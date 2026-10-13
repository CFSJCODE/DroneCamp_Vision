"""Treinamento, avaliação e busca de hiperparâmetros do detector YOLO.

Função no projeto: é o arquivo que de fato treina o modelo. Ele chama
``model.train`` da Ultralytics e audita o resultado antes de dar a execução
como concluída.

O que faz:
- ``train_pilot``: treino piloto com as revisões humanas do CEASA (o usado hoje);
  os pesos só sugerem caixas na revisão.
- ``train``: fine-tuning de produção; exige dataset aprovado e três edificações.
- ``evaluate``: métricas por classe em val ou test.
- ``tune``: vários treinos curtos para procurar hiperparâmetros.
- ``review_training_result``: confere pesos, ``results.csv`` e classes no fim.
- ``resume_pilot``: retoma um treino piloto interrompido a partir de ``last.pt``.
- ``prepare_dataset``: valida dataset e procedência (treino de produção, evaluate, tune).
- ``is_pilot_checkpoint`` / ``pilot_inference_config``: reconhecem pesos piloto e
  ajustam o imgsz da inferência ao do treino.
- ``review_tuning_result``: confere o resultado do ``tune``.
- Internas: ``_resolve_and_snapshot`` (caminhos absolutos + hashes + contagens),
  ``_tiled_training_dataset`` (janelas das fotos grandes), ``_apply_repeat_factor_sampling``
  (classes raras), ``_fit_staging``/``_mirror_results``/``_copy_back_fit`` (área de
  gravação em disco interno), ``_write_training_summary`` e ``_review_training_csv``.

Quando mexer: épocas, imgsz, batch, patience, lr e aumentos ficam em
``configs/project.yaml`` (seções ``training`` e ``pilot.training``), não aqui.
Mexa neste arquivo para mudar o fluxo: o que é validado antes, quais parâmetros
chegam ao ``model.train`` ou como o resultado é auditado. Para mudar como o
dataset piloto é montado (splits, duplicatas), o arquivo é ``pilot.py``.

Quem chama:
- ``cli.py``: comandos ``train``, ``evaluate``, ``tune`` (produção),
  ``train-pilot`` (``train_pilot`` ou, com ``--resume``, ``resume_pilot``) e ``resume-pilot``.
- ``cycle.run_cycle`` (comando ``retrain-cycle``) e ``hparam_bandit``
  (``tune-pilot-bandit``) chamam ``train_pilot``.
- A plataforma (seção Treinamento) inicia ``train-pilot`` como tarefa; o
  ``operations.py`` lê ``fit/results.csv``, ``fit/heartbeat`` e ``summary.json``.
- ``prediction``, ``suggestions``, ``model_gate``, ``gpu_eval``, ``calibration``,
  ``autolabel`` e ``exporting`` usam ``pilot_inference_config``/``is_pilot_checkpoint``.

O que lê e grava no disco (cada execução numa pasta NOVA em ``runs/``):
- Lê: o ``dataset.yaml`` recebido e as fotos/labels dele, ``configs/taxonomy.json``,
  os pesos de partida e, no piloto, os registros de revisão (via ``pilot.py``).
- Grava em ``runs/<etapa>_<data>_<id>/``:
  ``summary.json`` (estado running/failed/completed e motivos), ``execution.json``
  (ambiente + parâmetros efetivos), ``dataset_resolved.yaml`` (caminhos absolutos),
  ``dataset_snapshot.json`` (SHA-256 de cada arquivo + contagem por classe),
  ``pilot_provenance.json`` ou ``dataset_validation.json``/``training_provenance.json``,
  ``fit/`` (pesos ``best.pt``/``last.pt``, ``results.csv``, gráficos, ``heartbeat``),
  e, conforme as opções, ``dataset_tiled_resolved.yaml``, ``tiling_manifest.json``,
  ``train_repeat_factor.txt``, ``sampling.json`` e ``gpu_eval/``.

Onde mexer (casos concretos):
- Mudar um hiperparâmetro do piloto (épocas, imgsz, batch, lr0, mosaic...):
  ``configs/project.yaml`` → ``pilot.training`` (ou ``training`` como base).
  Nada aqui.
- Opção nova do ``train-pilot`` que vire parâmetro da Ultralytics: crie a opção
  em ``cli.py`` e inclua o nome na tupla de ``overrides`` de lá; ela chega
  automaticamente em ``parameters`` (etapa 4 de ``train_pilot``).
- Opção nova que NÃO é da Ultralytics (como ``tile`` e ``repeat_factor_threshold``):
  retire-a de ``parameters`` com ``.pop`` antes do ``model.train`` (ver EXTENSAO:
  na etapa 4 de ``train_pilot``), senão a Ultralytics recusa o argumento.
- Callback novo durante o treino (ex.: avisar a plataforma por época): perto
  de ``model.add_callback`` na etapa 5 de ``train_pilot``.
- Verificação nova do resultado do treino: ``review_training_result`` (arquivos)
  ou ``_review_training_csv`` (colunas e valores do ``results.csv``).
- Campo novo no ``summary.json``: ``_write_training_summary``.
"""

from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import math
import os
import shutil
import time

import yaml

from .backend import check_domain_names, load_detector, load_inference_model, runtime_info
from .config import ProjectConfig, detection_names, load_taxonomy
from .dataset import validate_dataset
from .io import file_hash, make_run_directory, write_json
from .model_family import GENERIC_DISTANCE_LOSS, distance_loss_columns

# Intervalo mínimo entre toques em fit/heartbeat (plataforma: treino vivo × interrompido).
# Unidade: segundos. Menor = arquivo regravado mais vezes; maior = a plataforma demora
# mais para perceber um treino parado (operations.py compara a idade do arquivo).
HEARTBEAT_SECONDS = 60


# ---------------------------------------------------------------------------
# Preparação dos dados: tudo o que é conferido antes de o YOLO ver o dataset.
# ---------------------------------------------------------------------------

def prepare_dataset(config: ProjectConfig, data_path: Path, run: Path) -> tuple[Path, dict]:
    """Registre a revisão e normalize caminhos antes de entregar dados ao YOLO.

    Usada pelos fluxos de PRODUÇÃO (``train``, ``evaluate``, ``tune``). O piloto
    não passa por aqui: usa ``pilot.validate_pilot_dataset`` + ``_resolve_and_snapshot``.

    Parâmetros:
        config (ProjectConfig): configuração do projeto.
        data_path (Path): ``dataset.yaml`` do dataset (absoluto).
        run (Path): pasta da execução, onde os relatórios são gravados.

    Retorna:
        tuple[Path, dict]: (caminho de ``run/dataset_resolved.yaml``, contagem de
        caixas ``{"train": {classe: n}, "val": {...}, "test": {...}}``).

    Erros:
        ValueError: dataset reprovado em ``validate_dataset``; procedência inválida
            (``review_provenance``); classe ativa sem nenhuma caixa no treino.

    Grava: ``dataset_validation.json``, ``training_provenance.json``,
    ``dataset_resolved.yaml`` e ``dataset_snapshot.json`` dentro de ``run``.
    """
    # 1. Estrutura: pastas, labels, classes e groups.csv (dataset.py).
    names = detection_names(load_taxonomy(config.taxonomy_path))
    report = validate_dataset(data_path, names)
    write_json(run / "dataset_validation.json", report)
    if not report["valid"]:
        raise ValueError(f"Dataset reprovado; consulte {run / 'dataset_validation.json'}.")
    # 2. Procedência: cada foto/label precisa vir de uma aprovação humana registrada.
    # ATENCAO: é aqui que se garante que a IA nunca aprova dados de treino de produção.
    from .review_provenance import validate_training_provenance
    try:
        provenance = validate_training_provenance(config, data_path)
    except ValueError as error:
        # Grava o motivo antes de propagar o erro, para a auditoria ficar no run.
        write_json(run / "training_provenance.json", {"valid": False, "reason": str(error)})
        raise
    write_json(run / "training_provenance.json", {"valid": True, **provenance})
    # 3. Caminhos absolutos e foto (hash) do que será treinado.
    normalized_path, class_counts = _resolve_and_snapshot(config, data_path, run, names)
    # Uma classe ativa sem exemplos não foi aprendida, mesmo que o treino termine.
    missing = [name for name, count in class_counts["train"].items() if count == 0]
    if missing:
        raise ValueError(f"Treino sem exemplos de classes: {', '.join(missing)}. Anote dados ou proponha nova taxonomia/versionamento antes de treinar.")
    return normalized_path, class_counts


def _resolve_and_snapshot(config: ProjectConfig, data_path: Path, run: Path, names: list[str]) -> tuple[Path, dict]:
    """Normalize caminhos e registre hashes e contagens do que será treinado.

    Parâmetros:
        config (ProjectConfig): usa ``taxonomy_path`` (hash gravado no snapshot).
        data_path (Path): ``dataset.yaml`` recebido (com ``path``, ``train``, ``val``, ``test``).
        run (Path): pasta da execução.
        names (list[str]): classes ativas na ordem dos IDs (``detection_names``).

    Retorna:
        tuple[Path, dict]: (``run/dataset_resolved.yaml``, contagem de caixas por
        split e classe, ex.: ``{"train": {"telha_quebrada": 12, ...}, "val": {...}, "test": {...}}``).

    Grava: ``run/dataset_resolved.yaml`` e ``run/dataset_snapshot.json``.

    ATENCAO: supõe a estrutura YOLO ``<root>/images/<split>`` e
    ``<root>/labels/<split>``; a pasta de labels é deduzida trocando ``images`` por ``labels``.
    """
    # Lê o dataset.yaml recebido e transforma path/train/val/test em caminhos absolutos.
    # utf-8-sig aceita arquivos gravados com BOM (comum em editores do Windows).
    original = yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))
    root = Path(original.get("path", ".")).expanduser()
    # "path" relativo é relativo à pasta do próprio dataset.yaml.
    root = (data_path.parent / root).resolve() if not root.is_absolute() else root.resolve()
    # names vem da taxonomia, não do YAML recebido: a taxonomia é o contrato.
    normalized = {"path": str(root), "names": names}
    for split in ("train", "val", "test"):
        split_path = Path(original[split]).expanduser()
        normalized[split] = str((root / split_path).resolve())
    # Não copie campos download ou outras instruções do YAML recebido.
    # ATENCAO: o YAML entregue à Ultralytics é sempre este, montado aqui; um "download:" no
    # YAML de origem poderia executar código/baixar dados, por isso não é copiado.
    normalized_path = run / "dataset_resolved.yaml"
    normalized_path.write_text(yaml.safe_dump(normalized, allow_unicode=True), encoding="utf-8")
    # Registra o SHA-256 de cada foto/label e conta as caixas por classe e split.
    snapshot = []
    class_counts = {split: {name: 0 for name in names} for split in ("train", "val", "test")}
    for split in ("train", "val", "test"):
        image_dir = Path(normalized[split])
        # images/<split> → labels/<split> (mesmo caminho relativo dentro de root).
        label_dir = root / "labels" / image_dir.relative_to(root / "images")
        # Hash de cada arquivo (fotos e labels), em ordem fixa (sorted) para o JSON ser estável.
        for directory in (image_dir, label_dir):
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    snapshot.append({"path": path.relative_to(root).as_posix(), "sha256": file_hash(path)})
        # Cada linha não vazia da label YOLO é "classe cx cy w h"; o 1º número é o ID da classe.
        for path in label_dir.rglob("*.txt"):
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    class_counts[split][names[int(line.split()[0])]] += 1
    # EXTENSAO: dado novo de auditoria do dataset (ex.: nº de fotos por split) entra neste dict.
    write_json(run / "dataset_snapshot.json", {
        "source_yaml_sha256": file_hash(data_path), "groups_sha256": file_hash(root / "groups.csv"),
        "taxonomy_sha256": file_hash(config.taxonomy_path), "files": snapshot,
        "class_counts": class_counts,
    })
    return normalized_path, class_counts


# ---------------------------------------------------------------------------
# Treinos: produção (train) e piloto (train_pilot). Ambos chamam model.train.
# ---------------------------------------------------------------------------

# ENTRADA: comando train (produção). Bloqueado na prática até haver dataset aprovado de três edificações.
def train(config: ProjectConfig, data_path: Path, weights: str | None = None) -> Path:
    """Ajuste os pesos pré-treinados; parâmetros explícitos ficam auditáveis.

    Parâmetros:
        config (ProjectConfig): configuração; hiperparâmetros de ``config.training``.
        data_path (Path): ``dataset.yaml`` do dataset de produção.
        weights (str | None): peso de partida; ``None`` usa ``model:`` do project.yaml.

    Retorna:
        Path: a pasta ``runs/train_<data>_<id>`` com ``summary.json`` "completed".

    Erros:
        ValueError: "Treino incompleto: ..." em qualquer falha de preparação,
            treino ou auditoria final (o motivo fica em ``summary.json``).
    """
    # Pasta nova runs/train_<data>_<id>; o summary começa como "running".
    run = make_run_directory(config.root, "train")
    _write_training_summary(run, ["A preparação e o treinamento ainda não concluíram."], state="running")
    try:
        dataset, counts = prepare_dataset(config, data_path.resolve(), run)
        expected_names = detection_names(load_taxonomy(config.taxonomy_path))
        model = load_detector(config, weights)
        # Hiperparâmetros vêm da seção training do configs/project.yaml.
        parameters = dict(config.training)
        # nms vem de prediction.nms (false no YOLO26, que é end-to-end).
        parameters["nms"] = config.prediction["nms"]
        write_json(run / "execution.json", {"runtime": runtime_info(model, config), "parameters": parameters,
                                           "class_counts": counts, "stage": "fine_tuning"})
        # Treino de verdade (Ultralytics); pesos e results.csv vão para run/fit.
        # exist_ok=False: a Ultralytics recusa uma pasta fit já existente (nunca sobrescreve).
        model.train(data=str(dataset), device=config.device, project=str(run), name="fit",
                    exist_ok=False, **parameters)
    except Exception as error:
        # Uma falha de preparação, treino ou callback nunca deixa uma execução aprovada.
        reason = f"Execução interrompida por {type(error).__name__}: {error}"
        _write_training_summary(run, [reason], state="failed")
        raise ValueError(f"Treino incompleto: {reason}. Revise {run}.") from error
    review_training_result(run, model, expected_names)
    return run


# ENTRADA: comando train-pilot e a tarefa "train-pilot" da seção Treinamento. Hiperparâmetros
# vêm de configs/project.yaml (pilot.training > training); não fixe valores aqui.
#
# De onde vem cada valor que chega ao model.train (configs/project.yaml, valores de hoje):
#   modelo inicial  --weights > pilot.model (yolo26l.pt) > model (yolo26l.pt)
#   device          device: (cpu), no topo do YAML
#   epochs          --epochs > pilot.training.epochs (80) > training.epochs (100)
#   imgsz           --imgsz  > pilot.training.imgsz (640) > training.imgsz (1024)
#   batch           --batch  > pilot.training.batch (4)   > training.batch (2)
#   patience        pilot.training.patience (0 = sem parada antecipada) > training.patience (20)
#   plots           pilot.training.plots (true)
#   workers, optimizer, lr0, lrf, seed, deterministic, cache, amp, close_mosaic,
#   mosaic, mixup, degrees, fliplr, flipud
#                   só em training: (0, AdamW, 0.001, 0.01, 42, true, false, false, 10,
#                   0.5, 0.0, 10.0, 0.5, 0.0), a não ser que pilot.training os repita
#   nms             prediction.nms (false), sempre sobrescrito
#   tile            --tile (retrain-cycle liga por padrão); NÃO vai ao model.train: ativa
#                   _tiled_training_dataset com pilot.tiling (patch 1280, overlap 0.2,
#                   negative_ratio 0.2 se ausente)
#   repeat_factor_threshold
#                   --repeat-factor-threshold > pilot.sampling.repeat_factor_threshold
#                   (desligado hoje); NÃO vai ao model.train: ativa _apply_repeat_factor_sampling
#   gpu_eval        --gpu-eval* > pilot.gpu_eval (enabled false, every 5, directml, val, 1800 s)
# O tune-pilot-bandit (hparam_bandit.py) também passa overrides (ex.: lr0, mosaic).
def apply_cpu_threads(threads: int | None) -> None:
    """Fixa as threads de CPU do treino acima do limite de 8 que a Ultralytics impõe.

    Parâmetros: ``threads`` (None ou 0 = mantém o padrão da Ultralytics).
    Retorna: None. Usa ``gpu_eval.limit_worker_threads``, que ajusta o torch e a
    constante ``NUM_THREADS`` da Ultralytics (ela reaplica esse valor no select_device).
    """
    if threads:
        from .gpu_eval import limit_worker_threads

        limit_worker_threads(int(threads))


def train_pilot(config: ProjectConfig, data_path: Path, weights: str | None = None, overrides: dict | None = None,
                gpu_eval: dict | None = None) -> Path:
    """Treino real e exploratório com o dataset piloto; nunca aprova o modelo para uso.

    ``gpu_eval`` (opções ``--gpu-eval*`` sobre ``pilot.gpu_eval``): enquanto a CPU
    treina, um worker avalia na GPU os checkpoints de N em N épocas (``gpu_eval.py``).

    Etapas (numeradas igual no corpo):
        1. cria ``runs/pilot_train_<data>_<id>`` e grava ``summary.json`` "running";
        2. reconfere o dataset piloto contra as revisões humanas e registra hashes;
        3. carrega o modelo inicial;
        4. monta os parâmetros (training < pilot.training < overrides) e aplica as
           opções que não são da Ultralytics: 4a janelas (``tile``), 4b classes raras;
        5. registra callbacks (espelho do results.csv, 5a batimento, 5b avaliação na
           GPU) e chama ``model.train``;
        6. audita o resultado (``review_training_result``) e marca "completed".

    Parâmetros:
        config (ProjectConfig): configuração do projeto.
        data_path (Path): ``dataset.yaml`` de ``data/pilot/<versão>/`` (build-pilot-data).
        weights (str | None): peso inicial (``--weights``); ``None`` usa
            ``pilot.model`` ou ``model``.
        overrides (dict | None): valores da linha de comando ou do bandit
            (``epochs``, ``imgsz``, ``batch``, ``repeat_factor_threshold``, ``tile``,
            ``lr0``...); vencem o YAML.
        gpu_eval (dict | None): opções ``enabled``, ``every``, ``provider``, ``split``
            (valores ``None`` são ignorados e ficam os de ``pilot.gpu_eval``).

    Retorna:
        Path: a pasta da execução, com ``summary.json`` ``"state": "completed"``,
        ``"pilot": true`` e ``"production_ready": false``.

    Erros:
        ValueError: "Treino piloto incompleto: ..." em qualquer falha (dataset,
            modelo, treino, callback, Ctrl+C) ou "Treino incompleto: ..." na
            auditoria final. O motivo também fica em ``summary.json``.

    Chamada por: ``cli.py`` (train-pilot), ``cycle.run_cycle`` (retrain-cycle) e
    ``hparam_bandit`` (tune-pilot-bandit).
    """
    from . import gpu_eval as gpu
    from .pilot import validate_pilot_dataset

    # 1. Pasta nova runs/pilot_train_<data>_<id>; summary marcado como piloto.
    run = make_run_directory(config.root, "pilot_train")
    # pilot_details acompanha a execução inteira e vai para execution.json e summary.json.
    # ATENCAO: production_ready é sempre False no piloto; pesos piloto só sugerem caixas.
    pilot_details = {"pilot": True, "production_ready": False}
    _write_training_summary(run, ["A preparação e o treinamento piloto ainda não concluíram."], state="running", **pilot_details)
    # gpu_settings = pilot.gpu_eval + opções --gpu-eval*; None quando a avaliação na GPU está desligada.
    worker, gpu_settings = None, gpu.gpu_eval_settings(config, gpu_eval)
    try:
        # 2. Reconfere hashes e labels do dataset piloto contra os registros humanos.
        names = detection_names(load_taxonomy(config.taxonomy_path))
        audit = validate_pilot_dataset(config, data_path.resolve())
        write_json(run / "pilot_provenance.json", audit)
        # dataset = run/dataset_resolved.yaml (caminhos absolutos); counts = caixas por split/classe.
        dataset, counts = _resolve_and_snapshot(config, data_path.resolve(), run, names)
        # 3. Modelo inicial: --weights, ou pilot.model, ou model do project.yaml.
        model = load_detector(config, weights or config.pilot.get("model") or config.model)
        # 4. Parâmetros: training < pilot.training < opções da linha de comando.
        # Junção de dicts: a chave repetida mais à direita vence (ver tabela acima da função).
        parameters = {**config.training, **config.pilot.get("training", {}), **(overrides or {})}
        parameters["nms"] = config.prediction["nms"]
        # cpu_threads (pilot.training): a Ultralytics fixa 8 threads de CPU em qualquer máquina;
        # no Ryzen 5 4600G (6 núcleos/12 threads) 12 threads deram passo 5% mais rápido (13,0 s
        # contra 13,7 s por lote de 4 do yolo26x, medido em 13/10 com o treino pausado).
        apply_cpu_threads(parameters.pop("cpu_threads", None))
        # EXTENSAO: opção nova que NÃO é argumento da Ultralytics = ``parameters.pop("nome", padrão)``
        # aqui (como tile e repeat_factor_threshold abaixo), antes do model.train; registre o efeito
        # em pilot_details para ir ao execution.json/summary.json.
        # 4a. Fotos grandes de drone: treino em janelas (pilot.tiling), na mesma escala
        # da inferência fatiada. O dataset piloto auditado acima continua sendo a fonte.
        if parameters.pop("tile", False):
            dataset, pilot_details["tiling"] = _tiled_training_dataset(config, run, data_path.resolve(),
                                                                       int(parameters.get("seed", 0)))
        # 4b. Classes raras: lista de treino com repetições (sampling.py). Não é
        # parâmetro da Ultralytics, por isso sai de ``parameters`` antes do train.
        threshold = parameters.pop("repeat_factor_threshold", None)
        # Sem opção na linha de comando: usa pilot.sampling.repeat_factor_threshold do YAML (se houver).
        if threshold is None:
            threshold = (config.pilot.get("sampling") or {}).get("repeat_factor_threshold")
        if threshold is not None:
            pilot_details["sampling"] = _apply_repeat_factor_sampling(run, dataset, float(threshold),
                                                                     int(parameters.get("seed", 0)))
        # Registro para a auditoria: classes sem caixa no treino e o aviso do dataset piloto.
        pilot_details["classes_without_training_boxes"] = [name for name, count in counts["train"].items() if count == 0]
        pilot_details["warning"] = audit["warning"]
        # class_counts conta caixas da unidade treinada: fotos, ou janelas com --tile
        # (uma caixa na sobreposição de duas janelas conta duas vezes).
        # ATENCAO: ``counts`` foi calculado na etapa 2, sobre o dataset original (fotos), e não é
        # recalculado depois do fatiamento; com --tile o valor gravado continua sendo o das fotos
        # (o fatiamento só informa totais em pilot_details["tiling"]["stats"], ex.: total_boxes_projected).
        pilot_details["label_unit"] = "janelas" if "tiling" in pilot_details else "fotos"
        # Área de gravação em disco interno (variável DRONECAMP_FIT_STAGING); None se não definida.
        staging = _fit_staging(run)
        if staging:
            pilot_details["fit_staging"] = str(staging)
        # execution.json: ambiente + parâmetros EFETIVOS (já sem tile/repeat_factor_threshold).
        execution = {"runtime": runtime_info(model, config), "parameters": parameters,
                     "class_counts": counts, "stage": "fine_tuning_piloto", **pilot_details}
        write_json(run / "execution.json", execution)
        # 5. Treino de verdade (Ultralytics); best.pt/last.pt vão para run/fit/weights
        #    (ou para a área de gravação no disco interno, copiada de volta no fim).
        # EXTENSAO: callback novo por época/lote = ``model.add_callback("on_...", função)`` aqui,
        # antes do model.train. A função recebe o ``trainer`` da Ultralytics e não deve lançar
        # exceção (uma exceção interrompe o treino e marca a execução como "failed").
        if staging:
            # Com staging, o results.csv é copiado para run/fit a cada época (monitor da plataforma).
            model.add_callback("on_fit_epoch_end", lambda trainer: _mirror_results(staging, run))
        # 5a. Batimento: fit/heartbeat é tocado a cada minuto para a plataforma distinguir
        #     "treinando" (época longa, results.csv parado) de "interrompido".
        # heartbeat é um dict (e não variáveis soltas) para a função interna poder alterar "at".
        heartbeat = {"path": run / "fit" / "heartbeat", "at": 0.0}

        def beat(trainer):
            """Callback de fim de lote: grava a hora UTC em ``fit/heartbeat`` no máximo 1x por minuto.

            Parâmetros:
                trainer: o treinador da Ultralytics (não é usado; exigido pela assinatura do callback).

            Retorna:
                None. Falhas de disco são ignoradas: o batimento nunca interrompe o treino.
            """
            now = time.time()
            # Só regrava se já passou HEARTBEAT_SECONDS desde o último toque.
            if now - heartbeat["at"] >= HEARTBEAT_SECONDS:
                heartbeat["at"] = now
                try:
                    heartbeat["path"].parent.mkdir(parents=True, exist_ok=True)
                    # Conteúdo: data/hora ISO 8601 em UTC (ex.: 2026-10-12T15:30:00.123456+00:00).
                    heartbeat["path"].write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
                except OSError:
                    pass

        model.add_callback("on_train_batch_end", beat)
        # 5b. CPU + GPU em conjunto: o worker avalia last.pt na GPU enquanto o treino continua.
        #     on_model_save dispara depois de last.pt ser gravado (on_fit_epoch_end viria antes).
        if gpu_settings:
            # work = pasta da fila de checkpoints (na área de gravação, se houver).
            work = gpu.work_directory(run)
            # Estado da avaliação na GPU, gravado no execution.json ao final.
            gpu_state = {"settings": gpu_settings, "state": "running", "enqueued": [], "skipped": [], "last_epoch": 0}
            pilot_details["gpu_eval"] = gpu_state
            # O worker é um processo separado (gpu-eval-worker); None se o DirectML não estiver disponível.
            worker = gpu.start_worker(config, run, data_path.resolve(), gpu_settings, int(parameters["imgsz"]))
            if worker is None:
                gpu_state["state"] = "skipped"
                print("gpu-eval: DirectML indisponível; o treino segue sem a avaliação paralela na GPU.")

            def enqueue(trainer):
                """Callback ``on_model_save``: põe o ``last.pt`` desta época na fila da GPU.

                Enfileira a cada ``gpu_settings["every"]`` épocas e na última; pula a
                época se a fila ainda estiver ocupada (a GPU é mais lenta que a época).

                Parâmetros:
                    trainer: treinador da Ultralytics (usa ``epoch`` 0-based, ``epochs`` e ``last``).

                Retorna:
                    None. Erros de fila/disco são registrados em ``skipped`` e não param o treino.
                """
                # trainer.epoch começa em 0; a época "humana" (results.csv) é epoch + 1.
                epoch = trainer.epoch + 1
                gpu_state["last_epoch"] = epoch
                final = epoch == trainer.epochs
                # Nada a fazer sem worker, se a época já foi enfileirada ou se last.pt ainda não existe.
                if worker is None or epoch in gpu_state["enqueued"] or not Path(trainer.last).is_file():
                    return
                # Só nas épocas múltiplas de "every" (ex.: 5, 10, 15...) e na última.
                if not (epoch % gpu_settings["every"] == 0 or final):
                    return
                # A GPU é mais lenta que a época: com fila ocupada pula a época (menos a última).
                if gpu.pending_checkpoints(work) and not final:
                    gpu_state["skipped"].append(epoch)
                    return
                try:
                    gpu.enqueue_checkpoint(work, Path(trainer.last), epoch)
                    gpu_state["enqueued"].append(epoch)
                except Exception as error:  # fila cheia/disco: o treino não para por causa da curva
                    gpu_state["skipped"].append(epoch)
                    print(f"gpu-eval: não foi possível enfileirar a época {epoch}: {error}")

            model.add_callback("on_model_save", enqueue)
        # trained só vira True se model.train voltar sem exceção (distingue concluído × interrompido).
        trained = False
        try:
            # ATENCAO: exist_ok=False faz a Ultralytics recusar uma pasta fit já existente; com
            # staging, project/name apontam para <staging>/<run>/fit em vez de <run>/fit.
            model.train(data=str(dataset), device=config.device, project=str(staging.parent if staging else run),
                        name=staging.name if staging else "fit", exist_ok=False, **parameters)
            trained = True
        finally:
            # Roda SEMPRE (sucesso, erro ou Ctrl+C): traz o fit/ de volta e fecha o worker da GPU.
            if staging:
                _copy_back_fit(staging, run)
            if gpu_settings:
                # A última época entra na fila mesmo que o callback não a tenha visto; só num
                # treino concluído (Ctrl+C ou erro deixam um last.pt parcial que não é a época final).
                last = run / "fit" / "weights" / "last.pt"
                final_epoch = int(gpu_state["last_epoch"])  # época real de last.pt (parada antecipada inclusa)
                if trained and worker is not None and last.is_file() and final_epoch and final_epoch not in gpu_state["enqueued"]:
                    try:
                        gpu.enqueue_checkpoint(work, last, final_epoch)
                        gpu_state["enqueued"].append(final_epoch)
                    except Exception as error:
                        print(f"gpu-eval: não foi possível enfileirar a época final: {error}")
                # Treino concluído: espera o worker até pilot.gpu_eval.timeout segundos; interrompido: não espera.
                timeout = float(gpu_settings["timeout"]) if trained else 0.0
                summary = gpu.finish_worker(run, worker, timeout) if worker is not None else {}
                state = "finished" if trained else "interrupted"
                pilot_details["gpu_eval"] = {**gpu_state, "state": state if worker is not None else "skipped", **summary}
                # worker = None evita que o except abaixo tente encerrá-lo de novo.
                worker = None
                write_json(run / "execution.json", {**execution, "gpu_eval": pilot_details["gpu_eval"]})
    except Exception as error:
        # Qualquer falha das etapas 2 a 5: encerra o worker (se ainda vivo) e grava "failed".
        if worker is not None:
            gpu.finish_worker(run, worker, 0)
        reason = f"Execução interrompida por {type(error).__name__}: {error}"
        _write_training_summary(run, [reason], state="failed", **pilot_details)
        raise ValueError(f"Treino piloto incompleto: {reason}. Revise {run}.") from error
    # 6. Auditoria final: só aqui o summary passa para "completed".
    review_training_result(run, model, names, **pilot_details)
    return run


# ---------------------------------------------------------------------------
# Área de gravação opcional (DRONECAMP_FIT_STAGING): o Ultralytics grava last.pt
# (~200 MB) a cada época. Num HD externo USB essa escrita já falhou com
# OSError 22 no meio do treino; num disco interno o risco cai. Ao final (ou na
# falha) tudo volta para run/fit, que continua sendo o único lugar de referência.
# ---------------------------------------------------------------------------

def _tiled_training_dataset(config: ProjectConfig, run: Path, source: Path, seed: int) -> tuple[Path, dict]:
    """Gera as janelas do dataset piloto (treino/val/teste separados como no original).

    As janelas ficam na área de gravação (disco interno) quando houver; o YAML com
    caminhos absolutos e o relatório do fatiamento ficam no run.

    Parâmetros:
        config (ProjectConfig): usa ``config.pilot["tiling"]`` (``patch`` em px,
            padrão 1280; ``overlap`` fração 0–1, padrão 0.2; ``negative_ratio``
            fração de janelas sem caixa mantidas, padrão 0.2).
        run (Path): pasta da execução.
        source (Path): ``dataset.yaml`` do dataset piloto original.
        seed (int): semente do sorteio das janelas negativas (vem de ``seed`` do treino).

    Retorna:
        tuple[Path, dict]: (``run/dataset_tiled_resolved.yaml``, resumo com
        ``patch``, ``overlap``, ``negative_ratio``, ``tiles_directory`` e ``stats``).

    Grava: ``<base>/tiled_dataset/`` (janelas), ``run/dataset_tiled_resolved.yaml`` e
    ``run/tiling_manifest.json``.

    Chamada por: ``train_pilot`` (etapa 4a, quando ``tile`` é verdadeiro).
    """
    from .tiling import build_tiled_dataset

    # Parâmetros do fatiamento: pilot.tiling do project.yaml (os mesmos da inferência fatiada).
    plan = config.pilot.get("tiling") or {}
    # Pasta base: <DRONECAMP_FIT_STAGING>/<nome do run> se a variável existir; senão o próprio run.
    base = Path(os.environ["DRONECAMP_FIT_STAGING"]).expanduser().resolve() / run.name \
        if os.environ.get("DRONECAMP_FIT_STAGING") else run
    output = base / "tiled_dataset"
    # EXTENSAO: parâmetro novo do fatiamento = chave nova em pilot.tiling, lida aqui com padrão,
    # passada a tiling.build_tiled_dataset e registrada no dict de retorno abaixo.
    report = build_tiled_dataset(source.parent, output, int(plan.get("patch", 1280)), float(plan.get("overlap", 0.2)),
                                 float(plan.get("negative_ratio", 0.2)), seed)
    original = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    # YAML novo com caminhos absolutos; só entram os splits que o fatiamento realmente gerou.
    resolved = {"path": str(output), "names": original["names"],
                **{split: str(output / "images" / split) for split in ("train", "val", "test")
                   if (output / "images" / split).is_dir()}}
    dataset = run / "dataset_tiled_resolved.yaml"
    dataset.write_text(yaml.safe_dump(resolved, allow_unicode=True), encoding="utf-8")
    # Cópia do manifesto no run, para a auditoria não depender da área de gravação.
    shutil.copy2(output / "tiling_manifest.json", run / "tiling_manifest.json")
    return dataset, {"patch": int(plan.get("patch", 1280)), "overlap": float(plan.get("overlap", 0.2)),
                     "negative_ratio": float(plan.get("negative_ratio", 0.2)), "tiles_directory": str(output),
                     "stats": report.get("stats", report)}


def _fit_staging(run: Path) -> Path | None:
    """Pasta ``fit`` na área de gravação, se a variável ``DRONECAMP_FIT_STAGING`` estiver definida.

    Parâmetros:
        run (Path): pasta da execução (o nome dela é repetido dentro da área de gravação).

    Retorna:
        Path | None: ``<DRONECAMP_FIT_STAGING>/<nome do run>/fit`` (a pasta-mãe é
        criada; ``fit`` não, porque a Ultralytics usa ``exist_ok=False``), ou
        ``None`` sem a variável (treino grava direto em ``run/fit``).

    Chamada por: ``train_pilot``.
    """
    base = os.environ.get("DRONECAMP_FIT_STAGING")
    if not base:
        return None
    staging = Path(base).expanduser().resolve() / run.name / "fit"
    staging.parent.mkdir(parents=True, exist_ok=True)
    return staging


def _mirror_results(staging: Path, run: Path) -> None:
    """Copia results.csv a cada época para o monitor da plataforma acompanhar o treino.

    Parâmetros:
        staging (Path): pasta ``fit`` na área de gravação.
        run (Path): pasta da execução (destino ``run/fit/results.csv``).

    Retorna:
        None. Sem ``results.csv`` ainda (1ª época), não faz nada.

    Chamada por: callback ``on_fit_epoch_end`` registrado em ``train_pilot``.
    """
    source = staging / "results.csv"
    if source.is_file():
        (run / "fit").mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, run / "fit" / "results.csv")


def _copy_back_fit(staging: Path, run: Path) -> None:
    """Copia a pasta ``fit`` inteira da área de gravação de volta para ``run/fit``.

    Parâmetros:
        staging (Path): pasta ``fit`` na área de gravação.
        run (Path): pasta da execução.

    Retorna:
        None. ``dirs_exist_ok=True`` mescla com o que já está em ``run/fit``
        (results.csv espelhado, heartbeat). A área de gravação NÃO é apagada.

    Chamada por: ``train_pilot`` (bloco ``finally``, em sucesso ou falha).
    """
    if staging.is_dir():
        shutil.copytree(staging, run / "fit", dirs_exist_ok=True)


# ENTRADA: train-pilot --resume <run> e resume-pilot <run> (cli.py).
# ATENCAO: a retomada não registra o batimento (fit/heartbeat), não reativa a avaliação na
# GPU e não faz a cópia de volta da área de gravação; a Ultralytics retoma com os argumentos
# gravados no last.pt (inclusive a pasta de saída do treino original).
def resume_pilot(config: ProjectConfig, run_path: Path) -> Path:
    """Retoma um treino piloto interrompido a partir de fit/weights/last.pt.

    Parâmetros:
        config (ProjectConfig): configuração do projeto.
        run_path (Path): pasta da execução interrompida (``runs/pilot_train_...``).

    Retorna:
        Path: a mesma pasta, com ``summary.json`` "completed" após a auditoria.

    Erros:
        ValueError: "Checkpoint last.pt não encontrado..." ou
            "Treino piloto incompleto: ..." (falha na retomada) ou a falha da auditoria.
    """
    run = run_path.resolve()
    last_pt = run / "fit" / "weights" / "last.pt"
    if not last_pt.is_file():
        raise ValueError(f"Checkpoint last.pt não encontrado em {last_pt}.")
    names = detection_names(load_taxonomy(config.taxonomy_path))
    pilot_details = {"pilot": True, "production_ready": False}
    # Volta o summary para "running" enquanto retoma (o anterior dizia "failed").
    _write_training_summary(run, ["A preparação e o treinamento piloto ainda não concluíram."], state="running", **pilot_details)
    try:
        model = load_detector(config, str(last_pt))
        apply_cpu_threads((config.pilot.get("training") or {}).get("cpu_threads"))
        # resume=True: a Ultralytics lê época, otimizador e argumentos de dentro do last.pt.
        model.train(resume=True)
    except Exception as error:
        reason = f"Execução interrompida por {type(error).__name__}: {error}"
        _write_training_summary(run, [reason], state="failed", **pilot_details)
        raise ValueError(f"Treino piloto incompleto: {reason}. Revise {run}.") from error
    review_training_result(run, model, names, **pilot_details)
    return run


def _apply_repeat_factor_sampling(run: Path, dataset: Path, threshold: float, seed: int) -> dict:
    """Troca ``train:`` do dataset_resolved.yaml por uma lista com fotos raras repetidas.

    Validação e teste não mudam: as métricas continuam medindo cada foto uma vez.

    Parâmetros:
        run (Path): pasta da execução.
        dataset (Path): YAML resolvido em uso (``dataset_resolved.yaml`` ou
            ``dataset_tiled_resolved.yaml``); é REGRAVADO com o novo ``train:``.
        threshold (float): limiar do repeat factor (ex.: 0.3); fotos com classes
            mais raras que isso são repetidas (ver ``sampling.py``).
        seed (int): semente do sorteio (vem de ``seed`` do treino).

    Retorna:
        dict: resumo de ``sampling.write_repeat_factor_list`` (inclui ``list``,
        o caminho do ``.txt`` com uma foto por linha).

    Grava: ``run/train_repeat_factor.txt``, ``run/sampling.json`` e o YAML ``dataset``.

    Chamada por: ``train_pilot`` (etapa 4b).
    """
    from .sampling import write_repeat_factor_list

    resolved = yaml.safe_load(dataset.read_text(encoding="utf-8"))
    image_dir = Path(resolved["train"])
    # images/train → labels/train dentro do mesmo "path".
    label_dir = Path(resolved["path"]) / "labels" / image_dir.relative_to(Path(resolved["path"]) / "images")
    summary = write_repeat_factor_list(image_dir, label_dir, run / "train_repeat_factor.txt", threshold, seed)
    # A Ultralytics aceita um .txt de caminhos em train: (repetir a linha = repetir a foto).
    resolved["train"] = summary["list"]
    dataset.write_text(yaml.safe_dump(resolved, allow_unicode=True), encoding="utf-8")
    write_json(run / "sampling.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Pesos piloto: identificação e tamanho de imagem usado na inferência.
# ---------------------------------------------------------------------------

def is_pilot_checkpoint(weights: str | Path | None) -> bool:
    """Pesos de runs/pilot_train_*/fit/weights carregam o aviso do piloto.

    Parâmetros:
        weights (str | Path | None): caminho do ``.pt`` (ex.: ``runs/pilot_train_x/fit/weights/best.pt``).

    Retorna:
        bool: True se ``<run>/summary.json`` (três pastas acima do arquivo) tiver
        ``"pilot": true``; False sem pesos, sem summary ou com JSON inválido.

    Chamada por: prediction e ``pilot_inference_config``.
    """
    if not weights:
        return False
    # best.pt → weights/ → fit/ → <run>/summary.json.
    summary = Path(weights).resolve().parent.parent.parent / "summary.json"
    try:
        return json.loads(summary.read_text(encoding="utf-8")).get("pilot") is True
    except (OSError, ValueError):
        return False


def pilot_inference_config(config: ProjectConfig, model) -> ProjectConfig:
    """Pesos piloto inferem no tamanho em que foram treinados, não no da produção.

    Parâmetros:
        config (ProjectConfig): configuração original (``prediction.imgsz`` = 1024 hoje).
        model: modelo carregado por ``load_inference_model``/``load_detector``.

    Retorna:
        ProjectConfig: cópia com ``prediction["imgsz"]`` trocado pelo do ONNX
        (entrada fixa) ou pelo ``imgsz`` do treino piloto (640 hoje); o próprio
        ``config`` quando não se aplica.

    Chamada por: suggestions, prediction, model_gate, gpu_eval, calibration,
    autolabel, exporting e ``evaluate``.
    """
    from dataclasses import replace

    # ONNX (load_inference_model): entrada fixa gravada nos metadados do arquivo.
    if getattr(model, "dronecamp_onnx", None):
        from .exporting import _onnx_imgsz

        imgsz = _onnx_imgsz(model.dronecamp_onnx)
        return replace(config, prediction={**config.prediction, "imgsz": imgsz}) if imgsz else config
    # .pt que não é piloto: mantém o imgsz de produção.
    if not is_pilot_checkpoint(getattr(model, "ckpt_path", None)):
        return config
    # imgsz usado no treino, gravado pela Ultralytics no checkpoint (train_args).
    trained = (getattr(model, "ckpt", None) or {}).get("train_args", {}).get("imgsz")
    if not isinstance(trained, int) or trained < 32:
        return config
    return replace(config, prediction={**config.prediction, "imgsz": trained})


# ---------------------------------------------------------------------------
# Auditoria do resultado: summary.json, results.csv e contrato das classes.
# ---------------------------------------------------------------------------

# Colunas de métricas de detecção que a Ultralytics grava no results.csv ("(B)" = caixas).
# Cada valor precisa estar entre 0 e 1. EXTENSAO: métrica nova obrigatória = nome novo aqui.
DETECTION_METRIC_COLUMNS = (
    "metrics/precision(B)", "metrics/recall(B)",
    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
)


def _write_training_summary(run: Path, reasons: list[str], **details) -> dict:
    """Conclusão é execução auditada; aprovação técnica continua sendo humana.

    Parâmetros:
        run (Path): pasta da execução.
        reasons (list[str]): motivos de não conclusão; lista vazia = ``complete: true``.
        **details: campos extras (``state`` = "running"/"failed"/"completed",
            ``pilot``, ``production_ready``, ``artifacts``, métricas...). Um campo
            em ``details`` com o mesmo nome de um campo fixo o sobrescreve.

    Retorna:
        dict: o conteúdo gravado em ``run/summary.json``.

    ATENCAO: ``human_acceptance`` começa sempre "pendente"; nenhum código aprova um modelo.
    A plataforma (operations.py) lê ``state``, ``complete`` e ``pilot`` deste arquivo.
    """
    # EXTENSAO: campo fixo novo do summary.json entra neste dict (e na leitura em operations.py, se for exibido).
    summary = {
        "schema_version": 1, "stage": "fine_tuning", "complete": not reasons,
        "reasons": reasons, "weights_directory": str(run / "fit/weights"),
        "test_evaluated": False, "human_acceptance": "pendente",
        "checkpoint_functional_validation": False, "export_parity_verified": False,
        "note": "Arquivos .pt não vazios não comprovam checkpoint funcional. Avaliar por classe e revisar erros antes de usar em inspeções.",
        **details,
    }
    write_json(run / "summary.json", summary)
    return summary


def _review_training_csv(path: Path, distance_loss: tuple[str, ...] = GENERIC_DISTANCE_LOSS) -> tuple[list[str], dict]:
    """Confira métricas e perdas reais, aceitando parada antecipada com épocas válidas.

    Parâmetros:
        path (Path): ``run/fit/results.csv`` gravado pela Ultralytics.
        distance_loss (tuple[str, ...]): perdas de distância aceitas para a família
            (``model_family.distance_loss_columns``); basta uma existir em train/ e val/.

    Retorna:
        tuple[list[str], dict]: (motivos de reprovação, detalhes). Detalhes:
        ``epochs_recorded`` (linhas), ``last_epoch``, ``last_metrics`` (métricas
        da última linha) e ``loss_columns_checked``. Lista vazia = CSV aprovado.

    Chamada por: ``review_training_result``.
    """
    reasons, rows = [], []
    # Leitura: nomes de coluna sem espaços (a Ultralytics alinha o CSV com espaços).
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = [name.strip() for name in (reader.fieldnames or [])]
            if len(columns) != len(set(columns)):
                return ["results.csv contém colunas duplicadas."], {"epochs_recorded": 0}
            reader.fieldnames = columns
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        return [f"Não foi possível ler results.csv: {type(error).__name__}."], {"epochs_recorded": 0}

    # Colunas que toda execução de detecção precisa registrar.
    # EXTENSAO: coluna obrigatória nova do results.csv = acrescente neste conjunto.
    required = {"epoch", *DETECTION_METRIC_COLUMNS, "train/box_loss", "train/cls_loss", "val/box_loss", "val/cls_loss"}
    missing = sorted(required.difference(columns))
    if missing:
        reasons.append("results.csv sem colunas obrigatórias: " + ", ".join(missing) + ".")
    # Perda de distância da família (model_family.py): l1_loss no YOLO26, dfl_loss em outras cabeças.
    if not any(f"train/{name}" in columns and f"val/{name}" in columns for name in distance_loss):
        reasons.append("results.csv precisa de perdas de distância correspondentes no treino e na validação "
                       f"({' ou '.join(distance_loss)}).")
    # Todas as colunas de perda presentes (train/..._loss e val/..._loss) são conferidas linha a linha.
    loss_columns = [name for name in columns if name.startswith(("train/", "val/")) and name.endswith("_loss")]
    if not rows:
        reasons.append("results.csv não contém nenhuma época de treinamento.")

    # Linha a linha: números finitos, métricas entre 0 e 1, épocas crescentes.
    last_epoch = 0
    last_metrics = {}
    # row_number começa em 2 porque a linha 1 do arquivo é o cabeçalho (facilita achar no editor).
    for row_number, row in enumerate(rows, start=2):
        # O DictReader põe valores sobrando na chave None (linha com mais campos que o cabeçalho).
        if None in row:
            reasons.append(f"results.csv linha {row_number} tem valores além das colunas declaradas.")
        numeric = {}
        for name in ["epoch", *DETECTION_METRIC_COLUMNS, *loss_columns]:
            if name not in columns:
                continue
            try:
                value = float(row[name])
            except (TypeError, ValueError):
                reasons.append(f"results.csv linha {row_number}: {name} não é um número.")
                continue
            # NaN ou infinito indicam treino divergente.
            if not math.isfinite(value):
                reasons.append(f"results.csv linha {row_number}: {name} não é finito.")
                continue
            if name in DETECTION_METRIC_COLUMNS and not 0 <= value <= 1:
                reasons.append(f"results.csv linha {row_number}: {name} precisa estar entre 0 e 1.")
                continue
            numeric[name] = value
        # Época: inteira, >= 1 e maior que a anterior (sem linhas repetidas ou fora de ordem).
        epoch = numeric.get("epoch")
        if epoch is not None:
            if epoch < 1 or not epoch.is_integer() or epoch <= last_epoch:
                reasons.append(f"results.csv linha {row_number}: época deve ser inteira, positiva e crescente.")
            else:
                last_epoch = int(epoch)
        # Métricas zeradas são um resultado possível, não aprovação de qualidade.
        last_metrics = {name: numeric[name] for name in DETECTION_METRIC_COLUMNS if name in numeric}
    return reasons, {"epochs_recorded": len(rows), "last_epoch": last_epoch, "last_metrics": last_metrics,
                     "loss_columns_checked": loss_columns}


# ATENCAO: um treino só conta como concluído se passar por esta auditoria (pesos, results.csv
# e classes). Se summary.json tem "complete": false, os motivos estão em "reasons".
def review_training_result(run: Path, model, expected_names: list[str], **details) -> dict:
    """Recuse retorno sem artefatos, métricas finitas ou contrato das classes ativas.

    Parâmetros:
        run (Path): pasta da execução (``run/fit`` precisa ter pesos e results.csv).
        model: o modelo que treinou (usado para os nomes de classe e a família).
        expected_names (list[str]): classes ativas (``detection_names``).
        **details: campos extras repassados ao ``summary.json`` (ex.: ``pilot_details``).

    Retorna:
        dict: o ``summary.json`` final com ``state: "completed"``.

    Erros:
        ValueError: "Treino incompleto: <motivos> Revise <run>." quando há qualquer
            motivo; o ``summary.json`` é gravado com ``state: "failed"`` antes.

    Chamada por: ``train``, ``train_pilot`` e ``resume_pilot``.
    """
    run = Path(run).resolve()
    reasons, artifacts = [], {}
    # 1. Arquivos obrigatórios da execução: pesos e histórico de épocas.
    # EXTENSAO: artefato obrigatório novo do treino = entrada nova neste dict (nome: caminho).
    paths = {"best_weights": run / "fit/weights/best.pt", "last_weights": run / "fit/weights/last.pt",
             "results_csv": run / "fit/results.csv"}
    for name, path in paths.items():
        try:
            # Um link não pode redirecionar a auditoria para artefatos de outra execução.
            contained = path.resolve().is_relative_to(run)
            present = contained and path.is_file()
            size = path.stat().st_size if present else 0
            artifacts[name] = {"path": str(path), "present": present, "size_bytes": size}
            if not contained:
                reasons.append(f"{name} aponta para fora da execução.")
            elif not present:
                reasons.append(f"{name} ausente ou não é um arquivo.")
            elif size == 0:
                reasons.append(f"{name} está vazio.")
        except (OSError, RuntimeError):
            artifacts[name] = {"path": str(path), "present": False, "size_bytes": 0}
            reasons.append(f"{name} não pôde ser verificado.")
    # 2. Épocas, perdas e métricas finitas no results.csv.
    csv_details = {"epochs_recorded": 0}
    if artifacts["results_csv"]["present"] and artifacts["results_csv"]["size_bytes"] > 0:
        csv_reasons, csv_details = _review_training_csv(paths["results_csv"], distance_loss_columns(model))
        reasons.extend(csv_reasons)
    # 3. O modelo treinado precisa ter exatamente as classes ativas da taxonomia.
    class_contract_verified = False
    try:
        check_domain_names(model, expected_names)
        class_contract_verified = True
    except (ValueError, AttributeError, KeyError, TypeError) as error:
        reasons.append(f"Contrato das classes após o treino inválido: {error}")
    # 4. summary.json final: "completed" só sem nenhum motivo de falha.
    summary = _write_training_summary(
        run, reasons, state="failed" if reasons else "completed", artifacts=artifacts,
        model_class_contract_verified=class_contract_verified, expected_class_names=expected_names, **csv_details,
        **details,
    )
    if reasons:
        raise ValueError(f"Treino incompleto: {' '.join(reasons)} Revise {run}.")
    return summary


# ---------------------------------------------------------------------------
# Avaliação e busca de hiperparâmetros (produção).
# ---------------------------------------------------------------------------

# ENTRADA: comando evaluate (cli.py).
def evaluate(config: ProjectConfig, data_path: Path, weights: str, split: str = "val",
             onnx_provider: str = "directml") -> Path:
    """Avalie checkpoint especializado; use test apenas para decisão final.

    Parâmetros:
        config (ProjectConfig): configuração do projeto.
        data_path (Path): ``dataset.yaml`` (passa por ``prepare_dataset``, ou seja,
            exige dataset de produção com procedência válida).
        weights (str): ``.pt`` ou ``.onnx`` a avaliar.
        split (str): ``"val"`` (padrão) ou ``"test"``.
        onnx_provider (str): ``"directml"`` ou ``"cpu"`` (só para ``.onnx``).

    Retorna:
        Path: ``runs/evaluate_<split>_<data>_<id>`` com ``evaluation.json``
        (métricas globais e por classe) e ``metrics/`` (gráficos da Ultralytics).

    Erros:
        ValueError: dataset/procedência inválidos ou classes do modelo diferentes
            das ativas.

    ATENCAO: para ``.pt``, o imgsz é ``prediction.imgsz`` (1024 hoje) mesmo para pesos
    piloto treinados em 640; só o ``.onnx`` usa ``pilot_inference_config`` aqui.
    """
    run = make_run_directory(config.root, f"evaluate_{split}")
    dataset, counts = prepare_dataset(config, data_path.resolve(), run)
    # .onnx exportado: GPU via DirectML (batch 1 e tamanho fixo do ONNX).
    model = load_inference_model(config, weights, onnx_provider)
    check_domain_names(model, detection_names(load_taxonomy(config.taxonomy_path)))
    onnx = getattr(model, "dronecamp_onnx", None)
    imgsz = pilot_inference_config(config, model).prediction["imgsz"] if onnx else config.prediction["imgsz"]
    # Não reutilize conf=.25 do predict: AP precisa incluir candidatos de baixa confiança.
    # batch: 1 no ONNX (entrada fixa); senão training.batch. workers=0: leitura no processo principal.
    metrics = model.val(data=str(dataset), split=split, device=config.device,
                        imgsz=imgsz, conf=0.001, rect=False,
                        nms=config.prediction["nms"], batch=1 if onnx else config.training["batch"],
                        workers=0, project=str(run), name="metrics", plots=True)
    # Precisão, recall e AP de cada classe medida no split.
    # ap_class_index lista só as classes presentes no split; index é a posição nessa lista.
    per_class = []
    for index, class_id in enumerate(metrics.box.ap_class_index):
        precision, recall, ap50, ap = metrics.box.class_result(index)
        per_class.append({"class_name": model.names[int(class_id)], "precision": float(precision),
                          "recall": float(recall), "ap50": float(ap50), "ap50_95": float(ap)})
    # EXTENSAO: campo novo no evaluation.json (ex.: matriz de confusão) entra neste dict.
    write_json(run / "evaluation.json", {
        "runtime": runtime_info(model, config) if not onnx else {"onnx": str(onnx), "onnx_sha256": file_hash(onnx),
                                                                "source_checkpoint": str(model.dronecamp_source)},
        "inference_runtime": model.dronecamp_runtime, "split": split, "class_counts": counts,
        "global_metrics": {key: float(value) for key, value in metrics.results_dict.items()},
        "per_class": per_class, "human_acceptance": "pendente",
    })
    return run


# ENTRADA: comando tune (cli.py). Para o piloto, a busca usada é tune-pilot-bandit (hparam_bandit.py).
def tune(config: ProjectConfig, data_path: Path, iterations: int, epochs: int, weights: str | None = None) -> Path:
    """Procure hiperparâmetros na validação; não consuma o teste final.

    Parâmetros:
        config (ProjectConfig): configuração; base dos parâmetros em ``config.training``.
        data_path (Path): ``dataset.yaml`` de produção (passa por ``prepare_dataset``).
        iterations (int): número de treinos curtos (tentativas), >= 1.
        epochs (int): épocas de cada tentativa, >= 1.
        weights (str | None): peso de partida; ``None`` usa ``model:``.

    Retorna:
        Path: ``runs/tune_<data>_<id>`` com ``search/`` (resultados da Ultralytics)
        e ``summary.json`` (ver ``review_tuning_result``).

    Erros:
        ValueError: iterations/epochs não positivos, dataset inválido ou
            "Tuning incompleto: ..." na conferência final.
    """
    if iterations < 1 or epochs < 1:
        raise ValueError("iterations e epochs precisam ser positivos.")
    run = make_run_directory(config.root, "tune")
    dataset, _ = prepare_dataset(config, data_path.resolve(), run)
    model = load_detector(config, weights)
    # Cada tentativa treina poucas épocas e varia lr0 e mosaic dentro do espaço abaixo.
    parameters = dict(config.training)
    # close_mosaic não pode passar do total de épocas de cada tentativa.
    parameters.update(epochs=epochs, close_mosaic=min(10, epochs), nms=config.prediction["nms"])
    write_json(run / "execution.json", {"runtime": runtime_info(model, config), "iterations": iterations,
                                       "parameters": parameters, "test_used": False})
    # EXTENSAO: hiperparâmetro novo na busca = par novo em space (nome: (mínimo, máximo)).
    model.tune(data=str(dataset), iterations=iterations, device=config.device, val=True,
               project=str(run), name="search", space={"lr0": (0.0001, 0.003), "mosaic": (0.0, 0.5)},
               **parameters)
    review_tuning_result(run, iterations)
    return run


def review_tuning_result(run: Path, expected_iterations: int) -> dict:
    """O tuner pode retornar normalmente mesmo se todos os treinos falharem.

    Lê ``search/tune_results.ndjson`` (uma linha JSON por tentativa) e conta as
    tentativas com métricas em todos os datasets e nenhuma falha registrada.

    Parâmetros:
        run (Path): pasta da execução do ``tune``.
        expected_iterations (int): número de tentativas pedido.

    Retorna:
        dict: o ``summary.json`` gravado (``attempted``, ``successful``,
        ``expected``, ``complete``, presença de ``best_hyperparameters.yaml`` e
        ``weights/best.pt``, ``test_used`` False, ``human_acceptance`` "pendente").

    Erros:
        ValueError: "Tuning incompleto: x/y tentativas..." se faltar tentativa
            bem-sucedida ou artefato.
    """
    search = run / "search"
    history = search / "tune_results.ndjson"
    records = []
    # NDJSON: um objeto JSON por linha; linhas vazias são ignoradas.
    if history.is_file():
        records = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
    successful = 0
    # Tentativa bem-sucedida: todo dataset tem métricas não vazias e nenhum está em failed_datasets.
    for record in records:
        datasets = record.get("datasets", {})
        failed = set(record.get("failed_datasets", []))
        if datasets and all(bool(metrics) and name not in failed for name, metrics in datasets.items()):
            successful += 1
    best_yaml = search / "best_hyperparameters.yaml"
    best_weights = search / "weights/best.pt"
    # Completo = todas as tentativas registradas e bem-sucedidas + melhores parâmetros e pesos gravados.
    complete = (len(records) == expected_iterations and successful == expected_iterations
                and best_yaml.is_file() and best_weights.is_file())
    summary = {"attempted": len(records), "successful": successful,
               "expected": expected_iterations, "complete": complete,
               "best_parameters_present": best_yaml.is_file(), "best_weights_present": best_weights.is_file(),
               "test_used": False, "human_acceptance": "pendente"}
    write_json(run / "summary.json", summary)
    if not complete:
        raise ValueError(f"Tuning incompleto: {successful}/{expected_iterations} tentativas com métricas e artefatos. Revise {run}.")
    return summary
