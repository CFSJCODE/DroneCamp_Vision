"""Treino piloto em GPU do Kaggle: pacote do dataset no PC e treino a partir dele.

Função no projeto: leva o treino piloto para uma GPU (T4 do Kaggle ou outra CUDA)
sem levar as revisões, os registros nem as fotos de 20 MP. No PC, o pacote é
montado com as mesmas janelas que o ``train-pilot --tile`` geraria; na GPU, o
treino roda a partir do pacote e grava um ``runs/pilot_train_*`` igual ao do PC,
que volta para o PC e entra no ``compare-models`` como qualquer outro.

O que faz:
- ``build_package`` (comando ``kaggle-package``, roda no PC):
  - treino novo (``--data``): reconfere o dataset piloto contra as revisões humanas
    (``pilot.validate_pilot_dataset``), gera as janelas (``pilot.tiling``, mesma
    semente) e copia só as janelas e labels;
  - continuação de um treino (``--resume-run``): copia as janelas que aquele treino
    usou, o ``last.pt`` e o ``results.csv``, para terminar as épocas na GPU sem
    parar nem alterar o treino do PC;
  - grava ``package.json`` (procedência, parâmetros, contagens e SHA-256 de cada
    arquivo) e um ``.zip`` pronto para virar um Dataset privado do Kaggle.
- ``verify_package``: confere hashes e inventário do pacote antes do treino.
- ``train_from_package`` (comando ``kaggle-train``, roda no Kaggle): treina com os
  parâmetros gravados no pacote (``training`` < ``pilot.training`` do PC) mais os
  ajustes de GPU (AMP, batch, workers), ou retoma o checkpoint do pacote.
- ``rebase_checkpoint``: cópia do ``last.pt`` com dados, pasta e device da máquina
  nova (o original não é alterado).

Quando mexer: para mudar o que vai no pacote, os ajustes padrão de GPU
(``GPU_DEFAULTS``) ou o limite de horas da sessão (``--max-hours``).

Quem chama: ``cli.py`` (``kaggle-package`` e ``kaggle-train``) e o notebook
``kaggle/treino_t4.ipynb``. Guia: ``docs/treino_kaggle.md``.

O que lê e grava:
- ``build_package`` lê o dataset piloto e os registros (treino novo) ou a pasta
  ``runs/pilot_train_*`` (continuação); grava ``<saída>/dronecamp_kaggle_<data>_<id>/``
  e ``<saída>/dronecamp_kaggle_<data>_<id>.zip``. Nada no dataset ou no run de
  origem é alterado.
- ``train_from_package`` só lê o pacote (no Kaggle ele fica em ``/kaggle/input``,
  somente leitura) e grava ``runs/pilot_train_<data>_<id>/`` com ``summary.json``,
  ``execution.json``, ``package_audit.json``, ``dataset_resolved.yaml`` e ``fit/``.

ATENCAO: o pacote contém recortes das fotos do cliente. Enviá-lo ao Kaggle (mesmo
como Dataset privado) depende de autorização do cliente; este módulo não envia nada.
"""

from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import os
import shutil
import subprocess
import time

import yaml

from .config import ProjectConfig, detection_names, load_taxonomy
from .io import file_hash, make_run_directory, write_json

# Tipo gravado em package.json; o treino recusa pacotes de outro tipo/versão.
PACKAGE_KIND = "dronecamp_kaggle_package_v1"
SPLITS = ("train", "val", "test")
# Ajustes aplicados quando o treino roda em GPU (o project.yaml foi calibrado para a CPU
# do 4600G: amp desligado, batch 4, workers 0). batch 16 numa T4 de 16 GB com yolo26x em
# 640 deixa folga; a Ultralytics acumula gradientes até nbs=64, então o lote efetivo do
# otimizador não muda. Opções da linha de comando vencem estes valores.
GPU_DEFAULTS = {"amp": True, "batch": 16, "workers": 4}
# Chaves de pilot.training que não são argumentos da Ultralytics (tratadas à parte).
NON_ULTRALYTICS_KEYS = ("cpu_threads", "tile", "repeat_factor_threshold")


# ---------------------------------------------------------------------------
# Montagem do pacote (PC).
# ---------------------------------------------------------------------------

# ENTRADA: comando kaggle-package (cli.py).
def build_package(config: ProjectConfig, output: Path, data_path: Path | None = None,
                  resume_run: Path | None = None, tile: bool = True, make_zip: bool = True) -> Path:
    """Monte um pacote autossuficiente para treinar o piloto numa GPU fora do PC.

    Parâmetros:
        config (ProjectConfig): configuração do projeto (taxonomia, ``pilot.tiling``,
            ``training`` e ``pilot.training``).
        output (Path): pasta onde o pacote e o ``.zip`` são criados (criada se faltar).
        data_path (Path | None): ``dataset.yaml`` do piloto (``build-pilot-data``)
            para um treino NOVO.
        resume_run (Path | None): pasta ``runs/pilot_train_*`` para CONTINUAR o treino
            a partir do ``last.pt`` dela. Informe exatamente um entre ``data_path`` e
            ``resume_run``.
        tile (bool): treino novo em janelas (padrão, como ``train-pilot --tile``);
            False copia as fotos inteiras.
        make_zip (bool): também grava ``<pacote>.zip`` ao lado da pasta.

    Retorna:
        Path: a pasta do pacote (``<output>/dronecamp_kaggle_<data>_<id>``).

    Erros:
        ValueError: as duas origens ou nenhuma; dataset piloto reprovado; run sem
            ``last.pt`` ou sem o YAML do dataset; arquivo de treino fora do dataset.
    """
    if bool(data_path) == bool(resume_run):
        raise ValueError("Informe --data (treino novo) ou --resume-run (continuar um treino), não os dois.")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = output / f"dronecamp_kaggle_{stamp}_{os.urandom(4).hex()}"
    target.mkdir()
    try:
        if data_path:
            details = _package_fresh(config, Path(data_path).resolve(), target, tile)
        else:
            details = _package_resume(config, Path(resume_run).resolve(), target)
        # Cópia das configurações usadas para montar o pacote (só para auditoria; o treino
        # usa os parâmetros e as classes gravados em package.json).
        (target / "configs").mkdir()
        shutil.copy2(config.root / "configs" / "project.yaml", target / "configs" / "project.yaml")
        shutil.copy2(config.taxonomy_path, target / "configs" / config.taxonomy_path.name)
        manifest = {
            "kind": PACKAGE_KIND, "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(config.root), "taxonomy_sha256": file_hash(config.taxonomy_path),
            "pilot": True, "production_ready": False, **details,
            "counts": dataset_counts(target / "dataset", details["names"]),
            # Inventário com hash de todos os arquivos (menos o próprio package.json).
            "files": [{"path": path.relative_to(target).as_posix(), "sha256": file_hash(path)}
                      for path in sorted(target.rglob("*")) if path.is_file()],
        }
        write_json(target / "package.json", manifest)
    except Exception:
        # Pacote pela metade nunca fica para trás com cara de pronto.
        shutil.rmtree(target, ignore_errors=True)
        raise
    if make_zip:
        # Conteúdo na raiz do zip: o Kaggle descompacta em /kaggle/input/<dataset>/.
        shutil.make_archive(str(target), "zip", root_dir=target)
    return target


def _package_fresh(config: ProjectConfig, data_path: Path, target: Path, tile: bool) -> dict:
    """Treino novo: audita o piloto, gera as janelas e devolve os campos do package.json.

    Parâmetros:
        config (ProjectConfig): configuração do projeto.
        data_path (Path): ``dataset.yaml`` do piloto.
        target (Path): pasta do pacote; as janelas vão para ``target/dataset``.
        tile (bool): gerar janelas (True) ou copiar as fotos inteiras.

    Retorna:
        dict: ``mode``, ``names``, ``parameters``, ``source``, ``audit`` e ``tiling``.
    """
    from .pilot import validate_pilot_dataset

    # Mesma auditoria do train-pilot: hashes, aprovação humana e labels recalculadas.
    audit = validate_pilot_dataset(config, data_path)
    names = detection_names(load_taxonomy(config.taxonomy_path))
    # Mesma junção do train_pilot (training < pilot.training); a GPU ajusta depois.
    parameters = {**config.training, **config.pilot.get("training", {})}
    parameters["nms"] = config.prediction["nms"]
    parameters["model"] = config.pilot.get("model") or config.model
    # Classes raras: registrado aqui e aplicado no treino (a lista depende dos caminhos de lá).
    threshold = (config.pilot.get("sampling") or {}).get("repeat_factor_threshold")
    if threshold is not None:
        parameters["repeat_factor_threshold"] = threshold
    for key in ("cpu_threads", "tile"):
        parameters.pop(key, None)
    tiling = None
    if tile:
        from .tiling import build_tiled_dataset

        plan = config.pilot.get("tiling") or {}
        # Mesmos parâmetros e semente de training._tiled_training_dataset: as janelas são as
        # mesmas que o train-pilot --tile geraria no PC.
        stats = build_tiled_dataset(data_path.parent, target / "dataset", int(plan.get("patch", 1280)),
                                    float(plan.get("overlap", 0.2)), float(plan.get("negative_ratio", 0.2)),
                                    int(parameters.get("seed", 0)))
        tiling = {"patch": int(plan.get("patch", 1280)), "overlap": float(plan.get("overlap", 0.2)),
                  "negative_ratio": float(plan.get("negative_ratio", 0.2)), "stats": stats}
        # Caminho absoluto do PC não vai para o pacote.
        manifest_path = target / "dataset" / "tiling_manifest.json"
        tiling_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tiling_manifest["source_dataset"] = data_path.parent.name
        write_json(manifest_path, tiling_manifest)
    else:
        original = yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))
        root = Path(original.get("path", "."))
        root = root if root.is_absolute() else (data_path.parent / root).resolve()
        _copy_splits(root, {split: root / original[split] for split in SPLITS if original.get(split)}, target / "dataset")
        _write_package_yaml(target / "dataset", names, {split: f"images/{split}" for split in SPLITS
                                                         if (target / "dataset" / "images" / split).is_dir()})
    return {"mode": "fresh", "names": names, "parameters": parameters, "audit": audit, "tiling": tiling,
            "source": {"dataset": data_path.parent.name, "dataset_yaml_sha256": file_hash(data_path),
                       "pilot_sha256": audit["pilot_sha256"]}}


def _package_resume(config: ProjectConfig, run: Path, target: Path) -> dict:
    """Continuação: copia as janelas do treino, o last.pt e o results.csv daquele run.

    Parâmetros:
        config (ProjectConfig): configuração do projeto (usada para achar a área de gravação).
        run (Path): pasta ``runs/pilot_train_*`` do treino a continuar.
        target (Path): pasta do pacote.

    Retorna:
        dict: ``mode``, ``names``, ``parameters``, ``source`` e ``checkpoint``.

    ATENCAO: o treino do PC continua intocado (só leitura). Se ele seguir rodando, as
    duas cópias divergem a partir da época do last.pt copiado; compare as duas no fim.
    """
    from .training import _fit_staging

    # last.pt mais novo: na área de gravação (DRONECAMP_FIT_STAGING) quando existir, como no resume_pilot.
    fit = run / "fit"
    staging = _fit_staging(run)
    if staging and (staging / "weights" / "last.pt").is_file():
        fit = staging
    last = fit / "weights" / "last.pt"
    if not last.is_file():
        raise ValueError(f"Checkpoint last.pt não encontrado em {last}.")
    # YAML que o treino de fato usou: o das janelas (--tile) ou o das fotos.
    dataset_yaml = next((run / name for name in ("dataset_tiled_resolved.yaml", "dataset_resolved.yaml")
                         if (run / name).is_file()), None)
    if dataset_yaml is None:
        raise ValueError(f"{run} não tem dataset_tiled_resolved.yaml nem dataset_resolved.yaml.")
    resolved = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8-sig"))
    root = Path(resolved["path"])
    names = list(resolved["names"].values()) if isinstance(resolved["names"], dict) else list(resolved["names"])
    sources, train_list = {}, None
    for split in SPLITS:
        if not resolved.get(split):
            continue
        value = Path(resolved[split])
        if value.suffix == ".txt":
            # Treino com classes raras repetidas: lista de fotos; a pasta copiada é images/train.
            train_list = [Path(line.strip()) for line in value.read_text(encoding="utf-8").splitlines() if line.strip()]
            value = root / "images" / split
        sources[split] = value
    _copy_splits(root, sources, target / "dataset")
    splits = {split: f"images/{split}" for split in sources}
    if train_list is not None:
        # "./" no início: a Ultralytics resolve a linha a partir da pasta do .txt.
        lines = []
        for path in train_list:
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Foto da lista de treino fora do dataset: {path}.")
            lines.append("./" + path.resolve().relative_to(root.resolve()).as_posix())
        (target / "dataset" / "train_list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        splits["train"] = "train_list.txt"
    _write_package_yaml(target / "dataset", names, splits)
    (target / "checkpoint").mkdir()
    shutil.copy2(last, target / "checkpoint" / "last.pt")
    # ATENCAO: com o treino do PC rodando, a cópia pode pegar o last.pt no meio da gravação
    # do fim de uma época; uma cópia que não abre é recusada (rode de novo em seguida).
    epoch = _checkpoint_epoch(target / "checkpoint" / "last.pt")
    if epoch is None:
        raise ValueError("A cópia do last.pt não abriu (o treino pode estar gravando a época); tente de novo.")
    if (fit / "results.csv").is_file():
        # Só as épocas do checkpoint copiado: o treino do PC pode ter fechado outra época
        # entre a cópia do last.pt e a do results.csv.
        copy_results_until(fit / "results.csv", target / "checkpoint" / "results.csv", epoch)
    # Relatórios do run de origem que explicam o que está sendo continuado.
    (target / "source_run").mkdir()
    for name in ("execution.json", "summary.json", "pilot_provenance.json", "tiling_manifest.json", "sampling.json"):
        if (run / name).is_file():
            shutil.copy2(run / name, target / "source_run" / name)
    execution = json.loads((run / "execution.json").read_text(encoding="utf-8")) if (run / "execution.json").is_file() else {}
    return {"mode": "resume", "names": names, "parameters": execution.get("parameters", {}),
            "source": {"run": run.name, "checkpoint_sha256": file_hash(target / "checkpoint" / "last.pt")},
            "checkpoint": {"path": "checkpoint/last.pt", "epoch_completed": epoch}}


def _copy_splits(root: Path, sources: dict[str, Path], destination: Path) -> None:
    """Copia ``images/<split>`` e ``labels/<split>`` de cada split para o pacote.

    Parâmetros:
        root (Path): raiz do dataset de origem (onde ficam ``images/`` e ``labels/``).
        sources (dict[str, Path]): ``{split: pasta de imagens}`` (dentro de ``root/images``).
        destination (Path): ``<pacote>/dataset``.

    Erros:
        ValueError: pasta de imagens fora de ``root/images``.
    """
    for split, image_dir in sources.items():
        image_dir = Path(image_dir).resolve()
        relative = image_dir.relative_to((root / "images").resolve())
        label_dir = root / "labels" / relative
        shutil.copytree(image_dir, destination / "images" / split)
        if label_dir.is_dir():
            shutil.copytree(label_dir, destination / "labels" / split)


def _write_package_yaml(dataset: Path, names: list[str], splits: dict[str, str]) -> None:
    """Grava ``dataset.yaml`` com caminhos relativos (``path: .``), válido em qualquer máquina."""
    content = {"path": ".", **splits, "nc": len(names), "names": names}
    (dataset / "dataset.yaml").write_text(yaml.safe_dump(content, sort_keys=False, allow_unicode=True), encoding="utf-8")


def dataset_counts(dataset: Path, names: list[str]) -> dict:
    """Conta imagens e caixas por classe em cada split do pacote.

    Retorna:
        dict: ``{split: {"images": n, "boxes": {classe: n}}}`` (só splits presentes).
    """
    counts = {}
    for split in SPLITS:
        image_dir, label_dir = dataset / "images" / split, dataset / "labels" / split
        if not image_dir.is_dir():
            continue
        boxes = {name: 0 for name in names}
        for label in label_dir.rglob("*.txt") if label_dir.is_dir() else ():
            for line in label.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    boxes[names[int(line.split()[0])]] += 1
        counts[split] = {"images": sum(1 for path in image_dir.rglob("*") if path.is_file()), "boxes": boxes}
    return counts


def copy_results_until(source: Path, destination: Path, epoch: int | None) -> int:
    """Copia o ``results.csv`` só até a época ``epoch`` (1-based); None copia tudo.

    Retorna:
        int: linhas de época copiadas.
    """
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return 0
    header, body = rows[0], rows[1:]
    column = next((index for index, name in enumerate(header) if name.strip() == "epoch"), None)
    if epoch is not None and column is not None:
        body = [row for row in body if row and float(row[column]) <= epoch]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows([header, *body])
    return len(body)


def _git_commit(root: Path) -> str | None:
    """Commit do código que montou o pacote (None fora de um clone git)."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _checkpoint_epoch(path: Path) -> int | None:
    """Última época concluída gravada no checkpoint (1-based); None se não der para ler."""
    try:
        import torch

        epoch = torch.load(path, map_location="cpu", weights_only=False).get("epoch")
        return int(epoch) + 1 if epoch is not None and int(epoch) >= 0 else None
    except Exception:  # checkpoint ilegível aqui não impede o pacote; o treino confere de novo
        return None


# ---------------------------------------------------------------------------
# Conferência e treino (Kaggle ou qualquer máquina com CUDA).
# ---------------------------------------------------------------------------

def find_package(path: Path) -> Path:
    """Acha a pasta do pacote: a própria ``path`` ou a única subpasta com ``package.json``.

    O Kaggle às vezes descompacta o zip dentro de mais uma pasta; esta função aceita
    ``/kaggle/input/<dataset>`` diretamente.

    Erros:
        ValueError: nenhum ou mais de um ``package.json`` encontrado.
    """
    path = Path(path).expanduser().resolve()
    if (path / "package.json").is_file():
        return path
    found = sorted(candidate.parent for candidate in path.rglob("package.json"))
    if len(found) != 1:
        raise ValueError(f"Esperado exatamente um package.json em {path}; encontrados {len(found)}.")
    return found[0]


def verify_package(package: Path) -> dict:
    """Confira tipo, hashes e inventário do pacote antes de treinar.

    Parâmetros:
        package (Path): pasta do pacote (com ``package.json``).

    Retorna:
        dict: o conteúdo de ``package.json``.

    Erros:
        ValueError: tipo desconhecido, arquivo ausente/alterado ou arquivo a mais em ``dataset/``.
    """
    manifest = json.loads((package / "package.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != PACKAGE_KIND:
        raise ValueError(f"package.json de outro tipo: {manifest.get('kind')!r}.")
    listed = set()
    for entry in manifest["files"]:
        path = package / entry["path"]
        if not path.resolve().is_relative_to(package.resolve()):
            raise ValueError(f"Arquivo fora do pacote: {entry['path']}.")
        if not path.is_file() or file_hash(path) != entry["sha256"]:
            raise ValueError(f"Arquivo ausente ou alterado no pacote: {entry['path']}.")
        listed.add(entry["path"])
    # Arquivo a mais no dataset entraria no treino sem estar no inventário.
    # Os .cache da Ultralytics (gravados ao lado das labels quando a pasta permite) não contam.
    extra = [path.relative_to(package).as_posix() for path in (package / "dataset").rglob("*")
             if path.is_file() and path.suffix != ".cache" and path.relative_to(package).as_posix() not in listed]
    if extra:
        raise ValueError(f"Arquivos fora do inventário no pacote: {extra[:5]}.")
    return manifest


def rebase_checkpoint(source: Path, destination: Path, updates: dict) -> Path:
    """Grava uma cópia do checkpoint com ``train_args`` atualizados para a máquina nova.

    A Ultralytics retoma com os argumentos guardados no ``last.pt`` (dataset em ``E:/``,
    pasta de saída em ``C:\\DroneCamp_runs``, ``device=cpu``, ``amp=False``). Só alguns
    podem ser trocados na chamada; ``amp`` não pode, por isso a cópia.

    Parâmetros:
        source (Path): ``last.pt`` original (não é alterado).
        destination (Path): onde gravar a cópia.
        updates (dict): chaves de ``train_args`` a trocar (ex.: ``data``, ``device``, ``amp``).

    Retorna:
        Path: ``destination``.
    """
    import torch

    # weights_only=False: checkpoint da Ultralytics gerado pelo próprio projeto (procedência no pacote).
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    checkpoint["train_args"] = {**(checkpoint.get("train_args") or {}), **updates}
    torch.save(checkpoint, destination)
    return destination


def align_optimizer_state(optimizer) -> int:
    """Põe o estado do otimizador no mesmo dispositivo, tipo e layout de memória de cada parâmetro.

    ATENCAO: um checkpoint do PC (CPU, Windows) guarda ``exp_avg``/``exp_avg_sq`` em NCHW
    (``exp_avg`` em float16, pela Ultralytics). Na GPU Linux a Ultralytics liga
    ``channels_last`` sozinha, e o AdamW fundido da CUDA recusa estado com strides
    diferentes dos pesos ("must have same dtype, device, and layout"). Copiar para
    ``torch.empty_like(param)`` herda o layout do peso sem mudar os valores.

    Parâmetros:
        optimizer: otimizador já com o ``load_state_dict`` do checkpoint.

    Retorna:
        int: quantos tensores foram ajustados.
    """
    import torch

    changed = 0
    for group in optimizer.param_groups:
        for param in group["params"]:
            for key, value in optimizer.state.get(param, {}).items():
                if key == "step" or not isinstance(value, torch.Tensor) or value.shape != param.shape:
                    continue
                if (value.device, value.dtype, value.stride()) != (param.device, param.dtype, param.stride()):
                    optimizer.state[param][key] = torch.empty_like(param).copy_(value)
                    changed += 1
    return changed


# ENTRADA: comando kaggle-train (cli.py) e notebook kaggle/treino_t4.ipynb.
def train_from_package(config: ProjectConfig, package: Path, device: str = "0", overrides: dict | None = None,
                       fresh: bool = False, weights: str | None = None, checkpoint: Path | None = None,
                       max_hours: float | None = None) -> Path:
    """Treine (ou continue) o piloto a partir de um pacote do ``kaggle-package``.

    Etapas (numeradas igual no corpo):
        1. acha e confere o pacote; cria ``runs/pilot_train_<data>_<id>``;
        2. grava ``dataset_resolved.yaml`` com caminhos absolutos do pacote;
        3a. continuação: cópia do checkpoint com dados/pasta/device novos e
            ``model.train(resume=True)``; ou
        3b. treino novo: parâmetros do pacote + ajustes de GPU + opções, e ``model.train``;
        4. limite de horas: para depois de uma época salva e guarda ``weights/resume.pt``;
        5. auditoria final (``review_training_result``) ou ``summary.json`` "interrupted".

    Parâmetros:
        config (ProjectConfig): configuração do projeto clonado (raiz para ``runs/`` e ``models/``).
        package (Path): pasta do pacote, ou uma pasta acima dele (``/kaggle/input/<dataset>``).
        device (str): ``"0"`` (primeira GPU), ``"0,1"`` ou ``"cpu"``.
        overrides (dict | None): ``epochs``, ``batch``, ``workers``, ``amp``... (vencem o pacote).
            Na continuação só valem os que a Ultralytics aceita ao retomar (batch, workers, amp aqui).
        fresh (bool): ignora o checkpoint de um pacote de continuação e treina do zero.
        weights (str | None): peso inicial do treino novo (padrão: o gravado no pacote).
        checkpoint (Path | None): ``last.pt``/``resume.pt`` de uma sessão anterior (continua
            dali em vez do checkpoint do pacote; o ``results.csv`` ao lado é reaproveitado).
        max_hours (float | None): para com segurança depois da primeira época que passar
            deste tempo (a sessão do Kaggle cai em 12 h).

    Retorna:
        Path: a pasta da execução.

    Erros:
        ValueError: pacote inválido; continuação pedida sem checkpoint; falha no treino.
    """
    from .backend import load_detector, runtime_info
    from .training import _apply_repeat_factor_sampling, _write_training_summary, review_training_result

    # 1. Pacote conferido antes de qualquer gravação; execução nova em runs/.
    package = find_package(package)
    manifest = verify_package(package)
    names = list(manifest["names"])
    run = make_run_directory(config.root, "pilot_train")
    details = {"pilot": True, "production_ready": False, "trained_on": "kaggle_package",
               "package": {"package_sha256": file_hash(package / "package.json"), "mode": manifest["mode"],
                           "source": manifest.get("source"), "git_commit": manifest.get("git_commit")},
               # Código que treinou (pode ser mais novo que o que montou o pacote).
               "runner_git_commit": _git_commit(config.root)}
    _write_training_summary(run, ["O treino a partir do pacote ainda não concluiu."], state="running", **details)
    write_json(run / "package_audit.json", {k: v for k, v in manifest.items() if k != "files"})
    overrides = dict(overrides or {})
    gpu = str(device).lower() != "cpu"
    # 2. YAML com caminhos absolutos do pacote (somente leitura no Kaggle; a Ultralytics só lê).
    dataset_root = package / "dataset"
    original = yaml.safe_load((dataset_root / "dataset.yaml").read_text(encoding="utf-8"))
    resolved = {"path": str(dataset_root), "names": names,
                **{split: str(dataset_root / original[split]) for split in SPLITS if original.get(split)}}
    dataset = run / "dataset_resolved.yaml"
    dataset.write_text(yaml.safe_dump(resolved, allow_unicode=True), encoding="utf-8")
    source_checkpoint = Path(checkpoint).resolve() if checkpoint else (
        package / manifest["checkpoint"]["path"] if manifest.get("checkpoint") and not fresh else None)
    state = {"stopped_by_time": False, "started": time.time()}

    def time_guard(trainer):
        """Callback ``on_fit_epoch_end`` (depois do last.pt da época): para antes de a sessão cair.

        A Ultralytics remove o otimizador do ``last.pt`` ao encerrar; por isso a cópia
        completa vai para ``weights/resume.pt``, que é o que a próxima sessão retoma.
        """
        if not max_hours or state["stopped_by_time"] or trainer.epoch + 1 >= trainer.epochs:
            return
        if time.time() - state["started"] >= float(max_hours) * 3600 and Path(trainer.last).is_file():
            shutil.copy2(trainer.last, Path(trainer.last).with_name("resume.pt"))
            state["stopped_by_time"] = True
            trainer.stop = True

    try:
        if source_checkpoint:
            # 3a. Continuação: results.csv anterior na pasta nova (a Ultralytics acrescenta linhas).
            if not source_checkpoint.is_file():
                raise ValueError(f"Checkpoint não encontrado: {source_checkpoint}.")
            if checkpoint:
                _check_lineage(source_checkpoint, details["package"]["package_sha256"])
            (run / "fit").mkdir()
            previous_csv = source_checkpoint.parent / "results.csv" if source_checkpoint.parent.name == "checkpoint" \
                else source_checkpoint.parent.parent / "results.csv"
            if previous_csv.is_file():
                copy_results_until(previous_csv, run / "fit" / "results.csv", _checkpoint_epoch(source_checkpoint))
            updates = {"data": str(dataset), "device": device, "project": str(run), "name": "fit",
                       "save_dir": str(run / "fit"), "exist_ok": True}
            if gpu:
                updates.update(GPU_DEFAULTS)
            updates.update({key: value for key, value in overrides.items() if key in ("batch", "workers", "amp")})
            rebased = rebase_checkpoint(source_checkpoint, run / "checkpoint_rebased.pt", updates)
            model = load_detector(config, str(rebased))
            parameters = {"resume": True, **updates}
            details["resumed_from"] = {"checkpoint": str(source_checkpoint), "sha256": file_hash(source_checkpoint),
                                       "epoch_completed": _checkpoint_epoch(source_checkpoint)}
            write_json(run / "execution.json", {"runtime": runtime_info(model, config), "parameters": parameters,
                                                "device": device, **details})
            model.add_callback("on_fit_epoch_end", time_guard)
            # Depois do resume_training (que carrega o otimizador) e antes do primeiro passo.
            model.add_callback("on_train_start", lambda trainer: state.update(
                optimizer_tensors_aligned=align_optimizer_state(trainer.optimizer)))
            # Argumentos explícitos além da cópia: data/device/save_dir/batch/workers são aceitos
            # pela Ultralytics ao retomar (BaseTrainer.check_resume).
            model.train(resume=True, data=str(dataset), device=device, save_dir=str(run / "fit"),
                        **{key: updates[key] for key in ("batch", "workers") if key in updates})
            details["optimizer_tensors_aligned"] = state.get("optimizer_tensors_aligned")
        else:
            # 3b. Treino novo com os parâmetros que o PC usaria, mais os ajustes de GPU.
            parameters = dict(manifest["parameters"])
            base = weights or parameters.pop("model", None) or config.pilot.get("model") or config.model
            parameters.pop("model", None)
            threshold = overrides.pop("repeat_factor_threshold", parameters.get("repeat_factor_threshold"))
            for key in NON_ULTRALYTICS_KEYS:
                parameters.pop(key, None)
            if gpu:
                parameters.update(GPU_DEFAULTS)
            parameters.update(overrides)
            if threshold is not None:
                details["sampling"] = _apply_repeat_factor_sampling(run, dataset, float(threshold),
                                                                    int(parameters.get("seed", 0)))
            model = load_detector(config, base)
            write_json(run / "execution.json", {"runtime": runtime_info(model, config), "parameters": parameters,
                                                "device": device, "base_weights": base, **details})
            model.add_callback("on_fit_epoch_end", time_guard)
            model.train(data=str(dataset), device=device, project=str(run), name="fit", exist_ok=False, **parameters)
    except Exception as error:
        reason = f"Execução interrompida por {type(error).__name__}: {error}"
        _write_training_summary(run, [reason], state="failed", **details)
        raise ValueError(f"Treino a partir do pacote incompleto: {reason}. Revise {run}.") from error
    finally:
        # A cópia rebaseada (centenas de MB) não é resultado; o checkpoint de origem continua no pacote.
        (run / "checkpoint_rebased.pt").unlink(missing_ok=True)
    if state["stopped_by_time"]:
        # 4. Parada pelo limite de horas: não é um treino concluído.
        epochs = _csv_epochs(run / "fit" / "results.csv")
        reason = (f"Parado pelo limite de {max_hours} h depois da época {epochs}. Continue com "
                  f"kaggle-train --checkpoint {run / 'fit' / 'weights' / 'resume.pt'}.")
        _write_training_summary(run, [reason], state="interrupted", epochs_recorded=epochs, **details)
        return run
    # 5. Mesma auditoria do train-pilot: pesos, results.csv finito e classes do pacote.
    review_training_result(run, model, names, **details)
    return run


def _check_lineage(checkpoint: Path, package_sha256: str) -> None:
    """Recusa continuar com um checkpoint treinado a partir de outro pacote.

    O ``resume.pt``/``last.pt`` de uma sessão anterior fica em ``<run>/fit/weights``; o
    ``summary.json`` daquele run registra o hash do ``package.json`` usado.

    Erros:
        ValueError: summary ausente/ilegível ou hash de pacote diferente.
    """
    summary_path = checkpoint.parent.parent.parent / "summary.json"
    try:
        recorded = json.loads(summary_path.read_text(encoding="utf-8"))["package"]["package_sha256"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Não foi possível conferir de que pacote veio {checkpoint} ({summary_path}).") from error
    if recorded != package_sha256:
        raise ValueError("O checkpoint veio de outro pacote; use o mesmo Dataset da sessão anterior.")


def _csv_epochs(path: Path) -> int:
    """Número de linhas de época no results.csv (0 se não existir)."""
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))
