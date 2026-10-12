"""Painel de operações: treinos, curvas, comparações e cobertura de classes.

Função no projeto: junta, em um único dicionário, o que a plataforma de
operações (página ``index.html`` da revisão) mostra fora da revisão de fotos:

- ``collect_runs``: cada treino em ``runs/`` com parâmetros, curvas por época
  (``fit/results.csv``), estado (concluído, em andamento, interrompido) e pesos;
- ``collect_comparisons``: arquivos ``runs/compare_*.json`` gerados por
  ``scripts/compare_pilot_models.py`` (acertos por divisão e por modelo);
- ``collect_datasets``: datasets piloto em ``data/pilot/*/dataset.yaml``;
- ``collect_operations``: tudo acima mais pesos disponíveis e padrões de treino.

Só lê arquivos: nada aqui treina, aprova dados ou altera pesos. A página recebe
uma cópia (instantâneo) ao ser gerada; o servidor local (``platform_server.py``)
chama ``collect_operations`` de novo a cada consulta para o monitoramento ao vivo.

Quando mexer: para mostrar uma métrica nova (acrescente a coluna em
``CURVE_COLUMNS``) ou um tipo novo de execução em ``runs/``.

Conteúdo do arquivo (funções públicas):
- ``run_id_from_path``: extrai o id da execução de um caminho de pesos;
- ``read_curves``: lê ``fit/results.csv`` (curvas por época);
- ``read_gpu_curve``: lê a avaliação paralela na GPU (``gpu_eval/*.jsonl``);
- ``collect_runs``: lista de treinos de ``runs/``;
- ``collect_comparisons``: comparações de modelos (``runs/compare_*.json``);
- ``collect_datasets``: datasets piloto (``data/pilot/*/dataset.yaml``);
- ``collect_review_files``: laudos e pacotes ZIP já gerados para uma revisão;
- ``collect_operations``: o instantâneo completo (``OPS`` na página).
Funções privadas: ``_read_json``, ``_number``, ``_relative``, ``_run_state``,
``_percent``, ``_name_runs``.

Quem chama:
- ``review_render.build_page``/``render-review``: instantâneo embutido no ``index.html``
  (``review_render`` também usa ``run_id_from_path``);
- ``platform_server.py``: ``GET /api/operations`` (a cada 5 s pela página),
  ``GET /api/reviews/<pasta>/files`` (``collect_review_files``) e a página servida;
- ``model_report.py``: ``RUN_NAME`` e ``read_curves``.

Disco (só leitura): ``runs/<execução>/execution.json``, ``summary.json``,
``pilot_provenance.json`` ou ``training_provenance.json``, ``fit/results.csv``,
``fit/heartbeat``, ``fit/weights/{best,last}.pt``, imagens de ``fit/``,
``gpu_eval/eval_curve.jsonl`` ou ``gpu_eval/curve.jsonl``; ``runs/nomes.json``;
``runs/compare_*.json``; ``data/pilot/*/dataset.yaml`` e ``pilot.json``;
``models/*.pt``; ``<revisão>/laudos/*/`` e ``<revisão>/exportacoes/*.zip``.

Na página (``review_templates/platform.js``), o resultado fica na variável ``OPS``
e é usado nas seções Treinamento, Monitoramento, Modelos, Início, Importar mídia
e Laudos e downloads.

Onde mexer:
- Métrica nova nas curvas: ``CURVE_COLUMNS`` (coluna do results.csv → nome curto) e
  o gráfico correspondente em platform.js (seções Treinamento/Monitoramento).
- Parâmetro de treino novo na tabela de execuções: ``SHOWN_PARAMETERS``.
- Tipo novo de execução em ``runs/`` (ex.: "finetune_..."): conjunto de estágios em
  ``collect_runs`` e, para nome legível, ``naming.STAGE_NAMES``.
- Gráfico novo de ``fit/`` (PNG/JPG da Ultralytics) na página: tupla ``plots`` em ``collect_runs``.
- Campo novo por execução: dict ``runs.append({...})`` em ``collect_runs``.
- Campo novo do instantâneo inteiro: dict devolvido por ``collect_operations``.
- Tipo novo de arquivo gerado para baixar: ``collect_review_files`` (e
  ``platform_server.DOWNLOAD_SUFFIXES`` para o servidor entregá-lo).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import math
import os
import re
import time

from .config import ProjectConfig
from .io import file_hash
from .naming import run_name

# ---------------------------------------------------------------------------
# Constantes: colunas das curvas, parâmetros mostrados, janela de "em andamento"
# e formato do nome das pastas de execução
# ---------------------------------------------------------------------------

# Colunas do results.csv da Ultralytics enviadas à página (as demais ficam no disco).
# EXTENSAO: métrica nova nas curvas da plataforma = coluna do results.csv (Ultralytics) → nome
# curto aqui; a página lê esse nome curto em OPS.runs[].curves.
# ATENCAO: "epoch" precisa continuar aqui (read_curves descarta linhas sem época) e "time"
# (segundos acumulados desde o início do treino) alimenta seconds_per_epoch em collect_runs.
# Os sufixos "(B)" são das métricas de caixa (box) da Ultralytics.
CURVE_COLUMNS = {
    "epoch": "epoch", "time": "time",
    "train/box_loss": "train_box", "train/cls_loss": "train_cls",
    "val/box_loss": "val_box", "val/cls_loss": "val_cls",
    "metrics/precision(B)": "precision", "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "map50", "metrics/mAP50-95(B)": "map50_95",
}
# Parâmetros de treino mostrados na tabela de execuções.
# Vêm de execution.json["parameters"]; um nome ausente no arquivo é simplesmente omitido.
# EXTENSAO: parâmetro novo na tabela = nome aqui (igual à chave em execution.json) e a
# coluna correspondente em platform.js (seção Treinamento).
SHOWN_PARAMETERS = ("epochs", "imgsz", "batch", "optimizer", "lr0", "patience", "mosaic", "close_mosaic", "seed")
# Sem nova época neste intervalo, um treino sem summary.json é tratado como interrompido.
# Unidade: segundos (15 minutos). Aumentar evita marcar como "interrompido" um treino lento;
# diminuir detecta mais cedo um treino que morreu.
RUNNING_WINDOW_SECONDS = 15 * 60
# Nome da pasta de execução: "<estágio>_<AAAAMMDDTHHMMSSZ>_<8 hex>",
# ex.: "pilot_train_20261007T010057Z_f5e5ba8b". Grupos: stage (letras/_; "+?" não guloso
# para o "_" antes da data não ser engolido), stamp (data UTC) e id (8 caracteres hexadecimais).
# ATENCAO: model_report.py importa esta regex; mudar o formato aqui muda o que os dois reconhecem.
RUN_NAME = re.compile(r"^(?P<stage>[a-z_]+?)_(?P<stamp>\d{8}T\d{6}Z)_(?P<id>[0-9a-f]{8})$")


# ---------------------------------------------------------------------------
# Leitura de arquivos isolados: id da execução, JSON opcional, curvas por época
# ---------------------------------------------------------------------------

def run_id_from_path(path) -> str | None:
    """Id da execução em um caminho de pesos ("runs/<execução>/fit/weights/best.pt", "/" ou "\\").

    Parâmetros:
        path: caminho (str, Path ou None), com separador "/" ou "\\" (caminhos do Windows).

    Retorna:
        O primeiro trecho do caminho que casa com ``RUN_NAME`` (ex.:
        "pilot_train_20261007T010057Z_f5e5ba8b") ou None (ex.: "models/yolo26l.pt").

    Chamada por: ``collect_comparisons`` e ``review_render`` (id dos pesos das sugestões).
    """
    # Normaliza "\" para "/", quebra em partes e devolve a primeira que parece uma pasta de execução.
    return next((part for part in str(path or "").replace("\\", "/").split("/") if RUN_NAME.match(part)), None)


def _read_json(path: Path) -> dict | None:
    """JSON opcional: arquivo ausente ou corrompido não derruba o painel.

    Parâmetros:
        path: caminho do arquivo JSON.

    Retorna:
        O conteúdo decodificado (normalmente dict) ou None se o arquivo não existe,
        não pode ser lido ou não é JSON válido.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _number(value: str) -> float | None:
    """Converte uma célula do results.csv em número, ou None se não der.

    Parâmetros:
        value: texto da célula (ex.: " 0.4312"), ou None se a coluna não existe.

    Retorna:
        float finito, ou None para célula vazia/ausente/não numérica e para "nan"/"inf"
        (o JSON da página não aceita NaN/Infinity).
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_curves(results_csv: Path) -> list[dict]:
    """Uma linha por época com as colunas de ``CURVE_COLUMNS`` (nomes curtos).

    Parâmetros:
        results_csv: caminho de ``runs/<execução>/fit/results.csv`` (gravado pela Ultralytics).

    Retorna:
        Lista de dicts ``{"epoch": int, "time": float|None, "train_box": ..., "map50": ...,
        "map50_95": ...}`` na ordem do arquivo; lista vazia se o arquivo não existe.
        Métrica ausente ou inválida vira None.

    Chamada por: ``collect_runs`` e ``model_report.py``.
    """
    # 1. Lê o CSV inteiro (arquivo ausente = treino ainda sem nenhuma época).
    try:
        with results_csv.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError:
        return []
    curves = []
    # 2. Converte cada linha (uma época).
    for row in rows:
        # A Ultralytics alinha colunas com espaços em algumas versões.
        # "if key" descarta a chave None que o DictReader cria para colunas sobrando.
        clean = {key.strip(): value for key, value in row.items() if key}
        # Renomeia as colunas para os nomes curtos e converte em número.
        point = {short: _number(clean.get(column)) for column, short in CURVE_COLUMNS.items()}
        # Só entra a linha que tem época (o CSV usa número de época a partir de 1).
        if point["epoch"] is not None:
            point["epoch"] = int(point["epoch"])
            curves.append(point)
    return curves


def read_gpu_curve(run_directory: Path) -> list[dict]:
    """Pontos da avaliação paralela na GPU: ``gpu_eval/eval_curve.jsonl`` (gpu_eval.py)
    ou ``gpu_eval/curve.jsonl`` (scripts/gpu_eval_watcher.py).

    Parâmetros:
        run_directory: pasta da execução (``runs/<execução>``).

    Retorna:
        Lista de pontos ``{"epoch": int, "map50": float, "map50_95": float|None,
        "split": str|None, "provider": str|None, "seconds": float|None}``, ordenada por
        época e depois por divisão; vazia se nenhum dos dois arquivos existe.

    Formatos aceitos (uma linha JSON por avaliação):
        - gpu_eval.py: ``{"epoch", "map50", "map50_95", "split", "provider_active"/"provider", "seconds"}``;
        - vigia: ``{"epoch", "sets": {"<divisão>": {"totals": {"map50": ...}}}, "provider", "seconds"}``.

    EXTENSAO: um terceiro formato de avaliação = um ``elif`` novo no laço das linhas.
    """
    points = []
    # Lê os dois arquivos possíveis (se ambos existirem, os pontos se somam).
    for name in ("eval_curve.jsonl", "curve.jsonl"):
        path = run_directory / "gpu_eval" / name
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            # Linha corrompida (ex.: gravação interrompida no meio) é ignorada.
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            # Precisa ser um objeto com época inteira.
            if not isinstance(entry, dict) or not isinstance(entry.get("epoch"), int):
                continue
            # Formato do gpu_eval.py: uma divisão por linha, mAP50 direto.
            if entry.get("map50") is not None:
                points.append({"epoch": entry["epoch"], "map50": entry["map50"], "map50_95": entry.get("map50_95"),
                               "split": entry.get("split"), "provider": entry.get("provider_active") or entry.get("provider"),
                               "seconds": entry.get("seconds")})
            elif isinstance(entry.get("sets"), dict):  # formato do vigia: um bloco por conjunto
                # Um ponto por divisão (val, test...), com o mAP50 de "totals".
                for split, values in entry["sets"].items():
                    map50 = (values.get("totals") or {}).get("map50")
                    if map50 is not None:
                        points.append({"epoch": entry["epoch"], "map50": map50, "map50_95": None, "split": split,
                                       "provider": entry.get("provider"), "seconds": entry.get("seconds")})
    # str(split) porque split pode ser None (não dá para comparar None com texto).
    return sorted(points, key=lambda point: (point["epoch"], str(point["split"])))


def _relative(path: Path, start: Path) -> str:
    """Caminho relativo com "/" para links da página (funciona aberto do disco).

    Parâmetros:
        path: arquivo de destino.
        start: pasta de referência (raiz ``sistema_ia`` ou pasta da página da revisão).

    Retorna:
        Caminho relativo com "/" (ex.: "../../runs/<execução>/fit/results.png"); pode
        conter ".." quando o arquivo está fora de ``start``.
    """
    return Path(os.path.relpath(path, start)).as_posix()


def _run_state(summary: dict | None, results_csv: Path, now: float) -> str:
    """completed / running / interrupted / failed a partir do que existe no disco.

    Parâmetros:
        summary: conteúdo de ``summary.json`` (gravado no fim do treino) ou None.
        results_csv: caminho de ``fit/results.csv``.
        now: horário atual em segundos (``time.time()``), para comparar com a data dos arquivos.

    Retorna:
        "completed" (summary diz concluído), "failed" (summary diz falha ou incompleto),
        "running" (results.csv ou heartbeat mudou há menos de ``RUNNING_WINDOW_SECONDS``),
        "interrupted" (há results.csv antigo, sem summary) ou "prepared" (nenhuma época ainda).

    EXTENSAO: estado novo = um retorno novo aqui e o texto/ícone correspondente em platform.js.
    """
    # 1. Com summary.json, o próprio treino declarou como terminou.
    if summary:
        if summary.get("state") == "completed" and summary.get("complete", True):
            return "completed"
        if summary.get("state") == "failed" or summary.get("complete") is False:
            return "failed"
    # fit/heartbeat é tocado a cada minuto pelo treino (training.py): uma época longa
    # deixa results.csv parado por mais que a janela sem que o treino tenha morrido.
    # 2. Sem summary conclusivo: algum dos dois arquivos mudou recentemente = em andamento.
    for marker in (results_csv, results_csv.parent / "heartbeat"):
        if marker.is_file() and now - marker.stat().st_mtime < RUNNING_WINDOW_SECONDS:
            return "running"
    # 3. Parado: com épocas gravadas = interrompido; sem nenhuma = só preparado.
    return "interrupted" if results_csv.is_file() else "prepared"


# ---------------------------------------------------------------------------
# Execuções de treino (runs/)
# ---------------------------------------------------------------------------

def collect_runs(root: Path, page_directory: Path | None = None, now: float | None = None) -> list[dict]:
    """Execuções de treino em ``runs/`` (mais recente primeiro).

    Parâmetros:
        root: pasta ``sistema_ia`` (raiz do projeto).
        page_directory: pasta da página da revisão; os links de imagens (``plots``) ficam
            relativos a ela, para funcionarem com a página aberta do disco. None = relativos a ``root``.
        now: horário atual em segundos (para testes); None usa ``time.time()``.

    Retorna:
        Lista de dicts, um por execução, com ``id``, ``stage``, ``started_at``, ``state``,
        ``parameters``, ``checkpoint``, ``device``, ``dataset_sha256``, ``buildings``,
        ``independent_evaluation``, ``class_counts``, ``epochs_planned``, ``epochs_done``,
        ``seconds_per_epoch``, ``best_epoch``, ``last``, ``curves``, ``gpu_eval``,
        ``label_unit``, ``weights`` (relativos a ``root``), ``plots`` (relativos a
        ``page_directory``), ``notes``, ``updated_at``, ``display_name`` e ``display_detail``.
        Lista vazia se ``runs/`` não existe.

    Chamada por: ``collect_operations``.
    """
    now = time.time() if now is None else now
    runs_dir = root / "runs"
    if not runs_dir.is_dir():
        return []
    base = page_directory or root
    runs = []
    # Uma passada por pasta dentro de runs/.
    for directory in runs_dir.iterdir():
        # 1. Só pastas com nome de execução e de um estágio de treino.
        # EXTENSAO: novo tipo de execução mostrado na plataforma = estágio novo neste conjunto
        # (e nome legível em naming.STAGE_NAMES).
        match = RUN_NAME.match(directory.name)
        if not directory.is_dir() or not match or match["stage"] not in {"pilot_train", "train", "tune"}:
            continue
        # 2. Arquivos da execução (todos opcionais; ausentes viram {} ou None).
        fit = directory / "fit"
        results_csv = fit / "results.csv"
        execution = _read_json(directory / "execution.json") or {}
        summary = _read_json(directory / "summary.json")
        # Proveniência: treino piloto ou treino de produção, o que existir.
        provenance = _read_json(directory / "pilot_provenance.json") or _read_json(directory / "training_provenance.json") or {}
        parameters = execution.get("parameters") or {}
        # 3. Curvas e estado.
        curves = read_curves(results_csv)
        state = _run_state(summary, results_csv, now)
        # Melhor época pelo mAP50 de validação (o mesmo critério do best.pt é mAP50-95 + mAP50).
        # 4. Melhor época: maior mAP50; empate decidido pelo mAP50-95 (None conta como 0).
        scored = [point for point in curves if point.get("map50") is not None]
        best = max(scored, key=lambda point: (point["map50"], point.get("map50_95") or 0), default=None)
        epochs_planned = parameters.get("epochs")
        # 5. Segundos por época = tempo acumulado da última linha ÷ número da última época
        # (max(1, ...) evita divisão por zero se a primeira época vier numerada 0).
        seconds_per_epoch = None
        if curves and curves[-1].get("time"):
            seconds_per_epoch = curves[-1]["time"] / max(1, curves[-1]["epoch"])
        # 6. Pesos existentes (best.pt/last.pt), relativos à raiz: é o que _job_arguments aceita.
        weights = {name: _relative(fit / "weights" / f"{name}.pt", root)
                   for name in ("best", "last") if (fit / "weights" / f"{name}.pt").is_file()}
        # 7. Imagens geradas pela Ultralytics em fit/, relativas à página (só as que existem).
        # EXTENSAO: imagem nova da Ultralytics na página = par ("nome_curto", "arquivo") nesta tupla;
        # a página lê run.plots.<nome_curto>.
        plots = {name: _relative(fit / file, base) for name, file in (
            ("results", "results.png"), ("confusion", "confusion_matrix_normalized.png"),
            ("pr_curve", "BoxPR_curve.png"), ("val_pred", "val_batch0_pred.jpg"), ("labels", "labels.jpg"))
            if (fit / file).is_file()}
        # 8. Linha da execução enviada à página (OPS.runs[]).
        # EXTENSAO: campo novo por execução = chave nova neste dict; a página lê em OPS.runs[].<chave>.
        runs.append({
            "id": directory.name,
            "stage": match["stage"],
            # Data de início vem do nome da pasta (AAAAMMDDTHHMMSSZ, UTC) e vira ISO 8601.
            "started_at": datetime.strptime(match["stamp"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat(),
            "state": state,
            # Só os parâmetros de SHOWN_PARAMETERS presentes, na ordem da tupla.
            "parameters": {key: parameters[key] for key in SHOWN_PARAMETERS if key in parameters},
            # Caminho gravado em outra máquina (Windows): só o nome do arquivo interessa.
            "checkpoint": str((execution.get("runtime") or {}).get("checkpoint") or "").replace("\\", "/").rsplit("/", 1)[-1] or None,
            "device": (execution.get("runtime") or {}).get("device_requested"),
            # Identidade do dataset: piloto (pilot_sha256) ou produção (dataset.yaml); o gráfico
            # Evolução dos modelos marca a troca porque validações diferentes não se comparam.
            "dataset_sha256": provenance.get("pilot_sha256") or provenance.get("dataset_yaml_sha256") or provenance.get("provenance_sha256"),
            "buildings": provenance.get("buildings") or [],
            "independent_evaluation": provenance.get("independent_evaluation"),
            "class_counts": execution.get("class_counts") or {},
            "epochs_planned": epochs_planned,
            "epochs_done": curves[-1]["epoch"] if curves else 0,
            "seconds_per_epoch": seconds_per_epoch,
            "best_epoch": best,
            "last": curves[-1] if curves else None,
            "curves": curves,
            "gpu_eval": read_gpu_curve(directory),
            # "fotos" (imagem inteira) ou "janelas" (treino em recortes/tiles).
            "label_unit": execution.get("label_unit") or "fotos",
            "weights": weights,
            "plots": plots,
            "notes": (summary or {}).get("reasons") or [],
            # Última escrita do results.csv (UTC), ou None se ainda não há épocas.
            "updated_at": datetime.fromtimestamp(results_csv.stat().st_mtime, timezone.utc).isoformat() if results_csv.is_file() else None,
        })
    # 9. Mais recente primeiro: split("_")[-2] é o carimbo de data AAAAMMDDTHHMMSSZ do id
    # (ordenar esse texto equivale a ordenar pela data).
    runs.sort(key=lambda run: run["id"].split("_")[-2], reverse=True)
    # 10. Nomes legíveis; runs/nomes.json (opcional) traz os nomes escolhidos pela equipe.
    _custom = _read_json(runs_dir / "nomes.json")
    _name_runs(runs, _custom if isinstance(_custom, dict) else {})
    return runs


def _percent(value) -> str:
    """Fração em porcentagem no formato brasileiro, com uma casa (0.4312 → "43,1%").

    Parâmetros:
        value: número entre 0 e 1 (float ou texto numérico).

    Retorna: texto com vírgula decimal e "%".
    """
    return f"{float(value) * 100:.1f}%".replace(".", ",")


def _name_runs(runs: list[dict], custom: dict) -> None:
    """Nome legível de cada execução (``display_name``) e um resumo (``display_detail``).

    O nome automático é o tipo da execução + o código curto do id
    ("Treino Piloto F5E5BA8B", ver ``naming.run_name``): o mesmo código aparece
    em comparações, pesos e relatórios, então a execução é reconhecível em toda
    a plataforma. ``runs/nomes.json`` troca o nome pelo escolhido pela equipe,
    com a chave sendo o id completo ou só o sufixo de 8 caracteres
    (ex.: ``{"56038eb9": "v7.4 referência"}``). O resumo junta o que a pessoa
    procura ao comparar: modelo base, épocas feitas, melhor mAP50 e se treinou
    em janelas.

    Parâmetros:
        runs: lista de ``collect_runs``; cada dict é MODIFICADO no lugar.
        custom: conteúdo de ``runs/nomes.json`` (id ou sufixo → nome), ou {}.

    Retorna: None (grava ``display_name`` e ``display_detail`` em cada execução).
    Exemplo de resumo: "yolo26l · 40/60 épocas · em janelas · mAP50 val 43,1% (época 37)".
    """
    for run in runs:
        # Nome: personalizado pelo id completo, senão pelo sufixo de 8 hex, senão automático.
        suffix = run["id"].rsplit("_", 1)[-1]
        run["display_name"] = run_name(run["id"], custom.get(run["id"]) or custom.get(suffix))
        # Resumo montado em partes, unidas por " · ".
        # EXTENSAO: informação nova no resumo da execução = um parts.append(...) aqui.
        parts = []
        # Modelo base sem a extensão (ex.: "yolo26l.pt" → "yolo26l").
        if run.get("checkpoint"):
            parts.append(str(run["checkpoint"]).rsplit(".", 1)[0])
        done, planned = run.get("epochs_done") or 0, run.get("epochs_planned")
        parts.append(f"{done}/{planned} épocas" if planned else f"{done} épocas")
        if run.get("label_unit") == "janelas":
            parts.append("em janelas")
        best = run.get("best_epoch")
        parts.append(f"mAP50 val {_percent(best['map50'])} (época {best['epoch']})" if best and best.get("map50") is not None
                     else "sem validação")
        run["display_detail"] = " · ".join(parts)


# ---------------------------------------------------------------------------
# Comparações de modelos, datasets piloto e arquivos gerados por revisão
# ---------------------------------------------------------------------------

def collect_comparisons(root: Path) -> list[dict]:
    """Comparações salvas por ``scripts/compare_pilot_models.py`` (``runs/compare_*.json``).

    Parâmetros:
        root: pasta ``sistema_ia``.

    Retorna:
        Lista (ordem alfabética do nome do arquivo) de dicts ``{"file", "dataset", "conf",
        "iou_match", "models": [{"weights", "run_id", "file", "weights_sha256", "imgsz",
        "splits"}]}``. Arquivos inválidos ou sem lista ``models`` são ignorados.

    Usada na página na seção Modelos (``OPS.comparisons``).
    """
    comparisons = []
    for path in sorted((root / "runs").glob("compare_*.json")):
        data = _read_json(path)
        # Ignora JSON corrompido ou de outro formato.
        if not data or not isinstance(data.get("models"), list):
            continue
        models = []
        for model in data["models"]:
            # Caminho dos pesos com "/" (pode ter sido gravado no Windows).
            weights = str(model.get("weights") or "").replace("\\", "/")
            # Liga o modelo à execução de treino que o gerou (None para models/*.pt).
            run_id = run_id_from_path(weights)
            # EXTENSAO: campo novo por modelo comparado = chave nova neste dict.
            models.append({"weights": weights, "run_id": run_id, "file": weights.rsplit("/", 1)[-1],
                           "weights_sha256": model.get("weights_sha256"), "imgsz": model.get("imgsz"),
                           "splits": model.get("splits") or {}})
        comparisons.append({"file": path.name, "dataset": data.get("dataset"), "conf": data.get("conf"),
                            "iou_match": data.get("iou_match"), "models": models})
    return comparisons


def collect_datasets(root: Path) -> list[dict]:
    """Datasets piloto disponíveis para um novo treino (``data/pilot/*/dataset.yaml``).

    Parâmetros:
        root: pasta ``sistema_ia``.

    Retorna:
        Lista (ordem alfabética da pasta) de dicts ``{"path" (relativo a root, ex.:
        "data/pilot/<nome>/dataset.yaml"), "name", "created_at", "sha256" (de pilot.json),
        "buildings", "splits" ({"train": n, "val": n, ...}), "boxes_for_training",
        "independent_evaluation"}``.

    Usada na página na seção Treinamento (seletor de dataset, ``OPS.datasets``). O ``path``
    é o valor que ``platform_server._job_arguments`` aceita em ``--data``.
    """
    datasets = []
    for yaml_path in sorted((root / "data" / "pilot").glob("*/dataset.yaml")):
        pilot_json = yaml_path.parent / "pilot.json"
        pilot = _read_json(pilot_json) or {}
        # Fotos por divisão a partir dos caminhos images/<split>/<hash>.jpg do pilot.json.
        # parts[1] é a divisão ("train", "val", "test") em "images/<split>/<arquivo>".
        splits: dict[str, int] = {}
        for entry in pilot.get("images") or []:
            parts = str(entry.get("image", "")).split("/")
            if len(parts) >= 3:
                splits[parts[1]] = splits.get(parts[1], 0) + 1
        # EXTENSAO: informação nova do dataset no seletor da seção Treinamento = chave nova aqui.
        datasets.append({"path": _relative(yaml_path, root), "name": yaml_path.parent.name,
                         "created_at": pilot.get("created_at"),
                         "sha256": file_hash(pilot_json) if pilot_json.is_file() else None,
                         "buildings": pilot.get("buildings") or [], "splits": splits,
                         "boxes_for_training": (pilot.get("duplicate_policy") or {}).get("boxes_for_training"),
                         "independent_evaluation": pilot.get("independent_evaluation")})
    return datasets


# ENTRADA: GET /api/reviews/<pasta>/files (platform_server.do_GET) → seção Laudos e downloads.
def collect_review_files(directory: Path | None) -> list[dict]:
    """Laudos (``laudos/<pasta>/``) e pacotes de evidências (``exportacoes/``) já gerados para uma revisão.

    Caminhos relativos à pasta da revisão: servem de link tanto na página aberta do
    disco quanto na servida em ``/r/<pasta>/``. Mais recentes primeiro.

    Parâmetros:
        directory: pasta da revisão (``data/reviews/<inspeção>``) ou None.

    Retorna:
        Lista de dicts ``{"kind": "pdf"|"html"|"json"|"zip", "label" (pasta do laudo ou None),
        "name", "href" (relativo à revisão), "bytes", "modified" (ISO 8601 UTC)}``,
        ordenada do mais recente para o mais antigo. Vazia se a pasta não existe.

    ATENCAO: o servidor só entrega estes arquivos se a pasta estiver em
    ``platform_server.DOWNLOAD_FOLDERS`` e a extensão em ``DOWNLOAD_SUFFIXES``.
    """
    if not directory or not Path(directory).is_dir():
        return []
    directory = Path(directory)
    files = []
    # 1. Cada laudo fica numa subpasta de laudos/ com até três formatos.
    # EXTENSAO: formato novo de laudo (ex.: "laudo.docx") = par novo nesta tupla
    # (e a extensão em platform_server.DOWNLOAD_SUFFIXES).
    for report in sorted(directory.glob("laudos/*/")):
        for name, kind in (("laudo.pdf", "pdf"), ("laudo.html", "html"), ("laudo.json", "json")):
            path = report / name
            if path.is_file():
                files.append({"kind": kind, "label": report.name, "name": f"laudos/{report.name}/{name}", "href": f"laudos/{report.name}/{name}",
                              "bytes": path.stat().st_size, "modified": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")})
    # 2. Pacotes ZIP de evidências.
    for path in sorted(directory.glob("exportacoes/*.zip")):
        files.append({"kind": "zip", "label": None, "name": f"exportacoes/{path.name}", "href": f"exportacoes/{path.name}",
                      "bytes": path.stat().st_size, "modified": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")})
    # 3. Mais recente primeiro (o texto ISO 8601 ordena igual à data).
    files.sort(key=lambda item: item["modified"], reverse=True)
    return files


# ---------------------------------------------------------------------------
# Instantâneo completo (OPS na página)
# ---------------------------------------------------------------------------

# ENTRADA: GET /api/operations (a cada 5 s pela página) e o instantâneo embutido no index.html.
def collect_operations(config: ProjectConfig, page_directory: Path | None = None) -> dict:
    """Instantâneo completo para a página (sem treinar, aprovar ou copiar nada).

    Parâmetros:
        config: configuração do projeto (``root``, ``training``, ``pilot``, ``device``, ``model``).
        page_directory: pasta da revisão aberta (onde fica o ``index.html``) ou None
            (espaço de trabalho sem inspeção).

    Retorna:
        dict com ``review_files``, ``generated_at``, ``review_directory``, ``device``,
        ``base_model``, ``pilot_defaults`` (epochs/imgsz/batch/patience), ``suggestion_conf``,
        ``runs``, ``comparisons``, ``datasets`` e ``weights`` (lista de .pt relativos a
        ``sistema_ia``: ``models/*.pt`` e depois os pesos de cada execução).

    Chamada por: ``review_render.build_page``/``render-review`` e ``platform_server``
    (``GET /api/operations`` e ``Handler._page``).

    EXTENSAO: dado novo para a página = chave nova no dict devolvido; a página lê em ``OPS.<chave>``.
    """
    root = config.root
    # Padrões do treino piloto: training geral do project.yaml sobrescrito por pilot.training.
    pilot_training = {**config.training, **(config.pilot.get("training") or {})}
    # Pesos disponíveis: primeiro os modelos base em models/, depois best/last de cada execução.
    weights = [_relative(path, root) for path in sorted((root / "models").glob("*.pt"))]
    runs = collect_runs(root, page_directory)
    for run in runs:
        weights.extend(run["weights"].values())
    return {
        # Laudos e pacotes já gerados nesta revisão (links relativos; o servidor lista ao vivo em /api/reviews/<pasta>/files).
        "review_files": collect_review_files(page_directory),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Pasta da revisão relativa a sistema_ia: a página monta os comandos do ciclo com ela.
        "review_directory": _relative(page_directory, root) if page_directory else None,
        "device": config.device,
        "base_model": config.pilot.get("model") or config.model,
        # EXTENSAO: padrão novo do formulário de treino = nome nesta tupla (vindo de project.yaml).
        "pilot_defaults": {key: pilot_training.get(key) for key in ("epochs", "imgsz", "batch", "patience")},
        "suggestion_conf": config.pilot.get("suggestion_conf"),
        "runs": runs,
        "comparisons": collect_comparisons(root),
        "datasets": collect_datasets(root),
        "weights": weights,
    }
