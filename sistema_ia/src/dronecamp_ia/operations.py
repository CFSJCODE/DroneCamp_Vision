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

# Colunas do results.csv da Ultralytics enviadas à página (as demais ficam no disco).
CURVE_COLUMNS = {
    "epoch": "epoch", "time": "time",
    "train/box_loss": "train_box", "train/cls_loss": "train_cls",
    "val/box_loss": "val_box", "val/cls_loss": "val_cls",
    "metrics/precision(B)": "precision", "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "map50", "metrics/mAP50-95(B)": "map50_95",
}
# Parâmetros de treino mostrados na tabela de execuções.
SHOWN_PARAMETERS = ("epochs", "imgsz", "batch", "optimizer", "lr0", "patience", "mosaic", "close_mosaic", "seed")
# Sem nova época neste intervalo, um treino sem summary.json é tratado como interrompido.
RUNNING_WINDOW_SECONDS = 15 * 60
RUN_NAME = re.compile(r"^(?P<stage>[a-z_]+?)_(?P<stamp>\d{8}T\d{6}Z)_(?P<id>[0-9a-f]{8})$")


def _read_json(path: Path) -> dict | None:
    """JSON opcional: arquivo ausente ou corrompido não derruba o painel."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _number(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_curves(results_csv: Path) -> list[dict]:
    """Uma linha por época com as colunas de ``CURVE_COLUMNS`` (nomes curtos)."""
    try:
        with results_csv.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError:
        return []
    curves = []
    for row in rows:
        # A Ultralytics alinha colunas com espaços em algumas versões.
        clean = {key.strip(): value for key, value in row.items() if key}
        point = {short: _number(clean.get(column)) for column, short in CURVE_COLUMNS.items()}
        if point["epoch"] is not None:
            point["epoch"] = int(point["epoch"])
            curves.append(point)
    return curves


def read_gpu_curve(run_directory: Path) -> list[dict]:
    """Pontos da avaliação paralela na GPU: ``gpu_eval/eval_curve.jsonl`` (gpu_eval.py)
    ou ``gpu_eval/curve.jsonl`` (scripts/gpu_eval_watcher.py)."""
    points = []
    for name in ("eval_curve.jsonl", "curve.jsonl"):
        path = run_directory / "gpu_eval" / name
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if not isinstance(entry, dict) or not isinstance(entry.get("epoch"), int):
                continue
            if entry.get("map50") is not None:
                points.append({"epoch": entry["epoch"], "map50": entry["map50"], "map50_95": entry.get("map50_95"),
                               "split": entry.get("split"), "provider": entry.get("provider_active") or entry.get("provider"),
                               "seconds": entry.get("seconds")})
            elif isinstance(entry.get("sets"), dict):  # formato do vigia: um bloco por conjunto
                for split, values in entry["sets"].items():
                    map50 = (values.get("totals") or {}).get("map50")
                    if map50 is not None:
                        points.append({"epoch": entry["epoch"], "map50": map50, "map50_95": None, "split": split,
                                       "provider": entry.get("provider"), "seconds": entry.get("seconds")})
    return sorted(points, key=lambda point: (point["epoch"], str(point["split"])))


def _relative(path: Path, start: Path) -> str:
    """Caminho relativo com "/" para links da página (funciona aberto do disco)."""
    return Path(os.path.relpath(path, start)).as_posix()


def _run_state(summary: dict | None, results_csv: Path, now: float) -> str:
    """completed / running / interrupted / failed a partir do que existe no disco."""
    if summary:
        if summary.get("state") == "completed" and summary.get("complete", True):
            return "completed"
        if summary.get("state") == "failed" or summary.get("complete") is False:
            return "failed"
    # fit/heartbeat é tocado a cada minuto pelo treino (training.py): uma época longa
    # deixa results.csv parado por mais que a janela sem que o treino tenha morrido.
    for marker in (results_csv, results_csv.parent / "heartbeat"):
        if marker.is_file() and now - marker.stat().st_mtime < RUNNING_WINDOW_SECONDS:
            return "running"
    return "interrupted" if results_csv.is_file() else "prepared"


def collect_runs(root: Path, page_directory: Path | None = None, now: float | None = None) -> list[dict]:
    """Execuções de treino em ``runs/`` (mais recente primeiro)."""
    now = time.time() if now is None else now
    runs_dir = root / "runs"
    if not runs_dir.is_dir():
        return []
    base = page_directory or root
    runs = []
    for directory in runs_dir.iterdir():
        match = RUN_NAME.match(directory.name)
        if not directory.is_dir() or not match or match["stage"] not in {"pilot_train", "train", "tune"}:
            continue
        fit = directory / "fit"
        results_csv = fit / "results.csv"
        execution = _read_json(directory / "execution.json") or {}
        summary = _read_json(directory / "summary.json")
        provenance = _read_json(directory / "pilot_provenance.json") or _read_json(directory / "training_provenance.json") or {}
        parameters = execution.get("parameters") or {}
        curves = read_curves(results_csv)
        state = _run_state(summary, results_csv, now)
        # Melhor época pelo mAP50 de validação (o mesmo critério do best.pt é mAP50-95 + mAP50).
        scored = [point for point in curves if point.get("map50") is not None]
        best = max(scored, key=lambda point: (point["map50"], point.get("map50_95") or 0), default=None)
        epochs_planned = parameters.get("epochs")
        seconds_per_epoch = None
        if curves and curves[-1].get("time"):
            seconds_per_epoch = curves[-1]["time"] / max(1, curves[-1]["epoch"])
        weights = {name: _relative(fit / "weights" / f"{name}.pt", root)
                   for name in ("best", "last") if (fit / "weights" / f"{name}.pt").is_file()}
        plots = {name: _relative(fit / file, base) for name, file in (
            ("results", "results.png"), ("confusion", "confusion_matrix_normalized.png"),
            ("pr_curve", "BoxPR_curve.png"), ("val_pred", "val_batch0_pred.jpg"), ("labels", "labels.jpg"))
            if (fit / file).is_file()}
        runs.append({
            "id": directory.name,
            "stage": match["stage"],
            "started_at": datetime.strptime(match["stamp"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat(),
            "state": state,
            "parameters": {key: parameters[key] for key in SHOWN_PARAMETERS if key in parameters},
            # Caminho gravado em outra máquina (Windows): só o nome do arquivo interessa.
            "checkpoint": str((execution.get("runtime") or {}).get("checkpoint") or "").replace("\\", "/").rsplit("/", 1)[-1] or None,
            "device": (execution.get("runtime") or {}).get("device_requested"),
            "dataset_sha256": provenance.get("pilot_sha256"),
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
            "label_unit": execution.get("label_unit") or "fotos",
            "weights": weights,
            "plots": plots,
            "notes": (summary or {}).get("reasons") or [],
            "updated_at": datetime.fromtimestamp(results_csv.stat().st_mtime, timezone.utc).isoformat() if results_csv.is_file() else None,
        })
    runs.sort(key=lambda run: run["id"].split("_")[-2], reverse=True)
    _name_runs(runs, _read_json(runs_dir / "nomes.json") or {})
    return runs


STAGE_LABELS = {"pilot_train": "Treino piloto", "train": "Treino de produção", "tune": "Ajuste de hiperparâmetros"}


def _percent(value) -> str:
    return f"{float(value) * 100:.1f}%".replace(".", ",")


def _name_runs(runs: list[dict], custom: dict) -> None:
    """Nome legível de cada execução (``display_name``) e um resumo (``display_detail``).

    O nome automático numera as execuções de cada etapa em ordem cronológica
    ("Treino piloto 7"); ``runs/nomes.json`` troca o nome pelo escolhido pela
    equipe, com a chave sendo o id completo ou só o sufixo de 8 caracteres
    (ex.: ``{"56038eb9": "v7.4 referência"}``). O resumo junta o que a pessoa
    procura ao comparar: modelo base, épocas feitas, melhor mAP50 e se treinou
    em janelas.
    """
    by_stage: dict[str, list[dict]] = {}
    for run in sorted(runs, key=lambda run: run["started_at"]):
        by_stage.setdefault(run["stage"], []).append(run)
    for stage, items in by_stage.items():
        for number, run in enumerate(items, start=1):
            suffix = run["id"].rsplit("_", 1)[-1]
            custom_name = custom.get(run["id"]) or custom.get(suffix)
            run["display_name"] = str(custom_name).strip() if custom_name else f"{STAGE_LABELS.get(stage, stage)} {number}"
            parts = []
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


def collect_comparisons(root: Path) -> list[dict]:
    """Comparações salvas por ``scripts/compare_pilot_models.py`` (``runs/compare_*.json``)."""
    comparisons = []
    for path in sorted((root / "runs").glob("compare_*.json")):
        data = _read_json(path)
        if not data or not isinstance(data.get("models"), list):
            continue
        models = []
        for model in data["models"]:
            weights = str(model.get("weights") or "").replace("\\", "/")
            run_id = next((part for part in weights.split("/") if RUN_NAME.match(part)), None)
            models.append({"weights": weights, "run_id": run_id, "file": weights.rsplit("/", 1)[-1],
                           "weights_sha256": model.get("weights_sha256"), "imgsz": model.get("imgsz"),
                           "splits": model.get("splits") or {}})
        comparisons.append({"file": path.name, "dataset": data.get("dataset"), "conf": data.get("conf"),
                            "iou_match": data.get("iou_match"), "models": models})
    return comparisons


def collect_datasets(root: Path) -> list[dict]:
    """Datasets piloto disponíveis para um novo treino (``data/pilot/*/dataset.yaml``)."""
    datasets = []
    for yaml_path in sorted((root / "data" / "pilot").glob("*/dataset.yaml")):
        pilot_json = yaml_path.parent / "pilot.json"
        pilot = _read_json(pilot_json) or {}
        # Fotos por divisão a partir dos caminhos images/<split>/<hash>.jpg do pilot.json.
        splits: dict[str, int] = {}
        for entry in pilot.get("images") or []:
            parts = str(entry.get("image", "")).split("/")
            if len(parts) >= 3:
                splits[parts[1]] = splits.get(parts[1], 0) + 1
        datasets.append({"path": _relative(yaml_path, root), "name": yaml_path.parent.name,
                         "created_at": pilot.get("created_at"),
                         "sha256": file_hash(pilot_json) if pilot_json.is_file() else None,
                         "buildings": pilot.get("buildings") or [], "splits": splits,
                         "boxes_for_training": (pilot.get("duplicate_policy") or {}).get("boxes_for_training"),
                         "independent_evaluation": pilot.get("independent_evaluation")})
    return datasets


def collect_operations(config: ProjectConfig, page_directory: Path | None = None) -> dict:
    """Instantâneo completo para a página (sem treinar, aprovar ou copiar nada)."""
    root = config.root
    pilot_training = {**config.training, **(config.pilot.get("training") or {})}
    weights = [_relative(path, root) for path in sorted((root / "models").glob("*.pt"))]
    runs = collect_runs(root, page_directory)
    for run in runs:
        weights.extend(run["weights"].values())
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Pasta da revisão relativa a sistema_ia: a página monta os comandos do ciclo com ela.
        "review_directory": _relative(page_directory, root) if page_directory else None,
        "device": config.device,
        "base_model": config.pilot.get("model") or config.model,
        "pilot_defaults": {key: pilot_training.get(key) for key in ("epochs", "imgsz", "batch", "patience")},
        "suggestion_conf": config.pilot.get("suggestion_conf"),
        "runs": runs,
        "comparisons": collect_comparisons(root),
        "datasets": collect_datasets(root),
        "weights": weights,
    }
