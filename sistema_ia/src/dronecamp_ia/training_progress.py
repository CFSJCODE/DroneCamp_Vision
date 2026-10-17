"""Evolução de um treino piloto: melhor época, regressão, tempo acumulado e identidade do checkpoint.

Função no projeto: responder "ainda há progresso?", "houve regressão?" e "por que
a época ficou mais lenta?" a partir do que o treino já gravou, sem tocar na pasta
do treino (que pode estar rodando). Funciona em dois níveis:

1. Só com ``fit/results.csv`` (todo treino, inclusive o 614d070a em andamento):
   melhor época pelo critério da Ultralytics, diferença da última época para a
   melhor, épocas sem melhora, taxa de aprendizagem, valores não finitos, duração
   por época e tempo acumulado CORRIGIDO após retomadas.
2. Com a instrumentação nova (``training_monitor.py``, treinos a partir deste
   código): tempos por etapa (mediana e p95 por lote), passos efetivos do
   otimizador, gradientes/perdas não finitos, recursos da máquina e a ficha de
   cada tentativa (``run_record.py``).

Quem chama: comando ``training-progress`` (``cli.py``) e ``quality_report.py``
(``checkpoint_identity``, para dizer de que época é cada peso avaliado).

O que lê e grava no disco:
- Lê: ``<run>/fit/results.csv``, ``<run>/fit/args.yaml``, ``<run>/execution.json``,
  ``<run>/instrumentation/*.jsonl`` e ``<run>/ficha/`` quando existem.
- Grava: ``runs/progress_<data>_<id>/progress.json`` e ``progress.md`` (pasta NOVA;
  a pasta do treino nunca é alterada).

Fórmulas e critérios:
- Melhor época = maior ``fitness``. Na Ultralytics 8.4.172 (versão fixada no
  pyproject) ``fitness = mAP50-95`` de validação (pesos [0, 0, 0, 1] sobre P, R,
  mAP50, mAP50-95); é a época que vira ``best.pt``.
- Tempo acumulado: a coluna ``time`` do results.csv conta segundos desde o início
  da TENTATIVA atual e volta a zero numa retomada. Aqui cada queda de ``time``
  marca uma nova tentativa; o tempo efetivo é a soma das durações das épocas que
  ficaram no CSV (a época refeita depois de uma queda conta uma vez, a última).

ATENCAO: gradientes, passos do otimizador e tempos por etapa que não foram
registrados durante o treino não podem ser reconstruídos a partir do results.csv;
essas seções ficam como "nao_registrado" nos treinos antigos.

Onde mexer:
- Outro critério de melhor época: ``FITNESS_WEIGHTS``.
- Nova coluna resumida: ``progress_report`` (EXTENSAO lá).
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .io import file_hash, make_run_directory, write_json

# Colunas de métricas de validação que a Ultralytics grava ("(B)" = caixas).
METRIC_COLUMNS = {"precision": "metrics/precision(B)", "recall": "metrics/recall(B)",
                  "map50": "metrics/mAP50(B)", "map50_95": "metrics/mAP50-95(B)"}
# Pesos do fitness da Ultralytics 8.4.172 sobre (P, R, mAP50, mAP50-95).
# ATENCAO: se a versão fixada da Ultralytics mudar, confira ``DetMetrics.fitness``.
FITNESS_WEIGHTS = (0.0, 0.0, 0.0, 1.0)


def _number(value) -> float | None:
    """Converte uma célula do CSV em float; vazio ou texto vira ``None`` (NaN/Inf ficam como float)."""
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def read_results(path: Path) -> list[dict]:
    """Lê o ``results.csv`` da Ultralytics em uma lista de dicts com nomes de coluna sem espaços.

    Erros: OSError se o arquivo não existir.
    """
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [{key.strip(): _number(value) for key, value in row.items() if key} for row in rows]


def fitness(row: dict) -> float | None:
    """Fitness da Ultralytics (``FITNESS_WEIGHTS``) de uma linha; ``None`` se faltar métrica."""
    values = [row.get(METRIC_COLUMNS[key]) for key in ("precision", "recall", "map50", "map50_95")]
    if any(value is None for value in values):
        return None
    # nan_to_num, como a Ultralytics: métrica não finita conta 0.
    return sum(weight * (value if math.isfinite(value) else 0.0) for weight, value in zip(FITNESS_WEIGHTS, values))


def _percentile(values: list[float], fraction: float) -> float | None:
    """Percentil por interpolação linear; ``None`` para lista vazia."""
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def epoch_timeline(rows: list[dict]) -> dict:
    """Durações por época e tempo acumulado corrigido para retomadas.

    Retorna:
        ``{"epochs": [{"epoch", "attempt", "seconds"}], "attempts": n, "redone_epochs": [...],
        "effective_seconds", "median_epoch_seconds", "p95_epoch_seconds"}``. A primeira
        época de cada tentativa inclui a preparação (carregar dados e modelo).
    """
    attempt, previous_time, previous_epoch = 1, None, None
    by_epoch: dict[int, dict] = {}
    redone = []
    for row in rows:
        epoch, elapsed = row.get("epoch"), row.get("time")
        if epoch is None or elapsed is None:
            continue
        # Queda do relógio ou época que não avança = a Ultralytics recomeçou: nova tentativa (retomada).
        if previous_time is not None and (elapsed < previous_time or
                                          (previous_epoch is not None and epoch <= previous_epoch)):
            attempt += 1
            previous_time = None
        previous_epoch = epoch
        seconds = elapsed - (previous_time or 0.0)
        previous_time = elapsed
        if int(epoch) in by_epoch:
            redone.append(int(epoch))
        by_epoch[int(epoch)] = {"epoch": int(epoch), "attempt": attempt, "seconds": round(seconds, 1)}
    epochs = [by_epoch[key] for key in sorted(by_epoch)]
    durations = [item["seconds"] for item in epochs]
    return {"epochs": epochs, "attempts": attempt if epochs else 0, "redone_epochs": redone,
            "effective_seconds": round(sum(durations), 1),
            "median_epoch_seconds": round(_percentile(durations, 0.5), 1) if durations else None,
            "p95_epoch_seconds": round(_percentile(durations, 0.95), 1) if durations else None}


def evolution(rows: list[dict], patience: int | None = None, planned_epochs: int | None = None) -> dict:
    """Melhor época, diferença para ela, épocas sem melhora e saúde numérica do results.csv."""
    scored = [(row, fitness(row)) for row in rows if row.get("epoch") is not None]
    valid = [(row, value) for row, value in scored if value is not None]
    if not valid:
        return {"state": "sem_validacao", "epochs_recorded": len(scored)}
    best_row, best_value = max(valid, key=lambda item: item[1])
    last_row = valid[-1][0]
    best_map50_row = max(valid, key=lambda item: item[0].get(METRIC_COLUMNS["map50"]) or 0.0)[0]
    since = int(last_row["epoch"] - best_row["epoch"])
    # Contagem de valores não finitos por coluna (perdas, métricas, lr).
    nonfinite = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                nonfinite[key] = nonfinite.get(key, 0) + 1
    lr_columns = sorted(key for key in rows[-1] if key.startswith("lr/"))
    deltas = {name: (round(last_row[column] - best_row[column], 4)
                     if last_row.get(column) is not None and best_row.get(column) is not None else None)
              for name, column in METRIC_COLUMNS.items()}
    val_losses = [key for key in rows[-1] if key.startswith("val/") and key.endswith("loss")]
    val_total = [(row["epoch"], sum(row[key] for key in val_losses if row.get(key) is not None)) for row in rows
                 if row.get("epoch") is not None and val_losses]
    min_val = min(val_total, key=lambda item: item[1]) if val_total else None
    # EXTENSAO: indicador novo de progresso = chave nova neste dict (e linha em _markdown).
    return {
        "state": "ok", "criterion": "maior fitness = mAP50-95 de validação (Ultralytics 8.4; vira best.pt)",
        "epochs_recorded": len(scored), "last_epoch": int(last_row["epoch"]), "planned_epochs": planned_epochs,
        "best_epoch": int(best_row["epoch"]), "best_fitness": round(best_value, 5),
        "best": {name: best_row.get(column) for name, column in METRIC_COLUMNS.items()},
        "last": {name: last_row.get(column) for name, column in METRIC_COLUMNS.items()},
        "last_minus_best": deltas, "epochs_since_best": since,
        "best_map50_epoch": int(best_map50_row["epoch"]),
        "patience": patience, "patience_left": (patience - since) if patience else None,
        "regression": since > 0 and (deltas["map50_95"] or 0) < 0,
        "learning_rate_last": {key: rows[-1].get(key) for key in lr_columns},
        "val_loss_min_epoch": int(min_val[0]) if min_val else None,
        "val_loss_rising_since_min": bool(min_val and val_total[-1][1] > min_val[1] and val_total[-1][0] > min_val[0]),
        "nonfinite_values": nonfinite,
    }


def _read_jsonl(path: Path) -> list[dict]:
    """Linhas JSON válidas de um .jsonl (linha corrompida por queda de energia é pulada)."""
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def instrumentation_summary(run: Path) -> dict:
    """Resumo da instrumentação nova (``training_monitor.py``); ``nao_registrado`` nos treinos antigos."""
    epochs = _read_jsonl(Path(run) / "instrumentation" / "epochs.jsonl")
    resources = _read_jsonl(Path(run) / "instrumentation" / "resources.jsonl")
    if not epochs and not resources:
        return {"state": "nao_registrado",
                "note": "Treino sem training_monitor: tempos por etapa, passos do otimizador e gradientes "
                        "não podem ser reconstruídos do results.csv."}
    # Última linha por época global (uma época refeita depois de uma queda substitui a anterior).
    latest = {}
    for row in epochs:
        latest[row.get("epoch")] = row
    ordered = [latest[key] for key in sorted(key for key in latest if key is not None)]
    totals = {"optimizer_steps": sum(row.get("optimizer_steps", 0) for row in ordered),
              "images": sum(row.get("images", 0) for row in ordered),
              "batches": sum(row.get("batches", 0) for row in ordered),
              "nonfinite_loss": sum(row.get("nonfinite_loss", 0) for row in ordered),
              "nonfinite_grad": sum(row.get("nonfinite_grad", 0) for row in ordered),
              "skipped_steps": sum(row.get("skipped_steps", 0) for row in ordered),
              "effective_seconds": round(sum(row.get("epoch_seconds", 0.0) for row in ordered), 1),
              "seconds_all_attempts": round(sum(row.get("epoch_seconds", 0.0) for row in epochs), 1)}
    peaks = {}
    for row in resources:
        for key, value in row.items():
            if isinstance(value, (int, float)) and key not in ("t", "pid"):
                peaks[key] = max(peaks.get(key, value), value)
    return {"state": "registrado", "epochs": ordered[-5:], "totals": totals, "resource_peaks": peaks,
            "resource_samples": len(resources), "last_resource_sample": resources[-1] if resources else None}


def _patience_and_epochs(run: Path) -> tuple[int | None, int | None]:
    """Paciência e épocas planejadas: ``fit/args.yaml`` da Ultralytics, senão ``execution.json``."""
    import yaml

    args_path = Path(run) / "fit" / "args.yaml"
    if args_path.is_file():
        args = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
        return args.get("patience"), args.get("epochs")
    execution = Path(run) / "execution.json"
    if execution.is_file():
        parameters = json.loads(execution.read_text(encoding="utf-8")).get("parameters", {})
        return parameters.get("patience"), parameters.get("epochs")
    return None, None


def checkpoint_identity(weights: Path, model=None) -> dict:
    """Identidade de um peso avaliado: hash, época e por que ele é o "melhor".

    Parâmetros:
        weights: caminho do ``.pt`` (ou ``.onnx`` exportado; o ``.pt`` de origem é procurado
            pelo chamador).
        model: modelo já carregado (opcional); ``model.ckpt["epoch"]`` é usado se ≥ 0.

    Retorna:
        ``{"path", "sha256", "bytes", "role", "epoch", "reason", "run"}``. ``role`` é
        ``best``/``last``/``outro`` pelo nome; ``epoch`` vem do results.csv da execução
        (o ``best.pt`` final da Ultralytics grava ``epoch = -1``).
    """
    weights = Path(weights)
    info = {"path": str(weights), "sha256": file_hash(weights) if weights.is_file() else None,
            "bytes": weights.stat().st_size if weights.is_file() else None,
            "role": "best" if weights.stem == "best" else "last" if weights.stem == "last" else "outro",
            "epoch": None, "reason": None, "run": None}
    ckpt_epoch = (getattr(model, "ckpt", None) or {}).get("epoch") if model is not None else None
    run = weights.parent.parent.parent if weights.parent.name == "weights" else None
    results = run / "fit" / "results.csv" if run else None
    if results and results.is_file():
        info["run"] = str(run)
        rows = read_results(results)
        report = evolution(rows)
        if info["role"] == "best" and report.get("state") == "ok":
            info.update({"epoch": report["best_epoch"],
                         "reason": f"best.pt: época {report['best_epoch']} de maior mAP50-95 de validação "
                                   f"({report['best_fitness']:.4f}) entre {report['epochs_recorded']} registradas"})
        elif info["role"] == "last" and rows:
            info.update({"epoch": int(rows[-1]["epoch"]), "reason": "last.pt: última época salva"})
    if info["epoch"] is None and isinstance(ckpt_epoch, int) and ckpt_epoch >= 0:
        info.update({"epoch": ckpt_epoch + 1, "reason": "época gravada no checkpoint"})
    if info["reason"] is None:
        info["reason"] = "peso informado à mão; sem results.csv ao lado para saber a época"
    return info


def progress_report(run: Path) -> dict:
    """Junta evolução, linha do tempo, instrumentação e ficha de um treino (sem gravar nada).

    Erros: ValueError se a pasta não tiver ``fit/results.csv``.
    """
    from .run_record import run_card_summary

    run = Path(run).resolve()
    results = run / "fit" / "results.csv"
    if not results.is_file():
        raise ValueError(f"{results} não existe: o treino ainda não terminou a primeira época ou a pasta está errada.")
    rows = read_results(results)
    patience, planned = _patience_and_epochs(run)
    timeline = epoch_timeline(rows)
    durations = timeline["epochs"]
    # Tendência de duração: média das 5 primeiras contra as 5 últimas épocas.
    first = [item["seconds"] for item in durations[:5]]
    last = [item["seconds"] for item in durations[-5:]]
    return {
        "schema_version": 1, "run": str(run), "results_csv_sha256": file_hash(results),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evolution": evolution(rows, patience, planned), "timeline": {
            **{key: value for key, value in timeline.items() if key != "epochs"},
            "first5_mean_seconds": round(sum(first) / len(first), 1) if first else None,
            "last5_mean_seconds": round(sum(last) / len(last), 1) if last else None,
            "last_epochs": durations[-10:]},
        "instrumentation": instrumentation_summary(run), "run_card": run_card_summary(run),
    }


def _fmt(value, digits: int = 3) -> str:
    """Número formatado para o Markdown ("—" para ausente)."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(report: dict) -> str:
    """Relatório curto em Markdown (abre no VS Code, no GitHub e no visualizador da plataforma)."""
    evo, timeline, inst = report["evolution"], report["timeline"], report["instrumentation"]
    lines = [f"# Evolução do treino `{Path(report['run']).name}`", ""]
    if evo.get("state") != "ok":
        lines += ["Sem época validada no results.csv ainda.", ""]
    else:
        lines += [
            f"- Critério da melhor época: {evo['criterion']}.",
            f"- Última época registrada: {evo['last_epoch']} de {_fmt(evo['planned_epochs'])}; "
            f"melhor: {evo['best_epoch']} (fitness {_fmt(evo['best_fitness'], 4)}); "
            f"épocas sem melhora: {evo['epochs_since_best']}"
            + (f"; paciência restante: {evo['patience_left']}" if evo.get("patience") else " (sem parada antecipada)") + ".",
            "", "| Métrica | Melhor época | Última época | Última − melhor |", "|---|---|---|---|"]
        for name in METRIC_COLUMNS:
            lines.append(f"| {name} | {_fmt(evo['best'][name])} | {_fmt(evo['last'][name])} | "
                         f"{_fmt(evo['last_minus_best'][name])} |")
        lines += ["", f"- Regressão em relação à melhor época: {'sim' if evo['regression'] else 'não'}.",
                  f"- Perda de validação mínima na época {_fmt(evo['val_loss_min_epoch'])}"
                  + ("; subiu desde então." if evo["val_loss_rising_since_min"] else "."),
                  f"- Valores não finitos no CSV: {evo['nonfinite_values'] or 'nenhum'}.",
                  f"- Taxa de aprendizagem na última época: {evo['learning_rate_last']}.", ""]
    lines += ["## Tempo", "",
              f"- Tentativas detectadas pelo relógio do results.csv: {timeline['attempts']}"
              f"; épocas refeitas: {timeline['redone_epochs'] or 'nenhuma'}.",
              f"- Tempo efetivo acumulado: {_fmt(timeline['effective_seconds'] / 3600 if timeline['effective_seconds'] else None, 2)} h"
              f" (mediana {_fmt(timeline['median_epoch_seconds'], 0)} s/época, p95 {_fmt(timeline['p95_epoch_seconds'], 0)} s).",
              f"- Média das 5 primeiras épocas: {_fmt(timeline['first5_mean_seconds'], 0)} s; "
              f"das 5 últimas: {_fmt(timeline['last5_mean_seconds'], 0)} s.", ""]
    lines += ["## Instrumentação", ""]
    if inst["state"] != "registrado":
        lines += [inst["note"], ""]
    else:
        totals = inst["totals"]
        lines += [f"- Passos do otimizador: {totals['optimizer_steps']}; lotes: {totals['batches']}; "
                  f"imagens: {totals['images']}; perdas não finitas: {totals['nonfinite_loss']}; "
                  f"gradientes não finitos: {totals['nonfinite_grad']}.",
                  f"- Tempo efetivo (instrumentação): {_fmt(totals['effective_seconds'] / 3600, 2)} h; "
                  f"incluindo trabalho perdido em quedas: {_fmt(totals['seconds_all_attempts'] / 3600, 2)} h.", "",
                  "| Época | Leitura (med/p95 s) | Preparo | Frente+perda | Trás | Otimizador | Validação | Checkpoint |",
                  "|---|---|---|---|---|---|---|---|"]
        for row in inst["epochs"]:
            stage = row.get("stages", {})

            def cell(name: str) -> str:
                item = stage.get(name) or {}
                return f"{_fmt(item.get('median'), 3)}/{_fmt(item.get('p95'), 3)}"

            lines.append(f"| {row.get('epoch')} | {cell('data_wait')} | {cell('preprocess')} | {cell('forward_loss')} | "
                         f"{cell('backward')} | {cell('optimizer_step')} | {_fmt(row.get('validation_seconds'), 1)} | "
                         f"{_fmt(row.get('checkpoint_seconds'), 1)} |")
        lines += ["", f"- Picos de recursos: {inst['resource_peaks']}.", ""]
    card = report["run_card"]
    lines += ["## Ficha", "", f"- Tentativas registradas: {card.get('attempts', 0)}; "
              f"integridade: {card.get('integrity', 'sem ficha')}.", ""]
    return "\n".join(lines)


def write_progress(config_root: Path, run: Path, output: Path | None = None) -> Path:
    """ENTRADA de ``training-progress``: grava ``progress.json`` e ``progress.md`` numa pasta NOVA.

    Parâmetros:
        config_root: pasta ``sistema_ia`` (onde fica ``runs/``).
        run: pasta do treino (``runs/pilot_train_*``); só é lida.
        output: pasta de saída (padrão ``runs/progress_<data>_<id>``).
    """
    report = progress_report(run)
    folder = Path(output) if output else make_run_directory(Path(config_root), "progress")
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "progress.json", report)
    (folder / "progress.md").write_text(_markdown(report), encoding="utf-8")
    return folder
