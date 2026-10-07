"""Relatório comparativo entre modelos: gráficos PNG, HTML autônomo e JSON normalizado.

Função no projeto: os números que dizem se um modelo piloto é melhor que outro
ficam espalhados em arquivos de formatos diferentes (``measurement.json`` do
``measure-models``, ``gate.json`` do ``compare-models``, ``runs/compare_*.json`` do
``scripts/compare_pilot_models.py``, ``runs/eval_por_classe_*.json`` e as curvas
``fit/results.csv`` de cada treino). Só uma medição (``measurement.json``) avalia
todos os modelos nas MESMAS fotos; por isso ela é a comparação principal e os
demais arquivos ficam numa seção "medições anteriores", não comparáveis entre si.
Este módulo junta tudo em uma pasta ``runs/report_*`` com gráficos comparativos
(um modelo = um rótulo e uma cor em todos eles), uma página HTML que abre do
disco sem internet e um ``models.json`` normalizado para quem quiser reprocessar.
Só lê arquivos: nada aqui treina, avalia ou aprova pesos.

O que faz:
- ``build_model_report``: monta a pasta do relatório a partir dos arquivos informados.
- ``load_measurement`` / ``load_gate`` / ``load_compare`` / ``load_eval`` / ``load_run``:
  leem cada formato e devolvem dicionários com as séries já identificadas no catálogo.
- ``ModelCatalog``: dá a cada série (SHA-256 dos pesos ou caminho, mais o regime de
  inferência quando há) um rótulo e uma cor fixos, na ordem em que aparece.
- ``wilson_interval``: intervalo de confiança de 95% do recall por classe.
- ``discover_runs``: acha em ``runs/`` os treinos piloto com ``fit/results.csv``.

Quem usa: o dono do projeto, para enxergar a diferença entre modelos piloto antes
de decidir o próximo treino; ``hparam_bandit.py`` pode gerar um ao fim da rodada.
Nunca importa Ultralytics/Torch: roda em qualquer máquina com matplotlib.

Quando mexer: para acrescentar um gráfico (escreva a função ``_chart_*`` e chame
em ``build_model_report``) ou para aceitar um formato novo de entrada.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path, PurePosixPath

from .config import ProjectConfig, detection_names, load_taxonomy
from .io import file_hash, make_run_directory, write_json
from .operations import RUN_NAME, read_curves

SCHEMA_VERSION = 1
# Uma cor por modelo, nesta ordem fixa; do nono modelo em diante tudo vira cinza "outros".
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
OTHERS_COLOR, OTHERS_LABEL = "#8a8a8a", "outros"
FN_COLOR = "#bdbdbd"
DPI = 130
GROUP_WIDTH = 0.8        # largura total de um grupo de barras, em unidades do eixo
BAR_GAP_PX = 2           # espaço entre barras vizinhas do mesmo grupo
LINE_PX = 2
MAX_MARKER_POINTS = 30   # acima disso os marcadores só borram a linha
SEEN_SUFFIX = " (visto no treino)"
SPLIT_ORDER = ("test", "val", "train")
FEW_BOXES = 3            # abaixo disso a classe é "pouco medida" (mesmo limite do gate)
WILSON_Z = 1.96          # intervalo de 95%
CLASS_KEYS = ("tp", "fp", "fn", "human_boxes", "precision", "recall", "ap50")
STRATA = {"fotos_fatiadas": "Fotos fatiadas (drone 20 MP)", "fotos_inteiras": "Fotos inteiras (laudo)"}
ARCHIVE_TITLE = "Medições anteriores (conjuntos de fotos diferentes; não comparáveis entre si)"
# Fundo claro, grade discreta e sem molduras: o leitor compara barras, não decoração.
STYLE = {
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.facecolor": "#fbfbfb",
    "axes.edgecolor": "#9a9a9a", "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "legend.fontsize": 8,
}


# ----------------------------------------------------------------------------
# Identidade dos modelos
# ----------------------------------------------------------------------------

def normalize_path(value: str | Path) -> str:
    """Caminhos gravados no Windows chegam com "\\"; o relatório compara com "/"."""
    return str(value).replace("\\", "/")


def run_id_of(path: str) -> str | None:
    """Id curto (8 hex) da pasta ``<etapa>_<data>_<id>`` presente no caminho, se houver."""
    for part in path.split("/"):
        match = RUN_NAME.match(part)
        if match:
            return match["id"]
    return None


class ModelCatalog:
    """Rótulo e cor únicos por série, na ordem de primeira aparição nas entradas.

    Uma série é um modelo (SHA-256 dos pesos quando o arquivo de entrada o traz;
    senão o caminho normalizado) mais, numa medição, o regime de inferência: o
    mesmo ``best.pt`` em "janelas" e "inteira" são duas séries, com cores
    vizinhas. Sem SHA, o mesmo ``best.pt`` gravado com caminho absoluto numa
    máquina e relativo noutra ainda é o mesmo modelo: casa por (id do treino,
    nome do arquivo).
    """

    def __init__(self) -> None:
        self.models: list[dict] = []
        self._by_id: dict[str, dict] = {}
        self._by_sha: dict[tuple, dict] = {}
        self._by_path: dict[tuple, dict] = {}
        self._by_run_file: dict[tuple, dict] = {}

    def register(self, weights: str | Path, sha256: str | None = None, regime: str | None = None) -> str:
        path = normalize_path(weights)
        run_id, file_name = run_id_of(path), PurePosixPath(path).name
        model = (self._by_sha.get((sha256, regime)) if sha256 else None) or self._by_path.get((path, regime))
        if model is None and run_id:
            model = self._by_run_file.get((run_id, file_name, regime))
        if model is None:
            index = len(self.models)
            base = sha256 or path
            model = {"id": f"{base}::{regime}" if regime else base,
                     "label": self._unique_label(run_id, file_name, sha256, regime), "weights": path,
                     "sha256": sha256, "run_id": run_id, "regime": regime,
                     "color": PALETTE[index] if index < len(PALETTE) else OTHERS_COLOR}
            self.models.append(model)
            self._by_id[model["id"]] = model
        if sha256 and not model["sha256"]:
            model["sha256"] = sha256
        if sha256:
            self._by_sha[(sha256, regime)] = model
        self._by_path[(path, regime)] = model
        if run_id:
            self._by_run_file.setdefault((run_id, file_name, regime), model)
        return model["id"]

    def _unique_label(self, run_id: str | None, file_name: str, sha256: str | None, regime: str | None) -> str:
        base = f"{run_id} · {file_name}" if run_id else file_name
        if regime:
            base = f"{base} · {regime}"
        used = {model["label"] for model in self.models}
        if base not in used:
            return base
        # Dois arquivos de mesmo nome sem treino distinguível: o SHA desempata.
        suffix = (sha256 or "")[:6] or str(len(self.models) + 1)
        return f"{base} ({suffix})"

    def by_id(self, model_id: str) -> dict:
        return self._by_id[model_id]

    def legend_label(self, model: dict) -> str:
        """Modelos além da paleta compartilham o cinza e a entrada "outros" na legenda."""
        return model["label"] if model["color"] != OTHERS_COLOR else OTHERS_LABEL

    def ordered(self, ids) -> list[dict]:
        """Modelos do catálogo presentes em ``ids``, na ordem de primeira aparição."""
        wanted = set(ids)
        return [model for model in self.models if model["id"] in wanted]


# ----------------------------------------------------------------------------
# Leitura das entradas
# ----------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _optional_json(path: Path) -> dict | None:
    """Arquivo opcional de um treino: ausente ou corrompido não derruba o relatório."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _ratio(numerator, denominator) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _finite(value):
    """NaN/infinito viram None: ``write_json`` recusa NaN e o HTML não deve mostrar "nan"."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return value if math.isfinite(value) else None


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return _finite(value)


def _metrics_block(metrics: dict) -> tuple[dict, dict]:
    """Dicionário por classe com ``_total`` (gate e medição): (totais, {classe: contagens})."""
    total = metrics.get("_total") or {}
    overall = {"map50": total.get("map50"), "precision": total.get("precision"), "recall": total.get("recall")}
    per_class = {name: {key: values.get(key) for key in CLASS_KEYS}
                 for name, values in metrics.items() if name != "_total" and isinstance(values, dict)}
    return overall, per_class


def load_measurement(path: Path, catalog: ModelCatalog) -> dict:
    """``measurement.json`` de ``model_gate.measure_models``: N séries nas mesmas fotos."""
    raw = _read_json(path)
    series, overall, per_class, strata = [], {}, {}, {}
    for entry in raw.get("models") or []:
        regime = entry.get("regime") or "inteira"
        series_id = catalog.register(entry.get("weights") or "modelo.pt", entry.get("sha256"), regime)
        series.append(series_id)
        overall[series_id], classes = _metrics_block(entry.get("metrics") or {})
        for name, values in classes.items():
            per_class.setdefault(name, {})[series_id] = values
        for stratum, metrics in (entry.get("strata") or {}).items():
            strata.setdefault(stratum, {})[series_id] = _metrics_block(metrics or {})[0]
    return {"file": str(path), "dataset": normalize_path(raw.get("dataset") or ""),
            "pilot_sha256": raw.get("pilot_sha256"), "splits": list(raw.get("splits") or []),
            "conf": raw.get("conf"), "ap_conf": raw.get("ap_conf"), "tiling": raw.get("tiling"),
            "images_evaluated": raw.get("images_evaluated"),
            "images_excluded_seen_in_training": raw.get("images_excluded_seen_in_training"),
            "strata_images": dict(raw.get("strata") or {}), "models": series, "overall": overall,
            "per_class": per_class, "strata": strata}


def load_gate(path: Path, catalog: ModelCatalog) -> dict:
    """``gate.json`` de ``model_gate.compare_models``; gates antigos podem não ter ``tiling``."""
    raw = _read_json(path)
    roles, overall, per_class = {}, {}, {}
    for role in ("baseline", "candidate"):
        entry = raw.get(role) or {}
        model_id = catalog.register(entry.get("weights") or f"{role}.pt", entry.get("sha256"))
        roles[role] = model_id
        overall[model_id], classes = _metrics_block(entry.get("metrics") or {})
        for name, values in classes.items():
            per_class.setdefault(name, {})[model_id] = values
    return {"file": str(path), "conf": raw.get("conf"), "splits": list(raw.get("splits") or []),
            "images_evaluated": raw.get("images_evaluated"),
            "images_excluded_seen_in_training": raw.get("images_excluded_seen_in_training"),
            "tiling": raw.get("tiling"), "adopt": raw.get("adopt"), "reasons": list(raw.get("reasons") or []),
            "regressions": list(raw.get("regressions") or []), "baseline": roles["baseline"],
            "candidate": roles["candidate"], "overall": overall, "per_class": per_class}


def load_compare(path: Path, catalog: ModelCatalog) -> dict:
    """``runs/compare_*.json`` de ``scripts/compare_pilot_models.py`` (contagens por split)."""
    raw = _read_json(path)
    models, splits = [], {}
    for entry in raw.get("models") or []:
        model_id = catalog.register(entry["weights"], entry.get("weights_sha256"))
        models.append(model_id)
        for split, counts in (entry.get("splits") or {}).items():
            truth, found = counts.get("truth") or 0, counts.get("found") or 0
            suggestions, correct = counts.get("suggestions") or 0, counts.get("correct") or 0
            splits.setdefault(split, {})[model_id] = {
                "images": counts.get("images"), "truth": truth, "found": found, "suggestions": suggestions,
                "correct": correct, "found_fraction": _ratio(found, truth),
                "correct_fraction": _ratio(correct, suggestions), "seen_in_training": split.endswith(SEEN_SUFFIX)}
    return {"file": str(path), "dataset": normalize_path(raw.get("dataset") or ""),
            "pilot_sha256": raw.get("pilot_sha256"), "conf": raw.get("conf"), "iou_match": raw.get("iou_match"),
            "models": models, "splits": splits}


def load_eval(path: Path, catalog: ModelCatalog) -> dict:
    """``runs/eval_por_classe_*.json``: ``model.val`` da Ultralytics mais contagens na confiança."""
    raw = _read_json(path)
    models = {}
    for weights, by_split in (raw.get("models") or {}).items():
        model_id = catalog.register(weights)
        models[model_id] = {}
        for split, values in (by_split or {}).items():
            overall = values.get("overall") or {}
            at_conf = {}
            for name, counts in (values.get("per_class_at_conf") or {}).items():
                tp, fp, fn = counts.get("tp") or 0, counts.get("fp") or 0, counts.get("fn") or 0
                truth = counts.get("truth")
                at_conf[name] = {"truth": tp + fn if truth is None else truth, "tp": tp, "fp": fp, "fn": fn,
                                 "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn)}
            ultralytics = {name: {"precision": item.get("P"), "recall": item.get("R"), "map50": item.get("mAP50")}
                           for name, item in (values.get("per_class_ultralytics") or {}).items()}
            models[model_id][split] = {
                "overall": {"map50": overall.get("mAP50"), "precision": overall.get("P"), "recall": overall.get("R"),
                            "map50_95": overall.get("mAP50-95")},
                "per_class_at_conf": at_conf, "per_class_ultralytics": ultralytics}
    return {"file": str(path), "conf": raw.get("conf"), "iou": raw.get("iou"), "models": models}


def _read_eval_curve(path: Path) -> list[dict]:
    """``gpu_eval/eval_curve.jsonl``: uma linha por época e split; linha truncada ou com erro é ignorada."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    points = []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue  # treino interrompido no meio da escrita
        # Época em que a avaliação falhou ({"epoch": n, "error": "..."}) não é um ponto.
        if (not isinstance(record, dict) or "error" in record or isinstance(record.get("epoch"), bool)
                or not isinstance(record.get("epoch"), int)):
            continue
        points.append({"epoch": record["epoch"], "split": str(record.get("split") or "test"),
                       "map50": _finite(record.get("map50")), "precision": _finite(record.get("precision")),
                       "recall": _finite(record.get("recall")), "seconds": _finite(record.get("seconds")),
                       "onnx_sha256": record.get("onnx_sha256")})
    return points


def _gpu_split(runs: list[dict]) -> str | None:
    """Split da curva na GPU: teste quando existe, senão validação, senão o primeiro gravado."""
    present = [point["split"] for run in runs for point in run["gpu_eval"] if point["map50"] is not None]
    for split in SPLIT_ORDER:
        if split in present:
            return split
    return present[0] if present else None


def load_run(directory: Path, catalog: ModelCatalog) -> dict:
    """Pasta ``runs/pilot_train_*``: curvas de validação, parâmetros, estado e avaliação na GPU."""
    directory = Path(directory)
    match = RUN_NAME.match(directory.name)
    execution = _optional_json(directory / "execution.json") or {}
    summary = _optional_json(directory / "summary.json") or {}
    parameters = execution.get("parameters") or {}
    weights = directory / "fit/weights/best.pt"
    # O treino entra no catálogo pelo seu best.pt: assim a curva usa a mesma cor
    # do modelo quando ele também aparece num gate ou numa comparação.
    model_id = catalog.register(weights, file_hash(weights) if weights.is_file() else None)
    curve = [{"epoch": point["epoch"], "map50": point.get("map50"), "precision": point.get("precision"),
              "recall": point.get("recall")} for point in read_curves(directory / "fit/results.csv")]
    best = max((point for point in curve if point["map50"] is not None), key=lambda point: point["map50"],
               default=None)
    return {"id": match["id"] if match else directory.name, "directory": str(directory), "model": model_id,
            "state": summary.get("state"), "parameters": {key: parameters.get(key) for key in ("epochs", "imgsz", "batch")},
            "checkpoint": (execution.get("runtime") or {}).get("checkpoint"), "epochs": len(curve),
            "best_val_map50": best["map50"] if best else None, "best_epoch": best["epoch"] if best else None,
            "curve": curve, "gpu_eval": _read_eval_curve(directory / "gpu_eval/eval_curve.jsonl")}


def discover_runs(root: Path) -> list[Path]:
    """Treinos piloto em ``runs/`` que já têm curva; pastas só preparadas ficam de fora."""
    return sorted(directory for directory in (root / "runs").glob("pilot_train_*")
                  if RUN_NAME.match(directory.name) and (directory / "fit/results.csv").is_file())


# ----------------------------------------------------------------------------
# Fontes dos gráficos (gate primeiro; eval_por_classe quando não há gate)
# ----------------------------------------------------------------------------

def _format_values(values) -> str:
    unique = sorted({value for value in values if value is not None}, key=str)
    return "/".join(f"{value:g}" if isinstance(value, (int, float)) else str(value) for value in unique) or "?"


def _gate_scope(gates: list[dict]) -> str:
    """Vale para gates e medições: ambos trazem ``conf`` e ``images_evaluated``."""
    return (f"conf {_format_values(gate['conf'] for gate in gates)}, "
            f"{_format_values(gate['images_evaluated'] for gate in gates)} fotos")


def wilson_interval(successes: int, total: int, z: float = WILSON_Z) -> tuple[float, float] | None:
    """Intervalo de Wilson para uma proporção: honesto com n pequeno, ao contrário do normal."""
    if not total:
        return None
    p, z2 = successes / total, z * z
    center = (p + z2 / (2 * total)) / (1 + z2 / total)
    half = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / (1 + z2 / total)
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


def _eval_split(evals: list[dict]) -> str | None:
    """Split honesto preferido: teste; senão validação; senão o primeiro que existir."""
    present = [split for item in evals for by_split in item["models"].values() for split in by_split]
    for split in SPLIT_ORDER:
        if split in present:
            return split
    return present[0] if present else None


def _overall_source(measurements: list[dict], gates: list[dict], evals: list[dict]) -> dict | None:
    values = {}
    if measurements:
        for measurement in measurements:
            for series_id, metrics in measurement["overall"].items():
                values.setdefault(series_id, metrics)
        scope = f"{_gate_scope(measurements)}, as mesmas para todas as séries"
        return {"kind": "measurement", "scope": scope,
                "title": f"Métricas gerais nas fotos fora do treino ({scope})", "values": values}
    if gates:
        for gate in gates:
            for model_id, metrics in gate["overall"].items():
                values.setdefault(model_id, metrics)
        return {"kind": "gate", "scope": _gate_scope(gates),
                "title": f"Métricas gerais nas fotos fora do treino ({_gate_scope(gates)})", "values": values}
    split = _eval_split(evals)
    if split is None:
        return None
    for item in evals:
        for model_id, by_split in item["models"].items():
            if split in by_split:
                values.setdefault(model_id, by_split[split]["overall"])
    scope = f"split {split}, model.val da Ultralytics"
    return {"kind": f"eval:{split}", "scope": scope, "title": f"Métricas gerais no {scope}", "values": values}


def _per_class_source(measurements: list[dict], gates: list[dict], evals: list[dict], names: list[str]) -> dict | None:
    classes: dict[str, dict] = {}
    if measurements or gates:
        # Medição e gate têm o mesmo bloco por classe; a medição manda quando existe.
        for item in measurements or gates:
            for name, by_model in item["per_class"].items():
                for model_id, values in by_model.items():
                    classes.setdefault(name, {}).setdefault(model_id, {
                        "tp": values.get("tp") or 0, "fp": values.get("fp") or 0, "fn": values.get("fn") or 0,
                        "boxes": values.get("human_boxes") or 0, "precision": values.get("precision"),
                        "recall": values.get("recall"), "ap50": values.get("ap50")})
        kind = "measurement" if measurements else "gate"
        scope = _gate_scope(measurements or gates)
    else:
        split = _eval_split(evals)
        if split is None:
            return None
        for item in evals:
            for model_id, by_split in item["models"].items():
                for name, values in (by_split.get(split) or {}).get("per_class_at_conf", {}).items():
                    classes.setdefault(name, {}).setdefault(model_id, {
                        "tp": values["tp"], "fp": values["fp"], "fn": values["fn"], "boxes": values["truth"],
                        "precision": values["precision"], "recall": values["recall"], "ap50": None})
        kind, scope = f"eval:{split}", f"split {split}, conf {_format_values(item['conf'] for item in evals)}"
    # Ordem fixa da taxonomia; classes desconhecidas (taxonomia antiga) vão ao fim.
    order = list(names) + sorted(set(classes) - set(names))
    return {"kind": kind, "scope": scope, "classes": order, "values": classes}


def _class_boxes(source: dict) -> dict[str, int]:
    """Caixas humanas por classe: o maior valor entre os modelos (devem coincidir num gate)."""
    return {name: max((values.get("boxes") or 0 for values in source["values"].get(name, {}).values()), default=0)
            for name in source["classes"]}


def _source_models(source: dict, catalog: ModelCatalog) -> list[dict]:
    """Modelos com dados na fonte: geral é ``{modelo: ...}``, por classe é ``{classe: {modelo: ...}}``."""
    if "classes" in source:
        ids = {model_id for by_model in source["values"].values() for model_id in by_model}
    else:
        ids = set(source["values"])
    return catalog.ordered(ids)


# ----------------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------------

def _plotting():
    """Backend Agg só na hora de desenhar: importar este módulo continua barato."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _points(pixels: float) -> float:
    """matplotlib mede linhas em pontos; a especificação dos gráficos é em pixels."""
    return pixels * 72 / DPI


def _lighten(color: str, amount: float) -> str:
    """Mistura a cor com branco (0 = igual, 1 = branco): o FP hachurado na cor do modelo."""
    channels = (int(color[index:index + 2], 16) for index in (1, 3, 5))
    return "#" + "".join(f"{round(channel + (255 - channel) * amount):02x}" for channel in channels)


def _data_per_pixel(ax, axis: str) -> float:
    """Quantas unidades do eixo cabem em um pixel (para a folga de 2 px entre barras)."""
    inverse = ax.transData.inverted()
    origin, step = inverse.transform((0.0, 0.0)), inverse.transform((1.0, 1.0))
    return abs(step[0] - origin[0]) if axis == "x" else abs(step[1] - origin[1])


def _save(plt, fig, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _fraction_axis(ax, which: str) -> None:
    """Eixo de 0,0 a 1,0 em frações (nunca porcentagem, nunca dois eixos)."""
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator

    axis = ax.yaxis if which == "y" else ax.xaxis
    (ax.set_ylim if which == "y" else ax.set_xlim)(0, 1.06)  # folga para o rótulo da barra cheia
    axis.set_major_locator(MultipleLocator(0.2))
    axis.set_major_formatter(FormatStrFormatter("%.1f"))


def _legend(ax, **kwargs) -> None:
    """Legenda sem repetição: vários modelos "outros" viram uma entrada só."""
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys(), **kwargs)


def _grouped_bars(ax, categories: list[str], series: list[tuple], horizontal: bool = False,
                  value_labels: bool = False) -> None:
    """Barras finas lado a lado por categoria; cada série é (rótulo, cor, valores) de um modelo."""
    count = max(len(series), 1)
    positions = list(range(len(categories)))
    if horizontal:
        ax.set_ylim(len(categories) - 0.5, -0.5)  # primeira categoria no topo
        ax.set_yticks(positions, categories)
    else:
        ax.set_xlim(-0.5, len(categories) - 0.5)
        ax.set_xticks(positions, categories)
    slot = GROUP_WIDTH / count
    width = max(slot - BAR_GAP_PX * _data_per_pixel(ax, "y" if horizontal else "x"), slot / 2)
    for index, (label, color, values, *rest) in enumerate(series):
        offset = (index - (count - 1) / 2) * slot
        intervals = rest[0] if rest else [None] * len(values)
        # Valor ausente (None) não vira barra zero: zero mede, ausente não.
        placed = [(position + offset, value, interval)
                  for position, value, interval in zip(positions, values, intervals) if value is not None]
        if not placed:
            continue
        at, heights, bounds = zip(*placed)
        errors = None
        if any(bound is not None for bound in bounds):
            # Barra de erro fina (intervalo de Wilson) do valor até cada extremo.
            errors = [[value - (bound[0] if bound else value) for value, bound in zip(heights, bounds)],
                      [(bound[1] if bound else value) - value for value, bound in zip(heights, bounds)]]
        error_kw = {"ecolor": "#444444", "elinewidth": 0.7, "capsize": 2, "capthick": 0.7}
        if horizontal:
            bars = ax.barh(at, heights, height=width, color=color, label=label, xerr=errors, error_kw=error_kw)
        else:
            bars = ax.bar(at, heights, width=width, color=color, label=label, yerr=errors, error_kw=error_kw)
        if value_labels:
            ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2, color="#333")


def _chart_overall(plt, path: Path, source: dict, models: list[dict], catalog: ModelCatalog) -> None:
    metrics = (("map50", "mAP50"), ("precision", "Precisão"), ("recall", "Recall"))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        series = [(catalog.legend_label(model), model["color"],
                   [_finite(source["values"][model["id"]].get(key)) for key, _ in metrics]) for model in models]
        _grouped_bars(ax, [label for _, label in metrics], series, value_labels=True)
        _fraction_axis(ax, "y")
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("Métrica")
        ax.set_ylabel("Valor (fração de 0,0 a 1,0)")
        ax.set_title(source["title"])
        if len(models) >= 2:
            _legend(ax, loc="upper right")
        _save(plt, fig, path)


def _chart_per_class_recall(plt, path: Path, source: dict, models: list[dict], catalog: ModelCatalog) -> None:
    classes, values, boxes = source["classes"], source["values"], _class_boxes(source)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 1.6 + 0.42 * len(classes)))
        series = []
        for model in models:
            recalls, intervals = [], []
            for name in classes:
                counts = values.get(name, {}).get(model["id"], {})
                # Com menos de 3 caixas a barra engana mais do que informa: só o texto.
                measured = boxes[name] >= FEW_BOXES and counts.get("recall") is not None
                recalls.append(_finite(counts["recall"]) if measured else None)
                intervals.append(wilson_interval(counts.get("tp") or 0, counts.get("boxes") or 0) if measured else None)
            series.append((catalog.legend_label(model), model["color"], recalls, intervals))
        _grouped_bars(ax, [f"{name} ({boxes[name]})" for name in classes], series, horizontal=True)
        _fraction_axis(ax, "x")
        ax.grid(axis="y", visible=False)
        for index, name in enumerate(classes):
            if boxes[name] < FEW_BOXES:
                text = "não medida" if not boxes[name] else "n<3: pouco medida"
                ax.text(0.01, index, text, va="center", ha="left", fontsize=8, color="#777", style="italic")
        ax.set_xlabel("Recall (caixas humanas reencontradas / caixas humanas; barra fina = IC 95% de Wilson)")
        ax.set_ylabel("Classe (número de caixas humanas)")
        ax.set_title(f"Recall por classe ({source['scope']})")
        if len(models) >= 2:
            _legend(ax, loc="lower right")
        _save(plt, fig, path)


def _chart_per_class_errors(plt, path: Path, source: dict, models: list[dict]) -> None:
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    values, boxes = source["values"], _class_boxes(source)
    # Classe sem caixa humana entra só quando algum modelo inventou caixas nela (alucinação).
    classes = [name for name in source["classes"] if boxes[name]
               or any((counts.get("fp") or 0) for counts in values.get(name, {}).values())]
    labels = [name if boxes[name] else f"{name} (0 caixas humanas)" for name in classes]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(models), sharey=True, squeeze=False,
                                 figsize=(2.8 * len(models) + 1.6, 0.34 * len(classes) + 2.2))
        positions = list(range(len(classes)))
        for ax, model in zip(axes[0], models):
            counts = [values.get(name, {}).get(model["id"], {}) for name in classes]
            tp = [item.get("tp") or 0 for item in counts]
            fp = [item.get("fp") or 0 for item in counts]
            fn = [item.get("fn") or 0 for item in counts]
            ax.barh(positions, tp, height=0.6, color=model["color"])
            ax.barh(positions, fp, left=tp, height=0.6, color=_lighten(model["color"], 0.55),
                    edgecolor=model["color"], hatch="//", linewidth=0.5)
            ax.barh(positions, fn, left=[a + b for a, b in zip(tp, fp)], height=0.6, color=FN_COLOR)
            ax.set_ylim(len(classes) - 0.5, -0.5)
            ax.set_yticks(positions, labels)
            ax.grid(axis="y", visible=False)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("Caixas")
            ax.set_title(model["label"])
        handles = [Patch(facecolor="#555555", label="TP: acertos (na cor do modelo)"),
                   Patch(facecolor="white", edgecolor="#555555", hatch="//", label="FP: falsos positivos"),
                   Patch(facecolor=FN_COLOR, label="FN: caixas humanas não encontradas")]
        fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f"Acertos e erros por classe ({source['scope']})")
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        _save(plt, fig, path)


def _chart_strata(plt, path: Path, measurements: list[dict], catalog: ModelCatalog) -> bool:
    """Recall total e mAP50 por estrato (fotos fatiadas do drone × fotos inteiras do laudo)."""
    strata: dict[str, dict] = {}
    for measurement in measurements:
        for stratum, by_series in measurement["strata"].items():
            for series_id, totals in by_series.items():
                strata.setdefault(stratum, {}).setdefault(series_id, totals)
    if not strata:
        return False
    order = sorted(strata, key=lambda name: (list(STRATA).index(name) if name in STRATA else len(STRATA), name))
    models = catalog.ordered({series_id for by_series in strata.values() for series_id in by_series})
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(4.2 * len(order) + 3, 4.4))
        for ax, (key, title) in zip(axes, (("recall", "Recall total"), ("map50", "mAP50"))):
            series = [(catalog.legend_label(model), model["color"],
                       [_finite(strata[name].get(model["id"], {}).get(key)) for name in order]) for model in models]
            _grouped_bars(ax, [STRATA.get(name, name) for name in order], series, value_labels=True)
            _fraction_axis(ax, "y")
            ax.grid(axis="x", visible=False)
            ax.set_xlabel("Estrato de fotos")
            ax.set_title(title)
        axes[0].set_ylabel("Valor (fração de 0,0 a 1,0)")
        if len(models) >= 2:
            _legend(axes[0], loc="upper left")
        fig.suptitle(f"Recall e mAP50 por estrato ({_gate_scope(measurements)})")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _save(plt, fig, path)
    return True


def _curve_points(run: dict, key: str, split: str | None) -> list[tuple[int, float]]:
    return sorted((point["epoch"], point["map50"]) for point in run[key]
                  if point["map50"] is not None and (split is None or point.get("split") == split))


def _chart_curves(plt, path: Path, runs: list[dict], catalog: ModelCatalog, key: str, title: str,
                  ylabel: str, split: str | None = None) -> bool:
    """Uma linha por treino; devolve False (e não grava) quando nenhum treino tem a curva."""
    from matplotlib.ticker import MaxNLocator

    drawable = [run for run in runs if _curve_points(run, key, split)]
    if not drawable:
        return False
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for run in drawable:
            model = catalog.by_id(run["model"])
            points = _curve_points(run, key, split)
            xs, ys = zip(*points)
            label = run["id"] if model["color"] != OTHERS_COLOR else OTHERS_LABEL
            ax.plot(xs, ys, color=model["color"], linewidth=_points(LINE_PX),
                    marker="o" if len(points) <= MAX_MARKER_POINTS else None, markersize=3.5, label=label)
        _fraction_axis(ax, "y")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Época")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if len(drawable) >= 2:
            _legend(ax, loc="best")
        _save(plt, fig, path)
    return True


def _split_base(split: str) -> tuple[str, bool]:
    seen = split.endswith(SEEN_SUFFIX)
    return (split[:-len(SEEN_SUFFIX)] if seen else split), seen


def _split_key(category: str) -> tuple:
    base = category.split("\n")[0]
    return (SPLIT_ORDER.index(base) if base in SPLIT_ORDER else len(SPLIT_ORDER), category)


def _chart_compare(plt, path: Path, compares: list[dict], catalog: ModelCatalog) -> None:
    # Fotos vistas no treino ficam num painel à parte: lá o modelo pode só ter decorado.
    panels = {False: {}, True: {}}
    datasets = {compare["dataset"] for compare in compares}
    for compare in compares:
        suffix = f"\n{PurePosixPath(compare['dataset']).parent.name}" if len(datasets) > 1 else ""
        for split, by_model in compare["splits"].items():
            base, seen = _split_base(split)
            for model_id, counts in by_model.items():
                panels[seen].setdefault(base + suffix, {}).setdefault(model_id, counts["found_fraction"])
    panels = {seen: data for seen, data in panels.items() if data}
    ids = {model_id for data in panels.values() for by_model in data.values() for model_id in by_model}
    models = catalog.ordered(ids)
    titles = {False: "Fotos fora do treino", True: "Fotos vistas no treino (memorização, não generalização)"}
    widths = [max(len(data), 1) for data in panels.values()]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(panels), sharey=True, squeeze=False,
                                 figsize=(2.2 * sum(widths) + 2.5, 4.4), gridspec_kw={"width_ratios": widths})
        for ax, (seen, data) in zip(axes[0], panels.items()):
            categories = sorted(data, key=_split_key)
            series = [(catalog.legend_label(model), model["color"],
                       [_finite(data[category].get(model["id"])) for category in categories]) for model in models]
            _grouped_bars(ax, categories, series, value_labels=True)
            _fraction_axis(ax, "y")
            ax.grid(axis="x", visible=False)
            ax.set_xlabel("Split")
            ax.set_title(titles[seen])
        axes[0][0].set_ylabel("Caixas humanas reencontradas (fração)")
        if len(models) >= 2:
            _legend(axes[0][0], loc="upper left")
        scope = f"conf {_format_values(c['conf'] for c in compares)}, IoU ≥ {_format_values(c['iou_match'] for c in compares)}"
        fig.suptitle(f"Caixas humanas reencontradas por split ({scope})")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _save(plt, fig, path)


# ----------------------------------------------------------------------------
# Notas e HTML
# ----------------------------------------------------------------------------

def _notes(measurements: list[dict], gates: list[dict], per_class: dict | None, overall: dict | None,
           archive: bool, catalog: ModelCatalog, skipped: list[str]) -> list[str]:
    notes = []
    for item, kind in [(m, "medição") for m in measurements] + [(g, "gate") for g in gates]:
        excluded = item["images_excluded_seen_in_training"]
        notes.append(f"Fotos vistas no treino de qualquer um dos modelos foram excluídas das métricas do {kind} "
                     f"({excluded if excluded is not None else '?'} excluídas).")
    notes.append("Nenhum resultado aqui aprova um modelo para produção: a avaliação usa uma única edificação.")
    if measurements:
        notes.append("Na medição principal todas as séries foram avaliadas nas mesmas fotos; o mesmo peso em dois "
                     "regimes (janelas × inteira) aparece como duas séries.")
    if archive:
        notes.append(f"{ARCHIVE_TITLE}: cada arquivo usou outro conjunto de fotos; compare só dentro de cada tabela.")
    for gate in gates:
        baseline, candidate = catalog.by_id(gate["baseline"])["label"], catalog.by_id(gate["candidate"])["label"]
        verdict = "sim" if gate["adopt"] else "não"
        reasons = "; ".join(gate["reasons"]) if gate["reasons"] else "sem ressalvas pela regra do gate"
        notes.append(f"Gate {PurePosixPath(normalize_path(gate['file'])).parent.name}: adotar o candidato "
                     f"{candidate} no lugar de {baseline}? {verdict}. Motivos: {reasons}")
    if per_class:
        boxes = _class_boxes(per_class)
        few = [f"{name} ({boxes[name]})" for name in per_class["classes"] if boxes[name] < FEW_BOXES]
        if few:
            notes.append(f"Classes pouco medidas (menos de {FEW_BOXES} caixas humanas): {', '.join(few)}. "
                         "Os números delas não sustentam conclusão.")
        if per_class["kind"].startswith("eval"):
            notes.append(f"Sem gate.json: métricas por classe vêm de eval_por_classe ({per_class['scope']}), "
                         "que não exclui fotos vistas no treino.")
    if overall and overall["kind"].startswith("eval"):
        notes.append(f"Métricas gerais vêm de eval_por_classe ({overall['scope']}): precisão e recall são os do "
                     "model.val da Ultralytics, não os da confiança das sugestões.")
    notes.extend(skipped)
    return notes


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "sim" if value else "não"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _cell(value, digits: int = 3) -> str:
    return _esc(_fmt(value, digits))


def _table(headers: list[str], rows: list[list[str]], caption: str | None = None) -> str:
    """Células já escapadas; os cabeçalhos são escapados aqui."""
    head = "".join(f"<th>{_esc(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    title = f"<caption>{_esc(caption)}</caption>" if caption else ""
    return f"<table>{title}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _figure(directory: Path, name: str, alt: str) -> str:
    """PNG embutido em base64: a página funciona copiada sozinha ou aberta do disco."""
    encoded = base64.b64encode((directory / name).read_bytes()).decode("ascii")
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{_esc(alt)}"><figcaption>{_esc(name)}</figcaption></figure>'


def _swatch(model: dict) -> str:
    return f'<span class="swatch" style="background:{_esc(model["color"])}"></span>{_esc(model["label"])}'


CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,Ubuntu,sans-serif;max-width:1100px;margin:2rem auto;
padding:0 1rem;color:#222;line-height:1.45;background:#fff}
h1{font-size:1.6rem;margin-bottom:.2rem}h2{font-size:1.2rem;margin-top:2.2rem;border-bottom:1px solid #ddd;padding-bottom:.2rem}
h3{font-size:1rem;margin-top:1.4rem}
table{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.92rem}
th,td{border:1px solid #ccc;padding:.35rem .55rem;text-align:left;vertical-align:top}
th{background:#eef1f4}tbody tr:nth-child(even){background:#f6f8fa}caption{text-align:left;font-weight:600;padding:.3rem 0}
figure{margin:1rem 0}img{max-width:100%;height:auto;border:1px solid #e3e3e3;background:#fff}
figcaption{font-size:.8rem;color:#666}code{font-size:.88em;word-break:break-all}
.muted{color:#666}.swatch{display:inline-block;width:.85em;height:.85em;border-radius:2px;vertical-align:-.05em;margin-right:.4em}
.archive{border-left:4px solid #e0a800;padding-left:1rem;margin-top:1rem}ul.notes li{margin:.3rem 0}
"""


def _overall_section(parts: list[str], directory: Path, source: dict | None, chart: str, charts: list[str],
                     gates: list[dict], catalog: ModelCatalog, level: str) -> None:
    parts.append(f"<{level}>Métricas gerais</{level}>")
    if not source:
        parts.append("<p class='muted'>Sem medição, gate.json nem eval_por_classe: nenhuma métrica geral.</p>")
        return
    if chart in charts:
        parts.append(_figure(directory, chart, "Métricas gerais por modelo"))
    roles = {}
    for gate in gates:
        roles.setdefault(gate["baseline"], "atual (baseline)")
        roles.setdefault(gate["candidate"], "candidato")
    values = source["values"]
    parts.append(_table(["Série", "Papel", "mAP50", "Precisão", "Recall"], [
        [_swatch(model), _cell(roles.get(model["id"])), _cell(values[model["id"]].get("map50")),
         _cell(values[model["id"]].get("precision")), _cell(values[model["id"]].get("recall"))]
        for model in _source_models(source, catalog)], caption=source["title"]))


def _recall_cell(counts: dict, boxes: int) -> str:
    if not boxes:
        return "não medida"
    if boxes < FEW_BOXES:
        return _esc("n<3: pouco medida")
    interval = wilson_interval(counts.get("tp") or 0, counts.get("boxes") or boxes)
    recall = counts.get("recall")
    if recall is None or interval is None:
        return "—"
    return _esc(f"{_fmt(recall)} [{_fmt(interval[0], 2)}–{_fmt(interval[1], 2)}]")


def _per_class_section(parts: list[str], directory: Path, source: dict | None, suffix: str, charts: list[str],
                       catalog: ModelCatalog, level: str) -> None:
    parts.append(f"<{level}>Por classe</{level}>")
    if not source:
        parts.append("<p class='muted'>Sem medição, gate.json nem eval_por_classe: nenhuma métrica por classe.</p>")
        return
    models = _source_models(source, catalog)
    for name, alt in ((f"per_class_recall{suffix}.png", "Recall por classe"),
                      (f"per_class_errors{suffix}.png", "Acertos e erros por classe")):
        if name in charts:
            parts.append(_figure(directory, name, alt))
    boxes = _class_boxes(source)
    headers = ["Classe", "Caixas humanas"]
    for model in models:
        headers += [f"TP/FP/FN · {model['label']}", f"Recall [IC95] · {model['label']}", f"AP50 · {model['label']}"]
    rows = []
    for name in source["classes"]:
        by_model = source["values"].get(name, {})
        row = [_esc(name), _cell(boxes[name])]
        for model in models:
            counts = by_model.get(model["id"], {})
            row.append(_esc(f"{counts.get('tp') or 0}/{counts.get('fp') or 0}/{counts.get('fn') or 0}"))
            row.append(_recall_cell(counts, boxes[name]))
            row.append(_cell(counts.get("ap50")) if boxes[name] >= FEW_BOXES else "—")
        rows.append(row)
    parts.append(_table(headers, rows, caption=f"Contagens na confiança das sugestões ({source['scope']}); "
                                               f"AP50 só com {FEW_BOXES} ou mais caixas humanas"))


def _compare_section(parts: list[str], directory: Path, compares: list[dict], chart: str, charts: list[str],
                     catalog: ModelCatalog, level: str) -> None:
    parts.append(f"<{level}>Reencontradas por split</{level}>")
    if not compares:
        parts.append("<p class='muted'>Nenhum compare_*.json informado.</p>")
        return
    if chart in charts:
        parts.append(_figure(directory, chart, "Caixas humanas reencontradas por split"))
    parts.append("<p class='muted'>Splits marcados como “visto no treino” contêm fotos que o modelo treinou: "
                 "medem memorização, não generalização, e ficam separados.</p>")
    for compare in compares:
        models = catalog.ordered(compare["models"])
        headers = (["Split"] + [f"Reencontradas / caixas · {model['label']}" for model in models]
                   + [f"Corretas / sugestões · {model['label']}" for model in models])
        rows = []
        for split in sorted(compare["splits"], key=lambda value: (_split_base(value)[1], _split_key(_split_base(value)[0]))):
            by_model = compare["splits"][split]
            found = [(f"{_cell(c['found'])} / {_cell(c['truth'])} ({_cell(c['found_fraction'], 2)})" if (c := by_model.get(model["id"])) else "—")
                     for model in models]
            correct = [(f"{_cell(c['correct'])} / {_cell(c['suggestions'])} ({_cell(c['correct_fraction'], 2)})" if (c := by_model.get(model["id"])) else "—")
                       for model in models]
            rows.append([_esc(split)] + found + correct)
        caption = f"{PurePosixPath(normalize_path(compare['file'])).name} · dataset {compare['dataset'] or '?'} · conf {_fmt(compare['conf'])} · IoU ≥ {_fmt(compare['iou_match'])}"
        parts.append(_table(headers, rows, caption=caption))


def _render_html(directory: Path, title: str, created_at: str, catalog: ModelCatalog, measurements: list[dict],
                 gates: list[dict], compares: list[dict], runs: list[dict], sources: dict, charts: list[str],
                 notes: list[str]) -> str:
    parts = [f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
             f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{_esc(title)}</title>'
             f"<style>{CSS}</style></head><body>",
             f"<h1>{_esc(title)}</h1><p class='muted'>Gerado em {_esc(created_at)} · pasta <code>{_esc(directory)}</code></p>"]

    # Séries: a mesma cor e o mesmo rótulo valem para todos os gráficos abaixo.
    parts.append("<h2>Modelos</h2>")
    parts.append(_table(["Série", "Arquivo", "SHA-256", "Treino", "Regime"], [
        [_swatch(model), f"<code>{_esc(model['weights'])}</code>",
         f"<code>{_esc((model['sha256'] or '')[:12] or '—')}</code>", _cell(model["run_id"]), _cell(model["regime"])]
        for model in catalog.models]))

    if measurements:
        _overall_section(parts, directory, sources["overall"], "overall_metrics.png", charts, [], catalog, "h2")
        _per_class_section(parts, directory, sources["per_class"], "", charts, catalog, "h2")
        parts.append("<h2>Estratos de fotos</h2>")
        if "strata_recall.png" in charts:
            parts.append(_figure(directory, "strata_recall.png", "Recall e mAP50 por estrato"))
            counts = {}
            for measurement in measurements:
                for stratum, number in measurement["strata_images"].items():
                    counts.setdefault(stratum, number)
            parts.append("<p class='muted'>" + _esc("Fotos por estrato: " + ", ".join(
                f"{STRATA.get(name, name)} = {_fmt(number)}" for name, number in counts.items())) + "</p>")
        else:
            parts.append("<p class='muted'>A medição não trouxe métricas por estrato.</p>")
        if gates or compares or sources["archive_overall"] or sources["archive_per_class"]:
            parts.append(f"<h2>{_esc(ARCHIVE_TITLE)}</h2><div class='archive'>")
            _overall_section(parts, directory, sources["archive_overall"], "overall_metrics_arquivo.png", charts,
                             gates, catalog, "h3")
            _per_class_section(parts, directory, sources["archive_per_class"], "_arquivo", charts, catalog, "h3")
            _compare_section(parts, directory, compares, "compare_found_arquivo.png", charts, catalog, "h3")
            parts.append("</div>")
    else:
        _overall_section(parts, directory, sources["overall"], "overall_metrics.png", charts, gates, catalog, "h2")
        _per_class_section(parts, directory, sources["per_class"], "", charts, catalog, "h2")
        _compare_section(parts, directory, compares, "compare_found.png", charts, catalog, "h2")

    parts.append("<h2>Curvas de treino</h2>")
    if runs:
        if "training_curves.png" in charts:
            parts.append(_figure(directory, "training_curves.png", "mAP50 de validação por época"))
        else:
            parts.append("<p class='muted'>Nenhum treino com fit/results.csv legível.</p>")
        rows = []
        for run in runs:
            model = catalog.by_id(run["model"])
            parameters = run["parameters"]
            rows.append([_esc(run["id"]), _swatch(model), _cell(run["state"] or "desconhecido"), _cell(run["epochs"]),
                         _cell(run["best_val_map50"]), _cell(run["best_epoch"]),
                         _esc(f"{_fmt(parameters.get('epochs'))} / {_fmt(parameters.get('imgsz'))} / {_fmt(parameters.get('batch'))}"),
                         _cell(len(run["gpu_eval"]))])
        parts.append(_table(["Treino", "Modelo", "Estado", "Épocas com curva", "Melhor mAP50 (val)", "Época",
                             "epochs / imgsz / batch", "Pontos da avaliação GPU"], rows))
        parts.append("<h3>Avaliação no teste durante o treino (GPU)</h3>")
        if "gpu_eval_curves.png" in charts:
            parts.append(_figure(directory, "gpu_eval_curves.png", "mAP50 por época na avaliação em GPU (fotos inteiras)"))
        else:
            parts.append("<p class='muted'>Nenhum treino tem gpu_eval/eval_curve.jsonl com mAP50.</p>")
    else:
        parts.append("<p class='muted'>Nenhum treino piloto informado ou encontrado em runs/.</p>")

    parts.append("<h2>Notas</h2><ul class='notes'>" + "".join(f"<li>{_esc(note)}</li>" for note in notes) + "</ul>")
    parts.append("</body></html>")
    return "\n".join(parts)


# ----------------------------------------------------------------------------
# Montagem do relatório
# ----------------------------------------------------------------------------

def _per_class_charts(plt, directory: Path, source: dict | None, suffix: str, catalog: ModelCatalog,
                      charts: list[str]) -> None:
    if not (source and source["values"]):
        return
    models = _source_models(source, catalog)
    _chart_per_class_recall(plt, directory / f"per_class_recall{suffix}.png", source, models, catalog)
    charts.append(f"per_class_recall{suffix}.png")
    _chart_per_class_errors(plt, directory / f"per_class_errors{suffix}.png", source, models)
    charts.append(f"per_class_errors{suffix}.png")


def _overall_chart(plt, directory: Path, source: dict | None, name: str, catalog: ModelCatalog,
                   charts: list[str]) -> None:
    if source and source["values"]:
        _chart_overall(plt, directory / name, source, _source_models(source, catalog), catalog)
        charts.append(name)


def build_model_report(config: ProjectConfig, output: str | Path | None = None, gates=(), compares=(), evals=(),
                       runs=None, title: str | None = None, measurements=()) -> Path:
    """Grava ``models.json``, os PNGs com dados e ``report.html`` na pasta de saída.

    ``measurements`` (``measurement.json``) é a comparação principal: N séries nas
    mesmas fotos. Com ela presente, gates, comparações e avaliações viram a seção
    "medições anteriores", com gráficos ``*_arquivo.png``. ``runs`` None usa todos
    os ``runs/pilot_train_*`` do projeto que já têm curva; ``output`` None cria
    ``runs/report_<data>_<id>``. Levanta ``ValueError`` quando nenhuma entrada
    traz dados.
    """
    names = detection_names(load_taxonomy(config.taxonomy_path))
    catalog = ModelCatalog()
    # Ordem de registro define cor e posição: medições, gates, comparações, avaliações, treinos.
    measurement_data = [load_measurement(Path(path), catalog) for path in measurements]
    gate_data = [load_gate(Path(path), catalog) for path in gates]
    compare_data = [load_compare(Path(path), catalog) for path in compares]
    eval_data = [load_eval(Path(path), catalog) for path in evals]
    run_directories = [Path(path) for path in runs] if runs is not None else discover_runs(Path(config.root))
    run_data, skipped = [], []
    for directory in run_directories:
        run = load_run(directory, catalog)
        if run["curve"] or run["gpu_eval"]:
            run_data.append(run)
        else:
            skipped.append(f"Treino {run['id']} ignorado: sem fit/results.csv nem gpu_eval/eval_curve.jsonl legíveis.")
    if not (measurement_data or gate_data or compare_data or eval_data or run_data):
        raise ValueError("Nenhum dado para o relatório: informe measurement.json, gate.json, compare_*.json, "
                         "eval_por_classe_*.json ou treinos em runs/pilot_train_* com fit/results.csv.")

    directory = Path(output) if output is not None else make_run_directory(Path(config.root), "report")
    if output is not None:
        directory.mkdir(parents=True, exist_ok=False)  # nunca sobrescreve um relatório anterior
    title = title or "Comparativo de modelos de detecção"
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Com medição, gate/eval vão para o arquivo; sem medição, eles são a fonte principal.
    archive = bool(measurement_data) and bool(gate_data or eval_data or compare_data)
    sources = {"overall": _overall_source(measurement_data, gate_data, eval_data),
               "per_class": _per_class_source(measurement_data, gate_data, eval_data, names),
               "archive_overall": _overall_source([], gate_data, eval_data) if archive else None,
               "archive_per_class": _per_class_source([], gate_data, eval_data, names) if archive else None}
    plt = _plotting()
    charts: list[str] = []
    _overall_chart(plt, directory, sources["overall"], "overall_metrics.png", catalog, charts)
    _per_class_charts(plt, directory, sources["per_class"], "", catalog, charts)
    if measurement_data and _chart_strata(plt, directory / "strata_recall.png", measurement_data, catalog):
        charts.append("strata_recall.png")
    _overall_chart(plt, directory, sources["archive_overall"], "overall_metrics_arquivo.png", catalog, charts)
    _per_class_charts(plt, directory, sources["archive_per_class"], "_arquivo", catalog, charts)
    if _chart_curves(plt, directory / "training_curves.png", run_data, catalog, "curve",
                     "mAP50 de validação por época", "mAP50 de validação"):
        charts.append("training_curves.png")
    gpu_split = _gpu_split(run_data)
    if gpu_split and _chart_curves(plt, directory / "gpu_eval_curves.png", run_data, catalog, "gpu_eval",
                                   f"mAP50 no split {gpu_split} durante o treino (avaliação na GPU)",
                                   f"mAP50 no split {gpu_split} (GPU, fotos inteiras)", split=gpu_split):
        charts.append("gpu_eval_curves.png")
    if any(compare["splits"] for compare in compare_data):
        compare_chart = "compare_found_arquivo.png" if measurement_data else "compare_found.png"
        _chart_compare(plt, directory / compare_chart, compare_data, catalog)
        charts.append(compare_chart)

    notes = _notes(measurement_data, gate_data, sources["per_class"], sources["overall"], archive, catalog, skipped)
    write_json(directory / "models.json", _clean({
        "schema_version": SCHEMA_VERSION, "created_at": created_at, "title": title,
        "models": [{key: model[key] for key in ("id", "label", "weights", "sha256", "run_id", "regime", "color")}
                   for model in catalog.models],
        "measurements": measurement_data, "gates": gate_data, "compares": compare_data, "evals": eval_data,
        "runs": [{key: run[key] for key in ("id", "directory", "model", "state", "parameters", "checkpoint", "epochs",
                                            "best_val_map50", "best_epoch", "gpu_eval")} for run in run_data],
        "charts": charts, "notes": notes,
    }))
    page = _render_html(directory, title, created_at, catalog, measurement_data, gate_data, compare_data, run_data,
                        sources, charts, notes)
    (directory / "report.html").write_text(page, encoding="utf-8")
    return directory
