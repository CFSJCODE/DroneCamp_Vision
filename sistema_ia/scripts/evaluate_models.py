"""Avaliação comparativa por classe, no mesmo pipeline das sugestões, com gráficos.

Para cada modelo (.pt ou .onnx exportado) e cada conjunto de fotos com gabarito
humano, roda ``detect_boxes`` (fatiamento de ``pilot.tiling`` nas fotos grandes) e
pareia caixas da mesma classe com IoU >= 0,5. Mede por classe:

- na confiança das sugestões (``pilot.suggestion_conf``): acertos, falsos
  positivos, omissões, precisão, recall e F1;
- AP50 (interpolação de todos os pontos, como no VOC) com as predições a partir
  de ``--ap-conf`` (padrão 0,01). Calculado aqui porque o ``val`` da Ultralytics
  avalia a foto inteira reduzida e não reproduz a inferência fatiada.

Conjuntos de fotos:
- ``--data data/pilot/V/dataset.yaml``: splits val e test do dataset piloto (fotos
  vistas no treino de um modelo são marcadas e ficam fora da média daquele modelo);
- ``--review REGISTRO.json FEEDBACK.json``: fotos confirmadas num arquivo de revisão
  (ex.: Lado A em andamento), gabarito independente para modelos que não as viram.

Uso:
  python scripts/evaluate_models.py --data data/pilot/ceasa_v9_piloto_s42/dataset.yaml \
      --review data/reviews/ceasa_lado_a_v1/registry.json data/reviews/ceasa_lado_a_v1/feedback_inbox/X.json \
      --weights runs/export_A/model.onnx --label A --weights runs/export_B/model.onnx --label B \
      --output runs/relatorio_modelos_X

Gera ``report.json``, gráficos PNG e ``index.html`` na pasta de saída. Nenhum
modelo é aprovado por este relatório: ele só organiza a evidência.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dronecamp_ia.backend import check_domain_names, load_inference_model  # noqa: E402
from dronecamp_ia.config import detection_names, load_config, load_taxonomy  # noqa: E402
from dronecamp_ia.io import file_hash, resolve_local_path, write_json  # noqa: E402
from dronecamp_ia.pilot import box_iou  # noqa: E402
from dronecamp_ia.suggestions import detect_boxes  # noqa: E402
from dronecamp_ia.training import pilot_inference_config  # noqa: E402

IOU_MATCH = 0.5
DEDUP_IOU = 0.7


# ---------------------------------------------------------------------------
# Gabaritos: fotos com caixas humanas.
# ---------------------------------------------------------------------------

def _dedup(boxes: list[dict]) -> list[dict]:
    """Mesma classe com IoU >= 0,7 conta uma vez (mesma política do build-pilot-data)."""
    kept = []
    for box in boxes:
        if not any(k["class_id"] == box["class_id"] and box_iou(k["bbox_xyxy"], box["bbox_xyxy"]) >= DEDUP_IOU for k in kept):
            kept.append({"class_id": int(box["class_id"]), "bbox_xyxy": [float(v) for v in box["bbox_xyxy"]]})
    return kept


def pilot_sets(data: Path) -> dict[str, list[dict]]:
    """Fotos val/test do dataset piloto com caixas do label YOLO."""
    from compare_pilot_models import ground_truth

    root = data.resolve().parent
    manifest = json.loads((root / "pilot.json").read_text(encoding="utf-8"))
    sets = defaultdict(list)
    for sample in manifest["images"]:
        if sample["split"] in {"val", "test"}:
            sets[f"{manifest['version']}:{sample['split']}"].append({
                "image": root / sample["image"], "sha256": sample["image_sha256"], "truth": _dedup(ground_truth(root, sample))})
    return dict(sets)


def review_set(registry: Path, feedback: Path) -> dict[str, list[dict]]:
    """Fotos confirmadas (confirmed_complete) de um arquivo de revisão exportado."""
    registry_data = json.loads(registry.read_text(encoding="utf-8"))
    sources = {item["image_sha256"]: item for item in registry_data["images"]}
    exported = json.loads(feedback.read_text(encoding="utf-8"))
    if exported.get("registry_sha256") != file_hash(registry):
        raise ValueError("Arquivo de revisão de outra versão do registro.")
    photos = []
    for item in exported["images"]:
        if not item.get("confirmed_complete") or item.get("status") not in {"positive", "negative"}:
            continue
        source = resolve_local_path(sources[item["image_sha256"]]["source_path"], ROOT)
        photos.append({"image": source, "sha256": item["image_sha256"], "truth": _dedup(item["boxes"])})
    return {f"{registry_data['version']}:revisado": photos}


# ---------------------------------------------------------------------------
# Métricas.
# ---------------------------------------------------------------------------

def _match(predictions: list[dict], truth: list[dict]) -> list[tuple[dict, bool]]:
    """Pareamento guloso por confiança: cada caixa humana conta uma vez."""
    used, out = set(), []
    for prediction in sorted(predictions, key=lambda value: -value["confidence"]):
        best, best_iou = None, IOU_MATCH
        for index, box in enumerate(truth):
            if index in used or box["class_id"] != prediction["class_id"]:
                continue
            overlap = box_iou(box["bbox_xyxy"], prediction["bbox_xyxy"])
            if overlap >= best_iou:
                best, best_iou = index, overlap
        if best is not None:
            used.add(best)
        out.append((prediction, best is not None))
    return out


def average_precision(scored: list[tuple[float, bool]], positives: int) -> float | None:
    """AP com interpolação de todos os pontos (envelope da curva precisão × recall)."""
    if positives == 0:
        return None
    scored = sorted(scored, key=lambda value: -value[0])
    tp = fp = 0
    points = []
    for _, hit in scored:
        tp, fp = tp + hit, fp + (not hit)
        points.append((tp / positives, tp / (tp + fp)))
    ap, previous_recall = 0.0, 0.0
    for index, (recall, _) in enumerate(points):
        precision = max(p for _, p in points[index:])
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return round(ap, 4)


def evaluate(model, config, photos: list[dict], names: list[str], conf: float, ap_conf: float,
             excluded: set[str]) -> dict:
    """Métricas por classe numa lista de fotos; fotos em ``excluded`` (vistas no treino) ficam fora."""
    per_class = {name: {"truth": 0, "tp": 0, "fp": 0, "scored": []} for name in names}
    started, counted = time.perf_counter(), 0
    for photo in photos:
        if photo["sha256"] in excluded:
            continue
        counted += 1
        predictions = detect_boxes(model, photo["image"], config, ap_conf)
        for box in photo["truth"]:
            per_class[names[box["class_id"]]]["truth"] += 1
        # AP: todas as predições >= ap_conf, pareadas uma vez.
        for prediction, hit in _match(predictions, photo["truth"]):
            per_class[names[prediction["class_id"]]]["scored"].append((prediction["confidence"], hit))
        # Ponto de operação das sugestões: só caixas >= conf, pareadas de novo.
        for prediction, hit in _match([p for p in predictions if p["confidence"] >= conf], photo["truth"]):
            per_class[names[prediction["class_id"]]]["tp" if hit else "fp"] += 1
    result = {}
    for name, entry in per_class.items():
        tp, fp, truth = entry["tp"], entry["fp"], entry["truth"]
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / truth if truth else None
        f1 = (2 * precision * recall / (precision + recall) if precision and recall else 0.0) if truth else None
        result[name] = {"truth": truth, "tp": tp, "fp": fp, "fn": truth - tp,
                        "precision": None if precision is None else round(precision, 4),
                        "recall": None if recall is None else round(recall, 4),
                        "f1": None if f1 is None else round(f1, 4),
                        "ap50": average_precision(entry["scored"], truth)}
    totals = {key: sum(value[key] for value in result.values()) for key in ("truth", "tp", "fp", "fn")}
    measured = [value["ap50"] for value in result.values() if value["ap50"] is not None]
    totals["precision"] = round(totals["tp"] / (totals["tp"] + totals["fp"]), 4) if totals["tp"] + totals["fp"] else None
    totals["recall"] = round(totals["tp"] / totals["truth"], 4) if totals["truth"] else None
    totals["map50"] = round(sum(measured) / len(measured), 4) if measured else None
    totals["photos"] = counted
    totals["seconds_per_photo"] = round((time.perf_counter() - started) / counted, 2) if counted else None
    return {"totals": totals, "per_class": result}


def trained_on(model) -> set[str]:
    """SHA-256 das fotos de treino do modelo (dataset_snapshot do run de origem)."""
    snapshot = Path(model.dronecamp_source).resolve().parent.parent.parent / "dataset_snapshot.json"
    try:
        files = json.loads(snapshot.read_text(encoding="utf-8"))["files"]
    except (OSError, ValueError, KeyError):
        return set()
    return {Path(item["path"]).stem for item in files if item["path"].startswith("images/train/")}


# ---------------------------------------------------------------------------
# Gráficos e página.
# ---------------------------------------------------------------------------

COLORS = ["#2a78d6", "#e3893b", "#4c9a5f", "#b5577a", "#7a5fc4", "#8a8f2e"]


def _charts(report: dict, output: Path) -> list[tuple[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [model["label"] for model in report["models"]]
    charts = []
    for set_name in report["sets"]:
        safe = set_name.replace(":", "_")
        # 1. Totais: precisão, recall e mAP50 por modelo.
        fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=130)
        metrics = [("precision", "Precisão"), ("recall", "Recall"), ("map50", "mAP50")]
        width = 0.8 / len(labels)
        for index, model in enumerate(report["models"]):
            totals = model["sets"][set_name]["totals"]
            values = [totals[key] or 0 for key, _ in metrics]
            bars = ax.bar([m + index * width for m in range(len(metrics))], values, width, label=model["label"],
                          color=COLORS[index % len(COLORS)])
            ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], fontsize=7, padding=2)
        ax.set_xticks([m + width * (len(labels) - 1) / 2 for m in range(len(metrics))], [n for _, n in metrics])
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{set_name} — totais (confiança {report['conf']:.2f}; AP desde {report['ap_conf']:.2f})", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        name = f"totais_{safe}.png"
        fig.savefig(output / name)
        plt.close(fig)
        charts.append((name, f"{set_name}: totais"))
        # 2. Por classe: recall e falsos positivos (só classes com gabarito ou com erros).
        classes = [c for c in report["classes"]
                   if any(m["sets"][set_name]["per_class"][c]["truth"] or m["sets"][set_name]["per_class"][c]["fp"]
                          for m in report["models"])]
        if not classes:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(classes) + 1.6), dpi=130, sharey=True)
        height = 0.8 / len(labels)
        for index, model in enumerate(report["models"]):
            per_class = model["sets"][set_name]["per_class"]
            positions = [c + index * height for c in range(len(classes))]
            recall = [per_class[c]["recall"] or 0 for c in classes]
            fps = [per_class[c]["fp"] for c in classes]
            axes[0].barh(positions, recall, height, color=COLORS[index % len(COLORS)], label=model["label"])
            axes[1].barh(positions, fps, height, color=COLORS[index % len(COLORS)])
        truth = [report["models"][0]["sets"][set_name]["per_class"][c]["truth"] for c in classes]
        axes[0].set_yticks([c + height * (len(labels) - 1) / 2 for c in range(len(classes))],
                           [f"{c} (n={n})" for c, n in zip(classes, truth)], fontsize=7)
        axes[0].invert_yaxis()
        axes[0].set_xlim(0, 1)
        axes[0].set_title("Recall por classe (ocorrências reencontradas)", fontsize=9)
        axes[1].set_title("Falsos positivos por classe", fontsize=9)
        axes[0].legend(fontsize=7, frameon=False, loc="lower right")
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="x", labelsize=7)
        fig.suptitle(set_name, fontsize=9)
        fig.tight_layout()
        name = f"classes_{safe}.png"
        fig.savefig(output / name)
        plt.close(fig)
        charts.append((name, f"{set_name}: por classe"))
    return charts


def _page(report: dict, charts: list[tuple[str, str]]) -> str:
    rows = []
    for set_name in report["sets"]:
        for model in report["models"]:
            t = model["sets"][set_name]["totals"]
            fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
            rows.append(f"<tr><td>{html.escape(set_name)}</td><td>{html.escape(model['label'])}</td><td>{t['photos']}</td>"
                        f"<td>{t['truth']}</td><td>{t['tp']}</td><td>{t['fp']}</td><td>{t['fn']}</td>"
                        f"<td>{fmt(t['precision'])}</td><td>{fmt(t['recall'])}</td><td>{fmt(t['map50'])}</td>"
                        f"<td>{t['seconds_per_photo']}</td></tr>")
    images = "".join(f"<figure><img src='{name}' alt='{html.escape(caption)}'><figcaption>{html.escape(caption)}</figcaption></figure>"
                     for name, caption in charts)
    models = "".join(f"<li><b>{html.escape(m['label'])}</b>: {html.escape(m['weights'])} ({html.escape(m['runtime'])})</li>"
                     for m in report["models"])
    return f"""<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Comparação de modelos</title>
<style>body{{font:14px/1.5 system-ui,sans-serif;margin:24px auto;max-width:1100px;padding:0 16px;color:#1d2733;background:#fff}}
table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{border-bottom:1px solid #dde3ea;padding:4px 6px;text-align:right}}
td:first-child,td:nth-child(2),th:first-child,th:nth-child(2){{text-align:left}}figure{{margin:16px 0}}img{{max-width:100%}}
figcaption{{font-size:12px;color:#5b6773}}.note{{background:#f3f6fa;padding:8px 12px;border-radius:6px}}</style>
<h1>Comparação de modelos</h1><p>{html.escape(report['created_at'])} · IoU {IOU_MATCH} · confiança das sugestões {report['conf']}
· fatiamento: {html.escape(json.dumps(report['tiling'], ensure_ascii=False))}</p>
<p class="note">Métricas no mesmo pipeline das sugestões. Fotos vistas no treino de um modelo ficam fora da conta daquele modelo.
Uma única edificação não mede generalização; nenhum modelo é aprovado por este relatório.</p>
<ul>{models}</ul>
<table><tr><th>Conjunto</th><th>Modelo</th><th>Fotos</th><th>Gabarito</th><th>Acertos</th><th>FP</th><th>Omissões</th>
<th>Precisão</th><th>Recall</th><th>mAP50</th><th>s/foto</th></tr>{''.join(rows)}</table>{images}</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, action="append", default=[], help="dataset.yaml de um dataset piloto.")
    parser.add_argument("--review", type=Path, nargs=2, action="append", default=[], metavar=("REGISTRO", "FEEDBACK"))
    parser.add_argument("--weights", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--onnx-provider", choices=["directml", "cpu"], default="directml")
    parser.add_argument("--no-tiling", action="store_true")
    parser.add_argument("--ap-conf", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Saída já existe; relatórios não são sobrescritos.")
    config = load_config()
    names = detection_names(load_taxonomy(config.taxonomy_path))
    conf = float(config.pilot.get("suggestion_conf", 0.15))
    tiling = None if args.no_tiling else config.pilot.get("tiling")
    if args.no_tiling:
        config = replace(config, pilot={**config.pilot, "tiling": {"enabled": False}})
    sets: dict[str, list[dict]] = {}
    for data in args.data:
        sets.update(pilot_sets(data))
    for registry, feedback in args.review:
        sets.update(review_set(registry, feedback))
    report = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "conf": conf,
              "ap_conf": args.ap_conf, "iou_match": IOU_MATCH, "tiling": tiling, "classes": names,
              "sets": {name: len(photos) for name, photos in sets.items()}, "models": []}
    for index, weights in enumerate(args.weights):
        model = load_inference_model(config, str(weights), args.onnx_provider)
        check_domain_names(model, names)
        model_config = pilot_inference_config(config, model)
        seen = trained_on(model)
        label = args.label[index] if index < len(args.label) else Path(model.dronecamp_source).parts[-4]
        entry = {"label": label, "weights": str(weights), "weights_sha256": file_hash(weights),
                 "source_checkpoint": str(model.dronecamp_source), "runtime": model.dronecamp_runtime, "sets": {}}
        for set_name, photos in sets.items():
            entry["sets"][set_name] = evaluate(model, model_config, photos, names, conf, args.ap_conf, seen)
            t = entry["sets"][set_name]["totals"]
            print(f"{label:24s} {set_name:36s} fotos {t['photos']:3d} | acertos {t['tp']:3d}/{t['truth']:3d}"
                  f" | FP {t['fp']:4d} | P {t['precision']} R {t['recall']} mAP50 {t['map50']} | {t['seconds_per_photo']} s/foto")
        report["models"].append(entry)
    args.output.mkdir(parents=True)
    write_json(args.output / "report.json", report)
    charts = _charts(report, args.output)
    (args.output / "index.html").write_text(_page(report, charts), encoding="utf-8")
    print(f"Relatório: {args.output / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
