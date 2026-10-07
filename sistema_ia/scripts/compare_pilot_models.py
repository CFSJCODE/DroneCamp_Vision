"""Compara pesos piloto nas mesmas fotos de um dataset piloto, com inferência real.

Para cada split: caixas humanas reencontradas (mesma classe, IoU >= 0,5) e
sugestões corretas na confiança das sugestões. Fotos que estiveram no treino de
um modelo são contadas à parte, para não confundir memorização com melhora.

Uso: python scripts/compare_pilot_models.py --data data/pilot/V/dataset.yaml \
        --weights A.pt --weights B.pt --output runs/compare_X.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronecamp_ia.backend import check_domain_names, load_inference_model  # noqa: E402
from dronecamp_ia.config import detection_names, load_config, load_taxonomy  # noqa: E402
from dronecamp_ia.io import file_hash, write_json  # noqa: E402
from dronecamp_ia.pilot import box_iou  # noqa: E402
from dronecamp_ia.review_data import review_image_size  # noqa: E402
from dronecamp_ia.suggestions import detect_boxes  # noqa: E402
from dronecamp_ia.training import pilot_inference_config  # noqa: E402

IOU_MATCH = 0.5


def ground_truth(root: Path, sample: dict) -> list[dict]:
    width, height = review_image_size(root / sample["image"])
    boxes = []
    for line in (root / sample["label"]).read_text(encoding="utf-8").splitlines():
        if line.strip():
            class_id, cx, cy, w, h = line.split()
            cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
            boxes.append({"class_id": int(class_id), "bbox_xyxy": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]})
    return boxes


def match(predictions: list[dict], truth: list[dict]) -> int:
    """Pareamento guloso por confiança: cada caixa humana conta uma vez."""
    used, hits = set(), 0
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
            hits += 1
    return hits


def trained_images(weights: Path) -> set[str]:
    snapshot = weights.resolve().parent.parent.parent / "dataset_snapshot.json"
    files = json.loads(snapshot.read_text(encoding="utf-8"))["files"]
    return {Path(item["path"]).stem for item in files if item["path"].startswith("images/train/")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, action="append", required=True)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--no-tiling", action="store_true",
                        help="Foto inteira reduzida ao imgsz (como antes); padrão: pilot.tiling para fotos grandes.")
    parser.add_argument("--onnx-provider", choices=["directml", "cpu"], default="directml",
                        help="Para --weights .onnx (comando export): directml usa a GPU.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config()
    names = detection_names(load_taxonomy(config.taxonomy_path))
    root = args.data.resolve().parent
    manifest = json.loads((root / "pilot.json").read_text(encoding="utf-8"))
    conf = float(args.conf if args.conf is not None else config.pilot.get("suggestion_conf", 0.15))
    report = {"dataset": str(args.data), "pilot_sha256": file_hash(root / "pilot.json"), "conf": conf,
              "iou_match": IOU_MATCH, "tiling": None if args.no_tiling else config.pilot.get("tiling"), "models": []}
    for weights in args.weights:
        # .onnx roda na GPU (DirectML); fotos de treino vêm do .pt de origem (export.json).
        model = load_inference_model(config, str(weights), args.onnx_provider)
        check_domain_names(model, names)
        model_config = pilot_inference_config(config, model)
        seen_in_training = trained_images(model.dronecamp_source)
        totals = defaultdict(lambda: {"images": 0, "truth": 0, "found": 0, "suggestions": 0, "correct": 0})
        for sample in manifest["images"]:
            truth = ground_truth(root, sample)
            predictions = detect_boxes(model, root / sample["image"], model_config, conf,
                                       {"enabled": False} if args.no_tiling else None)
            hits = match(predictions, truth)
            group = sample["split"] + (" (visto no treino)" if sample["image_sha256"] in seen_in_training else "")
            entry = totals[group]
            entry["images"] += 1
            entry["truth"] += len(truth)
            entry["found"] += hits
            entry["suggestions"] += len(predictions)
            entry["correct"] += hits
        report["models"].append({"weights": str(weights), "weights_sha256": file_hash(weights),
                                 "runtime": model.dronecamp_runtime, "source_checkpoint": str(model.dronecamp_source),
                                 "imgsz": model_config.prediction["imgsz"], "splits": dict(sorted(totals.items()))})
        print(f"\n{weights}")
        for group, entry in sorted(totals.items()):
            print(f"  {group:28s} fotos {entry['images']:2d} | humanas reencontradas {entry['found']:3d}/{entry['truth']:3d}"
                  f" | sugestões corretas {entry['correct']:3d}/{entry['suggestions']:3d}")
    write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
