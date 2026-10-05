"""Pareamento entre caixas do modelo e caixas humanas, e contagem por classe.

Função no projeto: é a régua comum dos módulos de aprendizado. Diz, para cada
sugestão do modelo, se ela reencontra uma caixa aprovada por um humano (mesma
classe e IoU >= 0,5) e conta acertos (TP), falsos positivos (FP) e omissões
(FN) por classe.

Quem usa:
- ``calibration.py``: rótulo 1/0 de cada sugestão para treinar o calibrador.
- ``active_learning.py``: recompensa do bandit (quanto o modelo errou na foto).
- ``model_gate.py``: precisão, recall, FP e omissões por classe na comparação.

Quando mexer: o limiar de IoU (``IOU_MATCH``) ou a regra de pareamento.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .pilot import box_iou

# Mesma régua do scripts/compare_pilot_models.py e do mAP50.
IOU_MATCH = 0.5


def yolo_label_boxes(label_path: Path, width: int, height: int) -> list[dict]:
    """Converte uma label YOLO (cx cy w h normalizados) em caixas [x1, y1, x2, y2] em pixels."""
    boxes = []
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
        boxes.append({"class_id": int(class_id), "bbox_xyxy": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]})
    return boxes


def match_predictions(predictions: list[dict], truth: list[dict], iou: float = IOU_MATCH) -> tuple[list[bool], set[int]]:
    """Pareamento guloso por confiança: cada caixa humana é reencontrada no máximo uma vez.

    Devolve ``(acertos, usadas)``: ``acertos[i]`` diz se ``predictions[i]`` (na ordem
    recebida) é um TP; ``usadas`` são os índices das caixas humanas reencontradas.
    """
    hits = [False] * len(predictions)
    used: set[int] = set()
    order = sorted(range(len(predictions)), key=lambda index: -float(predictions[index].get("confidence", 0.0)))
    for index in order:
        prediction = predictions[index]
        best, best_iou = None, iou
        for position, box in enumerate(truth):
            if position in used or box["class_id"] != prediction["class_id"]:
                continue
            overlap = box_iou(box["bbox_xyxy"], prediction["bbox_xyxy"])
            if overlap >= best_iou:
                best, best_iou = position, overlap
        if best is not None:
            used.add(best)
            hits[index] = True
    return hits, used


def image_errors(predictions: list[dict], truth: list[dict]) -> dict:
    """TP, FP e FN de uma foto, no total e por classe."""
    hits, used = match_predictions(predictions, truth)
    tp, fp, fn = Counter(), Counter(), Counter()
    for prediction, hit in zip(predictions, hits):
        (tp if hit else fp)[prediction["class_id"]] += 1
    for position, box in enumerate(truth):
        if position not in used:
            fn[box["class_id"]] += 1
    return {"tp": sum(tp.values()), "fp": sum(fp.values()), "fn": sum(fn.values()),
            "by_class": {"tp": tp, "fp": fp, "fn": fn}}


def per_class_report(errors: list[dict], names: list[str]) -> dict:
    """Soma os erros de várias fotos e calcula precisão e recall por classe.

    Classe sem nenhuma caixa humana nem sugestão fica com ``None``: não medida,
    o que é diferente de zero.
    """
    totals = {key: Counter() for key in ("tp", "fp", "fn")}
    for item in errors:
        for key in totals:
            totals[key].update(item["by_class"][key])
    report = {}
    for class_id, name in enumerate(names):
        tp, fp, fn = (totals[key][class_id] for key in ("tp", "fp", "fn"))
        report[name] = {
            "tp": tp, "fp": fp, "fn": fn, "human_boxes": tp + fn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        }
    tp, fp, fn = (sum(totals[key].values()) for key in ("tp", "fp", "fn"))
    report["_total"] = {"tp": tp, "fp": fp, "fn": fn, "human_boxes": tp + fn,
                        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                        "recall": round(tp / (tp + fn), 4) if tp + fn else None}
    return report
