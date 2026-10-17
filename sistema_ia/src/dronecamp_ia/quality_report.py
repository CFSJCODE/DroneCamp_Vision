"""Avaliação detalhada de pesos prontos: qualidade por classe, operação, localização, condições e erros visuais.

Função no projeto: é a ENTRADA do comando ``evaluate-detailed``. Roda um ou mais
pesos (``.pt`` ou ``.onnx``) nas fotos ORIGINAIS de um dataset piloto, com o
mesmo fatiamento das sugestões (``suggestions.detect_boxes``, que junta as
janelas sobrepostas numa lista por foto), e grava para cada peso o relatório de
``detection_metrics.evaluate_detections`` mais:

- protocolo completo: dataset e hash do ``pilot.json``, taxonomia, divisões,
  confiança, IoU, limite de detecções, regime de inferência, unidade (foto
  original) e fotos excluídas por terem sido vistas no treino;
- identidade do checkpoint: hash, época e motivo de ser o "melhor"
  (``training_progress.checkpoint_identity``);
- condições de captura: edificação e cena (do ``pilot.json``), período (mês da
  data EXIF) e, com ``--conditions``, material, ângulo, iluminação ou qualquer
  coluna de um CSV ``image_sha256,campo1,campo2...``;
- recortes dos erros em ``errors/<rótulo>/<classe>/``: caixa humana em verde,
  previsão em vermelho, nome com tipo, confiança e IoU; cada item do JSON aponta
  para o recorte, a foto, a classe, a confiança e o checkpoint.

Nada aqui adota modelo, altera revisão ou preenche laudo: a adoção continua sendo
``compare-models`` e a interpretação dos erros é humana.

Quem chama: ``cli.py`` (``evaluate-detailed``).

O que lê e grava no disco:
- Lê: ``pilot.json`` e fotos/labels do dataset piloto, os pesos, o
  ``dataset_snapshot.json`` dos treinos (fotos a excluir), ``results.csv`` ao lado
  dos pesos e o CSV de condições.
- Grava: ``runs/quality_<data>_<id>/quality.json``, ``report.md`` e ``errors/``;
  e, quando o treino de origem tem ficha (treinos novos), ``ficha/evaluation_*.json``
  apontando para esta avaliação.

Onde mexer:
- Novo campo de condição automático: ``_conditions``.
- Tamanho/cores dos recortes: ``_save_error_crops``.
- Coluna nova na tabela: ``_markdown``.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from .detection_metrics import DEFAULT_EXAMPLES, DEFAULT_FP_BUDGETS, DEFAULT_SIZE_EDGES, evaluate_detections
from .io import file_hash, make_run_directory, write_json

# Campos de condição sempre tentados (do pilot.json e do EXIF).
AUTO_CONDITIONS = ("edificacao", "cena", "periodo")
# Margem (px) em volta das caixas no recorte de erro e lado máximo do recorte salvo.
CROP_MARGIN, CROP_MAX_SIDE = 96, 768


def _exif_period(path: Path) -> str | None:
    """Mês da captura (``AAAA-MM``) pela data EXIF da foto; ``None`` sem EXIF."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            exif = image.getexif()
            value = exif.get_ifd(0x8769).get(36867) or exif.get(306)  # DateTimeOriginal ou DateTime
        return str(value)[:7].replace(":", "-") if value else None
    except Exception:  # noqa: BLE001 - foto sem EXIF ou formato sem suporte
        return None


def load_conditions_csv(path: Path | None) -> tuple[dict, list[str]]:
    """Lê o CSV de condições: ``{image_sha256: {campo: valor}}`` e a lista de campos.

    A primeira coluna precisa se chamar ``image_sha256`` (o mesmo hash do ``pilot.json``).
    """
    if not path:
        return {}, []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or reader.fieldnames[0] != "image_sha256":
            raise ValueError("O CSV de condições precisa começar pela coluna image_sha256.")
        fields = [name for name in reader.fieldnames[1:] if name]
        rows = {row["image_sha256"]: {name: (row.get(name) or "").strip() for name in fields} for row in reader}
    return rows, fields


def _conditions(sample: dict, image: Path, extra: dict) -> dict:
    """Condições de uma foto: edificação, cena, período EXIF e as colunas do CSV."""
    return {"edificacao": sample.get("building_group"), "cena": sample.get("scene_group"),
            "periodo": _exif_period(image), **extra.get(sample.get("image_sha256"), {})}


def _save_error_crops(errors: list[dict], root: Path, output: Path, label: str) -> None:
    """Grava um recorte JPEG por erro e acrescenta ``crop`` ao item (caminho relativo à saída)."""
    from PIL import Image, ImageDraw

    counters: dict[tuple, int] = {}
    for item in errors:
        boxes = [box for box in (item.get("truth_bbox_xyxy"), item.get("bbox_xyxy")) if box]
        if not boxes:
            continue
        try:
            with Image.open(root / item["image"]) as image:
                image = image.convert("RGB")
                x1 = max(0, int(min(box[0] for box in boxes)) - CROP_MARGIN)
                y1 = max(0, int(min(box[1] for box in boxes)) - CROP_MARGIN)
                x2 = min(image.width, int(max(box[2] for box in boxes)) + CROP_MARGIN)
                y2 = min(image.height, int(max(box[3] for box in boxes)) + CROP_MARGIN)
                crop = image.crop((x1, y1, x2, y2))
                draw = ImageDraw.Draw(crop)
                width = max(2, round(max(crop.size) / 250))
                if item.get("truth_bbox_xyxy"):
                    bx = item["truth_bbox_xyxy"]
                    draw.rectangle([bx[0] - x1, bx[1] - y1, bx[2] - x1, bx[3] - y1], outline=(0, 200, 0), width=width)
                if item.get("bbox_xyxy"):
                    bx = item["bbox_xyxy"]
                    draw.rectangle([bx[0] - x1, bx[1] - y1, bx[2] - x1, bx[3] - y1], outline=(230, 0, 0), width=width)
                crop.thumbnail((CROP_MAX_SIDE, CROP_MAX_SIDE))
                kind = item.get("fp_type") or item.get("fn_reason") or item["type"]
                key = (item["class"], item["type"])
                counters[key] = counters.get(key, 0) + 1
                folder = output / "errors" / label / item["class"]
                folder.mkdir(parents=True, exist_ok=True)
                confidence = f"{item['confidence']:.2f}" if item.get("confidence") is not None else "na"
                name = f"{item['type']}_{counters[key]:02d}_{kind}_conf{confidence}_iou{item.get('iou') or 0:.2f}.jpg"
                crop.save(folder / name, quality=88)
                item["crop"] = (folder / name).relative_to(output).as_posix()
        except OSError as error:
            item["crop_error"] = f"{type(error).__name__}: {error}"


def evaluate_weights(config, data_path: Path, weights: list[str], labels: list[str] | None = None,
                     splits: tuple[str, ...] = ("test",), conf: float | None = None, onnx_provider: str = "directml",
                     regime: str = "janelas", size_edges: tuple[float, ...] = DEFAULT_SIZE_EDGES,
                     fp_budgets: tuple[float, ...] = DEFAULT_FP_BUDGETS, conditions_csv: Path | None = None,
                     examples: int = DEFAULT_EXAMPLES, crops: bool = True, output: Path | None = None,
                     predict=None) -> Path:
    """ENTRADA de ``evaluate-detailed``: avalia cada peso nas MESMAS fotos e grava o relatório.

    Parâmetros:
        config: ``ProjectConfig``.
        data_path: ``dataset.yaml`` do dataset piloto (``pilot.json`` ao lado).
        weights: pesos ``.pt``/``.onnx``; ``labels``: rótulos (padrão: nome da pasta do run).
        splits: divisões avaliadas (padrão ``test``; ``val`` para acompanhar treino).
        conf: confiança operacional (padrão: a das sugestões, ``pilot.suggestion_conf``).
        onnx_provider: ``directml`` ou ``cpu`` para ``.onnx``.
        regime: ``janelas`` (como as sugestões) ou ``inteira``.
        size_edges, fp_budgets, examples: ver ``detection_metrics``.
        conditions_csv: CSV ``image_sha256,...`` com material, ângulo, iluminação etc.
        crops: grava os recortes dos erros.
        output: pasta de saída (padrão ``runs/quality_<data>_<id>``).
        predict: função ``(model, model_config, image) -> previsões`` (testes); padrão
            ``suggestions.detect_boxes`` com confiança ``AP_CONF``.

    Erros: ValueError sem fotos fora do treino de todos os pesos.
    """
    from .config import detection_names, load_taxonomy
    from .model_gate import AP_CONF, REGIMES, _load_models
    from .matching import yolo_label_boxes
    from .review_data import review_image_size
    from .run_record import record_evaluation
    from .training_progress import checkpoint_identity

    if regime not in REGIMES:
        raise ValueError(f"regime deve ser um de {REGIMES}.")
    # 1. Taxonomia, manifesto, confiança e modelos (fotos vistas no treino de qualquer peso saem).
    taxonomy_path = config.taxonomy_path
    names = detection_names(load_taxonomy(taxonomy_path))
    root = Path(data_path).resolve().parent
    manifest = json.loads((root / "pilot.json").read_text(encoding="utf-8"))
    conf = float(conf if conf is not None else config.pilot.get("suggestion_conf", config.prediction["conf"]))
    loaded, excluded = _load_models(config, list(weights), names, onnx_provider)
    candidates = [sample for sample in manifest["images"] if sample["split"] in splits]
    samples = [sample for sample in candidates if sample["image_sha256"] not in excluded]
    excluded_here = len(candidates) - len(samples)
    if not samples:
        raise ValueError("Nenhuma foto das divisões escolhidas ficou fora do treino de todos os pesos.")
    extra, csv_fields = load_conditions_csv(conditions_csv)
    fields = tuple(dict.fromkeys((*AUTO_CONDITIONS, *csv_fields)))
    if predict is None:
        from .suggestions import detect_boxes

        tiling = None if regime == "janelas" else {"enabled": False}

        def predict(model, model_config, image):
            return detect_boxes(model, image, model_config, AP_CONF, tiling)

    folder = Path(output) if output else make_run_directory(config.root, "quality")
    folder.mkdir(parents=True, exist_ok=True)
    labels = list(labels or [])
    report = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "models": {}}
    for position, entry in enumerate(loaded):
        label = labels[position] if position < len(labels) else (
            entry["source"].parent.parent.parent.name if entry["source"].parent.name == "weights" else entry["source"].stem)
        if label in report["models"]:
            label = f"{label}_{position}"
        checkpoint = checkpoint_identity(entry["source"], entry["model"])
        # 2. Previsões consolidadas por foto original + caixas humanas + condições.
        rows = []
        for sample in samples:
            image = root / sample["image"]
            width, height = review_image_size(image)
            rows.append({"image": sample["image"], "image_sha256": sample["image_sha256"],
                         "truth": yolo_label_boxes(root / sample["label"], width, height),
                         "predictions": predict(entry["model"], entry["config"], image),
                         "conditions": _conditions(sample, image, extra)})
        protocol = {
            "dataset": str(Path(data_path).resolve()), "pilot_sha256": file_hash(root / "pilot.json"),
            "taxonomy_sha256": file_hash(taxonomy_path), "splits": list(splits), "regime": regime,
            "tiling": config.pilot.get("tiling") if regime == "janelas" else None,
            "max_det": config.prediction.get("max_det"), "imgsz": entry["config"].prediction["imgsz"],
            "consolidation": "detect_boxes: janelas sobrepostas unidas por NMS por classe (merge_iou) na foto original",
            "images_excluded_seen_in_training": excluded_here, "runtime": entry["model"].dronecamp_runtime,
            "weights_executed": str(entry["used"]), "checkpoint": checkpoint}
        result = evaluate_detections(rows, names, conf, AP_CONF, size_edges, fp_budgets, fields, examples, protocol)
        for item in result["errors"]:
            item["checkpoint_sha256"] = checkpoint["sha256"]
            item["checkpoint_epoch"] = checkpoint["epoch"]
        # 3. Recortes dos erros (verde = humano, vermelho = modelo).
        if crops:
            _save_error_crops(result["errors"], root, folder, label)
        report["models"][label] = result
        # 4. Vínculo na ficha do treino de origem (só treinos novos, que já têm ficha).
        if checkpoint.get("run"):
            record_evaluation(Path(checkpoint["run"]), {
                "evaluation_folder": str(folder), "label": label, "checkpoint": checkpoint,
                "protocol": {key: value for key, value in result["protocol"].items() if key != "checkpoint"},
                "total": result["total"]})
    report["images"] = [{"image": sample["image"], "image_sha256": sample["image_sha256"], "split": sample["split"]}
                        for sample in samples]
    report["note"] = ("Mesmas fotos e mesma régua para todos os pesos; nenhum modelo é adotado aqui. "
                      "Classe sem caixa humana = não avaliada (suporte 0).")
    write_json(folder / "quality.json", report)
    (folder / "report.md").write_text(_markdown(report), encoding="utf-8")
    return folder


def _fmt(value, digits: int = 3) -> str:
    """Número para tabela ("—" quando ausente)."""
    if value is None:
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _markdown(report: dict) -> str:
    """Relatório em Markdown: protocolo, tabela por classe, operação, tamanho, condições e erros."""
    lines = ["# Avaliação detalhada por classe", "", report["note"], ""]
    for label, result in report["models"].items():
        protocol, total = result["protocol"], result["total"]
        checkpoint = protocol.get("checkpoint", {})
        lines += [f"## {label}", "",
                  f"- Checkpoint: `{checkpoint.get('path')}` (sha256 {str(checkpoint.get('sha256'))[:12]}, "
                  f"época {_fmt(checkpoint.get('epoch'))}). {checkpoint.get('reason')}.",
                  f"- Protocolo: divisões {protocol['splits']}, regime {protocol['regime']}, unidade {protocol['unit']}, "
                  f"confiança {protocol['conf']}, IoU {protocol['iou_match']}, AP com confiança ≥ {protocol['ap_conf']}, "
                  f"limite {protocol.get('max_det')} detecções; fotos {total['images']} (negativas {total['negative_images']}), "
                  f"caixas humanas {total['support_boxes']}; excluídas por terem sido vistas no treino: "
                  f"{protocol['images_excluded_seen_in_training']}.",
                  f"- Total: P {_fmt(total['precision'])}, R {_fmt(total['recall'])}, F1 {_fmt(total['f1'])}, "
                  f"mAP50 {_fmt(total['map50'])}, mAP75 {_fmt(total['map75'])}, mAP50-95 {_fmt(total['map50_95'])} "
                  f"({total['classes_evaluated']} classes avaliadas); FP por foto {_fmt(total['fp_per_image'])}.",
                  f"- Não avaliadas (sem caixa humana nestas fotos): {', '.join(total['classes_not_evaluated']) or 'nenhuma'}.",
                  "", "| Classe | Status | Fotos | Caixas | TP | FP | FN | P | R | F1 | AP50 | AP75 | AP50-95 | IoU mediana | FP fundo/classe/imprecisa/dup |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, row in result["per_class"].items():
            types = row["fp_types"]
            lines.append(f"| {name} | {row['status']} | {row['support_images']} | {row['support_boxes']} | {row['tp']} | "
                         f"{row['fp']} | {row['fn']} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | {_fmt(row['f1'])} | "
                         f"{_fmt(row['ap50'])} | {_fmt(row['ap75'])} | {_fmt(row['ap50_95'])} | "
                         f"{_fmt(row['iou_matched'].get('median'))} | {types['fundo']}/{types['classe_trocada']}/"
                         f"{types['caixa_imprecisa']}/{types['duplicata']} |")
        operating = result["operating"]["_total"]
        best = operating.get("best_f1") or {}
        lines += ["", f"- Operação: melhor F1 {_fmt(best.get('f1'))} na confiança {_fmt(best.get('conf'), 2)} "
                  f"(FP por foto {_fmt(best.get('fp_per_image'))}); recall com até N falsos alarmes por foto: "
                  + ", ".join(f"{budget}: {_fmt(value['recall'])}" for budget, value in operating["recall_at_fp_per_image"].items())
                  + ".", "", "| Classe | Faixa | Caixas | TP | FN | Recall | Previsões | Precisão |", "|---|---|---|---|---|---|---|---|"]
        for name, row in result["per_class"].items():
            if row["status"] != "avaliada":
                continue
            for bucket, values in row["by_size"].items():
                lines.append(f"| {name} | {bucket} | {values['support_boxes']} | {values['tp']} | {values['fn']} | "
                             f"{_fmt(values['recall'])} | {values['predictions']} | {_fmt(values['precision'])} |")
        for field, block in result["conditions"].items():
            lines += ["", f"Condição **{field}** (fotos sem etiqueta: {_fmt(block['unknown_share'])}):", "",
                      "| Valor | Fotos | Caixas | P | R | F1 | mAP50 | FP/foto |", "|---|---|---|---|---|---|---|---|"]
            for value, row in block["values"].items():
                lines.append(f"| {value} | {row['images']} | {row['support_boxes']} | {_fmt(row['precision'])} | "
                             f"{_fmt(row['recall'])} | {_fmt(row['f1'])} | {_fmt(row['map50'])} | {_fmt(row['fp_per_image'])} |")
        crops = [item for item in result["errors"] if item.get("crop")]
        lines += ["", f"Exemplos de erro gravados: {len(crops)} em `errors/{label}/` (verde = caixa humana, "
                  "vermelho = previsão).", ""]
    return "\n".join(lines)
