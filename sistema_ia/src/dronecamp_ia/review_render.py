"""Página local de conferência, evidências e rascunho de relatório rastreável."""

from collections import Counter
from pathlib import Path
import csv
import json
import shutil

from PIL import Image, ImageDraw, ImageFont

from .config import ProjectConfig, detection_names, load_taxonomy
from .io import file_hash, resolve_local_path, write_json
from .review_data import boxes_to_yolo, review_image_size, validate_boxes, write_review_summary

COLORS = ["#c73838", "#a23490", "#b36c00", "#265bc4", "#7646bb", "#c34c18", "#087d78", "#397821", "#5b4b99", "#166d91", "#8b5018", "#405ac0", "#895394"]


def _font(size: int):
    for path in (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/segoeui.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _load_suggestions(directory: Path, registry_path: Path, class_count: int) -> dict | None:
    """Sugestões só aparecem se pertencem a este registro; nunca viram caixas sozinhas."""
    path = directory / "suggestions.json"
    if not path.is_file():
        return None
    suggestions = json.loads(path.read_text(encoding="utf-8"))
    if suggestions.get("schema_version") != 1 or suggestions.get("registry_sha256") != file_hash(registry_path):
        raise ValueError("suggestions.json pertence a outra versão do registro; gere as sugestões novamente.")
    for values in suggestions["images"].values():
        for value in values:
            if type(value.get("class_id")) is not int or not 0 <= value["class_id"] < class_count:
                raise ValueError("Sugestão com classe inválida.")
            if not 0 <= float(value.get("confidence", -1)) <= 1:
                raise ValueError("Sugestão com confiança inválida.")
    return suggestions


def render_review_package(config: ProjectConfig, registry_path: Path) -> Path:
    """Copie evidências para visualização; fotos e registros anteriores permanecem."""
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("taxonomy_sha256") != file_hash(config.taxonomy_path):
        raise ValueError("Taxonomia mudou: migre a revisão antes de gerar as evidências.")
    directory = registry_path.parent
    images_dir = directory / "images"
    overlays_dir = directory / "evidence"
    # Cada registro tem sua pasta: rótulos de uma versão anterior não entram na atual.
    labels_dir = directory / "labels_proposed" / file_hash(registry_path)
    for target in (images_dir, overlays_dir, labels_dir):
        target.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy(config.taxonomy_path)
    names = detection_names(taxonomy)
    digests = [item["image_sha256"] for item in registry["images"]]
    if len(set(digests)) != len(digests):
        raise ValueError("Imagem duplicada no registro de revisão.")
    labels = {item["id"]: item["label"] for item in taxonomy["classes"] if item["phase"] == 1}
    suggestions = _load_suggestions(directory, registry_path, len(names))
    browser_images, rows = [], []
    for item in registry["images"]:
        digest = item["image_sha256"]
        source = resolve_local_path(item["source_path"], config.root)
        if file_hash(source) != digest:
            raise ValueError(f"A foto original mudou: {item['filename']}.")
        if review_image_size(source) != (item["width"], item["height"]):
            raise ValueError("Dimensões do registro divergem da foto revisada.")
        validate_boxes(item["boxes"], item["width"], item["height"], len(names))
        copied = images_dir / f"{digest}{source.suffix.lower()}"
        shutil.copy2(source, copied)
        with Image.open(source) as original:
            evidence = original.convert("RGB")
        draw = ImageDraw.Draw(evidence)
        font = _font(max(14, int(item["width"] / 75)))
        for number, box in enumerate(item["boxes"], start=1):
            coordinates = box["bbox_xyxy"]
            color = COLORS[box["class_id"]]
            draw.rectangle(coordinates, outline=color, width=3)
            text = f"{number}: {labels[box['class_id']]}"
            x, y = coordinates[:2]
            bounds = draw.textbbox((x, max(0, y - 22)), text, font=font)
            draw.rectangle(bounds, fill=color)
            draw.text((x, max(0, y - 22)), text, fill="white", font=font)
        evidence.save(overlays_dir / f"{digest}.jpg", quality=92)
        # Estes .txt são propostas, deliberadamente separados de data/dataset.
        if item["status"] in {"positive", "negative"}:
            (labels_dir / f"{digest}.txt").write_text(boxes_to_yolo(item["boxes"], item["width"], item["height"]), encoding="utf-8")
        browser_images.append({**item, "source_path": None, "image_url": copied.relative_to(directory).as_posix(),
                               "evidence_url": f"evidence/{digest}.jpg",
                               "model_suggestions": suggestions["images"].get(digest, []) if suggestions else []})
        rows.append([digest, item["filename"], ";".join(map(str,item["pages"])), item["status"], len(item["boxes"]),
                     item["visual_review_status"], item["technical_status"], item["notes"], item.get("second_review_notes", "")])
    with (directory / "review_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_sha256", "filename", "pages", "triage_status", "proposed_boxes", "visual_review_status", "technical_status", "notes", "second_review_notes"])
        writer.writerows(rows)
    # O aviso inclui expansões ainda não revistas de todas as migrações anteriores.
    pending_class_ids = set()
    for item in registry["images"]:
        if item.get("taxonomy_recheck_pending"):
            pending_class_ids.update(set(labels) - set(item.get("classes_reviewed_before_migration", [])))
    payload = {"registry_sha256": file_hash(registry_path), "version": registry["version"], "images": browser_images,
               "taxonomy_recheck_pending": any(item.get("taxonomy_recheck_pending") for item in registry["images"]),
               "classes_requiring_review": [labels[key] for key in sorted(pending_class_ids)],
               "classes": [{"id": key, "label": value, "color": COLORS[key]} for key,value in labels.items()],
               "duplicate_iou": config.pilot.get("duplicate_iou"),
               "suggestions": {key: suggestions[key] for key in ("model_label", "conf", "warning", "weights_sha256", "pilot")} if suggestions else None}
    template = (Path(__file__).parent / "review_templates/index.html").read_text(encoding="utf-8")
    # JSON embutido evita fetch de arquivos locais e fecha a possibilidade de </script> nos textos.
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    (directory / "index.html").write_text(template.replace("__REVIEW_DATA__", serialized), encoding="utf-8")
    write_json(directory / "browser_data.json", payload)
    summary = write_review_summary(directory, registry)
    report = ["# Revisão visual do CEASA — rascunho para conferência", "",
              f"Versão: {registry['version']}. {summary['unique_images']} imagens únicas, {summary['total_proposed_boxes']} caixas propostas.",
              f"Registro de origem: `{registry_path.name}`; SHA-256: `{file_hash(registry_path)}`.",
              f"Triagem: {summary['image_status_counts']}. Aprovações humanas registradas: {summary['human_approved_images']}.",
              f"Correções humanas de taxonomia anterior preservadas: {summary['historical_human_review_images']} fotos; não equivalem a aprovação das classes atuais.",
              "", "Este registro contém propostas visuais. Não é laudo técnico e não representa aprovação humana ou qualidade de um modelo especializado.",
              "", "| Foto original | Páginas | Triagem | Ocorrências propostas | Evidência demarcada |", "| --- | --- | --- | --- | --- |"]
    for item in registry["images"]:
        counter = Counter(labels[box["class_id"]] for box in item["boxes"])
        occurrences = "; ".join(f"{name}: {count}" for name,count in counter.items()) or "Sem caixa proposta"
        digest = item["image_sha256"]
        suffix = Path(item["source_path"]).suffix.lower()
        photo_link = f"[{item['filename']}](images/{digest}{suffix})"
        report.append(f"| {photo_link} | {', '.join(map(str,item['pages']))} | {item['status']} | {occurrences} | [Ver caixas](evidence/{digest}.jpg) |")
    report.extend(["", "Contagens referem-se a caixas por foto, não a defeitos físicos únicos. Cenas relacionadas e recortes podem repetir ocorrências.",
                   "", "Aprovação técnica, gravidade, prioridade, causa, metros lineares, áreas e diagnóstico hidráulico não foram inferidos."])
    (directory / "review_report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    return directory / "index.html"
