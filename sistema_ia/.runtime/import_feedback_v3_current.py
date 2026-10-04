"""Preservar correções humanas da v3 e preparar conferência das classes atuais."""

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import shutil

from dronecamp_ia.config import load_config, load_taxonomy, detection_names
from dronecamp_ia.io import file_hash, write_json
from dronecamp_ia.review_data import (
    boxes_to_yolo, import_human_feedback, migrate_review_taxonomy,
    review_image_size, validate_boxes,
)


def preserve_inputs(root: Path, feedback: Path) -> Path:
    """Registrar hashes antes de qualquer escrita em versões novas."""
    digest = file_hash(feedback)
    manifest = root / ".runtime" / f"feedback_{digest[:12]}_before.json"
    if manifest.exists():
        raise ValueError("Esta importação já foi iniciada; preservar as versões existentes.")
    paths = [feedback, root / "models/yolo26l.pt", root / "configs/taxonomy.json"]
    paths.extend(root.glob("configs/taxonomy_ceasa_*.json"))
    paths.extend(path for path in (root / "data/reviews").rglob("*") if path.is_file())
    paths.extend(path for path in (root / "data/reference/ceasa").rglob("*") if path.is_file())
    write_json(manifest, {"files": {str(path.resolve()): file_hash(path) for path in paths}})
    return manifest


def export_historical_annotations(root: Path, registry_path: Path, taxonomy_path: Path, digest: str) -> Path:
    """Arquivar somente labels humanos válidos da taxonomia de origem.

    Este acervo não tem YAML de treino ou divisões artificiais. É uma referência
    de dez classes: não comprova conferência das treze classes atuais.
    """
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    names = detection_names(load_taxonomy(taxonomy_path))
    directory = root / "data/annotation_corpus" / f"ceasa_v3_{digest[:12]}"
    if directory.exists():
        raise ValueError("Acervo já existe; não sobrescrever anotações históricas.")
    records = []
    counts = Counter()
    for item in registry["images"]:
        if item.get("human_approved") is not True:
            continue
        if item.get("technical_status") != "human_visual_reviewed" or not item.get("human_reviewer"):
            raise ValueError("A aprovação histórica precisa de estado e autor.")
        source = Path(item["source_path"])
        if file_hash(source) != item["image_sha256"] or review_image_size(source) != (item["width"], item["height"]):
            raise ValueError("Original alterado antes de arquivar a correção humana.")
        validate_boxes(item["boxes"], item["width"], item["height"], len(names))
        records.append(item)
        counts.update(box["class_id"] for box in item["boxes"])
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "images").mkdir()
    (directory / "labels").mkdir()
    samples = []
    for item in records:
        source = Path(item["source_path"])
        filename = item["image_sha256"] + source.suffix.lower()
        copied = directory / "images" / filename
        label = directory / "labels" / (item["image_sha256"] + ".txt")
        shutil.copyfile(source, copied)
        if file_hash(copied) != item["image_sha256"]:
            raise ValueError("Original mudou durante a cópia; não usar o acervo incompleto.")
        label.write_text(boxes_to_yolo(item["boxes"], item["width"], item["height"]), encoding="utf-8")
        samples.append({"image": copied.relative_to(directory).as_posix(),
                        "image_sha256": item["image_sha256"], "label": label.relative_to(directory).as_posix(),
                        "label_sha256": file_hash(label), "building_group": item["building_group"],
                        "approved_taxonomy_sha256": file_hash(taxonomy_path)})
    write_json(directory / "manifest.json", {
        "schema_version": 1, "kind": "historical_human_annotation_corpus",
        "ready_for_training": False, "active_classes_at_review": names,
        "taxonomy_sha256": file_hash(taxonomy_path), "taxonomy_path": str(taxonomy_path),
        "source_registry_path": str(registry_path), "source_registry_sha256": file_hash(registry_path),
        "human_approved_images_for_original_taxonomy": len(samples),
        "boxes_by_class_id": dict(counts), "samples": samples,
        "note": "Referência humana da v3, dez classes. Conferência das classes atuais e validação/teste independentes pendentes.",
    })
    return directory


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/project.yaml")
    feedback_source = Path("C:/Users/claud/Desktop/dronecamp-revisao-ceasa_v3.json")
    feedback = json.loads(feedback_source.read_text(encoding="utf-8"))
    digest = file_hash(feedback_source)
    source_registry = root / "data/reviews/ceasa_v3/registry_with_candidates.json"
    if feedback.get("registry_sha256") != file_hash(source_registry):
        raise ValueError("Feedback não corresponde ao registro v3 preservado.")
    previous_taxonomy = root / "configs/taxonomy_ceasa_v3_pedaco_telha_rufo_deslocado.json"
    historical_config = replace(config, taxonomy_path=previous_taxonomy)
    historical = root / "data/reviews" / f"ceasa_v3_humana_{digest[:12]}" / "registry.json"
    current = root / "data/reviews" / f"ceasa_v6_corrigida_{digest[:12]}" / "registry.json"
    if historical.exists() or current.exists():
        raise ValueError("Versões desta revisão já existem; não reimportar ou sobrescrever.")
    preserve_inputs(root, feedback_source)
    historical.parent.mkdir(parents=True, exist_ok=False)
    archived_feedback = historical.parent / "feedback.json"
    shutil.copy2(feedback_source, archived_feedback)
    if file_hash(archived_feedback) != digest:
        raise ValueError("Cópia do feedback diverge do original.")
    approved = import_human_feedback(historical_config, source_registry, archived_feedback, historical)
    migrated = migrate_review_taxonomy(config, historical, previous_taxonomy, current)
    corpus = export_historical_annotations(root, historical, previous_taxonomy, digest)
    print(json.dumps({"historical_registry": str(historical), "current_registry": str(current),
                      "historical_human_approved": sum(item.get("human_approved") is True for item in approved["images"]),
                      "current_human_approved": sum(item.get("human_approved") is True for item in migrated["images"]),
                      "current_proposed_boxes": sum(len(item["boxes"]) for item in migrated["images"]),
                      "historical_annotation_corpus": str(corpus)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
