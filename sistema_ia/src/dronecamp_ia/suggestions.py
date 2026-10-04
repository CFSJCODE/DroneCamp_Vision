"""Sugestões do detector na página de revisão: candidatas, nunca aprovação.

O revisor aceita, corrige ou descarta cada caixa. Só o feedback humano
importado entra no próximo dataset, fechando o ciclo de melhoria do modelo.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .backend import check_domain_names, load_detector
from .config import ProjectConfig, detection_names, load_taxonomy
from .dataset import IMAGE_EXTENSIONS
from .io import file_hash, write_json
from .review_data import REVIEW_SCHEMA, review_image_size, validate_boxes
from .training import is_pilot_checkpoint, pilot_inference_config

SUGGESTION_WARNING = ("Sugestões automáticas: aceite somente o que você conferir. "
                      "O modelo pode errar a classe, a caixa ou deixar de ver ocorrências.")


def detect_boxes(model, path: Path, config: ProjectConfig, conf: float) -> list[dict]:
    """Inferência real; caixas são arredondadas e recortadas aos pixels da foto."""
    width, height = review_image_size(path)
    parameters = {**config.prediction, "conf": conf}
    result = next(iter(model.predict(source=str(path), device=config.device, save=False, verbose=False,
                                     stream=True, rect=False, **parameters)))
    boxes = []
    if result.boxes is None:
        return boxes
    for (x1, y1, x2, y2), confidence, class_id in zip(result.boxes.xyxy.cpu().tolist(),
                                                     result.boxes.conf.cpu().tolist(),
                                                     result.boxes.cls.cpu().tolist()):
        coordinates = [max(0, min(width, round(x1))), max(0, min(height, round(y1))),
                       max(0, min(width, round(x2))), max(0, min(height, round(y2)))]
        candidate = {"class_id": int(class_id), "bbox_xyxy": coordinates, "confidence": round(float(confidence), 4)}
        try:
            validate_boxes([candidate], width, height, len(result.names))
        except ValueError:
            continue  # caixa degenerada após arredondamento
        boxes.append(candidate)
    boxes.sort(key=lambda value: -value["confidence"])
    for index, box in enumerate(boxes, start=1):
        box["id"] = f"s{index:03d}"
    return boxes


def suggest_for_registry(config: ProjectConfig, registry_path: Path, weights: str, conf: float | None = None) -> Path:
    """Grave suggestions.json ao lado do registro e regenere a página de revisão."""
    from .review_render import render_review_package

    registry_path = Path(registry_path).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("taxonomy_sha256") != file_hash(config.taxonomy_path):
        raise ValueError("Taxonomia mudou: migre a revisão antes de sugerir caixas.")
    conf = float(conf if conf is not None else config.pilot.get("suggestion_conf", config.prediction["conf"]))
    if not 0 < conf <= 1:
        raise ValueError("conf deve estar no intervalo (0, 1].")
    model = load_detector(config, weights)
    check_domain_names(model, detection_names(load_taxonomy(config.taxonomy_path)))
    pilot = is_pilot_checkpoint(model.ckpt_path)
    config = pilot_inference_config(config, model)
    images = {}
    for item in registry["images"]:
        source = Path(item["source_path"])
        if file_hash(source) != item["image_sha256"]:
            raise ValueError(f"A foto original mudou: {item.get('filename')}.")
        images[item["image_sha256"]] = detect_boxes(model, source, config, conf)
    checkpoint = Path(model.ckpt_path).resolve()
    write_json(registry_path.parent / "suggestions.json", {
        "schema_version": 1, "registry_sha256": file_hash(registry_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weights": str(checkpoint), "weights_sha256": file_hash(checkpoint), "pilot": pilot,
        "model_label": "Modelo piloto (não validado)" if pilot else "Modelo especializado",
        "conf": conf, "imgsz": config.prediction["imgsz"], "warning": SUGGESTION_WARNING, "images": images,
        "total_suggestions": sum(len(values) for values in images.values()),
    })
    return render_review_package(config, registry_path)


def create_review_for_new_images(config: ProjectConfig, source: Path, output: Path, building_group: str,
                                 weights: str, conf: float | None = None) -> Path:
    """Fotos novas viram uma revisão própria, com originais congelados por hash."""
    source, output = Path(source).expanduser().resolve(), Path(output)
    if output.exists():
        raise ValueError("Saída já existe; escolha uma nova versão da revisão.")
    if not building_group or not building_group.strip() or len(building_group) > 100:
        raise ValueError("Informe o grupo da edificação/campanha (ex.: galpao_b_2026_10).")
    files = [source] if source.is_file() else sorted(
        path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    files = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        raise ValueError(f"Nenhuma foto suportada em {source}.")
    for path in files:
        # Rotação EXIF precisa de cópia canonicalizada antes da anotação.
        review_image_size(path)
    output = output.resolve()
    originals = output / "originals"
    originals.mkdir(parents=True)
    images, seen = [], set()
    for path in files:
        digest = file_hash(path)
        if digest in seen:
            continue
        seen.add(digest)
        frozen = originals / f"{digest}{path.suffix.lower()}"
        shutil.copyfile(path, frozen)
        width, height = review_image_size(frozen)
        images.append({
            "image_sha256": digest, "filename": path.name, "pages": [], "width": width, "height": height,
            "status": "ambiguous", "boxes": [], "annotator": "sem_anotacao_humana",
            "notes": "Foto nova: nenhuma caixa revisada. Use as sugestões do modelo como ponto de partida.",
            "visual_review_status": "model_suggestions_only", "technical_status": "pending_human_review",
            "human_approved": False, "building_group": building_group.strip(),
            "scene_group": f"{building_group.strip()}_{digest[:12]}", "source_path": str(frozen),
            "original_path": str(path), "severity": None,
        })
    registry = {
        "schema_version": REVIEW_SCHEMA, "version": output.name,
        "taxonomy_sha256": file_hash(config.taxonomy_path),
        "taxonomy_version": load_taxonomy(config.taxonomy_path)["version"],
        "sources": [{"folder": str(source), "files": len(images)}], "images": images,
        "scope": "fotos_novas_com_sugestoes_do_modelo_pendentes_de_revisao",
    }
    registry_path = output / "registry.json"
    write_json(registry_path, registry)
    return suggest_for_registry(config, registry_path, weights, conf)
