"""Sugestões do detector na página de revisão: candidatas, nunca aprovação.

Função no projeto: usa um modelo treinado (em geral o piloto) para propor caixas
que o revisor aceita, corrige ou descarta na página. É o comando ``suggest``.

O que faz:
- ``detect_boxes``: roda o modelo numa foto e devolve caixas com classe e confiança.
- ``suggest_for_registry``: grava ``suggestions.json`` ao lado de uma revisão
  existente e regenera a página.
- ``create_review_for_new_images``: cria uma revisão nova para fotos novas
  (cópia congelada por hash em ``originals/``) já com as sugestões.

Só o feedback humano exportado e importado entra no próximo dataset; é isso que
fecha o ciclo de melhoria do modelo.

Quando mexer: confiança mínima padrão (``pilot.suggestion_conf`` em
``configs/project.yaml``) ou o que é gravado em ``suggestions.json``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .backend import check_domain_names, load_detector
from .config import ProjectConfig, detection_names, load_taxonomy
from .dataset import IMAGE_EXTENSIONS
from .io import file_hash, resolve_local_path, write_json
from .review_data import REVIEW_SCHEMA, review_image_size, validate_boxes
from .training import is_pilot_checkpoint, pilot_inference_config

SUGGESTION_WARNING = ("Sugestões automáticas: aceite somente o que você conferir. "
                      "O modelo pode errar a classe, a caixa ou deixar de ver ocorrências.")


# ---------------------------------------------------------------------------
# Inferência numa foto.
# ---------------------------------------------------------------------------

def detect_boxes(model, path: Path, config: ProjectConfig, conf: float) -> list[dict]:
    """Inferência real; caixas são arredondadas e recortadas aos pixels da foto."""
    # rect=False: entrada quadrada, igual à usada na exportação ONNX.
    width, height = review_image_size(path)
    parameters = {**config.prediction, "conf": conf}
    result = next(iter(model.predict(source=str(path), device=config.device, save=False, verbose=False,
                                     stream=True, rect=False, **parameters)))
    # Converte cada detecção em {classe, caixa em pixels inteiros, confiança}.
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
    # Mais confiantes primeiro; ids s001, s002... usados pela página ao descartar.
    boxes.sort(key=lambda value: -value["confidence"])
    for index, box in enumerate(boxes, start=1):
        box["id"] = f"s{index:03d}"
    return boxes


# ---------------------------------------------------------------------------
# Sugestões numa revisão existente ou numa revisão nova de fotos novas.
# ---------------------------------------------------------------------------

def suggest_for_registry(config: ProjectConfig, registry_path: Path, weights: str, conf: float | None = None) -> Path:
    """Grave suggestions.json ao lado do registro e regenere a página de revisão."""
    from .review_render import render_review_package

    # 1. Registro na taxonomia atual e confiança mínima (padrão: pilot.suggestion_conf).
    registry_path = Path(registry_path).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("taxonomy_sha256") != file_hash(config.taxonomy_path):
        raise ValueError("Taxonomia mudou: migre a revisão antes de sugerir caixas.")
    conf = float(conf if conf is not None else config.pilot.get("suggestion_conf", config.prediction["conf"]))
    if not 0 < conf <= 1:
        raise ValueError("conf deve estar no intervalo (0, 1].")
    # 2. Modelo com as 13 classes; pesos piloto inferem no tamanho do treino (640).
    model = load_detector(config, weights)
    check_domain_names(model, detection_names(load_taxonomy(config.taxonomy_path)))
    pilot = is_pilot_checkpoint(model.ckpt_path)
    config = pilot_inference_config(config, model)
    # 3. Roda o modelo em cada foto do registro (conferindo o hash do original).
    images = {}
    for item in registry["images"]:
        source = resolve_local_path(item["source_path"], config.root)
        if file_hash(source) != item["image_sha256"]:
            raise ValueError(f"A foto original mudou: {item.get('filename')}.")
        images[item["image_sha256"]] = detect_boxes(model, source, config, conf)
    # 4. suggestions.json fica preso a esta versão do registro (hash) e aos pesos usados.
    checkpoint = Path(model.ckpt_path).resolve()
    write_json(registry_path.parent / "suggestions.json", {
        "schema_version": 1, "registry_sha256": file_hash(registry_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weights": str(checkpoint), "weights_sha256": file_hash(checkpoint), "pilot": pilot,
        "model_label": "Modelo piloto (não validado)" if pilot else "Modelo especializado",
        "conf": conf, "imgsz": config.prediction["imgsz"], "warning": SUGGESTION_WARNING, "images": images,
        "total_suggestions": sum(len(values) for values in images.values()),
    })
    # 5. Regenera index.html com as sugestões tracejadas.
    return render_review_package(config, registry_path)


def create_review_for_new_images(config: ProjectConfig, source: Path, output: Path, building_group: str,
                                 weights: str, conf: float | None = None) -> Path:
    """Fotos novas viram uma revisão própria, com originais congelados por hash."""
    # 1. Pasta de saída nova, grupo da edificação e lista de fotos suportadas.
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
    # 2. Copia cada foto para originals/<sha256> (fotos repetidas entram uma vez).
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
    # 3. Registro novo (todas ambíguas, sem caixas) e, em seguida, as sugestões.
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
