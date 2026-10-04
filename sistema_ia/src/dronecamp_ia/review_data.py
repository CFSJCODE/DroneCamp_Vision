"""Registros de revisão: propostas, decisões humanas e migrações de taxonomia.

Função no projeto: mantém os ``registry.json`` de ``data/reviews/<versão>/``,
que dizem, para cada foto, as caixas, o status (positiva/negativa/ambígua) e
quem decidiu. O treino só aprende com o que um humano aprovou aqui.

O que faz:
- ``review_image_size`` / ``validate_boxes`` / ``boxes_to_yolo``: utilidades de
  foto e caixa usadas em todo o projeto.
- ``build_review_registry`` / ``apply_visual_audits``: montam a revisão do CEASA a
  partir de propostas e segunda leitura por IA (comando ``prepare-review``).
- ``import_human_feedback``: importa o JSON exportado pela página (``import-review``).
- ``apply_ai_proposals``: leva propostas visuais de IA como candidatas (``add-ai-proposals``).
- ``migrate_review_taxonomy``: acrescenta classes preservando caixas (``migrate-review``).

Toda saída é uma versão nova: registros anteriores nunca são sobrescritos.
Propostas visuais nunca viram aprovação humana sozinhas.

Quando mexer: para mudar o que o arquivo exportado pela página pode conter ou
as regras de aprovação (ex.: exigir ``confirmed_complete``).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import math
import re

from PIL import Image

from .config import ProjectConfig, detection_names, load_taxonomy
from .io import file_hash, resolve_local_path, write_json

# Versão do formato dos registros e status possíveis de uma foto.
REVIEW_SCHEMA = 1
IMAGE_STATUSES = {"positive", "negative", "ambiguous", "excluded"}


# ---------------------------------------------------------------------------
# Utilidades de foto e caixa (usadas também por pilot.py e suggestions.py).
# ---------------------------------------------------------------------------

def review_image_size(path: Path) -> tuple[int, int]:
    """Use coordenadas nos pixels originais, sem rotação EXIF implícita.

    Fotos de celular com rotação declarada precisam de uma cópia canonicalizada
    e de novo hash antes da anotação; nunca gire silenciosamente uma revisão.
    """
    with Image.open(path) as image:
        if image.getexif().get(274, 1) != 1:
            raise ValueError("Orientação EXIF exige canonicalização da foto antes da revisão.")
        image.load()
        return image.size


def validate_boxes(boxes: list[dict], width: int, height: int, class_count: int) -> None:
    """Rejeite caixas inválidas antes de desenhar, importar feedback ou treinar."""
    if not isinstance(boxes, list) or len(boxes) > 2000:
        raise ValueError("boxes deve ser uma lista com no máximo 2000 ocorrências.")
    for position, box in enumerate(boxes, start=1):
        if not isinstance(box, dict):
            raise ValueError(f"Caixa {position}: esperado um objeto com classe e coordenadas.")
        class_id = box.get("class_id")
        coordinates = box.get("bbox_xyxy")
        if type(class_id) is not int or not 0 <= class_id < class_count:
            raise ValueError(f"Caixa {position}: ID de classe inválido.")
        if not isinstance(coordinates, list) or len(coordinates) != 4:
            raise ValueError(f"Caixa {position}: esperado [x1,y1,x2,y2].")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in coordinates):
            raise ValueError(f"Caixa {position}: coordenadas devem ser números finitos.")
        x1, y1, x2, y2 = coordinates
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Caixa {position}: geometria fora da imagem ou área vazia.")


def boxes_to_yolo(boxes: list[dict], width: int, height: int) -> str:
    """A conversão não aprova a anotação; a proveniência define seu uso.

    Formato YOLO: uma linha por caixa, ``classe cx cy largura altura`` normalizados (0–1).
    """
    lines = []
    for box in boxes:
        x1, y1, x2, y2 = box["bbox_xyxy"]
        lines.append(f"{box['class_id']} {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
    return "\n".join(lines) + ("\n" if lines else "")


# ---------------------------------------------------------------------------
# Revisão inicial do CEASA: propostas por IA e segunda leitura (prepare-review).
# ---------------------------------------------------------------------------

def build_review_registry(config: ProjectConfig, directory: Path) -> dict:
    """Consolide lotes sem perder duplicatas documentais ou inventar decisões."""
    manifest_path = config.root / "data/reference/ceasa/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    originals = manifest_path.parent
    occurrences = defaultdict(list)
    for image in manifest["images"]:
        occurrences[image["sha256"]].append(image)
    proposals = {}
    taxonomy = load_taxonomy(config.taxonomy_path)
    class_count = len(detection_names(taxonomy))
    sources = []
    for path in sorted((directory / "proposals").glob("*.json")):
        batch = json.loads(path.read_text(encoding="utf-8"))
        if batch.get("schema_version") != REVIEW_SCHEMA:
            raise ValueError(f"Schema inválido em {path.name}.")
        declared_taxonomy = batch.get("taxonomy_sha256")
        if declared_taxonomy and declared_taxonomy != file_hash(config.taxonomy_path):
            raise ValueError("Proposta pertence a outra taxonomia; faça migração explícita.")
        if not declared_taxonomy and class_count > 8:
            raise ValueError("Lote histórico não declara as novas classes. Use migrate-review com a taxonomia anterior.")
        sources.append({"file": path.name, "sha256": file_hash(path), "annotator": batch["annotator"]})
        for item in batch["images"]:
            digest = item["image_sha256"]
            if digest not in occurrences or digest in proposals:
                raise ValueError(f"Imagem desconhecida ou proposta duplicada: {digest}.")
            first = occurrences[digest][0]
            path_original = originals / first["filename"]
            if file_hash(path_original) != digest:
                raise ValueError(f"Original alterado: {first['filename']}.")
            width, height = review_image_size(path_original)
            if (item["width"], item["height"]) != (width, height):
                raise ValueError(f"Dimensões da proposta divergem da foto: {first['filename']}.")
            if item["status"] not in IMAGE_STATUSES:
                raise ValueError("Status de triagem inválido.")
            validate_boxes(item["boxes"], width, height, class_count)
            if item["status"] == "positive" and not item["boxes"]:
                raise ValueError("Imagem positiva precisa de caixas.")
            if item["status"] == "negative" and item["boxes"]:
                raise ValueError("Imagem negativa não pode conter caixas.")
            proposals[digest] = {
                **item, "filename": first["filename"], "width": width, "height": height,
                "pages": sorted({occurrence["page"] for occurrence in occurrences[digest]}),
                "annotator": batch["annotator"], "visual_review_status": "first_pass_ai",
                "technical_status": "pending_human_review", "human_approved": False,
                "building_group": "ceasa_pavilhao_a", "source_path": str(path_original),
                "severity": None,
            }
    missing = set(occurrences) - set(proposals)
    if missing:
        raise ValueError(f"Revisão incompleta: faltam {len(missing)} imagens únicas.")
    registry = {
        "schema_version": REVIEW_SCHEMA, "version": directory.name,
        "taxonomy_sha256": file_hash(config.taxonomy_path), "taxonomy_version": load_taxonomy(config.taxonomy_path)["version"],
        "source_manifest_sha256": file_hash(manifest_path), "source_pdf_sha256": manifest["source_pdf_sha256"],
        "sources": sources, "images": sorted(proposals.values(), key=lambda item: (min(item["pages"]), item["filename"])),
        "scope": "propostas_visuais_por_ia_pendentes_de_validacao_tecnica",
    }
    write_json(directory / "registry.json", registry)
    write_review_summary(directory, registry)
    return registry


def write_review_summary(directory: Path, registry: dict) -> dict:
    """review_summary.json: contagens de fotos, caixas por classe e aprovações."""
    counts = Counter(item["status"] for item in registry["images"])
    per_class = Counter(box["class_id"] for item in registry["images"] for box in item["boxes"])
    summary = {
        "unique_images": len(registry["images"]), "image_status_counts": dict(counts),
        "boxes_by_class_id": dict(per_class), "total_proposed_boxes": sum(per_class.values()),
        "human_approved_images": sum(bool(item.get("human_approved")) for item in registry["images"]),
        "historical_human_review_images": sum(
            item.get("visual_review_status") == "prior_taxonomy_human_review"
            and item.get("previous_human_approved") is True for item in registry["images"]),
        "second_visual_review_images": sum(item.get("visual_review_status") == "second_pass_ai" for item in registry["images"]),
        "prior_taxonomy_second_visual_review_images": sum(
            item.get("previous_visual_review_status") == "second_pass_ai"
            or item.get("historical_visual_review_status") == "second_pass_ai"
            for item in registry["images"]),
        "taxonomy_recheck_pending_images": sum(bool(item.get("taxonomy_recheck_pending")) for item in registry["images"]),
        "independent_buildings": len({item["building_group"] for item in registry["images"]}),
        "ready_for_independent_evaluation": False,
        "note": "Propostas visuais são revisáveis; uma única edificação não mede generalização para outros locais.",
    }
    write_json(directory / "review_summary.json", summary)
    return summary


def apply_visual_audits(config: ProjectConfig, directory: Path) -> dict:
    """Uma segunda leitura de IA melhora a proposta, sem simular responsável humano."""
    registry_path = directory / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("taxonomy_sha256") != file_hash(config.taxonomy_path):
        raise ValueError("Taxonomia mudou: migre a revisão antes de auditá-la.")
    images = {item["image_sha256"]: item for item in registry["images"]}
    if len(images) != len(registry["images"]):
        raise ValueError("Imagem duplicada no registro de revisão.")
    seen = set()
    count = len(detection_names(load_taxonomy(config.taxonomy_path)))
    audit_sources = []
    for path in sorted((directory / "audits").glob("*.json")):
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit.get("schema_version") != REVIEW_SCHEMA:
            raise ValueError(f"Schema inválido em {path.name}.")
        audit_sources.append({"filename": path.name, "sha256": file_hash(path), "reviewer": audit["reviewer"]})
        for review in audit["images"]:
            digest = review["image_sha256"]
            if digest in seen or digest not in images:
                raise ValueError("Imagem duplicada/desconhecida na segunda revisão.")
            seen.add(digest)
            image = images[digest]
            if audit["reviewer"] == image["annotator"]:
                raise ValueError("A segunda leitura precisa de outro revisor.")
            if "boxes" in review:
                validate_boxes(review["boxes"], image["width"], image["height"], count)
                image["boxes"] = review["boxes"]
            status = review.get("status", image["status"])
            if status not in IMAGE_STATUSES or (status == "negative" and image["boxes"]):
                raise ValueError("Decisão de revisão incompatível com as caixas.")
            if status == "positive" and not image["boxes"]:
                raise ValueError("Revisão positiva sem ocorrência.")
            image.update(status=status, visual_review_status="second_pass_ai", second_reviewer=audit["reviewer"],
                         second_review_notes=review["notes"], human_approved=False)
    if seen != set(images):
        raise ValueError(f"Segunda leitura incompleta: {len(seen)}/{len(images)} imagens.")
    registry["audits"] = audit_sources
    write_json(directory / "registry_second_pass.json", registry)
    write_review_summary(directory, registry)
    return registry


# ---------------------------------------------------------------------------
# Decisões humanas exportadas pela página (import-review).
# ---------------------------------------------------------------------------

def import_human_feedback(config: ProjectConfig, registry_path: Path, feedback_path: Path, output: Path) -> dict:
    """Valide arquivo exportado pelo revisor e preserve o registro anterior."""
    # 1. O feedback precisa ter sido gerado a partir desta versão exata do registro (hash).
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or registry.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError("Schema inválido no registro de revisão.")
    if not isinstance(feedback, dict):
        raise ValueError("Feedback deve ser um objeto JSON.")
    if output.exists():
        raise ValueError("Saída já existe; use uma nova versão da revisão.")
    if feedback.get("schema_version") != REVIEW_SCHEMA or feedback.get("registry_sha256") != file_hash(registry_path):
        raise ValueError("Feedback não corresponde à versão original da revisão.")
    reviewer_value = feedback.get("reviewer_id", "")
    reviewer = reviewer_value.strip() if isinstance(reviewer_value, str) else ""
    if not reviewer or len(reviewer) > 100:
        raise ValueError("Informe um identificador de revisor válido.")
    class_count = len(detection_names(load_taxonomy(config.taxonomy_path)))
    if registry["taxonomy_sha256"] != file_hash(config.taxonomy_path):
        raise ValueError("Taxonomia mudou: migre a revisão antes de aprovar.")
    if not isinstance(registry.get("images"), list) or any(not isinstance(item, dict) for item in registry["images"]):
        raise ValueError("O registro deve conter uma lista de imagens válida.")
    by_id = {item["image_sha256"]: item for item in registry["images"]}
    if len(by_id) != len(registry["images"]):
        raise ValueError("Imagem duplicada no registro de revisão.")
    # 2. Cada decisão: foto conhecida e intacta, caixas válidas, revisão integral confirmada.
    decisions = feedback.get("images")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("Feedback deve conter uma lista não vazia de decisões.")
    seen = set()
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ValueError("Cada decisão deve ser um objeto JSON.")
        digest = decision.get("image_sha256")
        if digest not in by_id or digest in seen:
            raise ValueError("Imagem desconhecida/duplicada no feedback.")
        seen.add(digest)
        item = by_id[digest]
        source = resolve_local_path(item["source_path"], config.root)
        if file_hash(source) != digest:
            raise ValueError("Imagem original diverge do arquivo revisado.")
        if review_image_size(source) != (item["width"], item["height"]):
            raise ValueError("Dimensões do registro divergem da foto revisada.")
        status = decision.get("status")
        if status not in IMAGE_STATUSES:
            raise ValueError("Status de feedback inválido.")
        boxes = decision.get("boxes", [])
        validate_boxes(boxes, item["width"], item["height"], class_count)
        approved = status in {"positive", "negative"}
        if approved and decision.get("confirmed_complete") is not True:
            raise ValueError("Aprovação exige confirmação explícita da revisão integral.")
        if (status == "negative" and boxes) or (status == "positive" and not boxes):
            raise ValueError("Status positivo/negativo incompatível com ocorrências.")
        # Positiva/negativa confirmada = aprovada para treino; ambígua continua pendente.
        item.update(status=status, boxes=boxes, human_approved=approved,
                    technical_status="human_visual_reviewed" if approved else "pending_human_review",
                    human_reviewer=reviewer, human_notes=decision.get("notes", ""),
                    human_review_at=feedback.get("exported_at"), severity=None,
                    taxonomy_recheck_pending=not approved)
    # 3. Grava a nova versão com o hash do feedback e o vínculo com o registro anterior.
    registry["human_feedback"] = {"sha256": file_hash(feedback_path), "reviewer_id": reviewer,
                                  "reviewed_images": len(seen), "source_registry_sha256": file_hash(registry_path)}
    registry["parent_registry_version"] = registry.get("version")
    registry["version"] = output.parent.name if output.parent.resolve() != registry_path.parent.resolve() else output.stem
    write_json(output, registry)
    return registry


# ---------------------------------------------------------------------------
# Propostas visuais de IA (add-ai-proposals) e migração de taxonomia.
# ---------------------------------------------------------------------------

# Nome de classe sugerida: minúsculas, números e "_" (ex.: telha_trincada).
PROPOSED_CLASS_SLUG = re.compile(r"[a-z][a-z0-9_]{2,60}")


def apply_ai_proposals(config: ProjectConfig, registry_path: Path, proposals_path: Path, output: Path) -> dict:
    """Propostas visuais de IA preenchem a revisão como candidatas, nunca como aprovação.

    Classes ainda inexistentes ficam em proposed_new_class, ao lado da classe
    ativa mais próxima; ativá-las exige nova taxonomia e migração explícita.
    """
    if output.exists():
        raise ValueError("Saída já existe; use uma nova versão da revisão.")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    proposals = json.loads(proposals_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or registry.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError("Schema inválido no registro de revisão.")
    if not isinstance(proposals, dict) or proposals.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError("Schema inválido no arquivo de propostas.")
    if registry.get("taxonomy_sha256") != file_hash(config.taxonomy_path):
        raise ValueError("Taxonomia mudou: migre a revisão antes de propor caixas.")
    annotator_value = proposals.get("annotator", "")
    annotator = annotator_value.strip() if isinstance(annotator_value, str) else ""
    if not annotator or len(annotator) > 100:
        raise ValueError("Informe o anotador (IA) das propostas.")
    class_count = len(detection_names(load_taxonomy(config.taxonomy_path)))
    by_id = {item["image_sha256"]: item for item in registry["images"]}
    if len(by_id) != len(registry["images"]):
        raise ValueError("Imagem duplicada no registro de revisão.")
    if not isinstance(proposals.get("images"), list) or not proposals["images"]:
        raise ValueError("Propostas devem conter uma lista não vazia de imagens.")
    seen = set()
    for proposal in proposals["images"]:
        digest = proposal.get("image_sha256") if isinstance(proposal, dict) else None
        if digest not in by_id or digest in seen:
            raise ValueError("Imagem desconhecida/duplicada nas propostas.")
        seen.add(digest)
        item = by_id[digest]
        if item.get("human_approved") is True:
            raise ValueError(f"{item.get('filename')} já tem decisão humana; proposta de IA não a substitui.")
        source = resolve_local_path(item["source_path"], config.root)
        if file_hash(source) != digest:
            raise ValueError(f"A foto original mudou: {item.get('filename')}.")
        status, boxes = proposal.get("status"), proposal.get("boxes", [])
        if status not in {"positive", "negative", "ambiguous"}:
            raise ValueError("Status de proposta inválido.")
        validate_boxes(boxes, item["width"], item["height"], class_count)
        if (status == "negative" and boxes) or (status == "positive" and not boxes):
            raise ValueError("Status positivo/negativo incompatível com as caixas propostas.")
        for box in boxes:
            slug = box.get("proposed_new_class")
            if slug is not None and (not isinstance(slug, str) or not PROPOSED_CLASS_SLUG.fullmatch(slug)):
                raise ValueError("proposed_new_class deve ser um slug minúsculo (ex.: telha_trincada).")
        item.update(status=status, boxes=boxes, annotator=annotator, notes=proposal.get("notes") or item.get("notes", ""),
                    visual_review_status="first_pass_ai", technical_status="pending_human_review", human_approved=False)
    registry["ai_proposals"] = {"sha256": file_hash(proposals_path), "annotator": annotator, "images": len(seen),
                                "source_registry_sha256": file_hash(registry_path),
                                "proposed_new_classes": sorted({box["proposed_new_class"] for item in registry["images"]
                                                                for box in item["boxes"] if box.get("proposed_new_class")})}
    registry["parent_registry_version"] = registry.get("version")
    registry["version"] = output.parent.name
    write_json(output, registry)
    write_review_summary(output.parent, registry)
    return registry


def migrate_review_taxonomy(config: ProjectConfig, registry_path: Path, previous_taxonomy: Path, output: Path) -> dict:
    """Acrescente classes sem inventar caixas ou transferir aprovações antigas.

    Apenas expansão com IDs ativos anteriores preservados é suportada. Mudanças
    de significado, remoções ou reordenação exigem uma migração de anotações
    específica. O registro anterior e a taxonomia de origem continuam intactos.
    """
    if output.exists():
        raise ValueError("Saída já existe; use uma nova versão da revisão.")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    old_taxonomy = load_taxonomy(previous_taxonomy)
    new_taxonomy = load_taxonomy(config.taxonomy_path)
    old_names, new_names = detection_names(old_taxonomy), detection_names(new_taxonomy)
    if registry.get("schema_version") != REVIEW_SCHEMA or registry.get("taxonomy_sha256") != file_hash(previous_taxonomy):
        raise ValueError("Taxonomia anterior não corresponde ao registro de origem.")
    if len(new_names) <= len(old_names) or new_names[:len(old_names)] != old_names:
        raise ValueError("Esta migração suporta somente acréscimo de classes com IDs anteriores preservados.")
    source_registry_sha256 = file_hash(registry_path)
    previous_taxonomy_sha256 = file_hash(previous_taxonomy)
    seen = set()
    for item in registry["images"]:
        digest = item["image_sha256"]
        source = resolve_local_path(item["source_path"], config.root)
        if digest in seen or file_hash(source) != digest:
            raise ValueError("Imagem duplicada ou original alterado durante a migração.")
        seen.add(digest)
        if review_image_size(source) != (item["width"], item["height"]):
            raise ValueError("Dimensões da foto divergem do registro anterior.")
        validate_boxes(item["boxes"], item["width"], item["height"], len(old_names))
        reviewer_value = item.get("human_reviewer")
        has_reviewer = isinstance(reviewer_value, str) and bool(reviewer_value.strip())
        # O indicador isolado não comprova uma revisão humana integral. O estado
        # técnico, o autor e a ausência de pendência precisam ser compatíveis.
        current_human_approval = (
            item.get("human_approved") is True
            and item.get("technical_status") == "human_visual_reviewed"
            and has_reviewer
            and item.get("status") in {"positive", "negative"}
            and not item.get("taxonomy_recheck_pending")
        )
        historical_human_approval = (
            item.get("previous_human_approved") is True
            and item.get("visual_review_status") == "prior_taxonomy_human_review"
            and has_reviewer
        )
        # Uma migração repetida não equivale a revisar classes acrescentadas.
        # A aprovação humana nova cobre a taxonomia antiga inteira; a pendência
        # conserva somente o escopo efetivamente registrado na rodada anterior.
        if item.get("taxonomy_recheck_pending"):
            reviewed_ids = item.get("classes_reviewed_before_migration", [])
            if (not isinstance(reviewed_ids, list)
                    or any(type(value) is not int or value not in range(len(old_names)) for value in reviewed_ids)
                    or len(set(reviewed_ids)) != len(reviewed_ids)):
                raise ValueError("Escopo de classes revisadas antes da migração é inválido.")
        else:
            reviewed_ids = list(range(len(old_names)))
        if current_human_approval:
            item["historical_human_review"] = {
                "taxonomy_sha256": previous_taxonomy_sha256,
                "classes_reviewed": reviewed_ids,
                "source_registry_sha256": source_registry_sha256,
                "reviewer": item["human_reviewer"],
                "review_at": item.get("human_review_at"),
            }
        item.setdefault("historical_visual_review_status",
                        item.get("previous_visual_review_status", item.get("visual_review_status")))
        item["previous_visual_review_status"] = item.get("visual_review_status")
        item["previous_human_approved"] = current_human_approval or historical_human_approval
        item["visual_review_status"] = (
            "prior_taxonomy_human_review" if item["previous_human_approved"]
            else "prior_taxonomy_ai_review"
        )
        item["classes_reviewed_before_migration"] = reviewed_ids
        item["taxonomy_recheck_pending"] = True
        item["human_approved"] = False
        item["technical_status"] = "pending_human_review"
        # Negativo na taxonomia anterior não comprova ausência das novas classes.
        if item["status"] == "negative":
            item["previous_status"] = "negative"
            item["status"] = "ambiguous"
        item["severity"] = None
    migration = {
        "source_registry_sha256": source_registry_sha256, "source_registry_path": str(registry_path.resolve()),
        "previous_taxonomy_sha256": previous_taxonomy_sha256, "previous_taxonomy_path": str(previous_taxonomy.resolve()),
        "previous_active_classes": old_names, "new_active_classes": new_names,
        "new_class_ids": list(range(len(old_names), len(new_names))),
        "note": "Caixas anteriores preservadas. Revisão integral das novas classes e aprovação humana pendentes.",
    }
    # O último vínculo continua disponível para leitores antigos. O histórico
    # mantém os hashes das etapas anteriores, inclusive das revisões humanas.
    migration_history = registry.get("taxonomy_migration_history", [])
    if not isinstance(migration_history, list) or any(not isinstance(value, dict) for value in migration_history):
        raise ValueError("Histórico de migrações deve ser uma lista de registros.")
    previous_migration = registry.get("taxonomy_migration")
    if previous_migration is not None:
        if not isinstance(previous_migration, dict):
            raise ValueError("Registro da migração anterior é inválido.")
        if not migration_history or migration_history[-1] != previous_migration:
            migration_history.append(previous_migration)
    migration_history.append(migration)
    registry["taxonomy_migration_history"] = migration_history
    registry["taxonomy_migration"] = migration
    registry["taxonomy_sha256"] = file_hash(config.taxonomy_path)
    registry["taxonomy_version"] = new_taxonomy["version"]
    registry["parent_registry_version"] = registry.get("version")
    registry["version"] = output.parent.name
    registry["scope"] = "propostas_preservadas_expansao_de_taxonomia_pendente_de_revisao"
    write_json(output, registry)
    write_review_summary(output.parent, registry)
    return registry
