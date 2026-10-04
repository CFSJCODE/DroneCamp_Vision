"""Dataset de PRODUÇÃO a partir da revisão humana (não treina modelos).

Função no projeto: equivalente de produção do ``pilot.py``. Gera
``data/versions/<versão>/`` com fotos, labels, ``groups.csv`` e
``provenance.json``, usando a divisão por edificação de ``configs/review_groups.json``.
É o comando ``build-reviewed-data``.

O que faz:
- ``_approved_images``: fotos com aprovação humana e grupo de edificação conhecido.
- ``_write_samples``: copia fotos e escreve labels por split.
- ``_readiness``: diz se o dataset já pode treinar (exige três edificações e
  exemplos das classes); hoje, com só o CEASA, a resposta é "não".
- ``build_approved_dataset``: executa tudo e devolve o relatório.

Mais imagens só ajudam quando seus rótulos são confiáveis. Este módulo mantém
aprovação, origem e grupos de edificações separados das propostas feitas por IA.

Quando mexer: regras de prontidão para produção ou a lista ``MVP_NAMES``
(sempre acrescentando classes ao fim, junto com a taxonomia).
"""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import yaml

from .config import ProjectConfig, detection_names, load_taxonomy
from .dataset import IMAGE_EXTENSIONS, SPLITS, validate_dataset
from .io import file_hash, write_json
from .review_data import boxes_to_yolo, review_image_size, validate_boxes

# As 13 classes ativas, na ordem dos IDs (precisa coincidir com configs/taxonomy.json).
MVP_NAMES = (
    "telha_quebrada", "telha_ausente", "residuos_telha", "reparo_telha",
    "rufo_ausente", "rufo_quebrado", "residuos_calha", "vegetacao_calha",
    "pedaco_telha",
    "rufo_deslocado",
    # Acrescente classes ao fim: os rótulos já revisados mantêm seus IDs.
    "pedaco_telha_sobreposto",
    "fixador_telha_frouxo",
    "reparo_rufo",
)


def _read_json(path: Path) -> dict:
    """Falhe antes de criar a versão se o contrato não puder ser lido."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"Schema inválido: {path.name}; esperado schema_version: 1.")
    return value


def _approved_images(registry: dict, groups: dict) -> tuple[list[dict], list[dict]]:
    """Selecione decisões explícitas; uma segunda revisão de IA não aprova nada."""
    approved, excluded = [], []
    seen = set()
    if not isinstance(registry.get("images"), list):
        raise ValueError("O registro deve conter images como lista.")
    for item in registry["images"]:
        digest = item.get("image_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Hash de imagem inválido no registro.")
        if digest in seen:
            raise ValueError("Imagem duplicada no registro de revisão.")
        seen.add(digest)
        if item.get("human_approved") is not True or item.get("status") not in {"positive", "negative"}:
            excluded.append({"image_sha256": digest, "status": item.get("status"),
                             "reason": "Sem aprovação humana explícita ou decisão ainda ambígua/excluída."})
            continue
        reviewer = item.get("human_reviewer")
        if item.get("technical_status") != "human_visual_reviewed" or not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("Aprovação precisa de autor e estado de revisão humana explícitos.")
        group = item.get("building_group")
        if not isinstance(group, str) or not group.strip() or group not in groups:
            raise ValueError("Cada imagem aprovada precisa de building_group e assignment explícito.")
        source_value = item.get("source_path")
        if not isinstance(source_value, str) or "://" in source_value:
            raise ValueError("source_path deve indicar um arquivo local conhecido.")
        source = Path(source_value)
        if not source.is_absolute() or not source.is_file() or source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError("source_path deve ser um arquivo de imagem absoluto e existente.")
        if file_hash(source) != digest:
            raise ValueError(f"Hash do original alterado: {source.name}.")
        size = review_image_size(source)
        if any(type(item.get(key)) is not int for key in ("width", "height")) or size != (item["width"], item["height"]):
            raise ValueError(f"Dimensões do original divergem do registro: {source.name}.")
        boxes = item.get("boxes")
        validate_boxes(boxes, *size, len(MVP_NAMES))
        if (item["status"] == "positive" and not boxes) or (item["status"] == "negative" and boxes):
            raise ValueError("Decisão positiva/negativa incompatível com as caixas.")
        approved.append({**item, "source_path": str(source.resolve()), "split": groups[group]})
    if not approved:
        raise ValueError("Nenhuma imagem tem aprovação humana explícita; nenhum dataset foi criado.")
    return approved, excluded


def _write_samples(directory: Path, images: list[dict]) -> list[dict]:
    """Copie bytes; nomes por hash evitam colisões entre campanhas diferentes."""
    provenance_images = []
    with (directory / "groups.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "group_id", "split"])
        writer.writeheader()
        for item in images:
            split, digest = item["split"], item["image_sha256"]
            source = Path(item["source_path"])
            filename = digest + source.suffix.lower()
            image_dir, label_dir = directory / "images" / split, directory / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            copied = image_dir / filename
            shutil.copyfile(source, copied)
            # Reconfira os bytes copiados: a origem pode mudar após o preflight.
            if file_hash(copied) != digest:
                raise ValueError("Original mudou durante a cópia; versão incompleta não deve ser utilizada.")
            label = label_dir / (digest + ".txt")
            label.write_text(boxes_to_yolo(item["boxes"], item["width"], item["height"]), encoding="utf-8")
            relative = copied.relative_to(directory).as_posix()
            writer.writerow({"image": relative, "group_id": item["building_group"], "split": split})
            provenance_images.append({
                "image_sha256": digest, "image": relative,
                "label": label.relative_to(directory).as_posix(), "label_sha256": file_hash(label),
                "source_path": item["source_path"], "building_group": item["building_group"],
                "split": split, "width": item["width"], "height": item["height"],
                "status": item["status"], "human_approved": True,
                "technical_status": item.get("technical_status"),
                "approval_author": item.get("human_reviewer"), "approved_at": item.get("human_review_at"),
                "visual_review_status": item.get("visual_review_status"),
            })
    return provenance_images


def _readiness(directory: Path, images: list[dict]) -> dict:
    """Prontidão é estrutura, cobertura de classes e separação por edificação."""
    validation = validate_dataset(directory / "dataset.yaml", list(MVP_NAMES))
    groups = {split: sorted({item["building_group"] for item in images if item["split"] == split}) for split in SPLITS}
    train_counts = Counter(box["class_id"] for item in images if item["split"] == "train" for box in item["boxes"])
    missing_classes = [index for index in range(len(MVP_NAMES)) if not train_counts[index]]
    reasons = list(validation["errors"])
    if missing_classes:
        reasons.append("Faltam positivos no treino para as classes: " + ", ".join(MVP_NAMES[index] for index in missing_classes) + ".")
    if not all(groups.values()) or len({group for values in groups.values() for group in values}) < 3:
        reasons.append("Treino, validação e teste exigem ao menos três grupos de edificações independentes; um único CEASA não pode ocupar os três splits.")
    return {
        "schema_version": 1, "ready_for_training": validation["valid"] and not reasons,
        "state": "ready" if validation["valid"] and not reasons else "draft",
        "dataset_path": str(directory / "dataset.yaml"), "provenance_path": str(directory / "provenance.json"),
        "validation": validation, "reasons": reasons, "groups_by_split": groups,
        "counts": validation["counts"], "train_boxes_by_class": {str(index): train_counts[index] for index in range(len(MVP_NAMES))},
        "note": "A checagem não prova diversidade real ou qualidade técnica dos rótulos; não iniciou treinamento.",
    }


def build_approved_dataset(config: ProjectConfig, registry_path: Path, assignments_path: Path, output_dir: Path) -> dict:
    """Exporte uma nova versão, recusando sobrescrita e imagens sem aprovação.

    ``assignments`` usa {schema_version: 1, groups: {building_group: split}}.
    Splits sem amostras ficam ausentes: um rascunho nunca simula diversidade.
    """
    # 1. Saída nova, taxonomia do MVP e registro na mesma taxonomia.
    registry_path, assignments_path, directory = Path(registry_path), Path(assignments_path), Path(output_dir)
    if directory.exists() or directory.is_symlink():
        raise ValueError("Saída já existe; datasets são imutáveis. Escolha uma nova versão.")
    registry, assignments = _read_json(registry_path), _read_json(assignments_path)
    taxonomy = load_taxonomy(config.taxonomy_path)
    if tuple(detection_names(taxonomy)) != MVP_NAMES or [item["id"] for item in taxonomy["classes"] if item["phase"] == 1] != list(range(len(MVP_NAMES))):
        raise ValueError("A taxonomia ativa deve manter os IDs e nomes do MVP.")
    if registry.get("taxonomy_sha256") != file_hash(config.taxonomy_path):
        raise ValueError("Taxonomia mudou; migre e revise o registro antes de construir dados.")
    # 2. Cada edificação (building_group) vai inteira para um split, conforme o arquivo de grupos.
    groups = assignments.get("groups")
    if not isinstance(groups, dict) or any(not isinstance(group, str) or not group.strip() or split not in SPLITS for group, split in groups.items()):
        raise ValueError("Assignments deve mapear grupos para train, val ou test.")
    images, excluded = _approved_images(registry, groups)
    input_hashes = {"registry_sha256": file_hash(registry_path), "assignments_sha256": file_hash(assignments_path),
                    "taxonomy_sha256": file_hash(config.taxonomy_path)}
    directory = directory.resolve()
    # Toda aprovação, geometria e origem foi conferida antes da primeira escrita.
    directory.mkdir(parents=True, exist_ok=False)
    # 3. Fotos, labels, dataset.yaml e provenance.json; por fim, a prontidão.
    provenance_images = _write_samples(directory, images)
    present_splits = {item["split"] for item in images}
    dataset_yaml = {"path": str(directory), **{split: f"images/{split}" for split in SPLITS if split in present_splits},
                    "nc": len(MVP_NAMES), "names": list(MVP_NAMES)}
    (directory / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(directory / "provenance.json", {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "version": directory.name, **input_hashes, "registry_path": str(registry_path.resolve()),
        "assignments_path": str(assignments_path.resolve()), "images": provenance_images, "excluded_images": excluded,
        "human_feedback": registry.get("human_feedback"), "policy": "Somente revisão humana explícita; arquivos anteriores preservados."})
    readiness = _readiness(directory, images)
    write_json(directory / "readiness.json", readiness)
    return readiness
