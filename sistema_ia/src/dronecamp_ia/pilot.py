"""Dataset piloto: aprende com as revisões humanas antes de existir diversidade.

O dataset de produção exige três edificações independentes. Enquanto só houver
o CEASA, o piloto usa as mesmas aprovações humanas, separa os splits por cena
(recortes da mesma área ficam juntos) e declara que nenhuma métrica dele mede
generalização. Pesos treinados aqui servem para sugerir caixas na revisão.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import csv
import json
from pathlib import Path
import shutil

import yaml

from .config import ProjectConfig, detection_names, load_taxonomy
from .dataset import IMAGE_EXTENSIONS, SPLITS, validate_dataset
from .io import file_hash, write_json
from .review_data import boxes_to_yolo, review_image_size, validate_boxes

PILOT_KIND = "piloto_edificacao_unica"
PILOT_WARNING = ("Piloto com fotos de uma única edificação: validação e teste não são independentes. "
                 "Use os pesos apenas para sugerir caixas à revisão humana.")


def _approved(registry_path: Path, taxonomy_digest: str, class_count: int) -> list[dict]:
    """Somente decisões humanas completas entram; o resto fica na fila de revisão."""
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or registry.get("schema_version") != 1:
        raise ValueError(f"Schema inválido no registro {registry_path}.")
    if registry.get("taxonomy_sha256") != taxonomy_digest:
        raise ValueError(f"Taxonomia mudou desde {registry_path.name}; migre e revise antes do piloto.")
    selected = []
    for item in registry.get("images", []):
        if item.get("human_approved") is not True or item.get("status") not in {"positive", "negative"}:
            continue
        reviewer = item.get("human_reviewer")
        if item.get("technical_status") != "human_visual_reviewed" or not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("Aprovação sem autor ou estado de revisão humana explícito.")
        source = Path(item["source_path"])
        if not source.is_absolute() or not source.is_file() or source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Original ausente: {item.get('filename')}.")
        if file_hash(source) != item["image_sha256"]:
            raise ValueError(f"Original alterado: {source.name}.")
        if review_image_size(source) != (item["width"], item["height"]):
            raise ValueError(f"Dimensões divergem do registro: {source.name}.")
        validate_boxes(item["boxes"], item["width"], item["height"], class_count)
        if (item["status"] == "positive") != bool(item["boxes"]):
            raise ValueError("Decisão positiva/negativa incompatível com as caixas.")
        # Sem cena declarada, cada foto vira o próprio grupo (não junta recortes).
        scene = item.get("scene_group") or f"foto_{item['image_sha256'][:12]}"
        selected.append({**item, "scene_group": scene, "registry_path": str(registry_path.resolve()),
                         "registry_sha256": file_hash(registry_path)})
    return selected


def split_by_scene(images: list[dict], seed: int, fractions: tuple[float, float]) -> dict[str, str]:
    """Distribua cenas inteiras; classes raras permanecem no treino.

    Uma cena só sai do treino se todas as suas classes continuarem com exemplos
    no treino. Por isso classes com uma única foto nunca são medidas.
    """
    scenes: dict[str, list[dict]] = defaultdict(list)
    for item in images:
        scenes[item["scene_group"]].append(item)
    order = sorted(scenes, key=lambda name: sha256(f"{seed}:{name}".encode("utf-8")).hexdigest())
    train_images = Counter(box_class for item in images for box_class in {box["class_id"] for box in item["boxes"]})
    targets = {"val": max(1, round(len(images) * fractions[0])), "test": max(1, round(len(images) * fractions[1]))}
    counts = Counter()
    assignment = {}
    for split in ("val", "test"):
        for scene in order:
            if scene in assignment or counts[split] >= targets[split]:
                continue
            members = scenes[scene]
            # Não ultrapasse muito a meta: cenas grandes ficam no treino.
            if counts[split] + len(members) > targets[split] + max(2, targets[split] // 2):
                continue
            classes = Counter(box_class for item in members for box_class in {box["class_id"] for box in item["boxes"]})
            if any(train_images[class_id] - count < 1 for class_id, count in classes.items()):
                continue
            assignment[scene] = split
            counts[split] += len(members)
            train_images.subtract(classes)
    for scene in order:
        assignment.setdefault(scene, "train")
    if not all(split in assignment.values() for split in SPLITS):
        raise ValueError("Fotos/cenas insuficientes para separar treino, validação e teste no piloto.")
    return assignment


def build_pilot_dataset(config: ProjectConfig, registry_paths: list[Path], output: Path,
                        seed: int = 42, fractions: tuple[float, float] = (0.15, 0.15)) -> dict:
    """Crie uma versão nova e imutável do dataset piloto a partir dos registros."""
    output = Path(output)
    if output.exists():
        raise ValueError("Saída já existe; datasets piloto também são imutáveis.")
    taxonomy = load_taxonomy(config.taxonomy_path)
    names = detection_names(taxonomy)
    digest = file_hash(config.taxonomy_path)
    images, seen = [], set()
    for registry_path in registry_paths:
        for item in _approved(Path(registry_path), digest, len(names)):
            # A mesma foto pode aparecer em revisões sucessivas: vale a mais recente.
            if item["image_sha256"] in seen:
                images = [value for value in images if value["image_sha256"] != item["image_sha256"]]
            seen.add(item["image_sha256"])
            images.append(item)
    if not images:
        raise ValueError("Nenhuma foto com aprovação humana; nada para treinar.")
    assignment = split_by_scene(images, seed, fractions)
    output = output.resolve()
    output.mkdir(parents=True)
    samples = []
    try:
        with (output / "groups.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image", "group_id", "split"])
            writer.writeheader()
            for item in sorted(images, key=lambda value: value["image_sha256"]):
                split = assignment[item["scene_group"]]
                source = Path(item["source_path"])
                image = output / "images" / split / f"{item['image_sha256']}{source.suffix.lower()}"
                label = output / "labels" / split / f"{item['image_sha256']}.txt"
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, image)
                if file_hash(image) != item["image_sha256"]:
                    raise ValueError("Original mudou durante a cópia.")
                label.write_text(boxes_to_yolo(item["boxes"], item["width"], item["height"]), encoding="utf-8")
                relative = image.relative_to(output).as_posix()
                writer.writerow({"image": relative, "group_id": item["scene_group"], "split": split})
                samples.append({
                    "image": relative, "image_sha256": item["image_sha256"],
                    "label": label.relative_to(output).as_posix(), "label_sha256": file_hash(label),
                    "split": split, "scene_group": item["scene_group"], "building_group": item.get("building_group"),
                    "status": item["status"], "boxes": len(item["boxes"]),
                    "classes": sorted({box["class_id"] for box in item["boxes"]}),
                    "approval_author": item["human_reviewer"], "approved_at": item.get("human_review_at"),
                    "registry_path": item["registry_path"], "registry_sha256": item["registry_sha256"],
                    "source_path": str(source.resolve()),
                })
        (output / "dataset.yaml").write_text(yaml.safe_dump(
            {"path": str(output), "train": "images/train", "val": "images/val", "test": "images/test",
             "nc": len(names), "names": names}, allow_unicode=True, sort_keys=False), encoding="utf-8")
        boxes_by_split = {split: Counter() for split in SPLITS}
        for item in images:
            boxes_by_split[assignment[item["scene_group"]]].update(box["class_id"] for box in item["boxes"])
        coverage = {name: {split: boxes_by_split[split][index] for split in SPLITS} for index, name in enumerate(names)}
        buildings = sorted({str(item.get("building_group")) for item in images})
        manifest = {
            "schema_version": 1, "kind": PILOT_KIND, "created_at": datetime.now(timezone.utc).isoformat(),
            "version": output.name, "taxonomy_sha256": digest, "seed": seed,
            "split_policy": "cena_inteira_por_split; classes raras mantidas no treino",
            "registries": sorted({(item["registry_path"], item["registry_sha256"]) for item in images}),
            "buildings": buildings, "independent_evaluation": len(buildings) >= 3,
            "production_ready": False, "warning": PILOT_WARNING,
            "images": samples, "boxes_by_class_and_split": coverage,
            "classes_without_training_boxes": [name for name in names if coverage[name]["train"] == 0],
            "classes_not_measured_in_val": [name for name in names if coverage[name]["val"] == 0],
        }
        write_json(output / "pilot.json", manifest)
    except Exception:
        # Uma versão incompleta não pode ser confundida com uma versão válida.
        (output / "INCOMPLETO.txt").write_text("Criação interrompida; não utilize esta pasta.\n", encoding="utf-8")
        raise
    report = validate_dataset(output / "dataset.yaml", names)
    write_json(output / "dataset_validation.json", report)
    if not report["valid"]:
        raise ValueError("Dataset piloto estruturalmente inválido: " + "; ".join(report["errors"]))
    return {"dataset": str(output / "dataset.yaml"), "counts": report["counts"],
            "classes_without_training_boxes": manifest["classes_without_training_boxes"],
            "classes_not_measured_in_val": manifest["classes_not_measured_in_val"],
            "buildings": buildings, "warning": PILOT_WARNING}


def validate_pilot_dataset(config: ProjectConfig, data_path: Path) -> dict:
    """Reconfira hashes e labels contra os registros humanos antes do treino piloto."""
    data_path = Path(data_path).resolve()
    names = detection_names(load_taxonomy(config.taxonomy_path))
    report = validate_dataset(data_path, names)
    if not report["valid"]:
        raise ValueError("Dataset piloto inválido: " + "; ".join(report["errors"]))
    root = Path(yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))["path"]).resolve()
    if (root / "INCOMPLETO.txt").exists():
        raise ValueError("Dataset piloto marcado como incompleto.")
    manifest = json.loads((root / "pilot.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != PILOT_KIND or manifest.get("taxonomy_sha256") != file_hash(config.taxonomy_path):
        raise ValueError("pilot.json ausente, de outro tipo ou de outra taxonomia.")
    registries = {}
    declared = set()
    for sample in manifest["images"]:
        image, label = root / sample["image"], root / sample["label"]
        if not image.resolve().is_relative_to(root) or not label.resolve().is_relative_to(root):
            raise ValueError("Amostra fora do dataset piloto.")
        if file_hash(image) != sample["image_sha256"] or file_hash(label) != sample["label_sha256"]:
            raise ValueError(f"Arquivo alterado após a criação do piloto: {sample['image']}.")
        registry_path = Path(sample["registry_path"])
        if registry_path not in registries:
            if file_hash(registry_path) != sample["registry_sha256"]:
                raise ValueError(f"Registro humano mudou: {registry_path.name}.")
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registries[registry_path] = {item["image_sha256"]: item for item in registry["images"]}
        reviewed = registries[registry_path].get(sample["image_sha256"])
        if not reviewed or reviewed.get("human_approved") is not True:
            raise ValueError("Amostra sem aprovação humana no registro de origem.")
        if label.read_text(encoding="utf-8") != boxes_to_yolo(reviewed["boxes"], reviewed["width"], reviewed["height"]):
            raise ValueError(f"Label diverge das caixas aprovadas: {sample['label']}.")
        declared.add(sample["image"])
    present = {path.relative_to(root).as_posix() for split in SPLITS
               for path in (root / "images" / split).rglob("*") if path.is_file()}
    if present != declared:
        raise ValueError("Inventário do piloto diverge do manifesto.")
    return {"valid": True, "kind": PILOT_KIND, "pilot_sha256": file_hash(root / "pilot.json"),
            "buildings": manifest["buildings"], "independent_evaluation": manifest["independent_evaluation"],
            "classes_without_training_boxes": manifest["classes_without_training_boxes"],
            "warning": PILOT_WARNING}
