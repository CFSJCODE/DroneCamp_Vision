"""Dataset piloto: monta, a partir das revisões humanas, o dataset do treino piloto.

Função no projeto: transforma registros de revisão (``data/reviews/*/registry.json``)
em um dataset YOLO (``data/pilot/<versão>/``) pronto para ``train-pilot``.

O que faz:
- ``_approved``: separa só as fotos com decisão humana completa e confere hashes.
- ``consolidate_duplicates``: junta caixas repetidas do mesmo objeto (um alvo só).
- ``split_by_scene``: divide treino/validação/teste por cena, sem misturar recortes.
- ``build_pilot_dataset``: copia fotos, escreve labels, ``dataset.yaml`` e ``pilot.json``.
- ``validate_pilot_dataset``: reconfere tudo antes de cada treino.

Por que piloto: o dataset de produção exige três edificações independentes.
Com só o CEASA, nenhuma métrica daqui mede generalização; os pesos treinados com
este dataset servem para sugerir caixas na revisão.

Quando mexer: proporção de validação/teste (``fractions``), regra de divisão por
cena ou critério de duplicatas. O limite de duplicatas fica em
``configs/project.yaml`` (``pilot.duplicate_iou``).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import csv
import json
from pathlib import Path
import re
import shutil

import yaml

from .config import ProjectConfig, detection_names, load_taxonomy
from .dataset import IMAGE_EXTENSIONS, SPLITS, validate_dataset
from .io import file_hash, resolve_local_path, write_json
from .review_data import boxes_to_yolo, review_image_size, validate_boxes

PILOT_KIND = "piloto_edificacao_unica"
PILOT_WARNING = ("Piloto com fotos de uma única edificação: validação e teste não são independentes. "
                 "Use os pesos apenas para sugerir caixas à revisão humana.")
# Nota que a página de revisão grava ao aceitar uma sugestão do modelo.
ACCEPTED_SUGGESTION_NOTE = "Sugestão do modelo aceita"


# ---------------------------------------------------------------------------
# Caixas duplicadas: IoU e consolidação (um objeto vira um único alvo de treino).
# ---------------------------------------------------------------------------

def box_iou(a: list[float], b: list[float]) -> float:
    """Interseção sobre união de duas caixas [x1, y1, x2, y2] (0 = disjuntas, 1 = iguais)."""
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def consolidate_duplicates(boxes: list[dict], iou_threshold: float | None) -> list[dict]:
    """Uma ocorrência, um alvo: caixas da mesma classe com IoU alto viram uma só.

    Aceitar uma sugestão sobre uma caixa já desenhada duplica o alvo e ensina
    previsões conflitantes. Mantém a caixa desenhada/editada pelo revisor; entre
    sugestões aceitas, a de maior confiança. O registro humano não é alterado.
    """
    if iou_threshold is None:
        return list(boxes)

    # Ordem de preferência: caixa do revisor > sugestão de maior confiança > ordem original.
    def priority(entry: tuple[int, dict]) -> tuple:
        position, box = entry
        note = box.get("note") or ""
        confidence = re.search(r"confiança (\d+)%", note)
        return (note.startswith(ACCEPTED_SUGGESTION_NOTE), -int(confidence.group(1)) if confidence else 0, position)

    # Mantém cada caixa que não repete (mesma classe e IoU alto) uma já mantida.
    kept: list[tuple[int, dict]] = []
    for position, box in sorted(enumerate(boxes), key=priority):
        if all(other["class_id"] != box["class_id"] or box_iou(other["bbox_xyxy"], box["bbox_xyxy"]) < iou_threshold
               for _, other in kept):
            kept.append((position, box))
    return [box for _, box in sorted(kept, key=lambda entry: entry[0])]


# ---------------------------------------------------------------------------
# Seleção das fotos aprovadas e divisão em treino/validação/teste.
# ---------------------------------------------------------------------------

def _approved(registry_path: Path, taxonomy_digest: str, class_count: int, root: Path) -> list[dict]:
    """Somente decisões humanas completas entram; o resto fica na fila de revisão."""
    # O registro precisa ser da taxonomia atual (mesmas 13 classes, mesmo hash).
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or registry.get("schema_version") != 1:
        raise ValueError(f"Schema inválido no registro {registry_path}.")
    if registry.get("taxonomy_sha256") != taxonomy_digest:
        raise ValueError(f"Taxonomia mudou desde {registry_path.name}; migre e revise antes do piloto.")
    selected = []
    for item in registry.get("images", []):
        # Fotos ambíguas ou sem decisão humana ficam fora do treino.
        if item.get("human_approved") is not True or item.get("status") not in {"positive", "negative"}:
            continue
        # Confere autor, foto original (SHA-256), dimensões e caixas.
        reviewer = item.get("human_reviewer")
        if item.get("technical_status") != "human_visual_reviewed" or not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("Aprovação sem autor ou estado de revisão humana explícito.")
        source = resolve_local_path(item["source_path"], root)
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
        selected.append({**item, "scene_group": scene, "source_path": str(source),
                         "registry_path": str(registry_path.resolve()),
                         "registry_sha256": file_hash(registry_path)})
    return selected


def split_by_scene(images: list[dict], seed: int, fractions: tuple[float, float]) -> dict[str, str]:
    """Distribua cenas inteiras; classes raras permanecem no treino.

    Uma cena só sai do treino se todas as suas classes continuarem com exemplos
    no treino. Por isso classes com uma única foto nunca são medidas.
    """
    # Agrupa as fotos por cena e embaralha as cenas de forma reprodutível (seed).
    scenes: dict[str, list[dict]] = defaultdict(list)
    for item in images:
        scenes[item["scene_group"]].append(item)
    order = sorted(scenes, key=lambda name: sha256(f"{seed}:{name}".encode("utf-8")).hexdigest())
    # Quantas fotos de treino cada classe tem; meta de fotos para validação e teste.
    train_images = Counter(box_class for item in images for box_class in {box["class_id"] for box in item["boxes"]})
    targets = {"val": max(1, round(len(images) * fractions[0])), "test": max(1, round(len(images) * fractions[1]))}
    counts = Counter()
    assignment = {}
    # Preenche validação e depois teste com cenas inteiras; o resto vai para treino.
    for split in ("val", "test"):
        for scene in order:
            if scene in assignment or counts[split] >= targets[split]:
                continue
            members = scenes[scene]
            # Não ultrapasse muito a meta: cenas grandes ficam no treino.
            if counts[split] + len(members) > targets[split] + max(2, targets[split] // 2):
                continue
            # Não tira do treino a última foto de nenhuma classe.
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


# ---------------------------------------------------------------------------
# Construção e validação do dataset piloto (data/pilot/<versão>/).
# ---------------------------------------------------------------------------

SEQUENCE_NAME = re.compile(r"^(?P<prefix>.*?)(?P<number>\d{3,})\.[A-Za-z0-9]+$")


def group_flight_sequences(images: list[dict], block: int) -> int:
    """Fotos sequenciais de um voo (ex.: DJI_0194…) viram blocos de ``block`` fotos.

    Fotos vizinhas de drone se sobrepõem: separadas por foto, uma cena quase igual
    cairia em treino e teste ao mesmo tempo. Só muda grupos automáticos (uma foto
    por grupo); cenas declaradas no registro são mantidas. Devolve quantas mudaram.
    """
    if block < 2:
        return 0
    changed = 0
    for item in images:
        scene, digest = item["scene_group"], item["image_sha256"][:12]
        automatic = scene == f"foto_{digest}" or scene.endswith(f"_{digest}")
        match = SEQUENCE_NAME.match(item.get("filename") or "")
        if automatic and match:
            building = item.get("building_group") or "sem_edificacao"
            item["scene_group"] = f"{building}_{match['prefix']}seq{int(match['number']) // block:04d}"
            changed += 1
    return changed


def build_pilot_dataset(config: ProjectConfig, registry_paths: list[Path], output: Path,
                        seed: int = 42, fractions: tuple[float, float] = (0.15, 0.15),
                        sequence_block: int = 0) -> dict:
    """Crie uma versão nova e imutável do dataset piloto a partir dos registros.

    ``fractions`` = (validação, teste): 15% das fotos para cada um.
    ``sequence_block``: agrupa fotos sequenciais de voo (ver ``group_flight_sequences``).
    """
    output = Path(output)
    if output.exists():
        raise ValueError("Saída já existe; datasets piloto também são imutáveis.")
    taxonomy = load_taxonomy(config.taxonomy_path)
    names = detection_names(taxonomy)
    digest = file_hash(config.taxonomy_path)
    # 1. Junta as fotos aprovadas de todos os registros informados (--registry).
    images, seen = [], set()
    for registry_path in registry_paths:
        for item in _approved(Path(registry_path), digest, len(names), config.root):
            # A mesma foto pode aparecer em revisões sucessivas: vale a mais recente.
            if item["image_sha256"] in seen:
                images = [value for value in images if value["image_sha256"] != item["image_sha256"]]
            seen.add(item["image_sha256"])
            images.append(item)
    sequence_grouped = group_flight_sequences(images, sequence_block)
    if not images:
        raise ValueError("Nenhuma foto com aprovação humana; nada para treinar.")
    # 2. Caixas de treino: duplicatas consolidadas conforme pilot.duplicate_iou.
    duplicate_iou = config.pilot.get("duplicate_iou")
    if duplicate_iou is not None and not 0 < float(duplicate_iou) <= 1:
        raise ValueError("pilot.duplicate_iou deve estar no intervalo (0, 1].")
    duplicate_iou = None if duplicate_iou is None else float(duplicate_iou)
    for item in images:
        item["training_boxes"] = consolidate_duplicates(item["boxes"], duplicate_iou)
    # 3. Divide as cenas em treino/validação/teste.
    assignment = split_by_scene(images, seed, fractions)
    output = output.resolve()
    output.mkdir(parents=True)
    samples = []
    try:
        # 4. Copia cada foto (nome = SHA-256), escreve a label YOLO e a linha do groups.csv.
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
                label.write_text(boxes_to_yolo(item["training_boxes"], item["width"], item["height"]), encoding="utf-8")
                relative = image.relative_to(output).as_posix()
                writer.writerow({"image": relative, "group_id": item["scene_group"], "split": split})
                samples.append({
                    "image": relative, "image_sha256": item["image_sha256"],
                    "label": label.relative_to(output).as_posix(), "label_sha256": file_hash(label),
                    "split": split, "scene_group": item["scene_group"], "building_group": item.get("building_group"),
                    "status": item["status"], "boxes": len(item["training_boxes"]),
                    "boxes_reviewed": len(item["boxes"]),
                    "duplicates_consolidated": len(item["boxes"]) - len(item["training_boxes"]),
                    "classes": sorted({box["class_id"] for box in item["boxes"]}),
                    "approval_author": item["human_reviewer"], "approved_at": item.get("human_review_at"),
                    "registry_path": item["registry_path"], "registry_sha256": item["registry_sha256"],
                    "source_path": str(source.resolve()),
                })
        # 5. dataset.yaml lido pela Ultralytics. path relativo ao YAML: o dataset
        # continua válido em outro clone ou máquina.
        (output / "dataset.yaml").write_text(yaml.safe_dump(
            {"path": ".", "train": "images/train", "val": "images/val", "test": "images/test",
             "nc": len(names), "names": names}, allow_unicode=True, sort_keys=False), encoding="utf-8")
        # 6. pilot.json: manifesto com hashes, origem, políticas e cobertura por classe.
        boxes_by_split = {split: Counter() for split in SPLITS}
        for item in images:
            boxes_by_split[assignment[item["scene_group"]]].update(box["class_id"] for box in item["training_boxes"])
        coverage = {name: {split: boxes_by_split[split][index] for split in SPLITS} for index, name in enumerate(names)}
        buildings = sorted({str(item.get("building_group")) for item in images})
        manifest = {
            "schema_version": 1, "kind": PILOT_KIND, "created_at": datetime.now(timezone.utc).isoformat(),
            "version": output.name, "taxonomy_sha256": digest, "seed": seed,
            "split_policy": "cena_inteira_por_split; classes raras mantidas no treino",
            "sequence_grouping": {"block": sequence_block, "images_regrouped": sequence_grouped} if sequence_block else None,
            "duplicate_policy": None if duplicate_iou is None else {
                "iou": duplicate_iou, "rule": "mesma classe e IoU >= limite viram uma caixa; prioridade: "
                "caixa desenhada/editada pelo revisor, depois sugestão aceita de maior confiança",
                "boxes_reviewed": sum(len(item["boxes"]) for item in images),
                "boxes_for_training": sum(len(item["training_boxes"]) for item in images)},
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
    # 7. Validação estrutural do que acabou de ser escrito.
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
    # 1. Estrutura do dataset e manifesto do tipo piloto, na taxonomia atual.
    data_path = Path(data_path).resolve()
    names = detection_names(load_taxonomy(config.taxonomy_path))
    report = validate_dataset(data_path, names)
    if not report["valid"]:
        raise ValueError("Dataset piloto inválido: " + "; ".join(report["errors"]))
    declared_root = Path(yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))["path"])
    root = (declared_root if declared_root.is_absolute() else data_path.parent / declared_root).resolve()
    if (root / "INCOMPLETO.txt").exists():
        raise ValueError("Dataset piloto marcado como incompleto.")
    manifest = json.loads((root / "pilot.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != PILOT_KIND or manifest.get("taxonomy_sha256") != file_hash(config.taxonomy_path):
        raise ValueError("pilot.json ausente, de outro tipo ou de outra taxonomia.")
    # 2. Cada foto/label: hash intacto, registro humano intacto e label recalculada igual.
    duplicate_iou = (manifest.get("duplicate_policy") or {}).get("iou")
    registries = {}
    declared = set()
    for sample in manifest["images"]:
        image, label = root / sample["image"], root / sample["label"]
        if not image.resolve().is_relative_to(root) or not label.resolve().is_relative_to(root):
            raise ValueError("Amostra fora do dataset piloto.")
        if file_hash(image) != sample["image_sha256"] or file_hash(label) != sample["label_sha256"]:
            raise ValueError(f"Arquivo alterado após a criação do piloto: {sample['image']}.")
        registry_path = resolve_local_path(sample["registry_path"], config.root)
        if registry_path not in registries:
            if file_hash(registry_path) != sample["registry_sha256"]:
                raise ValueError(f"Registro humano mudou: {registry_path.name}.")
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registries[registry_path] = {item["image_sha256"]: item for item in registry["images"]}
        reviewed = registries[registry_path].get(sample["image_sha256"])
        if not reviewed or reviewed.get("human_approved") is not True:
            raise ValueError("Amostra sem aprovação humana no registro de origem.")
        expected = consolidate_duplicates(reviewed["boxes"], duplicate_iou)
        if label.read_text(encoding="utf-8") != boxes_to_yolo(expected, reviewed["width"], reviewed["height"]):
            raise ValueError(f"Label diverge das caixas aprovadas: {sample['label']}.")
        declared.add(sample["image"])
    # 3. Nenhuma foto a mais ou a menos nas pastas além das listadas no manifesto.
    present = {path.relative_to(root).as_posix() for split in SPLITS
               for path in (root / "images" / split).rglob("*") if path.is_file()}
    if present != declared:
        raise ValueError("Inventário do piloto diverge do manifesto.")
    return {"valid": True, "kind": PILOT_KIND, "pilot_sha256": file_hash(root / "pilot.json"),
            "buildings": manifest["buildings"], "independent_evaluation": manifest["independent_evaluation"],
            "classes_without_training_boxes": manifest["classes_without_training_boxes"],
            "warning": PILOT_WARNING}
