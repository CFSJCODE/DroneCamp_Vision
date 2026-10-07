"""Fatiamento de imagens em alta resolução (Tiling / Patch Slicing) para YOLO.

Função no projeto: em inspeções aéreas por drone, imagens com alta resolução
(ex.: 5472 × 3648 pixels) contêm não conformidades diminutas (fissuras de telha,
parafusos frouxos, corrosão pontual) que ocupam poucos pixels. Redimensionar a foto
inteira diretamente para 640 ou 1280 comprimiria esses defeitos em ruídos de poucos pixels.

Este módulo implementa a estratégia industrial de Tiling / SAHI (Slicing Aided Hyper Inference):
- Divide a foto original de 20 MP em recortes deslizantes de dimensão uniforme (ex.: 1280 × 1280)
  com sobreposição configurável (ex.: 20% de overlap);
- Projeta geometricamente as anotações humanas (bounding boxes) para o sistema de coordenadas local
  de cada fatia, recortando ou descartando anotações que cruzam as bordas;
- Controla a taxa de fatias puramente negativas (sem defeito) para preservar o equilíbrio do dataset;
- Permite recombinação de predições via NMS global após inferência em fatias.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import shutil

from PIL import Image
import yaml

from .io import file_hash, write_json
from .review_data import boxes_to_yolo


def calculate_patch_windows(width: int, height: int, patch_size: int = 1280, overlap_ratio: float = 0.2) -> list[tuple[int, int, int, int]]:
    """Calcula janelas de corte (x1, y1, x2, y2) com sobreposição cobrindo 100% da imagem."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Dimensões inválidas: {width}x{height}")
    if patch_size <= 0:
        raise ValueError("patch_size deve ser maior que zero.")
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio deve estar entre 0.0 e 0.99.")

    # Se a foto for menor que o patch, retorna a própria imagem
    if width <= patch_size and height <= patch_size:
        return [(0, 0, width, height)]

    effective_patch_w = min(patch_size, width)
    effective_patch_h = min(patch_size, height)
    stride_x = max(1, int(effective_patch_w * (1.0 - overlap_ratio)))
    stride_y = max(1, int(effective_patch_h * (1.0 - overlap_ratio)))

    # Pontos de início nos eixos X e Y
    x_starts = []
    x = 0
    while x + effective_patch_w < width:
        x_starts.append(x)
        x += stride_x
    x_starts.append(width - effective_patch_w)
    x_starts = sorted(set(x_starts))

    y_starts = []
    y = 0
    while y + effective_patch_h < height:
        y_starts.append(y)
        y += stride_y
    y_starts.append(height - effective_patch_h)
    y_starts = sorted(set(y_starts))

    windows = []
    for ys in y_starts:
        for xs in x_starts:
            windows.append((xs, ys, xs + effective_patch_w, ys + effective_patch_h))
    return windows


def project_boxes_to_patch(boxes: list[dict], window: tuple[int, int, int, int], min_visibility: float = 0.3) -> list[dict]:
    """Projeta caixas delimitadoras absolutas [x1, y1, x2, y2] para o sistema de coordenadas local do patch.

    Se uma caixa for cortada pela margem do patch, a área visível é calculada.
    Se a proporção da área for >= min_visibility, a caixa é mantida recortada nos limites do patch.
    """
    wx1, wy1, wx2, wy2 = window
    projected = []

    for box in boxes:
        bx1, by1, bx2, by2 = box["bbox_xyxy"]
        box_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        if box_area <= 0:
            continue

        # Interseção entre a caixa e a janela do patch
        ix1 = max(bx1, wx1)
        iy1 = max(by1, wy1)
        ix2 = min(bx2, wx2)
        iy2 = min(by2, wy2)

        if ix2 <= ix1 or iy2 <= iy1:
            continue

        inter_area = (ix2 - ix1) * (iy2 - iy1)
        visibility = inter_area / box_area

        if visibility >= min_visibility:
            # Converte para coordenadas locais (relativas a wx1, wy1)
            lx1 = ix1 - wx1
            ly1 = iy1 - wy1
            lx2 = ix2 - wx1
            ly2 = iy2 - wy1
            projected.append({
                "class_id": box["class_id"],
                "bbox_xyxy": [round(lx1, 1), round(ly1, 1), round(lx2, 1), round(ly2, 1)],
                "original_box_id": box.get("id"),
                "visibility": round(visibility, 3)
            })

    return projected


def parse_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    """Lê arquivo de labels YOLO normalizado (class_id x_center y_center w h) e converte em xyxy absoluto."""
    if not label_path.is_file():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            cid = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            boxes.append({"class_id": cid, "bbox_xyxy": [x1, y1, x2, y2]})
    return boxes


def build_tiled_dataset(
    source_dataset_dir: Path,
    output_dir: Path,
    patch_size: int = 1280,
    overlap_ratio: float = 0.2,
    negative_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """Gera um dataset fatiado (patches 1280x1280) a partir de um dataset YOLO existente.

    Args:
        source_dataset_dir: Pasta do dataset original (contendo dataset.yaml e images/labels).
        output_dir: Pasta de destino onde o novo dataset fatiado será criado.
        patch_size: Tamanho do patch quadrado em pixels (padrão 1280).
        overlap_ratio: Fração de sobreposição entre patches vizinhos (padrão 0.20 = 20%).
        negative_ratio: Fração máxima permitida de patches vazios sem defeitos (padrão 0.20).
        seed: Semente para amostragem determinística dos patches negativos.
    """
    source_dataset_dir = Path(source_dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()

    yaml_file = source_dataset_dir / "dataset.yaml"
    if not yaml_file.is_file():
        raise FileNotFoundError(f"dataset.yaml não encontrado em {source_dataset_dir}")

    config_data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    names = config_data.get("names", {})
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]

    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "original_images": 0,
        "tiled_images": 0,
        "positive_patches": 0,
        "negative_patches": 0,
        "total_boxes_projected": 0,
        "by_split": {},
    }

    splits = [s for s in ("train", "val", "test") if (source_dataset_dir / "images" / s).is_dir()]
    if not splits:
        raise ValueError(f"Nenhuma subpasta train/val encontrada em {source_dataset_dir / 'images'}")

    for split in splits:
        img_src_dir = source_dataset_dir / "images" / split
        lbl_src_dir = source_dataset_dir / "labels" / split
        img_out_dir = output_dir / "images" / split
        lbl_out_dir = output_dir / "labels" / split
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted([p for p in img_src_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
        stats["original_images"] += len(image_files)

        split_positive_patches = []
        split_negative_patches = []

        for img_path in image_files:
            with Image.open(img_path) as im:
                w, h = im.size

            lbl_path = lbl_src_dir / f"{img_path.stem}.txt"
            original_boxes = parse_yolo_labels(lbl_path, w, h)
            windows = calculate_patch_windows(w, h, patch_size=patch_size, overlap_ratio=overlap_ratio)

            with Image.open(img_path) as im:
                for idx, window in enumerate(windows):
                    wx1, wy1, wx2, wy2 = window
                    patch_w = wx2 - wx1
                    patch_h = wy2 - wy1
                    patch_boxes = project_boxes_to_patch(original_boxes, window)

                    patch_name = f"{img_path.stem}_p{idx:02d}_{wx1}_{wy1}"
                    patch_img_dest = img_out_dir / f"{patch_name}{img_path.suffix.lower()}"
                    patch_lbl_dest = lbl_out_dir / f"{patch_name}.txt"

                    patch_info = {
                        "window": window,
                        "source": img_path,
                        "dest_img": patch_img_dest,
                        "dest_lbl": patch_lbl_dest,
                        "boxes": patch_boxes,
                        "patch_w": patch_w,
                        "patch_h": patch_h,
                    }

                    if patch_boxes:
                        split_positive_patches.append(patch_info)
                    else:
                        split_negative_patches.append(patch_info)

        # Controle de patches negativos para evitar diluição excessiva do dataset
        max_negatives = int(len(split_positive_patches) * negative_ratio) if split_positive_patches else len(split_negative_patches)
        selected_negatives = rng.sample(split_negative_patches, min(max_negatives, len(split_negative_patches)))

        selected_patches = split_positive_patches + selected_negatives
        rng.shuffle(selected_patches)

        split_boxes_count = 0
        # Agrupa por foto: cada JPEG de 20 MP é decodificado uma única vez (~120 ms cada).
        by_source: dict[Path, list[dict]] = {}
        for p in selected_patches:
            by_source.setdefault(p["source"], []).append(p)
        for source, patches in by_source.items():
            with Image.open(source) as im:
                im.load()
                for p in patches:
                    wx1, wy1, wx2, wy2 = p["window"]
                    im.crop((wx1, wy1, wx2, wy2)).save(p["dest_img"], quality=95)
                    # Salva o arquivo de rótulos YOLO
                    if p["boxes"]:
                        yolo_text = boxes_to_yolo(p["boxes"], p["patch_w"], p["patch_h"])
                        p["dest_lbl"].write_text(yolo_text, encoding="utf-8")
                        split_boxes_count += len(p["boxes"])
                    else:
                        p["dest_lbl"].touch()

        stats["by_split"][split] = {
            "positive_patches": len(split_positive_patches),
            "negative_patches_kept": len(selected_negatives),
            "total_patches": len(selected_patches),
            "boxes": split_boxes_count,
        }
        stats["tiled_images"] += len(selected_patches)
        stats["positive_patches"] += len(split_positive_patches)
        stats["negative_patches"] += len(selected_negatives)
        stats["total_boxes_projected"] += split_boxes_count

    # Grava dataset.yaml correspondente
    output_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }
    if "test" in splits:
        output_yaml["test"] = "images/test"
    (output_dir / "dataset.yaml").write_text(yaml.safe_dump(output_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source_dataset_dir),
        "patch_size": patch_size,
        "overlap_ratio": overlap_ratio,
        "negative_ratio": negative_ratio,
        "stats": stats,
    }
    write_json(output_dir / "tiling_manifest.json", manifest)
    return stats


def merge_tiled_predictions(predictions: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Recombina predições feitas em fatias locais de volta para o sistema global da imagem com NMS por classe.

    Args:
        predictions: Lista de dicionários com:
          - "bbox_xyxy": [x1, y1, x2, y2] nas coordenadas do patch
          - "score": confiança (float)
          - "class_id": ID da classe (int)
          - "window": [wx1, wy1, wx2, wy2] coordenadas do patch na foto original
        iou_threshold: Limiar de IoU para suprimir duplicatas em áreas de sobreposição.
    """
    if not predictions:
        return []

    # 1. Translação de cada caixa para coordenadas absolutas da foto global
    global_boxes = []
    for pred in predictions:
        bx1, by1, bx2, by2 = pred["bbox_xyxy"]
        wx1, wy1 = pred["window"][0], pred["window"][1]
        gx1 = bx1 + wx1
        gy1 = by1 + wy1
        gx2 = bx2 + wx1
        gy2 = by2 + wy1
        global_boxes.append({
            "class_id": pred["class_id"],
            "score": float(pred.get("score", 1.0)),
            "bbox_xyxy": [gx1, gy1, gx2, gy2],
        })

    # 2. Non-Maximum Suppression por classe
    def compute_iou(b1: list[float], b2: list[float]) -> float:
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2])
        iy2 = min(b1[3], b2[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    by_class: dict[int, list[dict]] = {}
    for box in global_boxes:
        by_class.setdefault(box["class_id"], []).append(box)

    merged = []
    for cid, items in by_class.items():
        # Ordena da maior para menor confiança
        items.sort(key=lambda x: x["score"], reverse=True)
        kept = []
        for candidate in items:
            if all(compute_iou(candidate["bbox_xyxy"], prev["bbox_xyxy"]) < iou_threshold for prev in kept):
                kept.append(candidate)
        merged.extend(kept)

    # Ordena resultado final por score descendente
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged
