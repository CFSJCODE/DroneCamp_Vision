"""Validação local e somente leitura de um dataset de detecção YOLO.

O conjunto só pode ser treinado depois de passar por esta revisão. Os caminhos
seguem a estrutura ``images/<split>`` / ``labels/<split>`` e ``groups.csv``
registra a unidade real de captura, para que frames do mesmo telhado, local ou
campanha não sejam distribuídos entre treino, validação e teste.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import re
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})
SPLITS = ("train", "val", "test")
BOX_TOLERANCE = 1e-6
CLASS_ID_PATTERN = re.compile(r"^[+-]?\d+$")


def _new_report() -> dict[str, Any]:
    """Cria uma resposta estável e compatível com JSON, inclusive em falhas."""
    return {
        "valid": False,
        "errors": [],
        "warnings": [],
        "counts": {
            split: {"images": 0, "labels": 0, "objects": 0, "negative_images": 0}
            for split in (*SPLITS, "total")
        },
    }


def _finish_report(report: dict[str, Any]) -> dict[str, Any]:
    for key in ("images", "labels", "objects", "negative_images"):
        report["counts"]["total"][key] = sum(report["counts"][split][key] for split in SPLITS)
    report["valid"] = not report["errors"]
    return report


def _inside(path: Path, directory: Path) -> bool:
    """Resolve caminhos para detectar também links que escapam do dataset."""
    return path.resolve().is_relative_to(directory.resolve())


def _relative_display(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _ultralytics_label_path(image_path: Path) -> Path:
    """Deriva a label como a biblioteca, sem importar seu runtime pesado.

    Ultralytics substitui o ÚLTIMO componente literal ``images`` no caminho,
    com comparação sensível a maiúsculas. Reproduzir esta regra impede que a
    revisão aprove uma label que o treino depois procuraria em outra pasta.
    """
    image_component = f"{os.sep}images{os.sep}"
    label_component = f"{os.sep}labels{os.sep}"
    label_text = label_component.join(os.fspath(image_path).rsplit(image_component, 1))
    return Path(label_text.rsplit(".", 1)[0] + ".txt")


def _read_names(value: Any, errors: list[str]) -> list[str] | None:
    """Aceita a lista e o mapa de IDs documentados pelo formato YOLO."""
    if isinstance(value, list):
        names = value
    elif isinstance(value, dict):
        normalized: dict[int, Any] = {}
        for key, name in value.items():
            if isinstance(key, bool) or not CLASS_ID_PATTERN.fullmatch(str(key)):
                errors.append("names deve usar IDs inteiros consecutivos iniciados em zero.")
                return None
            class_id = int(key)
            if class_id in normalized:
                errors.append("names contém IDs duplicados após normalização.")
                return None
            normalized[class_id] = name
        if sorted(normalized) != list(range(len(normalized))):
            errors.append("names deve usar IDs inteiros consecutivos iniciados em zero.")
            return None
        names = [normalized[index] for index in range(len(normalized))]
    else:
        errors.append("O YAML deve conter names como lista ou mapa de IDs.")
        return None
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        errors.append("names deve conter nomes de classe não vazios.")
        return None
    if len(set(names)) != len(names):
        errors.append("names contém nomes de classe duplicados.")
        return None
    return names


def _read_config(data_path: Path, report: dict[str, Any]) -> tuple[dict[str, Any], Path] | None:
    try:
        with data_path.open("r", encoding="utf-8-sig") as stream:
            config = yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        report["errors"].append(f"Não foi possível ler o YAML do dataset: {error}")
        return None
    if not isinstance(config, dict):
        report["errors"].append("O YAML do dataset deve conter um mapa de configuração.")
        return None
    root_value = config.get("path", ".")
    if not isinstance(root_value, str) or not root_value.strip() or "://" in root_value:
        report["errors"].append("path deve indicar um diretório local não vazio.")
        return None
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = data_path.parent / root
    root = root.resolve()
    if not root.is_dir():
        report["errors"].append(f"A raiz do dataset não existe ou não é diretório: {root}")
        return None
    if "download" in config:
        report["warnings"].append("O campo download do YAML foi ignorado: esta validação não baixa nem executa conteúdo.")
    return config, root


def _split_directories(config: dict[str, Any], root: Path, errors: list[str]) -> dict[str, Path]:
    directories: dict[str, Path] = {}
    images_root = (root / "images").resolve()
    for split in SPLITS:
        value = config.get(split)
        if not isinstance(value, str) or not value.strip() or "://" in value:
            errors.append(f"{split}: informe um diretório local simples; listas, URLs e arquivos de lista não são suportados.")
            continue
        directory = Path(value).expanduser()
        if not directory.is_absolute():
            directory = root / directory
        directory = directory.resolve()
        if not directory.is_dir():
            errors.append(f"{split}: o diretório de imagens não existe: {directory}")
            continue
        if not _inside(directory, images_root) or directory == images_root:
            errors.append(f"{split}: o diretório deve ficar dentro de path/images, em uma subpasta independente.")
            continue
        directories[split] = directory
    # Pastas iguais ou aninhadas compartilham imagens mesmo sem arquivos iguais.
    for position, left_split in enumerate(SPLITS):
        left = directories.get(left_split)
        if left is None:
            continue
        for right_split in SPLITS[position + 1 :]:
            right = directories.get(right_split)
            if right is not None and (_inside(left, right) or _inside(right, left)):
                errors.append(f"Splits {left_split} e {right_split} usam diretórios iguais ou sobrepostos.")
    return directories


def _validate_label(label_path: Path, display: str, class_count: int, errors: list[str]) -> tuple[int, bool]:
    """Retorna objetos válidos e se o arquivo registra um negativo explícito."""
    try:
        contents = label_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        errors.append(f"{display}: não foi possível ler a anotação: {error}")
        return 0, False
    if contents.startswith("\ufeff"):
        # Não remova o BOM silenciosamente: o loader YOLO usa UTF-8 comum.
        errors.append(f"{display}: label com BOM UTF-8 não é compatível com o loader YOLO; salve UTF-8 sem BOM após revisão.")
        return 0, False
    if not contents.strip():
        return 0, True
    valid_objects = 0
    for line_number, line in enumerate(contents.splitlines(), start=1):
        if not line.strip():
            continue
        prefix = f"{display}, linha {line_number}"
        values = line.split()
        if len(values) != 5:
            errors.append(f"{prefix}: esperado class_id x_center y_center width height (5 valores).")
            continue
        if not CLASS_ID_PATTERN.fullmatch(values[0]):
            errors.append(f"{prefix}: class_id deve ser inteiro.")
            continue
        class_id = int(values[0])
        if not 0 <= class_id < class_count:
            errors.append(f"{prefix}: class_id {class_id} está fora dos IDs definidos em names.")
            continue
        try:
            x_center, y_center, width, height = map(float, values[1:])
        except ValueError:
            errors.append(f"{prefix}: coordenadas devem ser números normalizados.")
            continue
        coordinates = (x_center, y_center, width, height)
        if not all(math.isfinite(number) for number in coordinates):
            errors.append(f"{prefix}: coordenadas devem ser finitas; NaN e infinito não são aceitos.")
            continue
        if not all(-BOX_TOLERANCE <= number <= 1 + BOX_TOLERANCE for number in coordinates):
            errors.append(f"{prefix}: coordenadas devem estar normalizadas no intervalo [0, 1].")
            continue
        if width <= 0 or height <= 0:
            errors.append(f"{prefix}: largura e altura devem ser maiores que zero.")
            continue
        edges = (x_center - width / 2, y_center - height / 2, x_center + width / 2, y_center + height / 2)
        if edges[0] < -BOX_TOLERANCE or edges[1] < -BOX_TOLERANCE or edges[2] > 1 + BOX_TOLERANCE or edges[3] > 1 + BOX_TOLERANCE:
            errors.append(f"{prefix}: a caixa ultrapassa os limites da imagem.")
            continue
        valid_objects += 1
    return valid_objects, False


def _image_digest(image_path: Path, display: str, errors: list[str]) -> str | None:
    try:
        with Image.open(image_path) as image:
            if image.width < 10 or image.height < 10:
                raise ValueError("largura e altura devem ter pelo menos 10 pixels, conforme o loader YOLO")
            if image.format in {"JPEG", "MPO"}:
                with image_path.open("rb") as stream:
                    stream.seek(-2, os.SEEK_END)
                    if stream.read() != b"\xff\xd9":
                        # A biblioteca repararia e substituiria esse original.
                        raise ValueError("JPEG sem marcador EOI final; revise a imagem antes do treino para impedir reparo automático do original")
            image.verify()
        # load() decodifica de fato os pixels; só abrir o cabeçalho não basta.
        with Image.open(image_path) as image:
            image.load()
        digest = hashlib.sha256()
        with image_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        errors.append(f"{display}: imagem ilegível ou inválida: {error}")
        return None


def _inspect_split(split: str, directory: Path, root: Path, class_count: int, report: dict[str, Any], hashes: dict[str, tuple[str, str]]) -> dict[str, str]:
    """Revisa imagens e suas labels, preservando o vínculo usado no manifesto."""
    errors = report["errors"]
    counts = report["counts"][split]
    image_root = (root / "images").resolve()
    label_root = (root / "labels").resolve()
    label_directory = label_root / directory.relative_to(image_root)
    images = sorted((path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS), key=lambda path: path.as_posix().casefold())
    counts["images"] = len(images)
    if not images:
        errors.append(f"{split}: nenhum arquivo de imagem suportado foi encontrado.")
    stem_paths: dict[str, Path] = {}
    expected_labels: set[Path] = set()
    used_images: dict[str, str] = {}
    for image_path in images:
        display = _relative_display(image_path, root)
        if not _inside(image_path, image_root):
            errors.append(f"{display}: o arquivo resolve para fora de path/images.")
            continue
        used_images[display] = split
        # Duas extensões com o mesmo stem apontariam para a mesma anotação.
        relative_image = image_path.relative_to(directory)
        stem_key = relative_image.with_suffix("").as_posix().casefold()
        if stem_key in stem_paths:
            errors.append(f"{split}: colisão de stem entre {_relative_display(stem_paths[stem_key], root)} e {display}; as imagens compartilhariam uma label.")
        else:
            stem_paths[stem_key] = image_path
        digest = _image_digest(image_path, display, errors)
        if digest is not None:
            if digest in hashes:
                previous_display, previous_split = hashes[digest]
                kind = "vazamento entre splits" if split != previous_split else "duplicação dentro do split"
                errors.append(f"{display}: {kind}; conteúdo idêntico a {previous_display} ({previous_split}).")
            else:
                hashes[digest] = (display, split)
        label_path = label_directory / relative_image.with_suffix(".txt")
        expected_labels.add(label_path.resolve())
        derived_label = _ultralytics_label_path(image_path)
        if derived_label.resolve() != label_path.resolve():
            errors.append(f"{display}: caminho de label incompatível com Ultralytics; a biblioteca procuraria {_relative_display(derived_label, root)}, mas o dataset registra {_relative_display(label_path, root)}. Use a pasta literal images e evite subpastas images aninhadas.")
            continue
        if not _inside(label_path, label_root):
            errors.append(f"{display}: a label resolve para fora de path/labels.")
            continue
        if not label_path.is_file():
            errors.append(f"{display}: label ausente em {_relative_display(label_path, root)}; negativos exigem um .txt vazio explícito.")
            continue
        counts["labels"] += 1
        objects, negative = _validate_label(label_path, _relative_display(label_path, root), class_count, errors)
        counts["objects"] += objects
        counts["negative_images"] += int(negative)
    if label_directory.is_dir():
        for label_path in sorted(label_directory.rglob("*")):
            if label_path.is_file() and label_path.suffix.lower() == ".txt" and label_path.resolve() not in expected_labels:
                errors.append(f"{split}: label órfã, sem imagem correspondente: {_relative_display(label_path, root)}.")
    if images and counts["negative_images"] == len(images):
        report["warnings"].append(f"{split}: todas as imagens são negativos explícitos; não há objetos anotados neste split.")
    return used_images


def _validate_groups(root: Path, used_images: dict[str, str], errors: list[str]) -> None:
    """Impede vazamento por unidade de captura, mesmo com frames diferentes."""
    manifest = root / "groups.csv"
    if not manifest.is_file():
        errors.append("groups.csv obrigatório ausente na raiz do dataset (colunas image,group_id,split).")
        return
    seen_images: set[str] = set()
    group_splits: dict[str, str] = {}
    try:
        with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not {"image", "group_id", "split"}.issubset(reader.fieldnames):
                errors.append("groups.csv deve conter as colunas image,group_id,split.")
                return
            for row_number, row in enumerate(reader, start=2):
                prefix = f"groups.csv, linha {row_number}"
                image_value = (row.get("image") or "").strip()
                group_id = (row.get("group_id") or "").strip()
                split = (row.get("split") or "").strip()
                if not image_value or not group_id or split not in SPLITS:
                    errors.append(f"{prefix}: image e group_id devem estar preenchidos e split deve ser train, val ou test.")
                    continue
                image_path = Path(image_value.replace("\\", "/"))
                if image_path.is_absolute() or not _inside(root / image_path, root):
                    errors.append(f"{prefix}: image deve ser um caminho relativo dentro da raiz do dataset.")
                    continue
                image_key = _relative_display((root / image_path).resolve(), root)
                if image_key in seen_images:
                    errors.append(f"{prefix}: imagem duplicada no manifesto: {image_key}.")
                seen_images.add(image_key)
                if image_key not in used_images:
                    errors.append(f"{prefix}: imagem não usada pelos splits configurados: {image_key}.")
                elif used_images[image_key] != split:
                    errors.append(f"{prefix}: split de {image_key} diverge do diretório configurado.")
                previous_split = group_splits.setdefault(group_id, split)
                if previous_split != split:
                    errors.append(f"{prefix}: vazamento por grupo {group_id!r} entre {previous_split} e {split}.")
    except (OSError, UnicodeError, csv.Error) as error:
        errors.append(f"Não foi possível ler groups.csv: {error}")
        return
    for image_key in sorted(set(used_images) - seen_images):
        errors.append(f"groups.csv: imagem sem grupo de captura: {image_key}.")


def validate_dataset(data_path: Path, expected_names: list[str]) -> dict[str, Any]:
    """Valida um YAML YOLO de detecção sem modificar arquivos ou acessar rede.

    ``expected_names`` é a taxonomia revisada do projeto: sua ordem define os
    IDs e não pode mudar silenciosamente entre anotação, treino e inferência.
    Cada split deve apontar para uma pasta independente sob ``path/images``.
    Labels usam o mesmo caminho relativo sob ``path/labels``. ``groups.csv`` é
    obrigatório e seu campo ``image`` é relativo à raiz ``path``.
    A derivação de labels também deve coincidir com a regra da Ultralytics:
    último componente literal ``images`` substituído por ``labels``. Labels
    precisam usar UTF-8 sem BOM; imagens exigem pelo menos 10 pixels em cada
    dimensão e JPEG com marcador EOI final, sem reparo automático do original.

    O manifesto depende de curadoria humana: nomes de grupo devem representar
    telhado, local ou campanha que precisam permanecer juntos, nunca frames
    individuais. Esta função verifica a consistência, não prova a origem real.
    """
    report = _new_report()
    if not expected_names or any(not isinstance(name, str) or not name.strip() for name in expected_names) or len(set(expected_names)) != len(expected_names):
        report["errors"].append("expected_names deve conter nomes de classe únicos e não vazios.")
        return _finish_report(report)
    config_result = _read_config(Path(data_path).resolve(), report)
    if config_result is None:
        return _finish_report(report)
    config, root = config_result
    names = _read_names(config.get("names"), report["errors"])
    if names is not None and names != expected_names:
        report["errors"].append("Os nomes e a ordem de IDs em names divergem da taxonomia esperada; revise a anotação antes do treino.")
    if "nc" in config and (type(config["nc"]) is not int or config["nc"] != len(expected_names)):
        report["errors"].append("nc diverge da quantidade de classes esperadas.")
    directories = _split_directories(config, root, report["errors"])
    hashes: dict[str, tuple[str, str]] = {}
    used_images: dict[str, str] = {}
    for split, directory in directories.items():
        used_images.update(_inspect_split(split, directory, root, len(expected_names), report, hashes))
    _validate_groups(root, used_images, report["errors"])
    return _finish_report(report)
