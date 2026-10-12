"""Validação local e somente leitura de um dataset de detecção YOLO.

Função no projeto: é o "porteiro" do treino. Todo dataset (piloto ou produção)
passa por ``validate_dataset`` antes de chegar à Ultralytics.

O que faz:
- Lê o ``dataset.yaml`` e confere as pastas ``images/<split>`` e ``labels/<split>``.
- Confere cada label YOLO: classe válida, 5 números, caixa dentro da foto.
- Detecta fotos duplicadas entre splits (vazamento) e labels órfãs.
- Exige ``groups.csv``: cada foto tem um grupo de captura, e um grupo nunca
  fica em dois splits (frames do mesmo telhado/local não se misturam).
- Devolve um relatório JSON com erros, avisos e contagens.

Quando mexer: para aceitar outro formato de imagem (``IMAGE_EXTENSIONS``) ou
adicionar uma checagem nova no dataset. Não muda nada no disco.

Conteúdo do arquivo:
- Constantes: ``IMAGE_EXTENSIONS``, ``SPLITS``, ``BOX_TOLERANCE``, ``CLASS_ID_PATTERN``
  (as duas primeiras são importadas por outros módulos).
- Função pública: ``validate_dataset(data_path, expected_names)``.
- Funções privadas, na ordem em que aparecem: ``_new_report``, ``_finish_report``,
  ``_inside``, ``_relative_display``, ``_ultralytics_label_path`` (relatório e caminhos);
  ``_read_names``, ``_read_config``, ``_split_directories`` (leitura do YAML);
  ``_validate_label``, ``_image_digest``, ``_inspect_split`` (labels e fotos);
  ``_validate_groups`` (groups.csv).

Quem chama:
- ``cli.py``: comando ``python -m dronecamp_ia validate-data`` (código de saída 2 se inválido).
- ``training.py``: antes de ``train``/``evaluate``/``tune`` (produção).
- ``review_dataset.py`` e ``review_provenance.py``: ao gerar/conferir datasets de produção.
- ``IMAGE_EXTENSIONS`` também é usado por ``ingest.py``, ``review_dataset.py``,
  ``review_provenance.py`` e ``cli.py`` (verificação de paridade).

Disco: SÓ LÊ (``dataset.yaml``, ``groups.csv``, ``images/**``, ``labels/**``).
Não grava nada; quem grava o relatório em JSON é o chamador (ex.: ``--output`` da CLI).

Formato YOLO conferido aqui (uma linha por objeto em ``labels/<split>/<foto>.txt``)::

    class_id x_centro y_centro largura altura

Os quatro números são NORMALIZADOS pelo tamanho da foto (0 a 1): x_centro e
largura divididos pela largura em pixels, y_centro e altura pela altura em pixels.
Um ``.txt`` vazio significa "foto negativa" (sem defeito) e é aceito de propósito.

Onde mexer:
- Aceitar um formato de foto novo (ex.: ``.heic``): acrescente a extensão em
  ``IMAGE_EXTENSIONS`` (vale para o projeto todo; confira se o Pillow abre o formato).
- Acrescentar uma checagem por linha de label (ex.: caixa mínima): em ``_validate_label``,
  no ponto marcado ``EXTENSAO:`` antes de ``valid_objects += 1``.
- Acrescentar uma checagem por foto (ex.: resolução mínima maior): em ``_image_digest``.
- Acrescentar uma checagem do dataset inteiro: em ``validate_dataset``, depois da etapa 4.
- Acrescentar uma contagem nova no relatório: em ``_new_report`` (chave) e em
  ``_finish_report`` (soma do total), e incremente em ``_inspect_split``.
- Acrescentar uma coluna obrigatória no groups.csv: em ``_validate_groups``.
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

# ---------------------------------------------------------------------------
# Constantes do módulo
# ---------------------------------------------------------------------------

# Formatos de foto aceitos, nomes dos splits e tolerância numérica das caixas.
# EXTENSAO: formato de imagem aceito em todo o projeto (dataset, export, importação de mídia).
# IMAGE_EXTENSIONS: sufixos em minúsculas (a comparação usa ``suffix.lower()``).
# Acrescentar um sufixo aqui faz a validação, a importação (ingest.py) e a
# geração de datasets (review_dataset.py) aceitarem o formato ao mesmo tempo.
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})
# SPLITS: nomes e ORDEM dos splits. A ordem importa: o hash da primeira foto
# vista fica como "original" e as repetições posteriores são acusadas.
# ATENCAO: estes nomes são as chaves do dataset.yaml e da coluna split do groups.csv;
# trocar ou acrescentar um nome exige mudar quem gera datasets (pilot.py, review_dataset.py).
SPLITS = ("train", "val", "test")
# BOX_TOLERANCE: folga numérica (adimensional, em unidades normalizadas 0..1)
# para aceitar arredondamentos como 1.0000001. Aumentar deixa passar caixas que
# saem levemente da foto; zerar rejeita erros de ponto flutuante legítimos.
BOX_TOLERANCE = 1e-6
# CLASS_ID_PATTERN: inteiro com sinal opcional, sem ponto decimal ("3", "+3", "-1").
# O sinal é aceito aqui só para a mensagem de erro seguinte ("fora dos IDs") ser mais clara.
CLASS_ID_PATTERN = re.compile(r"^[+-]?\d+$")


# ---------------------------------------------------------------------------
# Relatório e utilidades de caminho.
# ---------------------------------------------------------------------------

def _new_report() -> dict[str, Any]:
    """Cria uma resposta estável e compatível com JSON, inclusive em falhas.

    Parâmetros: nenhum.

    Retorna:
        dict com ``valid`` (bool, começa False), ``errors`` (lista de str),
        ``warnings`` (lista de str) e ``counts``: um dict por split
        (``train``, ``val``, ``test``) e mais ``total``, cada um com
        ``images``, ``labels``, ``objects`` e ``negative_images`` (int).

    EXTENSAO: contagem nova no relatório: acrescente a chave neste dict interno,
    some em ``_finish_report`` (tupla de chaves) e incremente em ``_inspect_split``.
    """
    return {
        "valid": False,
        "errors": [],
        "warnings": [],
        # Um contador zerado para cada split e um para o total (somado no fim).
        "counts": {
            split: {"images": 0, "labels": 0, "objects": 0, "negative_images": 0}
            for split in (*SPLITS, "total")
        },
    }


def _finish_report(report: dict[str, Any]) -> dict[str, Any]:
    """Fecha o relatório: soma os totais e decide se o dataset é válido.

    Parâmetros:
        report: o dict criado por ``_new_report`` e preenchido pelas etapas.

    Retorna:
        o mesmo dict (alterado no lugar), com ``counts["total"]`` somado e
        ``valid`` = True somente se não houver nenhum erro. Avisos não invalidam.

    Chamada por: ``validate_dataset`` em todos os pontos de retorno.
    """
    # Total = soma dos três splits para cada contador.
    # EXTENSAO: ao criar uma contagem nova em _new_report, acrescente a chave nesta tupla.
    for key in ("images", "labels", "objects", "negative_images"):
        report["counts"]["total"][key] = sum(report["counts"][split][key] for split in SPLITS)
    # ATENCAO: a regra de validade é "nenhum erro"; uma checagem que deve só alertar
    # tem de ir para report["warnings"], nunca para report["errors"].
    report["valid"] = not report["errors"]
    return report


def _inside(path: Path, directory: Path) -> bool:
    """Resolve caminhos para detectar também links que escapam do dataset.

    Parâmetros:
        path: arquivo ou pasta a testar.
        directory: pasta que deveria conter ``path``.

    Retorna:
        True se ``path``, depois de resolver links simbólicos e ``..``, fica dentro
        de ``directory`` (ou é a própria pasta); False caso contrário.
    """
    return path.resolve().is_relative_to(directory.resolve())


def _relative_display(path: Path, root: Path) -> str:
    """Texto curto de um caminho para mensagens do relatório.

    Parâmetros:
        path: caminho a exibir.
        root: raiz do dataset.

    Retorna:
        o caminho relativo à raiz com barras ``/`` (ex.: ``images/train/a.jpg``);
        se ``path`` não estiver sob ``root``, devolve o caminho completo como texto.

    ATENCAO: este texto também é usado como CHAVE de foto (``used_images`` e o
    confronto com a coluna image do groups.csv), não só para exibição.
    """
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        # Fora da raiz: relative_to falha; mostra o caminho absoluto.
        return str(path)


def _ultralytics_label_path(image_path: Path) -> Path:
    """Deriva a label como a biblioteca, sem importar seu runtime pesado.

    Ultralytics substitui o ÚLTIMO componente literal ``images`` no caminho,
    com comparação sensível a maiúsculas. Reproduzir esta regra impede que a
    revisão aprove uma label que o treino depois procuraria em outra pasta.

    Parâmetros:
        image_path: caminho da foto (ex.: ``.../ds/images/train/a.jpg``).

    Retorna:
        caminho da label que a Ultralytics procuraria
        (ex.: ``.../ds/labels/train/a.txt``).

    ATENCAO: se a Ultralytics mudar essa regra, atualize aqui; senão o validador
    aprova datasets que o treino lê sem labels (todas as fotos viram negativas).
    """
    # Componentes "/images/" e "/labels/" com o separador do sistema (\ no Windows).
    image_component = f"{os.sep}images{os.sep}"
    label_component = f"{os.sep}labels{os.sep}"
    # rsplit(..., 1) corta só na ÚLTIMA ocorrência de "/images/"; o join recola com "/labels/".
    label_text = label_component.join(os.fspath(image_path).rsplit(image_component, 1))
    # Troca a extensão da foto (o que vem depois do último ponto) por ".txt".
    return Path(label_text.rsplit(".", 1)[0] + ".txt")


# ---------------------------------------------------------------------------
# Leitura do dataset.yaml: classes, raiz e pastas de cada split.
# ---------------------------------------------------------------------------

def _read_names(value: Any, errors: list[str]) -> list[str] | None:
    """Aceita a lista e o mapa de IDs documentados pelo formato YOLO.

    Parâmetros:
        value: o campo ``names`` do YAML, como lista (``["a", "b"]``) ou como mapa
            de ID para nome (``{0: "a", 1: "b"}``; chaves texto "0" também valem).
        errors: lista de erros do relatório; recebe a mensagem em caso de falha.

    Retorna:
        a lista de nomes na ordem dos IDs (posição = class_id), ou None quando
        o campo é inválido (a mensagem já foi acrescentada a ``errors``).
    """
    # Formato 1: lista simples; a posição já é o ID.
    if isinstance(value, list):
        names = value
    # Formato 2: mapa ID -> nome; normaliza para uma lista ordenada por ID.
    elif isinstance(value, dict):
        normalized: dict[int, Any] = {}
        for key, name in value.items():
            # bool é subclasse de int em Python (True == 1); por isso é recusado à parte.
            if isinstance(key, bool) or not CLASS_ID_PATTERN.fullmatch(str(key)):
                errors.append("names deve usar IDs inteiros consecutivos iniciados em zero.")
                return None
            class_id = int(key)
            # "1" e 1 viram o mesmo ID depois de int(); isso seria ambíguo.
            if class_id in normalized:
                errors.append("names contém IDs duplicados após normalização.")
                return None
            normalized[class_id] = name
        # IDs precisam ser exatamente 0, 1, 2, ..., n-1 (sem buracos nem negativos).
        if sorted(normalized) != list(range(len(normalized))):
            errors.append("names deve usar IDs inteiros consecutivos iniciados em zero.")
            return None
        names = [normalized[index] for index in range(len(normalized))]
    else:
        errors.append("O YAML deve conter names como lista ou mapa de IDs.")
        return None
    # Cada nome precisa ser texto não vazio.
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        errors.append("names deve conter nomes de classe não vazios.")
        return None
    # Nomes repetidos tornariam ambígua a ligação nome <-> ID.
    if len(set(names)) != len(names):
        errors.append("names contém nomes de classe duplicados.")
        return None
    return names


def _read_config(data_path: Path, report: dict[str, Any]) -> tuple[dict[str, Any], Path] | None:
    """Lê o ``dataset.yaml`` e resolve a pasta raiz (campo ``path``).

    Parâmetros:
        data_path: caminho absoluto do ``dataset.yaml``.
        report: relatório em construção (recebe erros e avisos).

    Retorna:
        tupla ``(config, root)``: o YAML como dict e a raiz absoluta resolvida;
        ou None se o YAML for ilegível, não for um mapa, ou ``path`` for inválido
        ou não existir (com o erro registrado em ``report["errors"]``).

    Chamada por: ``validate_dataset`` (etapa 1).
    """
    # 1. Leitura do YAML. "utf-8-sig" tolera BOM no YAML (comum em arquivos salvos no Windows).
    try:
        with data_path.open("r", encoding="utf-8-sig") as stream:
            config = yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        report["errors"].append(f"Não foi possível ler o YAML do dataset: {error}")
        return None
    # 2. O conteúdo tem de ser um mapa (chave: valor), não lista nem texto solto.
    if not isinstance(config, dict):
        report["errors"].append("O YAML do dataset deve conter um mapa de configuração.")
        return None
    # 3. Campo path: raiz local do dataset. URLs ("://") são recusadas (nada de rede).
    root_value = config.get("path", ".")
    if not isinstance(root_value, str) or not root_value.strip() or "://" in root_value:
        report["errors"].append("path deve indicar um diretório local não vazio.")
        return None
    # 4. path relativo é resolvido a partir da pasta do próprio YAML (como na Ultralytics).
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = data_path.parent / root
    root = root.resolve()
    if not root.is_dir():
        report["errors"].append(f"A raiz do dataset não existe ou não é diretório: {root}")
        return None
    # 5. "download" é um script/URL que a Ultralytics executaria; aqui é só avisado e ignorado.
    if "download" in config:
        report["warnings"].append("O campo download do YAML foi ignorado: esta validação não baixa nem executa conteúdo.")
    return config, root


def _split_directories(config: dict[str, Any], root: Path, errors: list[str]) -> dict[str, Path]:
    """Resolve e confere a pasta de imagens de cada split (``train``, ``val``, ``test``).

    Parâmetros:
        config: o YAML lido por ``_read_config``.
        root: raiz absoluta do dataset.
        errors: lista de erros do relatório.

    Retorna:
        dict ``{split: pasta absoluta}`` só com os splits válidos. Um split
        inválido fica de fora (e gera erro), então o restante ainda é inspecionado.

    Regras: cada split é UMA pasta local (não lista, nem URL, nem arquivo .txt
    com lista de fotos), existe, fica dentro de ``root/images`` mas não é a
    própria ``images``, e nenhum split se sobrepõe a outro.
    """
    directories: dict[str, Path] = {}
    images_root = (root / "images").resolve()
    # 1. Confere cada split isoladamente.
    for split in SPLITS:
        value = config.get(split)
        # ATENCAO: listas de pastas e arquivos de lista (formatos aceitos pela Ultralytics)
        # são recusados de propósito: dificultariam provar que não há vazamento.
        if not isinstance(value, str) or not value.strip() or "://" in value:
            errors.append(f"{split}: informe um diretório local simples; listas, URLs e arquivos de lista não são suportados.")
            continue
        # Caminho relativo é relativo à raiz (campo path), como na Ultralytics.
        directory = Path(value).expanduser()
        if not directory.is_absolute():
            directory = root / directory
        directory = directory.resolve()
        if not directory.is_dir():
            errors.append(f"{split}: o diretório de imagens não existe: {directory}")
            continue
        # Precisa estar sob images/ e ser uma subpasta (images/train), não a própria images/.
        if not _inside(directory, images_root) or directory == images_root:
            errors.append(f"{split}: o diretório deve ficar dentro de path/images, em uma subpasta independente.")
            continue
        directories[split] = directory
    # Pastas iguais ou aninhadas compartilham imagens mesmo sem arquivos iguais.
    # 2. Compara todos os pares (train×val, train×test, val×test) uma vez cada.
    for position, left_split in enumerate(SPLITS):
        left = directories.get(left_split)
        if left is None:
            continue
        for right_split in SPLITS[position + 1 :]:
            right = directories.get(right_split)
            if right is not None and (_inside(left, right) or _inside(right, left)):
                errors.append(f"Splits {left_split} e {right_split} usam diretórios iguais ou sobrepostos.")
    return directories


# ---------------------------------------------------------------------------
# Conferência de cada label, foto e split.
# ---------------------------------------------------------------------------

def _validate_label(label_path: Path, display: str, class_count: int, errors: list[str]) -> tuple[int, bool]:
    """Retorna objetos válidos e se o arquivo registra um negativo explícito.

    Parâmetros:
        label_path: arquivo ``.txt`` YOLO da foto.
        display: caminho relativo para as mensagens (ex.: ``labels/train/a.txt``).
        class_count: quantidade de classes; IDs válidos vão de 0 a ``class_count - 1``.
        errors: lista de erros do relatório (um erro por linha inválida).

    Retorna:
        tupla ``(objetos_validos, negativo)``: quantas linhas passaram em todas as
        checagens, e True se o arquivo está vazio (foto sem defeito, "negativo").

    Cada linha deve ter ``class_id x_centro y_centro largura altura`` com as
    coordenadas normalizadas 0..1 (ver cabeçalho do módulo).
    """
    # 1. Leitura do arquivo em UTF-8 comum.
    try:
        contents = label_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        errors.append(f"{display}: não foi possível ler a anotação: {error}")
        return 0, False
    # 2. BOM (﻿ no começo) quebraria a leitura do primeiro class_id na Ultralytics.
    if contents.startswith("﻿"):
        # Não remova o BOM silenciosamente: o loader YOLO usa UTF-8 comum.
        errors.append(f"{display}: label com BOM UTF-8 não é compatível com o loader YOLO; salve UTF-8 sem BOM após revisão.")
        return 0, False
    # 3. Arquivo vazio (ou só espaços) = negativo explícito, válido.
    if not contents.strip():
        return 0, True
    # 4. Confere linha a linha; uma linha ruim gera erro e passa para a próxima.
    valid_objects = 0
    for line_number, line in enumerate(contents.splitlines(), start=1):
        if not line.strip():
            continue
        prefix = f"{display}, linha {line_number}"
        values = line.split()
        # 4a. Exatamente 5 campos (formatos com segmentação/pontos não são aceitos).
        if len(values) != 5:
            errors.append(f"{prefix}: esperado class_id x_center y_center width height (5 valores).")
            continue
        # 4b. class_id inteiro ("1.0" é recusado) e dentro do intervalo das classes.
        if not CLASS_ID_PATTERN.fullmatch(values[0]):
            errors.append(f"{prefix}: class_id deve ser inteiro.")
            continue
        class_id = int(values[0])
        if not 0 <= class_id < class_count:
            errors.append(f"{prefix}: class_id {class_id} está fora dos IDs definidos em names.")
            continue
        # 4c. As quatro coordenadas precisam ser números.
        try:
            x_center, y_center, width, height = map(float, values[1:])
        except ValueError:
            errors.append(f"{prefix}: coordenadas devem ser números normalizados.")
            continue
        coordinates = (x_center, y_center, width, height)
        # 4d. float() aceita "nan" e "inf"; aqui eles são recusados.
        if not all(math.isfinite(number) for number in coordinates):
            errors.append(f"{prefix}: coordenadas devem ser finitas; NaN e infinito não são aceitos.")
            continue
        # 4e. Normalizadas: cada valor entre 0 e 1 (com a folga BOX_TOLERANCE).
        # Valores em pixels (ex.: 512) caem aqui, um erro comum de conversão.
        if not all(-BOX_TOLERANCE <= number <= 1 + BOX_TOLERANCE for number in coordinates):
            errors.append(f"{prefix}: coordenadas devem estar normalizadas no intervalo [0, 1].")
            continue
        # 4f. Caixa com área: largura e altura positivas.
        if width <= 0 or height <= 0:
            errors.append(f"{prefix}: largura e altura devem ser maiores que zero.")
            continue
        # 4g. Converte centro+tamanho para bordas (x1, y1, x2, y2), ainda normalizadas:
        #     x1 = xc - w/2, y1 = yc - h/2, x2 = xc + w/2, y2 = yc + h/2.
        #     Um centro dentro da foto pode ter borda fora; isso é conferido aqui.
        edges = (x_center - width / 2, y_center - height / 2, x_center + width / 2, y_center + height / 2)
        if edges[0] < -BOX_TOLERANCE or edges[1] < -BOX_TOLERANCE or edges[2] > 1 + BOX_TOLERANCE or edges[3] > 1 + BOX_TOLERANCE:
            errors.append(f"{prefix}: a caixa ultrapassa os limites da imagem.")
            continue
        # EXTENSAO: checagem nova por caixa (ex.: área mínima, proporção máxima): acrescente
        # aqui um "if ...: errors.append(f"{prefix}: ..."); continue" antes da contagem.
        valid_objects += 1
    return valid_objects, False


def _image_digest(image_path: Path, display: str, errors: list[str]) -> str | None:
    """Confere se a foto abre e decodifica, e calcula o SHA-256 dos bytes.

    Parâmetros:
        image_path: caminho da foto.
        display: caminho relativo para mensagens.
        errors: lista de erros do relatório.

    Retorna:
        o SHA-256 em hexadecimal (64 caracteres) do arquivo inteiro, usado para
        achar fotos idênticas; ou None se a foto for inválida (erro registrado).

    Erros registrados (não lançados): foto menor que 10 px em algum lado, JPEG
    sem o marcador final EOI (``FF D9``), arquivo corrompido ou "decompression bomb".

    EXTENSAO: checagem nova por foto (ex.: resolução mínima maior, modo de cor):
    acrescente um ``raise ValueError("...")`` dentro do primeiro ``with``.
    """
    try:
        # 1. Abre só o cabeçalho e confere o tamanho mínimo exigido pela Ultralytics.
        with Image.open(image_path) as image:
            if image.width < 10 or image.height < 10:
                raise ValueError("largura e altura devem ter pelo menos 10 pixels, conforme o loader YOLO")
            # 2. JPEG (e MPO, JPEG de câmeras com várias imagens): os 2 últimos bytes
            #    devem ser o marcador EOI FF D9 (fim da imagem).
            if image.format in {"JPEG", "MPO"}:
                with image_path.open("rb") as stream:
                    stream.seek(-2, os.SEEK_END)
                    if stream.read() != b"\xff\xd9":
                        # A biblioteca repararia e substituiria esse original.
                        # ATENCAO: a Ultralytics regrava o JPEG "consertado" por cima do original;
                        # por isso a foto é recusada aqui, antes de qualquer treino.
                        raise ValueError("JPEG sem marcador EOI final; revise a imagem antes do treino para impedir reparo automático do original")
            # 3. verify() checa a estrutura do arquivo sem decodificar os pixels.
            image.verify()
        # load() decodifica de fato os pixels; só abrir o cabeçalho não basta.
        # (Depois de verify() o objeto não pode mais ser usado, por isso abre de novo.)
        with Image.open(image_path) as image:
            image.load()
        # 4. Hash dos bytes em blocos de 1 MiB (não carrega o arquivo inteiro na memória).
        digest = hashlib.sha256()
        with image_path.open("rb") as stream:
            # iter(função, b"") chama stream.read até ele devolver b"" (fim do arquivo).
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        errors.append(f"{display}: imagem ilegível ou inválida: {error}")
        return None


def _inspect_split(split: str, directory: Path, root: Path, class_count: int, report: dict[str, Any], hashes: dict[str, tuple[str, str]]) -> dict[str, str]:
    """Revisa imagens e suas labels, preservando o vínculo usado no manifesto.

    Parâmetros:
        split: ``"train"``, ``"val"`` ou ``"test"``.
        directory: pasta absoluta das imagens do split (sob ``root/images``).
        root: raiz do dataset.
        class_count: número de classes da taxonomia.
        report: relatório em construção (erros, avisos e ``counts[split]``).
        hashes: dict compartilhado entre os splits, ``{sha256: (display, split)}``,
            preenchido aqui para detectar duplicatas e vazamento entre splits.

    Retorna:
        dict ``{caminho relativo da foto: split}`` (ex.: ``{"images/train/a.jpg": "train"}``),
        usado depois por ``_validate_groups`` para confrontar com o groups.csv.

    Chamada por: ``validate_dataset`` (etapa 3), uma vez por split.
    """
    errors = report["errors"]
    counts = report["counts"][split]
    image_root = (root / "images").resolve()
    label_root = (root / "labels").resolve()
    # Pasta das labels espelha a das imagens: images/train/x -> labels/train/x.
    label_directory = label_root / directory.relative_to(image_root)
    # 1. Lista recursiva das fotos com extensão aceita, em ordem estável
    #    (casefold = minúsculas robustas) para o relatório não mudar entre execuções.
    images = sorted((path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS), key=lambda path: path.as_posix().casefold())
    counts["images"] = len(images)
    if not images:
        errors.append(f"{split}: nenhum arquivo de imagem suportado foi encontrado.")
    stem_paths: dict[str, Path] = {}
    expected_labels: set[Path] = set()
    used_images: dict[str, str] = {}
    # 2. Cada foto: caminho seguro, stem único, hash, label correspondente e conteúdo da label.
    for image_path in images:
        display = _relative_display(image_path, root)
        # 2a. Link simbólico que aponta para fora de images/ é recusado.
        if not _inside(image_path, image_root):
            errors.append(f"{display}: o arquivo resolve para fora de path/images.")
            continue
        used_images[display] = split
        # Duas extensões com o mesmo stem apontariam para a mesma anotação.
        # 2b. stem_key = caminho sem extensão, em minúsculas (a.jpg e A.png colidem).
        relative_image = image_path.relative_to(directory)
        stem_key = relative_image.with_suffix("").as_posix().casefold()
        if stem_key in stem_paths:
            errors.append(f"{split}: colisão de stem entre {_relative_display(stem_paths[stem_key], root)} e {display}; as imagens compartilhariam uma label.")
        else:
            stem_paths[stem_key] = image_path
        # 2c. Conteúdo idêntico (mesmo SHA-256): dentro do split é duplicação,
        #     entre splits é vazamento (a mesma foto no treino e na validação).
        digest = _image_digest(image_path, display, errors)
        if digest is not None:
            if digest in hashes:
                previous_display, previous_split = hashes[digest]
                kind = "vazamento entre splits" if split != previous_split else "duplicação dentro do split"
                errors.append(f"{display}: {kind}; conteúdo idêntico a {previous_display} ({previous_split}).")
            else:
                hashes[digest] = (display, split)
        # 2d. Label esperada pelo layout do dataset (labels/<split>/<mesmo caminho>.txt).
        label_path = label_directory / relative_image.with_suffix(".txt")
        expected_labels.add(label_path.resolve())
        # 2e. Tem de coincidir com a label que a Ultralytics procuraria de verdade.
        derived_label = _ultralytics_label_path(image_path)
        if derived_label.resolve() != label_path.resolve():
            errors.append(f"{display}: caminho de label incompatível com Ultralytics; a biblioteca procuraria {_relative_display(derived_label, root)}, mas o dataset registra {_relative_display(label_path, root)}. Use a pasta literal images e evite subpastas images aninhadas.")
            continue
        if not _inside(label_path, label_root):
            errors.append(f"{display}: a label resolve para fora de path/labels.")
            continue
        # 2f. ATENCAO: label ausente é ERRO, não negativo. Foto sem defeito precisa de
        # um .txt vazio explícito, para "esqueci de anotar" não virar "não tem defeito".
        if not label_path.is_file():
            errors.append(f"{display}: label ausente em {_relative_display(label_path, root)}; negativos exigem um .txt vazio explícito.")
            continue
        # 2g. Confere o conteúdo da label e soma as contagens do split.
        counts["labels"] += 1
        objects, negative = _validate_label(label_path, _relative_display(label_path, root), class_count, errors)
        counts["objects"] += objects
        counts["negative_images"] += int(negative)
    # 3. Labels órfãs: .txt em labels/<split> que nenhuma foto usa (foto apagada ou renomeada).
    if label_directory.is_dir():
        for label_path in sorted(label_directory.rglob("*")):
            if label_path.is_file() and label_path.suffix.lower() == ".txt" and label_path.resolve() not in expected_labels:
                errors.append(f"{split}: label órfã, sem imagem correspondente: {_relative_display(label_path, root)}.")
    # 4. Aviso (não erro): split só com negativos não ensina nem mede defeito algum.
    if images and counts["negative_images"] == len(images):
        report["warnings"].append(f"{split}: todas as imagens são negativos explícitos; não há objetos anotados neste split.")
    return used_images


# ---------------------------------------------------------------------------
# groups.csv (unidade de captura) e função principal.
# ---------------------------------------------------------------------------

def _validate_groups(root: Path, used_images: dict[str, str], errors: list[str]) -> None:
    """Impede vazamento por unidade de captura, mesmo com frames diferentes.

    Parâmetros:
        root: raiz do dataset (onde fica ``groups.csv``).
        used_images: ``{caminho relativo da foto: split}`` vindo de ``_inspect_split``.
        errors: lista de erros do relatório.

    Retorna: None (só acrescenta erros).

    Formato do ``groups.csv`` (UTF-8, com ou sem BOM), colunas obrigatórias::

        image,group_id,split
        images/train/a.jpg,telhado_01,train

    Regras: cada linha completa; ``image`` relativo à raiz e usado por um split;
    o split da linha igual ao da pasta; cada foto listada uma vez; um ``group_id``
    sempre no mesmo split; toda foto dos splits presente no manifesto.

    EXTENSAO: coluna obrigatória nova: acrescente o nome no conjunto
    ``{"image", "group_id", "split"}`` abaixo e leia/valide o valor no laço
    de linhas; mude junto quem grava o groups.csv (pilot.py, review_dataset.py, tiling.py).
    """
    manifest = root / "groups.csv"
    # 1. O manifesto é obrigatório.
    if not manifest.is_file():
        errors.append("groups.csv obrigatório ausente na raiz do dataset (colunas image,group_id,split).")
        return
    seen_images: set[str] = set()
    # group_splits: primeiro split visto para cada group_id.
    group_splits: dict[str, str] = {}
    try:
        with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            # 2. Cabeçalho com as três colunas obrigatórias (colunas extras são permitidas).
            if reader.fieldnames is None or not {"image", "group_id", "split"}.issubset(reader.fieldnames):
                errors.append("groups.csv deve conter as colunas image,group_id,split.")
                return
            # 3. Cada linha de dados. start=2 porque a linha 1 do arquivo é o cabeçalho.
            for row_number, row in enumerate(reader, start=2):
                prefix = f"groups.csv, linha {row_number}"
                image_value = (row.get("image") or "").strip()
                group_id = (row.get("group_id") or "").strip()
                split = (row.get("split") or "").strip()
                # 3a. Campos preenchidos e split conhecido.
                if not image_value or not group_id or split not in SPLITS:
                    errors.append(f"{prefix}: image e group_id devem estar preenchidos e split deve ser train, val ou test.")
                    continue
                # 3b. Aceita "\" do Windows, mas exige caminho relativo dentro da raiz.
                image_path = Path(image_value.replace("\\", "/"))
                if image_path.is_absolute() or not _inside(root / image_path, root):
                    errors.append(f"{prefix}: image deve ser um caminho relativo dentro da raiz do dataset.")
                    continue
                # 3c. Normaliza para o mesmo formato das chaves de used_images
                #     (resolve "..", links e barras), para a comparação funcionar.
                image_key = _relative_display((root / image_path).resolve(), root)
                if image_key in seen_images:
                    errors.append(f"{prefix}: imagem duplicada no manifesto: {image_key}.")
                seen_images.add(image_key)
                # 3d. A foto precisa existir num split e no MESMO split informado.
                if image_key not in used_images:
                    errors.append(f"{prefix}: imagem não usada pelos splits configurados: {image_key}.")
                elif used_images[image_key] != split:
                    errors.append(f"{prefix}: split de {image_key} diverge do diretório configurado.")
                # 3e. setdefault grava o split na primeira vez e devolve o já gravado
                #     nas seguintes; se divergir, o grupo cruzou splits (vazamento).
                previous_split = group_splits.setdefault(group_id, split)
                if previous_split != split:
                    errors.append(f"{prefix}: vazamento por grupo {group_id!r} entre {previous_split} e {split}.")
    except (OSError, UnicodeError, csv.Error) as error:
        errors.append(f"Não foi possível ler groups.csv: {error}")
        return
    # 4. Fotos dos splits que o manifesto esqueceu (diferença de conjuntos).
    for image_key in sorted(set(used_images) - seen_images):
        errors.append(f"groups.csv: imagem sem grupo de captura: {image_key}.")


# ENTRADA: validação de dataset (comando validate-data da CLI, training.train/evaluate/tune,
# review_dataset.py e review_provenance.py).
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

    Parâmetros:
        data_path: caminho do ``dataset.yaml`` (relativo ou absoluto).
        expected_names: nomes das classes na ordem dos IDs (ex.: vindo de
            ``detection_names(load_taxonomy(...))`` ou ``MVP_NAMES``).

    Retorna:
        o relatório de ``_new_report``: ``{"valid", "errors", "warnings", "counts"}``.
        Nunca lança exceção por dataset ruim; tudo vira mensagem em ``errors``.

    ATENCAO: os IDs de classe são contrato entre labels, pesos treinados e revisões;
    por isso ``names`` do YAML precisa ser IDÊNTICO a ``expected_names`` (nomes e ordem).
    """
    report = _new_report()
    # 0. A própria taxonomia esperada precisa ser válida (nomes únicos e não vazios).
    if not expected_names or any(not isinstance(name, str) or not name.strip() for name in expected_names) or len(set(expected_names)) != len(expected_names):
        report["errors"].append("expected_names deve conter nomes de classe únicos e não vazios.")
        return _finish_report(report)
    # 1. YAML legível e raiz existente.
    config_result = _read_config(Path(data_path).resolve(), report)
    if config_result is None:
        return _finish_report(report)
    config, root = config_result
    # 2. Classes do YAML iguais às da taxonomia, na mesma ordem.
    names = _read_names(config.get("names"), report["errors"])
    if names is not None and names != expected_names:
        report["errors"].append("Os nomes e a ordem de IDs em names divergem da taxonomia esperada; revise a anotação antes do treino.")
    # nc (número de classes) é opcional; se existir, tem de ser int e bater com a taxonomia.
    # "type(...) is not int" recusa também True/False (bool é subclasse de int).
    if "nc" in config and (type(config["nc"]) is not int or config["nc"] != len(expected_names)):
        report["errors"].append("nc diverge da quantidade de classes esperadas.")
    # 3. Cada split: fotos, labels e duplicatas entre splits (pelo hash da foto).
    directories = _split_directories(config, root, report["errors"])
    # hashes é compartilhado entre os splits para detectar a mesma foto em dois deles.
    hashes: dict[str, tuple[str, str]] = {}
    used_images: dict[str, str] = {}
    for split, directory in directories.items():
        used_images.update(_inspect_split(split, directory, root, len(expected_names), report, hashes))
    # 4. groups.csv cobre todas as fotos e nenhum grupo cruza splits.
    _validate_groups(root, used_images, report["errors"])
    # EXTENSAO: checagem nova do dataset inteiro (ex.: mínimo de objetos por classe):
    # acrescente aqui, gravando em report["errors"] (bloqueia o treino) ou
    # report["warnings"] (só alerta), antes de fechar o relatório.
    return _finish_report(report)
