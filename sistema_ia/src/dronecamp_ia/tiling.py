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

Funções públicas:
- ``calculate_patch_windows(width, height, patch_size, overlap_ratio)``: lista de janelas
  ``(x1, y1, x2, y2)`` em pixels da foto original.
- ``project_boxes_to_patch(boxes, window, min_visibility)``: leva caixas da foto para a janela.
- ``parse_yolo_labels(label_path, img_w, img_h)``: label YOLO normalizada -> caixas xyxy em pixels.
- ``build_tiled_dataset(...)``: gera um dataset YOLO inteiro feito de janelas (treino).
- ``merge_tiled_predictions(predictions, iou_threshold)``: devolve as detecções das janelas
  para a foto inteira e remove duplicatas (NMS por classe) (inferência).

Quem chama:
- ``cli.py``: comando ``python -m dronecamp_ia tile-dataset`` -> ``build_tiled_dataset``.
- ``training._tiled_training_dataset``: treino piloto em janelas (``pilot.tiling`` /
  ``train-pilot --tile``) -> ``build_tiled_dataset``.
- ``prediction._tiled_detections`` e ``suggestions.py``: inferência fatiada
  -> ``calculate_patch_windows`` + ``merge_tiled_predictions``.

Disco:
- Lê: ``<origem>/dataset.yaml``, ``<origem>/images/<split>/*`` e ``<origem>/labels/<split>/*.txt``.
- Grava (em ``<destino>``): ``images/<split>/<foto>_pNN_<x>_<y>.<ext>``, a label ``.txt`` de cada
  janela, ``dataset.yaml`` e ``tiling_manifest.json``. NÃO grava ``groups.csv``.

Sistemas de coordenadas usados aqui (todos com origem no canto superior esquerdo):
- YOLO normalizado: ``cx cy w h`` divididos pela largura/altura da imagem (0 a 1).
- xyxy em pixels da FOTO ORIGINAL ("global"): ``[x1, y1, x2, y2]``.
- xyxy em pixels da JANELA ("local"): o mesmo, mas subtraindo o canto ``(wx1, wy1)`` da janela.
  Global -> local: ``x_local = x_global - wx1``; local -> global: ``x_global = x_local + wx1``
  (idem para y com ``wy1``). Não há mudança de escala: a janela é um recorte, não um redimensionamento.

Onde mexer:
- Mudar tamanho/sobreposição das janelas no treino e nas sugestões: ``configs/project.yaml``
  (``pilot.tiling``), não aqui.
- Mudar quanto de uma caixa precisa aparecer na janela para ela ser mantida:
  ``min_visibility`` de ``project_boxes_to_patch`` (hoje fixo em 0.3 no ``build_tiled_dataset``).
- Aceitar outro formato de foto no fatiamento: conjunto de extensões em ``build_tiled_dataset``.
- Mudar o nome dos arquivos das janelas: ``patch_name`` em ``build_tiled_dataset``.
- Trocar o critério de fusão das predições (ex.: NMS entre classes, WBF): ``merge_tiled_predictions``.
"""

from __future__ import annotations

# Observação: Counter, json, math, shutil e file_hash são importados mas não usados
# neste arquivo hoje; ficam disponíveis caso uma extensão precise deles.
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


# ---------------------------------------------------------------------------
# Geometria: janelas de corte, projeção de caixas e leitura de labels YOLO
# ---------------------------------------------------------------------------

def calculate_patch_windows(width: int, height: int, patch_size: int = 1280, overlap_ratio: float = 0.2) -> list[tuple[int, int, int, int]]:
    """Calcula janelas de corte (x1, y1, x2, y2) com sobreposição cobrindo 100% da imagem.

    Parâmetros:
        width: largura da foto em pixels (ex.: 5472).
        height: altura da foto em pixels (ex.: 3648).
        patch_size: lado da janela quadrada em pixels (ex.: 1280).
        overlap_ratio: fração de sobreposição entre janelas vizinhas (0.2 = 20%);
            aceita de 0.0 (sem sobreposição) até menos de 1.0.

    Retorna:
        lista de tuplas ``(x1, y1, x2, y2)`` em pixels da foto original, com x2/y2
        exclusivos (prontas para ``Image.crop`` e para fatiar arrays ``[y1:y2, x1:x2]``).
        Ordem: linha por linha (y crescente), e dentro da linha x crescente.
        Foto menor ou igual ao patch nos dois lados -> uma única janela, a foto inteira.

    Erros:
        ValueError: dimensões <= 0, ``patch_size`` <= 0 ou ``overlap_ratio`` fora de [0, 1).

    Exemplo: 5472 × 3648, patch 1280, overlap 0.2 -> passo 1024 px; em x as janelas
    começam em 0, 1024, 2048, 3072, 4096 e a última encostada na borda (4192).

    ATENCAO: a ordem e as coordenadas destas janelas entram no nome dos arquivos do
    dataset fatiado (``_pNN_<x>_<y>``); mudar a fórmula muda os nomes gerados.
    """
    # 1. Validação dos parâmetros.
    if width <= 0 or height <= 0:
        raise ValueError(f"Dimensões inválidas: {width}x{height}")
    if patch_size <= 0:
        raise ValueError("patch_size deve ser maior que zero.")
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio deve estar entre 0.0 e 0.99.")

    # Se a foto for menor que o patch, retorna a própria imagem
    if width <= patch_size and height <= patch_size:
        return [(0, 0, width, height)]

    # 2. Tamanho real da janela em cada eixo: se só um lado da foto é menor que o
    #    patch, a janela usa aquele lado inteiro (a janela deixa de ser quadrada).
    effective_patch_w = min(patch_size, width)
    effective_patch_h = min(patch_size, height)
    # 3. Passo (stride) entre inícios de janela = tamanho × (1 - sobreposição),
    #    truncado para pixel inteiro; nunca menor que 1 para o laço terminar.
    stride_x = max(1, int(effective_patch_w * (1.0 - overlap_ratio)))
    stride_y = max(1, int(effective_patch_h * (1.0 - overlap_ratio)))

    # Pontos de início nos eixos X e Y
    # 4. Eixo X: avança de stride em stride enquanto a janela ainda não toca a borda
    #    direita; depois acrescenta uma última janela encostada na borda
    #    (width - patch). Essa última pode sobrepor MAIS que overlap_ratio à anterior,
    #    mas garante que nenhuma faixa da foto fique de fora. set() remove o início
    #    repetido quando o passo cai exatamente na borda.
    x_starts = []
    x = 0
    while x + effective_patch_w < width:
        x_starts.append(x)
        x += stride_x
    x_starts.append(width - effective_patch_w)
    x_starts = sorted(set(x_starts))

    # 5. Eixo Y: mesma regra do eixo X, com a altura.
    y_starts = []
    y = 0
    while y + effective_patch_h < height:
        y_starts.append(y)
        y += stride_y
    y_starts.append(height - effective_patch_h)
    y_starts = sorted(set(y_starts))

    # 6. Combina todos os inícios (produto cartesiano) em janelas (x1, y1, x2, y2).
    windows = []
    for ys in y_starts:
        for xs in x_starts:
            windows.append((xs, ys, xs + effective_patch_w, ys + effective_patch_h))
    return windows


def project_boxes_to_patch(boxes: list[dict], window: tuple[int, int, int, int], min_visibility: float = 0.3) -> list[dict]:
    """Projeta caixas delimitadoras absolutas [x1, y1, x2, y2] para o sistema de coordenadas local do patch.

    Se uma caixa for cortada pela margem do patch, a área visível é calculada.
    Se a proporção da área for >= min_visibility, a caixa é mantida recortada nos limites do patch.

    Parâmetros:
        boxes: lista de dicts com ``"class_id"`` (int), ``"bbox_xyxy"`` ([x1, y1, x2, y2]
            em pixels da foto original) e, opcionalmente, ``"id"``.
        window: janela ``(wx1, wy1, wx2, wy2)`` em pixels da foto original
            (como devolvida por ``calculate_patch_windows``).
        min_visibility: fração mínima (0 a 1) da área da caixa que precisa cair dentro
            da janela para ela ser mantida. 0.3 = pelo menos 30% visível.

    Retorna:
        lista de dicts, um por caixa mantida: ``class_id``, ``bbox_xyxy`` em pixels
        LOCAIS da janela (arredondados a 0.1 px), ``original_box_id`` (o ``id`` de
        origem ou None) e ``visibility`` (fração visível, 3 casas).

    ATENCAO: uma caixa com menos de ``min_visibility`` visível some da janela, mas o
    pedaço dela continua na imagem recortada; essa janela pode virar "negativa" com
    um defeito parcial à vista. É o compromisso padrão do fatiamento.
    """
    wx1, wy1, wx2, wy2 = window
    projected = []

    for box in boxes:
        # 1. Área da caixa original (max(0, ...) protege contra caixas invertidas).
        bx1, by1, bx2, by2 = box["bbox_xyxy"]
        box_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        if box_area <= 0:
            continue

        # Interseção entre a caixa e a janela do patch
        # 2. Interseção de dois retângulos: maior dos inícios e menor dos fins.
        ix1 = max(bx1, wx1)
        iy1 = max(by1, wy1)
        ix2 = min(bx2, wx2)
        iy2 = min(by2, wy2)

        # Sem sobreposição (largura ou altura da interseção <= 0): a caixa está fora da janela.
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        # 3. Visibilidade = área dentro da janela / área total da caixa.
        inter_area = (ix2 - ix1) * (iy2 - iy1)
        visibility = inter_area / box_area

        # EXTENSAO: regra diferente por classe (ex.: exigir 0.6 para classes pequenas):
        # troque min_visibility por um valor buscado pelo box["class_id"] aqui.
        if visibility >= min_visibility:
            # Converte para coordenadas locais (relativas a wx1, wy1)
            # 4. Global -> local: subtrai o canto superior esquerdo da janela.
            #    Usa a INTERSEÇÃO, então a caixa já sai recortada nas bordas da janela.
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
    """Lê arquivo de labels YOLO normalizado (class_id x_center y_center w h) e converte em xyxy absoluto.

    Parâmetros:
        label_path: arquivo ``.txt`` YOLO da foto.
        img_w: largura da foto em pixels.
        img_h: altura da foto em pixels.

    Retorna:
        lista de ``{"class_id": int, "bbox_xyxy": [x1, y1, x2, y2]}`` em pixels (float)
        da foto original. Arquivo ausente -> lista vazia (tratada como foto negativa).

    Conversão YOLO -> xyxy em pixels (inversa de ``review_data.boxes_to_yolo``)::

        x1 = (cx - w/2) × largura     y1 = (cy - h/2) × altura
        x2 = (cx + w/2) × largura     y2 = (cy + h/2) × altura

    ATENCAO: não valida nada (quem valida é ``dataset.validate_dataset``). Linhas com
    menos de 5 campos são ignoradas em silêncio; colunas a mais (ex.: confiança) são
    descartadas; um class_id não inteiro lança ValueError do ``int()``.
    """
    if not label_path.is_file():
        return []
    boxes = []
    # Uma linha por objeto: "class_id cx cy w h".
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            cid = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            # Centro ± metade do tamanho dá as bordas (ainda 0..1); × tamanho da foto dá pixels.
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            boxes.append({"class_id": cid, "bbox_xyxy": [x1, y1, x2, y2]})
    return boxes


# ---------------------------------------------------------------------------
# Treino: geração de um dataset YOLO inteiro feito de janelas
# ---------------------------------------------------------------------------

# ENTRADA: comando tile-dataset (cli.py) e treino piloto em janelas
# (training._tiled_training_dataset, ligado por pilot.tiling / train-pilot --tile).
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

    Retorna:
        dict de estatísticas: ``original_images``, ``tiled_images`` (janelas gravadas),
        ``positive_patches``, ``negative_patches`` (negativas mantidas),
        ``total_boxes_projected`` e ``by_split`` (``{split: {positive_patches,
        negative_patches_kept, total_patches, boxes}}``). O mesmo dict vai, junto com os
        parâmetros, para ``<output_dir>/tiling_manifest.json``.

    Erros:
        FileNotFoundError: não existe ``dataset.yaml`` na pasta de origem.
        ValueError: nenhuma das pastas ``images/train``, ``images/val``, ``images/test`` existe;
            ou repassado de ``calculate_patch_windows`` (parâmetros inválidos).
        OSError: foto ilegível (``Image.open``) ou falha de gravação.

    Etapas: lê o YAML -> para cada split, calcula as janelas de cada foto e projeta as
    caixas -> sorteia as negativas -> recorta e grava as janelas e labels -> grava
    ``dataset.yaml`` e ``tiling_manifest.json``.

    ATENCAO: ``negative_ratio`` é aplicado sobre o NÚMERO DE JANELAS POSITIVAS do split
    (máximo de negativas = positivas × negative_ratio), não sobre o total de janelas.
    Com 0.2, as negativas ficam em no máximo ~17% do total. Se o split não tiver
    nenhuma janela positiva, TODAS as negativas são mantidas.

    ATENCAO: não grava ``groups.csv``; por isso o dataset gerado não passa em
    ``dataset.validate_dataset``. O treino piloto em janelas não o valida de novo
    (a origem já foi validada antes do fatiamento).

    ATENCAO: ``output_dir`` não é limpo antes; janelas de execuções anteriores com outros
    parâmetros continuam na pasta e entrariam no treino. Use uma pasta nova.
    """
    # 1. Caminhos absolutos (os nomes gravados no manifesto ficam independentes do cwd).
    source_dataset_dir = Path(source_dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()

    # 2. dataset.yaml da origem: só o campo names é usado (as pastas são fixas: images/<split>).
    yaml_file = source_dataset_dir / "dataset.yaml"
    if not yaml_file.is_file():
        raise FileNotFoundError(f"dataset.yaml não encontrado em {source_dataset_dir}")

    config_data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    names = config_data.get("names", {})
    # names em formato mapa {0: "a", 1: "b"} vira lista ordenada pelo ID numérico
    # (int(x) evita a ordem de texto "10" < "2").
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]

    # Gerador aleatório próprio, com semente: a mesma semente sorteia as mesmas negativas.
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # EXTENSAO: estatística nova no relatório/manifesto: acrescente a chave aqui e
    # some no fim do laço de splits (junto de stats["tiled_images"] += ...).
    stats = {
        "original_images": 0,
        "tiled_images": 0,
        "positive_patches": 0,
        "negative_patches": 0,
        "total_boxes_projected": 0,
        "by_split": {},
    }

    # 3. Splits presentes na origem (os ausentes são simplesmente pulados).
    splits = [s for s in ("train", "val", "test") if (source_dataset_dir / "images" / s).is_dir()]
    if not splits:
        raise ValueError(f"Nenhuma subpasta train/val encontrada em {source_dataset_dir / 'images'}")

    # 4. Um split por vez; as janelas de um split nunca vão para outro (sem vazamento).
    for split in splits:
        img_src_dir = source_dataset_dir / "images" / split
        lbl_src_dir = source_dataset_dir / "labels" / split
        img_out_dir = output_dir / "images" / split
        lbl_out_dir = output_dir / "labels" / split
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        # 4a. Fotos do split: só o primeiro nível da pasta (iterdir não desce em subpastas).
        # EXTENSAO: formato de foto aceito no fatiamento: acrescente a extensão neste conjunto
        # (ele é independente de dataset.IMAGE_EXTENSIONS; .tif/.bmp hoje ficam de fora).
        image_files = sorted([p for p in img_src_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
        stats["original_images"] += len(image_files)

        split_positive_patches = []
        split_negative_patches = []

        # 4b. Planejamento: para cada foto calcula janelas e caixas projetadas,
        #     sem gravar nada ainda (a seleção das negativas vem depois).
        for img_path in image_files:
            # Só o cabeçalho é lido aqui para saber o tamanho (rápido).
            with Image.open(img_path) as im:
                w, h = im.size

            # Label com o mesmo nome da foto; ausente = foto negativa (lista vazia).
            lbl_path = lbl_src_dir / f"{img_path.stem}.txt"
            # YOLO normalizado -> xyxy em pixels da foto original.
            original_boxes = parse_yolo_labels(lbl_path, w, h)
            windows = calculate_patch_windows(w, h, patch_size=patch_size, overlap_ratio=overlap_ratio)

            with Image.open(img_path) as im:
                for idx, window in enumerate(windows):
                    wx1, wy1, wx2, wy2 = window
                    # Tamanho real desta janela (pode ser menor que patch_size em fotos pequenas).
                    patch_w = wx2 - wx1
                    patch_h = wy2 - wy1
                    # xyxy da foto -> xyxy local da janela (min_visibility padrão, 0.3).
                    patch_boxes = project_boxes_to_patch(original_boxes, window)

                    # Nome: <foto>_p<índice com 2 dígitos>_<x1>_<y1>; o canto no nome permite
                    # saber de onde o recorte saiu na foto original.
                    patch_name = f"{img_path.stem}_p{idx:02d}_{wx1}_{wy1}"
                    patch_img_dest = img_out_dir / f"{patch_name}{img_path.suffix.lower()}"
                    patch_lbl_dest = lbl_out_dir / f"{patch_name}.txt"

                    # Tudo o que a etapa de gravação precisa saber sobre a janela.
                    patch_info = {
                        "window": window,
                        "source": img_path,
                        "dest_img": patch_img_dest,
                        "dest_lbl": patch_lbl_dest,
                        "boxes": patch_boxes,
                        "patch_w": patch_w,
                        "patch_h": patch_h,
                    }

                    # Positiva = ao menos uma caixa visível o bastante; senão, negativa.
                    if patch_boxes:
                        split_positive_patches.append(patch_info)
                    else:
                        split_negative_patches.append(patch_info)

        # Controle de patches negativos para evitar diluição excessiva do dataset
        # 4c. Máximo de negativas = positivas × negative_ratio (ver ATENCAO na docstring);
        #     split sem positivas mantém todas as negativas. O sorteio usa a semente.
        max_negatives = int(len(split_positive_patches) * negative_ratio) if split_positive_patches else len(split_negative_patches)
        selected_negatives = rng.sample(split_negative_patches, min(max_negatives, len(split_negative_patches)))

        # Embaralha só a lista de trabalho; a ordem não afeta os arquivos gerados.
        selected_patches = split_positive_patches + selected_negatives
        rng.shuffle(selected_patches)

        split_boxes_count = 0
        # Agrupa por foto: cada JPEG de 20 MP é decodificado uma única vez (~120 ms cada).
        by_source: dict[Path, list[dict]] = {}
        for p in selected_patches:
            by_source.setdefault(p["source"], []).append(p)
        # 4d. Gravação: decodifica a foto uma vez e recorta todas as janelas escolhidas dela.
        for source, patches in by_source.items():
            with Image.open(source) as im:
                im.load()
                for p in patches:
                    wx1, wy1, wx2, wy2 = p["window"]
                    # crop usa (x1, y1, x2, y2) em pixels com x2/y2 exclusivos.
                    # quality=95 vale para JPEG/WebP; o PNG ignora esse parâmetro.
                    im.crop((wx1, wy1, wx2, wy2)).save(p["dest_img"], quality=95)
                    # Salva o arquivo de rótulos YOLO
                    if p["boxes"]:
                        # xyxy LOCAL em pixels -> YOLO normalizado pelo tamanho da JANELA
                        # (não da foto): cx = (x1+x2)/(2·patch_w), w = (x2-x1)/patch_w etc.
                        yolo_text = boxes_to_yolo(p["boxes"], p["patch_w"], p["patch_h"])
                        p["dest_lbl"].write_text(yolo_text, encoding="utf-8")
                        split_boxes_count += len(p["boxes"])
                    else:
                        # Negativa explícita: .txt vazio (o formato exigido por dataset.py).
                        p["dest_lbl"].touch()

        # 4e. Estatísticas do split e acumulado geral.
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
    # 5. YAML do dataset fatiado, com caminhos relativos à própria pasta (path: ".").
    # ATENCAO: "val" é sempre escrito, mesmo se a origem não tiver images/val.
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

    # 6. Manifesto do fatiamento: parâmetros usados + estatísticas, para auditoria do treino.
    # EXTENSAO: parâmetro novo do fatiamento: registre-o também aqui (o training.py copia
    # este arquivo para runs/<treino>/tiling_manifest.json).
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


# ---------------------------------------------------------------------------
# Inferência: junta as detecções das janelas na foto inteira (NMS por classe)
# ---------------------------------------------------------------------------

def merge_tiled_predictions(predictions: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Recombina predições feitas em fatias locais de volta para o sistema global da imagem com NMS por classe.

    Args:
        predictions: Lista de dicionários com:
          - "bbox_xyxy": [x1, y1, x2, y2] nas coordenadas do patch
          - "score": confiança (float)
          - "class_id": ID da classe (int)
          - "window": [wx1, wy1, wx2, wy2] coordenadas do patch na foto original
        iou_threshold: Limiar de IoU para suprimir duplicatas em áreas de sobreposição.

    Retorna:
        lista de ``{"class_id", "score", "bbox_xyxy"}`` com caixas em pixels da FOTO
        ORIGINAL, já sem duplicatas, ordenada da maior para a menor confiança.
        Lista vazia se ``predictions`` estiver vazia. ``score`` ausente vale 1.0.

    Chamada por: ``prediction._tiled_detections`` e ``suggestions.py`` (com
    ``pilot.tiling.merge_iou``). Uma passada na foto inteira entra com
    ``window = [0, 0, largura, altura]`` (deslocamento zero).

    ATENCAO: o NMS é POR CLASSE: duas caixas de classes diferentes no mesmo lugar
    são ambas mantidas. Para fundir entre classes, agrupe tudo numa só lista em ``by_class``.
    """
    if not predictions:
        return []

    # 1. Translação de cada caixa para coordenadas absolutas da foto global
    #    Local -> global: soma o canto superior esquerdo da janela (wx1, wy1).
    #    Só a translação é feita; as caixas já estão na escala da foto (a janela é recorte).
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
        """IoU (interseção sobre união) de duas caixas xyxy na mesma escala.

        Parâmetros:
            b1, b2: caixas ``[x1, y1, x2, y2]``.

        Retorna:
            float de 0.0 (sem sobreposição) a 1.0 (caixas idênticas).
        """
        # Retângulo de interseção: maior dos inícios, menor dos fins.
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2])
        iy2 = min(b1[3], b2[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        # União = soma das áreas menos a parte contada duas vezes.
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    # 3. Separa as caixas por classe (o NMS roda dentro de cada grupo).
    by_class: dict[int, list[dict]] = {}
    for box in global_boxes:
        by_class.setdefault(box["class_id"], []).append(box)

    # 4. NMS guloso: percorre da maior para a menor confiança e só mantém a caixa
    #    se ela não sobrepõe (IoU < limite) nenhuma caixa já mantida.
    # EXTENSAO: outra fusão (ex.: média ponderada das caixas sobrepostas, WBF): troque este bloco.
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
