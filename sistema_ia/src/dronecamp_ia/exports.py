"""Exportações comerciais: pacote de evidências (ZIP) e laudo técnico preliminar.

Função no projeto: transforma uma revisão (``registry.json`` + decisões da página)
em material entregável ao cliente, sem nunca inventar gravidade, causa ou
diagnóstico. O servidor da plataforma (``platform_server.py``) chama estas
funções; a página de revisão vai usar ``draw_boxes`` para desenhar as caixas.

O que faz:
- ``draw_boxes``: desenha caixas (e, opcionalmente, sugestões tracejadas do
  modelo) numa foto, no mesmo estilo das evidências de ``review_render``.
- ``resolve_decisions``: para cada foto do registro decide quais caixas e qual
  status valem (decisão humana da página > registro) e conta as sugestões do
  modelo ainda sem decisão.
- ``build_evidence_package``: ZIP com as fotos demarcadas, ``ocorrencias.csv``,
  ``fotos.csv``, ``manifesto.json`` (hashes) e ``LEIA-ME.txt``.
- ``build_report``: pasta com ``laudo.html`` (autossuficiente, pronto para
  imprimir em A4), ``laudo.json``, ``laudo.pdf`` (se ``reportlab`` estiver
  instalado: extra ``reports``) e ``manifesto.json``. Estrutura ABNT, normas e
  texto fixo do laudo ficam em ``laudo_abnt.py``; aqui só os dados.
- ``occurrences_csv``: o texto do ``ocorrencias.csv`` sem gerar o ZIP.

Toda saída é nova, dentro de ``output_dir``; registros, fotos e revisões não
são tocados. Fotos são conferidas pelo SHA-256 antes de qualquer desenho.
Gravidade, diagnóstico e recomendação ficam explicitamente em branco
("pendente de validação técnica") para o responsável técnico preencher.

Quando mexer: colunas dos CSV, aparência das evidências ou campos do
``laudo.json`` (texto e formatação do laudo: ``laudo_abnt.py``).

Mapa do arquivo (seções, na ordem):
1. Constantes (status, colunas dos CSV, rótulos, filtros).
2. Desenho das evidências: ``_font``, ``_palette``, ``_label``,
   ``_dashed_rectangle``, ``draw_boxes``.
3. Decisões: ``_js_round``, ``_same_box``, ``_pending_suggestions``,
   ``_normalize_suggestions``, ``_resolve_entries``, ``resolve_decisions``.
4. Contexto comum: ``_Context``, ``_suggestion_file``, ``_model_meta``,
   ``_load_context``, ``_checked_source``, ``_render_evidence``,
   ``_filter_entries``.
5. Utilidades: ``_number``, ``_utc_now``, ``_safe_name``, ``_unique_path``,
   ``_csv_text``, ``_evidence_name``, ``_occurrence_rows``, ``_photo_rows``,
   ``_counts``, ``_readme``.
6. Pacote ZIP: ``_HashingWriter``, ``build_evidence_package``, ``occurrences_csv``.
7. Laudo: ``build_report``, ``_report_items``, ``_build_report_files``.

Quem chama:
- ``platform_server.py``: ``POST /api/export/evidence`` → ``_export_evidence``
  → ``build_evidence_package`` (grava em ``<revisão>/exportacoes/``);
  ``POST /api/report`` → ``_build_report`` → ``build_report`` (grava em
  ``<revisão>/laudos/``); a contagem de sugestões pendentes da lista de
  inspeções usa ``resolve_decisions``.
- ``tests/test_exports.py`` usa ``draw_boxes``, ``occurrences_csv`` e as demais.

O que lê do disco (só leitura):
- ``data/reviews/<inspeção>/registry.json`` (registro de revisão);
- ``suggestions.json`` e ``ai_notes.json`` ao lado do registro (opcionais; só
  valem se o ``registry_sha256`` gravado neles for o do registro atual);
- ``configs/taxonomy.json``;
- as fotos originais (``source_path`` de cada imagem do registro).

O que grava: um arquivo/pasta NOVO por chamada, com carimbo UTC no nome
(``<AAAAMMDDTHHMMSSZ>_evidencias_<versão>.zip`` ou
``<AAAAMMDDTHHMMSSZ>_laudo_<versão>/``). Em caso de erro, nada parcial fica.

Onde mexer:
- Coluna nova no ``ocorrencias.csv``/``fotos.csv``: ``OCCURRENCE_COLUMNS``/
  ``PHOTO_COLUMNS`` + ``_occurrence_rows``/``_photo_rows`` (mesma posição).
- Novo status de foto: ``review_data.IMAGE_STATUSES``, ``STATUS_LABELS`` aqui,
  a ordem de ``by_status`` em ``_counts`` e o texto ``facts`` do laudo.
- Novo filtro de fotos do ZIP: ``ONLY_CHOICES`` + ``_filter_entries`` + a
  validação em ``platform_server._export_evidence``.
- Cores, espessuras e placas das caixas: ``draw_boxes`` (as cores por classe
  vêm de ``review_render.COLORS``).
- Arquivo novo dentro do ZIP: ``build_evidence_package``, chamando ``small``
  antes do manifesto (o hash entra no manifesto sozinho).
- Campo novo no ``laudo.json``: dicionário ``report`` em
  ``_build_report_files`` (e, para aparecer no documento, ``laudo_abnt.render_html``
  e ``laudo_abnt.write_pdf``).
- Campo novo em cada item da seção 6 do laudo: ``_report_items``, ``items.append``.

ATENCAO (regras que valem para o arquivo todo):
- A IA nunca aprova: sugestões do modelo nunca viram ocorrência no CSV nem no
  laudo; só caixas de decisão humana ou do registro.
- Gravidade, diagnóstico, recomendação, risco e prioridade são SEMPRE
  ``PENDING_TEXT``; nunca preencha por inferência do modelo.
- Nunca sobrescrever: ``_unique_path`` garante nome novo; erro apaga o parcial.
- Foto só é desenhada depois de conferida por SHA-256 e dimensões (``_checked_source``).
- Regra de "sugestão pendente" tem paridade com a página
  (``review_templates/platform.js``: ``sameBox`` e ``pendingSuggestionsFor``):
  mudou aqui, mude lá.
"""

# Anotações de tipo como texto (permite "dict | None" e referências adiante).
from __future__ import annotations

# ---------------------------------------------------------------------------
# Importações
# ---------------------------------------------------------------------------

# Biblioteca padrão: contagens (Counter), ordem estável dos arquivos do
# manifesto (OrderedDict), dataclass do contexto, carimbo UTC, hash, CSV em
# memória (io.StringIO/BytesIO), JSON, geometria do tracejado (math), nome
# seguro (re), remoção de laudo parcial (shutil) e o ZIP.
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import csv
import io
import json
import math
import re
import shutil
import zipfile

# Pillow: abrir, reduzir e desenhar as fotos.
from PIL import Image, ImageDraw, ImageFont

# Módulos do pacote:
# - config: tipo ProjectConfig e taxonomia;
# - io: hash, caminho local de um source_path gravado em outra máquina, JSON;
# - laudo_abnt: estrutura ABNT, textos fixos, HTML e PDF do laudo;
# - naming: nome amigável da inspeção;
# - review_data: contrato do registro (status válidos, versão do schema,
#   tamanho da foto sem rotação EXIF e validação de caixas).
from .config import ProjectConfig, detection_names, load_taxonomy
from .io import file_hash, resolve_local_path, write_json
from . import laudo_abnt
# Texto dos campos do responsável técnico e limitações do método: fonte única em laudo_abnt.
from .laudo_abnt import LIMITATIONS as METHODOLOGY, PENDING_TEXT, PRELIMINARY_TEXT
from .naming import review_name
from .review_data import IMAGE_STATUSES, REVIEW_SCHEMA, review_image_size, validate_boxes

# ---------------------------------------------------------------------------
# Constantes: status, colunas dos CSV, rótulos e filtros
# ---------------------------------------------------------------------------

# Decisão exportada pela página pode estar "pending" (rascunho sem status escolhido).
# Conjunto = os quatro status do registro (positive, negative, ambiguous, excluded) + "pending".
# ATENCAO: "pending" só é aceito em DECISÕES da página; no registro o status
# tem de estar em IMAGE_STATUSES (ver _resolve_entries).
DECISION_STATUSES = IMAGE_STATUSES | {"pending"}
# Cor do texto nas placas das caixas (contraste ≥ 4,5:1 com todas as cores de classe).
# Também é a cor de FUNDO das placas das sugestões. ATENCAO: ao mudar, confira
# o contraste com todas as cores de review_render.COLORS.
PLATE_TEXT = "#08111F"
# Colunas do ocorrencias.csv (uma linha por caixa) e do fotos.csv (uma linha por foto).
# EXTENSAO: coluna nova no ocorrencias.csv/fotos.csv = nome aqui + valor na mesma posição em
# _occurrence_rows/_photo_rows (mais abaixo). A ordem das listas precisa casar.
# ocorrencias.csv: posição da foto, hash, arquivo, status da foto, origem
# (revisao_humana/registro), número da caixa na foto, id e nome da classe,
# coordenadas em pixels da foto original e nota da caixa.
OCCURRENCE_COLUMNS = ["numero_foto", "image_sha256", "filename", "status", "origem", "caixa", "classe_id", "classe",
                      "x1", "y1", "x2", "y2", "nota"]
# fotos.csv: posição, hash, arquivo, tamanho em px, páginas do laudo de
# referência (separadas por ";"), grupo de edificação, cena, status, origem,
# quantidade de caixas, sugestões pendentes, notas e caminho da evidência no ZIP.
PHOTO_COLUMNS = ["numero_foto", "image_sha256", "filename", "largura", "altura", "paginas", "grupo_edificacao", "cena",
                 "status", "origem", "caixas", "sugestoes_pendentes", "notas", "evidencia"]
# Rótulos em português de cada status de foto (laudo.json e legendas das figuras).
# EXTENSAO: novo status: acrescente o rótulo aqui (senão KeyError em
# _build_report_files e _report_items) e em review_data.IMAGE_STATUSES.
STATUS_LABELS = {"positive": "positiva", "negative": "negativa", "ambiguous": "ambígua", "excluded": "excluída",
                 "pending": "pendente (rascunho sem decisão)"}
# Rótulos da origem das caixas de cada foto (decisão da página ou registro).
# EXTENSAO: nova origem de decisão: rótulo aqui + o valor em _resolve_entries.
SOURCE_LABELS = {"revisao_humana": "revisão humana", "registro": "registro (sem decisão humana nesta exportação)"}
# Filtros aceitos no parâmetro "only" do ZIP de evidências (ver _filter_entries).
# EXTENSAO: novo filtro: acrescente aqui, trate em _filter_entries e libere em
# platform_server._export_evidence (que valida o mesmo conjunto à parte).
ONLY_CHOICES = ("all", "decided", "positive")

# ---------------------------------------------------------------------------
# Desenho das evidências (mesmo estilo de review_render e da página).
# ---------------------------------------------------------------------------

# Fontes com acentos do pt-BR: Windows (como em review_render), depois Linux/macOS; por fim a padrão do Pillow.
# A primeira que existir no computador é usada (ver _font).
# EXTENSAO: outra fonte: acrescente o caminho na posição de prioridade desejada.
FONT_CANDIDATES = (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/segoeui.ttf"),
                   Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                   Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
                   Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"), Path("/Library/Fonts/Arial.ttf"),
                   Path("/System/Library/Fonts/Supplemental/Arial.ttf"))


def _font(size: int):
    """Primeira fonte TrueType existente; senão a padrão do Pillow (sem acentos nas versões bitmap).

    Parâmetros:
        size (int): tamanho da fonte em pixels.

    Retorna:
        ImageFont.FreeTypeFont | ImageFont.ImageFont: fonte pronta para
        ``draw.text``.

    Chamada por: ``draw_boxes``.
    """
    # Percorre FONT_CANDIDATES na ordem de prioridade.
    for path in FONT_CANDIDATES:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    # Nenhuma TrueType instalada: fonte embutida do Pillow (>= 10.1 aceita tamanho).
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow antigo: fonte bitmap sem tamanho.
        return ImageFont.load_default()


def _palette() -> list[str]:
    """Cores por classe de ``review_render`` (import tardio: aquele módulo passará a importar este).

    Retorna:
        list[str]: cores hexadecimais ("#RRGGBB"); a classe ``n`` usa a cor
        ``n % len(cores)``.

    Chamada por: ``_load_context`` (vira ``_Context.colors``).
    """
    from .review_render import COLORS
    return COLORS


def _label(labels: dict[int, str], class_id: int) -> str:
    """Nome legível da classe para as placas.

    Parâmetros:
        labels (dict[int, str]): id → rótulo da taxonomia (ex.: 0 → "Telha quebrada").
        class_id (int): id da classe da caixa.

    Retorna:
        str: o rótulo, ou "classe N" se o id não estiver em ``labels``.
    """
    return labels.get(class_id, f"classe {class_id}")


def _dashed_rectangle(draw: ImageDraw.ImageDraw, box: list[float], color: str, width: int, dash: tuple[int, int]) -> None:
    """Tracejado simulado com segmentos (o Pillow não tem linha tracejada nativa).

    Parâmetros:
        draw (ImageDraw.ImageDraw): desenhador da imagem de destino.
        box (list[float]): ``[x1, y1, x2, y2]`` em pixels da imagem desenhada
            (já multiplicados por ``scale``).
        color (str): cor "#RRGGBB".
        width (int): espessura da linha em pixels.
        dash (tuple[int, int]): (comprimento do traço, comprimento do vão) em pixels.

    Retorna: None (desenha no lugar).

    Chamada por: ``draw_boxes`` (sugestões do modelo).
    """
    x1, y1, x2, y2 = box
    # Os 4 lados como 4 segmentos: cantos em sentido horário, voltando ao primeiro.
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    on, off = dash
    # zip(corners, corners[1:]) dá pares (início, fim) de cada lado.
    for (ax, ay), (bx, by) in zip(corners, corners[1:]):
        length = math.hypot(bx - ax, by - ay)
        # Lado de comprimento zero (caixa degenerada): nada a desenhar.
        if length == 0:
            continue
        # Vetor unitário na direção do lado.
        ux, uy = (bx - ax) / length, (by - ay) / length
        # Anda pelo lado desenhando "on" px e pulando "off" px.
        position = 0.0
        while position < length:
            end = min(position + on, length)
            draw.line([(ax + ux * position, ay + uy * position), (ax + ux * end, ay + uy * end)], fill=color, width=width)
            position = end + off


def draw_boxes(image: Image.Image, boxes: list[dict], labels: dict[int, str], colors: list[str], *,
               suggestions: list[dict] | None = None, scale: float = 1.0) -> Image.Image:
    """Devolva uma cópia RGB da foto com as caixas desenhadas; a imagem recebida não muda.

    Caixas: retângulo na cor da classe (largura 3) e placa na cor da classe
    acima da caixa com texto escuro "n: Classe". Sugestões (opcional): retângulo
    tracejado, placa escura abaixo da caixa com texto na cor da classe
    "IA n: Classe NN%".

    ``scale`` é a razão entre a imagem recebida e a foto original (1.0 =
    resolução original). As coordenadas das caixas e sugestões estão sempre em
    pixels da foto original e são multiplicadas por ``scale``; espessura e
    fonte acompanham o tamanho da imagem desenhada.

    Parâmetros:
        image (PIL.Image.Image): foto (qualquer modo; é convertida em RGB).
        boxes (list[dict]): caixas confirmadas/do registro, cada uma com
            ``class_id`` (int) e ``bbox_xyxy`` ([x1, y1, x2, y2] em pixels da
            foto original). A numeração "1:", "2:"... segue a ordem da lista
            e é a mesma da coluna ``caixa`` do ``ocorrencias.csv``.
        labels (dict[int, str]): id da classe → rótulo.
        colors (list[str]): paleta "#RRGGBB" (``review_render.COLORS``).
        suggestions (list[dict] | None): sugestões pendentes do modelo
            (``class_id``, ``bbox_xyxy``, ``confidence`` 0..1); None = não desenha.
        scale (float): 0 < scale ≤ 16.

    Retorna:
        PIL.Image.Image: nova imagem RGB com as marcações.

    Erros:
        ValueError("scale deve ser um fator positivo.") se scale fora de (0, 16].

    Chamada por: ``_render_evidence`` (ZIP e laudo) e testes.
    """
    # Validação do fator de escala.
    if not (0 < scale <= 16):
        raise ValueError("scale deve ser um fator positivo.")
    # convert("RGB") sempre devolve uma cópia: a foto recebida não é alterada.
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    # Espessura: 3 px na resolução original, proporcional à escala, mínimo 2.
    thickness = max(2, round(3 * scale))
    # Fonte proporcional à largura desenhada (1/75), mínimo 14 px.
    font = _font(max(14, int(canvas.width / 75)))
    # 1. Caixas (linha cheia, placa colorida ACIMA da caixa).
    for number, box in enumerate(boxes, start=1):
        # Coordenadas da foto original → pixels da imagem desenhada.
        coordinates = [value * scale for value in box["bbox_xyxy"]]
        # Cor por classe; o módulo evita IndexError se houver mais classes que cores.
        # ATENCAO: mesma regra de cor de review_render e da página (paridade visual).
        color = colors[box["class_id"] % len(colors)]
        draw.rectangle(coordinates, outline=color, width=thickness)
        # EXTENSAO: texto da placa (ex.: incluir a nota da caixa): mude aqui.
        text = f"{number}: {_label(labels, box['class_id'])}"
        x, y = coordinates[:2]
        # Placa 22 px acima do topo da caixa; max(0, ...) mantém dentro da foto.
        # ATENCAO: o deslocamento de 22 px é fixo, mas a fonte cresce com a
        # largura da foto; em fotos largas a placa pode cobrir o topo da caixa.
        anchor = (x, max(0, y - 22))
        # Fundo da placa = retângulo que envolve o texto (textbbox), na cor da classe.
        draw.rectangle(draw.textbbox(anchor, text, font=font), fill=color)
        draw.text(anchor, text, fill=PLATE_TEXT, font=font)
    # 2. Sugestões do modelo (tracejado, placa escura ABAIXO da caixa).
    if suggestions:
        small = _font(max(13, int(canvas.width / 80)))
        # Traço de 10 px e vão de 6 px na resolução original (mínimos 4 e 3).
        dash = (max(4, round(10 * scale)), max(3, round(6 * scale)))
        for number, suggestion in enumerate(suggestions, start=1):
            coordinates = [value * scale for value in suggestion["bbox_xyxy"]]
            color = colors[suggestion["class_id"] % len(colors)]
            _dashed_rectangle(draw, coordinates, color, thickness, dash)
            # Confiança 0..1 → porcentagem inteira (ex.: 0.734 → 73).
            confidence = round(float(suggestion.get("confidence", 0)) * 100)
            text = f"IA {number}: {_label(labels, suggestion['class_id'])} {confidence}%"
            # Placa na borda inferior da caixa (coordinates[3] = y2), sem passar
            # da borda de baixo da foto.
            anchor = (coordinates[0], min(canvas.height - 22, coordinates[3]))
            draw.rectangle(draw.textbbox(anchor, text, font=small), fill=PLATE_TEXT)
            draw.text(anchor, text, fill=color, font=small)
    return canvas


# ---------------------------------------------------------------------------
# Decisões: página > registro; sugestões pendentes.
# ---------------------------------------------------------------------------

def _js_round(value: float) -> int:
    """Arredondamento da página (Math.round: meio para cima), não o do Python (meio par).

    O ``round`` do Python arredonda 2.5 para 2 (meio para o par); o
    ``Math.round`` do JavaScript dá 3. Para comparar caixas da mesma forma que
    a página, usa-se floor(valor + 0,5).

    Parâmetros:
        value (float): coordenada em pixels.

    Retorna:
        int: valor arredondado como no JavaScript.
    """
    return math.floor(value + 0.5)


def _same_box(box: dict, suggestion: dict) -> bool:
    """Diz se uma caixa aceita corresponde a uma sugestão do modelo.

    Quando o revisor aceita uma sugestão, a página cria uma caixa com as
    coordenadas arredondadas (``Math.round``). Mesma classe e as quatro
    coordenadas iguais após o arredondamento = a sugestão foi aceita.

    Parâmetros:
        box (dict): caixa com ``class_id`` e ``bbox_xyxy``.
        suggestion (dict): sugestão com ``class_id`` e ``bbox_xyxy``.

    Retorna:
        bool: True se forem a mesma caixa.

    ATENCAO: paridade com ``sameBox`` em ``review_templates/platform.js``.
    """
    return box["class_id"] == suggestion["class_id"] and all(
        _js_round(a) == _js_round(b) for a, b in zip(box["bbox_xyxy"], suggestion["bbox_xyxy"]))


def _pending_suggestions(candidates: list[dict], boxes: list[dict], dismissed: list) -> list[dict]:
    """Sugestão pendente = não aceita como caixa (mesma classe e geometria) nem descartada.

    Parâmetros:
        candidates (list[dict]): sugestões do modelo para a foto (com ``id``
            como "s001", ``class_id``, ``bbox_xyxy``, ``confidence``).
        boxes (list[dict]): caixas que valem para a foto.
        dismissed (list): ids das sugestões descartadas pelo revisor.

    Retorna:
        list[dict]: as sugestões ainda sem decisão, na ordem original.

    ATENCAO: paridade com ``pendingSuggestionsFor`` em ``platform.js``.
    """
    return [suggestion for suggestion in candidates
            if not any(_same_box(box, suggestion) for box in boxes) and suggestion.get("id") not in dismissed]


def _normalize_suggestions(suggestions: dict | None, class_count: int) -> dict[str, list[dict]]:
    """Aceita o mapa sha → lista ou o conteúdo inteiro de suggestions.json; rejeita sugestões inválidas.

    Parâmetros:
        suggestions (dict | None): ``{"<sha256>": [sugestão, ...]}`` ou o JSON
            inteiro de ``suggestions.json`` (que tem esse mapa em ``images``).
        class_count (int): número de classes ativas; ``class_id`` precisa
            estar em ``0..class_count-1``.

    Retorna:
        dict[str, list[dict]]: mapa sha256 → cópia da lista de sugestões
        ({} quando não há sugestões).

    Erros (ValueError): não é dicionário; valor de uma foto não é lista;
    sugestão sem ``class_id`` inteiro válido ("Sugestão com classe inválida."),
    ``bbox_xyxy`` sem 4 números finitos ("... coordenadas inválidas.") ou
    ``confidence`` fora de 0..1 ("... confiança inválida.").

    Chamada por: ``_resolve_entries``.
    """
    # Nada informado: nenhuma sugestão.
    if not suggestions:
        return {}
    if not isinstance(suggestions, dict):
        raise ValueError("suggestions deve ser um dicionário image_sha256 → lista de sugestões.")
    # Se veio o suggestions.json inteiro, usa o mapa em "images"; senão o próprio dicionário.
    mapping = suggestions.get("images") if isinstance(suggestions.get("images"), dict) else suggestions
    normalized = {}
    # Valida foto a foto e sugestão a sugestão.
    for digest, values in mapping.items():
        if not isinstance(values, list):
            raise ValueError(f"Sugestões de {digest} devem ser uma lista.")
        for value in values:
            # type(...) is not int rejeita também bool (True é int em Python) e float.
            if not isinstance(value, dict) or type(value.get("class_id")) is not int or not 0 <= value["class_id"] < class_count:
                raise ValueError("Sugestão com classe inválida.")
            # Coordenadas: lista de exatamente 4 números finitos (sem NaN/inf).
            coordinates = value.get("bbox_xyxy")
            if (not isinstance(coordinates, list) or len(coordinates) != 4
                    or any(type(item) not in (int, float) or not math.isfinite(item) for item in coordinates)):
                raise ValueError("Sugestão com coordenadas inválidas.")
            # Confiança ausente vira -1 e cai no erro abaixo.
            try:
                confidence = float(value.get("confidence", -1))
            except (TypeError, ValueError):
                raise ValueError("Sugestão com confiança inválida.") from None
            if not 0 <= confidence <= 1:
                raise ValueError("Sugestão com confiança inválida.")
            # EXTENSAO: nova validação de sugestão (ex.: id obrigatório): acrescente aqui.
        normalized[digest] = list(values)
    return normalized


def _resolve_entries(registry: dict, decisions: dict | None, class_count: int, suggestions: dict | None) -> list[dict]:
    """Versão interna de ``resolve_decisions`` com a lista das sugestões pendentes e a posição.

    Parâmetros:
        registry (dict): conteúdo do ``registry.json`` (usa ``images``: cada
            uma com ``image_sha256``, ``filename``, ``width``, ``height``,
            ``status``, ``boxes``, notas e aprovação humana).
        decisions (dict | None): mapa sha256 → decisão da página (ver
            ``resolve_decisions``); None/{} = só o registro.
        class_count (int): número de classes ativas.
        suggestions (dict | None): sugestões do modelo (ver ``_normalize_suggestions``).

    Retorna:
        list[dict]: uma entrada por foto, na ordem do registro, com:
        ``position`` (1, 2, ... = número da foto nos arquivos exportados),
        ``image`` (o item do registro), ``status``, ``boxes``, ``notes``,
        ``source`` ("revisao_humana" ou "registro"), ``confirmed_complete``,
        ``pending_suggestions`` (quantidade) e ``pending`` (a lista).

    Erros (ValueError): registro sem lista de imagens; imagem duplicada;
    decisão para foto fora do registro; status inválido; caixas inválidas
    (``validate_boxes``); positiva sem caixas; negativa com caixas; notas ou
    descartes com tipo errado; sugestões inválidas.

    Chamada por: ``resolve_decisions`` e ``_load_context``.
    """
    # 1. Validação do registro: lista de objetos, sem hash repetido.
    images = registry.get("images") if isinstance(registry, dict) else None
    if not isinstance(images, list) or any(not isinstance(item, dict) for item in images):
        raise ValueError("O registro deve conter uma lista de imagens válida.")
    known = [item.get("image_sha256") for item in images]
    if len(set(known)) != len(known):
        raise ValueError("Imagem duplicada no registro de revisão.")
    # 2. Validação das decisões: dicionário cujas chaves são hashes do registro.
    decisions = decisions or {}
    if not isinstance(decisions, dict):
        raise ValueError("decisions deve ser um dicionário image_sha256 → decisão.")
    unknown = set(decisions) - set(known)
    if unknown:
        raise ValueError(f"Decisão para imagem fora do registro: {sorted(unknown)[0]}.")
    # 3. Sugestões validadas (mapa sha → lista).
    candidates = _normalize_suggestions(suggestions, class_count)
    entries = []
    # 4. Uma entrada por foto; position começa em 1 e é o "numero_foto".
    # ATENCAO: a posição vem da ORDEM do registro; reordenar o registro muda
    # a numeração das fotos, dos nomes das evidências e das figuras.
    for position, item in enumerate(images, start=1):
        digest = item["image_sha256"]
        decision = decisions.get(digest)
        # 4a. Foto COM decisão da página: a decisão humana prevalece sobre o registro.
        if decision is not None:
            if not isinstance(decision, dict):
                raise ValueError(f"Decisão de {item.get('filename')} deve ser um objeto.")
            status = decision.get("status")
            # EXTENSAO: novo status de decisão: DECISION_STATUSES (via
            # review_data.IMAGE_STATUSES) e STATUS_LABELS.
            if status not in DECISION_STATUSES:
                raise ValueError(f"Status inválido na decisão de {item.get('filename')}: {status!r}.")
            boxes = decision.get("boxes", [])
            # Caixas dentro da foto, classe válida, até 2000 (review_data.validate_boxes).
            validate_boxes(boxes, item["width"], item["height"], class_count)
            # Coerência status × caixas: positiva precisa de caixa; negativa não pode ter.
            if status == "positive" and not boxes:
                raise ValueError(f"Decisão positiva sem caixas: {item.get('filename')}.")
            if status == "negative" and boxes:
                raise ValueError(f"Decisão negativa com caixas: {item.get('filename')}.")
            notes = decision.get("notes") or ""
            dismissed = decision.get("dismissed_suggestions") or []
            if not isinstance(notes, str) or not isinstance(dismissed, list):
                raise ValueError(f"Notas ou sugestões descartadas inválidas em {item.get('filename')}.")
            # "confirmed_complete": o revisor marcou que demarcou tudo na foto.
            source, confirmed = "revisao_humana", decision.get("confirmed_complete") is True
        # 4b. Foto SEM decisão: vale o que está gravado no registro.
        else:
            status = item.get("status")
            # No registro "pending" não é status válido (só nas decisões).
            if status not in IMAGE_STATUSES:
                raise ValueError(f"Status inválido no registro: {item.get('filename')}.")
            boxes = item.get("boxes", [])
            validate_boxes(boxes, item["width"], item["height"], class_count)
            # Notas: as do revisor humano têm prioridade sobre as notas gerais.
            notes = item.get("human_notes") or item.get("notes") or ""
            dismissed, source, confirmed = [], "registro", item.get("human_approved") is True
        # 5. Sugestões ainda sem decisão para esta foto.
        pending = _pending_suggestions(candidates.get(digest, []), boxes, dismissed)
        # EXTENSAO: novo campo por foto disponível para CSV/laudo: acrescente
        # aqui (e em "keys" de resolve_decisions se for público).
        entries.append({"position": position, "image": item, "status": status, "boxes": boxes, "notes": notes,
                        "source": source, "confirmed_complete": confirmed, "pending_suggestions": len(pending),
                        "pending": pending})
    return entries


def resolve_decisions(registry: dict, decisions: dict | None, class_count: int,
                      suggestions: dict | None = None) -> list[dict]:
    """Para cada foto do registro, o status e as caixas que valem nesta exportação.

    ``decisions`` (chave ``image_sha256``) é o objeto exportado pela página:
    ``{"status", "boxes", "notes", "confirmed_complete", "dismissed_suggestions"}``.
    Foto com decisão usa as caixas da decisão (origem ``revisao_humana``);
    sem decisão, as do registro (origem ``registro``). Caixas são validadas
    com ``validate_boxes``; status fora de positive/negative/ambiguous/
    excluded/pending levanta ``ValueError``.

    ``suggestions`` (opcional): mapa ``image_sha256`` → sugestões do modelo
    (``browser_data.json["images"][i]["model_suggestions"]`` ou
    ``suggestions.json["images"]``). ``pending_suggestions`` conta as que não
    foram aceitas como caixa nem descartadas.

    Parâmetros:
        registry (dict): conteúdo do ``registry.json``.
        decisions (dict | None): mapa sha256 → decisão da página.
        class_count (int): número de classes ativas (``len(detection_names(...))``).
        suggestions (dict | None): sugestões do modelo.

    Retorna:
        list[dict]: uma entrada por foto, na ordem do registro, com as chaves
        ``image``, ``status``, ``boxes``, ``notes``, ``source``,
        ``pending_suggestions`` (int) e ``confirmed_complete`` (bool).

    Erros: os mesmos ``ValueError`` de ``_resolve_entries``.

    Chamada por: ``platform_server`` (contagem de sugestões pendentes na lista
    de inspeções) e testes.
    """
    # Só as chaves públicas (sem "position" e sem a lista "pending").
    # EXTENSAO: expor mais um campo da entrada: acrescente o nome aqui.
    keys = ("image", "status", "boxes", "notes", "source", "pending_suggestions", "confirmed_complete")
    return [{key: entry[key] for key in keys} for entry in _resolve_entries(registry, decisions, class_count, suggestions)]


# ---------------------------------------------------------------------------
# Contexto comum: registro, taxonomia, sugestões e fotos conferidas.
# ---------------------------------------------------------------------------

@dataclass
class _Context:
    """Tudo o que o ZIP e o laudo precisam, carregado e validado uma única vez.

    Montado por ``_load_context``; lido por ``build_evidence_package``,
    ``occurrences_csv``, ``build_report`` e as funções auxiliares.

    Atributos:
        registry (dict): conteúdo do ``registry.json``.
        registry_path (Path): caminho do registro.
        registry_sha256 (str): hash do arquivo do registro (vai nos manifestos).
        version (str): versão da revisão (``registry["version"]`` ou o nome da pasta).
        taxonomy_sha256 (str): hash de ``configs/taxonomy.json``.
        classes (list[dict]): classes ativas (fase 1) da taxonomia.
        labels (dict[int, str]): id → rótulo das classes ativas.
        colors (list[str]): paleta de ``review_render.COLORS``.
        entries (list[dict]): saída de ``_resolve_entries`` (todas as fotos).
        decisions_count (int): quantas decisões da página foram recebidas.
        suggestion_model (dict | None): metadados do modelo que gerou as sugestões.
        suggestion_note (str | None): aviso quando um suggestions.json foi ignorado.
        reviewers (list[str]): revisores do registro, sem repetição.

    EXTENSAO: novo dado comum ao ZIP e ao laudo: acrescente o atributo aqui
    (com valor padrão se for opcional) e preencha em ``_load_context``.
    """

    registry: dict
    registry_path: Path
    registry_sha256: str
    version: str
    taxonomy_sha256: str
    classes: list[dict]                 # classes ativas (fase 1) com criterion/reference_severity
    labels: dict[int, str]
    colors: list[str]
    entries: list[dict]
    decisions_count: int
    suggestion_model: dict | None       # label, conf, weights_sha256... de suggestions.json
    suggestion_note: str | None = None  # quando um suggestions.json foi ignorado
    reviewers: list[str] = field(default_factory=list)


def _suggestion_file(registry_path: Path, registry_sha256: str) -> tuple[dict | None, str | None]:
    """suggestions.json ao lado do registro, só se pertencer a esta versão exata.

    Parâmetros:
        registry_path (Path): caminho do ``registry.json``.
        registry_sha256 (str): hash atual do registro.

    Retorna:
        tuple: ``(dados, None)`` se o arquivo existe e é desta versão;
        ``(None, None)`` se não existe; ``(None, aviso)`` se é de outra versão
        do registro ou de outro schema (o aviso vai para o manifesto/laudo).

    Erros: JSON malformado levanta ``json.JSONDecodeError`` (ValueError).

    ATENCAO: sugestões de outra versão do registro poderiam apontar fotos ou
    caixas que não existem mais; por isso o hash do registro tem de bater.
    """
    path = registry_path.parent / "suggestions.json"
    if not path.is_file():
        return None, None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("registry_sha256") != registry_sha256:
        return None, "suggestions.json ao lado do registro pertence a outra versão e foi ignorado."
    return data, None


def _model_meta(data: dict) -> dict:
    """Só os metadados do modelo de um suggestions.json (sem as sugestões em si).

    Parâmetros:
        data (dict): conteúdo do suggestions.json.

    Retorna:
        dict: as chaves de ``keys`` presentes em ``data`` (rótulo do modelo,
        limiar, hash dos pesos, se é piloto, calibrador, data, aviso).
    """
    # EXTENSAO: novo metadado do modelo no manifesto/laudo: acrescente a chave aqui.
    keys = ("model_label", "conf", "weights_sha256", "pilot", "calibrator", "created_at", "warning")
    return {key: data.get(key) for key in keys if key in data}


def _load_context(config: ProjectConfig, registry_path: Path, decisions: dict | None, suggestions: dict | None) -> _Context:
    """Lê e valida registro, taxonomia, sugestões e decisões; monta o ``_Context``.

    Parâmetros:
        config (ProjectConfig): usa ``taxonomy_path``.
        registry_path (Path): ``data/reviews/<inspeção>/registry.json``.
        decisions (dict | None): mapa sha256 → decisão da página.
        suggestions (dict | None): sugestões passadas pelo chamador; None =
            procurar ``suggestions.json`` ao lado do registro.

    Retorna:
        _Context: pronto para o ZIP/laudo.

    Erros (ValueError): schema do registro diferente de ``REVIEW_SCHEMA``;
    taxonomia mudou desde a criação do registro; e todos os de
    ``_resolve_entries``. Arquivo ausente levanta OSError.

    Chamada por: ``build_evidence_package``, ``occurrences_csv``, ``build_report``.
    """
    # 1. Registro: JSON com o schema esperado.
    registry_path = Path(registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or registry.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError("Schema inválido no registro de revisão.")
    # 2. Taxonomia: o hash gravado no registro tem de ser o da taxonomia atual.
    # ATENCAO: ids de classe são contrato; com taxonomia diferente o id 3 pode
    # ser outra classe. A migração (taxonomy_migration) resolve isso antes.
    taxonomy_sha256 = file_hash(config.taxonomy_path)
    if registry.get("taxonomy_sha256") != taxonomy_sha256:
        raise ValueError("Taxonomia mudou: migre a revisão antes de exportar.")
    taxonomy = load_taxonomy(config.taxonomy_path)
    # Número de classes ativas (detection_names também confere ids consecutivos).
    class_count = len(detection_names(taxonomy))
    # Só as classes da fase 1 (as que o modelo e a revisão usam).
    classes = [item for item in taxonomy["classes"] if item["phase"] == 1]
    registry_sha256 = file_hash(registry_path)
    # Sugestões: o que veio por parâmetro; senão, suggestions.json desta versão do registro.
    # Três casos:
    # - nada passado: lê o suggestions.json (se for desta versão) e seus metadados;
    # - passado o suggestions.json inteiro (tem "images"): metadados dele mesmo;
    # - passado só o mapa sha → lista: metadados do suggestions.json em disco, se houver.
    suggestion_note, model = None, None
    if suggestions is None:
        file_data, suggestion_note = _suggestion_file(registry_path, registry_sha256)
        if file_data is not None:
            suggestions, model = file_data.get("images", {}), _model_meta(file_data)
    elif isinstance(suggestions, dict) and isinstance(suggestions.get("images"), dict):
        model = _model_meta(suggestions)
    else:
        file_data, _ = _suggestion_file(registry_path, registry_sha256)
        model = _model_meta(file_data) if file_data else None
    # 3. Decisões aplicadas sobre o registro (valida tudo).
    entries = _resolve_entries(registry, decisions, class_count, suggestions)
    # 4. Revisores: o do feedback geral + o de cada foto, sem repetição, na ordem.
    reviewers = []
    for value in [registry.get("human_feedback", {}).get("reviewer_id")] + [item.get("human_reviewer") for item in registry["images"]]:
        if isinstance(value, str) and value.strip() and value.strip() not in reviewers:
            reviewers.append(value.strip())
    # 5. Contexto. version: a do registro ou, na falta, o nome da pasta da revisão.
    return _Context(registry=registry, registry_path=registry_path, registry_sha256=registry_sha256,
                    version=str(registry.get("version") or registry_path.parent.name), taxonomy_sha256=taxonomy_sha256,
                    classes=classes, labels={item["id"]: item["label"] for item in classes}, colors=_palette(),
                    entries=entries, decisions_count=len(decisions or {}), suggestion_model=model,
                    suggestion_note=suggestion_note, reviewers=reviewers)


def _checked_source(config: ProjectConfig, item: dict) -> Path:
    """Foto original intacta (hash e dimensões) ou ValueError; nunca desenha sobre foto alterada.

    Parâmetros:
        config (ProjectConfig): usa ``root`` para reancorar caminhos gravados
            em outra máquina (``io.resolve_local_path``).
        item (dict): imagem do registro (``source_path``, ``filename``,
            ``image_sha256``, ``width``, ``height``).

    Retorna:
        Path: caminho local da foto conferida.

    Erros (ValueError): foto não encontrada; SHA-256 diferente ("A foto
    original mudou"); largura/altura diferentes das do registro.

    ATENCAO: as coordenadas das caixas valem só para a foto exata do
    registro; nunca remova esta conferência.
    """
    source = resolve_local_path(item["source_path"], config.root)
    if not source.is_file():
        raise ValueError(f"Foto original não encontrada: {item['filename']}.")
    if file_hash(source) != item["image_sha256"]:
        raise ValueError(f"A foto original mudou: {item['filename']}.")
    # review_image_size: tamanho em pixels sem aplicar rotação EXIF.
    if review_image_size(source) != (item["width"], item["height"]):
        raise ValueError(f"Dimensões do registro divergem da foto: {item['filename']}.")
    return source


def _render_evidence(source: Path, entry: dict, context: _Context, max_side: int | None, include_suggestions: bool) -> Image.Image:
    """Foto (reduzida a ``max_side`` quando pedido) com as caixas desenhadas.

    Parâmetros:
        source (Path): foto já conferida por ``_checked_source``.
        entry (dict): entrada de ``_resolve_entries`` (``boxes``, ``pending``).
        context (_Context): rótulos e cores.
        max_side (int | None): lado maior máximo em pixels; None = original.
        include_suggestions (bool): desenha as sugestões pendentes tracejadas.

    Retorna:
        PIL.Image.Image: imagem RGB nova (quem chama deve fechá-la).

    Chamada por: ``build_evidence_package`` e ``_build_report_files``.
    """
    # Lê a foto e fecha o arquivo logo (convert cria uma cópia em memória).
    with Image.open(source) as original:
        image = original.convert("RGB")
    # Redução só quando a foto é maior que max_side; nunca amplia.
    # scale = fator aplicado às coordenadas das caixas em draw_boxes.
    scale = 1.0
    if max_side and max(image.size) > max_side:
        scale = max_side / max(image.size)
        # LANCZOS: melhor qualidade na redução; max(1, ...) evita lado zero.
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    return draw_boxes(image, entry["boxes"], context.labels, context.colors,
                      suggestions=entry["pending"] if include_suggestions else None, scale=scale)


def _filter_entries(entries: list[dict], only: str) -> list[dict]:
    """Seleciona as fotos que entram no ZIP de evidências.

    Parâmetros:
        entries (list[dict]): todas as entradas (``_Context.entries``).
        only (str): "all" (todas), "decided" (só fotos com decisão humana
            nesta exportação, exceto rascunhos "pending") ou "positive" (só
            fotos com status positivo, de qualquer origem).

    Retorna:
        list[dict]: nova lista, mantendo a ordem (e a posição original).

    Erros: ValueError se ``only`` não estiver em ``ONLY_CHOICES``.
    """
    if only not in ONLY_CHOICES:
        raise ValueError(f"only deve ser um de {ONLY_CHOICES}.")
    if only == "positive":
        return [entry for entry in entries if entry["status"] == "positive"]
    # ATENCAO: fotos aprovadas no próprio registro (human_approved) NÃO contam
    # como "decided" aqui; só decisões enviadas pela página nesta exportação.
    if only == "decided":
        return [entry for entry in entries if entry["source"] == "revisao_humana" and entry["status"] != "pending"]
    # EXTENSAO: novo filtro: acrescente um "if only == ..." acima desta linha
    # e o nome em ONLY_CHOICES.
    return list(entries)


# ---------------------------------------------------------------------------
# Utilidades de formatação e arquivos.
# ---------------------------------------------------------------------------

def _number(value: float):
    """Coordenadas inteiras sem ".0"; fracionárias com duas casas.

    Parâmetros:
        value (float): coordenada em pixels.

    Retorna:
        int | float: ex.: 120.0 → 120; 120.456 → 120.46.
    """
    return int(value) if float(value).is_integer() else round(float(value), 2)


def _utc_now() -> datetime:
    """Data e hora atuais em UTC (fuso explícito).

    Retorna:
        datetime: com ``tzinfo=UTC``; usada no carimbo dos nomes e em
        ``generated_at`` (formato ISO, ex.: "2026-10-12T14:03:07+00:00").
    """
    return datetime.now(timezone.utc)


def _safe_name(value: str) -> str:
    """Trecho de nome de arquivo seguro em qualquer sistema.

    Troca toda sequência de caracteres fora de ``A-Z a-z 0-9 . _ -`` por "_"
    (acentos e espaços também), tira "." e "_" das pontas e corta em 80
    caracteres.

    Parâmetros:
        value (str): texto livre (versão da revisão, nome da foto).

    Retorna:
        str: o nome seguro, ou "revisao" se sobrar vazio.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")[:80] or "revisao"


def _unique_path(directory: Path, stem: str, suffix: str = "") -> Path:
    """Nunca sobrescreva uma exportação anterior gerada no mesmo segundo.

    Parâmetros:
        directory (Path): pasta de destino.
        stem (str): nome base (ex.: "20261012T140307Z_evidencias_v3").
        suffix (str): extensão com ponto (".zip") ou "" para pasta.

    Retorna:
        Path: ``stem+suffix`` se livre; senão ``stem_2``, ``stem_3``... (o
        primeiro que não existir). Não cria nada.
    """
    candidate = directory / f"{stem}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = directory / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def _csv_text(header: list[str], rows: list[list]) -> str:
    """Monta um CSV em memória (separador vírgula, aspas quando necessário).

    Parâmetros:
        header (list[str]): nomes das colunas.
        rows (list[list]): linhas, na mesma ordem das colunas.

    Retorna:
        str: o texto do CSV (fins de linha "\\r\\n", padrão do módulo csv).
        O BOM do Excel é acrescentado por quem grava (``small(..., bom=True)``).
    """
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()


def _evidence_name(entry: dict) -> str:
    """Nome da evidência no ZIP: ``NNN_<nome seguro da foto>.jpg``.

    Parâmetros:
        entry (dict): entrada com ``position`` e ``image.filename``.

    Retorna:
        str: ex.: posição 7 e "DJI 0042.JPG" → "007_DJI_0042.jpg".

    ATENCAO: usado em dois lugares que precisam casar (nome do arquivo no ZIP
    e coluna ``evidencia`` do fotos.csv).
    """
    filename = Path(entry["image"]["filename"])
    return f"{entry['position']:03d}_{_safe_name(filename.stem)}.jpg"


def _occurrence_rows(entries: list[dict], labels: dict[int, str]) -> list[list]:
    """Linhas do ``ocorrencias.csv``: uma por caixa, de todas as fotos recebidas.

    Parâmetros:
        entries (list[dict]): entradas selecionadas.
        labels (dict[int, str]): id → rótulo da classe.

    Retorna:
        list[list]: valores na ordem de ``OCCURRENCE_COLUMNS``.

    ATENCAO: inclui caixas de fotos com qualquer status (inclusive
    ambíguas/rascunho); a coluna ``status`` diz qual. Sugestões do modelo
    nunca entram aqui.
    """
    rows = []
    for entry in entries:
        item = entry["image"]
        # number = número da caixa dentro da foto (o mesmo da placa "n:" da evidência).
        for number, box in enumerate(entry["boxes"], start=1):
            x1, y1, x2, y2 = (_number(value) for value in box["bbox_xyxy"])
            # EXTENSAO: coluna nova = valor nesta lista, na MESMA posição do
            # nome em OCCURRENCE_COLUMNS.
            rows.append([entry["position"], item["image_sha256"], item["filename"], entry["status"], entry["source"], number,
                         box["class_id"], labels.get(box["class_id"], ""), x1, y1, x2, y2, box.get("note", "") or ""])
    return rows


def _photo_rows(entries: list[dict]) -> list[list]:
    """Linhas do ``fotos.csv``: uma por foto selecionada.

    Parâmetros:
        entries (list[dict]): entradas selecionadas.

    Retorna:
        list[list]: valores na ordem de ``PHOTO_COLUMNS``; ``paginas`` vira
        texto "3;4;7" e ``evidencia`` o caminho dentro do ZIP.
    """
    rows = []
    for entry in entries:
        item = entry["image"]
        # EXTENSAO: coluna nova = valor nesta lista, na MESMA posição do nome
        # em PHOTO_COLUMNS.
        rows.append([entry["position"], item["image_sha256"], item["filename"], item["width"], item["height"],
                     ";".join(map(str, item.get("pages", []))), item.get("building_group", ""), item.get("scene_group", ""),
                     entry["status"], entry["source"], len(entry["boxes"]), entry["pending_suggestions"], entry["notes"],
                     f"evidencias/{_evidence_name(entry)}"])
    return rows


def _counts(entries: list[dict], labels: dict[int, str]) -> dict:
    """Totais usados no manifesto do ZIP e no laudo.

    Parâmetros:
        entries (list[dict]): entradas consideradas.
        labels (dict[int, str]): id → rótulo da classe.

    Retorna:
        dict com ``images``; ``by_status`` (os cinco status, zero quando
        ausentes); ``boxes`` (todas as caixas); ``confirmed_boxes`` e
        ``confirmed_boxes_by_class`` (só caixas de fotos POSITIVAS, por
        rótulo); ``pending_suggestions``; ``decided_by_human`` (fotos com
        decisão da página, inclusive rascunhos "pending").
    """
    status = Counter(entry["status"] for entry in entries)
    # Caixas confirmadas = caixas de fotos com status "positive", contadas por classe.
    confirmed = Counter(box["class_id"] for entry in entries if entry["status"] == "positive" for box in entry["boxes"])
    # EXTENSAO: novo status: acrescente na tupla de by_status (ordem de exibição).
    # EXTENSAO: novo total: acrescente uma chave neste dicionário.
    return {"images": len(entries), "by_status": {key: status.get(key, 0) for key in ("positive", "negative", "ambiguous", "excluded", "pending")},
            "boxes": sum(len(entry["boxes"]) for entry in entries), "confirmed_boxes": sum(confirmed.values()),
            "confirmed_boxes_by_class": {labels[key]: confirmed[key] for key in sorted(confirmed)},
            "pending_suggestions": sum(entry["pending_suggestions"] for entry in entries),
            "decided_by_human": sum(entry["source"] == "revisao_humana" for entry in entries)}


def _readme(context: _Context, generated_at: str, parameters: dict) -> str:
    """Texto do ``LEIA-ME.txt`` do ZIP: identificação, aviso e conteúdo.

    Parâmetros:
        context (_Context): versão, nome e hash do registro.
        generated_at (str): data ISO UTC da geração.
        parameters (dict): parâmetros da exportação (only, max_side...).

    Retorna:
        str: texto com linhas separadas por "\\n".

    EXTENSAO: arquivo novo no ZIP: descreva-o na seção "CONTEÚDO" abaixo.
    ATENCAO: o AVISO diz que o pacote NÃO é laudo; não remova.
    """
    return "\n".join([
        "PACOTE DE EVIDÊNCIAS — DroneCamp Vision", "",
        f"Inspeção: {review_name(context.version, context.registry.get('display_name'))} ({context.version})", f"Registro: {context.registry_path.name} (SHA-256 {context.registry_sha256})",
        f"Gerado em: {generated_at}", f"Parâmetros: {json.dumps(parameters, ensure_ascii=False)}", "",
        "AVISO",
        "As caixas desenhadas nas fotos são evidências visuais revisáveis. Este pacote NÃO é laudo técnico.",
        "Nenhuma gravidade, causa, medida ou diagnóstico foi atribuída automaticamente.",
        'Origem "revisao_humana": caixas vindas da revisão humana na plataforma; "registro": caixas do registro',
        "de propostas, sem decisão humana nesta exportação.",
        "Sugestões automáticas do modelo (quando incluídas) aparecem tracejadas e não constituem ocorrências.",
        "Caixas em fotos semelhantes podem repetir o mesmo defeito; contagens são por caixa, não por defeito físico.", "",
        "CONTEÚDO",
        "evidencias/NNN_<arquivo>.jpg  foto com as caixas (NNN = posição da foto no registro)",
        "ocorrencias.csv               uma linha por caixa (" + ", ".join(OCCURRENCE_COLUMNS) + ")",
        "fotos.csv                     uma linha por foto (status, origem, caixas, sugestões pendentes, notas)",
        "manifesto.json                SHA-256 de cada arquivo, do registro e da taxonomia; parâmetros e contagens",
        "", "CSV em UTF-8 com BOM (abre direto no Excel).", ""])


# ---------------------------------------------------------------------------
# Pacote de evidências (ZIP).
# ---------------------------------------------------------------------------

class _HashingWriter:
    """Encaminha a escrita ao zip calculando SHA-256 e tamanho, sem guardar a imagem em memória.

    Objeto "tipo arquivo" passado ao ``Image.save``: cada bloco de bytes vai
    direto para a entrada do ZIP e, ao mesmo tempo, atualiza o hash e o
    tamanho que entram no ``manifesto.json``.

    Atributos:
        stream: a entrada aberta com ``archive.open(info, "w")``.
        digest: objeto sha256 acumulado.
        size (int): bytes escritos.
    """

    def __init__(self, stream):
        """Parâmetros: stream: destino com ``write`` (entrada do ZIP). Retorna: None."""
        self.stream, self.digest, self.size = stream, sha256(), 0

    def write(self, data) -> int:
        """Escreve ``data`` (bytes) no destino, atualiza hash e tamanho.

        Retorna:
            int: quantidade de bytes escritos (contrato de arquivo do Python).
        """
        self.stream.write(data)
        self.digest.update(data)
        self.size += len(data)
        return len(data)

    def flush(self) -> None:
        """Nada a fazer (o ZIP descarrega ao fechar a entrada); exigido pelo Pillow. Retorna: None."""
        pass


# ENTRADA: POST /api/export/evidence (botão do ZIP de evidências na seção Laudos).
def build_evidence_package(config: ProjectConfig, registry_path: Path, output_dir: Path, *, decisions: dict | None = None,
                           suggestions: dict | None = None, max_side: int | None = 2048, only: str = "all",
                           include_suggestions: bool = False) -> dict:
    """ZIP ``<stamp UTC>_evidencias_<versão>.zip`` em ``output_dir`` com as fotos demarcadas.

    ``only``: "all" (todas as fotos), "decided" (só fotos com decisão humana,
    exceto rascunhos pendentes) ou "positive" (só positivas). ``max_side``
    reduz o lado maior das evidências (None = resolução original).
    ``include_suggestions`` desenha tracejadas as sugestões do modelo ainda
    pendentes (nunca entram no CSV de ocorrências).

    O zip é escrito uma imagem por vez (nada é montado inteiro em memória).
    Devolve ``{"zip", "images", "boxes", "bytes", "sha256"}``.

    Parâmetros:
        config (ProjectConfig): configuração (taxonomia, raiz do projeto).
        registry_path (Path): ``registry.json`` da inspeção.
        output_dir (Path): pasta de saída (criada se não existir); o servidor
            usa ``<pasta da revisão>/exportacoes``.
        decisions (dict | None): mapa sha256 → decisão da página.
        suggestions (dict | None): sugestões; None = ``suggestions.json`` da versão.
        max_side (int | None): inteiro ≥ 16 ou None (padrão 2048 px).
        only (str): filtro (``ONLY_CHOICES``).
        include_suggestions (bool): desenha sugestões pendentes.

    Retorna:
        dict: ``zip`` (caminho), ``images`` (fotos no ZIP), ``boxes`` (linhas
        do ocorrencias.csv), ``bytes`` (tamanho do ZIP), ``sha256`` (do ZIP).

    Erros (ValueError): ``max_side`` inválido; ``only`` inválido; registro,
    taxonomia, decisões ou sugestões inválidos (``_load_context``); foto
    ausente ou alterada (``_checked_source``). Qualquer erro apaga o ZIP parcial.

    Arquivos dentro do ZIP: ``evidencias/NNN_<foto>.jpg``, ``ocorrencias.csv``,
    ``fotos.csv``, ``LEIA-ME.txt`` e ``manifesto.json`` (por último, com o hash
    de todos os anteriores).
    """
    # 1. Validação de max_side (type(...) is not int recusa bool e float).
    if max_side is not None and (type(max_side) is not int or max_side < 16):
        raise ValueError("max_side deve ser None ou um inteiro ≥ 16.")
    # 2. Contexto validado e fotos selecionadas pelo filtro.
    context = _load_context(config, registry_path, decisions, suggestions)
    selected = _filter_entries(context.entries, only)
    # 3. Carimbo de tempo único para nome, manifesto e datas internas do ZIP.
    generated = _utc_now()
    generated_at = generated.isoformat(timespec="seconds")
    # Parâmetros gravados no LEIA-ME e no manifesto (rastreabilidade).
    # EXTENSAO: novo parâmetro da exportação: acrescente na assinatura e aqui.
    parameters = {"only": only, "max_side": max_side, "include_suggestions": include_suggestions,
                  "decisions": context.decisions_count}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Nome: AAAAMMDDTHHMMSSZ_evidencias_<versão>.zip, com _2, _3... se já existir.
    path = _unique_path(output_dir, f"{generated.strftime('%Y%m%dT%H%M%SZ')}_evidencias_{_safe_name(context.version)}", ".zip")
    # Hash e tamanho de cada arquivo do ZIP, na ordem em que entram (vai no manifesto).
    files: OrderedDict[str, dict] = OrderedDict()
    # Data/hora (ano, mês, dia, hora, min, seg) gravada em todas as entradas do ZIP.
    stamp = generated.timetuple()[:6]

    def small(archive: zipfile.ZipFile, name: str, text: str, bom: bool = False) -> None:
        """Grava um arquivo de texto pequeno no ZIP e registra hash e tamanho.

        Parâmetros:
            archive (zipfile.ZipFile): ZIP aberto para escrita.
            name (str): caminho dentro do ZIP (ex.: "fotos.csv").
            text (str): conteúdo.
            bom (bool): True prefixa o BOM UTF-8 ("\\ufeff") para o Excel
                reconhecer acentos.

        Retorna: None (atualiza ``files``).
        """
        data = (("\ufeff" if bom else "") + text).encode("utf-8")
        info = zipfile.ZipInfo(name, date_time=stamp)
        # Texto comprime bem (DEFLATED); 0o644 << 16 = permissão rw-r--r-- ao extrair no Linux/macOS.
        info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
        archive.writestr(info, data)
        files[name] = {"sha256": sha256(data).hexdigest(), "bytes": len(data)}

    # 4. Escrita do ZIP. Qualquer exceção (inclusive Ctrl+C: BaseException)
    # apaga o arquivo parcial e repropaga o erro.
    try:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            # 4a. Evidências, uma foto por vez (memória limitada a uma imagem).
            for entry in selected:
                # Confere hash e dimensões antes de desenhar.
                source = _checked_source(config, entry["image"])
                evidence = _render_evidence(source, entry, context, max_side, include_suggestions)
                name = f"evidencias/{_evidence_name(entry)}"
                info = zipfile.ZipInfo(name, date_time=stamp)
                info.compress_type, info.external_attr = zipfile.ZIP_STORED, 0o644 << 16  # JPEG já é comprimido.
                # O JPEG é codificado direto dentro do ZIP, passando pelo _HashingWriter.
                # EXTENSAO: qualidade/formato das evidências: mude o "JPEG", quality=92
                # (e a extensão em _evidence_name se trocar o formato).
                with archive.open(info, "w") as stream:
                    writer = _HashingWriter(stream)
                    evidence.save(writer, "JPEG", quality=92)
                files[name] = {"sha256": writer.digest.hexdigest(), "bytes": writer.size,
                               "width": evidence.width, "height": evidence.height}
                evidence.close()
            # 4b. CSVs (com BOM para o Excel) e LEIA-ME.
            occurrence_rows = _occurrence_rows(selected, context.labels)
            small(archive, "ocorrencias.csv", _csv_text(OCCURRENCE_COLUMNS, occurrence_rows), bom=True)
            small(archive, "fotos.csv", _csv_text(PHOTO_COLUMNS, _photo_rows(selected)), bom=True)
            small(archive, "LEIA-ME.txt", _readme(context, generated_at, parameters))
            # EXTENSAO: arquivo novo no ZIP: chame small(archive, "nome", texto) aqui,
            # ANTES do manifesto, para o hash dele entrar em "files"; descreva-o em _readme.
            # 4c. Manifesto por último: hashes de todos os arquivos acima, do
            # registro e da taxonomia, parâmetros, contagens e modelo das sugestões.
            # EXTENSAO: novo campo no manifesto do ZIP: acrescente neste dicionário.
            manifest = {"schema_version": 1, "kind": "pacote_de_evidencias", "generated_at": generated_at,
                        "generator": "dronecamp_ia.exports.build_evidence_package", "version": context.version,
                        "registry_file": context.registry_path.name, "registry_sha256": context.registry_sha256,
                        "taxonomy_sha256": context.taxonomy_sha256,
                        "taxonomy_version": context.registry.get("taxonomy_version"), "parameters": parameters,
                        "counts": _counts(selected, context.labels), "suggestion_model": context.suggestion_model,
                        "suggestion_note": context.suggestion_note, "files": dict(files),
                        "note": "Caixas são evidências revisáveis, não laudo. Sem gravidade ou diagnóstico automático."}
            small(archive, "manifesto.json", json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
    except BaseException:
        path.unlink(missing_ok=True)  # Nada de zip parcial.
        raise
    # 5. Resumo para o servidor (ele devolve à página o caminho relativo).
    return {"zip": str(path), "images": len(selected), "boxes": len(occurrence_rows), "bytes": path.stat().st_size,
            "sha256": file_hash(path)}


def occurrences_csv(config: ProjectConfig, registry_path: Path, decisions: dict | None = None) -> str:
    """Texto do ``ocorrencias.csv`` (todas as fotos, uma linha por caixa), sem abrir as fotos.

    Codifique em UTF-8 com BOM (``utf-8-sig``) ao gravar ou enviar, como no zip.

    Parâmetros:
        config (ProjectConfig): configuração.
        registry_path (Path): ``registry.json``.
        decisions (dict | None): mapa sha256 → decisão da página.

    Retorna:
        str: CSV com cabeçalho ``OCCURRENCE_COLUMNS`` (sem BOM). Igual ao
        ``ocorrencias.csv`` do ZIP com ``only="all"``.

    Erros: os ``ValueError`` de ``_load_context``.

    Chamada por: testes (não há rota no servidor hoje).
    """
    # suggestions=None: sugestões não afetam as linhas (nunca são ocorrências).
    context = _load_context(config, registry_path, decisions, None)
    return _csv_text(OCCURRENCE_COLUMNS, _occurrence_rows(context.entries, context.labels))


# ---------------------------------------------------------------------------
# Laudo técnico preliminar (HTML / JSON / PDF).
# ---------------------------------------------------------------------------

# ENTRADA: POST /api/report (botão de laudo na seção Laudos). Monta laudo.json e chama
# laudo_abnt.render_html/write_pdf. ATENCAO: gravidade, diagnóstico e recomendação ficam
# "pendente de validação técnica"; nunca preencha esses campos por inferência do modelo.
def build_report(config: ProjectConfig, registry_path: Path, output_dir: Path, *, decisions: dict | None = None,
                 suggestions: dict | None = None, reviewer: str = "", title: str | None = None, client: str = "",
                 site: str = "", inspector: str = "", notes: str = "", details: dict | None = None, pdf: bool = True,
                 max_side: int = 1400, only_positive_images: bool = True) -> dict:
    """Pasta ``<stamp UTC>_laudo_<versão>/`` com ``laudo.html``, ``laudo.json``, ``laudo.pdf`` e ``manifesto.json``.

    O laudo segue a estrutura ABNT de ``laudo_abnt`` (capa, folha de rosto,
    sumário, seções numeradas, referências e anexos) e a lista de verificação do
    laudo CEASA: um item por categoria com ocorrência confirmada, com figuras
    numeradas e Conformidade / Risco / Diagnóstico / Recomendação / Prioridade
    sempre "pendente de validação técnica".

    ``details`` (opcional): dados do imóvel e da inspeção informados pelo
    usuário (chaves em ``laudo_abnt.DETAIL_FIELDS``: data da inspeção, tipo de
    cobertura, área, equipamento, responsável técnico, ART...). O que não for
    informado sai como "não informado"; nada é estimado. ``client`` vira o
    solicitante e ``inspector`` o responsável técnico quando ``details`` não os traz.

    O HTML é autossuficiente (CSS inline, evidências em base64 reduzidas a
    ``max_side``, pronto para imprimir em A4). O PDF só é gerado com
    ``pdf=True`` e ``reportlab`` instalado (``pip install dronecamp-ia[reports]``);
    caso contrário ``pdf`` volta None e ``pdf_reason`` explica.
    ``only_positive_images=False`` inclui as figuras das demais fotos no Anexo A.

    Descrições do VLM (``ai_notes.json`` ao lado do registro, desta versão)
    entram como subsídio pendente. Devolve ``{"directory", "html", "pdf",
    "pdf_reason", "json", "manifest", "images" (fotos com figura), "occurrences"
    (caixas confirmadas), "pending_suggestions"}``.

    Parâmetros (todos de texto já limitados pelo servidor):
        config (ProjectConfig), registry_path (Path), output_dir (Path: o
        servidor usa ``<pasta da revisão>/laudos``), decisions, suggestions:
        como em ``build_evidence_package``.
        reviewer (str): revisor que gerou o laudo (somado aos do registro).
        title (str | None): título; None = ``laudo_abnt.DEFAULT_TITLE``.
        client, site, inspector, notes (str): cabeçalho do laudo.
        details (dict | None): campos de ``laudo_abnt.DETAIL_FIELDS``.
        pdf (bool): tenta gerar o PDF.
        max_side (int): lado maior das figuras em px (inteiro ≥ 16; padrão 1400).
        only_positive_images (bool): True = só fotos positivas viram figura.

    Erros (ValueError): ``max_side`` inválido; ``details`` com chave
    desconhecida (``laudo_abnt.normalize_details``); os de ``_load_context`` e
    ``_checked_source``. Qualquer erro apaga a pasta do laudo. Falha só no
    PDF NÃO é erro: vai em ``pdf_reason``.
    """
    # 1. Validações e normalização dos dados informados pelo usuário.
    if type(max_side) is not int or max_side < 16:
        raise ValueError("max_side deve ser um inteiro ≥ 16.")
    details = laudo_abnt.normalize_details(details, client=client, inspector=inspector)
    context = _load_context(config, registry_path, decisions, suggestions)
    # 2. Pasta nova AAAAMMDDTHHMMSSZ_laudo_<versão> (com _2, _3... se já existir).
    generated = _utc_now()
    generated_at = generated.isoformat(timespec="seconds")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    directory = _unique_path(output_dir, f"{generated.strftime('%Y%m%dT%H%M%SZ')}_laudo_{_safe_name(context.version)}")
    directory.mkdir()
    # 3. Geração dos arquivos; erro (inclusive Ctrl+C) apaga a pasta inteira.
    # EXTENSAO: novo parâmetro do laudo: acrescente na assinatura, repasse aqui
    # e em _build_report_files, e leia do corpo da requisição em
    # platform_server._build_report.
    try:
        return _build_report_files(config, context, directory, generated_at, reviewer=reviewer, title=title, client=client,
                                   site=site, inspector=inspector, notes=notes, details=details, pdf=pdf, max_side=max_side,
                                   only_positive_images=only_positive_images)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)  # Nada de laudo pela metade.
        raise


def _report_items(context: _Context, occurrences: list[dict], entries: list[dict], ai_notes: dict | None,
                  included: set[int]) -> tuple[list[dict], list[dict], list[dict]]:
    """Itens da seção 6 (um por categoria com ocorrência), categorias sem ocorrência e figuras numeradas.

    Cada foto positiva vira uma figura, colocada no primeiro item (ordem da taxonomia) em que aparece; os
    itens seguintes só a citam ("figure_refs"). Fotos incluídas sem ocorrência confirmada ficam no Anexo A.

    Parâmetros:
        context (_Context): classes ativas (ordem da taxonomia), rótulos e versão.
        occurrences (list[dict]): caixas de fotos positivas (registros
            montados em ``_build_report_files``, com ``numero_foto``,
            ``class_id``, ``caixa``, ``nota``, ``image_sha256``, ``bbox_xyxy``).
        entries (list[dict]): todas as entradas de ``_resolve_entries``.
        ai_notes (dict | None): conteúdo do ``ai_notes.json`` desta versão.
        included (set[int]): posições das fotos que ganham figura.

    Retorna:
        tuple ``(items, without, figures)``:
        - items: um dict por categoria com ocorrência (``number``, ``label``,
          ``question``, ``statement``, ``figures`` novas, ``figure_refs``,
          ``notes``, ``ai_support``, ``model_reference`` e os campos técnicos
          sempre ``PENDING_TEXT``);
        - without: categorias sem ocorrência (para "não constatado");
        - figures: figuras numeradas 1, 2, ... (``numero_foto``, ``caption``,
          ``source``, ``item``: número do item ou None = Anexo A).
    """
    figures: list[dict] = []
    # figure_of: posição da foto → número da figura (evita figura duplicada).
    figure_of: dict[int, int] = {}
    by_position = {entry["position"]: entry for entry in entries}
    items, without = [], []

    def add_figure(position: int, item_number: int | None) -> int:
        """Cria a próxima figura para a foto ``position``.

        Parâmetros:
            position (int): número da foto.
            item_number (int | None): item da seção 6 onde a figura aparece;
                None = Anexo A.

        Retorna:
            int: o número da figura criada.
        """
        entry = by_position[position]
        image = entry["image"]
        # Legenda: classes presentes (só para foto positiva) ou o status da foto.
        present = sorted({box["class_id"] for box in entry["boxes"]})
        names = ", ".join(context.labels.get(class_id, f"classe {class_id}") for class_id in present) if entry["status"] == "positive" else ""
        caption = f"{names} (fotografia {position:03d})" if names else f"Fotografia {position:03d} — {STATUS_LABELS[entry['status']]}"
        # Origem neutra: o registro não guarda como a foto foi obtida (há revisões com imagens de referência).
        source = (f"registro de revisão {context.version}; arquivo {image['filename']}, {image['width']} × {image['height']} px, "
                  f"SHA-256 {image['image_sha256']}; demarcações: {SOURCE_LABELS[entry['source']]}")
        # EXTENSAO: novo dado da figura (ex.: coordenadas GPS): acrescente neste dicionário
        # e mostre em laudo_abnt.render_html/write_pdf.
        figures.append({"number": len(figures) + 1, "numero_foto": position, "caption": caption, "source": source, "item": item_number})
        figure_of[position] = len(figures)
        return len(figures)

    # 1. Um item por categoria ativa, na ordem da taxonomia.
    for category in context.classes:
        confirmed = [box for box in occurrences if box["class_id"] == category["id"]]
        # Pergunta da lista de verificação do laudo CEASA e a referência do
        # laudo-modelo (item, conformidade, recomendação). Categoria sem
        # entrada em CHECKLIST recebe uma pergunta genérica e nenhuma referência.
        # EXTENSAO: pergunta/recomendação de referência de uma classe nova:
        # acrescente o slug em laudo_abnt.CHECKLIST.
        question, model_item, conformity, recommendation = laudo_abnt.CHECKLIST.get(
            category["slug"], (f"Há ocorrências de {category['label'].lower()}?", None, None, None))
        # Sem ocorrência confirmada: vai para a lista "sem ocorrência".
        if not confirmed:
            without.append({"class_id": category["id"], "slug": category["slug"], "label": category["label"], "question": question})
            continue
        number = len(items) + 1
        # Fotos onde a categoria aparece (posições, ordenadas).
        positions = sorted({box["numero_foto"] for box in confirmed})
        # Figuras novas só para fotos ainda sem figura e incluídas no corpo.
        new_figures = [add_figure(position, number) for position in positions if position not in figure_of and position in included]
        # Todas as figuras citadas pelo item (novas ou de itens anteriores).
        refs = sorted(figure_of[position] for position in positions if position in figure_of)
        # " (Figura 3)" ou " (Figuras 3, 5)" ou "" se nenhuma.
        figure_text = (f" (Figura {refs[0]})" if len(refs) == 1 else f" (Figuras {', '.join(map(str, refs))})") if refs else ""
        # Constatação factual: contagens e critério, sem juízo técnico.
        # ATENCAO: texto só com fatos da revisão; nada de gravidade ou causa.
        statement = (f"Na revisão das fotografias foram confirmadas {len(confirmed)} demarcação(ões) de “{category['label']}” em "
                     f"{len(positions)} fotografia(s){figure_text}. Critério de identificação adotado: "
                     f"{category.get('criterion') or 'conforme taxonomia'}. Estes pontos podem vir a ser quantificados em campo; a "
                     "contagem por fotografia pode repetir o mesmo elemento em imagens sobrepostas.")
        # Notas da fotografia (acesso, observações da cena) antes das notas de cada caixa.
        photo_notes = [f"fotografia {position:03d}: {by_position[position]['notes']}" for position in positions
                       if by_position[position]["notes"]]
        # dict.fromkeys remove repetidas mantendo a ordem; no máximo 8 notas por item.
        notes = list(dict.fromkeys(photo_notes + [box["nota"] for box in confirmed if box["nota"]]))[:8]
        # Descrições do VLM para as caixas confirmadas: no máximo 6 por item,
        # 600 caracteres cada, sempre com status "pendente" (subsídio, não constatação).
        ai_rows = []
        for box in confirmed:
            note = laudo_abnt.match_ai_note(ai_notes, box["image_sha256"], {"class_id": box["class_id"], "bbox_xyxy": box["bbox_xyxy"]})
            if note and len(ai_rows) < 6:
                ai_rows.append({"figure": figure_of.get(box["numero_foto"]), "foto": box["numero_foto"], "box": box["caixa"],
                                "text": str(note["description"])[:600], "model": note.get("model_used"),
                                "class_guess": note.get("class_guess"), "status": "pendente"})
        # EXTENSAO: novo campo em cada item da seção 6: acrescente neste dicionário
        # e mostre em laudo_abnt.render_html/write_pdf.
        # ATENCAO: conformidade, risco, diagnóstico, recomendação e prioridade
        # ficam SEMPRE PENDING_TEXT; "model_reference" e "reference_severity"
        # são só a referência histórica do laudo-modelo.
        items.append({"number": number, "class_id": category["id"], "slug": category["slug"], "label": category["label"],
                      "question": question, "boxes": len(confirmed), "images": len(positions), "figures": new_figures,
                      "figure_refs": refs, "statement": statement, "notes": notes, "ai_support": ai_rows,
                      "criterion": category.get("criterion", ""), "reference_severity": category.get("reference_severity"),
                      "model_reference": {"item": model_item, "conformidade": conformity, "recomendacao": recommendation} if model_item else None,
                      "conformidade": PENDING_TEXT, "risco": PENDING_TEXT, "diagnostico": PENDING_TEXT,
                      "recomendacao": PENDING_TEXT, "prioridade": PENDING_TEXT})
    # 2. Fotos incluídas que não viraram figura em nenhum item → Anexo A.
    for position in sorted(included - set(figure_of)):  # Fotos sem ocorrência confirmada pedidas no corpo → Anexo A.
        add_figure(position, None)
    return items, without, figures


def _build_report_files(config: ProjectConfig, context: _Context, directory: Path, generated_at: str, *, reviewer: str,
                        title: str | None, client: str, site: str, inspector: str, notes: str, details: dict, pdf: bool,
                        max_side: int, only_positive_images: bool) -> dict:
    """Monta o ``laudo.json`` e grava HTML, JSON, PDF e manifesto na pasta já criada.

    Parâmetros:
        config (ProjectConfig): para conferir as fotos originais.
        context (_Context): contexto validado.
        directory (Path): pasta nova e vazia do laudo.
        generated_at (str): data ISO UTC da geração.
        reviewer, title, client, site, inspector, notes, details, pdf,
        max_side, only_positive_images: como em ``build_report`` (``details``
        já normalizado).

    Retorna:
        dict: o mesmo de ``build_report``.

    Erros: os de ``_checked_source``; falha no PDF NÃO levanta (vai em ``pdf_reason``).

    Chamada por: ``build_report`` (que apaga a pasta se algo falhar aqui).
    """
    entries, labels = context.entries, context.labels
    # Revisores: o informado agora primeiro, depois os do registro; vazios
    # removidos; dict.fromkeys tira repetidos mantendo a ordem.
    reviewers = [value for value in [reviewer.strip()] + context.reviewers if value]
    reviewers = list(dict.fromkeys(reviewers))
    # Grupos de edificação presentes nas fotos (sem vazio), em ordem alfabética.
    buildings = sorted({str(entry["image"].get("building_group") or "") for entry in entries} - {""})
    # 1. Dados estruturados (laudo.json) — a mesma fonte alimenta HTML e PDF.
    # included: posições das fotos que ganham figura (só positivas, ou todas
    # com only_positive_images=False).
    included = {entry["position"] for entry in entries if entry["status"] == "positive" or not only_positive_images}
    # images: uma linha por foto; occurrences: caixas de fotos positivas;
    # drafts: caixas de fotos não positivas (ambíguas, rascunhos...), que não
    # entram nas constatações.
    images, occurrences, drafts = [], [], []
    for entry in entries:
        item = entry["image"]
        # EXTENSAO: novo campo por foto no laudo.json: acrescente neste dicionário.
        images.append({"numero_foto": entry["position"], "image_sha256": item["image_sha256"], "filename": item["filename"],
                       "width": item["width"], "height": item["height"], "pages": list(item.get("pages", [])),
                       "building_group": item.get("building_group"), "scene_group": item.get("scene_group"),
                       "status": entry["status"], "status_label": STATUS_LABELS[entry["status"]], "source": entry["source"],
                       "source_label": SOURCE_LABELS[entry["source"]], "confirmed_complete": entry["confirmed_complete"],
                       "boxes": len(entry["boxes"]), "pending_suggestions": entry["pending_suggestions"], "notes": entry["notes"],
                       "included_in_report": entry["position"] in included})
        # Um registro por caixa; "caixa" = número da caixa na foto (placa "n:").
        for number, box in enumerate(entry["boxes"], start=1):
            # EXTENSAO: novo campo por ocorrência no laudo.json: acrescente aqui.
            # ATENCAO: gravidade/diagnóstico/recomendação ficam PENDING_TEXT.
            record = {"numero_foto": entry["position"], "image_sha256": item["image_sha256"], "filename": item["filename"],
                      "caixa": number, "class_id": box["class_id"], "classe": labels.get(box["class_id"]),
                      "bbox_xyxy": [_number(value) for value in box["bbox_xyxy"]], "nota": box.get("note", "") or "",
                      "status_foto": entry["status"], "origem": entry["source"], "gravidade": PENDING_TEXT,
                      "diagnostico": PENDING_TEXT, "recomendacao": PENDING_TEXT}
            # Positiva → ocorrência confirmada; demais → rascunho.
            (occurrences if entry["status"] == "positive" else drafts).append(record)
    counts = _counts(entries, labels)
    # Resumo por classe (todas as classes ativas, inclusive com zero).
    by_class = []
    for category in context.classes:
        confirmed = [box for box in occurrences if box["class_id"] == category["id"]]
        by_class.append({"class_id": category["id"], "slug": category["slug"], "label": category["label"], "boxes": len(confirmed),
                         "images": len({box["numero_foto"] for box in confirmed}), "criterion": category.get("criterion", ""),
                         "reference_severity": category.get("reference_severity")})
    # Sugestões pendentes por classe (Anexo B), só classes com alguma sugestão.
    pending_by_class = []
    for category in context.classes:
        # Pares (posição da foto, sugestão) desta classe em todas as fotos.
        matching = [(entry["position"], suggestion) for entry in entries for suggestion in entry["pending"] if suggestion["class_id"] == category["id"]]
        if matching:
            pending_by_class.append({"class_id": category["id"], "label": category["label"], "count": len(matching),
                                     "images": len({position for position, _ in matching})})
    # Descrições do VLM (ai_notes.json desta versão do registro) e aviso se ignoradas.
    ai_notes, ai_note = laudo_abnt.load_ai_notes(context.registry_path, context.registry_sha256)
    items, without, figures = _report_items(context, occurrences, entries, ai_notes, included)
    # Cada ocorrência/rascunho ganha o número da figura da sua foto (ou None).
    figure_by_photo = {figure["numero_foto"]: figure["number"] for figure in figures}
    for record in occurrences + drafts:
        record["figura"] = figure_by_photo.get(record["numero_foto"])
    positive_photos = counts["by_status"]["positive"]
    # Revisada = decisão humana nesta exportação (não rascunho) ou aprovação humana gravada no registro.
    # (soma de booleanos: True conta 1.)
    reviewed = sum((entry["source"] == "revisao_humana" and entry["status"] != "pending")
                   or (entry["source"] == "registro" and entry["image"].get("human_approved") is True) for entry in entries)
    # Parágrafo de fatos da conclusão: só contagens, nenhuma conclusão técnica.
    # A frase "e N ainda não foram revisadas" só aparece se houver fotos não revisadas;
    # a lista de categorias entre parênteses só se houver itens.
    # EXTENSAO: novo status: acrescente a contagem neste texto.
    facts = (f"A inspeção {review_name(context.version, context.registry.get('display_name'))} (registro {context.version}) reúne {counts['images']} fotografias, das quais {reviewed} têm decisão humana "
             f"registrada" + (f" e {counts['images'] - reviewed} ainda não foram revisadas" if reviewed < counts["images"] else "")
             + f". Situação das fotografias: {positive_photos} com ocorrências "
             f"confirmadas, {counts['by_status']['negative']} sem ocorrências, {counts['by_status']['ambiguous']} ambíguas, "
             f"{counts['by_status']['excluded']} excluídas e {counts['by_status']['pending']} pendentes. Foram confirmadas "
             f"{counts['confirmed_boxes']} demarcações em {len(items)} categoria(s) da lista de verificação"
             + (f" ({', '.join(item['label'] for item in items)})" if items else "")
             + f". Permanecem {counts['pending_suggestions']} sugestões automáticas sem decisão humana (Anexo B), que não integram as constatações.")
    # laudo.json (schema 2). Seções:
    # header (identificação), details (dados informados), normative_references,
    # summary, items (seção 6), items_without_occurrence, manual_checklist
    # (perguntas que só a vistoria em campo responde), figures, images,
    # occurrences, draft_boxes, pending_suggestions (Anexo B), ai_support
    # (Anexo C), conclusion, technical_responsibility, provenance, methodology/limitations.
    # EXTENSAO: nova seção ou campo do laudo: acrescente neste dicionário e
    # mostre em laudo_abnt.render_html e laudo_abnt.write_pdf (os dois leem daqui).
    # ATENCAO: ao mudar o formato de forma incompatível, suba "schema_version".
    report = {
        "schema_version": 2, "document": "laudo_tecnico_inspecao_cobertura_drone", "status": PRELIMINARY_TEXT,
        "header": {"title": title or laudo_abnt.DEFAULT_TITLE, "client": client, "site": site,
                   "buildings": buildings, "issued_at": generated_at, "reviewers": reviewers, "inspector": inspector, "notes": notes,
                   "registry_version": context.version, "registry_display_name": review_name(context.version, context.registry.get("display_name")),
                   "registry_sha256": context.registry_sha256,
                   "taxonomy_version": context.registry.get("taxonomy_version"), "taxonomy_sha256": context.taxonomy_sha256,
                   "suggestion_model": context.suggestion_model},
        "details": details,
        "normative_references": laudo_abnt.NORMATIVE_REFERENCES,
        "summary": {"images": counts["images"], **counts["by_status"], "decided_by_human": counts["decided_by_human"],
                    "confirmed_boxes": counts["confirmed_boxes"], "draft_boxes": len(drafts), "by_class": by_class,
                    "pending_suggestions": counts["pending_suggestions"]},
        "items": items, "items_without_occurrence": without, "manual_checklist": laudo_abnt.MANUAL_CHECKLIST,
        "figures": figures, "images": images, "occurrences": occurrences, "draft_boxes": drafts,
        "pending_suggestions": {"total": counts["pending_suggestions"], "by_class": pending_by_class,
                                "note": "Sugestões automáticas pendentes não constituem ocorrências; aguardam decisão humana na plataforma de revisão."},
        "ai_support": laudo_abnt.ai_summary(ai_notes, ai_note, {category["slug"]: category["label"] for category in context.classes}),
        # ATENCAO: a conclusão técnica fica sempre pendente; só "facts" é automático.
        "conclusion": {"facts": facts, "reviewed_images": reviewed, "technical_conclusion": PENDING_TEXT},
        "technical_responsibility": {"name": details["responsible_name"], "title": details["responsible_title"],
                                     "registry": details["responsible_registry"], "art": details["art_number"],
                                     "statement": "Este laudo foi elaborado com base em inspeção visual aérea, não substituindo "
                                                  "inspeções presenciais in loco. Sua validade depende da revisão, complementação e "
                                                  "assinatura do responsável técnico habilitado, com a respectiva ART."},
        "provenance": {"generated_at": generated_at, "generator": "dronecamp_ia.exports.build_report",
                       "registry_file": context.registry_path.name, "review_directory": context.registry_path.parent.name,
                       "registry_sha256": context.registry_sha256, "taxonomy_sha256": context.taxonomy_sha256,
                       "decisions_provided": context.decisions_count, "suggestion_model": context.suggestion_model,
                       "suggestion_note": context.suggestion_note, "ai_notes_note": ai_note,
                       "parameters": {"max_side": max_side, "only_positive_images": only_positive_images, "pdf_requested": pdf}},
        "methodology": METHODOLOGY, "limitations": METHODOLOGY,
    }
    # 2. Evidências reduzidas (JPEG qualidade 82), uma foto por vez, só das fotos com figura.
    # evidences: posição da foto → bytes JPEG (embutidos em base64 no HTML e no PDF).
    # As sugestões do modelo nunca são desenhadas no laudo (include_suggestions=False).
    evidences: dict[int, bytes] = {}
    for entry in entries:
        if entry["position"] not in included:
            continue
        # Confere hash e dimensões antes de desenhar.
        source = _checked_source(config, entry["image"])
        evidence = _render_evidence(source, entry, context, max_side, include_suggestions=False)
        buffer = io.BytesIO()
        # EXTENSAO: qualidade das figuras do laudo: mude quality=82 (afeta o tamanho do HTML/PDF).
        evidence.save(buffer, "JPEG", quality=82)
        evidence.close()
        evidences[entry["position"]] = buffer.getvalue()
    # 3. Arquivos.
    html_path, json_path, pdf_path, manifest_path = (directory / name for name in ("laudo.html", "laudo.json", "laudo.pdf", "manifesto.json"))
    # 3a. HTML autossuficiente (laudo_abnt.render_html) e JSON estruturado.
    html_path.write_text(laudo_abnt.render_html(report, evidences), encoding="utf-8")
    write_json(json_path, report)
    # 3b. PDF opcional: não pedido → motivo; reportlab ausente → motivo;
    # falha na geração → apaga o PDF parcial e registra o motivo (HTML/JSON continuam).
    pdf_reason = None
    if not pdf:
        pdf_reason = "PDF não solicitado (pdf=False)."
    else:
        try:
            import reportlab  # noqa: F401 - só confere a instalação.
        except ImportError:
            pdf_reason = "reportlab não instalado: pip install dronecamp-ia[reports] (ou pip install reportlab)."
        else:
            try:
                laudo_abnt.write_pdf(pdf_path, report, evidences)
            except Exception as error:  # HTML e JSON continuam válidos; o motivo vai na resposta.
                pdf_path.unlink(missing_ok=True)
                pdf_reason = f"falha ao gerar o PDF: {type(error).__name__}: {error}"
    # 3c. Manifesto: hash e tamanho dos arquivos que existem (o PDF pode faltar).
    # EXTENSAO: arquivo novo no laudo: grave-o acima e acrescente o caminho nesta tupla.
    files = {}
    for path in (html_path, json_path, pdf_path):
        if path.is_file():
            files[path.name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    write_json(manifest_path, {"schema_version": 1, "kind": "laudo_preliminar", "generated_at": generated_at,
                               "generator": "dronecamp_ia.exports.build_report", "version": context.version,
                               "registry_file": context.registry_path.name, "registry_sha256": context.registry_sha256,
                               "taxonomy_sha256": context.taxonomy_sha256, "status": PRELIMINARY_TEXT,
                               "parameters": report["provenance"]["parameters"], "counts": counts, "pdf_reason": pdf_reason,
                               "files": files})
    # 4. Resumo para o servidor (que converte em links relativos para a página).
    return {"directory": str(directory), "html": str(html_path), "pdf": str(pdf_path) if pdf_path.is_file() else None,
            "pdf_reason": pdf_reason, "json": str(json_path), "manifest": str(manifest_path), "images": len(evidences),
            "occurrences": len(occurrences), "pending_suggestions": counts["pending_suggestions"]}
