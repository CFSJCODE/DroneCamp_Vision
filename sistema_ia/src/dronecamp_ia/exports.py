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
  instalado: extra ``reports``) e ``manifesto.json``.
- ``occurrences_csv``: o texto do ``ocorrencias.csv`` sem gerar o ZIP.

Toda saída é nova, dentro de ``output_dir``; registros, fotos e revisões não
são tocados. Fotos são conferidas pelo SHA-256 antes de qualquer desenho.
Gravidade, diagnóstico e recomendação ficam explicitamente em branco
("pendente de validação técnica") para o responsável técnico preencher.

Quando mexer: colunas dos CSV, texto fixo do laudo (metodologia, avisos),
aparência das evidências ou campos do ``laudo.json``.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import base64
import csv
import html
import io
import json
import math
import re
import shutil
import zipfile

from PIL import Image, ImageDraw, ImageFont

from .config import ProjectConfig, detection_names, load_taxonomy
from .io import file_hash, resolve_local_path, write_json
from .review_data import IMAGE_STATUSES, REVIEW_SCHEMA, review_image_size, validate_boxes

# Decisão exportada pela página pode estar "pending" (rascunho sem status escolhido).
DECISION_STATUSES = IMAGE_STATUSES | {"pending"}
# Texto fixo dos campos que só o responsável técnico pode preencher.
PENDING_TEXT = "pendente de validação técnica"
PRELIMINARY_TEXT = "preliminar, sujeito a validação por responsável técnico"
# Cor do texto nas placas das caixas (contraste ≥ 4,5:1 com todas as cores de classe).
PLATE_TEXT = "#08111F"
# Colunas do ocorrencias.csv (uma linha por caixa) e do fotos.csv (uma linha por foto).
OCCURRENCE_COLUMNS = ["numero_foto", "image_sha256", "filename", "status", "origem", "caixa", "classe_id", "classe",
                      "x1", "y1", "x2", "y2", "nota"]
PHOTO_COLUMNS = ["numero_foto", "image_sha256", "filename", "largura", "altura", "paginas", "grupo_edificacao", "cena",
                 "status", "origem", "caixas", "sugestoes_pendentes", "notas", "evidencia"]
STATUS_LABELS = {"positive": "positiva", "negative": "negativa", "ambiguous": "ambígua", "excluded": "excluída",
                 "pending": "pendente (rascunho sem decisão)"}
SOURCE_LABELS = {"revisao_humana": "revisão humana", "registro": "registro (sem decisão humana nesta exportação)"}
ONLY_CHOICES = ("all", "decided", "positive")

# Texto fixo de metodologia e limitações (laudo e LEIA-ME). Nada aqui é inferido dos dados.
METHODOLOGY = [
    "As fotografias foram congeladas por hash SHA-256 no registro de revisão; cada evidência deste documento foi "
    "gerada a partir do arquivo original conferido contra esse hash.",
    "As caixas resultam da revisão humana na plataforma local (origem \"revisão humana\") ou do registro de propostas "
    "(origem \"registro\"). Caixas de origem \"registro\" não contam com decisão humana nesta exportação.",
    "As sugestões automáticas vêm de um modelo piloto não validado. Não constituem ocorrências: aparecem somente no "
    "Anexo B e aguardam decisão humana.",
    "Não houve medição física, avaliação hidráulica, estrutural ou de estanqueidade. Nenhuma gravidade, causa, área, "
    "extensão em metros ou diagnóstico foi inferida automaticamente; esses campos ficam em branco para o responsável "
    "técnico.",
    "Caixas em fotografias semelhantes (mesma cena, recortes ou ângulos próximos) podem registrar o mesmo defeito mais "
    "de uma vez. As contagens referem-se a caixas por foto, não a defeitos físicos únicos.",
    "Este documento é preliminar e não substitui laudo técnico assinado por profissional habilitado.",
]


# ---------------------------------------------------------------------------
# Desenho das evidências (mesmo estilo de review_render e da página).
# ---------------------------------------------------------------------------

# Fontes com acentos do pt-BR: Windows (como em review_render), depois Linux/macOS; por fim a padrão do Pillow.
FONT_CANDIDATES = (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/segoeui.ttf"),
                   Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                   Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
                   Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"), Path("/Library/Fonts/Arial.ttf"),
                   Path("/System/Library/Fonts/Supplemental/Arial.ttf"))


def _font(size: int):
    """Primeira fonte TrueType existente; senão a padrão do Pillow (sem acentos nas versões bitmap)."""
    for path in FONT_CANDIDATES:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow antigo: fonte bitmap sem tamanho.
        return ImageFont.load_default()


def _palette() -> list[str]:
    """Cores por classe de ``review_render`` (import tardio: aquele módulo passará a importar este)."""
    from .review_render import COLORS
    return COLORS


def _label(labels: dict[int, str], class_id: int) -> str:
    return labels.get(class_id, f"classe {class_id}")


def _dashed_rectangle(draw: ImageDraw.ImageDraw, box: list[float], color: str, width: int, dash: tuple[int, int]) -> None:
    """Tracejado simulado com segmentos (o Pillow não tem linha tracejada nativa)."""
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    on, off = dash
    for (ax, ay), (bx, by) in zip(corners, corners[1:]):
        length = math.hypot(bx - ax, by - ay)
        if length == 0:
            continue
        ux, uy = (bx - ax) / length, (by - ay) / length
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
    """
    if not (0 < scale <= 16):
        raise ValueError("scale deve ser um fator positivo.")
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    thickness = max(2, round(3 * scale))
    font = _font(max(14, int(canvas.width / 75)))
    for number, box in enumerate(boxes, start=1):
        coordinates = [value * scale for value in box["bbox_xyxy"]]
        color = colors[box["class_id"] % len(colors)]
        draw.rectangle(coordinates, outline=color, width=thickness)
        text = f"{number}: {_label(labels, box['class_id'])}"
        x, y = coordinates[:2]
        anchor = (x, max(0, y - 22))
        draw.rectangle(draw.textbbox(anchor, text, font=font), fill=color)
        draw.text(anchor, text, fill=PLATE_TEXT, font=font)
    if suggestions:
        small = _font(max(13, int(canvas.width / 80)))
        dash = (max(4, round(10 * scale)), max(3, round(6 * scale)))
        for number, suggestion in enumerate(suggestions, start=1):
            coordinates = [value * scale for value in suggestion["bbox_xyxy"]]
            color = colors[suggestion["class_id"] % len(colors)]
            _dashed_rectangle(draw, coordinates, color, thickness, dash)
            confidence = round(float(suggestion.get("confidence", 0)) * 100)
            text = f"IA {number}: {_label(labels, suggestion['class_id'])} {confidence}%"
            anchor = (coordinates[0], min(canvas.height - 22, coordinates[3]))
            draw.rectangle(draw.textbbox(anchor, text, font=small), fill=PLATE_TEXT)
            draw.text(anchor, text, fill=color, font=small)
    return canvas


# ---------------------------------------------------------------------------
# Decisões: página > registro; sugestões pendentes.
# ---------------------------------------------------------------------------

def _js_round(value: float) -> int:
    """Arredondamento da página (Math.round: meio para cima), não o do Python (meio par)."""
    return math.floor(value + 0.5)


def _same_box(box: dict, suggestion: dict) -> bool:
    return box["class_id"] == suggestion["class_id"] and all(
        _js_round(a) == _js_round(b) for a, b in zip(box["bbox_xyxy"], suggestion["bbox_xyxy"]))


def _pending_suggestions(candidates: list[dict], boxes: list[dict], dismissed: list) -> list[dict]:
    """Sugestão pendente = não aceita como caixa (mesma classe e geometria) nem descartada."""
    return [suggestion for suggestion in candidates
            if not any(_same_box(box, suggestion) for box in boxes) and suggestion.get("id") not in dismissed]


def _normalize_suggestions(suggestions: dict | None, class_count: int) -> dict[str, list[dict]]:
    """Aceita o mapa sha → lista ou o conteúdo inteiro de suggestions.json; rejeita sugestões inválidas."""
    if not suggestions:
        return {}
    if not isinstance(suggestions, dict):
        raise ValueError("suggestions deve ser um dicionário image_sha256 → lista de sugestões.")
    mapping = suggestions.get("images") if isinstance(suggestions.get("images"), dict) else suggestions
    normalized = {}
    for digest, values in mapping.items():
        if not isinstance(values, list):
            raise ValueError(f"Sugestões de {digest} devem ser uma lista.")
        for value in values:
            if not isinstance(value, dict) or type(value.get("class_id")) is not int or not 0 <= value["class_id"] < class_count:
                raise ValueError("Sugestão com classe inválida.")
            coordinates = value.get("bbox_xyxy")
            if (not isinstance(coordinates, list) or len(coordinates) != 4
                    or any(type(item) not in (int, float) or not math.isfinite(item) for item in coordinates)):
                raise ValueError("Sugestão com coordenadas inválidas.")
            try:
                confidence = float(value.get("confidence", -1))
            except (TypeError, ValueError):
                raise ValueError("Sugestão com confiança inválida.") from None
            if not 0 <= confidence <= 1:
                raise ValueError("Sugestão com confiança inválida.")
        normalized[digest] = list(values)
    return normalized


def _resolve_entries(registry: dict, decisions: dict | None, class_count: int, suggestions: dict | None) -> list[dict]:
    """Versão interna de ``resolve_decisions`` com a lista das sugestões pendentes e a posição."""
    images = registry.get("images") if isinstance(registry, dict) else None
    if not isinstance(images, list) or any(not isinstance(item, dict) for item in images):
        raise ValueError("O registro deve conter uma lista de imagens válida.")
    known = [item.get("image_sha256") for item in images]
    if len(set(known)) != len(known):
        raise ValueError("Imagem duplicada no registro de revisão.")
    decisions = decisions or {}
    if not isinstance(decisions, dict):
        raise ValueError("decisions deve ser um dicionário image_sha256 → decisão.")
    unknown = set(decisions) - set(known)
    if unknown:
        raise ValueError(f"Decisão para imagem fora do registro: {sorted(unknown)[0]}.")
    candidates = _normalize_suggestions(suggestions, class_count)
    entries = []
    for position, item in enumerate(images, start=1):
        digest = item["image_sha256"]
        decision = decisions.get(digest)
        if decision is not None:
            if not isinstance(decision, dict):
                raise ValueError(f"Decisão de {item.get('filename')} deve ser um objeto.")
            status = decision.get("status")
            if status not in DECISION_STATUSES:
                raise ValueError(f"Status inválido na decisão de {item.get('filename')}: {status!r}.")
            boxes = decision.get("boxes", [])
            validate_boxes(boxes, item["width"], item["height"], class_count)
            if status == "positive" and not boxes:
                raise ValueError(f"Decisão positiva sem caixas: {item.get('filename')}.")
            if status == "negative" and boxes:
                raise ValueError(f"Decisão negativa com caixas: {item.get('filename')}.")
            notes = decision.get("notes") or ""
            dismissed = decision.get("dismissed_suggestions") or []
            if not isinstance(notes, str) or not isinstance(dismissed, list):
                raise ValueError(f"Notas ou sugestões descartadas inválidas em {item.get('filename')}.")
            source, confirmed = "revisao_humana", decision.get("confirmed_complete") is True
        else:
            status = item.get("status")
            if status not in IMAGE_STATUSES:
                raise ValueError(f"Status inválido no registro: {item.get('filename')}.")
            boxes = item.get("boxes", [])
            validate_boxes(boxes, item["width"], item["height"], class_count)
            notes = item.get("human_notes") or item.get("notes") or ""
            dismissed, source, confirmed = [], "registro", item.get("human_approved") is True
        pending = _pending_suggestions(candidates.get(digest, []), boxes, dismissed)
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
    """
    keys = ("image", "status", "boxes", "notes", "source", "pending_suggestions", "confirmed_complete")
    return [{key: entry[key] for key in keys} for entry in _resolve_entries(registry, decisions, class_count, suggestions)]


# ---------------------------------------------------------------------------
# Contexto comum: registro, taxonomia, sugestões e fotos conferidas.
# ---------------------------------------------------------------------------

@dataclass
class _Context:
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
    """suggestions.json ao lado do registro, só se pertencer a esta versão exata."""
    path = registry_path.parent / "suggestions.json"
    if not path.is_file():
        return None, None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("registry_sha256") != registry_sha256:
        return None, "suggestions.json ao lado do registro pertence a outra versão e foi ignorado."
    return data, None


def _model_meta(data: dict) -> dict:
    keys = ("model_label", "conf", "weights_sha256", "pilot", "calibrator", "created_at", "warning")
    return {key: data.get(key) for key in keys if key in data}


def _load_context(config: ProjectConfig, registry_path: Path, decisions: dict | None, suggestions: dict | None) -> _Context:
    registry_path = Path(registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or registry.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError("Schema inválido no registro de revisão.")
    taxonomy_sha256 = file_hash(config.taxonomy_path)
    if registry.get("taxonomy_sha256") != taxonomy_sha256:
        raise ValueError("Taxonomia mudou: migre a revisão antes de exportar.")
    taxonomy = load_taxonomy(config.taxonomy_path)
    class_count = len(detection_names(taxonomy))
    classes = [item for item in taxonomy["classes"] if item["phase"] == 1]
    registry_sha256 = file_hash(registry_path)
    # Sugestões: o que veio por parâmetro; senão, suggestions.json desta versão do registro.
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
    entries = _resolve_entries(registry, decisions, class_count, suggestions)
    reviewers = []
    for value in [registry.get("human_feedback", {}).get("reviewer_id")] + [item.get("human_reviewer") for item in registry["images"]]:
        if isinstance(value, str) and value.strip() and value.strip() not in reviewers:
            reviewers.append(value.strip())
    return _Context(registry=registry, registry_path=registry_path, registry_sha256=registry_sha256,
                    version=str(registry.get("version") or registry_path.parent.name), taxonomy_sha256=taxonomy_sha256,
                    classes=classes, labels={item["id"]: item["label"] for item in classes}, colors=_palette(),
                    entries=entries, decisions_count=len(decisions or {}), suggestion_model=model,
                    suggestion_note=suggestion_note, reviewers=reviewers)


def _checked_source(config: ProjectConfig, item: dict) -> Path:
    """Foto original intacta (hash e dimensões) ou ValueError; nunca desenha sobre foto alterada."""
    source = resolve_local_path(item["source_path"], config.root)
    if not source.is_file():
        raise ValueError(f"Foto original não encontrada: {item['filename']}.")
    if file_hash(source) != item["image_sha256"]:
        raise ValueError(f"A foto original mudou: {item['filename']}.")
    if review_image_size(source) != (item["width"], item["height"]):
        raise ValueError(f"Dimensões do registro divergem da foto: {item['filename']}.")
    return source


def _render_evidence(source: Path, entry: dict, context: _Context, max_side: int | None, include_suggestions: bool) -> Image.Image:
    """Foto (reduzida a ``max_side`` quando pedido) com as caixas desenhadas."""
    with Image.open(source) as original:
        image = original.convert("RGB")
    scale = 1.0
    if max_side and max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    return draw_boxes(image, entry["boxes"], context.labels, context.colors,
                      suggestions=entry["pending"] if include_suggestions else None, scale=scale)


def _filter_entries(entries: list[dict], only: str) -> list[dict]:
    if only not in ONLY_CHOICES:
        raise ValueError(f"only deve ser um de {ONLY_CHOICES}.")
    if only == "positive":
        return [entry for entry in entries if entry["status"] == "positive"]
    if only == "decided":
        return [entry for entry in entries if entry["source"] == "revisao_humana" and entry["status"] != "pending"]
    return list(entries)


# ---------------------------------------------------------------------------
# Utilidades de formatação e arquivos.
# ---------------------------------------------------------------------------

def _number(value: float):
    """Coordenadas inteiras sem ".0"; fracionárias com duas casas."""
    return int(value) if float(value).is_integer() else round(float(value), 2)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")[:80] or "revisao"


def _unique_path(directory: Path, stem: str, suffix: str = "") -> Path:
    """Nunca sobrescreva uma exportação anterior gerada no mesmo segundo."""
    candidate = directory / f"{stem}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = directory / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def _csv_text(header: list[str], rows: list[list]) -> str:
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()


def _evidence_name(entry: dict) -> str:
    filename = Path(entry["image"]["filename"])
    return f"{entry['position']:03d}_{_safe_name(filename.stem)}.jpg"


def _occurrence_rows(entries: list[dict], labels: dict[int, str]) -> list[list]:
    rows = []
    for entry in entries:
        item = entry["image"]
        for number, box in enumerate(entry["boxes"], start=1):
            x1, y1, x2, y2 = (_number(value) for value in box["bbox_xyxy"])
            rows.append([entry["position"], item["image_sha256"], item["filename"], entry["status"], entry["source"], number,
                         box["class_id"], labels.get(box["class_id"], ""), x1, y1, x2, y2, box.get("note", "") or ""])
    return rows


def _photo_rows(entries: list[dict]) -> list[list]:
    rows = []
    for entry in entries:
        item = entry["image"]
        rows.append([entry["position"], item["image_sha256"], item["filename"], item["width"], item["height"],
                     ";".join(map(str, item.get("pages", []))), item.get("building_group", ""), item.get("scene_group", ""),
                     entry["status"], entry["source"], len(entry["boxes"]), entry["pending_suggestions"], entry["notes"],
                     f"evidencias/{_evidence_name(entry)}"])
    return rows


def _counts(entries: list[dict], labels: dict[int, str]) -> dict:
    status = Counter(entry["status"] for entry in entries)
    confirmed = Counter(box["class_id"] for entry in entries if entry["status"] == "positive" for box in entry["boxes"])
    return {"images": len(entries), "by_status": {key: status.get(key, 0) for key in ("positive", "negative", "ambiguous", "excluded", "pending")},
            "boxes": sum(len(entry["boxes"]) for entry in entries), "confirmed_boxes": sum(confirmed.values()),
            "confirmed_boxes_by_class": {labels[key]: confirmed[key] for key in sorted(confirmed)},
            "pending_suggestions": sum(entry["pending_suggestions"] for entry in entries),
            "decided_by_human": sum(entry["source"] == "revisao_humana" for entry in entries)}


def _readme(context: _Context, generated_at: str, parameters: dict) -> str:
    return "\n".join([
        "PACOTE DE EVIDÊNCIAS — DroneCamp Vision", "",
        f"Revisão: {context.version}", f"Registro: {context.registry_path.name} (SHA-256 {context.registry_sha256})",
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
    """Encaminha a escrita ao zip calculando SHA-256 e tamanho, sem guardar a imagem em memória."""

    def __init__(self, stream):
        self.stream, self.digest, self.size = stream, sha256(), 0

    def write(self, data) -> int:
        self.stream.write(data)
        self.digest.update(data)
        self.size += len(data)
        return len(data)

    def flush(self) -> None:
        pass


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
    """
    if max_side is not None and (type(max_side) is not int or max_side < 16):
        raise ValueError("max_side deve ser None ou um inteiro ≥ 16.")
    context = _load_context(config, registry_path, decisions, suggestions)
    selected = _filter_entries(context.entries, only)
    generated = _utc_now()
    generated_at = generated.isoformat(timespec="seconds")
    parameters = {"only": only, "max_side": max_side, "include_suggestions": include_suggestions,
                  "decisions": context.decisions_count}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _unique_path(output_dir, f"{generated.strftime('%Y%m%dT%H%M%SZ')}_evidencias_{_safe_name(context.version)}", ".zip")
    files: OrderedDict[str, dict] = OrderedDict()
    stamp = generated.timetuple()[:6]

    def small(archive: zipfile.ZipFile, name: str, text: str, bom: bool = False) -> None:
        data = (("\ufeff" if bom else "") + text).encode("utf-8")
        info = zipfile.ZipInfo(name, date_time=stamp)
        info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
        archive.writestr(info, data)
        files[name] = {"sha256": sha256(data).hexdigest(), "bytes": len(data)}

    try:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for entry in selected:
                source = _checked_source(config, entry["image"])
                evidence = _render_evidence(source, entry, context, max_side, include_suggestions)
                name = f"evidencias/{_evidence_name(entry)}"
                info = zipfile.ZipInfo(name, date_time=stamp)
                info.compress_type, info.external_attr = zipfile.ZIP_STORED, 0o644 << 16  # JPEG já é comprimido.
                with archive.open(info, "w") as stream:
                    writer = _HashingWriter(stream)
                    evidence.save(writer, "JPEG", quality=92)
                files[name] = {"sha256": writer.digest.hexdigest(), "bytes": writer.size,
                               "width": evidence.width, "height": evidence.height}
                evidence.close()
            occurrence_rows = _occurrence_rows(selected, context.labels)
            small(archive, "ocorrencias.csv", _csv_text(OCCURRENCE_COLUMNS, occurrence_rows), bom=True)
            small(archive, "fotos.csv", _csv_text(PHOTO_COLUMNS, _photo_rows(selected)), bom=True)
            small(archive, "LEIA-ME.txt", _readme(context, generated_at, parameters))
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
    return {"zip": str(path), "images": len(selected), "boxes": len(occurrence_rows), "bytes": path.stat().st_size,
            "sha256": file_hash(path)}


def occurrences_csv(config: ProjectConfig, registry_path: Path, decisions: dict | None = None) -> str:
    """Texto do ``ocorrencias.csv`` (todas as fotos, uma linha por caixa), sem abrir as fotos.

    Codifique em UTF-8 com BOM (``utf-8-sig``) ao gravar ou enviar, como no zip.
    """
    context = _load_context(config, registry_path, decisions, None)
    return _csv_text(OCCURRENCE_COLUMNS, _occurrence_rows(context.entries, context.labels))


# ---------------------------------------------------------------------------
# Laudo técnico preliminar (HTML / JSON / PDF).
# ---------------------------------------------------------------------------

def _esc(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _box_rows_html(boxes: list[dict], labels: dict[int, str]) -> str:
    rows = []
    for number, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = (_number(value) for value in box["bbox_xyxy"])
        rows.append(f"<tr><td>{number}</td><td>{_esc(labels.get(box['class_id']))}</td>"
                    f"<td>{x1}, {y1} – {x2}, {y2}</td><td>{_esc(box.get('note', ''))}</td></tr>")
    return "".join(rows)


def _report_html(report: dict, evidences: dict[int, bytes], labels: dict[int, str]) -> str:
    header, summary = report["header"], report["summary"]
    style = """
:root { color-scheme: light; }
@page { size: A4; margin: 16mm 14mm; }
body { margin: 0; background: #fff; color: #111827; font: 11pt/1.45 "Segoe UI", Arial, Helvetica, sans-serif; }
main { max-width: 190mm; margin: 0 auto; padding: 10mm 8mm; }
h1 { font-size: 20pt; margin: 0 0 4px; }
h2 { font-size: 14pt; margin: 28px 0 10px; border-bottom: 2px solid #CBD5E1; padding-bottom: 4px; break-before: page; page-break-before: always; }
h2.primeira { break-before: auto; page-break-before: auto; margin-top: 18px; }
h3 { font-size: 11.5pt; margin: 0 0 6px; }
.banner { border: 2px solid #B45309; background: #FFF7ED; color: #7C2D12; padding: 8px 12px; font-weight: 600; margin: 10px 0 16px; }
table { border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 6px 0 12px; }
th, td { border: 1px solid #CBD5E1; padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #F1F5F9; font-weight: 600; }
td.pendente, span.pendente { color: #92400E; font-style: italic; }
.foto { border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 12px; margin: 0 0 16px; break-inside: avoid; page-break-inside: avoid; }
figure { margin: 6px 0 8px; }
img.evidencia { display: block; max-width: 100%; height: auto; border: 1px solid #CBD5E1; }
figcaption { font-size: 9pt; color: #475569; margin-top: 4px; }
.mono { font-family: Consolas, "Courier New", monospace; font-size: 8.5pt; word-break: break-all; }
ol.metodologia li { margin-bottom: 6px; }
footer { margin-top: 24px; font-size: 8.5pt; color: #64748B; border-top: 1px solid #E2E8F0; padding-top: 6px; }
"""
    model = header.get("suggestion_model") or {}
    if model:
        model_text = f"{model.get('model_label') or 'modelo'} · pesos SHA-256 {model.get('weights_sha256') or '—'}"
        if model.get("conf") is not None:
            model_text += f" · confiança mínima {model['conf']}"
    else:
        model_text = "nenhum arquivo de sugestões nesta revisão"
    # Pares (rótulo, HTML já escapado) da tabela de identificação.
    identification = [("Cliente", _esc(header["client"] or "—")), ("Local / edificação", _esc(header["site"] or "—")),
                      ("Grupos de edificação no registro", _esc(", ".join(header["buildings"]) or "—")),
                      ("Data de emissão", _esc(header["issued_at"])), ("Revisor(es)", _esc(", ".join(header["reviewers"]) or "—")),
                      ("Inspetor / responsável em campo", _esc(header["inspector"] or "—")),
                      ("Versão da revisão", _esc(header["registry_version"])),
                      ("SHA-256 do registro", f'<span class="mono">{_esc(header["registry_sha256"])}</span>'),
                      ("Taxonomia", f'{_esc(header["taxonomy_version"] or "—")} · <span class="mono">{_esc(header["taxonomy_sha256"])}</span>'),
                      ("Modelo das sugestões automáticas", _esc(model_text)), ("Situação do documento", _esc(PRELIMINARY_TEXT))]
    if header["notes"]:
        identification.append(("Observações", _esc(header["notes"])))
    parts = [f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
             f"<title>{_esc(header['title'])}</title><style>{style}</style></head><body><main>",
             f'<section id="capa"><h1>{_esc(header["title"])}</h1>',
             f'<div class="banner">Documento {_esc(PRELIMINARY_TEXT)}. As caixas são evidências visuais revisáveis; '
             f"gravidade, diagnóstico e recomendação não foram inferidos e ficam em branco.</div><table>",
             "".join(f'<tr><th style="width:34%">{_esc(key)}</th><td>{value}</td></tr>' for key, value in identification),
             "</table></section>",
             '<section id="resumo"><h2>1. Resumo</h2><table><tr><th>Fotos no registro</th><th>Com ocorrências confirmadas (positivas)</th>'
             "<th>Negativas</th><th>Ambíguas</th><th>Excluídas</th><th>Pendentes (rascunho)</th><th>Com decisão humana</th></tr>",
             f"<tr><td>{summary['images']}</td><td>{summary['positive']}</td><td>{summary['negative']}</td><td>{summary['ambiguous']}</td>"
             f"<td>{summary['excluded']}</td><td>{summary['pending']}</td><td>{summary['decided_by_human']}</td></tr></table>",
             "<h3>Ocorrências confirmadas por classe</h3><table><tr><th>Classe</th><th>Ocorrências (caixas)</th><th>Fotos</th>"
             "<th>Critério de identificação</th><th>Gravidade de referência da taxonomia (não aplicada automaticamente)</th></tr>"]
    for row in summary["by_class"]:
        parts.append(f"<tr><td>{_esc(row['label'])}</td><td>{row['boxes']}</td><td>{row['images']}</td><td>{_esc(row['criterion'] or '—')}</td>"
                     f"<td>{_esc(row['reference_severity'] or '—')}</td></tr>")
    parts.append(f"<tr><th>Total</th><th>{summary['confirmed_boxes']}</th><th>{summary['positive']}</th><th colspan=\"2\">"
                 "Contagem de caixas por foto; fotos semelhantes podem repetir o mesmo defeito.</th></tr></table></section>")
    parts.append('<section id="ocorrencias"><h2>2. Ocorrências por foto</h2>'
                 f'<p>Cada foto traz a evidência demarcada e três campos reservados ao responsável técnico, preenchidos como "{_esc(PENDING_TEXT)}".</p>')
    shown = [image for image in report["images"] if image["included_in_report"]]
    if not shown:
        parts.append("<p>Nenhuma foto com ocorrências confirmadas nesta revisão.</p>")
    for image in shown:
        number = image["numero_foto"]
        encoded = base64.b64encode(evidences[number]).decode("ascii")
        boxes = [box for box in report["occurrences"] + report["draft_boxes"] if box["numero_foto"] == number]
        box_rows = _box_rows_html([{"class_id": box["class_id"], "bbox_xyxy": box["bbox_xyxy"], "note": box["nota"]} for box in boxes], labels)
        parts.append(f'<article class="foto" id="foto-{number:03d}"><h3>Foto {number:03d} — {_esc(image["filename"])}</h3>'
                     f'<figure><img class="evidencia" src="data:image/jpeg;base64,{encoded}" alt="Evidência da foto {number:03d} com {len(boxes)} caixa(s)">'
                     f'<figcaption>{_esc(image["filename"])} · {image["width"]} × {image["height"]} px · SHA-256 <span class="mono">{_esc(image["image_sha256"])}</span></figcaption></figure>'
                     f'<table><tr><th style="width:26%">Grupo / cena</th><td>{_esc(image["building_group"] or "—")} / {_esc(image["scene_group"] or "—")}</td></tr>'
                     f'<tr><th>Páginas de origem</th><td>{_esc(", ".join(map(str, image["pages"])) or "—")}</td></tr>'
                     f'<tr><th>Status</th><td>{_esc(image["status_label"])}{" · revisão integral confirmada" if image["confirmed_complete"] else ""}</td></tr>'
                     f'<tr><th>Origem das caixas</th><td>{_esc(image["source_label"])}</td></tr>'
                     f'<tr><th>Notas da revisão</th><td>{_esc(image["notes"] or "—")}</td></tr></table>'
                     f'<table><tr><th>Nº</th><th>Classe</th><th>Coordenadas (x1, y1 – x2, y2)</th><th>Nota</th></tr>'
                     f'{box_rows or "<tr><td colspan=4>Sem caixas.</td></tr>"}</table>'
                     f'<table class="avaliacao"><tr><th style="width:26%">Gravidade</th><td class="pendente">{_esc(PENDING_TEXT)}</td></tr>'
                     f'<tr><th>Diagnóstico</th><td class="pendente">{_esc(PENDING_TEXT)}</td></tr>'
                     f'<tr><th>Recomendação</th><td class="pendente">{_esc(PENDING_TEXT)}</td></tr></table></article>')
    parts.append("</section>")
    others = [image for image in report["images"] if image["status"] != "positive"]
    parts.append('<section id="anexo-a"><h2>Anexo A — Fotos sem ocorrências confirmadas, ambíguas, excluídas ou pendentes</h2>')
    if others:
        parts.append("<table><tr><th>Nº</th><th>Arquivo</th><th>Status</th><th>Origem</th><th>Caixas não confirmadas</th><th>Notas</th></tr>")
        parts.extend(f"<tr><td>{image['numero_foto']:03d}</td><td>{_esc(image['filename'])}</td><td>{_esc(image['status_label'])}</td>"
                     f"<td>{_esc(image['source_label'])}</td><td>{image['boxes']}</td><td>{_esc(image['notes'] or '—')}</td></tr>" for image in others)
        parts.append("</table>")
    else:
        parts.append("<p>Todas as fotos do registro têm ocorrências confirmadas.</p>")
    parts.append("</section>")
    pending = report["pending_suggestions"]
    parts.append('<section id="anexo-b"><h2>Anexo B — Sugestões automáticas pendentes</h2>'
                 f"<p>{_esc(pending['note'])}</p>")
    if pending["by_class"]:
        parts.append("<table><tr><th>Classe</th><th>Sugestões pendentes</th><th>Fotos</th></tr>")
        parts.extend(f"<tr><td>{_esc(row['label'])}</td><td>{row['count']}</td><td>{row['images']}</td></tr>" for row in pending["by_class"])
        parts.append(f"<tr><th>Total</th><th>{pending['total']}</th><th></th></tr></table>")
    else:
        parts.append("<p>Nenhuma sugestão automática pendente nesta revisão.</p>")
    parts.append("</section>")
    parts.append('<section id="metodologia"><h2>Metodologia e limitações</h2><ol class="metodologia">'
                 + "".join(f"<li>{_esc(text)}</li>" for text in METHODOLOGY) + "</ol></section>")
    parts.append(f"<footer>Gerado em {_esc(report['provenance']['generated_at'])} por DroneCamp Vision "
                 f"({_esc(report['provenance']['generator'])}). Documento {_esc(PRELIMINARY_TEXT)}.</footer></main></body></html>")
    return "".join(parts)


def _write_pdf(path: Path, report: dict, evidences: dict[int, bytes], labels: dict[int, str]) -> None:
    """PDF com reportlab (platypus): capa, resumo, ocorrências por foto, anexos e metodologia.

    Fontes padrão (Helvetica, codificação WinAnsi): acentos do pt-BR funcionam.
    """
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    base = ParagraphStyle("base", fontName="Helvetica", fontSize=9.5, leading=12.5)
    cell = ParagraphStyle("cell", parent=base, fontSize=8, leading=10)
    cell_bold = ParagraphStyle("cell_bold", parent=cell, fontName="Helvetica-Bold")
    pending_style = ParagraphStyle("pending", parent=cell, fontName="Helvetica-Oblique", textColor=rl_colors.HexColor("#92400E"))
    title_style = ParagraphStyle("title", parent=base, fontName="Helvetica-Bold", fontSize=18, leading=22, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=13, leading=16, spaceBefore=10, spaceAfter=6)
    h3 = ParagraphStyle("h3", parent=base, fontName="Helvetica-Bold", fontSize=10.5, leading=13, spaceBefore=6, spaceAfter=4)
    banner = ParagraphStyle("banner", parent=base, fontName="Helvetica-Bold", textColor=rl_colors.HexColor("#7C2D12"),
                            backColor=rl_colors.HexColor("#FFF7ED"), borderColor=rl_colors.HexColor("#B45309"), borderWidth=1,
                            borderPadding=6, spaceBefore=8, spaceAfter=14)

    def p(text, style=base):
        return Paragraph(html.escape("" if text is None else str(text)), style)

    grid = TableStyle([("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#CBD5E1")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                       ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#F1F5F9"))])
    key_grid = TableStyle([("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#CBD5E1")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("BACKGROUND", (0, 0), (0, -1), rl_colors.HexColor("#F1F5F9"))])

    def table(rows, widths, style=grid, header=True):
        data = [[p(value, cell_bold if header and index == 0 else cell) if not isinstance(value, Paragraph) else value
                 for value in row] for index, row in enumerate(rows)]
        return Table(data, colWidths=widths, repeatRows=1 if header else 0, style=style)

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title=report["header"]["title"], author="DroneCamp Vision", subject=PRELIMINARY_TEXT)
    width = doc.width
    header, summary = report["header"], report["summary"]
    model = header.get("suggestion_model") or {}
    model_text = (f"{model.get('model_label') or 'modelo'} · pesos SHA-256 {model.get('weights_sha256') or '—'}") if model else "nenhum arquivo de sugestões nesta revisão"
    story = [p(header["title"], title_style),
             p(f"Documento {PRELIMINARY_TEXT}. As caixas são evidências visuais revisáveis; gravidade, diagnóstico e recomendação "
               "não foram inferidos e ficam em branco.", banner)]
    identification = [("Cliente", header["client"] or "—"), ("Local / edificação", header["site"] or "—"),
                      ("Grupos de edificação no registro", ", ".join(header["buildings"]) or "—"), ("Data de emissão", header["issued_at"]),
                      ("Revisor(es)", ", ".join(header["reviewers"]) or "—"), ("Inspetor / responsável em campo", header["inspector"] or "—"),
                      ("Versão da revisão", header["registry_version"]), ("SHA-256 do registro", header["registry_sha256"]),
                      ("Taxonomia", f"{header['taxonomy_version'] or '—'} · {header['taxonomy_sha256']}"),
                      ("Modelo das sugestões automáticas", model_text), ("Situação do documento", PRELIMINARY_TEXT)]
    if header["notes"]:
        identification.append(("Observações", header["notes"]))
    story.append(table([[p(key, cell_bold), p(value, cell)] for key, value in identification], [width * 0.32, width * 0.68], key_grid, header=False))
    story += [PageBreak(), p("1. Resumo", h2),
              table([["Fotos", "Positivas", "Negativas", "Ambíguas", "Excluídas", "Pendentes", "Com decisão humana"],
                     [summary["images"], summary["positive"], summary["negative"], summary["ambiguous"], summary["excluded"], summary["pending"], summary["decided_by_human"]]],
                    [width / 7] * 7),
              Spacer(1, 6), p("Ocorrências confirmadas por classe", h3)]
    class_rows = [["Classe", "Caixas", "Fotos", "Critério de identificação", "Gravidade de referência da taxonomia (não aplicada automaticamente)"]]
    class_rows += [[row["label"], row["boxes"], row["images"], row["criterion"] or "—", row["reference_severity"] or "—"] for row in summary["by_class"]]
    class_rows.append(["Total", summary["confirmed_boxes"], summary["positive"], "Contagem de caixas por foto; fotos semelhantes podem repetir o mesmo defeito.", ""])
    story.append(table(class_rows, [width * 0.2, width * 0.08, width * 0.08, width * 0.38, width * 0.26]))
    story += [PageBreak(), p("2. Ocorrências por foto", h2),
              p(f'Cada foto traz a evidência demarcada e três campos reservados ao responsável técnico, preenchidos como "{PENDING_TEXT}".')]
    shown = [image for image in report["images"] if image["included_in_report"]]
    if not shown:
        story.append(p("Nenhuma foto com ocorrências confirmadas nesta revisão."))
    for image in shown:
        number = image["numero_foto"]
        data = evidences[number]
        image_width, image_height = ImageReader(io.BytesIO(data)).getSize()
        ratio = min(width / image_width, (110 * mm) / image_height, 1.0)
        boxes = [box for box in report["occurrences"] + report["draft_boxes"] if box["numero_foto"] == number]
        box_rows = [["Nº", "Classe", "Coordenadas (x1, y1 – x2, y2)", "Nota"]]
        for index, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = (_number(value) for value in box["bbox_xyxy"])
            box_rows.append([index, labels.get(box["class_id"], ""), f"{x1}, {y1} – {x2}, {y2}", box["nota"]])
        block = [p(f"Foto {number:03d} — {image['filename']}", h3),
                 RLImage(io.BytesIO(data), width=image_width * ratio, height=image_height * ratio),
                 p(f"{image['filename']} · {image['width']} × {image['height']} px · SHA-256 {image['image_sha256']}", cell), Spacer(1, 4),
                 table([[p("Grupo / cena", cell_bold), p(f"{image['building_group'] or '—'} / {image['scene_group'] or '—'}", cell)],
                        [p("Páginas de origem", cell_bold), p(", ".join(map(str, image["pages"])) or "—", cell)],
                        [p("Status", cell_bold), p(image["status_label"] + (" · revisão integral confirmada" if image["confirmed_complete"] else ""), cell)],
                        [p("Origem das caixas", cell_bold), p(image["source_label"], cell)],
                        [p("Notas da revisão", cell_bold), p(image["notes"] or "—", cell)]], [width * 0.26, width * 0.74], key_grid, header=False),
                 Spacer(1, 4),
                 table(box_rows if len(box_rows) > 1 else box_rows + [["", "Sem caixas.", "", ""]], [width * 0.07, width * 0.3, width * 0.3, width * 0.33]),
                 Spacer(1, 4),
                 table([[p("Gravidade", cell_bold), p(PENDING_TEXT, pending_style)], [p("Diagnóstico", cell_bold), p(PENDING_TEXT, pending_style)],
                        [p("Recomendação", cell_bold), p(PENDING_TEXT, pending_style)]], [width * 0.26, width * 0.74], key_grid, header=False),
                 Spacer(1, 10)]
        story.append(KeepTogether(block[:3]))
        story.extend(block[3:])
    story += [PageBreak(), p("Anexo A — Fotos sem ocorrências confirmadas, ambíguas, excluídas ou pendentes", h2)]
    others = [image for image in report["images"] if image["status"] != "positive"]
    if others:
        rows = [["Nº", "Arquivo", "Status", "Origem", "Caixas não confirmadas", "Notas"]]
        rows += [[f"{image['numero_foto']:03d}", image["filename"], image["status_label"], image["source_label"], image["boxes"], image["notes"] or "—"] for image in others]
        story.append(table(rows, [width * 0.06, width * 0.2, width * 0.14, width * 0.2, width * 0.1, width * 0.3]))
    else:
        story.append(p("Todas as fotos do registro têm ocorrências confirmadas."))
    pending = report["pending_suggestions"]
    story += [Spacer(1, 10), p("Anexo B — Sugestões automáticas pendentes", h2), p(pending["note"])]
    if pending["by_class"]:
        rows = [["Classe", "Sugestões pendentes", "Fotos"]] + [[row["label"], row["count"], row["images"]] for row in pending["by_class"]]
        rows.append(["Total", pending["total"], ""])
        story.append(table(rows, [width * 0.5, width * 0.25, width * 0.25]))
    else:
        story.append(p("Nenhuma sugestão automática pendente nesta revisão."))
    story += [Spacer(1, 10), p("Metodologia e limitações", h2)]
    story += [p(f"{index}. {text}") for index, text in enumerate(METHODOLOGY, start=1)]
    story.append(Spacer(1, 8))
    story.append(p(f"Gerado em {report['provenance']['generated_at']} por DroneCamp Vision ({report['provenance']['generator']}).", cell))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(document.leftMargin, 9 * mm, f"{header['title']} — {PRELIMINARY_TEXT}")
        canvas.drawRightString(A4[0] - document.rightMargin, 9 * mm, f"Página {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def build_report(config: ProjectConfig, registry_path: Path, output_dir: Path, *, decisions: dict | None = None,
                 suggestions: dict | None = None, reviewer: str = "", title: str | None = None, client: str = "",
                 site: str = "", inspector: str = "", notes: str = "", pdf: bool = True, max_side: int = 1400,
                 only_positive_images: bool = True) -> dict:
    """Pasta ``<stamp UTC>_laudo_<versão>/`` com ``laudo.html``, ``laudo.json``, ``laudo.pdf`` e ``manifesto.json``.

    O HTML é autossuficiente (CSS inline, evidências em base64 reduzidas a
    ``max_side``, pronto para imprimir em A4). O PDF só é gerado com
    ``pdf=True`` e ``reportlab`` instalado (``pip install dronecamp-ia[reports]``);
    caso contrário ``pdf`` volta None e ``pdf_reason`` explica.
    ``only_positive_images=False`` inclui evidência de todas as fotos no corpo.

    Nada de gravidade, causa, medida ou diagnóstico é inferido: esses campos
    saem como "pendente de validação técnica". Devolve ``{"directory", "html",
    "pdf", "pdf_reason", "json", "manifest", "images" (fotos com evidência no
    corpo), "occurrences" (caixas confirmadas), "pending_suggestions"}``.
    """
    if type(max_side) is not int or max_side < 16:
        raise ValueError("max_side deve ser um inteiro ≥ 16.")
    context = _load_context(config, registry_path, decisions, suggestions)
    generated = _utc_now()
    generated_at = generated.isoformat(timespec="seconds")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    directory = _unique_path(output_dir, f"{generated.strftime('%Y%m%dT%H%M%SZ')}_laudo_{_safe_name(context.version)}")
    directory.mkdir()
    try:
        return _build_report_files(config, context, directory, generated_at, reviewer=reviewer, title=title, client=client,
                                   site=site, inspector=inspector, notes=notes, pdf=pdf, max_side=max_side,
                                   only_positive_images=only_positive_images)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)  # Nada de laudo pela metade.
        raise


def _build_report_files(config: ProjectConfig, context: _Context, directory: Path, generated_at: str, *, reviewer: str,
                        title: str | None, client: str, site: str, inspector: str, notes: str, pdf: bool, max_side: int,
                        only_positive_images: bool) -> dict:
    entries, labels = context.entries, context.labels
    reviewers = [value for value in [reviewer.strip()] + context.reviewers if value]
    reviewers = list(dict.fromkeys(reviewers))
    buildings = sorted({str(entry["image"].get("building_group") or "") for entry in entries} - {""})
    # 1. Dados estruturados (laudo.json) — a mesma fonte alimenta HTML e PDF.
    images, occurrences, drafts = [], [], []
    for entry in entries:
        item = entry["image"]
        included = entry["status"] == "positive" or not only_positive_images
        images.append({"numero_foto": entry["position"], "image_sha256": item["image_sha256"], "filename": item["filename"],
                       "width": item["width"], "height": item["height"], "pages": list(item.get("pages", [])),
                       "building_group": item.get("building_group"), "scene_group": item.get("scene_group"),
                       "status": entry["status"], "status_label": STATUS_LABELS[entry["status"]], "source": entry["source"],
                       "source_label": SOURCE_LABELS[entry["source"]], "confirmed_complete": entry["confirmed_complete"],
                       "boxes": len(entry["boxes"]), "pending_suggestions": entry["pending_suggestions"], "notes": entry["notes"],
                       "included_in_report": included})
        for number, box in enumerate(entry["boxes"], start=1):
            record = {"numero_foto": entry["position"], "image_sha256": item["image_sha256"], "filename": item["filename"],
                      "caixa": number, "class_id": box["class_id"], "classe": labels.get(box["class_id"]),
                      "bbox_xyxy": [_number(value) for value in box["bbox_xyxy"]], "nota": box.get("note", "") or "",
                      "status_foto": entry["status"], "origem": entry["source"], "gravidade": PENDING_TEXT,
                      "diagnostico": PENDING_TEXT, "recomendacao": PENDING_TEXT}
            (occurrences if entry["status"] == "positive" else drafts).append(record)
    counts = _counts(entries, labels)
    by_class = []
    for category in context.classes:
        confirmed = [box for box in occurrences if box["class_id"] == category["id"]]
        by_class.append({"class_id": category["id"], "slug": category["slug"], "label": category["label"], "boxes": len(confirmed),
                         "images": len({box["numero_foto"] for box in confirmed}), "criterion": category.get("criterion", ""),
                         "reference_severity": category.get("reference_severity")})
    pending_by_class = []
    for category in context.classes:
        matching = [(entry["position"], suggestion) for entry in entries for suggestion in entry["pending"] if suggestion["class_id"] == category["id"]]
        if matching:
            pending_by_class.append({"class_id": category["id"], "label": category["label"], "count": len(matching),
                                     "images": len({position for position, _ in matching})})
    report = {
        "schema_version": 1, "document": "laudo_preliminar_inspecao_assistida", "status": PRELIMINARY_TEXT,
        "header": {"title": title or f"Laudo preliminar de inspeção assistida — {context.version}", "client": client, "site": site,
                   "buildings": buildings, "issued_at": generated_at, "reviewers": reviewers, "inspector": inspector, "notes": notes,
                   "registry_version": context.version, "registry_sha256": context.registry_sha256,
                   "taxonomy_version": context.registry.get("taxonomy_version"), "taxonomy_sha256": context.taxonomy_sha256,
                   "suggestion_model": context.suggestion_model},
        "summary": {"images": counts["images"], **counts["by_status"], "decided_by_human": counts["decided_by_human"],
                    "confirmed_boxes": counts["confirmed_boxes"], "draft_boxes": len(drafts), "by_class": by_class,
                    "pending_suggestions": counts["pending_suggestions"]},
        "images": images, "occurrences": occurrences, "draft_boxes": drafts,
        "pending_suggestions": {"total": counts["pending_suggestions"], "by_class": pending_by_class,
                                "note": "Sugestões automáticas pendentes não constituem ocorrências; aguardam decisão humana na plataforma de revisão."},
        "provenance": {"generated_at": generated_at, "generator": "dronecamp_ia.exports.build_report",
                       "registry_file": context.registry_path.name, "review_directory": context.registry_path.parent.name,
                       "registry_sha256": context.registry_sha256, "taxonomy_sha256": context.taxonomy_sha256,
                       "decisions_provided": context.decisions_count, "suggestion_model": context.suggestion_model,
                       "suggestion_note": context.suggestion_note,
                       "parameters": {"max_side": max_side, "only_positive_images": only_positive_images, "pdf_requested": pdf}},
        "methodology": METHODOLOGY,
    }
    # 2. Evidências reduzidas (JPEG qualidade 82), uma foto por vez, só das fotos que entram no corpo.
    evidences: dict[int, bytes] = {}
    for entry in entries:
        if not (entry["status"] == "positive" or not only_positive_images):
            continue
        source = _checked_source(config, entry["image"])
        evidence = _render_evidence(source, entry, context, max_side, include_suggestions=False)
        buffer = io.BytesIO()
        evidence.save(buffer, "JPEG", quality=82)
        evidence.close()
        evidences[entry["position"]] = buffer.getvalue()
    # 3. Arquivos.
    html_path, json_path, pdf_path, manifest_path = (directory / name for name in ("laudo.html", "laudo.json", "laudo.pdf", "manifesto.json"))
    html_path.write_text(_report_html(report, evidences, labels), encoding="utf-8")
    write_json(json_path, report)
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
                _write_pdf(pdf_path, report, evidences, labels)
            except Exception as error:  # HTML e JSON continuam válidos; o motivo vai na resposta.
                pdf_path.unlink(missing_ok=True)
                pdf_reason = f"falha ao gerar o PDF: {type(error).__name__}: {error}"
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
    return {"directory": str(directory), "html": str(html_path), "pdf": str(pdf_path) if pdf_path.is_file() else None,
            "pdf_reason": pdf_reason, "json": str(json_path), "manifest": str(manifest_path), "images": len(evidences),
            "occurrences": len(occurrences), "pending_suggestions": counts["pending_suggestions"]}
