"""Apoio de um modelo de visão e linguagem (VLM) gratuito: descrever ocorrências e
rascunhar texto de apoio ao laudo, sempre como pendência para decisão humana.

Função no projeto: o detector YOLO acha e classifica caixas; ele não explica o
que viu. Um VLM gratuito (OpenRouter, roteador ``openrouter/free`` ou um slug
fixo) recebe o recorte de cada caixa e descreve só o que está visível. O texto
serve para (1) ajudar o revisor a decidir recortes ambíguos (segunda opinião
independente do detector), (2) medir a concordância com as classes humanas e
(3) rascunhar o texto de apoio do laudo. Nada aqui aprova caixa, atribui
gravidade, diagnóstico ou recomendação; esses campos ficam ``None`` com
``"pendente de validação técnica"``.

O que faz:
- ``load_vlm_settings``: lê ``configs/vlm_assist.json`` e a chave de
  ``OPENROUTER_API_KEY`` (ambiente ou ``.env`` local, nunca de arquivos versionados).
- ``OpenRouterClient``: chamada ``chat/completions`` compatível com OpenAI via
  ``urllib`` (sem dependência nova); ``transport`` injetável nos testes.
- ``describe_occurrences``: recorta cada caixa (humana e, opcionalmente, sugerida),
  pergunta ao VLM e grava ``ai_notes.json`` ao lado do registro, retomável.
- ``agreement_report``: concordância da classe sugerida pelo VLM com a humana.
- ``draft_report_support``: ``laudo_apoio.json``/``.md`` com as ocorrências
  revisadas, descrições pendentes e campos técnicos em branco.
- ``list_free_vision_models``: modelos gratuitos com entrada de imagem no OpenRouter.

Quando mexer: prompt (``PROMPT_VERSION`` muda e invalida notas antigas),
tamanho dos recortes ou campos gravados. Sem chave ou sem rede o comando falha
com explicação; nunca inventa descrições.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import base64
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

from PIL import Image

from .config import ProjectConfig, load_taxonomy
from .io import file_hash, resolve_local_path, write_json

SETTINGS_FILE = "vlm_assist.json"
ENV_KEY = "OPENROUTER_API_KEY"
ENV_MODEL = "OPENROUTER_MODEL"
ENV_BASE_URL = "OPENROUTER_BASE_URL"
PROMPT_VERSION = 1
NOTES_FILE = "ai_notes.json"
PENDING = "pendente de validação técnica"
WARNING = ("Descrições geradas por IA a partir de recortes: apoiam a revisão, não a substituem. "
           "Classe, gravidade, diagnóstico e recomendações dependem de decisão humana.")

# Respostas aceitas para a classe sugerida além dos slugs da taxonomia.
NO_CLASS = ("nenhuma", "incerto")
LEGIBILITY = ("boa", "limitada", "insuficiente")


class VLMUnavailable(RuntimeError):
    """Serviço, chave ou rede indisponíveis: o comando para sem gravar descrições."""


# ---------------------------------------------------------------------------
# Configuração e chave (a chave nunca é gravada nem impressa).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VLMSettings:
    provider: str
    base_url: str
    model: str
    api_key: str | None
    timeout_seconds: float
    pause_seconds: float
    max_output_tokens: int
    temperature: float
    crop_margin: float
    max_crop_side: int
    min_crop_side: int
    max_boxes_per_image: int
    app_title: str
    settings_sha256: str

    def describe(self) -> dict:
        """O que fica registrado nas saídas: sem chave, só se havia uma."""
        return {"provider": self.provider, "base_url": self.base_url, "model_requested": self.model,
                "api_key_present": bool(self.api_key), "settings_sha256": self.settings_sha256,
                "crop_margin": self.crop_margin, "max_crop_side": self.max_crop_side, "prompt_version": PROMPT_VERSION}


def _read_dotenv(path: Path) -> dict[str, str]:
    """``CHAVE=valor`` por linha; só as variáveis deste módulo são lidas."""
    values = {}
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if key in (ENV_KEY, ENV_MODEL, ENV_BASE_URL):
            values[key] = value.strip().strip("'\"")
    return values


def load_vlm_settings(config: ProjectConfig, model: str | None = None, environ: dict | None = None) -> VLMSettings:
    """Lê configs/vlm_assist.json; a chave vem do ambiente ou de sistema_ia/.env."""
    path = config.taxonomy_path.parent / SETTINGS_FILE
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError(f"{SETTINGS_FILE} precisa de schema_version: 1.")
    env = dict(os.environ if environ is None else environ)
    for key, value in _read_dotenv(config.root / ".env").items():
        env.setdefault(key, value)
    api_key = (env.get(ENV_KEY) or "").strip() or None
    if api_key and not re.fullmatch(r"[A-Za-z0-9._\-]{16,}", api_key):
        raise ValueError(f"{ENV_KEY} tem formato inesperado; confira a chave (sem espaços ou aspas).")
    return VLMSettings(
        provider=str(raw.get("provider", "openrouter")),
        base_url=str(env.get(ENV_BASE_URL) or raw.get("base_url", "https://openrouter.ai/api/v1")).rstrip("/"),
        model=str(model or env.get(ENV_MODEL) or raw.get("model", "openrouter/free")),
        api_key=api_key, timeout_seconds=float(raw.get("timeout_seconds", 90)),
        pause_seconds=float(raw.get("pause_seconds", 1.0)), max_output_tokens=int(raw.get("max_output_tokens", 400)),
        temperature=float(raw.get("temperature", 0.0)), crop_margin=float(raw.get("crop_margin", 0.3)),
        max_crop_side=int(raw.get("max_crop_side", 768)), min_crop_side=int(raw.get("min_crop_side", 96)),
        max_boxes_per_image=int(raw.get("max_boxes_per_image", 20)), app_title=str(raw.get("app_title", "DroneCamp Vision")),
        settings_sha256=file_hash(path),
    )


# ---------------------------------------------------------------------------
# Cliente HTTP (API compatível com OpenAI). ``transport`` é trocado nos testes.
# ---------------------------------------------------------------------------

def _urllib_transport(url: str, headers: dict, body: bytes | None, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST" if body else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


class OpenRouterClient:
    def __init__(self, settings: VLMSettings, transport=None):
        if not settings.api_key:
            raise VLMUnavailable(f"Defina {ENV_KEY} (variável de ambiente ou arquivo .env em sistema_ia/) "
                                 "para usar o apoio do VLM; a chave não fica em nenhum arquivo versionado.")
        self.settings = settings
        self.transport = transport or _urllib_transport
        self.calls = 0

    def _headers(self) -> dict:
        # Cabeçalhos opcionais do OpenRouter (HTTP-Referer/X-Title) só identificam o app nos rankings.
        return {"Authorization": f"Bearer {self.settings.api_key}", "Content-Type": "application/json",
                "X-Title": self.settings.app_title}

    def _request(self, path: str, payload: dict | None) -> dict:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        try:
            status, raw = self.transport(f"{self.settings.base_url}{path}", self._headers(), body, self.settings.timeout_seconds)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise VLMUnavailable(f"Sem acesso a {self.settings.base_url}: {type(error).__name__}: {error}") from error
        self.calls += 1
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        if status in (401, 403):
            raise VLMUnavailable(f"OpenRouter recusou a chave (HTTP {status}). Gere outra chave e defina {ENV_KEY}.")
        if status == 429:
            raise VLMUnavailable("Limite de uso dos modelos gratuitos atingido (HTTP 429); tente mais tarde ou troque o modelo.")
        if status >= 400:
            raise VLMUnavailable(f"OpenRouter respondeu HTTP {status}: {text[:300]}")
        try:
            data = json.loads(text)
        except ValueError as error:
            raise VLMUnavailable(f"Resposta não é JSON: {text[:200]}") from error
        if isinstance(data, dict) and data.get("error"):
            raise VLMUnavailable(f"Erro do OpenRouter: {json.dumps(data['error'], ensure_ascii=False)[:300]}")
        return data

    def chat(self, messages: list[dict]) -> tuple[str, dict]:
        """Texto da resposta e metadados (modelo que de fato respondeu, uso de tokens)."""
        payload = {"model": self.settings.model, "messages": messages, "temperature": self.settings.temperature,
                   "max_tokens": self.settings.max_output_tokens}
        data = self._request("/chat/completions", payload)
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise VLMUnavailable(f"Resposta sem conteúdo: {json.dumps(data, ensure_ascii=False)[:300]}") from error
        if isinstance(content, list):  # alguns provedores devolvem partes
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        meta = {"model_used": data.get("model"), "provider": data.get("provider"), "usage": data.get("usage"),
                "finish_reason": choice.get("finish_reason")}
        return str(content or ""), meta

    def models(self) -> list[dict]:
        data = self._request("/models", None)
        return list(data.get("data", [])) if isinstance(data, dict) else []


def list_free_vision_models(settings: VLMSettings, transport=None) -> list[dict]:
    """Modelos gratuitos (sufixo ``:free`` ou preço zero) que aceitam imagem na entrada."""
    client = OpenRouterClient(settings, transport)
    rows = []
    for item in client.models():
        identifier = str(item.get("id", ""))
        pricing = item.get("pricing") or {}
        free = identifier.endswith(":free") or identifier == "openrouter/free" or (
            str(pricing.get("prompt", "1")) in ("0", "0.0") and str(pricing.get("completion", "1")) in ("0", "0.0"))
        modalities = (item.get("architecture") or {}).get("input_modalities") or []
        if free and "image" in modalities:
            rows.append({"id": identifier, "name": item.get("name"), "context_length": item.get("context_length"),
                         "input_modalities": modalities})
    return sorted(rows, key=lambda row: row["id"])


# ---------------------------------------------------------------------------
# Recortes e prompt.
# ---------------------------------------------------------------------------

def crop_box(image: Image.Image, bbox: list[float], margin: float, max_side: int, min_side: int) -> tuple[bytes, list[int]]:
    """JPEG do recorte com margem (contexto) e lado ≤ max_side; devolve também o recorte usado."""
    width, height = image.size
    x1, y1, x2, y2 = (float(value) for value in bbox)
    pad_x, pad_y = (x2 - x1) * margin, (y2 - y1) * margin
    # Garante um recorte mínimo: caixas de 20 px viram um quadrado legível.
    extra_x = max(0.0, (min_side - (x2 - x1 + 2 * pad_x)) / 2)
    extra_y = max(0.0, (min_side - (y2 - y1 + 2 * pad_y)) / 2)
    left = max(0, int(x1 - pad_x - extra_x))
    top = max(0, int(y1 - pad_y - extra_y))
    right = min(width, int(round(x2 + pad_x + extra_x)))
    bottom = min(height, int(round(y2 + pad_y + extra_y)))
    if right <= left or bottom <= top:
        raise ValueError(f"Caixa fora da foto: {bbox}.")
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    scale = max_side / max(crop.size)
    if scale < 1:
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.LANCZOS)
    buffer = BytesIO()
    crop.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue(), [left, top, right, bottom]


def taxonomy_prompt(taxonomy: dict) -> str:
    """Classes ativas com critério, na língua do revisor, para o VLM escolher um slug."""
    lines = []
    for item in taxonomy["classes"]:
        if item.get("phase") == 1:
            lines.append(f"- {item['slug']}: {item['label']}. {item.get('criterion', '')}".strip())
    return "\n".join(lines)


def build_messages(taxonomy: dict, crop_jpeg: bytes) -> list[dict]:
    """Prompt fixo (versão PROMPT_VERSION): só o visível, sem gravidade nem recomendação."""
    system = (
        "Você apoia a revisão humana de inspeções de telhados industriais feitas por drone. "
        "Receberá o recorte de uma foto em torno de uma possível ocorrência. Descreva apenas o que está visível. "
        "Não estime gravidade, causa, risco, urgência nem recomendação; não presuma o que não aparece. "
        "Responda somente com um objeto JSON com as chaves: "
        "\"descricao\" (1 a 3 frases em português, objetivas), "
        "\"classe_sugerida\" (um slug da lista, \"nenhuma\" se não houver não conformidade visível ou \"incerto\"), "
        "\"evidencias\" (lista de até 4 detalhes visíveis que sustentam a descrição), "
        "\"legibilidade\" (\"boa\", \"limitada\" ou \"insuficiente\", conforme a nitidez do recorte)."
    )
    user_text = ("Classes do projeto (slug: nome. critério):\n" + taxonomy_prompt(taxonomy) +
                 "\n\nDescreva o recorte e responda só com o JSON pedido.")
    data_url = "data:image/jpeg;base64," + base64.b64encode(crop_jpeg).decode("ascii")
    return [{"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": user_text},
                                         {"type": "image_url", "image_url": {"url": data_url}}]}]


def prompt_digest(taxonomy: dict) -> str:
    """Identifica o prompt (versão + classes) para invalidar notas antigas."""
    system = build_messages(taxonomy, b"")[0]["content"]
    return sha256(json.dumps([PROMPT_VERSION, system, taxonomy_prompt(taxonomy)]).encode("utf-8")).hexdigest()


def resume_digest(settings: VLMSettings, prompt_sha256: str) -> str:
    """Chave da retomada: prompt + o que muda a resposta (modelo pedido, temperatura, tokens).

    Trocar ``--model``/``OPENROUTER_MODEL`` invalida as notas antigas: os recortes
    são enviados de novo ao modelo novo. Os parâmetros do recorte já entram no
    hash do JPEG de cada caixa.
    """
    return sha256(json.dumps([prompt_sha256, settings.model, settings.temperature, settings.max_output_tokens]).encode("utf-8")).hexdigest()


def parse_answer(text: str, names: list[str]) -> dict:
    """Aceita JSON puro, com cerca ``json`` ou com texto em volta; o que não encaixa vira ``incerto``."""
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end > start:
            candidate = candidate[start:end + 1]
    try:
        data = json.loads(candidate)
        if not isinstance(data, dict):
            raise ValueError
    except ValueError:
        return {"description": text.strip()[:600] or None, "class_guess": "incerto", "evidence": [],
                "legibility": None, "parsed": False}
    guess = str(data.get("classe_sugerida", "incerto")).strip().lower()
    if guess not in names and guess not in NO_CLASS:
        guess = "incerto"
    legibility = str(data.get("legibilidade", "")).strip().lower() or None
    evidence = data.get("evidencias")
    if not isinstance(evidence, list):
        evidence = [str(evidence)] if evidence else []
    return {"description": (str(data.get("descricao", "")).strip() or None),
            "class_guess": guess, "evidence": [str(item)[:200] for item in evidence][:4],
            "legibility": legibility if legibility in LEGIBILITY else None, "parsed": True}


# ---------------------------------------------------------------------------
# Descrição das ocorrências de uma revisão (ai_notes.json).
# ---------------------------------------------------------------------------

def _boxes_to_describe(registry: dict, suggestions: dict | None, include_suggestions: bool, per_image: int) -> list[tuple[dict, dict, str]]:
    """(foto, caixa, origem) das caixas humanas e, opcionalmente, das sugestões pendentes."""
    rows = []
    for item in registry["images"]:
        count = 0
        # Caixas de foto sem aprovação humana podem ser propostas de IA (first_pass_ai):
        # entram como "proposta" e ficam fora da concordância.
        source = "humana" if item.get("human_approved") is True else "proposta"
        for box in item.get("boxes", []):
            if count >= per_image:
                break
            rows.append((item, box, source))
            count += 1
        if include_suggestions and suggestions:
            for box in suggestions.get("images", {}).get(item["image_sha256"], []):
                if count >= per_image:
                    break
                rows.append((item, box, "sugestao"))
                count += 1
    return rows


def _slug(names: list[str], class_id) -> str | None:
    """Slug da classe ativa; caixas fora da taxonomia (registro antigo) ficam sem slug."""
    try:
        index = int(class_id)
    except (TypeError, ValueError):
        return None
    return names[index] if 0 <= index < len(names) else None


def _load_suggestions(registry_path: Path) -> dict | None:
    path = registry_path.parent / "suggestions.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("registry_sha256") != file_hash(registry_path):
        raise ValueError("suggestions.json pertence a outra versão do registro; gere as sugestões novamente.")
    return data


def describe_occurrences(config: ProjectConfig, registry_path: Path, include_suggestions: bool = False,
                         limit: int | None = None, model: str | None = None, transport=None,
                         resume: bool = True, environ: dict | None = None) -> Path:
    """Pergunte ao VLM o que há em cada recorte e grave ``ai_notes.json`` ao lado do registro.

    Retomável: notas já gravadas para o mesmo registro, prompt e recorte (hash do
    JPEG) não são pedidas de novo, o que respeita os limites dos modelos gratuitos.
    ``limit`` corta o número de chamadas desta execução. A classe humana ou
    sugerida não é mostrada ao VLM, para a concordância medir uma opinião
    independente.
    """
    registry_path = Path(registry_path).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    taxonomy = load_taxonomy(config.taxonomy_path)
    if registry.get("taxonomy_sha256") != file_hash(config.taxonomy_path):
        raise ValueError("Taxonomia mudou: migre a revisão antes de pedir descrições.")
    names = [item["slug"] for item in taxonomy["classes"] if item.get("phase") == 1]
    settings = load_vlm_settings(config, model, environ)
    client = OpenRouterClient(settings, transport)  # falha cedo sem chave
    digest = prompt_digest(taxonomy)
    resume_key = resume_digest(settings, digest)
    registry_sha = file_hash(registry_path)
    notes_path = registry_path.parent / NOTES_FILE
    previous = {}
    if resume and notes_path.is_file():
        old = json.loads(notes_path.read_text(encoding="utf-8"))
        if old.get("registry_sha256") == registry_sha and old.get("resume_sha256") == resume_key:
            previous = {entry["crop_sha256"]: entry for entries in old.get("images", {}).values() for entry in entries
                        if entry.get("description")}
    rows = _boxes_to_describe(registry, _load_suggestions(registry_path) if include_suggestions else None,
                              include_suggestions, settings.max_boxes_per_image)
    images: dict[str, list[dict]] = {}
    calls, errors, started = 0, 0, time.monotonic()
    for item, box, source in rows:
        path = resolve_local_path(item["source_path"], config.root)
        if file_hash(path) != item["image_sha256"]:
            raise ValueError(f"A foto original mudou: {item.get('filename')}.")
        with Image.open(path) as image:
            jpeg, crop = crop_box(image, box["bbox_xyxy"], settings.crop_margin, settings.max_crop_side, settings.min_crop_side)
        crop_sha = sha256(jpeg).hexdigest()
        entry = {"box_id": box.get("id"), "source": source, "class_id": box.get("class_id"),
                 "class_slug": _slug(names, box.get("class_id")),
                 "confidence": box.get("confidence"), "bbox_xyxy": box["bbox_xyxy"], "crop_xyxy": crop,
                 "crop_sha256": crop_sha, "status": "pendente", "severity": None, "diagnosis": None, "recommendation": None}
        if crop_sha in previous:
            cached = previous[crop_sha]
            entry.update({key: cached.get(key) for key in ("description", "class_guess", "evidence", "legibility",
                                                             "model_used", "parsed", "described_at")})
            entry["reused"] = True
        elif limit is not None and calls >= limit:
            entry.update({"description": None, "class_guess": None, "evidence": [], "legibility": None,
                          "model_used": None, "parsed": False, "skipped": "limite desta execução"})
        else:
            if calls and settings.pause_seconds > 0 and transport is None:
                time.sleep(settings.pause_seconds)
            try:
                text, meta = client.chat(build_messages(taxonomy, jpeg))
                calls += 1
                entry.update(parse_answer(text, names))
                entry.update({"model_used": meta.get("model_used"), "usage": meta.get("usage"),
                              "described_at": datetime.now(timezone.utc).isoformat(), "raw_answer": text[:1200]})
            except VLMUnavailable as error:
                # Chave recusada ou rede fora: para tudo, sem gravar descrições parciais inventadas.
                if calls == 0:
                    raise
                errors += 1
                entry.update({"description": None, "class_guess": None, "evidence": [], "legibility": None,
                              "model_used": None, "parsed": False, "error": str(error)[:300]})
        images.setdefault(item["image_sha256"], []).append(entry)
    described = [entry for entries in images.values() for entry in entries if entry.get("description")]
    report = {
        "schema_version": 1, "kind": "vlm_notes", "registry_sha256": registry_sha, "registry_version": registry.get("version"),
        "created_at": datetime.now(timezone.utc).isoformat(), "prompt_sha256": digest, "resume_sha256": resume_key,
        "settings": settings.describe(),
        "include_suggestions": include_suggestions, "calls_this_run": calls, "errors_this_run": errors,
        "seconds": round(time.monotonic() - started, 1), "warning": WARNING, "human_acceptance": "pendente",
        "images": images, "total_boxes": len(rows), "described": len(described),
        "agreement": agreement_report(images, names),
    }
    write_json(notes_path, report)
    return notes_path


def agreement_report(images: dict[str, list[dict]], names: list[str]) -> dict:
    """Concordância do VLM com a classe humana (só caixas humanas descritas), por classe.

    Mede se a descrição independente aponta a mesma classe; não valida o VLM
    como detector. ``nenhuma`` em caixa humana conta como discordância.
    """
    per_class = {name: {"boxes": 0, "agree": 0, "none_or_uncertain": 0, "confusions": {}} for name in names}
    total = agree = 0
    for entries in images.values():
        for entry in entries:
            if entry.get("source") != "humana" or not entry.get("description") or entry.get("class_slug") is None:
                continue
            row = per_class[entry["class_slug"]]
            row["boxes"] += 1
            total += 1
            guess = entry.get("class_guess")
            if guess == entry["class_slug"]:
                row["agree"] += 1
                agree += 1
            elif guess in NO_CLASS or guess is None:
                row["none_or_uncertain"] += 1
            else:
                row["confusions"][guess] = row["confusions"].get(guess, 0) + 1
    for row in per_class.values():
        row["agreement_rate"] = round(row["agree"] / row["boxes"], 3) if row["boxes"] else None
    return {"human_boxes_described": total, "agree": agree, "agreement_rate": round(agree / total, 3) if total else None,
            "per_class": {name: row for name, row in per_class.items() if row["boxes"]},
            "note": "Concordância de uma leitura independente do recorte com a classe humana; sem a classe no prompt."}


# ---------------------------------------------------------------------------
# Texto de apoio ao laudo: ocorrências revisadas + descrições pendentes.
# ---------------------------------------------------------------------------

def draft_report_support(config: ProjectConfig, registry_path: Path, output: Path | None = None) -> Path:
    """Grave ``laudo_apoio.json`` e ``laudo_apoio.md`` com as ocorrências da revisão.

    Entram só fotos com aprovação humana e suas caixas (registro na taxonomia
    atual, conferida pelo hash); a descrição do VLM
    (``ai_notes.json``, quando existir e for desta versão do registro) aparece
    marcada como pendente. Gravidade, diagnóstico e recomendação ficam em branco
    para o responsável técnico; nada é preenchido por inferência.
    """
    registry_path = Path(registry_path).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    # IDs de classe mudaram entre versões da taxonomia: um registro antigo sairia com slugs errados.
    if registry.get("taxonomy_sha256") != file_hash(config.taxonomy_path):
        raise ValueError("Taxonomia mudou: migre a revisão (migrate-review) antes de gerar o apoio ao laudo.")
    taxonomy = load_taxonomy(config.taxonomy_path)
    classes = {item["id"]: item for item in taxonomy["classes"] if item.get("phase") == 1}
    registry_sha = file_hash(registry_path)
    notes_path = registry_path.parent / NOTES_FILE
    notes = {}
    notes_meta = None
    if notes_path.is_file():
        data = json.loads(notes_path.read_text(encoding="utf-8"))
        if data.get("registry_sha256") == registry_sha:
            notes = data.get("images", {})
            notes_meta = {"created_at": data.get("created_at"), "model_requested": (data.get("settings") or {}).get("model_requested"),
                          "described": data.get("described"), "agreement": data.get("agreement")}
    occurrences, skipped = [], 0
    for item in registry["images"]:
        if not item.get("human_approved"):
            skipped += 1
            continue
        by_crop = {entry.get("box_id"): entry for entry in notes.get(item["image_sha256"], []) if entry.get("source") == "humana"}
        by_index = [entry for entry in notes.get(item["image_sha256"], []) if entry.get("source") == "humana"]
        for index, box in enumerate(item.get("boxes", [])):
            category = classes.get(int(box["class_id"]), {})
            note = by_crop.get(box.get("id")) if box.get("id") is not None else None
            if note is None and index < len(by_index) and by_index[index].get("bbox_xyxy") == box["bbox_xyxy"]:
                note = by_index[index]
            occurrences.append({
                "occurrence": len(occurrences) + 1, "image_sha256": item["image_sha256"], "filename": item.get("filename"),
                "pages": item.get("pages", []), "building_group": item.get("building_group"), "scene_group": item.get("scene_group"),
                "box_id": box.get("id"), "bbox_xyxy": box["bbox_xyxy"], "class_id": box["class_id"],
                "class_slug": category.get("slug"), "class_label": category.get("label"),
                "criterion": category.get("criterion"), "human_reviewer": item.get("human_reviewer"),
                "human_notes": item.get("human_notes") or None,
                "reference_severity_historical_only": category.get("reference_severity"),
                "ai_description": {"text": note.get("description"), "evidence": note.get("evidence", []),
                                   "legibility": note.get("legibility"), "class_guess": note.get("class_guess"),
                                   "model_used": note.get("model_used"), "status": "pendente"} if note and note.get("description") else None,
                "severity": None, "diagnosis": None, "recommendation": None, "validation": PENDING,
            })
    report = {"schema_version": 1, "kind": "report_support", "registry_sha256": registry_sha,
              "registry_version": registry.get("version"), "created_at": datetime.now(timezone.utc).isoformat(),
              "taxonomy_version": taxonomy.get("version"), "approved_images": len(registry["images"]) - skipped,
              "images_without_human_approval": skipped, "occurrences": occurrences, "ai_notes": notes_meta,
              "warning": WARNING, "human_acceptance": "pendente"}
    directory = Path(output) if output is not None else registry_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "laudo_apoio.json", report)
    (directory / "laudo_apoio.md").write_text(_support_markdown(report), encoding="utf-8")
    return directory / "laudo_apoio.json"


def _support_markdown(report: dict) -> str:
    lines = [f"# Apoio ao laudo — revisão {report.get('registry_version')}", "",
             f"Gerado em {report['created_at']}. {report['warning']}", "",
             f"Fotos com aprovação humana: {report['approved_images']}; sem aprovação (fora deste rascunho): "
             f"{report['images_without_human_approval']}.", ""]
    if report.get("ai_notes"):
        meta = report["ai_notes"]
        lines += [f"Descrições de IA: {meta.get('described')} recortes descritos por `{meta.get('model_requested')}` "
                  f"em {meta.get('created_at')} (pendentes de validação).", ""]
    for occurrence in report["occurrences"]:
        lines.append(f"## Ocorrência {occurrence['occurrence']} — {occurrence['class_label']} (`{occurrence['class_slug']}`)")
        lines.append(f"- Foto: `{occurrence['filename']}` (páginas {occurrence['pages'] or '—'}), caixa {occurrence['bbox_xyxy']}.")
        if occurrence.get("human_notes"):
            lines.append(f"- Nota do revisor: {occurrence['human_notes']}")
        if occurrence.get("ai_description"):
            ai = occurrence["ai_description"]
            evidence = "; ".join(ai.get("evidence") or []) or "—"
            lines.append(f"- Descrição da IA (pendente): {ai['text']} Evidências: {evidence}. Legibilidade: {ai.get('legibility') or '—'}.")
        else:
            lines.append("- Descrição da IA: não gerada.")
        lines += ["- Gravidade: a definir pelo responsável técnico.", "- Diagnóstico: a definir pelo responsável técnico.",
                  "- Recomendação: a definir pelo responsável técnico.", ""]
    return "\n".join(lines)
