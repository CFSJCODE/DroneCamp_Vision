"""Nomes legíveis para o que a plataforma mostra (treinos, modelos, inspeções, arquivos).

Função no projeto: transformar identificadores de disco como
``pilot_train_20261007T010057Z_f5e5ba8b`` ou ``ceasa_v8_revisao002_ff34e226416c``
em nomes que uma pessoa lê de relance ("Treino Piloto F5E5BA8B",
"CEASA Pavilhão A v8 · Revisão 002 FF34E226").

Regra geral: tipo em português + código curto (8 primeiros caracteres do hash,
em maiúsculas). Os nomes são DERIVADOS dos ids: pastas, arquivos e ids em disco
nunca mudam, porque registros, datasets, comparações e pesos se referenciam por
esses caminhos. Onde a rastreabilidade importa, a página mostra o id cru como
texto secundário ou dica (``title``).

O mesmo algoritmo existe em ``review_templates/platform.js`` (bloco "Nomes
legíveis"), para a página aberta do disco e para listas que só a página vê.
``tests/test_naming.py`` executa os dois com os mesmos ids e exige resultados
iguais: ao mudar uma regra aqui, mude lá também.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re

# Etapas de ``runs/<etapa>_<AAAAMMDDTHHMMSSZ>_<hash8>``.
STAGE_NAMES = {
    "pilot_train": "Treino Piloto", "train": "Treino de Produção", "tune": "Ajuste de Hiperparâmetros",
    "export": "Exportação", "export_demo": "Exportação de Demonstração", "predict_demo": "Predição de Demonstração",
    "autolabel_eval": "Avaliação da Marcação Automática", "calibrator": "Calibrador",
    "feedback_validation": "Validação de Revisão", "review_validation": "Validação de Revisão",
    "review_opening_fix": "Correção de Abertura",
}
RUN_ID = re.compile(r"^(?P<stage>[a-z0-9_]+?)_(?P<stamp>\d{8}T\d{6}Z)_(?P<hash>[0-9a-f]{8})$")
# Locais conhecidos (prefixo da pasta da inspeção até a versão vN).
PLACES = {"ceasa": "CEASA Pavilhão A", "ceasa_lado_a": "CEASA Lado A", "ceasa_ladoa": "CEASA Lado A",
          "internet": "Fotos da Internet"}
# Palavras das pastas sem acento → forma escrita. Conectivos ficam minúsculos.
WORDS = {"revisao": "Revisão", "rev": "Revisão", "prerevisao": "Pré-revisão", "sugestoes": "Sugestões",
         "humana": "Humana", "corrigida": "Corrigida", "taxonomia": "Taxonomia", "parcial": "Parcial",
         "piloto": "Piloto", "propostas": "Propostas", "busca": "Busca", "aberta": "Aberta", "modelo": "Modelo",
         "ladoa": "Lado A", "galpao": "Galpão", "pavilhao": "Pavilhão", "inspecao": "Inspeção", "ceasa": "CEASA",
         "claude": "Claude", "base": "Base", "demo": "Demonstração", "evidencias": "Evidências", "laudo": "Laudo"}
# Qualificadores inteiros que pedem uma redação própria (comparados já sem o hash).
PHRASES = {"humana": "Revisão Humana", "corrigida": "Revisão Corrigida", "sugestoes_modelo": "Sugestões do Modelo"}
CONNECTORS = {"de", "da", "do", "das", "dos", "e", "a", "o", "em"}
WORD_NUMBER = re.compile(r"^(?P<word>[a-z]+?)(?P<number>\d+)$")


def short_hash(value: str) -> str:
    """Código curto mostrado ao lado do nome: 8 primeiros caracteres em maiúsculas."""
    return str(value)[:8].upper()


def _is_hash(token: str) -> bool:
    """Trecho hexadecimal de 8+ caracteres (``ff34e226416c``, ``deadbeef``); palavras reais raramente são só a–f."""
    return len(token) >= 8 and re.fullmatch(r"[0-9a-f]+", token) is not None


def _word(token: str, first: bool) -> str:
    """Uma palavra da pasta na forma escrita: dicionário, versão (v6), palavra+número (revisao002)."""
    if token in WORDS:
        return WORDS[token]
    if re.fullmatch(r"v\d+[a-z]?", token):
        return token
    if re.fullmatch(r"s\d+", token):
        return f"Semente {token[1:]}"
    if token.isdigit():
        return token
    match = WORD_NUMBER.match(token)
    if match and match["word"] in WORDS:
        return f"{WORDS[match['word']]} {match['number']}"
    if token in CONNECTORS and not first:
        return token
    if len(token) <= 2 and token.isalpha():
        return token.upper()
    return token[:1].upper() + token[1:]


def humanize(slug: str) -> str:
    """Nome genérico: ``galpao_norte_b`` → "Galpão Norte B"; hashes viram código curto."""
    tokens = [token for token in re.split(r"[_\-\s]+", str(slug).strip().lower()) if token]
    return " ".join(short_hash(token) if _is_hash(token) else _word(token, index == 0) for index, token in enumerate(tokens))


def _split_review(name: str) -> tuple[str | None, str | None, list[str], str | None]:
    """(local, versão, qualificadores, hash) de ``<local>_v<N>_<qualificadores>_<hash>``."""
    tokens = [token for token in str(name).lower().split("_") if token]
    position = next((index for index, token in enumerate(tokens) if re.fullmatch(r"v\d+[a-z]?", token)), None)
    if position is None or position == 0:
        return None, None, tokens, None
    place_key = "_".join(tokens[:position])
    place = PLACES.get(place_key) or humanize(place_key)
    rest = tokens[position + 1:]
    digest = rest.pop() if rest and _is_hash(rest[-1]) else None
    return place, tokens[position], rest, digest


def review_name(name: str, display_name: str | None = None) -> str:
    """Nome da inspeção a partir da pasta em ``data/reviews``.

    ``ceasa_v8_revisao002_ff34e226416c`` → "CEASA Pavilhão A v8 · Revisão 002 FF34E226";
    ``ceasa_lado_a_v1`` → "CEASA Lado A v1". ``display_name`` salvo no registro
    (nome digitado no envio pela plataforma) tem prioridade.
    """
    if display_name and str(display_name).strip():
        return str(display_name).strip()
    if not name:
        return ""
    place, version, rest, digest = _split_review(name)
    if version is None:
        return humanize(name)
    qualifier = PHRASES.get("_".join(rest)) or " ".join(_word(token, index == 0) for index, token in enumerate(rest))
    text = f"{place} {version}"
    tail = " ".join(part for part in (qualifier, short_hash(digest) if digest else "") if part)
    return f"{text} · {tail}" if tail else text


def run_name(run_id: str, custom: str | None = None) -> str:
    """``pilot_train_20261007T010057Z_f5e5ba8b`` → "Treino Piloto F5E5BA8B" (nome salvo tem prioridade)."""
    if custom and str(custom).strip():
        return str(custom).strip()
    match = RUN_ID.match(str(run_id or ""))
    if not match:
        return humanize(run_id or "")
    stage = STAGE_NAMES.get(match["stage"]) or humanize(match["stage"])
    return f"{stage} {short_hash(match['hash'])}"


def run_started(run_id: str) -> datetime | None:
    """Início (UTC) gravado no id da execução, ou None."""
    match = RUN_ID.match(str(run_id or ""))
    return datetime.strptime(match["stamp"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc) if match else None


def weights_name(path: str, base_model: str | None = None) -> str:
    """Pesos: ``runs/<id>/fit/weights/best.pt`` → "Treino Piloto 56038EB9 · melhor época"; ``models/yolo26l.pt`` → "Modelo Base YOLO26l"."""
    parts = str(path or "").replace("\\", "/").split("/")
    run = next((part for part in parts if RUN_ID.match(part)), None)
    file = parts[-1] if parts else ""
    stem = file.rsplit(".", 1)[0]
    if run:
        kind = {"best": "melhor época", "last": "última época"}.get(stem, file)
        return f"{run_name(run)} · {kind}"
    family = re.fullmatch(r"yolo(?P<version>v?\d+)(?P<size>[nsmlx]?)", stem, re.IGNORECASE)
    if family:
        return f"Modelo Base YOLO{family['version'].upper()}{family['size'].lower()}"
    return humanize(stem) if stem else ""


def _date(stamp: str) -> str | None:
    """``20261007`` → "07/10/2026"."""
    return f"{stamp[6:8]}/{stamp[4:6]}/{stamp[:4]}" if re.fullmatch(r"\d{8}", stamp) else None


def comparison_name(file: str) -> str:
    """``compare_v9_f5e5ba8b_vs_56038eb9_20261007.json`` → "Comparação F5E5BA8B × 56038EB9 · 07/10/2026"."""
    stem = str(file or "").rsplit("/", 1)[-1]
    stem = stem[:-5] if stem.endswith(".json") else stem
    tokens = [token for token in stem.split("_") if token]
    if tokens and tokens[0] in ("compare", "eval"):
        kind = "Comparação" if tokens[0] == "compare" else "Avaliação por Classe"
        tokens = tokens[1:] if tokens[0] == "compare" else tokens[3:] if tokens[1:3] == ["por", "classe"] else tokens[1:]
    else:
        return humanize(stem)
    date = _date(tokens.pop()) if tokens and _date(tokens[-1]) else None
    hashes = [short_hash(token) for token in tokens if _is_hash(token)]
    if hashes:
        subject = " × ".join(hashes)
    else:
        subject = " ".join(_word(token, index == 0) for index, token in enumerate(tokens if "vs" not in tokens else []))
    return " · ".join(part for part in (f"{kind} {subject}".strip(), date) if part)


def dataset_name(name: str) -> str:
    """``ceasa_v8_piloto_s42`` → "Dataset Piloto CEASA Pavilhão A v8 · Semente 42"."""
    place, version, rest, digest = _split_review(name)
    if version is None:
        return f"Dataset {humanize(name)}"
    seed = next((token for token in rest if re.fullmatch(r"s\d+", token)), None)
    words = [token for token in rest if token not in ("piloto", seed)]
    kind = "Dataset Piloto" if "piloto" in rest else "Dataset"
    qualifier = " ".join(_word(token, index == 0) for index, token in enumerate(words))
    parts = [f"{kind} {place} {version}", qualifier, f"Semente {seed[1:]}" if seed else "", short_hash(digest) if digest else ""]
    return " · ".join(part for part in parts if part)


def format_datetime(value: datetime | str | None) -> str:
    """Data legível ``dd/mm/aaaa hh:mm`` (horário de Brasília, UTC−3) para textos gerados no servidor."""
    if not value:
        return ""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone(timedelta(hours=-3))).strftime("%d/%m/%Y %H:%M")


def generated_file_name(file_name: str) -> tuple[str, str | None]:
    """Laudo ou pacote gerado: (nome, data ISO) a partir de ``<AAAAMMDDTHHMMSSZ>_laudo_<inspeção>``."""
    match = re.match(r"^(?P<stamp>\d{8}T\d{6}Z)_(?P<kind>laudo|evidencias)_", str(file_name or ""))
    if not match:
        return humanize(str(file_name).rsplit(".", 1)[0]), None
    when = datetime.strptime(match["stamp"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    kind = "Laudo Técnico" if match["kind"] == "laudo" else "Pacote de Evidências"
    return kind, when.isoformat()
