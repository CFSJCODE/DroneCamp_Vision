"""Consumo da IA generativa (VLM do OpenRouter): registro local e consulta de limites do provedor.

Função no projeto: alimenta o submenu "Uso e limites" do painel Assistente IA da
plataforma. Cada chamada ao modelo (descrição de ocorrências, chat do assistente,
teste da chave) grava UMA linha de metadados em ``.runtime/ai_usage.jsonl``; a
página agrega essas linhas por período. Separadamente, o saldo e o teto da chave
são consultados no endpoint oficial do OpenRouter, sem gerar inferência.

O que faz:
- ``completion_usage``: tokens e custo informados pelo provedor; ausente fica None
  (nunca vira zero por inferência).
- ``record_usage``: acrescenta uma linha no ``ai_usage.jsonl``; falha de disco só
  avisa no terminal e nunca derruba a resposta da IA.
- ``local_usage``: totais por período (24 h, 7 dias, 30 dias, tudo) e a tabela
  por recurso/modelo.
- ``provider_usage``: ``GET /key`` e ``GET /credits`` do OpenRouter (teto da chave,
  consumo acumulado, saldo da conta); outros provedores devolvem "sem consulta".

O que grava: só ``sistema_ia/.runtime/ai_usage.jsonl`` (ignorado pelo git). Cada
linha tem data, recurso, provedor, modelo pedido e o que respondeu, resultado,
duração, código HTTP da falha, tokens e custo. NUNCA grava pergunta, resposta,
recorte, dados da inspeção ou a chave.

Quem chama:
- ``vlm_assist.OpenRouterClient`` (cada chamada de ``chat``/``ping``) via o
  parâmetro ``usage_log`` montado por ``usage_logger``.
- ``platform_server`` (rotas ``GET /api/ai/usage`` e ``GET /api/ai/provider-usage``).

Onde mexer:
- Recurso novo que chama o modelo: passe ``usage_logger(root, "<recurso>")`` ao
  criar o ``OpenRouterClient`` e acrescente o nome legível em ``FEATURES`` (Python)
  e em ``AI_FEATURES`` no platform.js.
- Período novo do filtro: ``PERIODS`` aqui e o ``<select id="aiUsagePeriod">`` do index.html.
- Campo novo por chamada: ``record_usage`` (dict ``entry``) e a agregação em ``local_usage``.

ATENCAO: o arquivo é só de metadados. Não acrescente texto do usuário, resposta do
modelo, nome de foto ou a chave em nenhuma linha.
"""

# ---------------------------------------------------------------------------
# Importações
# ---------------------------------------------------------------------------

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Constantes do módulo
# ---------------------------------------------------------------------------

# Arquivo de registro, relativo a sistema_ia/ (config.root). Uma linha JSON por chamada.
# ATENCAO: o .gitignore da raiz ignora este caminho; se mudar o nome, mude lá também.
USAGE_FILE = Path(".runtime") / "ai_usage.jsonl"
# Filtros de período da página: chave → (rótulo, dias; 0 = sem limite).
# EXTENSAO: período novo = chave aqui + <option> no <select id="aiUsagePeriod"> do index.html.
PERIODS = {"24h": ("Últimas 24 horas", 1), "7d": ("Últimos 7 dias", 7), "30d": ("Últimos 30 dias", 30), "all": ("Todo o período", 0)}
# Nome legível de cada recurso que chama o modelo (o JS tem a mesma tabela em AI_FEATURES).
# EXTENSAO: recurso novo = chave aqui e em AI_FEATURES no platform.js.
FEATURES = {"chat": "Chat do assistente", "describe": "Descrição de ocorrências", "connection-test": "Teste da chave"}
# Únicos resultados gravados: resposta recebida ou falha (HTTP, rede, resposta vazia).
STATUSES = ("success", "error")
# Endereço oficial do OpenRouter: só para ele a chave é enviada nas consultas de saldo.
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Leitura segura dos números informados pelo provedor
# ---------------------------------------------------------------------------

def _finite(value) -> float | None:
    """Número finito (int/float, não bool) ou None.

    Parâmetros: ``value`` (qualquer valor vindo do JSON do provedor).
    Retorna: o próprio número como float, ou None para texto, bool, NaN ou infinito.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def _tokens(value) -> int | None:
    """Contagem de tokens válida (inteiro >= 0) ou None.

    Parâmetros: ``value`` (campo ``prompt_tokens``/``completion_tokens``/``total_tokens``).
    Retorna: int, ou None se faltar, for negativo, fracionário ou não numérico.
    """
    number = _finite(value)
    return int(number) if number is not None and number >= 0 and number == int(number) else None


def completion_usage(usage) -> dict:
    """Tokens e custo de uma resposta ``chat/completions``; o que faltar fica None.

    Parâmetros:
    - ``usage``: o objeto ``usage`` da resposta (dict) ou None.

    Retorna: ``{"input_tokens", "output_tokens", "total_tokens", "cost_usd"}``.
    ``cost_usd`` vem do campo ``cost`` que o OpenRouter devolve (USD); zero só
    aparece quando o provedor informou zero.

    Chamada por: ``vlm_assist.OpenRouterClient._measured``.
    """
    data = usage if isinstance(usage, dict) else {}
    return {"input_tokens": _tokens(data.get("prompt_tokens")), "output_tokens": _tokens(data.get("completion_tokens")),
            "total_tokens": _tokens(data.get("total_tokens")), "cost_usd": _finite(data.get("cost"))}


# ---------------------------------------------------------------------------
# Registro local (uma linha JSON por chamada)
# ---------------------------------------------------------------------------

def record_usage(root: Path, entry: dict, now: datetime | None = None) -> None:
    """Acrescenta uma linha de metadados em ``.runtime/ai_usage.jsonl``.

    Parâmetros:
    - ``root``: pasta ``sistema_ia`` (``config.root``).
    - ``entry``: dict com ``feature``, ``provider``, ``model``, ``model_used``,
      ``status`` ("success"/"error"), ``duration_ms``, ``http_status`` e os campos
      de ``completion_usage``. Chaves fora dessa lista são descartadas.
    - ``now``: data da chamada (os testes fixam); None = agora em UTC.

    Retorna: None. Erro de disco (pasta somente leitura, disco cheio) só imprime um
    aviso: a telemetria nunca impede a resposta da IA.

    Chamada por: o ``usage_log`` criado por ``usage_logger``.
    """
    allowed = ("feature", "provider", "model", "model_used", "status", "duration_ms", "http_status",
               "input_tokens", "output_tokens", "total_tokens", "cost_usd")
    # ATENCAO: lista fechada de chaves; nada de texto livre (pergunta, resposta, chave) entra aqui.
    line = {"at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")}
    line.update({key: entry.get(key) for key in allowed})
    try:
        path = Path(root) / USAGE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError:
        print("Aviso: não foi possível registrar o consumo da IA em .runtime/ai_usage.jsonl.", file=sys.stderr)


def usage_logger(root: Path, feature: str):
    """Cria a função ``usage_log`` entregue ao ``OpenRouterClient``.

    Parâmetros:
    - ``root``: pasta ``sistema_ia``.
    - ``feature``: recurso que está chamando o modelo (chave de ``FEATURES``).

    Retorna: função ``log(entry: dict) -> None`` que completa ``feature`` e grava.

    Chamada por: ``vlm_assist.describe_occurrences`` e ``ai_panel`` (chat e teste da chave).
    """
    def log(entry: dict) -> None:
        record_usage(root, {**entry, "feature": feature})
    return log


def read_usage(root: Path) -> list[dict]:
    """Linhas válidas do ``ai_usage.jsonl`` (as corrompidas são ignoradas).

    Parâmetros: ``root`` (pasta ``sistema_ia``).
    Retorna: lista de dicts na ordem do arquivo; lista vazia se o arquivo não existir.
    """
    path = Path(root) / USAGE_FILE
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            value = json.loads(line)
        except ValueError:
            continue  # linha cortada por queda de energia: não derruba o painel
        if isinstance(value, dict) and value.get("status") in STATUSES:
            rows.append(value)
    return rows


def _parse_at(value) -> datetime | None:
    """Data ISO da linha → datetime com fuso (ou None se ilegível)."""
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _sum(rows: list[dict], key: str) -> tuple[float | None, int]:
    """Soma de um campo numérico e quantas linhas o informaram.

    Retorna: ``(soma ou None se nenhuma linha informou, quantidade de linhas com valor)``.
    Zero informado conta como informado (soma 0, não None).
    """
    values = [row[key] for row in rows if _finite(row.get(key)) is not None]
    return (sum(values) if values else None), len(values)


def local_usage(root: Path, period: str = "30d", now: datetime | None = None) -> dict:
    """Totais de consumo desta instalação no período pedido.

    ENTRADA: rota ``GET /api/ai/usage?period=...`` (platform_server).

    Parâmetros:
    - ``root``: pasta ``sistema_ia``.
    - ``period``: chave de ``PERIODS`` ("24h", "7d", "30d", "all").
    - ``now``: instante de referência (os testes fixam); None = agora.

    Retorna: dict com ``period``, ``period_label``, ``requests``, ``successful``,
    ``errors``, ``input_tokens``, ``output_tokens``, ``total_tokens``,
    ``token_reports`` (chamadas que informaram tokens), ``cost_usd``,
    ``cost_reports``, ``average_duration_ms``, ``tracking_since`` (primeira linha
    do arquivo inteiro) e ``features`` (lista por recurso/provedor/modelo com
    ``events``, ``errors``, ``total_tokens`` e ``cost_usd``).
    Campos sem nenhuma informação ficam None ("Não informado" na página).

    Erros: ``ValueError`` para período desconhecido.
    """
    if period not in PERIODS:
        raise ValueError(f"Período desconhecido: {period}.")
    label, days = PERIODS[period]
    now = now or datetime.now(timezone.utc)
    rows = read_usage(root)
    first = min((moment for moment in (_parse_at(row.get("at")) for row in rows) if moment), default=None)
    # 1. Recorta o período (janela móvel a partir de agora).
    if days:
        start = now - timedelta(days=days)
        rows = [row for row in rows if (_parse_at(row.get("at")) or now) >= start]
    # 2. Totais gerais.
    total_tokens, token_reports = _sum(rows, "total_tokens")
    cost, cost_reports = _sum(rows, "cost_usd")
    durations = [row["duration_ms"] for row in rows if _finite(row.get("duration_ms")) is not None]
    # 3. Tabela por recurso, provedor e modelo (o modelo que respondeu, quando o provedor informou).
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row.get("feature") or "?", row.get("provider") or "?", row.get("model_used") or row.get("model") or "?")
        groups.setdefault(key, []).append(row)
    features = []
    for (feature, provider, model), items in sorted(groups.items(), key=lambda pair: -len(pair[1])):
        tokens, reports = _sum(items, "total_tokens")
        spent, _ = _sum(items, "cost_usd")
        features.append({"feature": feature, "feature_label": FEATURES.get(feature, feature), "provider": provider, "model": model,
                         "events": len(items), "errors": sum(1 for item in items if item.get("status") == "error"),
                         "token_reports": reports, "total_tokens": tokens, "cost_usd": spent})
    return {"period": period, "period_label": label, "requests": len(rows),
            "successful": sum(1 for row in rows if row.get("status") == "success"),
            "errors": sum(1 for row in rows if row.get("status") == "error"),
            "input_tokens": _sum(rows, "input_tokens")[0], "output_tokens": _sum(rows, "output_tokens")[0],
            "total_tokens": total_tokens, "token_reports": token_reports, "cost_usd": cost, "cost_reports": cost_reports,
            "average_duration_ms": (sum(durations) / len(durations)) if durations else None,
            "tracking_since": first.isoformat(timespec="seconds") if first else None, "features": features}


# ---------------------------------------------------------------------------
# Saldo e limites no provedor (sem inferência)
# ---------------------------------------------------------------------------

def _empty_provider(state: str, message: str) -> dict:
    """Resposta de ``provider_usage`` com todos os números ausentes (None)."""
    return {"state": state, "message": message, "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "account_balance": None, "key_limit": None, "key_remaining": None, "key_usage": None,
            "daily_usage": None, "monthly_usage": None, "reset": None, "is_free_tier": None, "label": None}


def parse_openrouter(key: dict | None, credits: dict | None) -> dict:
    """Converte as respostas de ``/key`` e ``/credits`` nos campos do painel.

    Parâmetros:
    - ``key``: ``data`` de ``GET /api/v1/key`` (ou None se a consulta falhou).
    - ``credits``: ``data`` de ``GET /api/v1/credits`` (ou None; exige chave de gerenciamento).

    Retorna: dict com ``account_balance`` (créditos − consumo da conta), ``key_limit``,
    ``key_remaining``, ``key_usage``, ``daily_usage``, ``monthly_usage``, ``reset``
    (renovação do teto da chave), ``is_free_tier`` e ``label`` (rótulo da chave,
    que o OpenRouter já devolve mascarado). Ausente = None.
    """
    data = key if isinstance(key, dict) else {}
    balance = credits if isinstance(credits, dict) else {}
    total, used = _finite(balance.get("total_credits")), _finite(balance.get("total_usage"))
    free = data.get("is_free_tier")
    return {"account_balance": (total - used) if total is not None and used is not None else None,
            "key_limit": _finite(data.get("limit")), "key_remaining": _finite(data.get("limit_remaining")),
            "key_usage": _finite(data.get("usage")), "daily_usage": _finite(data.get("usage_daily")),
            "monthly_usage": _finite(data.get("usage_monthly")),
            "reset": data.get("limit_reset") if isinstance(data.get("limit_reset"), str) else None,
            "is_free_tier": free if isinstance(free, bool) else None,
            "label": data.get("label") if isinstance(data.get("label"), str) else None}


def provider_usage(settings, transport=None) -> dict:
    """Teto, consumo e saldo da chave no OpenRouter, sem chamar nenhum modelo.

    ENTRADA: rota ``GET /api/ai/provider-usage`` (platform_server).

    Parâmetros:
    - ``settings``: ``vlm_assist.VLMSettings`` em vigor.
    - ``transport``: transporte HTTP falso (testes); None usa o real.

    Retorna: dict de ``_empty_provider`` preenchido por ``parse_openrouter`` e
    ``state``: "available" (consultado), "unconfigured" (sem chave),
    "unsupported" (URL que não é a do OpenRouter) ou "error" (as duas consultas
    falharam). ``message`` explica o estado em português.

    ATENCAO: a chave só vai para ``OPENROUTER_BASE``; com outra URL (servidor
    compatível) nada é enviado, para a chave não vazar para um endereço trocado.
    """
    from .vlm_assist import OpenRouterClient, VLMUnavailable
    if not settings.api_key:
        return _empty_provider("unconfigured", "Cadastre a chave em Configuração da IA para consultar os limites do provedor.")
    if settings.base_url.rstrip("/") != OPENROUTER_BASE:
        return _empty_provider("unsupported", "Este endereço não é o do OpenRouter; consulte saldo e cotas no painel do próprio provedor.")
    client = OpenRouterClient(settings, transport)
    results = {}
    # Cada consulta isolada: /credits pode exigir chave de gerenciamento e falhar sozinha.
    for name in ("key", "credits"):
        try:
            value = client._request(f"/{name}", None)
            results[name] = value.get("data") if isinstance(value, dict) and isinstance(value.get("data"), dict) else None
        except VLMUnavailable:
            results[name] = None
    if results["key"] is None and results["credits"] is None:
        return _empty_provider("error", "Não foi possível consultar o OpenRouter. Verifique a conexão e a chave e tente atualizar.")
    message = ("Dados consultados no OpenRouter. O saldo da conta e o teto da chave são independentes."
               if results["credits"] is not None else
               "Teto e consumo da chave consultados no OpenRouter. O saldo da conta exige chave de gerenciamento; veja-o no painel do provedor.")
    return {**_empty_provider("available", message), **parse_openrouter(results["key"], results["credits"])}
