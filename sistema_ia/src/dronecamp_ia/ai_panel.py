"""Painel Assistente IA da plataforma: configuração da chave/modelo e chat sobre o projeto.

Função no projeto: back-end dos submenus "Configuração da IA" e "Chat" da seção
Assistente IA (o submenu "Uso e limites" usa ``ai_usage.py``). Segue a mesma
estrutura do painel de IA do NetInventory: configurar o provedor (a chave é
testada antes de salvar e nunca volta ao navegador), acompanhar o consumo e
conversar com respostas baseadas nos dados reais do projeto.

O que faz:
- ``ai_status``: provedor, modelo, se há chave (só os 4 últimos caracteres), de
  onde ela vem (variável de ambiente ou ``sistema_ia/.env``) e se a página pode trocá-la.
- ``save_settings``: testa chave+modelo com uma chamada de 1 token e, só se
  passar, grava ``OPENROUTER_API_KEY``/``OPENROUTER_MODEL`` no ``.env`` local.
- ``delete_key``: tira a chave do ``.env``.
- ``project_context``: instantâneo compacto do projeto (inspeções, classes,
  treinos, comparações, notas da IA) que acompanha cada pergunta do chat.
- ``ask``: envia a conversa + o instantâneo ao modelo e devolve a resposta.

O que lê: ``configs/vlm_assist.json`` e ``sistema_ia/.env`` (via
``vlm_assist.load_vlm_settings``), ``registry.json``/``suggestions.json``/
``ai_notes.json`` da inspeção aberta e o que o servidor já reuniu
(``Workspace.describe``, ``operations.collect_operations``, ``class_catalog``).
O que grava: só ``sistema_ia/.env`` (ignorado pelo git) e, por ``ai_usage``, o
``.runtime/ai_usage.jsonl``. A conversa não é gravada em lugar nenhum.

Quem chama: ``platform_server`` (rotas ``/api/ai/status``, ``/api/ai/settings``,
``/api/ai/key/delete``, ``/api/ai/models`` e ``/api/ai/chat``).

Onde mexer:
- Regras do assistente (o que pode ou não responder): ``CHAT_RULES``.
- Dado novo que o chat deve conhecer: ``project_context`` (mantenha compacto;
  ``CONTEXT_LIMIT`` corta o excesso).
- Outro provedor compatível com OpenAI: ``PROVIDERS`` (nome e links) e a URL
  base em ``configs/vlm_assist.json`` ou ``OPENROUTER_BASE_URL`` no ``.env``.

ATENCAO: a chave nunca é devolvida, impressa ou gravada fora do ``.env``. A IA
do chat só informa: não aprova caixa, não define gravidade, diagnóstico ou
recomendação e não declara melhoria de modelo sem comparação registrada.
"""

# ---------------------------------------------------------------------------
# Importações
# ---------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

# ProjectConfig: caminhos do projeto; load_taxonomy: classes da taxonomia em vigor.
from .config import ProjectConfig
# file_hash: confere se sugestões/notas pertencem à versão atual do registro.
from .io import file_hash
# Leitura de configuração, cliente HTTP e lista de modelos gratuitos (mesmo código da CLI).
from .vlm_assist import (ENV_KEY, ENV_MODEL, NOTES_FILE, SETTINGS_FILE, OpenRouterClient,
                         VLMUnavailable, list_free_vision_models, load_vlm_settings)
# usage_logger: cada chamada do chat e do teste da chave entra no consumo local.
from .ai_usage import OPENROUTER_BASE, usage_logger

# ---------------------------------------------------------------------------
# Constantes do módulo
# ---------------------------------------------------------------------------

# Nome, link para criar a chave e painel de consumo de cada provedor reconhecido pela URL base.
# EXTENSAO: provedor compatível novo = entrada aqui com a URL base como chave.
PROVIDERS = {OPENROUTER_BASE: {"name": "OpenRouter", "key_help_url": "https://openrouter.ai/keys",
                               "dashboard_url": "https://openrouter.ai/settings/credits"}}
# Formato aceito para a chave (o mesmo de load_vlm_settings) e para o slug do modelo.
KEY_PATTERN = re.compile(r"[A-Za-z0-9._\-]{16,}")
MODEL_PATTERN = re.compile(r"[A-Za-z0-9._:/\-]{1,200}")
# Limites do chat: mensagens por conversa, caracteres por mensagem e tamanho da resposta.
CHAT_MAX_MESSAGES = 20
CHAT_MAX_CHARS = 4000
CHAT_MAX_TOKENS = 900
# Tamanho máximo do instantâneo do projeto enviado com cada pergunta (caracteres de JSON).
CONTEXT_LIMIT = 24000
# Regras fixas do assistente (mensagem de sistema). Mudou aqui = mude também o texto de ajuda da página.
CHAT_RULES = (
    "Você é o Assistente IA da plataforma DroneCamp Vision (inspeção assistida de telhados industriais por drone). "
    "Responda em português do Brasil, de forma objetiva, usando SOMENTE o instantâneo do projeto em JSON abaixo. "
    "Se a informação pedida não estiver no instantâneo, diga claramente que ela não consta nos dados da plataforma. "
    "Nunca invente números, fotos, classes ou treinos. "
    "Sugestões do modelo e notas da IA são pendências até decisão humana; não as trate como ocorrências confirmadas. "
    "Não atribua gravidade, diagnóstico, recomendação técnica nem aprovação de caixa ou de laudo: diga que dependem de validação técnica. "
    "Não afirme que um modelo melhorou sem uma comparação registrada no instantâneo; receber fotos, treinar ou gerar pesos não comprova melhoria. "
    "Cite os nomes legíveis (campo nome) e não os ids internos."
)


# ---------------------------------------------------------------------------
# Configuração: estado, gravação e remoção da chave
# ---------------------------------------------------------------------------

def _env_path(config: ProjectConfig) -> Path:
    """Caminho do ``.env`` local (``sistema_ia/.env``, ignorado pelo git)."""
    return config.root / ".env"


def _dotenv_lines(path: Path) -> list[str]:
    """Linhas atuais do ``.env`` (lista vazia se ele não existir)."""
    try:
        return path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return []


def _dotenv_has(path: Path, name: str) -> bool:
    """Diz se o ``.env`` define ``name`` com valor não vazio."""
    for line in _dotenv_lines(path):
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and key.strip() == name and value.strip().strip("'\""):
            return True
    return False


def _write_dotenv(path: Path, values: dict[str, str | None]) -> None:
    """Atualiza variáveis no ``.env`` preservando as outras linhas e comentários.

    Parâmetros:
    - ``path``: ``sistema_ia/.env``.
    - ``values``: nome → valor novo; ``None`` remove a variável.

    Retorna: None. Grava num arquivo temporário e troca (não deixa ``.env`` pela
    metade se o PC desligar no meio) e tenta deixar o arquivo legível só pelo dono.

    ATENCAO: o ``.env`` é ignorado pelo git (``.gitignore``); nunca mude o destino
    para uma pasta versionada.
    """
    pending = dict(values)
    lines = []
    for line in _dotenv_lines(path):
        key = line.strip().removeprefix("export ").partition("=")[0].strip()
        if key in pending:
            value = pending.pop(key)
            if value is not None:
                lines.append(f"{key}={value}")
            continue  # variável removida: a linha some
        lines.append(line)
    lines += [f"{key}={value}" for key, value in pending.items() if value is not None]
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass  # Windows ignora as permissões POSIX; o .env já fica fora do git
    os.replace(temporary, path)


def ai_status(config: ProjectConfig, environ: dict | None = None) -> dict:
    """Estado da IA mostrado no submenu Configuração da IA (sem a chave).

    ENTRADA: rota ``GET /api/ai/status``.

    Parâmetros:
    - ``config``: configuração do projeto.
    - ``environ``: variáveis de ambiente (os testes passam um dict); None = ``os.environ``.

    Retorna: dict com ``configured`` (há chave), ``provider``, ``provider_name``,
    ``base_url``, ``model``, ``model_default`` (do ``vlm_assist.json``), ``key_hint``
    (4 últimos caracteres), ``key_source`` ("ambiente", ".env" ou None),
    ``key_locked`` (a chave vem do ambiente do sistema e a página não a troca),
    ``key_help_url``, ``dashboard_url``, ``settings_file``, ``env_file`` e
    ``error`` (configuração ilegível ou chave com formato errado).
    """
    env = os.environ if environ is None else environ
    path = _env_path(config)
    status = {"configured": False, "provider": "openrouter", "provider_name": "OpenRouter", "base_url": OPENROUTER_BASE,
              "model": None, "model_default": None, "key_hint": None, "key_source": None, "key_locked": bool(env.get(ENV_KEY)),
              "key_help_url": PROVIDERS[OPENROUTER_BASE]["key_help_url"], "dashboard_url": PROVIDERS[OPENROUTER_BASE]["dashboard_url"],
              "settings_file": f"configs/{SETTINGS_FILE}", "env_file": ".env", "error": None}
    # 1. Origem da chave: o ambiente do sistema vence o .env (mesma regra de load_vlm_settings).
    if env.get(ENV_KEY):
        status["key_source"] = "ambiente"
    elif _dotenv_has(path, ENV_KEY):
        status["key_source"] = ".env"
    # 2. Configuração efetiva; erro de formato vira mensagem (a chave nunca entra nela).
    try:
        raw = json.loads((config.taxonomy_path.parent / SETTINGS_FILE).read_text(encoding="utf-8"))
        status["model_default"] = raw.get("model")
        settings = load_vlm_settings(config, environ=dict(env))
    except (OSError, ValueError) as error:
        status["error"] = str(error)
        return status
    known = PROVIDERS.get(settings.base_url.rstrip("/"))
    status.update({"configured": bool(settings.api_key), "provider": settings.provider, "base_url": settings.base_url,
                   "provider_name": known["name"] if known else "Servidor compatível com OpenAI", "model": settings.model,
                   "key_hint": settings.api_key[-4:] if settings.api_key else None,
                   "dashboard_url": known["dashboard_url"] if known else None})
    return status


def save_settings(config: ProjectConfig, body: dict, transport=None, environ: dict | None = None) -> dict:
    """Testa a chave e o modelo e, só se a chamada passar, grava no ``.env``.

    ENTRADA: rota ``POST /api/ai/settings`` (botão "Testar e salvar").

    Parâmetros:
    - ``body``: ``{"api_key": "..." (opcional: vazio mantém a atual), "model": "slug"}``.
    - ``transport``/``environ``: substitutos nos testes.

    Retorna: ``ai_status`` depois de gravar, mais ``tested_model`` (o modelo que respondeu).

    Erros (``ValueError``, a página mostra a mensagem):
    - modelo vazio ou com caracteres inválidos; chave com formato inesperado;
    - chave nova quando a atual vem da variável de ambiente do sistema (``key_locked``);
    - nenhuma chave disponível; falha do teste (``VLMUnavailable`` convertida).

    ATENCAO: o teste é uma chamada real de 1 token (entra no consumo local como
    "Teste da chave"); a chave só é gravada depois dele.
    """
    env = dict(os.environ if environ is None else environ)
    model = str(body.get("model") or "").strip()
    api_key = str(body.get("api_key") or "").strip()
    # 1. Validação dos campos (mensagens sem repetir a chave).
    if not MODEL_PATTERN.fullmatch(model):
        raise ValueError("Informe o modelo (ex.: openrouter/free ou um slug terminado em :free).")
    if api_key and not KEY_PATTERN.fullmatch(api_key):
        raise ValueError("A chave tem formato inesperado; cole só a chave, sem espaços ou aspas.")
    if api_key and env.get(ENV_KEY):
        raise ValueError(f"A chave atual vem da variável de ambiente {ENV_KEY} do sistema; troque-a lá ou remova a variável.")
    # 2. Configuração que será testada: a atual com o modelo (e a chave, se veio) trocados.
    settings = replace(load_vlm_settings(config, model=model, environ=env), **({"api_key": api_key} if api_key else {}))
    if not settings.api_key:
        raise ValueError("Cole a chave da API do OpenRouter para ativar a IA.")
    # 3. Teste real (1 token). Falhou = nada é gravado.
    try:
        tested = OpenRouterClient(settings, transport, usage_log=usage_logger(config.root, "connection-test")).ping()
    except VLMUnavailable as error:
        raise ValueError(f"Teste falhou; nada foi salvo. {error}") from error
    # 4. Grava no .env: a chave só se veio uma nova; o modelo sempre (vence o vlm_assist.json).
    _write_dotenv(_env_path(config), {**({ENV_KEY: api_key} if api_key else {}), ENV_MODEL: model})
    return {**ai_status(config, environ), "tested_model": tested.get("model_used")}


def delete_key(config: ProjectConfig, environ: dict | None = None) -> dict:
    """Remove ``OPENROUTER_API_KEY`` do ``.env`` (o modelo escolhido continua).

    ENTRADA: rota ``POST /api/ai/key/delete`` (botão "Remover chave").

    Retorna: ``ai_status`` depois da remoção.
    Erros: ``ValueError`` se a chave vier da variável de ambiente do sistema
    (a página não consegue apagá-la) ou se não houver chave no ``.env``.
    """
    env = os.environ if environ is None else environ
    if env.get(ENV_KEY):
        raise ValueError(f"A chave vem da variável de ambiente {ENV_KEY} do sistema; remova-a nas variáveis do Windows.")
    if not _dotenv_has(_env_path(config), ENV_KEY):
        raise ValueError("Não há chave gravada no .env.")
    _write_dotenv(_env_path(config), {ENV_KEY: None})
    return ai_status(config, environ)


def free_models(config: ProjectConfig, transport=None, environ: dict | None = None) -> dict:
    """Modelos gratuitos com entrada de imagem, para o seletor de modelo.

    ENTRADA: rota ``GET /api/ai/models`` (botão "Listar modelos gratuitos").
    Retorna: ``{"models": [...]}`` como ``list_free_vision_models`` devolve.
    Erros: ``ValueError`` com a mensagem de ``VLMUnavailable`` (sem chave, sem rede).
    """
    try:
        return {"models": list_free_vision_models(load_vlm_settings(config, environ=environ), transport)}
    except VLMUnavailable as error:
        raise ValueError(str(error)) from error


# ---------------------------------------------------------------------------
# Chat: instantâneo do projeto + conversa
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict | None:
    """JSON de um arquivo, ou None se faltar ou estiver ilegível."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _open_inspection(registry_path: Path, labels: dict[int, str], summary: dict | None) -> dict:
    """Resumo da inspeção aberta: fotos, status proposto, caixas e sugestões por classe, notas da IA."""
    registry = _read_json(registry_path) or {}
    images = registry.get("images") or []
    boxes: dict[str, int] = {}
    status: dict[str, int] = {}
    for image in images:
        status[image.get("status") or "sem status"] = status.get(image.get("status") or "sem status", 0) + 1
        for box in image.get("boxes") or []:
            name = labels.get(box.get("class_id"), f"classe {box.get('class_id')}")
            boxes[name] = boxes.get(name, 0) + 1
    result = {"nome": (summary or {}).get("display_name") or registry_path.parent.name, "fotos": len(images),
              "fotos_aprovadas_por_humano": sum(1 for image in images if image.get("human_approved") is True),
              "fotos_por_status_proposto": status, "caixas_registradas_por_classe": boxes,
              "observacao": "Caixas do registro salvo; rascunhos ainda não salvos no navegador não entram."}
    if summary:
        result.update({"fotos_decididas": summary.get("decided"), "sugestoes_pendentes": summary.get("suggestions_pending")})
    # Sugestões do modelo por classe (só se pertencem a esta versão do registro).
    registry_sha = file_hash(registry_path)
    suggestions = _read_json(registry_path.parent / "suggestions.json")
    if suggestions and suggestions.get("registry_sha256") == registry_sha:
        per_class: dict[str, int] = {}
        for values in (suggestions.get("images") or {}).values():
            for value in values if isinstance(values, list) else []:
                name = labels.get(value.get("class_id"), f"classe {value.get('class_id')}")
                per_class[name] = per_class.get(name, 0) + 1
        result["sugestoes_do_modelo"] = {"modelo": suggestions.get("model_label"), "confianca_minima": suggestions.get("conf"),
                                         "total": suggestions.get("total_suggestions"), "por_classe": per_class,
                                         "situacao": "pendentes até decisão humana"}
    # Concordância das notas do VLM com as classes humanas (describe-occurrences).
    notes = _read_json(registry_path.parent / NOTES_FILE)
    if notes and notes.get("registry_sha256") == registry_sha and isinstance(notes.get("agreement"), dict):
        result["notas_da_ia"] = {"modelo_pedido": (notes.get("settings") or {}).get("model_requested"),
                                 "concordancia_com_humanos": notes["agreement"], "situacao": "pendentes de validação técnica"}
    return result


def project_context(config: ProjectConfig, workspace: dict, operations: dict, classes: dict | None,
                    registry_path: Path | None) -> dict:
    """Instantâneo compacto do projeto que acompanha cada pergunta do chat.

    Parâmetros:
    - ``workspace``: ``Workspace.describe(...)`` (lista de inspeções com resumo).
    - ``operations``: ``operations.collect_operations(...)`` (treinos, comparações, pesos).
    - ``classes``: ``class_catalog.list_classes(...)`` ou None se o catálogo falhou.
    - ``registry_path``: ``registry.json`` da inspeção aberta na página, ou None.

    Retorna: dict só com nomes legíveis e contagens (sem caminhos de foto, sem chave).

    EXTENSAO: dado novo para o chat = chave nova aqui; mantenha o total abaixo de
    ``CONTEXT_LIMIT`` (o que passar é cortado em ``ask``).
    """
    catalog = (classes or {}).get("classes") or []
    labels = {entry.get("id"): entry.get("display_name") or entry.get("name") for entry in catalog}
    reviews = workspace.get("reviews") or []
    current = next((review for review in reviews if review.get("current")), None)
    context = {
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inspecoes": [{"nome": review.get("display_name") or review.get("name"), "fotos": review.get("images"),
                       "fotos_decididas": review.get("decided"), "sugestoes_pendentes": review.get("suggestions_pending"),
                       "edificacoes": len(review.get("buildings") or []), "atualizada_em": review.get("updated_at"),
                       "aberta_na_pagina": bool(review.get("current"))} for review in reviews[:30]],
        "classes": [{"id": entry.get("id"), "nome": entry.get("display_name") or entry.get("name"),
                     "ativa": entry.get("active", True), "no_detector": entry.get("in_detector"),
                     "criterio": (entry.get("criterion") or "")[:160]} for entry in catalog],
        "dispositivo_de_treino": operations.get("device"), "modelo_base": operations.get("base_model"),
        "treinos": [{"nome": run.get("display_name") or run.get("id"), "detalhe": run.get("display_detail"),
                     "estado": run.get("state"), "inicio": run.get("started_at"),
                     "avaliacao_em_edificacao_independente": run.get("independent_evaluation")}
                    for run in (operations.get("runs") or [])[:15]],
        "comparacoes": [{"arquivo": comparison.get("file"), "conf": comparison.get("conf"),
                         "modelos": [{"treino": model.get("run_id"), "teste": (model.get("splits") or {}).get("test")}
                                     for model in comparison.get("models") or []]}
                        for comparison in (operations.get("comparisons") or [])[:8]],
    }
    if registry_path is not None:
        context["inspecao_aberta"] = _open_inspection(registry_path, labels, current)
    return context


def _validate_messages(messages) -> list[dict]:
    """Confere a conversa vinda da página: papéis, tamanhos e última mensagem do usuário."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("Envie a conversa como uma lista de mensagens.")
    if len(messages) > CHAT_MAX_MESSAGES:
        raise ValueError(f"Conversa longa demais (máximo {CHAT_MAX_MESSAGES} mensagens); limpe a conversa e pergunte de novo.")
    clean = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
            raise ValueError("Mensagem com papel inválido.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip() or len(content) > CHAT_MAX_CHARS:
            raise ValueError(f"Cada mensagem precisa ter texto com até {CHAT_MAX_CHARS} caracteres.")
        clean.append({"role": message["role"], "content": content})
    if clean[-1]["role"] != "user":
        raise ValueError("A última mensagem precisa ser uma pergunta.")
    return clean


def ask(config: ProjectConfig, messages, context: dict, transport=None, environ: dict | None = None) -> dict:
    """Responde a pergunta do chat com base no instantâneo do projeto.

    ENTRADA: rota ``POST /api/ai/chat``.

    Parâmetros:
    - ``messages``: conversa ``[{"role": "user"|"assistant", "content": "..."}]``
      (fica só na memória da página; aqui também não é gravada).
    - ``context``: ``project_context(...)``.

    Retorna: ``{"answer": texto, "model_used": modelo que respondeu}``.

    Erros: ``ValueError`` para conversa inválida, IA sem chave ou falha do provedor
    (mensagem de ``VLMUnavailable``).
    """
    conversation = _validate_messages(messages)
    settings = load_vlm_settings(config, environ=environ)
    if not settings.api_key:
        raise ValueError("IA desativada: cadastre a chave em Assistente IA → Configuração da IA.")
    # Resposta mais longa e um pouco mais livre que a descrição de recortes (que usa temperatura 0).
    settings = replace(settings, max_output_tokens=CHAT_MAX_TOKENS, temperature=0.2)
    snapshot = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    if len(snapshot) > CONTEXT_LIMIT:
        snapshot = snapshot[:CONTEXT_LIMIT] + "…(instantâneo cortado por tamanho)"
    system = {"role": "system", "content": f"{CHAT_RULES}\n\nINSTANTÂNEO DO PROJETO (JSON):\n{snapshot}"}
    try:
        client = OpenRouterClient(settings, transport, usage_log=usage_logger(config.root, "chat"))
        answer, meta = client.chat([system, *conversation])
    except VLMUnavailable as error:
        raise ValueError(str(error)) from error
    if not answer.strip():
        raise ValueError("O modelo não devolveu texto; tente de novo ou escolha outro modelo em Configuração da IA.")
    return {"answer": answer.strip(), "model_used": meta.get("model_used")}
