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
  ``urllib`` (sem dependência nova); ``transport`` injetável nos testes; cada
  chamada pode ser registrada em ``.runtime/ai_usage.jsonl`` (``usage_log``, ver ai_usage.py).
- ``describe_occurrences``: recorta cada caixa (humana e, opcionalmente, sugerida),
  pergunta ao VLM e grava ``ai_notes.json`` ao lado do registro, retomável.
- ``agreement_report``: concordância da classe sugerida pelo VLM com a humana.
- ``draft_report_support``: ``laudo_apoio.json``/``.md`` com as ocorrências
  revisadas, descrições pendentes e campos técnicos em branco.
- ``list_free_vision_models``: modelos gratuitos com entrada de imagem no OpenRouter.

Quando mexer: prompt (``PROMPT_VERSION`` muda e invalida notas antigas),
tamanho dos recortes ou campos gravados. Sem chave ou sem rede o comando falha
com explicação; nunca inventa descrições.

Conteúdo do arquivo (na ordem em que aparece):
- Constantes do módulo (nomes de arquivos, variáveis de ambiente, versão do prompt).
- ``VLMUnavailable``: exceção de serviço/chave/rede indisponível.
- Configuração: ``VLMSettings`` (+ ``describe``), ``_read_dotenv``, ``load_vlm_settings``.
- Cliente HTTP: ``_urllib_transport``, ``OpenRouterClient`` (``chat``, ``models``),
  ``list_free_vision_models``.
- Recortes e prompt: ``crop_box``, ``taxonomy_prompt``, ``build_messages``,
  ``prompt_digest``, ``resume_digest``, ``parse_answer``.
- Descrição das ocorrências: ``_boxes_to_describe``, ``_slug``, ``_load_suggestions``,
  ``describe_occurrences``, ``agreement_report``.
- Apoio ao laudo: ``draft_report_support``, ``_support_markdown``.

Quem chama:
- CLI (cli.py): ``describe-occurrences`` → ``describe_occurrences``;
  ``report-support`` → ``draft_report_support``; ``list-free-models`` →
  ``load_vlm_settings`` + ``list_free_vision_models``.
- ``laudo_abnt.py`` e ``exports.py`` LEEM o ``ai_notes.json`` gravado aqui
  (``load_ai_notes``/``match_ai_note``/``ai_summary``): as chaves ``images``,
  ``registry_sha256``, ``settings.model_requested``, ``agreement`` e, por entrada,
  ``source``, ``description``, ``class_id`` e ``bbox_xyxy`` são contrato com eles.

O que lê do disco:
- ``configs/vlm_assist.json`` (parâmetros), ``configs/taxonomy.json`` (classes),
  ``sistema_ia/.env`` (só a chave/modelo/URL), ``registry.json`` da revisão,
  ``suggestions.json`` (opcional) e as fotos originais apontadas no registro.
O que grava:
- ``<pasta da revisão>/ai_notes.json`` (``describe_occurrences``);
- ``laudo_apoio.json`` e ``laudo_apoio.md`` (``draft_report_support``).
Nada aqui altera o ``registry.json``.

Onde mexer:
- Mudar o texto do prompt ou as chaves do JSON pedido ao VLM: ``build_messages``
  (e suba ``PROMPT_VERSION``); leitura da resposta em ``parse_answer``.
- Novo parâmetro de configuração (ex.: ``top_p``): campo em ``VLMSettings``, leitura em
  ``load_vlm_settings``, chave em ``configs/vlm_assist.json`` e, se ele mudar a
  resposta, inclua-o em ``resume_digest`` e no ``payload`` de ``OpenRouterClient.chat``.
- Outro provedor compatível com OpenAI: ``OPENROUTER_BASE_URL`` / ``base_url`` no
  JSON; cabeçalhos em ``OpenRouterClient._headers``.
- Novo campo em cada nota do ``ai_notes.json``: dicionário ``entry`` em
  ``describe_occurrences`` (e na cópia do cache, tupla de chaves reaproveitadas).
- Novo campo por ocorrência no apoio ao laudo: dicionário em ``draft_report_support``
  e a linha correspondente em ``_support_markdown``.
- Novo tipo de origem de caixa (além de humana/proposta/sugestao): ``_boxes_to_describe``
  e o filtro de ``agreement_report``.

ATENCAO: a IA nunca aprova caixa nem define gravidade, diagnóstico ou recomendação.
Esses campos são sempre ``None`` e ``status``/``validation`` ficam pendentes.
"""

# ---------------------------------------------------------------------------
# Importações
# ---------------------------------------------------------------------------

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

# Pillow: abre a foto original e gera o recorte JPEG enviado ao VLM.
from PIL import Image

# ProjectConfig: caminhos do projeto (root, taxonomy_path); load_taxonomy: lê configs/taxonomy.json.
from .config import ProjectConfig, load_taxonomy
# file_hash: SHA-256 de arquivo; resolve_local_path: caminho do registro → caminho local;
# write_json: grava JSON (formato padrão do projeto).
from .io import file_hash, resolve_local_path, write_json
# completion_usage/usage_logger: metadados de consumo de cada chamada ao modelo
# (submenu Uso e limites do Assistente IA da plataforma).
from .ai_usage import completion_usage, usage_logger

# ---------------------------------------------------------------------------
# Constantes do módulo (nomes de arquivo, variáveis de ambiente, textos fixos)
# ---------------------------------------------------------------------------

# Nome do arquivo de parâmetros, procurado na mesma pasta da taxonomia (configs/).
SETTINGS_FILE = "vlm_assist.json"
# Variável de ambiente (ou linha do .env) com a chave do OpenRouter.
# ATENCAO: só o NOME da variável fica no código; o valor nunca é gravado nem impresso.
ENV_KEY = "OPENROUTER_API_KEY"
# Variável opcional que troca o modelo do JSON (perde para o --model da CLI).
ENV_MODEL = "OPENROUTER_MODEL"
# Variável opcional que troca a URL base da API (outro provedor compatível com OpenAI).
ENV_BASE_URL = "OPENROUTER_BASE_URL"
# Versão do prompt. Entra em prompt_digest: subir o número invalida as notas antigas
# e força a redescrição de todos os recortes na próxima execução.
# ATENCAO: suba este número sempre que mudar o texto de build_messages.
PROMPT_VERSION = 1
# Nome do arquivo de notas gravado ao lado do registry.json.
# ATENCAO: laudo_abnt.AI_NOTES_FILE procura este mesmo nome; mude os dois juntos.
NOTES_FILE = "ai_notes.json"
# Texto fixo dos campos técnicos ainda não validados por um responsável humano.
PENDING = "pendente de validação técnica"
# Aviso gravado em ai_notes.json e no laudo_apoio (json e md).
WARNING = ("Descrições geradas por IA a partir de recortes: apoiam a revisão, não a substituem. "
           "Classe, gravidade, diagnóstico e recomendações dependem de decisão humana.")

# Respostas aceitas para a classe sugerida além dos slugs da taxonomia.
# "nenhuma" = sem não conformidade visível; "incerto" = o VLM não soube (ou respondeu algo inválido).
# EXTENSAO: nova resposta especial do VLM: acrescente aqui, cite-a no prompt de build_messages
# e decida como agreement_report a conta (hoje entra em "none_or_uncertain").
NO_CLASS = ("nenhuma", "incerto")
# Valores aceitos para "legibilidade"; qualquer outro vira None em parse_answer.
# EXTENSAO: novo grau de legibilidade: acrescente aqui e no texto do prompt (build_messages).
LEGIBILITY = ("boa", "limitada", "insuficiente")


class VLMUnavailable(RuntimeError):
    """Serviço, chave ou rede indisponíveis: o comando para sem gravar descrições.

    Lançada por ``OpenRouterClient`` (chave ausente/recusada, HTTP 429, erro HTTP,
    resposta não JSON ou sem conteúdo) e capturada em ``describe_occurrences``:
    na primeira chamada ela interrompe tudo; depois disso vira um erro por caixa.
    """


# ---------------------------------------------------------------------------
# Configuração e chave (a chave nunca é gravada nem impressa).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VLMSettings:
    """Parâmetros efetivos de uma execução (JSON + ambiente + argumento ``--model``).

    Criado só por ``load_vlm_settings``. Imutável (``frozen``) para que nenhum
    trecho altere o modelo ou a chave no meio de uma execução.

    Campos:
    - ``provider``: rótulo do provedor (``"openrouter"``), só informativo.
    - ``base_url``: URL base da API sem barra final (ex.: ``https://openrouter.ai/api/v1``).
    - ``model``: slug pedido (``openrouter/free`` ou um slug fixo terminando em ``:free``).
    - ``api_key``: chave do OpenRouter ou None. ATENCAO: nunca a grave nem a imprima;
      ``describe`` registra só se ela existe.
    - ``timeout_seconds``: tempo máximo por requisição HTTP, em segundos.
    - ``pause_seconds``: espera entre chamadas consecutivas (respeita o limite dos modelos gratuitos).
    - ``max_output_tokens``: limite de tokens da resposta (``max_tokens`` do payload).
    - ``temperature``: aleatoriedade da resposta (0.0 = mais determinística).
    - ``crop_margin``: margem de contexto em torno da caixa, em fração do lado (0.3 = 30 % de cada lado).
    - ``max_crop_side``: maior lado do JPEG enviado, em pixels (recortes maiores são reduzidos).
    - ``min_crop_side``: menor lado do recorte antes da redução, em pixels (caixas pequenas ganham contexto).
    - ``max_boxes_per_image``: quantas caixas por foto são descritas, no máximo.
    - ``app_title``: nome do app enviado no cabeçalho ``X-Title``.
    - ``settings_sha256``: SHA-256 do ``vlm_assist.json`` lido (rastreabilidade).

    EXTENSAO: novo parâmetro: campo aqui, leitura em ``load_vlm_settings`` e, se
    deve aparecer nas saídas, em ``describe``.
    """

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
        """O que fica registrado nas saídas: sem chave, só se havia uma.

        Retorna: dict gravado em ``ai_notes.json["settings"]`` com ``provider``,
        ``base_url``, ``model_requested``, ``api_key_present`` (bool),
        ``settings_sha256``, ``crop_margin``, ``max_crop_side`` e ``prompt_version``.

        ATENCAO: nunca acrescente ``api_key`` aqui; este dict vai para arquivos.
        ``model_requested`` é lido por ``laudo_abnt.ai_summary`` e por ``draft_report_support``.
        EXTENSAO: novo campo de rastreabilidade nas saídas: acrescente a chave neste dict.
        """
        return {"provider": self.provider, "base_url": self.base_url, "model_requested": self.model,
                "api_key_present": bool(self.api_key), "settings_sha256": self.settings_sha256,
                "crop_margin": self.crop_margin, "max_crop_side": self.max_crop_side, "prompt_version": PROMPT_VERSION}


def _read_dotenv(path: Path) -> dict[str, str]:
    """``CHAVE=valor`` por linha; só as variáveis deste módulo são lidas.

    Parâmetros:
    - ``path``: caminho do ``.env`` (normalmente ``sistema_ia/.env``, fora do git).

    Retorna: dict só com as chaves ``OPENROUTER_API_KEY``, ``OPENROUTER_MODEL`` e
    ``OPENROUTER_BASE_URL`` encontradas; vazio se o arquivo não existir ou não puder ser lido.

    Aceita linhas ``export CHAVE=valor`` e valores entre aspas simples ou duplas.
    Não trata comentários no fim da linha (``CHAVE=valor # nota`` mantém ``# nota``).

    Chamada por: ``load_vlm_settings``.
    """
    values = {}
    # Leitura tolerante: utf-8-sig remove o BOM que alguns editores do Windows gravam;
    # arquivo ausente ou ilegível não é erro (a chave pode vir do ambiente).
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return values
    # Uma variável por linha; linhas vazias, comentários e linhas sem "=" são ignoradas.
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        # partition corta só no primeiro "=": valores com "=" no meio continuam inteiros.
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        # Só as variáveis deste módulo; o resto do .env não é carregado.
        # EXTENSAO: nova variável lida do .env: acrescente o nome nesta tupla (e uma constante ENV_*).
        if key in (ENV_KEY, ENV_MODEL, ENV_BASE_URL):
            # Remove espaços e as aspas em volta do valor ('valor' ou "valor").
            values[key] = value.strip().strip("'\"")
    return values


# ATENCAO: a chave do OpenRouter vem só de OPENROUTER_API_KEY (ambiente) ou de sistema_ia/.env.
# Nunca a escreva em código, configs, docs ou commits (o repositório é público).
def load_vlm_settings(config: ProjectConfig, model: str | None = None, environ: dict | None = None) -> VLMSettings:
    """Lê configs/vlm_assist.json; a chave vem do ambiente ou de sistema_ia/.env.

    ENTRADA: comando ``list-free-models`` (cli.py) e, por dentro, ``describe_occurrences``.

    Parâmetros:
    - ``config``: ``ProjectConfig`` do projeto (usa ``taxonomy_path`` para achar
      ``configs/`` e ``root`` para achar o ``.env``).
    - ``model``: slug do modelo vindo do ``--model`` da CLI; None usa ambiente ou JSON.
    - ``environ``: dict de variáveis de ambiente (os testes passam um dict próprio);
      None usa ``os.environ``.

    Precedência: modelo = ``model`` > ``OPENROUTER_MODEL`` > JSON > ``openrouter/free``;
    URL = ``OPENROUTER_BASE_URL`` > JSON > padrão do OpenRouter; variáveis do ambiente
    vencem as do ``.env``.

    Retorna: ``VLMSettings`` preenchido (``api_key`` pode ser None; quem precisa da
    chave é ``OpenRouterClient``, que falha cedo sem ela).

    Erros:
    - ``ValueError("vlm_assist.json precisa de schema_version: 1.")`` se a versão do JSON for outra;
    - ``ValueError`` se a chave tiver formato inesperado (espaços, aspas, menos de 16 caracteres);
    - ``OSError``/``json.JSONDecodeError`` se o JSON não existir ou estiver malformado.
    """
    # 1. Lê o JSON de parâmetros (configs/vlm_assist.json, ao lado da taxonomia).
    path = config.taxonomy_path.parent / SETTINGS_FILE
    raw = json.loads(path.read_text(encoding="utf-8"))
    # ATENCAO: só a versão 1 do formato é entendida; mudar o formato exige subir a versão aqui e no JSON.
    if raw.get("schema_version") != 1:
        raise ValueError(f"{SETTINGS_FILE} precisa de schema_version: 1.")
    # 2. Junta ambiente e .env: setdefault faz o ambiente vencer o .env.
    env = dict(os.environ if environ is None else environ)
    for key, value in _read_dotenv(config.root / ".env").items():
        env.setdefault(key, value)
    # 3. Chave: string vazia vira None. A regex aceita só letras, dígitos, ".", "_" e "-",
    #    com 16 caracteres ou mais; pega cópias com espaço, aspas ou quebra de linha.
    # ATENCAO: a mensagem de erro não inclui a chave; mantenha assim.
    api_key = (env.get(ENV_KEY) or "").strip() or None
    if api_key and not re.fullmatch(r"[A-Za-z0-9._\-]{16,}", api_key):
        raise ValueError(f"{ENV_KEY} tem formato inesperado; confira a chave (sem espaços ou aspas).")
    # 4. Monta as configurações com valores padrão para cada chave ausente do JSON.
    # EXTENSAO: novo parâmetro do vlm_assist.json: leia-o aqui com raw.get("nome", padrão)
    # e acrescente o campo correspondente em VLMSettings.
    return VLMSettings(
        provider=str(raw.get("provider", "openrouter")),
        # rstrip("/") evita barra dupla ao concatenar "/chat/completions" em _request.
        base_url=str(env.get(ENV_BASE_URL) or raw.get("base_url", "https://openrouter.ai/api/v1")).rstrip("/"),
        model=str(model or env.get(ENV_MODEL) or raw.get("model", "openrouter/free")),
        api_key=api_key, timeout_seconds=float(raw.get("timeout_seconds", 90)),
        pause_seconds=float(raw.get("pause_seconds", 1.0)), max_output_tokens=int(raw.get("max_output_tokens", 400)),
        temperature=float(raw.get("temperature", 0.0)), crop_margin=float(raw.get("crop_margin", 0.3)),
        max_crop_side=int(raw.get("max_crop_side", 768)), min_crop_side=int(raw.get("min_crop_side", 96)),
        max_boxes_per_image=int(raw.get("max_boxes_per_image", 20)), app_title=str(raw.get("app_title", "DroneCamp Vision")),
        # Hash do JSON lido: as saídas registram exatamente qual configuração foi usada.
        settings_sha256=file_hash(path),
    )


# ---------------------------------------------------------------------------
# Cliente HTTP (API compatível com OpenAI). ``transport`` é trocado nos testes.
# ---------------------------------------------------------------------------

def _urllib_transport(url: str, headers: dict, body: bytes | None, timeout: float) -> tuple[int, bytes]:
    """Transporte HTTP padrão (biblioteca padrão ``urllib``, sem dependência nova).

    Parâmetros:
    - ``url``: endereço completo (base + caminho, ex.: ``.../api/v1/chat/completions``).
    - ``headers``: cabeçalhos HTTP (incluem a autorização).
    - ``body``: corpo JSON em bytes; None faz um GET, bytes fazem um POST.
    - ``timeout``: tempo máximo em segundos.

    Retorna: ``(status HTTP, corpo em bytes)``. Erros HTTP (4xx/5xx) NÃO lançam
    exceção: voltam como status para ``OpenRouterClient._request`` decidir a mensagem.

    Erros: falhas de rede (``URLError``, ``TimeoutError``, ``OSError``) sobem para
    ``_request``, que as converte em ``VLMUnavailable``.

    ATENCAO: os testes substituem esta função pelo parâmetro ``transport`` com a
    mesma assinatura ``(url, headers, body, timeout) -> (status, bytes)``.
    """
    # Método decidido pela presença de corpo: POST para chat, GET para /models.
    request = urllib.request.Request(url, data=body, headers=headers, method="POST" if body else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        # HTTPError também é uma resposta: devolve o código e o corpo com a explicação do servidor.
        return error.code, error.read()


class OpenRouterClient:
    """Cliente mínimo da API do OpenRouter (formato OpenAI ``chat/completions``).

    Guarda as configurações, o transporte HTTP e um contador de requisições.
    Métodos públicos: ``chat`` (uma pergunta com imagem) e ``models`` (lista de modelos).

    Chamado por: ``describe_occurrences`` e ``list_free_vision_models``.
    EXTENSAO: novo endpoint da API (ex.: ``/generation`` para custo): crie um método
    que chame ``self._request("/caminho", payload_ou_None)``.
    """

    def __init__(self, settings: VLMSettings, transport=None, usage_log=None):
        """Cria o cliente e falha cedo se não houver chave.

        Parâmetros:
        - ``settings``: ``VLMSettings`` de ``load_vlm_settings``.
        - ``transport``: função ``(url, headers, body, timeout) -> (status, bytes)``;
          None usa ``_urllib_transport`` (os testes passam um falso).
        - ``usage_log``: função ``log(entry: dict)`` que registra os metadados de
          consumo de cada chamada ao modelo (``ai_usage.usage_logger``); None não registra.

        Erros: ``VLMUnavailable`` quando ``settings.api_key`` é None ou vazio, com a
        instrução de definir ``OPENROUTER_API_KEY``.
        """
        # Sem chave nenhuma chamada é feita: o comando para antes de recortar fotos.
        if not settings.api_key:
            raise VLMUnavailable(f"Defina {ENV_KEY} (variável de ambiente ou arquivo .env em sistema_ia/) "
                                 "para usar o apoio do VLM; a chave não fica em nenhum arquivo versionado.")
        self.settings = settings
        self.transport = transport or _urllib_transport
        self.usage_log = usage_log
        # Requisições que chegaram a ter resposta (inclui as com erro HTTP).
        self.calls = 0
        # Código HTTP da última resposta (None antes da primeira ou sem rede); vai para o registro de consumo.
        self.last_status = None

    def _headers(self) -> dict:
        """Cabeçalhos de toda requisição.

        Retorna: dict com ``Authorization: Bearer <chave>``, ``Content-Type`` JSON e
        ``X-Title`` (nome do app).

        ATENCAO: este dict contém a chave; nunca o grave em arquivo nem o imprima.
        EXTENSAO: cabeçalho novo (ex.: ``HTTP-Referer``): acrescente a chave neste dict.
        """
        # Cabeçalhos opcionais do OpenRouter (HTTP-Referer/X-Title) só identificam o app nos rankings.
        return {"Authorization": f"Bearer {self.settings.api_key}", "Content-Type": "application/json",
                "X-Title": self.settings.app_title}

    def _request(self, path: str, payload: dict | None) -> dict:
        """Faz uma requisição e traduz falhas em ``VLMUnavailable`` com mensagem clara.

        Parâmetros:
        - ``path``: caminho relativo à ``base_url`` (ex.: ``"/chat/completions"``, ``"/models"``).
        - ``payload``: dict enviado como JSON (POST) ou None (GET).

        Retorna: o JSON da resposta já decodificado (normalmente um dict).

        Erros (todos ``VLMUnavailable``):
        - sem rede/tempo esgotado: ``"Sem acesso a <base_url>: ..."``;
        - HTTP 401/403: chave recusada;
        - HTTP 429: limite dos modelos gratuitos;
        - outro HTTP >= 400: status e os 300 primeiros caracteres do corpo;
        - corpo que não é JSON, ou JSON com a chave ``"error"``.
        """
        # 1. Serializa o payload (None = GET sem corpo).
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        # 2. Envia; falhas de rede viram VLMUnavailable (o comando explica e para).
        # last_status zera antes: falha sem resposta HTTP não herda o código da chamada anterior.
        self.last_status = None
        try:
            status, raw = self.transport(f"{self.settings.base_url}{path}", self._headers(), body, self.settings.timeout_seconds)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise VLMUnavailable(f"Sem acesso a {self.settings.base_url}: {type(error).__name__}: {error}") from error
        self.calls += 1
        self.last_status = status
        # 3. Decodifica o corpo; errors="replace" evita exceção com bytes inválidos.
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        # 4. Traduz os códigos HTTP em mensagens para o usuário.
        # EXTENSAO: tratamento específico de outro código HTTP (ex.: 402 sem créditos, 503):
        # acrescente um if aqui, antes do teste genérico status >= 400.
        if status in (401, 403):
            raise VLMUnavailable(f"OpenRouter recusou a chave (HTTP {status}). Gere outra chave e defina {ENV_KEY}.")
        if status == 429:
            raise VLMUnavailable("Limite de uso dos modelos gratuitos atingido (HTTP 429); tente mais tarde ou troque o modelo.")
        if status >= 400:
            raise VLMUnavailable(f"OpenRouter respondeu HTTP {status}: {text[:300]}")
        # 5. Converte em JSON.
        try:
            data = json.loads(text)
        except ValueError as error:
            raise VLMUnavailable(f"Resposta não é JSON: {text[:200]}") from error
        # 6. O OpenRouter às vezes responde HTTP 200 com {"error": {...}} (falha do provedor sorteado).
        if isinstance(data, dict) and data.get("error"):
            raise VLMUnavailable(f"Erro do OpenRouter: {json.dumps(data['error'], ensure_ascii=False)[:300]}")
        return data

    def chat(self, messages: list[dict]) -> tuple[str, dict]:
        """Texto da resposta e metadados (modelo que de fato respondeu, uso de tokens).

        Parâmetros:
        - ``messages``: lista no formato OpenAI (``build_messages`` monta: system + user com imagem).

        Retorna: ``(texto, meta)``; ``meta`` tem ``model_used`` (o roteador
        ``openrouter/free`` escolhe um modelo e devolve o nome dele aqui),
        ``provider``, ``usage`` (tokens) e ``finish_reason``.

        Erros: ``VLMUnavailable`` (de ``_request``) ou quando a resposta não tem
        ``choices[0].message.content``.

        Chamada por: ``describe_occurrences``.
        """
        # Corpo da requisição no formato OpenAI.
        # EXTENSAO: novo parâmetro de geração (ex.: "top_p", "response_format"): acrescente aqui,
        # crie o campo em VLMSettings e inclua-o em resume_digest se ele mudar a resposta.
        payload = {"model": self.settings.model, "messages": messages, "temperature": self.settings.temperature,
                   "max_tokens": self.settings.max_output_tokens}

        def extract(data: dict) -> tuple[str, dict]:
            # Extrai o texto da primeira escolha; estrutura diferente vira VLMUnavailable
            # (e entra no registro de consumo como falha, por estar dentro de _measured).
            try:
                choice = data["choices"][0]
                content = choice["message"]["content"]
            except (KeyError, IndexError, TypeError) as error:
                raise VLMUnavailable(f"Resposta sem conteúdo: {json.dumps(data, ensure_ascii=False)[:300]}") from error
            if isinstance(content, list):  # alguns provedores devolvem partes
                # Junta só as partes de texto ({"type": "text", "text": ...}).
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            # Metadados gravados em ai_notes.json (model_used, usage).
            meta = {"model_used": data.get("model"), "provider": data.get("provider"), "usage": data.get("usage"),
                    "finish_reason": choice.get("finish_reason")}
            return str(content or ""), meta

        return self._measured(payload, extract)

    def ping(self) -> dict:
        """Chamada mínima (1 token) que confirma chave, rede e modelo; usada antes de salvar a chave.

        Retorna: ``{"model_used": ...}`` com o modelo que respondeu (o roteador
        ``openrouter/free`` informa qual sorteou).

        Erros: ``VLMUnavailable`` com a mensagem de ``_request`` (chave recusada, 429, rede...).
        Resposta sem texto NÃO é erro aqui: alguns modelos gastam o único token em raciocínio.

        Chamada por: ``ai_panel.save_settings`` (submenu Configuração da IA).
        """
        payload = {"model": self.settings.model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
        return self._measured(payload, lambda data: {"model_used": data.get("model")})

    def _measured(self, payload: dict, extract):
        """Faz ``POST /chat/completions`` e registra os metadados de consumo (sucesso ou falha).

        Parâmetros:
        - ``payload``: corpo da requisição (formato OpenAI).
        - ``extract``: função que recebe o JSON da resposta e devolve o resultado de quem chamou;
          pode lançar ``VLMUnavailable`` (resposta sem conteúdo), que também conta como falha.

        Retorna: o que ``extract`` devolver.

        ATENCAO: o registro (``usage_log``) recebe só metadados: modelo, tempo, código HTTP,
        tokens e custo. Nunca passe ``payload`` ou o texto da resposta para ele.
        Chamada por: ``chat`` e ``ping``.
        """
        started = time.monotonic()
        entry = {"provider": self.settings.provider, "model": self.settings.model}
        try:
            data = self._request("/chat/completions", payload)
            result = extract(data)
        except VLMUnavailable:
            if self.usage_log:
                self.usage_log({**entry, "status": "error", "duration_ms": round((time.monotonic() - started) * 1000),
                                "http_status": self.last_status if (self.last_status or 0) >= 400 else None})
            raise
        if self.usage_log:
            self.usage_log({**entry, "status": "success", "duration_ms": round((time.monotonic() - started) * 1000),
                            "model_used": data.get("model") if isinstance(data, dict) else None,
                            **completion_usage(data.get("usage") if isinstance(data, dict) else None)})
        return result

    def models(self) -> list[dict]:
        """Lista os modelos do provedor (GET ``/models``).

        Retorna: lista de dicts como vêm da API (``id``, ``name``, ``pricing``,
        ``architecture.input_modalities``, ``context_length``...); lista vazia se
        a resposta não for um dict.

        Chamada por: ``list_free_vision_models``.
        """
        data = self._request("/models", None)
        return list(data.get("data", [])) if isinstance(data, dict) else []


def list_free_vision_models(settings: VLMSettings, transport=None) -> list[dict]:
    """Modelos gratuitos (sufixo ``:free`` ou preço zero) que aceitam imagem na entrada.

    ENTRADA: comando ``list-free-models`` (cli.py), que imprime e opcionalmente grava a lista.

    Parâmetros:
    - ``settings``: ``VLMSettings`` (precisa de chave).
    - ``transport``: transporte HTTP falso nos testes; None = rede real.

    Retorna: lista ordenada por ``id`` de dicts ``{"id", "name", "context_length",
    "input_modalities"}``. Os ids servem para ``--model`` ou ``"model"`` do JSON.

    Erros: ``VLMUnavailable`` sem chave, sem rede ou com chave recusada.
    """
    client = OpenRouterClient(settings, transport)
    rows = []
    # Filtra cada modelo do catálogo por dois critérios: gratuito e com entrada de imagem.
    for item in client.models():
        identifier = str(item.get("id", ""))
        pricing = item.get("pricing") or {}
        # Gratuito = slug ":free", o roteador "openrouter/free" ou preço "0" no prompt E na resposta.
        # O padrão "1" faz um modelo sem preço declarado contar como pago.
        free = identifier.endswith(":free") or identifier == "openrouter/free" or (
            str(pricing.get("prompt", "1")) in ("0", "0.0") and str(pricing.get("completion", "1")) in ("0", "0.0"))
        modalities = (item.get("architecture") or {}).get("input_modalities") or []
        if free and "image" in modalities:
            # EXTENSAO: nova coluna na listagem (ex.: preço, descrição): acrescente a chave neste dict.
            rows.append({"id": identifier, "name": item.get("name"), "context_length": item.get("context_length"),
                         "input_modalities": modalities})
    return sorted(rows, key=lambda row: row["id"])


# ---------------------------------------------------------------------------
# Recortes e prompt.
# ---------------------------------------------------------------------------

def crop_box(image: Image.Image, bbox: list[float], margin: float, max_side: int, min_side: int) -> tuple[bytes, list[int]]:
    """JPEG do recorte com margem (contexto) e lado ≤ max_side; devolve também o recorte usado.

    Parâmetros:
    - ``image``: foto original aberta pelo Pillow (pixels na orientação gravada).
    - ``bbox``: caixa em pixels absolutos ``[x1, y1, x2, y2]`` (formato xyxy do registro,
      origem no canto superior esquerdo; NÃO é o formato YOLO normalizado).
    - ``margin``: fração da largura/altura da caixa acrescentada de CADA lado (0.3 = 30 %).
    - ``max_side``: maior lado do JPEG final, em pixels.
    - ``min_side``: menor tamanho do recorte antes da redução, em pixels.

    Retorna: ``(bytes do JPEG qualidade 90, [left, top, right, bottom])``; a lista é
    o recorte efetivo em pixels da foto (gravado como ``crop_xyxy``).

    Erros: ``ValueError("Caixa fora da foto: ...")`` se o recorte ficar vazio.

    ATENCAO: o hash deste JPEG (``crop_sha256``) é a chave da retomada em
    ``describe_occurrences``; mudar margem, tamanho ou qualidade invalida o cache
    automaticamente (o que é desejado).
    """
    width, height = image.size
    x1, y1, x2, y2 = (float(value) for value in bbox)
    # Margem proporcional ao tamanho da caixa, de cada lado.
    pad_x, pad_y = (x2 - x1) * margin, (y2 - y1) * margin
    # Garante um recorte mínimo: caixas de 20 px viram um quadrado legível.
    # extra = metade do que falta para a largura (caixa + 2 margens) chegar a min_side.
    extra_x = max(0.0, (min_side - (x2 - x1 + 2 * pad_x)) / 2)
    extra_y = max(0.0, (min_side - (y2 - y1 + 2 * pad_y)) / 2)
    # Limita o recorte às bordas da foto (perto da borda ele fica menor que min_side).
    # int() trunca para dentro no canto superior esquerdo; round() no canto inferior direito.
    left = max(0, int(x1 - pad_x - extra_x))
    top = max(0, int(y1 - pad_y - extra_y))
    right = min(width, int(round(x2 + pad_x + extra_x)))
    bottom = min(height, int(round(y2 + pad_y + extra_y)))
    # Caixa totalmente fora da foto (registro corrompido ou foto trocada).
    if right <= left or bottom <= top:
        raise ValueError(f"Caixa fora da foto: {bbox}.")
    # Recorta e converte para RGB (JPEG não aceita RGBA nem paleta).
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    # Reduz proporcionalmente se o maior lado passar de max_side (nunca amplia).
    scale = max_side / max(crop.size)
    if scale < 1:
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.LANCZOS)
    # Codifica em memória (nada é gravado no disco).
    buffer = BytesIO()
    crop.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue(), [left, top, right, bottom]


def taxonomy_prompt(taxonomy: dict) -> str:
    """Classes ativas com critério, na língua do revisor, para o VLM escolher um slug.

    Parâmetros:
    - ``taxonomy``: dict de ``configs/taxonomy.json`` (``load_taxonomy``).

    Retorna: texto com uma linha por classe ativa (fase 1), no formato
    ``- <slug>: <label>. <criterion>``.

    ATENCAO: o texto entra em ``prompt_digest``; editar ``label``/``criterion`` na
    taxonomia invalida as notas antigas (os recortes são descritos de novo).
    """
    lines = []
    # Só a fase 1 (classes que o detector treina hoje); fase 2 fica fora do prompt.
    # EXTENSAO: para oferecer outras fases ao VLM, mude o filtro aqui E a lista ``names``
    # de describe_occurrences (senão a resposta vira "incerto" em parse_answer).
    for item in taxonomy["classes"]:
        if item.get("phase") == 1:
            lines.append(f"- {item['slug']}: {item['label']}. {item.get('criterion', '')}".strip())
    return "\n".join(lines)


def build_messages(taxonomy: dict, crop_jpeg: bytes) -> list[dict]:
    """Prompt fixo (versão PROMPT_VERSION): só o visível, sem gravidade nem recomendação.

    Parâmetros:
    - ``taxonomy``: dict da taxonomia (as classes ativas vão no texto do usuário).
    - ``crop_jpeg``: bytes do JPEG do recorte (``crop_box``); ``b""`` em ``prompt_digest``.

    Retorna: lista de mensagens no formato OpenAI: ``[system, user]``, em que o
    ``user`` tem uma parte de texto e uma parte ``image_url`` com o JPEG em data URL base64.

    ATENCAO: a classe humana ou sugerida NÃO é enviada; a concordância de
    ``agreement_report`` só faz sentido com uma leitura independente.
    ATENCAO: o prompt proíbe gravidade, causa, risco e recomendação (regra do projeto).
    EXTENSAO: nova chave na resposta do VLM: acrescente-a no texto ``system`` abaixo,
    leia-a em ``parse_answer``, suba ``PROMPT_VERSION`` e, se for para o laudo,
    copie-a em ``draft_report_support``.
    """
    # Instruções fixas do sistema; as chaves do JSON pedido (descricao, classe_sugerida,
    # evidencias, legibilidade) são as que parse_answer lê.
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
    # Texto do usuário: a lista de classes ativas e o pedido.
    user_text = ("Classes do projeto (slug: nome. critério):\n" + taxonomy_prompt(taxonomy) +
                 "\n\nDescreva o recorte e responda só com o JSON pedido.")
    # Imagem embutida como data URL (sem upload separado).
    data_url = "data:image/jpeg;base64," + base64.b64encode(crop_jpeg).decode("ascii")
    return [{"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": user_text},
                                         {"type": "image_url", "image_url": {"url": data_url}}]}]


def prompt_digest(taxonomy: dict) -> str:
    """Identifica o prompt (versão + classes) para invalidar notas antigas.

    Parâmetros:
    - ``taxonomy``: dict da taxonomia.

    Retorna: SHA-256 hexadecimal de ``[PROMPT_VERSION, texto do system, lista de classes]``
    (gravado como ``prompt_sha256`` em ``ai_notes.json``).

    ATENCAO: o texto ``user`` fixo ("Descreva o recorte...") não entra no hash; se
    ele mudar, suba ``PROMPT_VERSION`` para invalidar as notas.
    """
    # Monta as mensagens com imagem vazia só para obter o texto do system.
    system = build_messages(taxonomy, b"")[0]["content"]
    return sha256(json.dumps([PROMPT_VERSION, system, taxonomy_prompt(taxonomy)]).encode("utf-8")).hexdigest()


def resume_digest(settings: VLMSettings, prompt_sha256: str) -> str:
    """Chave da retomada: prompt + o que muda a resposta (modelo pedido, temperatura, tokens).

    Trocar ``--model``/``OPENROUTER_MODEL`` invalida as notas antigas: os recortes
    são enviados de novo ao modelo novo. Os parâmetros do recorte já entram no
    hash do JPEG de cada caixa.

    Parâmetros:
    - ``settings``: configurações da execução.
    - ``prompt_sha256``: resultado de ``prompt_digest``.

    Retorna: SHA-256 hexadecimal (gravado como ``resume_sha256`` em ``ai_notes.json``).

    EXTENSAO: novo parâmetro que mude a resposta do VLM: acrescente-o na lista
    abaixo, senão notas antigas seriam reaproveitadas indevidamente.
    """
    return sha256(json.dumps([prompt_sha256, settings.model, settings.temperature, settings.max_output_tokens]).encode("utf-8")).hexdigest()


def parse_answer(text: str, names: list[str]) -> dict:
    """Aceita JSON puro, com cerca ``json`` ou com texto em volta; o que não encaixa vira ``incerto``.

    Parâmetros:
    - ``text``: resposta crua do VLM.
    - ``names``: slugs das classes ativas (fase 1), na ordem dos IDs.

    Retorna: dict com
    - ``description``: texto da descrição (ou None);
    - ``class_guess``: slug válido, ``"nenhuma"`` ou ``"incerto"``;
    - ``evidence``: lista de até 4 textos de até 200 caracteres;
    - ``legibility``: um valor de ``LEGIBILITY`` ou None;
    - ``parsed``: True se havia JSON utilizável; False se a resposta foi guardada
      como texto livre (até 600 caracteres) em ``description``.

    Nunca lança exceção: resposta fora do formato vira ``class_guess="incerto"``.
    Chamada por: ``describe_occurrences``.
    """
    candidate = text.strip()
    # 1. Isola o objeto JSON: primeiro procura cerca markdown ```json {...} ```;
    #    a regex captura o primeiro "{...}" (não guloso) seguido do fechamento da cerca.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        # Sem cerca: pega do primeiro "{" ao último "}" (ignora texto antes e depois).
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end > start:
            candidate = candidate[start:end + 1]
    # 2. Decodifica; se não for um objeto JSON, guarda o texto livre como descrição não estruturada.
    try:
        data = json.loads(candidate)
        if not isinstance(data, dict):
            raise ValueError
    except ValueError:
        return {"description": text.strip()[:600] or None, "class_guess": "incerto", "evidence": [],
                "legibility": None, "parsed": False}
    # 3. Classe sugerida: só slug ativo ou resposta especial; o resto vira "incerto".
    # ATENCAO: o VLM nunca cria classe nova; slugs fora da taxonomia são descartados aqui.
    guess = str(data.get("classe_sugerida", "incerto")).strip().lower()
    if guess not in names and guess not in NO_CLASS:
        guess = "incerto"
    # 4. Legibilidade e evidências: normaliza tipos (evidência solta vira lista de um item).
    legibility = str(data.get("legibilidade", "")).strip().lower() or None
    evidence = data.get("evidencias")
    if not isinstance(evidence, list):
        evidence = [str(evidence)] if evidence else []
    # 5. Resultado com limites de tamanho (evita respostas enormes em ai_notes.json).
    # EXTENSAO: nova chave pedida no prompt: leia data.get("nova_chave") e acrescente-a neste dict
    # (e no dict de erro acima, com valor None, para manter as mesmas chaves).
    return {"description": (str(data.get("descricao", "")).strip() or None),
            "class_guess": guess, "evidence": [str(item)[:200] for item in evidence][:4],
            "legibility": legibility if legibility in LEGIBILITY else None, "parsed": True}


# ---------------------------------------------------------------------------
# Descrição das ocorrências de uma revisão (ai_notes.json).
# ---------------------------------------------------------------------------

def _boxes_to_describe(registry: dict, suggestions: dict | None, include_suggestions: bool, per_image: int) -> list[tuple[dict, dict, str]]:
    """(foto, caixa, origem) das caixas humanas e, opcionalmente, das sugestões pendentes.

    Parâmetros:
    - ``registry``: conteúdo do ``registry.json`` (lista ``images`` com ``boxes``).
    - ``suggestions``: conteúdo do ``suggestions.json`` ou None; ``images`` é um dict
      ``{image_sha256: [caixas sugeridas]}``.
    - ``include_suggestions``: True para incluir também as sugestões do modelo.
    - ``per_image``: máximo de caixas por foto (``max_boxes_per_image``), somando as duas origens.

    Retorna: lista de tuplas ``(item da foto, caixa, origem)``, em que origem é
    ``"humana"`` (foto aprovada), ``"proposta"`` (foto ainda não aprovada) ou ``"sugestao"``.

    ATENCAO: só ``"humana"`` entra em ``agreement_report`` e em ``draft_report_support``.
    EXTENSAO: nova origem de caixa: acrescente outro laço aqui com o rótulo novo e
    decida se ``agreement_report`` deve contá-la.
    """
    rows = []
    for item in registry["images"]:
        # Contador por foto: caixas do registro vêm primeiro e consomem o limite antes das sugestões.
        count = 0
        # Caixas de foto sem aprovação humana podem ser propostas de IA (first_pass_ai):
        # entram como "proposta" e ficam fora da concordância.
        source = "humana" if item.get("human_approved") is True else "proposta"
        # Caixas do registro, na ordem gravada.
        for box in item.get("boxes", []):
            if count >= per_image:
                break
            rows.append((item, box, source))
            count += 1
        # Sugestões pendentes (suggestions.json), procuradas pelo hash da foto.
        if include_suggestions and suggestions:
            for box in suggestions.get("images", {}).get(item["image_sha256"], []):
                if count >= per_image:
                    break
                rows.append((item, box, "sugestao"))
                count += 1
    return rows


def _slug(names: list[str], class_id) -> str | None:
    """Slug da classe ativa; caixas fora da taxonomia (registro antigo) ficam sem slug.

    Parâmetros:
    - ``names``: slugs da fase 1 na ordem dos IDs (índice = ``class_id``).
    - ``class_id``: ID numérico da classe (int ou texto numérico).

    Retorna: o slug, ou None se o ID não for número ou estiver fora da lista.

    ATENCAO: usa o ID como ÍNDICE da lista; isso só vale porque as classes da fase 1
    têm IDs contíguos 0..N-1 em ``configs/taxonomy.json``. Se a taxonomia ganhar
    buracos ou classes de fase 1 com ID depois das de fase 2, troque por busca pelo ``id``.
    """
    try:
        index = int(class_id)
    except (TypeError, ValueError):
        return None
    return names[index] if 0 <= index < len(names) else None


def _load_suggestions(registry_path: Path) -> dict | None:
    """Lê ``suggestions.json`` ao lado do registro, se existir e for desta versão.

    Parâmetros:
    - ``registry_path``: caminho do ``registry.json``.

    Retorna: o dict das sugestões, ou None se o arquivo não existir.

    Erros: ``ValueError`` se ``suggestions.json`` tiver ``registry_sha256`` diferente
    do hash atual do registro (sugestões de outra versão; gere de novo com ``suggest``).
    """
    path = registry_path.parent / "suggestions.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    # ATENCAO: sugestões só valem para o registro exato em que foram geradas (hash do arquivo).
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

    ENTRADA: comando ``describe-occurrences`` (cli.py; ``--no-resume`` vira ``resume=False``).

    Parâmetros:
    - ``config``: ``ProjectConfig`` do projeto.
    - ``registry_path``: caminho do ``registry.json`` da revisão.
    - ``include_suggestions``: inclui as caixas de ``suggestions.json`` (origem ``"sugestao"``).
    - ``limit``: máximo de chamadas NOVAS ao VLM nesta execução (None = sem limite);
      as caixas que passarem ficam com ``skipped``.
    - ``model``: slug do modelo (``--model``); None usa ambiente/JSON.
    - ``transport``: transporte HTTP falso nos testes (também desliga a pausa entre chamadas).
    - ``resume``: False ignora o ``ai_notes.json`` anterior e redescreve tudo.
    - ``environ``: variáveis de ambiente para os testes.

    Retorna: ``Path`` do ``ai_notes.json`` gravado. Chaves principais do arquivo:
    ``registry_sha256``, ``prompt_sha256``, ``resume_sha256``, ``settings``,
    ``calls_this_run``, ``errors_this_run``, ``images`` (``{image_sha256: [notas]}``),
    ``total_boxes``, ``described`` e ``agreement``.

    Erros:
    - ``ValueError`` se a taxonomia mudou desde o registro, se a foto original mudou
      (hash diferente) ou se ``suggestions.json`` for de outra versão;
    - ``VLMUnavailable`` sem chave ou se a PRIMEIRA chamada falhar (nada é gravado).

    ATENCAO: o arquivo inteiro é regravado a cada execução (não é incremental no disco);
    a retomada copia as notas antigas válidas para o arquivo novo.
    """
    # 1. Lê o registro e confere que ele usa a taxonomia atual (IDs de classe = contrato).
    registry_path = Path(registry_path).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    taxonomy = load_taxonomy(config.taxonomy_path)
    # ATENCAO: com taxonomia diferente, os class_id do registro apontariam para slugs errados.
    if registry.get("taxonomy_sha256") != file_hash(config.taxonomy_path):
        raise ValueError("Taxonomia mudou: migre a revisão antes de pedir descrições.")
    # Slugs ativos na ordem dos IDs (índice = class_id; ver _slug).
    names = [item["slug"] for item in taxonomy["classes"] if item.get("phase") == 1]
    # 2. Configurações e cliente; sem chave o comando falha aqui, antes de abrir fotos.
    settings = load_vlm_settings(config, model, environ)
    # Cada chamada entra no consumo local (.runtime/ai_usage.jsonl, submenu Uso e limites).
    client = OpenRouterClient(settings, transport, usage_log=usage_logger(config.root, "describe"))  # falha cedo sem chave
    # 3. Chaves de retomada: prompt, parâmetros do modelo e versão do registro.
    digest = prompt_digest(taxonomy)
    resume_key = resume_digest(settings, digest)
    registry_sha = file_hash(registry_path)
    notes_path = registry_path.parent / NOTES_FILE
    # 4. Cache das notas anteriores: só reaproveita se registro E prompt/modelo forem os mesmos.
    #    O dict "previous" é indexado pelo hash do JPEG do recorte (crop_sha256);
    #    notas sem descrição (erro, pulada por limite) não entram e são pedidas de novo.
    previous = {}
    if resume and notes_path.is_file():
        old = json.loads(notes_path.read_text(encoding="utf-8"))
        if old.get("registry_sha256") == registry_sha and old.get("resume_sha256") == resume_key:
            previous = {entry["crop_sha256"]: entry for entries in old.get("images", {}).values() for entry in entries
                        if entry.get("description")}
    # 5. Lista das caixas a descrever (humanas/propostas e, se pedido, sugestões).
    rows = _boxes_to_describe(registry, _load_suggestions(registry_path) if include_suggestions else None,
                              include_suggestions, settings.max_boxes_per_image)
    # images: {image_sha256: [nota por caixa]}, mesmo formato lido por laudo_abnt.match_ai_note.
    images: dict[str, list[dict]] = {}
    # calls = chamadas bem-sucedidas nesta execução; errors = falhas após a primeira chamada.
    calls, errors, started = 0, 0, time.monotonic()
    # 6. Laço principal: uma nota por caixa.
    for item, box, source in rows:
        # 6a. Abre a foto original e confere que ela é a mesma registrada (hash).
        path = resolve_local_path(item["source_path"], config.root)
        # ATENCAO: foto alterada depois do registro = coordenadas sem sentido; o comando para.
        if file_hash(path) != item["image_sha256"]:
            raise ValueError(f"A foto original mudou: {item.get('filename')}.")
        # 6b. Recorta a caixa com margem e calcula o hash do JPEG (chave do cache).
        with Image.open(path) as image:
            jpeg, crop = crop_box(image, box["bbox_xyxy"], settings.crop_margin, settings.max_crop_side, settings.min_crop_side)
        crop_sha = sha256(jpeg).hexdigest()
        # 6c. Nota base da caixa. Gravidade, diagnóstico e recomendação ficam SEMPRE None.
        # ATENCAO: "source", "description", "class_id" e "bbox_xyxy" são lidos por laudo_abnt.match_ai_note
        # e draft_report_support; não renomeie sem mudar os dois.
        # ATENCAO: a IA nunca aprova: status "pendente" e campos técnicos None.
        # EXTENSAO: novo campo fixo por caixa (ex.: "page"): acrescente a chave neste dict.
        entry = {"box_id": box.get("id"), "source": source, "class_id": box.get("class_id"),
                 "class_slug": _slug(names, box.get("class_id")),
                 "confidence": box.get("confidence"), "bbox_xyxy": box["bbox_xyxy"], "crop_xyxy": crop,
                 "crop_sha256": crop_sha, "status": "pendente", "severity": None, "diagnosis": None, "recommendation": None}
        # 6d. Três caminhos: reaproveitar nota antiga, pular por limite ou chamar o VLM.
        if crop_sha in previous:
            # Mesmo recorte já descrito com o mesmo prompt/modelo: copia a resposta, sem chamada.
            # EXTENSAO: campo novo vindo da resposta do VLM: acrescente-o nesta tupla para que
            # sobreviva à retomada.
            cached = previous[crop_sha]
            entry.update({key: cached.get(key) for key in ("description", "class_guess", "evidence", "legibility",
                                                             "model_used", "parsed", "described_at")})
            entry["reused"] = True
        elif limit is not None and calls >= limit:
            # Limite de chamadas desta execução atingido: a caixa fica para a próxima execução.
            entry.update({"description": None, "class_guess": None, "evidence": [], "legibility": None,
                          "model_used": None, "parsed": False, "skipped": "limite desta execução"})
        else:
            # Pausa entre chamadas reais (não na primeira, nem nos testes com transporte falso).
            if calls and settings.pause_seconds > 0 and transport is None:
                time.sleep(settings.pause_seconds)
            try:
                # Pergunta ao VLM e interpreta a resposta.
                text, meta = client.chat(build_messages(taxonomy, jpeg))
                calls += 1
                entry.update(parse_answer(text, names))
                # Resposta crua (até 1200 caracteres) fica gravada para auditoria.
                entry.update({"model_used": meta.get("model_used"), "usage": meta.get("usage"),
                              "described_at": datetime.now(timezone.utc).isoformat(), "raw_answer": text[:1200]})
            except VLMUnavailable as error:
                # Chave recusada ou rede fora: para tudo, sem gravar descrições parciais inventadas.
                # Na primeira chamada sobe o erro; depois dela, registra o erro nesta caixa e segue
                # (as próximas caixas continuam tentando; ver errors_this_run).
                if calls == 0:
                    raise
                errors += 1
                entry.update({"description": None, "class_guess": None, "evidence": [], "legibility": None,
                              "model_used": None, "parsed": False, "error": str(error)[:300]})
        # 6e. Agrupa a nota sob o hash da foto.
        images.setdefault(item["image_sha256"], []).append(entry)
    # 7. Monta o relatório final e grava ai_notes.json (sobrescreve o anterior).
    described = [entry for entries in images.values() for entry in entries if entry.get("description")]
    # ATENCAO: "registry_sha256", "images", "settings" e "agreement" são lidos por laudo_abnt/exports;
    # "human_acceptance" fica sempre "pendente" (a IA nunca aceita as próprias notas).
    # EXTENSAO: novo campo de resumo no ai_notes.json: acrescente a chave neste dict
    # (e, se a CLI deve imprimi-lo, na tupla de chaves do comando describe-occurrences em cli.py).
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

    Parâmetros:
    - ``images``: ``{image_sha256: [notas]}`` como montado em ``describe_occurrences``.
    - ``names``: slugs ativos (fase 1).

    Retorna: dict com ``human_boxes_described``, ``agree``, ``agreement_rate``
    (0 a 1 com 3 casas, ou None sem caixas), ``per_class`` (só classes com caixas:
    ``boxes``, ``agree``, ``none_or_uncertain``, ``confusions`` = ``{slug sugerido: contagem}``,
    ``agreement_rate``) e ``note``.

    Chamada por: ``describe_occurrences`` (gravado em ``ai_notes.json["agreement"]``,
    lido por ``laudo_abnt.ai_summary``).
    """
    # Uma linha de contagem por classe ativa (inclusive as que não aparecem; filtradas no fim).
    # EXTENSAO: nova métrica por classe: acrescente o contador neste dict e atualize-o no laço.
    per_class = {name: {"boxes": 0, "agree": 0, "none_or_uncertain": 0, "confusions": {}} for name in names}
    total = agree = 0
    for entries in images.values():
        for entry in entries:
            # Só caixas humanas, descritas, com classe ativa conhecida.
            # ATENCAO: propostas e sugestões ficam fora: a métrica compara com decisão humana.
            if entry.get("source") != "humana" or not entry.get("description") or entry.get("class_slug") is None:
                continue
            row = per_class[entry["class_slug"]]
            row["boxes"] += 1
            total += 1
            guess = entry.get("class_guess")
            # Três desfechos: mesma classe, "nenhuma"/"incerto"/None, ou outra classe (confusão).
            if guess == entry["class_slug"]:
                row["agree"] += 1
                agree += 1
            elif guess in NO_CLASS or guess is None:
                row["none_or_uncertain"] += 1
            else:
                row["confusions"][guess] = row["confusions"].get(guess, 0) + 1
    # Taxa por classe com 3 casas; None quando a classe não teve caixas.
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

    ENTRADA: comando ``report-support`` (cli.py). Não chama o VLM nem precisa de chave.

    Parâmetros:
    - ``config``: ``ProjectConfig`` do projeto.
    - ``registry_path``: caminho do ``registry.json``.
    - ``output``: pasta de saída; None grava ao lado do registro.

    Retorna: ``Path`` do ``laudo_apoio.json`` (o ``.md`` fica na mesma pasta).

    Erros: ``ValueError`` se a taxonomia mudou desde o registro (rode ``migrate-review``).

    ATENCAO: sobrescreve ``laudo_apoio.json``/``.md`` existentes na pasta de saída.
    ATENCAO: ``severity``, ``diagnosis`` e ``recommendation`` ficam sempre None; a
    gravidade de referência da taxonomia vai só como histórico
    (``reference_severity_historical_only``), nunca como gravidade da ocorrência.
    """
    # 1. Registro e taxonomia atual.
    registry_path = Path(registry_path).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    # IDs de classe mudaram entre versões da taxonomia: um registro antigo sairia com slugs errados.
    if registry.get("taxonomy_sha256") != file_hash(config.taxonomy_path):
        raise ValueError("Taxonomia mudou: migre a revisão (migrate-review) antes de gerar o apoio ao laudo.")
    taxonomy = load_taxonomy(config.taxonomy_path)
    # Classes ativas indexadas pelo ID (aqui a busca é pelo campo "id", não pela posição).
    classes = {item["id"]: item for item in taxonomy["classes"] if item.get("phase") == 1}
    registry_sha = file_hash(registry_path)
    # 2. Notas do VLM: usadas só se forem desta versão exata do registro.
    notes_path = registry_path.parent / NOTES_FILE
    notes = {}
    notes_meta = None
    if notes_path.is_file():
        data = json.loads(notes_path.read_text(encoding="utf-8"))
        if data.get("registry_sha256") == registry_sha:
            notes = data.get("images", {})
            # Resumo das notas que vai para o cabeçalho do apoio ao laudo.
            notes_meta = {"created_at": data.get("created_at"), "model_requested": (data.get("settings") or {}).get("model_requested"),
                          "described": data.get("described"), "agreement": data.get("agreement")}
    # 3. Uma ocorrência por caixa de cada foto aprovada por humano.
    occurrences, skipped = [], 0
    for item in registry["images"]:
        # ATENCAO: fotos sem aprovação humana ficam fora do rascunho (só são contadas).
        if not item.get("human_approved"):
            skipped += 1
            continue
        # Duas formas de achar a nota do VLM de cada caixa:
        # by_crop: pelo id da caixa (box_id); by_index: pela posição, para caixas antigas sem id.
        by_crop = {entry.get("box_id"): entry for entry in notes.get(item["image_sha256"], []) if entry.get("source") == "humana"}
        by_index = [entry for entry in notes.get(item["image_sha256"], []) if entry.get("source") == "humana"]
        for index, box in enumerate(item.get("boxes", [])):
            # Classe fora da fase 1 (não deveria ocorrer com taxonomia conferida) vira dict vazio.
            category = classes.get(int(box["class_id"]), {})
            # Nota pelo id; sem id, pela posição, mas só se as coordenadas forem idênticas
            # (evita casar a nota de outra caixa quando a lista mudou).
            note = by_crop.get(box.get("id")) if box.get("id") is not None else None
            if note is None and index < len(by_index) and by_index[index].get("bbox_xyxy") == box["bbox_xyxy"]:
                note = by_index[index]
            # Registro da ocorrência. "occurrence" é a numeração sequencial (1, 2, 3...).
            # EXTENSAO: novo campo por ocorrência no laudo_apoio.json: acrescente a chave aqui
            # e a linha correspondente em _support_markdown.
            occurrences.append({
                "occurrence": len(occurrences) + 1, "image_sha256": item["image_sha256"], "filename": item.get("filename"),
                "pages": item.get("pages", []), "building_group": item.get("building_group"), "scene_group": item.get("scene_group"),
                "box_id": box.get("id"), "bbox_xyxy": box["bbox_xyxy"], "class_id": box["class_id"],
                "class_slug": category.get("slug"), "class_label": category.get("label"),
                "criterion": category.get("criterion"), "human_reviewer": item.get("human_reviewer"),
                "human_notes": item.get("human_notes") or None,
                "reference_severity_historical_only": category.get("reference_severity"),
                # Descrição da IA só quando há nota com texto; status sempre "pendente".
                "ai_description": {"text": note.get("description"), "evidence": note.get("evidence", []),
                                   "legibility": note.get("legibility"), "class_guess": note.get("class_guess"),
                                   "model_used": note.get("model_used"), "status": "pendente"} if note and note.get("description") else None,
                # ATENCAO: campos técnicos em branco para o responsável técnico; nunca inferidos.
                "severity": None, "diagnosis": None, "recommendation": None, "validation": PENDING,
            })
    # 4. Relatório completo.
    # EXTENSAO: novo campo de cabeçalho do apoio ao laudo: acrescente a chave aqui e,
    # se deve aparecer no .md, em _support_markdown.
    report = {"schema_version": 1, "kind": "report_support", "registry_sha256": registry_sha,
              "registry_version": registry.get("version"), "created_at": datetime.now(timezone.utc).isoformat(),
              "taxonomy_version": taxonomy.get("version"), "approved_images": len(registry["images"]) - skipped,
              "images_without_human_approval": skipped, "occurrences": occurrences, "ai_notes": notes_meta,
              "warning": WARNING, "human_acceptance": "pendente"}
    # 5. Grava JSON e Markdown na pasta de saída (criada se preciso).
    directory = Path(output) if output is not None else registry_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "laudo_apoio.json", report)
    (directory / "laudo_apoio.md").write_text(_support_markdown(report), encoding="utf-8")
    return directory / "laudo_apoio.json"


def _support_markdown(report: dict) -> str:
    """Versão legível (Markdown) do apoio ao laudo.

    Parâmetros:
    - ``report``: dict montado em ``draft_report_support``.

    Retorna: texto Markdown com cabeçalho, contagens, resumo das notas de IA e uma
    seção ``## Ocorrência N`` por caixa, com gravidade/diagnóstico/recomendação
    sempre "a definir pelo responsável técnico".

    Chamada por: ``draft_report_support``.
    """
    # Cabeçalho: versão da revisão, data, aviso e contagem de fotos.
    lines = [f"# Apoio ao laudo — revisão {report.get('registry_version')}", "",
             f"Gerado em {report['created_at']}. {report['warning']}", "",
             f"Fotos com aprovação humana: {report['approved_images']}; sem aprovação (fora deste rascunho): "
             f"{report['images_without_human_approval']}.", ""]
    # Resumo das descrições de IA, se houver ai_notes.json desta versão.
    if report.get("ai_notes"):
        meta = report["ai_notes"]
        lines += [f"Descrições de IA: {meta.get('described')} recortes descritos por `{meta.get('model_requested')}` "
                  f"em {meta.get('created_at')} (pendentes de validação).", ""]
    # Uma seção por ocorrência.
    # EXTENSAO: nova linha por ocorrência no .md: acrescente um lines.append neste laço
    # (o campo precisa existir no dict da ocorrência em draft_report_support).
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
        # ATENCAO: estes três campos são sempre deixados para o responsável técnico.
        lines += ["- Gravidade: a definir pelo responsável técnico.", "- Diagnóstico: a definir pelo responsável técnico.",
                  "- Recomendação: a definir pelo responsável técnico.", ""]
    return "\n".join(lines)
