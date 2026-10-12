"""Cadastro das classes pela plataforma: nomes de exibição, cores, ativo/inativo e classes propostas.

Função no projeto: rotas ``/api/classes`` do comando ``platform`` (seção Classes).
As edições ficam em ``configs/taxonomy_overrides.json``, ao lado da taxonomia, e
**nunca reescrevem** ``taxonomy.json``. Motivo: revisões, datasets piloto,
calibradores e laudos guardam o SHA-256 de ``taxonomy.json``; trocar a cor de
uma classe dentro dele invalidaria todos esses arquivos de uma vez.

O que cada operação faz:
- ``list_classes``: taxonomia + edições + classes propostas, com o que cada uma
  exige (``in_detector``: classe ativa que os pesos atuais já aprenderam;
  ``requires_training``: só passa a ser detectada depois de dados aprovados e
  novo treino, mesmo quando já é ativa na taxonomia, como o ID 13 na v6).
- ``create_class``: classe proposta com ID depois do maior da taxonomia, sempre
  ``requires_training``. Ela não entra no detector nem nas caixas da revisão:
  isso exige uma nova versão da taxonomia (como v5 → v6) e ``migrate-review``.
- ``update_class``: nome de exibição, cor e ativo/inativo de qualquer classe;
  nome técnico (slug) e critério só das propostas. O slug de uma classe da
  taxonomia é o nome que os pesos treinados conferem (``check_domain_names``).
- ``deactivate_class``: exclusão lógica (``active: false``); nada é apagado e o
  histórico registra cada mudança.

Inativa não muda o modelo nem as caixas já feitas: a página só deixa de
oferecer a classe para caixas novas.

As edições das classes da taxonomia ficam guardadas **pelo slug**, não pelo ID:
uma migração desloca IDs do catálogo (na v6 o antigo ID 13 virou 23) e o slug é
o que não muda. As propostas são conferidas contra a taxonomia a cada leitura
(``_reconcile``): proposta cujo slug entrou na taxonomia é dada como absorvida e
sai da lista; proposta cujo ID uma expansão da taxonomia ocupou recebe o
próximo ID livre. Tudo fica no ``history``.

Quando mexer: para levar uma proposta à detecção, crie a nova versão da
taxonomia com snapshot e migração (ver ``docs/analise_taxonomia_classes.md``);
não edite IDs existentes.

Conteúdo do arquivo (funções públicas):
- ``overrides_path``: caminho de ``configs/taxonomy_overrides.json``;
- ``load_overrides``: lê e concilia as edições com a taxonomia;
- ``default_color``: cor padrão de uma classe pelo ID (paleta de ``review_render``);
- ``list_classes``: listagem completa da seção Classes (``GET /api/classes``);
- ``create_class``: nova classe proposta (``POST /api/classes``);
- ``update_class``: edição de uma classe (``PUT /api/classes/<id>``);
- ``deactivate_class``: exclusão lógica (``DELETE /api/classes/<id>``);
- ``apply_to_page_classes``: aplica nome/cor/ativo às classes embutidas na página
  servida (``platform_server.make_handler`` → ``Handler._page``).
Funções privadas: ``_empty``, ``_reconcile``, ``_save``, ``_now``, ``_text``,
``_color``, ``_slug``, ``_entry``, ``_find``, ``_get``.

Quem chama: só ``platform_server.py`` (rotas acima) e ``tests/test_class_catalog.py``.
A página (``review_templates/platform.js``, seção "Classes (cadastro)") chama as
rotas com ``classRequest``.

Disco: LÊ ``configs/taxonomy.json`` (via ``config.load_taxonomy``) e
``configs/taxonomy_overrides.json``; GRAVA só ``configs/taxonomy_overrides.json``
(com arquivo temporário ``.json.tmp`` + troca atômica).

Formato de ``taxonomy_overrides.json`` (schema_version 1)::

    {"schema_version": 1, "note": "...",
     "classes":  {"<slug>": {"display_name": "...", "color": "#RRGGBB", "active": false, "updated_at": "..."}},
     "proposed": [{"id": 24, "slug": "...", "label": "...", "color": "#RRGGBB", "active": true,
                   "requires_training": true, "criterion": "...", "created_at": "...", "created_by": "..."}],
     "history":  [{"at": "...", "action": "create|update|absorbed|renumbered", "id": 24, "by": "...", "changes": {...}}]}

Onde mexer:
- Para permitir editar um campo novo de classe pela página (ex.: "descrição"):
  acrescente o nome em ``allowed`` de ``update_class``, valide-o ali (como
  ``display_name``/``color``), devolva-o em ``_entry`` e, se a página o usar nas
  caixas, copie-o em ``apply_to_page_classes``. Do lado da página, ajuste a seção
  "Classes (cadastro)" de ``platform.js`` e o formulário em ``index.html`` (``view-classes``).
- Para mudar o limite de tamanho de nomes/critério: ``MAX_NAME``/``MAX_CRITERION``.
- Para mudar a regra de nome técnico: regex ``SLUG`` e a mensagem em ``_slug``.
- Para mudar a paleta padrão: ``review_render.COLORS`` (não aqui).
- Para guardar mais/menos histórico: ``MAX_HISTORY``.
- Uma ação nova no histórico (ex.: "restore"): acrescente-a onde a ação acontece,
  no mesmo formato ``{"at", "action", "id", "by", "changes"}``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
import threading

from .config import ProjectConfig, load_taxonomy
from .io import file_hash

# ---------------------------------------------------------------------------
# Constantes: arquivo de edições, mensagens, regras de validação e trava
# ---------------------------------------------------------------------------

# Nome do arquivo de edições, criado na MESMA pasta de taxonomy.json (configs/).
# ATENCAO: mudar este nome faz a plataforma "esquecer" as edições já feitas (o arquivo
# antigo deixa de ser lido). Se mudar, renomeie também o arquivo existente em configs/.
OVERRIDES_FILE = "taxonomy_overrides.json"
# Aviso mostrado no topo da seção Classes (campo "notice" de list_classes).
TRAINING_NOTICE = "Mudanças nas classes exigem novo ciclo de treinamento para ter efeito nos modelos."
# Português em snake_case, como os slugs da taxonomia (ex.: pedaco_calha_sobre_telha).
# Regex: começa com letra minúscula, seguida de letras/números; depois, zero ou mais grupos
# "_xxx" (sem "_" duplo, sem "_" no fim, sem acento, sem maiúscula).
# EXTENSAO: para aceitar outro formato de nome técnico, mude esta regex E a mensagem em _slug.
SLUG = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
# Cor no formato HTML "#RRGGBB" (6 dígitos hexadecimais; maiúsculas ou minúsculas na entrada).
COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
# Tamanho máximo (em caracteres, depois de juntar espaços) do nome de exibição.
MAX_NAME = 80
# Tamanho máximo (em caracteres) do critério de anotação de uma classe proposta.
MAX_CRITERION = 1000
# Quantas entradas de histórico ficam no arquivo; as mais antigas são descartadas em _save.
# Aumentar deixa o arquivo maior; diminuir perde histórico antigo na próxima gravação.
MAX_HISTORY = 500
# Trava reentrante (RLock) do processo: o servidor é multi-thread (ThreadingHTTPServer),
# então duas requisições simultâneas não podem ler-modificar-gravar o arquivo ao mesmo tempo.
# RLock (e não Lock) porque create_class/update_class seguram a trava e chamam
# load_overrides, que também a pega.
# ATENCAO: a trava só protege dentro de UM processo; dois servidores rodando ao mesmo tempo
# sobre a mesma pasta podem perder edições um do outro.
_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Erros, leitura e gravação do arquivo de edições
# ---------------------------------------------------------------------------

class ClassNotFound(LookupError):
    """ID que não existe na taxonomia nem entre as propostas.

    Lançada por ``_find`` (logo, por ``update_class`` e ``deactivate_class``).
    ``platform_server.Handler._class_change`` a transforma em resposta HTTP 404.
    """


def overrides_path(config: ProjectConfig) -> Path:
    """Caminho do arquivo de edições: ``configs/taxonomy_overrides.json``.

    Parâmetros:
        config: configuração do projeto; usa ``config.taxonomy_path`` (caminho de taxonomy.json).

    Retorna:
        ``Path`` na mesma pasta de ``taxonomy.json``, com o nome ``OVERRIDES_FILE``.
        O arquivo pode não existir ainda (é criado na primeira edição).
    """
    return config.taxonomy_path.with_name(OVERRIDES_FILE)


def _empty() -> dict:
    """Conteúdo inicial do arquivo de edições quando ele ainda não existe.

    Parâmetros: nenhum.

    Retorna:
        dict no formato schema_version 1, com ``classes`` vazio (edições por slug),
        ``proposed`` vazio (classes propostas) e ``history`` vazio.

    EXTENSAO: uma chave nova de primeiro nível no arquivo (ex.: "aliases") começa aqui
    e precisa ser aceita/validada em ``load_overrides``.
    """
    return {"schema_version": 1, "note": "Edições feitas pela plataforma; taxonomy.json não é alterado. Chaves de classes = slug.",
            "classes": {}, "proposed": [], "history": []}


def _reconcile(taxonomy: dict, overrides: dict) -> bool:
    """Confere as propostas contra a taxonomia atual; devolve True se algo mudou (e precisa gravar).

    Por que existe: quando a equipe cria uma nova versão da taxonomia (ex.: v6), uma
    classe proposta pode ter sido promovida (o slug agora existe na taxonomia) ou o ID
    que ela usava pode ter sido ocupado por outra classe nova. Esta função arruma a
    lista ``proposed`` sem intervenção manual e registra tudo no histórico.

    Parâmetros:
        taxonomy: conteúdo de ``taxonomy.json`` (dict com ``classes``: lista de
            ``{"id": int, "slug": str, "label": str, ...}``).
        overrides: conteúdo de ``taxonomy_overrides.json``; É MODIFICADO no lugar
            (``proposed`` e ``history``).

    Retorna:
        True se alguma proposta foi absorvida ou renumerada (o chamador deve gravar);
        False se nada mudou.

    Chamada por: ``load_overrides`` a cada leitura.
    """
    # Mapa slug → ID e conjunto de IDs já usados pela taxonomia atual.
    slugs = {item["slug"]: item["id"] for item in taxonomy["classes"]}
    ids = {item["id"] for item in taxonomy["classes"]}
    changed = False
    # Propostas que continuam na lista (as absorvidas ficam de fora).
    kept = []
    # Percorre cada proposta, na ordem em que foram criadas.
    for item in overrides["proposed"]:
        # Caso 1: o nome técnico da proposta já existe na taxonomia.
        if item["slug"] in slugs:
            # Entrou na taxonomia (promovida numa nova versão): a proposta cumpriu o papel.
            overrides["history"].append({"at": _now(), "action": "absorbed", "id": item["id"], "by": None,
                                         "changes": {"name": item["slug"], "taxonomy_id": slugs[item["slug"]]}})
            changed = True
            continue
        # Caso 2: o ID da proposta foi ocupado por outra classe da taxonomia.
        if item["id"] in ids:
            # A taxonomia cresceu até este ID com outra classe: a proposta recebe o próximo livre.
            # Próximo livre = maior ID entre a taxonomia e as propostas já mantidas + 1.
            # ATENCAO: o cálculo não olha as propostas que ainda vêm DEPOIS nesta lista; se uma
            # delas já usa o número escolhido aqui, duas propostas podem ficar com o mesmo ID.
            new_id = max(ids | {value["id"] for value in kept}) + 1
            overrides["history"].append({"at": _now(), "action": "renumbered", "id": new_id, "by": None,
                                         "changes": {"name": item["slug"], "previous_id": item["id"]}})
            item["id"] = new_id
            changed = True
        kept.append(item)
    # Substitui a lista de propostas pela lista conciliada.
    overrides["proposed"] = kept
    return changed


def load_overrides(config: ProjectConfig, taxonomy: dict | None = None) -> dict:
    """Lê as edições e as concilia com a taxonomia; grava se a conciliação mudou algo.

    Parâmetros:
        config: configuração do projeto (``taxonomy_path`` aponta para ``configs/taxonomy.json``).
        taxonomy: taxonomia já carregada, para não ler o arquivo de novo; ``None`` lê
            de ``config.taxonomy_path``.

    Retorna:
        dict no formato de ``_empty`` (``schema_version``, ``classes``, ``proposed``,
        ``history``). Se o arquivo não existe, devolve um ``_empty()`` novo (sem gravar).

    Erros:
        ValueError "taxonomy_overrides.json fora do formato esperado (schema_version 1)."
        quando o arquivo existe mas não tem a estrutura esperada; ValueError do
        ``json.loads`` se o JSON estiver corrompido; OSError se não puder ler.

    Chamada por: ``list_classes``, ``create_class``, ``update_class``.
    """
    path = overrides_path(config)
    # Sem arquivo ainda: nenhuma edição feita pela plataforma.
    if not path.is_file():
        return _empty()
    with _LOCK:
        # 1. Lê o JSON.
        value = json.loads(path.read_text(encoding="utf-8"))
        # 2. Confere a estrutura mínima (versão do formato e tipos das chaves principais).
        # EXTENSAO: ao criar schema_version 2, trate a migração do formato aqui.
        if (not isinstance(value, dict) or value.get("schema_version") != 1 or not isinstance(value.get("classes"), dict)
                or not isinstance(value.get("proposed"), list)):
            raise ValueError(f"{path.name} fora do formato esperado (schema_version 1).")
        # 3. Arquivos antigos podiam não ter histórico.
        value.setdefault("history", [])
        # 4. Concilia propostas com a taxonomia atual e grava se algo mudou.
        if _reconcile(taxonomy or load_taxonomy(config.taxonomy_path), value):
            _save(config, value)
    return value


# ATENCAO: a seção Classes grava SÓ em configs/taxonomy_overrides.json. Nunca escreva em
# taxonomy.json daqui: o hash dele está em revisões, datasets e laudos.
def _save(config: ProjectConfig, value: dict) -> None:
    """Grava em arquivo temporário e troca de uma vez: uma queda no meio não deixa JSON pela metade.

    Parâmetros:
        config: configuração do projeto (define o caminho via ``overrides_path``).
        value: conteúdo completo do arquivo de edições; o ``history`` é cortado no lugar
            para as últimas ``MAX_HISTORY`` entradas.

    Retorna: None.

    Erros: OSError se a pasta ``configs/`` não aceitar escrita.

    Chamada por: ``load_overrides`` (após conciliação), ``create_class`` e ``update_class``,
    sempre com ``_LOCK`` seguro pelo chamador.
    """
    path = overrides_path(config)
    # Mantém só as entradas mais recentes do histórico (fatia negativa = últimas N).
    value["history"] = value["history"][-MAX_HISTORY:]
    # Grava primeiro em "taxonomy_overrides.json.tmp" e depois troca pelo definitivo.
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # os.replace é atômico no mesmo disco: ou fica o arquivo antigo, ou o novo inteiro.
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# Utilitários: cor padrão, data/hora e validação dos campos enviados pela página
# ---------------------------------------------------------------------------

def default_color(class_id: int) -> str:
    """Cor padrão de uma classe, a mesma que a página de revisão usa sem edição.

    Parâmetros:
        class_id: ID numérico da classe (ex.: 0, 13, 24).

    Retorna:
        Cor "#RRGGBB" da paleta ``review_render.COLORS``, escolhida pelo resto da divisão
        do ID pelo tamanho da paleta (a paleta se repete para IDs grandes).

    ATENCAO: a importação fica dentro da função para evitar importação circular
    (``review_render`` importa módulos que usam este).
    """
    from .review_render import COLORS
    return COLORS[class_id % len(COLORS)]


def _now() -> str:
    """Data e hora atuais em UTC, formato ISO 8601 sem frações (ex.: "2026-10-12T14:03:00+00:00").

    Parâmetros: nenhum.

    Retorna: texto usado em ``created_at``, ``updated_at`` e no ``history``.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _text(body: dict, key: str, limit: int, required: bool = False) -> str | None:
    """Campo de texto do corpo da requisição, com espaços normalizados e limite de tamanho.

    Parâmetros:
        body: corpo JSON enviado pela página (dict).
        key: nome do campo (ex.: "display_name", "criterion").
        limit: tamanho máximo em caracteres, depois de normalizar espaços.
        required: True exige o campo; False aceita campo ausente/``None`` (devolve None).

    Retorna:
        O texto com espaços repetidos/quebras de linha trocados por um espaço só,
        ou None quando o campo é opcional e não veio.

    Erros:
        ValueError "Informe <key>." se o campo é obrigatório e falta, ou se veio vazio /
        não é texto; ValueError "<key> passa de <limit> caracteres." se é longo demais.
    """
    value = body.get(key)
    # Campo opcional ausente: nada a validar.
    if value is None and not required:
        return None
    # Precisa ser texto com pelo menos um caractere visível.
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Informe {key}.")
    # split() sem argumento quebra em qualquer espaço/tab/quebra; join junta com um espaço só.
    value = " ".join(value.split())
    if len(value) > limit:
        raise ValueError(f"{key} passa de {limit} caracteres.")
    return value


def _color(body: dict) -> str | None:
    """Cor do corpo da requisição, validada e em maiúsculas.

    Parâmetros:
        body: dict com a chave opcional ``"color"`` (ex.: ``{"color": "#ff5c70"}``).

    Retorna:
        "#RRGGBB" em maiúsculas (ex.: "#FF5C70") ou None se a cor não veio.

    Erros:
        ValueError "Cor no formato #RRGGBB." se não casar com a regex ``COLOR``.
    """
    value = body.get("color")
    if value is None:
        return None
    if not isinstance(value, str) or not COLOR.match(value):
        raise ValueError("Cor no formato #RRGGBB.")
    return value.upper()


def _slug(body: dict) -> str:
    """Nome técnico (slug) do corpo da requisição, campo ``"name"``.

    Parâmetros:
        body: dict com a chave ``"name"`` (ex.: ``{"name": "calha_danificada"}``).

    Retorna:
        O slug como veio (sem transformar), se for válido.

    Erros:
        ValueError com a regra do nome técnico se não casar com ``SLUG`` ou se tiver
        menos de 3 ou mais de 60 caracteres.

    EXTENSAO: limites de tamanho do nome técnico ficam aqui (3 a 60); mude a mensagem junto.
    """
    value = body.get("name")
    if not isinstance(value, str) or not SLUG.match(value) or not 3 <= len(value) <= 60:
        raise ValueError("Nome técnico em português snake_case: letras minúsculas sem acento, números e _ (3 a 60 caracteres).")
    return value


# ---------------------------------------------------------------------------
# Leitura: uma entrada da listagem, listagem completa e busca por ID
# ---------------------------------------------------------------------------

def _entry(item: dict, edits: dict, proposed: bool) -> dict:
    """Monta a linha de uma classe como a seção Classes da página a recebe.

    Parâmetros:
        item: a classe crua, vinda de ``taxonomy.json`` (``id``, ``slug``, ``label``,
            ``phase``, ``color``, ``active``, ``criterion``...) ou de ``proposed``.
        edits: edições da plataforma para esta classe (``display_name``, ``color``,
            ``active``, ``updated_at``); vazio para propostas (elas guardam tudo no próprio item).
        proposed: True se a classe é uma proposta (não está na taxonomia).

    Retorna:
        dict com ``id``, ``name`` (slug), ``display_name``, ``original_display_name``,
        ``color``, ``active``, ``phase``, ``source`` ("taxonomia"/"proposta"),
        ``in_detector``, ``requires_training``, ``criterion``, ``name_editable``,
        ``created_at`` e ``updated_at``.

    EXTENSAO: campo novo mostrado na tabela de classes = uma chave nova aqui; a página
    lê essas chaves em ``renderClasses`` (platform.js, seção "Classes (cadastro)").
    """
    class_id = item["id"]
    phase = item.get("phase")
    # Ativa na taxonomia (entra em detection_names) não é o mesmo que já aprendida pelos pesos:
    # o ID 13 da v6 é ativo e ainda exige treino (os pesos piloto têm 13 classes).
    # Regra: proposta sempre exige treino; da taxonomia, vale o campo requires_training e,
    # se ele faltar, exige treino toda classe que não é da fase 1.
    requires_training = True if proposed else bool(item.get("requires_training", phase != 1))
    # Prioridade de cada campo: edição da plataforma > valor da taxonomia > padrão.
    # "active": edição, senão o da taxonomia, senão True; só False explícito desativa.
    # "in_detector": só classe da taxonomia, fase 1 e já aprendida pelos pesos.
    return {"id": class_id, "name": item["slug"],
            "display_name": edits.get("display_name") or item["label"], "original_display_name": item["label"],
            "color": edits.get("color") or item.get("color") or default_color(class_id),
            "active": edits.get("active", item.get("active", True)) is not False,
            "phase": "proposta" if proposed else phase, "source": "proposta" if proposed else "taxonomia",
            "in_detector": not proposed and phase == 1 and not requires_training,
            "requires_training": requires_training,
            "criterion": item.get("criterion"), "name_editable": proposed,
            "created_at": item.get("created_at"), "updated_at": edits.get("updated_at") or item.get("updated_at")}


# ENTRADA: GET /api/classes (platform_server.do_GET) → seção Classes da página.
def list_classes(config: ProjectConfig) -> dict:
    """Listagem completa da seção Classes: taxonomia + edições + propostas, com resumo.

    Parâmetros:
        config: configuração do projeto (``taxonomy_path`` e ``root``).

    Retorna:
        dict com:
        - ``taxonomy_version``: versão declarada em taxonomy.json (ex.: "v6");
        - ``taxonomy_sha256``: hash do arquivo de taxonomia (identidade usada em revisões);
        - ``overrides_file``: caminho do arquivo de edições relativo a ``sistema_ia``
          (ou só o nome, se estiver fora);
        - ``notice``: ``TRAINING_NOTICE``;
        - ``classes``: lista de linhas no formato de ``_entry`` (taxonomia primeiro, depois propostas);
        - ``summary``: contagens ``total``, ``in_detector``, ``inactive``, ``requires_training``.

    Erros: OSError/ValueError de ``load_taxonomy``/``load_overrides`` (o servidor responde 500).
    """
    # 1. Lê taxonomia e edições (a leitura das edições já concilia as propostas).
    taxonomy = load_taxonomy(config.taxonomy_path)
    overrides = load_overrides(config, taxonomy)
    edits = overrides["classes"]  # chave = slug: sobrevive ao deslocamento de IDs nas migrações
    # 2. Uma linha por classe da taxonomia, com as edições do slug dela (ou nenhuma).
    classes = [_entry(item, edits.get(item["slug"], {}), False) for item in taxonomy["classes"]]
    # 3. Depois, uma linha por classe proposta.
    classes += [_entry(item, {}, True) for item in overrides["proposed"]]
    # 4. Resposta com metadados e contagens (sum de booleanos conta os True).
    # EXTENSAO: contagem nova no resumo do topo da seção Classes = chave nova em "summary".
    return {"taxonomy_version": taxonomy.get("version"), "taxonomy_sha256": file_hash(config.taxonomy_path),
            "overrides_file": overrides_path(config).relative_to(config.root).as_posix() if overrides_path(config).is_relative_to(config.root) else OVERRIDES_FILE,
            "notice": TRAINING_NOTICE, "classes": classes,
            "summary": {"total": len(classes), "in_detector": sum(item["in_detector"] for item in classes),
                        "inactive": sum(not item["active"] for item in classes),
                        "requires_training": sum(item["requires_training"] for item in classes)}}


def _find(config: ProjectConfig, overrides: dict, class_id: int) -> tuple[dict, bool]:
    """Procura uma classe pelo ID, primeiro na taxonomia e depois nas propostas.

    Parâmetros:
        config: configuração do projeto (para ler a taxonomia).
        overrides: edições já carregadas (de onde vêm as propostas).
        class_id: ID numérico procurado.

    Retorna:
        Tupla ``(item, proposta)``: o dict cru da classe (o MESMO objeto da lista, para
        poder ser editado no lugar) e True se é proposta, False se é da taxonomia.

    Erros:
        ClassNotFound "Classe <id> não existe." se o ID não aparece em nenhuma das duas.
    """
    taxonomy = load_taxonomy(config.taxonomy_path)
    # Busca na taxonomia oficial.
    for item in taxonomy["classes"]:
        if item["id"] == class_id:
            return item, False
    # Busca nas propostas.
    for item in overrides["proposed"]:
        if item["id"] == class_id:
            return item, True
    raise ClassNotFound(f"Classe {class_id} não existe.")


def _get(config: ProjectConfig, class_id: int) -> dict:
    """Linha atualizada de uma classe (formato ``_entry``), relendo tudo do disco.

    Parâmetros:
        config: configuração do projeto.
        class_id: ID da classe recém-criada/editada.

    Retorna: o dict da classe, como em ``list_classes()["classes"]``.

    Erros: StopIteration se o ID não estiver na listagem (não acontece logo após
    criar/editar, porque a classe acabou de ser gravada).
    """
    return next(item for item in list_classes(config)["classes"] if item["id"] == class_id)


# ---------------------------------------------------------------------------
# Escrita: criar, editar e desativar classes
# ---------------------------------------------------------------------------

# ENTRADA: POST /api/classes (platform_server.do_POST) → botão de criar classe da seção Classes.
def create_class(config: ProjectConfig, body: dict, author: str | None = None) -> dict:
    """Cria uma classe PROPOSTA (não entra no detector nem nas caixas da revisão).

    Parâmetros:
        config: configuração do projeto.
        body: corpo da requisição, com ``name`` (slug, obrigatório), ``display_name``
            (obrigatório, até ``MAX_NAME``), ``color`` ("#RRGGBB", opcional) e
            ``criterion`` (opcional, até ``MAX_CRITERION``). O servidor já tirou ``author``.
        author: nome do revisor que pediu a criação (vai para ``created_by`` e ``history``).

    Retorna:
        A linha da classe criada, no formato de ``_entry``.

    Erros:
        ValueError de validação (``_slug``, ``_text``, ``_color``) ou
        "Já existe uma classe com o nome técnico <slug>." se o slug já existe.

    ATENCAO: o ID é sempre o maior entre taxonomia e propostas + 1. Nunca reaproveite IDs
    nem altere IDs de classes da taxonomia (eles são o contrato com pesos e labels).
    """
    # 1. Valida tudo ANTES de pegar a trava (erros de digitação não seguram o arquivo).
    slug = _slug(body)
    display = _text(body, "display_name", MAX_NAME, required=True)
    color = _color(body)
    criterion = _text(body, "criterion", MAX_CRITERION)
    with _LOCK:
        # 2. Lê estado atual (taxonomia + propostas) dentro da trava.
        taxonomy = load_taxonomy(config.taxonomy_path)
        overrides = load_overrides(config, taxonomy)
        existing = taxonomy["classes"] + overrides["proposed"]
        # 3. Nome técnico não pode repetir (nem na taxonomia nem nas propostas).
        if any(item["slug"] == slug for item in existing):
            raise ValueError(f"Já existe uma classe com o nome técnico {slug}.")
        # 4. Próximo ID livre depois do maior existente.
        class_id = max(item["id"] for item in existing) + 1
        # 5. Monta a proposta no mesmo formato das classes da taxonomia (slug, label...).
        # EXTENSAO: campo novo guardado na proposta = chave nova aqui (e leitura em _entry).
        item = {"id": class_id, "slug": slug, "label": display, "color": color or default_color(class_id), "active": True,
                "requires_training": True, "criterion": criterion, "created_at": _now(), "created_by": author}
        overrides["proposed"].append(item)
        # 6. Registra no histórico e grava.
        overrides["history"].append({"at": item["created_at"], "action": "create", "id": class_id, "by": author,
                                     "changes": {"name": slug, "display_name": display}})
        _save(config, overrides)
    # 7. Devolve a linha já no formato da listagem.
    return _get(config, class_id)


# ENTRADA: PUT /api/classes/<id> (platform_server.Handler._class_change) → edição na seção Classes.
def update_class(config: ProjectConfig, class_id: int, body: dict, author: str | None = None) -> dict:
    """Edita uma classe: nome de exibição, cor e ativo/inativo de qualquer uma; slug e critério só de propostas.

    Parâmetros:
        config: configuração do projeto.
        class_id: ID da classe a editar.
        body: só com chaves de ``allowed``: ``display_name`` (texto), ``color``
            ("#RRGGBB" ou ``None`` para voltar à cor padrão), ``active`` (bool),
            ``name`` (slug, só propostas) e ``criterion`` (texto, só propostas).
        author: revisor que fez a mudança (vai para o ``history``).

    Retorna:
        A linha atualizada da classe, no formato de ``_entry``.

    Erros:
        ValueError "Campos não editáveis: ..." (chave fora de ``allowed``),
        "Nada para alterar.", "active deve ser true ou false.", erros de ``_text``/
        ``_color``/``_slug``, nome técnico repetido, ou tentativa de mudar slug/critério
        de uma classe da taxonomia; ClassNotFound se o ID não existe.

    Como grava:
        - proposta: altera o próprio item em ``proposed`` (``display_name`` vira ``label``,
          ``name`` vira ``slug``);
        - classe da taxonomia: grava em ``classes[<slug>]``; valor ``None`` ou nome igual
          ao original remove a edição (volta ao padrão da taxonomia).
    """
    # 1. Só campos conhecidos.
    # EXTENSAO: campo novo editável pela página = nome aqui + validação abaixo + leitura em _entry.
    allowed = {"display_name", "color", "active", "name", "criterion"}
    unknown = set(body) - allowed
    if unknown:
        raise ValueError(f"Campos não editáveis: {', '.join(sorted(unknown))}.")
    if not body:
        raise ValueError("Nada para alterar.")
    if "active" in body and not isinstance(body["active"], bool):
        raise ValueError("active deve ser true ou false.")
    # 2. Valida os campos que valem para qualquer classe; "changes" guarda só o que foi pedido.
    changes = {}
    if "display_name" in body:
        changes["display_name"] = _text(body, "display_name", MAX_NAME, required=True)
    if "color" in body:
        # color None = pedido explícito para voltar à cor padrão.
        changes["color"] = _color({"color": body["color"]}) if body["color"] is not None else None
    if "active" in body:
        changes["active"] = body["active"]
    with _LOCK:
        # 3. Acha a classe (taxonomia ou proposta) dentro da trava.
        overrides = load_overrides(config)
        item, proposed = _find(config, overrides, class_id)
        # ATENCAO: slug e critério da taxonomia NÃO mudam por aqui; os pesos treinados conferem o slug.
        if ("name" in body or "criterion" in body) and not proposed:
            raise ValueError("Nome técnico e critério das classes da taxonomia só mudam numa nova versão da taxonomia "
                             "(os pesos treinados conferem o nome). Altere o nome de exibição.")
        # 4. Novo slug de proposta: válido e sem repetir nenhum outro (exceto ela mesma).
        if "name" in body:
            slug = _slug(body)
            taken = load_taxonomy(config.taxonomy_path)["classes"] + [value for value in overrides["proposed"] if value is not item]
            if any(value["slug"] == slug for value in taken):
                raise ValueError(f"Já existe uma classe com o nome técnico {slug}.")
            changes["name"] = slug
        if "criterion" in body:
            changes["criterion"] = _text(body, "criterion", MAX_CRITERION)
        stamp = _now()
        # 5. Aplica as mudanças no lugar certo.
        if proposed:
            # Proposta: os nomes de campo da página viram os nomes internos (display_name→label, name→slug).
            mapping = {"display_name": "label", "name": "slug"}
            for key, value in changes.items():
                # Cor None numa proposta = grava a cor padrão do ID (a proposta precisa de uma cor).
                item[mapping.get(key, key)] = value if not (key == "color" and value is None) else default_color(class_id)
            item["updated_at"] = stamp
        else:
            # Classe da taxonomia: edição guardada pelo SLUG em overrides["classes"].
            edits = overrides["classes"].setdefault(item["slug"], {})
            for key, value in changes.items():
                if value is None or (key == "display_name" and value == item["label"]):
                    edits.pop(key, None)  # cor vazia ou nome igual ao original = volta ao padrão da taxonomia
                else:
                    edits[key] = value
            edits["updated_at"] = stamp
        # 6. Histórico e gravação.
        overrides["history"].append({"at": stamp, "action": "update", "id": class_id, "by": author, "changes": changes})
        _save(config, overrides)
    # 7. Devolve a linha relida do disco.
    return _get(config, class_id)


# ENTRADA: DELETE /api/classes/<id> (platform_server.Handler._class_change) → botão "Desativar".
def deactivate_class(config: ProjectConfig, class_id: int, author: str | None = None) -> dict:
    """Exclusão lógica: a classe some das opções de caixa nova; nada é apagado.

    Parâmetros:
        config: configuração do projeto.
        class_id: ID da classe a desativar.
        author: revisor que pediu (vai para o ``history``).

    Retorna: a linha atualizada (``active`` False), formato de ``_entry``.

    Erros: os mesmos de ``update_class`` (ClassNotFound se o ID não existe).

    ATENCAO: não existe exclusão física de classe; reativar = ``update_class`` com ``active: true``.
    """
    return update_class(config, class_id, {"active": False}, author)


# ---------------------------------------------------------------------------
# Página servida: aplica as edições às classes embutidas no index.html
# ---------------------------------------------------------------------------

def apply_to_page_classes(config: ProjectConfig, classes: list[dict]) -> list[dict]:
    """Nome de exibição, cor e ativo/inativo nas classes embutidas na página servida.

    Só as classes ativas da taxonomia entram (as caixas da revisão usam esses
    IDs); um arquivo de edições inválido deixa a página com os dados originais.

    Parâmetros:
        config: configuração do projeto.
        classes: lista ``payload["classes"]`` de ``browser_data.json`` (cada item com
            ``id``, ``slug``, ``label``, ``color``...).

    Retorna:
        Nova lista, na mesma ordem: cada classe da fase 1 com mesmo ID e mesmo slug recebe
        ``label`` (nome de exibição), ``color``, ``active`` e ``requires_training`` da
        listagem; as demais passam sem mudança. A lista de entrada não é modificada.

    Chamada por: ``platform_server.make_handler`` → ``Handler._page`` a cada página servida.

    EXTENSAO: campo editado que deve valer nas caixas da revisão = acrescente-o no dict
    montado no fim do laço (e use-o em platform.js).
    """
    # 1. Classes da taxonomia, fase 1, já com as edições aplicadas, indexadas por ID.
    # Qualquer problema no arquivo de edições: devolve a lista original (a página abre mesmo assim).
    try:
        current = {item["id"]: item for item in list_classes(config)["classes"] if item["source"] == "taxonomia" and item["phase"] == 1}
    except (OSError, ValueError, KeyError):
        return classes
    result = []
    # 2. Copia cada classe da página, aplicando a edição quando ID e slug batem.
    for category in classes:
        entry = current.get(category.get("id"))
        # ATENCAO: confere o slug além do ID. Se a página foi gerada com outra taxonomia (IDs
        # deslocados), aplicar pela posição pintaria a classe errada; nesse caso fica como está.
        # (Classe da página sem "slug" é aceita pelo ID: "or entry['name']".)
        if entry is None or entry["name"] != (category.get("slug") or entry["name"]):
            result.append(category)
            continue
        result.append({**category, "label": entry["display_name"], "color": entry["color"], "active": entry["active"],
                       "requires_training": entry["requires_training"]})
    return result
