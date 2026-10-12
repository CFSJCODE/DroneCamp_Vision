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

O que o arquivo contém (funções públicas):
    - ``short_hash``: código curto de 8 caracteres em maiúsculas.
    - ``humanize``: nome genérico a partir de qualquer slug (``galpao_norte_b``).
    - ``review_name``: nome da inspeção (pasta em ``data/reviews/``).
    - ``run_name`` / ``run_started``: nome e horário de início de uma execução em ``runs/``.
    - ``weights_name``: nome de um arquivo de pesos (``best.pt``, ``yolo26l.pt``).
    - ``comparison_name``: nome de um JSON de comparação/avaliação de modelos.
    - ``dataset_name``: nome de um dataset montado a partir de uma revisão.
    - ``format_datetime``: data e hora ``dd/mm/aaaa hh:mm`` no horário de Brasília.
    - ``generated_file_name``: nome de laudo ou pacote de evidências gerado.
    Funções privadas: ``_is_hash``, ``_word``, ``_split_review``, ``_date``.

Quem chama:
    - ``review_name``: exports.py (texto do laudo e manifesto), laudo_abnt.py
      (``registry_label``), review_render.py (nome mostrado na página de revisão),
      ingest.py (resposta do envio de fotos) e platform_server.py (lista de inspeções).
    - ``run_name``: operations.py (lista de execuções em ``runs/`` na plataforma).
    - As demais (``weights_name``, ``comparison_name``, ``dataset_name``,
      ``generated_file_name``) são exercidas por tests/test_naming.py, que as compara
      com a cópia em platform.js usada pela página.

Disco: este módulo NÃO lê nem grava nada. Só transforma texto em texto.

Onde mexer:
    - Nova etapa de execução em ``runs/`` (ex.: ``distill_<stamp>_<hash>``): acrescente
      em ``STAGE_NAMES`` aqui E em ``STAGES`` no bloco "Nomes legíveis" do platform.js.
    - Novo local de inspeção com nome bonito (ex.: ``galpao_sul`` → "Galpão Sul da Fábrica"):
      acrescente em ``PLACES`` aqui E em ``PLACES`` do platform.js.
    - Palavra de pasta sem acento que deve sair acentuada (ex.: ``vistoria`` → "Vistoria",
      ``agua`` → "Água"): acrescente em ``WORDS`` aqui E em ``WORDS`` do platform.js.
    - Qualificador inteiro com redação própria (ex.: ``revisao_final`` → "Revisão Final
      Aprovada"): acrescente em ``PHRASES`` aqui E no platform.js.
    - Novo tipo de arquivo de comparação (prefixo além de ``compare``/``eval``): veja
      ``comparison_name`` (marcador EXTENSAO) e a função ``comparison`` do platform.js.
    - Novo tipo de arquivo gerado (além de laudo e evidências): veja ``generated_file_name``.
    Depois de qualquer mudança: acrescente o id de exemplo nas listas do topo de
    tests/test_naming.py e rode ``python -m unittest tests.test_naming``.

ATENCAO: o bloco "Nomes legíveis" do platform.js (entre os comentários
"Nomes legíveis" e "fim dos nomes legíveis") precisa continuar IGUAL a este
algoritmo. Toda regra alterada aqui deve ser replicada lá, na mesma ordem de testes,
senão a página e o servidor mostram nomes diferentes para o mesmo id e o teste
``test_page_uses_the_same_names`` falha.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re

# ---------------------------------------------------------------------------
# Tabelas de nomes (espelhadas no bloco "Nomes legíveis" do platform.js)
# ---------------------------------------------------------------------------

# Etapas de ``runs/<etapa>_<AAAAMMDDTHHMMSSZ>_<hash8>``.
# EXTENSAO: tipo novo de pasta em runs/ = nome aqui E no bloco "Nomes legíveis" do
# platform.js. ATENCAO: os dois algoritmos precisam dar o mesmo resultado (tests/test_naming.py).
# Chave = prefixo da pasta (antes do carimbo de data); valor = nome mostrado ao usuário.
# Etapa que não estiver aqui ainda ganha nome, via ``humanize`` (ex.: "Distill"), só que sem tradução.
STAGE_NAMES = {
    "pilot_train": "Treino Piloto", "train": "Treino de Produção", "tune": "Ajuste de Hiperparâmetros",
    "export": "Exportação", "export_demo": "Exportação de Demonstração", "predict_demo": "Predição de Demonstração",
    "autolabel_eval": "Avaliação da Marcação Automática", "calibrator": "Calibrador",
    "feedback_validation": "Validação de Revisão", "review_validation": "Validação de Revisão",
    "review_opening_fix": "Correção de Abertura",
}
# Expressão regular do id de execução: ``<etapa>_<AAAAMMDDTHHMMSSZ>_<hash8>``.
#   - ``stage``: letras minúsculas, dígitos e "_" (o ``+?`` é preguiçoso, então a etapa
#     termina no primeiro "_" seguido de um carimbo válido; ``pilot_train`` fica inteiro);
#   - ``stamp``: data e hora UTC, ex.: ``20261007T010057Z`` (lido por ``run_started``);
#   - ``hash``: exatamente 8 caracteres hexadecimais minúsculos.
# ATENCAO: é a mesma regex de ``RUN_ID`` no platform.js. Pasta que não casa com ela
# (ex.: ``detect``) cai em ``humanize``.
RUN_ID = re.compile(r"^(?P<stage>[a-z0-9_]+?)_(?P<stamp>\d{8}T\d{6}Z)_(?P<hash>[0-9a-f]{8})$")
# Locais conhecidos (prefixo da pasta da inspeção até a versão vN).
# Chave = tudo antes do token de versão (``ceasa_lado_a`` em ``ceasa_lado_a_v1``), já minúsculo.
# Local fora desta tabela é escrito por ``humanize`` (``galpao_norte`` → "Galpão Norte").
# EXTENSAO: local novo com nome próprio = chave aqui E em PLACES do platform.js.
PLACES = {"ceasa": "CEASA Pavilhão A", "ceasa_lado_a": "CEASA Lado A", "ceasa_ladoa": "CEASA Lado A",
          "internet": "Fotos da Internet"}
# Palavras das pastas sem acento → forma escrita. Conectivos ficam minúsculos.
# Também vale para palavra+número (``revisao002`` → "Revisão 002", ver ``WORD_NUMBER``).
# EXTENSAO: palavra nova acentuada ou com grafia própria = chave (minúscula, sem acento)
# aqui E em WORDS do platform.js.
WORDS = {"revisao": "Revisão", "rev": "Revisão", "prerevisao": "Pré-revisão", "sugestoes": "Sugestões",
         "humana": "Humana", "corrigida": "Corrigida", "taxonomia": "Taxonomia", "parcial": "Parcial",
         "piloto": "Piloto", "propostas": "Propostas", "busca": "Busca", "aberta": "Aberta", "modelo": "Modelo",
         "ladoa": "Lado A", "galpao": "Galpão", "pavilhao": "Pavilhão", "inspecao": "Inspeção", "ceasa": "CEASA",
         "claude": "Claude", "base": "Base", "demo": "Demonstração", "evidencias": "Evidências", "laudo": "Laudo"}
# Qualificadores inteiros que pedem uma redação própria (comparados já sem o hash).
# Chave = os tokens depois da versão unidos por "_" (``ceasa_v3_humana_<hash>`` → ``humana``).
# EXTENSAO: frase nova = chave aqui E em PHRASES do platform.js.
PHRASES = {"humana": "Revisão Humana", "corrigida": "Revisão Corrigida", "sugestoes_modelo": "Sugestões do Modelo"}
# Conectivos que ficam minúsculos no meio do nome ("Galpão de Cargas"); na primeira posição
# são capitalizados normalmente. ATENCAO: mesmo conjunto de CONNECTORS no platform.js.
CONNECTORS = {"de", "da", "do", "das", "dos", "e", "a", "o", "em"}
# Palavra seguida de número colado: ``revisao002`` → word="revisao", number="002".
# O ``+?`` preguiçoso garante que todos os dígitos finais vão para ``number``.
WORD_NUMBER = re.compile(r"^(?P<word>[a-z]+?)(?P<number>\d+)$")


# ---------------------------------------------------------------------------
# Peças básicas: código curto, detecção de hash, uma palavra, slug genérico
# ---------------------------------------------------------------------------

def short_hash(value: str) -> str:
    """Código curto mostrado ao lado do nome: 8 primeiros caracteres em maiúsculas.

    Parâmetros:
        value: hash ou id qualquer (``ff34e226416c``); é convertido com ``str``.

    Retorna:
        str com até 8 caracteres em maiúsculas (``"FF34E226"``).

    Chamada por: quase todas as funções deste módulo. Espelho: ``shortHash`` no platform.js.
    """
    return str(value)[:8].upper()


def _is_hash(token: str) -> bool:
    """Trecho hexadecimal de 8+ caracteres (``ff34e226416c``, ``deadbeef``); palavras reais raramente são só a–f.

    Parâmetros:
        token: um pedaço do id já em minúsculas (separado por "_").

    Retorna:
        True se o token tiver 8 ou mais caracteres e só dígitos 0-9 e letras a-f.

    ATENCAO: palavras de 8+ letras só com a–f (raras, ex.: ``deadbeef``) seriam tratadas
    como hash. Espelho: ``isHash`` no platform.js.
    """
    return len(token) >= 8 and re.fullmatch(r"[0-9a-f]+", token) is not None


def _word(token: str, first: bool) -> str:
    """Uma palavra da pasta na forma escrita: dicionário, versão (v6), palavra+número (revisao002).

    Parâmetros:
        token: um pedaço do id, minúsculo (``revisao002``, ``v8``, ``s42``, ``norte``).
        first: True se for a primeira palavra do nome (conectivos não ficam minúsculos no início).

    Retorna:
        str com a palavra escrita para leitura (``"Revisão 002"``, ``"v8"``, ``"Semente 42"``).

    Chamada por: ``humanize``, ``review_name``, ``comparison_name``, ``dataset_name``.
    ATENCAO: a ORDEM dos testes abaixo define o resultado e é a mesma de ``word`` no
    platform.js. Mudar a ordem aqui sem mudar lá quebra a paridade.
    EXTENSAO: regra nova de escrita (ex.: ``l3`` → "Lote 3") entra como mais um ``if``
    nesta sequência, no ponto certo da prioridade, e igual em ``word`` do platform.js.
    """
    # 1. Palavra conhecida do dicionário (acentuação e grafia próprias).
    if token in WORDS:
        return WORDS[token]
    # 2. Token de versão (``v8``, ``v10``, ``v8b``): fica como está, minúsculo.
    if re.fullmatch(r"v\d+[a-z]?", token):
        return token
    # 3. Semente do dataset (``s42``) vira "Semente 42".
    if re.fullmatch(r"s\d+", token):
        return f"Semente {token[1:]}"
    # 4. Número puro (``2026``) fica como está.
    if token.isdigit():
        return token
    # 5. Palavra conhecida + número colado (``revisao002`` → "Revisão 002").
    match = WORD_NUMBER.match(token)
    if match and match["word"] in WORDS:
        return f"{WORDS[match['word']]} {match['number']}"
    # 6. Conectivo no meio do nome fica minúsculo ("Galpão de Cargas").
    if token in CONNECTORS and not first:
        return token
    # 7. Siglas curtas de 1 ou 2 letras viram maiúsculas (``b`` → "B", ``sp`` → "SP").
    if len(token) <= 2 and token.isalpha():
        return token.upper()
    # 8. Caso geral: primeira letra maiúscula, resto como veio.
    return token[:1].upper() + token[1:]


def humanize(slug: str) -> str:
    """Nome genérico: ``galpao_norte_b`` → "Galpão Norte B"; hashes viram código curto.

    Parâmetros:
        slug: qualquer identificador (separadores aceitos: "_", "-" e espaço).

    Retorna:
        str com as palavras escritas por ``_word`` e unidas por espaço; "" se o slug for vazio.

    Chamada por: todas as funções de nome como último recurso, quando o id não segue o
    formato esperado. Espelho: ``humanize`` no platform.js.
    """
    # Divide por "_", "-" ou espaços (um ou mais seguidos) e descarta pedaços vazios.
    tokens = [token for token in re.split(r"[_\-\s]+", str(slug).strip().lower()) if token]
    # Cada token: hash → código curto; senão, ``_word`` (``index == 0`` marca a primeira palavra).
    return " ".join(short_hash(token) if _is_hash(token) else _word(token, index == 0) for index, token in enumerate(tokens))


# ---------------------------------------------------------------------------
# Inspeções (pastas em data/reviews/)
# ---------------------------------------------------------------------------

def _split_review(name: str) -> tuple[str | None, str | None, list[str], str | None]:
    """(local, versão, qualificadores, hash) de ``<local>_v<N>_<qualificadores>_<hash>``.

    Parâmetros:
        name: nome da pasta da inspeção ou do dataset (``ceasa_v8_revisao002_ff34e226416c``).

    Retorna:
        Tupla ``(local, versão, resto, hash)``:
        - local: nome escrito do local (``PLACES`` ou ``humanize``), ex.: "CEASA Pavilhão A";
        - versão: o token de versão (``"v8"``);
        - resto: lista dos tokens entre a versão e o hash (``["revisao002"]``);
        - hash: o hash final, se houver (``"ff34e226416c"``), senão None.
        Se não houver token de versão (ou ele for o primeiro), volta
        ``(None, None, todos_os_tokens, None)`` e quem chama usa ``humanize``.

    Chamada por: ``review_name`` e ``dataset_name``. Espelho: ``splitReview`` no platform.js.
    """
    # 1. Quebra por "_" em minúsculas e descarta pedaços vazios (``a__b``).
    tokens = [token for token in str(name).lower().split("_") if token]
    # 2. Posição do primeiro token de versão (``v8``, ``v10``, ``v8b``); None se não houver.
    position = next((index for index, token in enumerate(tokens) if re.fullmatch(r"v\d+[a-z]?", token)), None)
    # Sem versão, ou versão no início (sem local antes): formato desconhecido.
    if position is None or position == 0:
        return None, None, tokens, None
    # 3. Local = tudo antes da versão; nome próprio em PLACES ou nome genérico.
    place_key = "_".join(tokens[:position])
    place = PLACES.get(place_key) or humanize(place_key)
    # 4. Resto = tudo depois da versão; se o último for hash, sai da lista e vira ``digest``.
    rest = tokens[position + 1:]
    digest = rest.pop() if rest and _is_hash(rest[-1]) else None
    return place, tokens[position], rest, digest


# ENTRADA: nome legível de uma inspeção (usado no laudo, na página de revisão e na lista da plataforma).
def review_name(name: str, display_name: str | None = None) -> str:
    """Nome da inspeção a partir da pasta em ``data/reviews``.

    ``ceasa_v8_revisao002_ff34e226416c`` → "CEASA Pavilhão A v8 · Revisão 002 FF34E226";
    ``ceasa_lado_a_v1`` → "CEASA Lado A v1". ``display_name`` salvo no registro
    (nome digitado no envio pela plataforma) tem prioridade.

    Parâmetros:
        name: nome da pasta da inspeção (a "versão" do registro).
        display_name: nome digitado pelo usuário e salvo no registry.json, ou None.

    Retorna:
        str: ``display_name`` aparado, se houver; senão "<local> <versão> · <qualificador> <HASH>";
        sem versão reconhecível, ``humanize(name)``; nome vazio → "".

    Chamada por: exports.py, laudo_abnt.registry_label, review_render.py, ingest.py,
    platform_server.py. Espelho: ``review`` no platform.js.
    """
    # 1. Nome digitado pelo usuário sempre vence.
    if display_name and str(display_name).strip():
        return str(display_name).strip()
    if not name:
        return ""
    # 2. Decompõe a pasta; sem versão, nome genérico.
    place, version, rest, digest = _split_review(name)
    if version is None:
        return humanize(name)
    # 3. Qualificador: frase própria (PHRASES) para o resto inteiro, senão palavra por palavra.
    qualifier = PHRASES.get("_".join(rest)) or " ".join(_word(token, index == 0) for index, token in enumerate(rest))
    # 4. Monta "<local> <versão>" e, se houver qualificador ou hash, " · <qualificador> <HASH>".
    text = f"{place} {version}"
    tail = " ".join(part for part in (qualifier, short_hash(digest) if digest else "") if part)
    return f"{text} · {tail}" if tail else text


# ---------------------------------------------------------------------------
# Execuções (pastas em runs/) e pesos
# ---------------------------------------------------------------------------

# ENTRADA: nome legível de uma execução em runs/ (lista de treinos da plataforma, operations.py).
def run_name(run_id: str, custom: str | None = None) -> str:
    """``pilot_train_20261007T010057Z_f5e5ba8b`` → "Treino Piloto F5E5BA8B" (nome salvo tem prioridade).

    Parâmetros:
        run_id: nome da pasta em ``runs/``.
        custom: nome dado pelo usuário a essa execução (salvo pela plataforma), ou None.

    Retorna:
        str: ``custom`` aparado, se houver; senão "<etapa traduzida> <HASH8>"; id fora do
        formato ``RUN_ID`` → ``humanize(run_id)``.

    Chamada por: operations.py (lista de execuções) e ``weights_name``. Espelho: ``run`` no platform.js.
    """
    # Nome escolhido pelo usuário vence.
    if custom and str(custom).strip():
        return str(custom).strip()
    match = RUN_ID.match(str(run_id or ""))
    # Pasta fora do padrão (ex.: ``detect``): nome genérico.
    if not match:
        return humanize(run_id or "")
    # Etapa conhecida → STAGE_NAMES; desconhecida → humanize do prefixo.
    stage = STAGE_NAMES.get(match["stage"]) or humanize(match["stage"])
    return f"{stage} {short_hash(match['hash'])}"


def run_started(run_id: str) -> datetime | None:
    """Início (UTC) gravado no id da execução, ou None.

    Parâmetros:
        run_id: nome da pasta em ``runs/`` (``pilot_train_20261007T010057Z_f5e5ba8b``).

    Retorna:
        datetime com fuso UTC lido do carimbo ``AAAAMMDDTHHMMSSZ`` (``%Y%m%dT%H%M%SZ``),
        ou None se o id não seguir ``RUN_ID``.
    """
    match = RUN_ID.match(str(run_id or ""))
    # O carimbo não traz fuso explícito além do "Z"; ``replace(tzinfo=utc)`` marca como UTC.
    return datetime.strptime(match["stamp"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc) if match else None


def weights_name(path: str, base_model: str | None = None) -> str:
    """Pesos: ``runs/<id>/fit/weights/best.pt`` → "Treino Piloto 56038EB9 · melhor época"; ``models/yolo26l.pt`` → "Modelo Base YOLO26l".

    Parâmetros:
        path: caminho do arquivo de pesos (barras "/" ou "\\"; relativo ou absoluto).
        base_model: não é usado pelo corpo atual (mantido na assinatura por compatibilidade).

    Retorna:
        str: pesos de um treino → "<run_name> · melhor época|última época|<arquivo>";
        modelo base YOLO → "Modelo Base YOLO<versão><tamanho>"; outro → ``humanize`` do nome
        sem extensão; caminho vazio → "".

    Espelho: ``weights`` no platform.js.
    """
    # 1. Normaliza separadores do Windows e quebra o caminho em partes.
    parts = str(path or "").replace("\\", "/").split("/")
    # 2. Primeira parte do caminho que é um id de execução (pasta em runs/).
    run = next((part for part in parts if RUN_ID.match(part)), None)
    # 3. Nome do arquivo e nome sem a última extensão (``best.pt`` → ``best``).
    file = parts[-1] if parts else ""
    stem = file.rsplit(".", 1)[0]
    # 4. Pesos produzidos por um treino: nome do treino + qual época.
    # EXTENSAO: outro arquivo de pesos com nome próprio (ex.: ``swa.pt``) entra neste dicionário
    # aqui E no mesmo ponto de ``weights`` no platform.js.
    if run:
        kind = {"best": "melhor época", "last": "última época"}.get(stem, file)
        return f"{run_name(run)} · {kind}"
    # 5. Modelo base Ultralytics: ``yolo26l``, ``yolov8n`` (versão com "v" opcional + tamanho n/s/m/l/x).
    family = re.fullmatch(r"yolo(?P<version>v?\d+)(?P<size>[nsmlx]?)", stem, re.IGNORECASE)
    if family:
        return f"Modelo Base YOLO{family['version'].upper()}{family['size'].lower()}"
    # 6. Qualquer outro arquivo: nome genérico.
    return humanize(stem) if stem else ""


# ---------------------------------------------------------------------------
# Comparações, datasets e datas
# ---------------------------------------------------------------------------

def _date(stamp: str) -> str | None:
    """``20261007`` → "07/10/2026".

    Parâmetros:
        stamp: token com exatamente 8 dígitos ``AAAAMMDD``.

    Retorna:
        str ``dd/mm/aaaa`` (fatias: dia ``[6:8]``, mês ``[4:6]``, ano ``[:4]``), ou None se
        o token não tiver 8 dígitos. Não valida se a data existe. Espelho: ``day`` no platform.js.
    """
    return f"{stamp[6:8]}/{stamp[4:6]}/{stamp[:4]}" if re.fullmatch(r"\d{8}", stamp) else None


def comparison_name(file: str) -> str:
    """``compare_v9_f5e5ba8b_vs_56038eb9_20261007.json`` → "Comparação F5E5BA8B × 56038EB9 · 07/10/2026".

    Parâmetros:
        file: nome ou caminho do JSON de comparação (``compare_...``) ou de avaliação por
            classe (``eval_por_classe_...`` ou ``eval_...``).

    Retorna:
        str "<tipo> <assunto> · <data>": assunto = hashes unidos por " × " ou, sem hashes,
        as palavras restantes (vazio se houver "vs" sem hashes); arquivo de outro tipo →
        ``humanize`` do nome sem ``.json``.

    Espelho: ``comparison`` no platform.js.
    """
    # 1. Só o nome do arquivo, sem pasta e sem ``.json``.
    stem = str(file or "").rsplit("/", 1)[-1]
    stem = stem[:-5] if stem.endswith(".json") else stem
    tokens = [token for token in stem.split("_") if token]
    # 2. Tipo pelo prefixo. ``compare`` descarta 1 token; ``eval_por_classe`` descarta 3;
    #    ``eval`` sem "por_classe" descarta 1.
    # EXTENSAO: prefixo novo de arquivo de modelos (ex.: ``gate_``) = mais um tipo neste
    # teste e no cálculo de quantos tokens descartar, aqui E em ``comparison`` do platform.js.
    if tokens and tokens[0] in ("compare", "eval"):
        kind = "Comparação" if tokens[0] == "compare" else "Avaliação por Classe"
        tokens = tokens[1:] if tokens[0] == "compare" else tokens[3:] if tokens[1:3] == ["por", "classe"] else tokens[1:]
    else:
        return humanize(stem)
    # 3. Último token com 8 dígitos é a data; sai da lista.
    date = _date(tokens.pop()) if tokens and _date(tokens[-1]) else None
    # 4. Os hashes dos pesos comparados viram códigos curtos ("F5E5BA8B × 56038EB9").
    hashes = [short_hash(token) for token in tokens if _is_hash(token)]
    if hashes:
        subject = " × ".join(hashes)
    else:
        # Sem hashes: as palavras restantes (``v8`` etc.); se houver "vs", nada (assunto vazio).
        subject = " ".join(_word(token, index == 0) for index, token in enumerate(tokens if "vs" not in tokens else []))
    # 5. Junta "<tipo> <assunto>" e a data com " · ", ignorando partes vazias.
    return " · ".join(part for part in (f"{kind} {subject}".strip(), date) if part)


def dataset_name(name: str) -> str:
    """``ceasa_v8_piloto_s42`` → "Dataset Piloto CEASA Pavilhão A v8 · Semente 42".

    Parâmetros:
        name: nome da pasta do dataset (``<local>_v<N>_<qualificadores>_s<semente>``).

    Retorna:
        str "Dataset [Piloto] <local> <versão> · <qualificadores> · Semente N · <HASH>"
        (partes vazias omitidas); sem versão → "Dataset <humanize(name)>".

    Espelho: ``dataset`` no platform.js.
    """
    place, version, rest, digest = _split_review(name)
    if version is None:
        return f"Dataset {humanize(name)}"
    # Semente (``s42``) e a palavra ``piloto`` saem do qualificador e viram partes próprias.
    seed = next((token for token in rest if re.fullmatch(r"s\d+", token)), None)
    words = [token for token in rest if token not in ("piloto", seed)]
    kind = "Dataset Piloto" if "piloto" in rest else "Dataset"
    qualifier = " ".join(_word(token, index == 0) for index, token in enumerate(words))
    parts = [f"{kind} {place} {version}", qualifier, f"Semente {seed[1:]}" if seed else "", short_hash(digest) if digest else ""]
    return " · ".join(part for part in parts if part)


def format_datetime(value: datetime | str | None) -> str:
    """Data legível ``dd/mm/aaaa hh:mm`` (horário de Brasília, UTC−3) para textos gerados no servidor.

    Parâmetros:
        value: datetime (com ou sem fuso), texto ISO 8601 (``2026-10-07T01:00:57Z``) ou None.

    Retorna:
        str ``dd/mm/aaaa hh:mm`` em UTC−3; "" para valor vazio; o próprio texto se ele não
        for ISO válido. datetime sem fuso é tratado como UTC.

    ATENCAO: o fuso é fixo em −3 h (sem horário de verão). Não há espelho deste no teste de paridade.
    """
    if not value:
        return ""
    # Texto: ``fromisoformat`` não aceita "Z" em versões antigas do Python, por isso a troca por +00:00.
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    # Sem fuso → assume UTC; depois converte para UTC−3 e formata.
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone(timedelta(hours=-3))).strftime("%d/%m/%Y %H:%M")


# ---------------------------------------------------------------------------
# Arquivos gerados (laudos e pacotes de evidências)
# ---------------------------------------------------------------------------

def generated_file_name(file_name: str) -> tuple[str, str | None]:
    """Laudo ou pacote gerado: (nome, data ISO) a partir de ``<AAAAMMDDTHHMMSSZ>_laudo_<inspeção>``.

    Parâmetros:
        file_name: nome da pasta/arquivo gerado (``20261011T140301Z_laudo_ceasa_v8_...`` ou
            ``20261011T140301Z_evidencias_ceasa_v4.zip``).

    Retorna:
        Tupla ``(nome, data_iso)``: ("Laudo Técnico" | "Pacote de Evidências", data UTC em ISO
        8601); fora do padrão → (``humanize`` do nome sem extensão, None).

    Espelho: ``generated`` no platform.js (o teste compara a tupla como lista).
    """
    # Regex: carimbo UTC no início, seguido do tipo ("laudo" ou "evidencias") e de "_".
    # EXTENSAO: tipo novo de arquivo gerado = acrescente na alternância ``laudo|evidencias``
    # e o nome em ``kind`` abaixo, aqui E em ``generated`` do platform.js.
    match = re.match(r"^(?P<stamp>\d{8}T\d{6}Z)_(?P<kind>laudo|evidencias)_", str(file_name or ""))
    if not match:
        return humanize(str(file_name).rsplit(".", 1)[0]), None
    when = datetime.strptime(match["stamp"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    kind = "Laudo Técnico" if match["kind"] == "laudo" else "Pacote de Evidências"
    return kind, when.isoformat()
