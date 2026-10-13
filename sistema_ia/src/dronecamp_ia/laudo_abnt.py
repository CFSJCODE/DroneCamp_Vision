"""Laudo técnico de inspeção de cobertura com drone, no formato ABNT e no estilo do laudo CEASA.

Função no projeto: ``exports.build_report`` monta os dados do laudo (``laudo.json``)
a partir da revisão; este módulo guarda o texto fixo do documento (normas,
metodologia, escalas de classificação, perguntas da lista de verificação) e
desenha o mesmo conteúdo em HTML (A4, autossuficiente) e em PDF (reportlab).

Estrutura (ABNT NBR 10719 para a apresentação do relatório técnico, ABNT NBR 16747
para o conteúdo do laudo de inspeção, ABNT NBR 6024/6027/6023 para seções,
sumário e referências), seguindo o laudo de referência do projeto
(``CEASA Minas - Pavilão A - Laudo.pdf``):

    capa · folha de rosto · sumário
    1 Identificação do laudo        2 Objetivo
    3 Normas e documentos aplicáveis 4 Metodologia (campo, processamento com IA,
                                       critérios de classificação, limitações)
    5 Descrição do imóvel / área inspecionada
    6 Registro fotográfico e análise de conformidades (um item por categoria, como
      as perguntas do laudo CEASA: constatação, figuras, subsídio de IA,
      Conformidade / Risco / Recomendação / Prioridade)
    7 Quadro-resumo  8 Recomendações gerais  9 Considerações finais
    10 Responsabilidade técnica  · Referências · Anexos A–E

Regra que não muda: conformidade, risco (gravidade), diagnóstico, recomendação e
prioridade são julgamentos do responsável técnico. Aqui eles saem sempre como
``PENDING_TEXT``; a gravidade de referência da taxonomia e o texto do laudo CEASA
aparecem só no Anexo E, identificados como "não aplicados". Descrições de modelos
de visão e linguagem (``ai_notes.json``) entram como subsídio pendente, nunca como
constatação. Dados do imóvel não informados saem como "não informado" (nada de
área, inclinação ou extensão estimadas pelo sistema).

Quando mexer: lista de normas (``NORMATIVE_REFERENCES``; confirme a edição
vigente no Catálogo ABNT), perguntas por categoria (``CHECKLIST``), escalas
(``CONFORMITY_SCALE``, ``PRIORITY_SCALE``) ou a aparência do HTML/PDF.
O que o arquivo contém (por seção, na ordem do arquivo):
    - Constantes de texto curtas: ``PENDING_TEXT``, ``PRELIMINARY_TEXT``, ``NOT_INFORMED``,
      ``DEFAULT_TITLE``, ``AI_NOTES_FILE``.
    - Texto fixo do documento: ``NORMATIVE_REFERENCES`` (seção 3 e Referências),
      ``CONFORMITY_SCALE`` e ``PRIORITY_SCALE`` (seção 4.3), ``CHECKLIST`` (perguntas da
      seção 6 e referências do Anexo E), ``MANUAL_CHECKLIST`` (seção 6.x "Itens não
      avaliados"), ``LIMITATIONS`` (seção 4.4), ``GENERAL_RECOMMENDATIONS`` (seção 8),
      ``DETAIL_FIELDS`` / ``PROPERTY_FIELDS`` (dados digitados pelo usuário; seções 1, 4.1, 5, 10).
    - ``normalize_details``: valida e limpa os dados técnicos digitados.
    - ``reference_text``: uma norma no formato de referência da NBR 6023.
    - Subsídios de IA: ``load_ai_notes``, ``match_ai_note``, ``ai_summary`` (lêem ``ai_notes.json``).
    - HTML: ``registry_label``, ``detector_step``, ``table_of_contents``, ``HTML_STYLE`` e
      ``render_html`` (mais os auxiliares privados ``_esc``, ``_or_blank``, ``_pct``,
      ``_rows``, ``_pending_cell``, ``_evaluation_html``, ``_figure_html``).
    - PDF: ``write_pdf`` (reportlab, importado só dentro da função).

Quem chama:
    - exports.py (``build_report``/``_build_report_files``/``_report_items``): usa
      ``normalize_details``, ``CHECKLIST``, ``match_ai_note``, ``load_ai_notes``,
      ``ai_summary``, ``NORMATIVE_REFERENCES``, ``MANUAL_CHECKLIST``, ``LIMITATIONS``,
      ``DEFAULT_TITLE``, ``render_html`` e ``write_pdf``.
    - platform_server.py (``_report_details``): usa ``DETAIL_FIELDS`` para recusar campos
      desconhecidos vindos do formulário "Dados técnicos" da plataforma.
    - Fluxo completo: botão "Gerar laudo" na plataforma → rota de laudo do platform_server
      → ``exports.build_report`` → ``render_html`` (laudo.html) e ``write_pdf`` (laudo.pdf).

Disco: lê ``ai_notes.json`` ao lado do ``registry.json`` (``load_ai_notes``) e grava o PDF
em ``write_pdf``. O ``laudo.html`` é gravado por exports.py com o texto de ``render_html``.

Onde mexer:
    - Mudar o TEXTO de uma seção: procure "Seção N" (ex.: "Seção 4 Metodologia") em
      ``render_html`` (HTML) e em ``write_pdf`` (PDF). Os dois formatos têm o texto
      escrito separadamente: altere nos DOIS.
    - Acrescentar uma SEÇÃO nova: (1) entrada em ``table_of_contents`` (sumário do HTML);
      (2) ``parts.append('<section id="...">...')`` em ``render_html`` no ponto da ordem,
      com o mesmo ``id`` da âncora; (3) ``heading("...")`` + conteúdo em ``write_pdf``, no
      mesmo ponto (o sumário do PDF é montado sozinho a partir de ``heading``); (4) se o
      texto depende de dados novos, acrescente a chave em ``exports.build_report`` (laudo.json).
    - Nova norma: ``NORMATIVE_REFERENCES``. Nova pergunta por categoria: ``CHECKLIST``.
    - Novo campo digitado pelo usuário: ``DETAIL_FIELDS`` (ver o EXTENSAO dele).
    - Aparência do HTML: ``HTML_STYLE``. Aparência do PDF: estilos ``ParagraphStyle`` no
      início de ``write_pdf``.
    Teste: ``python -m unittest tests.test_exports`` gera HTML e PDF.

ATENCAO: nenhuma função aqui pode preencher conformidade, risco, diagnóstico,
recomendação ou prioridade com valor calculado: todos saem como ``PENDING_TEXT``.
"""

from __future__ import annotations

import base64
import html
import io
import json
import math
from pathlib import Path

from .naming import review_name

# ---------------------------------------------------------------------------
# Textos curtos usados em todo o documento
# ---------------------------------------------------------------------------

# Texto que ocupa todo campo de julgamento técnico (conformidade, risco, diagnóstico,
# recomendação, prioridade, ART ausente). ATENCAO: exports.py importa esta constante e a
# grava no laudo.json; mudar a frase muda os dois formatos e o JSON.
PENDING_TEXT = "pendente de validação técnica"
# Situação do documento (capa, folha de rosto, rodapé do PDF, metadado "subject" do PDF).
PRELIMINARY_TEXT = "preliminar, sujeito a validação por responsável técnico"
# Valor mostrado quando o usuário deixou um campo de DETAIL_FIELDS em branco (ver _or_blank).
NOT_INFORMED = "não informado"
# Título padrão do laudo quando a plataforma não manda um título (usado por exports.build_report).
DEFAULT_TITLE = "Laudo técnico de inspeção de cobertura com uso de drone"
# Nome do arquivo de descrições do VLM (gravado por vlm_assist.py ao lado do registry.json).
AI_NOTES_FILE = "ai_notes.json"

# ---------------------------------------------------------------------------
# Texto fixo do documento.
# ---------------------------------------------------------------------------

# Seção 3 (tabela de normas) e REFERÊNCIAS (via reference_text, em ordem alfabética).
# Cada item: code (sigla e número, ex. "ABNT NBR 16747"), year (int ou None), title (título
# oficial) e use (frase "Aplicação neste laudo" da tabela da seção 3).
# ATENCAO: o prefixo de "code" decide a entidade em reference_text ("ABNT ", "ANAC " ou BRASIL).
# Normas citadas (seção 3 e Referências no formato da NBR 6023). "ano" só onde a
# edição é estável; sem ano, a referência sai sem data e o Anexo E pede conferência.
# EXTENSAO: norma nova = mais um dict nesta lista (mesmas quatro chaves). Aparece sozinha na
# seção 3, nas Referências e no Anexo E (aviso de edição) dos dois formatos, porque exports.py
# copia esta lista para report["normative_references"].
NORMATIVE_REFERENCES = [
    {"code": "ABNT NBR 16747", "year": 2020, "title": "Inspeção predial — Diretrizes, conceitos, terminologia, requisitos e procedimento",
     "use": "Conteúdo do laudo de inspeção, classificação das irregularidades e organização das prioridades (patamares de urgência)."},
    {"code": "ABNT NBR 5674", "year": None, "title": "Manutenção de edificações — Requisitos para o sistema de gestão de manutenção",
     "use": "Inserção das ações recomendadas no programa de manutenção preventiva e corretiva da edificação."},
    {"code": "ABNT NBR 7196", "year": None, "title": "Telhas de fibrocimento sem amianto — Execução de coberturas e fechamentos laterais",
     "use": "Referência de execução para telhas, fixações, recobrimentos e acessórios de fibrocimento, junto às especificações do fabricante."},
    {"code": "ABNT NBR 15575-5", "year": None, "title": "Edificações habitacionais — Desempenho — Parte 5: Requisitos para os sistemas de coberturas",
     "use": "Referência de desempenho (estanqueidade, durabilidade) para o sistema de cobertura; aplicação subsidiária em edificação não habitacional."},
    {"code": "ABNT NBR 10844", "year": 1989, "title": "Instalações prediais de águas pluviais — Procedimento",
     "use": "Referência para calhas, condutores e escoamento das águas pluviais."},
    {"code": "ABNT NBR 16280", "year": None, "title": "Reforma em edificações — Sistema de gestão de reformas — Requisitos",
     "use": "Planejamento e execução das intervenções de reparo decorrentes deste laudo."},
    {"code": "ABNT NBR 13752", "year": None, "title": "Perícias de engenharia na construção civil",
     "use": "Terminologia e conduta técnica na elaboração de laudos de engenharia."},
    {"code": "ABNT NBR 10719", "year": 2015, "title": "Informação e documentação — Relatório técnico e/ou científico — Apresentação",
     "use": "Estrutura e apresentação do documento (elementos pré-textuais, textuais e pós-textuais)."},
    {"code": "ABNT NBR 6023", "year": 2018, "title": "Informação e documentação — Referências — Elaboração",
     "use": "Formato das referências."},
    {"code": "ABNT NBR 6024", "year": 2012, "title": "Informação e documentação — Numeração progressiva das seções de um documento — Apresentação",
     "use": "Numeração das seções."},
    {"code": "ABNT NBR 6027", "year": 2012, "title": "Informação e documentação — Sumário — Apresentação",
     "use": "Sumário."},
    {"code": "ANAC RBAC-E nº 94", "year": 2017, "title": "Requisitos gerais para aeronaves não tripuladas de uso civil",
     "use": "Operação da aeronave não tripulada (drone) na coleta das imagens."},
    {"code": "Lei nº 6.496", "year": 1977, "title": "Institui a Anotação de Responsabilidade Técnica (ART) na prestação de serviços de engenharia",
     "use": "Emissão do laudo com ART pelo profissional habilitado."},
]

# Seção 4.3 (Critérios de classificação), duas tabelas: (nome, critério).
# ATENCAO: são só as escalas impressas para o responsável técnico usar; o sistema nunca
# escolhe um valor delas. EXTENSAO: novo nível = mais uma tupla (aparece em HTML e PDF).
# Escala do laudo CEASA (conformidade) e patamares de urgência da NBR 16747.
CONFORMITY_SCALE = [
    ("Conforme", "Atende às especificações do fabricante e às normas aplicáveis."),
    ("Não conformidade baixa", "Irregularidade sem perda de desempenho imediata; correção programável."),
    ("Não conformidade média", "Irregularidade com perda parcial de desempenho ou risco de agravamento."),
    ("Não conformidade grave", "Irregularidade que compromete o desempenho do sistema (estanqueidade, escoamento, fixação)."),
    ("Não conformidade gravíssima", "Irregularidade com risco imediato à edificação, aos usuários ou à operação."),
    ("Não é possível constatar", "As imagens não permitem avaliar o item; requer verificação in loco."),
    ("Não se aplica", "O item não existe ou não é pertinente à edificação."),
]
# Patamares de urgência da ABNT NBR 16747 (tabela "Prioridade" da seção 4.3).
PRIORITY_SCALE = [
    ("Prioridade 1", "Ações necessárias quando a perda de desempenho compromete a saúde ou a segurança dos usuários, ou a "
                     "funcionalidade dos sistemas, com risco de paralisação."),
    ("Prioridade 2", "Ações necessárias quando a perda de desempenho compromete parcialmente a funcionalidade, sem prejuízo "
                     "à operação direta dos sistemas nem à segurança."),
    ("Prioridade 3", "Ações que podem ser planejadas e programadas, sem comprometimento da vida útil ou da durabilidade."),
]

# Formato: slug da categoria da taxonomia → (pergunta, item do laudo CEASA, conformidade
# dada pelo laudo CEASA, recomendação do laudo CEASA). Use None nos três últimos quando o
# laudo CEASA não tem item equivalente (aí o Anexo E mostra "—").
# Uso: exports._report_items lê a pergunta (título em negrito de cada item 6.N e da tabela
# 6.x "Categorias sem ocorrência") e guarda os outros três em item["model_reference"].
# Pergunta da lista de verificação por categoria (redação do laudo CEASA, item entre parênteses) e
# o texto que aquele laudo deu ao caso dele. A referência só aparece no Anexo E, como apoio ao
# responsável técnico; nunca é aplicada às ocorrências deste laudo.
# EXTENSAO: categoria nova na taxonomia = acrescente o slug aqui (o mesmo de
# configs/taxonomy.json). Sem entrada, exports.py usa a pergunta genérica
# "Há ocorrências de <rótulo>?" e o Anexo E fica sem referência.
# ATENCAO: a chave precisa ser exatamente o "slug" da taxonomia, senão a entrada é ignorada.
# Taxonomia v7 (13/10/2026): classes fundidas. A pergunta cobre todas as classes v6 que
# formam a v7; a referência do laudo CEASA é a da classe v6 dominante (a de mais caixas),
# e os itens das demais ficam citados na pergunta para o responsável técnico.
CHECKLIST = {
    "telha_danificada": ("Existem telhas quebradas, trincadas ou furadas (inclusive translúcidas)?", "7", "Não conformidade gravíssima",
                         "Substituir as telhas quebradas por telhas novas."),
    "telha_ausente": ("É possível perceber ausência de telhas na área da cobertura?", "8", "Não conformidade gravíssima", "Instalar uma nova telha."),
    "residuos_cobertura": ("Existe acúmulo de resíduos sobre as telhas?", "10", "Não conformidade média", "Limpar o telhado removendo todos os resíduos."),
    "objeto_solto_cobertura": ("Há pedaços de telha, de calha, de rufo ou outros objetos soltos sobre a cobertura (itens 10 e 32)?", "10",
                               "Não conformidade média", "Remover os fragmentos e objetos junto com a limpeza do telhado."),
    "calha_obstruida": ("Há acúmulo de resíduos ou vegetação nas calhas e nas saídas (itens 28 e 29)?", "28", "Não conformidade gravíssima",
                        "Limpeza manual das calhas, remoção da vegetação, lavagem dos tubos de queda e inspeção das caixas de areia."),
    "rufo_ausente": ("É possível perceber ausência de rufos ou cumeeiras na área da cobertura?", "20", "Não conformidade grave",
                     "Instalar rufos nos trechos ausentes."),
    "rufo_danificado": ("Há rufos ou cumeeiras danificados, deslocados ou desalinhados?", "21", "Não conformidade gravíssima",
                        "Substituir os rufos danificados."),
    "reparo_material_aplicado": ("Existem reparos ou material aplicado em telhas, rufos ou fixadores? Que material ou técnica foi utilizada (itens 13 e 18)?",
                                 "13", "Não conformidade baixa",
                                 "Realizar teste de estanqueidade nos reparos; substituir as telhas com reparos em excesso ou refazer os reparos com falha."),
    "fixador_irregular": ("Existem parafusos de fixação de telhas ou rufos frouxos, faltando ou enferrujados (itens 15, 16, 17, 22 e 26)?", "17",
                          "Não conformidade média", "Apertar, repor ou substituir os fixadores conforme o fabricante."),
    "calha_danificada": ("As calhas estão danificadas?", None, None, None),
    "telha_deslocada": ("A sobreposição das telhas está correta em relação às especificações dos fabricantes?", "4",
                        "Não conformidade média", "Remover a telha antiga e fixar a telha nova conforme o fabricante."),
    "corrosao_metalica": ("Há corrosão nas superfícies metálicas da cobertura?", None, None, None),
    "vedacao_falha": ("Há falhas ou ausências de selante em juntas de rufos, fixadores ou passagens pela telha (itens 23, 24 e 34)?", "23",
                      "Não conformidade média", "Aplicar selante nas juntas, nos fixadores e nas passagens e testar a estanqueidade."),
    "agua_empocada": ("Há água empoçada visível sobre telhas, calhas ou lajes?", None, None, None),
    "crescimento_biologico": ("Há musgo, líquen ou bolor sobre a cobertura?", None, None, None),
}

# Seção 6, último subitem "Itens não avaliados pela detecção automática" (HTML e PDF).
# EXTENSAO: pergunta nova que o sistema não cobre = mais uma string nesta lista.
# Itens da lista de verificação do laudo CEASA que a detecção automática não cobre: entram como
# "não avaliados pelo sistema" (a NBR 16747 pede a lista do que não foi inspecionado).
MANUAL_CHECKLIST = [
    "Qual é o tipo de telha utilizada na cobertura?",
    "Qual é a área total da cobertura?",
    "A inclinação do telhado está correta?",
    "Existem tirantes de contraventamento nas telhas? Estão dispostos de maneira correta?",
    "Há incompatibilidade nas peças utilizadas no telhado?",
    "Existem trincas ou furos nas telhas?",
    "Há sinais de acúmulo de água nas telhas ou nas calhas?",
    "Os parafusos de fixação das telhas estão nas posições corretas?",
    "As calhas estão danificadas?",
    "Existem saídas de chaminés ou exaustores no telhado? As juntas estão vedadas?",
    "Existem árvores, construções ou obras vizinhas mais altas que a cobertura?",
]

# Seção 4.4 Limitações (lista numerada). exports.py importa esta lista como METHODOLOGY e a
# grava no laudo.json em "methodology" e "limitations"; render_html e write_pdf leem
# report["limitations"].
# EXTENSAO: limitação nova = mais uma string aqui (aparece nos dois formatos e no LEIA-ME).
# Limitações (seção 4.4 e LEIA-ME). Nada aqui é inferido dos dados.
LIMITATIONS = [
    "As fotografias foram congeladas por hash SHA-256 no registro de revisão; cada figura deste documento foi gerada a "
    "partir do arquivo original conferido contra esse hash.",
    "As demarcações resultam da revisão humana na plataforma (origem \"revisão humana\") ou do registro de propostas "
    "(origem \"registro\"). Demarcações de origem \"registro\" não contam com decisão humana nesta emissão.",
    "As sugestões automáticas do detector vêm de um modelo piloto não validado. Não constituem ocorrências: aparecem "
    "somente no Anexo B e aguardam decisão humana.",
    "Descrições geradas por modelo de visão e linguagem, quando presentes, são subsídio à revisão e não constituem "
    "constatação técnica.",
    "Não houve medição física, avaliação hidráulica, estrutural ou de estanqueidade. Nenhuma gravidade, causa, área, "
    "extensão em metros ou diagnóstico foi inferida automaticamente; esses campos ficam pendentes para o responsável "
    "técnico.",
    "Demarcações em fotografias semelhantes (mesma cena, recortes ou ângulos próximos) podem registrar o mesmo elemento "
    "físico mais de uma vez. As contagens referem-se a demarcações por fotografia, não a defeitos físicos únicos.",
    "Este laudo foi elaborado com base em inspeção visual aérea e não substitui inspeções presenciais in loco nem laudo "
    "assinado por profissional habilitado com ART.",
]

# Seção 8 Recomendações gerais (lista com marcadores), lida direto daqui pelos dois formatos.
# EXTENSAO: recomendação geral nova = mais uma string aqui.
# ATENCAO: só textos genéricos; recomendações por item são do responsável técnico (pendentes).
GENERAL_RECOMMENDATIONS = [
    "As ações corretivas e a sua ordem de execução serão definidas pelo responsável técnico conforme os patamares de "
    "prioridade da ABNT NBR 16747, a partir das constatações da seção 6.",
    "As intervenções decorrentes deste laudo devem ser planejadas e executadas conforme a ABNT NBR 16280 e incorporadas "
    "ao programa de manutenção da edificação (ABNT NBR 5674).",
    "Recomenda-se registro fotográfico completo pós-reparo e vistoria periódica por imagens aéreas para acompanhamento.",
]

# Campos do imóvel e da inspeção que o usuário pode informar (chave → rótulo). Nada é calculado.
# EXTENSAO: campo novo no laudo = (1) chave aqui; (2) input com data-detail="<chave>" no
# formulário "Dados técnicos" de index.html; (3) onde ele aparece em render_html e write_pdf.
# Chave fora desta lista é recusada pelo servidor (normalize_details).
# Onde cada chave aparece: report_number (folha de rosto, seção 1), inspection_date (1, 4.1),
# requester (capa, rosto, 1), address (capa, 1), roof_type/slope/support_structure/
# gutter_type/area (seção 5, via PROPERTY_FIELDS), equipment e flight_description (4.1),
# company (capa), responsible_* e art_number (rosto, 1, 10 via exports "technical_responsibility").
DETAIL_FIELDS = {
    "report_number": "Número do laudo",
    "inspection_date": "Data da inspeção",
    "requester": "Solicitante",
    "address": "Endereço",
    "roof_type": "Tipo de cobertura",
    "slope": "Inclinação média do telhado",
    "support_structure": "Estrutura de apoio",
    "gutter_type": "Tipo de calha",
    "area": "Área total estimada",
    "equipment": "Equipamento utilizado",
    "flight_description": "Descrição dos voos",
    "company": "Empresa responsável pela inspeção",
    "responsible_name": "Responsável técnico",
    "responsible_title": "Título profissional",
    "responsible_registry": "Registro profissional (CREA/CAU)",
    "art_number": "ART nº",
}
# Chaves de DETAIL_FIELDS que formam a tabela da seção 5 (Descrição do imóvel), nesta ordem.
# EXTENSAO: novo dado do imóvel na seção 5 = chave em DETAIL_FIELDS e também aqui.
PROPERTY_FIELDS = ("roof_type", "slope", "support_structure", "gutter_type", "area")


# ---------------------------------------------------------------------------
# Validação dos dados digitados e formato das referências
# ---------------------------------------------------------------------------

# ENTRADA: exports.build_report chama esta função antes de montar o laudo.json.
def normalize_details(details: dict | None, *, client: str = "", inspector: str = "") -> dict[str, str]:
    """Só texto informado pelo usuário (aparado, até 500 caracteres); chaves desconhecidas são recusadas.

    Parâmetros:
        details: dict chave → valor vindo do formulário "Dados técnicos" (ou None). Chaves
            permitidas: as de ``DETAIL_FIELDS``. Valores: str, int ou float.
        client: nome do cliente digitado no envio; usado se "requester" vier vazio.
        inspector: nome do inspetor; usado se "responsible_name" vier vazio.

    Retorna:
        dict com TODAS as chaves de ``DETAIL_FIELDS`` (as ausentes viram ""), valores como
        texto aparado e cortado em 500 caracteres.

    Erros:
        ValueError("details deve ser um dicionário.") se não for dict;
        ValueError("Campo desconhecido nos dados do laudo: <chave>.") para chave fora da lista;
        ValueError("Valor inválido em <chave>.") para lista, dict, bool etc.

    Chamada por: exports.build_report.
    """
    details = details or {}
    if not isinstance(details, dict):
        raise ValueError("details deve ser um dicionário.")
    # 1. Recusa chave desconhecida (ex.: erro de digitação em "art_numbr" não pode sumir em silêncio).
    unknown = set(details) - set(DETAIL_FIELDS)
    if unknown:
        raise ValueError(f"Campo desconhecido nos dados do laudo: {sorted(unknown)[0]}.")
    # 2. Percorre todas as chaves conhecidas, na ordem de DETAIL_FIELDS.
    values = {}
    for key in DETAIL_FIELDS:
        value = details.get(key)
        if value is None:
            value = ""
        # bool é subclasse de int em Python, por isso a exclusão explícita.
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise ValueError(f"Valor inválido em {key}.")
        values[key] = str(value).strip()[:500]
    # 3. Padrões: solicitante = cliente e responsável técnico = inspetor, se o usuário não preencheu.
    values["requester"] = values["requester"] or client.strip()
    values["responsible_name"] = values["responsible_name"] or inspector.strip()
    return values


def reference_text(norm: dict) -> str:
    """Referência no formato da NBR 6023 (entidade. Código: título. Local: editora, ano.).

    Parâmetros:
        norm: um item de ``NORMATIVE_REFERENCES`` (chaves code, title, year).

    Retorna:
        str, ex.: "ASSOCIAÇÃO BRASILEIRA DE NORMAS TÉCNICAS. NBR 16747: Inspeção predial — ...
        Rio de Janeiro: ABNT, 2020." Sem ano, termina em "ABNT." (sem data).

    Chamada por: render_html e write_pdf (seção REFERÊNCIAS, também como chave de ordenação).
    """
    code, title, year = norm["code"], norm["title"], norm.get("year")
    # Entidade e local pelo prefixo do código; ``code[5:]`` tira "ABNT " / "ANAC " da frente.
    # EXTENSAO: outra entidade (ex.: "ISO ", "NR ") = mais um elif aqui com o texto da NBR 6023.
    if code.startswith("ABNT "):
        text = f"ASSOCIAÇÃO BRASILEIRA DE NORMAS TÉCNICAS. {code[5:]}: {title}. Rio de Janeiro: ABNT"
    elif code.startswith("ANAC "):
        text = f"AGÊNCIA NACIONAL DE AVIAÇÃO CIVIL. {code[5:]}: {title}. Brasília, DF: ANAC"
    else:
        text = f"BRASIL. {code}: {title}. Brasília, DF"
    # Ano só quando conhecido (year=None → referência sem data).
    return f"{text}, {year}." if year else f"{text}."


# ---------------------------------------------------------------------------
# Subsídios de IA (ai_notes.json do vlm_assist, quando existir).
# ---------------------------------------------------------------------------

def _round(value: float) -> int:
    """Arredonda meio para cima (2.5 → 3), como ``Math.round`` do JS.

    Parâmetros:
        value: coordenada em pixels (float ou int).

    Retorna:
        int arredondado. ATENCAO: ``round`` do Python arredonda 2.5 para 2 (bancário); aqui
        precisa ser meio para cima para casar as coordenadas da caixa humana com as gravadas
        em ai_notes.json.
    """
    return math.floor(float(value) + 0.5)


def load_ai_notes(registry_path: Path, registry_sha256: str) -> tuple[dict | None, str | None]:
    """``ai_notes.json`` ao lado do registro, só se for desta versão exata do registro.

    Parâmetros:
        registry_path: caminho do ``registry.json`` da revisão.
        registry_sha256: SHA-256 atual desse registry.json.

    Retorna:
        Tupla ``(notas, aviso)``: ``(dict, None)`` se o arquivo existir, for legível e tiver
        ``registry_sha256`` igual; ``(None, None)`` se não existir; ``(None, "<motivo>")`` se
        estiver ilegível ou for de outra versão do registro (o aviso vai para o Anexo C).

    Chamada por: exports._build_report_files.
    ATENCAO: o hash garante que descrições feitas para caixas antigas não entrem num laudo
    de uma revisão já alterada. Rode de novo o vlm_assist depois de mudar a revisão.
    """
    path = Path(registry_path).parent / AI_NOTES_FILE
    # Sem arquivo: laudo segue sem subsídio de IA, sem aviso.
    if not path.is_file():
        return None, None
    # Lê o JSON; erro de leitura ou JSON inválido vira aviso, nunca erro do laudo.
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, f"{AI_NOTES_FILE} ilegível; ignorado."
    # Confere formato mínimo e se o arquivo é da MESMA versão do registro (hash).
    if not isinstance(data, dict) or data.get("registry_sha256") != registry_sha256 or not isinstance(data.get("images"), dict):
        return None, f"{AI_NOTES_FILE} ao lado do registro pertence a outra versão e foi ignorado."
    return data, None


def match_ai_note(notes: dict | None, image_sha256: str, box: dict) -> dict | None:
    """Descrição do VLM para esta caixa humana (mesma classe e coordenadas arredondadas), se houver.

    Parâmetros:
        notes: conteúdo de ai_notes.json (de ``load_ai_notes``) ou None.
        image_sha256: hash da foto (chave de ``notes["images"]``).
        box: dict com ``class_id`` (int) e ``bbox_xyxy`` ([x1, y1, x2, y2] em pixels da foto original).

    Retorna:
        O dict da entrada do ai_notes.json que casa (com "description", "model_used" etc.), ou None.

    Chamada por: exports._report_items (subsídio de IA de cada item 6.N).
    """
    if not notes:
        return None
    # Uma entrada casa quando: veio de caixa humana, tem descrição, mesma classe e as 4
    # coordenadas xyxy iguais depois de arredondar (``_round``) dos dois lados.
    for entry in notes["images"].get(image_sha256, []) or []:
        if (isinstance(entry, dict) and entry.get("source") == "humana" and entry.get("description")
                and entry.get("class_id") == box["class_id"] and isinstance(entry.get("bbox_xyxy"), list)
                and len(entry["bbox_xyxy"]) == 4
                and all(_round(a) == _round(b) for a, b in zip(entry["bbox_xyxy"], box["bbox_xyxy"]))):
            return entry
    return None


def ai_summary(notes: dict | None, note: str | None, labels_by_slug: dict[str, str]) -> dict:
    """Bloco ``ai_support`` do laudo.json (Anexo C): modelo, contagens e concordância por classe.

    Parâmetros:
        notes: conteúdo de ai_notes.json ou None.
        note: aviso de ``load_ai_notes`` (arquivo ignorado) ou None.
        labels_by_slug: slug da categoria → rótulo legível (da taxonomia).

    Retorna:
        Sem notas: ``{"available": False, "note": <motivo>}``. Com notas: ``available`` True,
        ``model``, ``created_at``, ``described``, ``total_boxes``, ``agreement_rate``,
        ``human_boxes_described``, ``per_class`` (lista de {slug, label, boxes, agree,
        agreement_rate}) e ``warning``.

    Chamada por: exports._build_report_files. Lido por render_html/write_pdf (seção 4.2 e Anexo C).
    """
    if not notes:
        return {"available": False, "note": note or "Nenhuma descrição por modelo de visão e linguagem nesta revisão "
                                                     "(ai_notes.json do vlm_assist)."}
    # Configurações do vlm_assist e relatório de concordância (vlm_assist.agreement_report).
    settings = notes.get("settings") or {}
    agreement = notes.get("agreement") or {}
    per_class = []
    # Uma linha por categoria; o rótulo vem da taxonomia, com o slug como reserva.
    for slug, row in (agreement.get("per_class") or {}).items():
        if isinstance(row, dict):
            per_class.append({"slug": slug, "label": labels_by_slug.get(slug, slug), "boxes": row.get("boxes"),
                              "agree": row.get("agree"), "agreement_rate": row.get("agreement_rate")})
    # EXTENSAO: dado novo do VLM no Anexo C = chave aqui e a linha correspondente no Anexo C de render_html e write_pdf.
    return {"available": True, "note": note, "created_at": notes.get("created_at"),
            "model": settings.get("model_requested") or settings.get("model"), "described": notes.get("described"),
            "total_boxes": notes.get("total_boxes"), "agreement_rate": agreement.get("agreement_rate"),
            "human_boxes_described": agreement.get("human_boxes_described"), "per_class": per_class,
            "warning": "Descrições automáticas de recortes; subsídio à revisão, sem valor de constatação técnica."}


# ---------------------------------------------------------------------------
# HTML (A4, autossuficiente).
# ---------------------------------------------------------------------------

def _esc(value) -> str:
    """Escapa texto para HTML (``&``, ``<``, ``>``, aspas); None vira "".

    Parâmetros:
        value: qualquer valor (convertido com ``str``).

    Retorna:
        str seguro para pôr dentro de tags e atributos.
    ATENCAO: todo texto vindo do usuário ou do laudo.json precisa passar por aqui em render_html.
    """
    return html.escape("" if value is None else str(value), quote=True)


def registry_label(header: dict) -> str:
    """Inspeção no laudo: nome legível seguido da pasta do registro, que garante a rastreabilidade.

    Parâmetros:
        header: report["header"] (usa ``registry_version`` e ``registry_display_name``).

    Retorna:
        str "<nome legível> (<pasta>)", ex.: "CEASA Pavilhão A v8 · Revisão 002 FF34E226
        (ceasa_v8_revisao002_ff34e226416c)"; só a pasta se o nome for igual a ela; "" sem pasta.

    Chamada por: render_html e write_pdf (folha de rosto, 4.2, Anexo E).
    """
    version = str(header.get("registry_version") or "")
    name = header.get("registry_display_name") or review_name(version)
    return f"{name} ({version})" if version and name != version else version


def _or_blank(value: str) -> str:
    """Valor digitado ou ``NOT_INFORMED`` ("não informado") quando vazio.

    Parâmetros:
        value: texto de ``details`` (já normalizado).

    Retorna:
        o próprio valor, ou "não informado". ATENCAO: o sistema nunca estima o dado faltante.
    """
    return value if value else NOT_INFORMED


def _pct(value) -> str:
    """Fração 0–1 em porcentagem inteira ("0.873" → "87%"); None → "—".

    Parâmetros:
        value: taxa de concordância (float 0–1) ou None.

    Retorna:
        str com a porcentagem arredondada ou travessão.
    """
    return "—" if value is None else f"{round(float(value) * 100)}%"


def detector_step(header: dict) -> str:
    """Etapa do detector na metodologia: só afirma a detecção quando há procedência do modelo.

    Parâmetros:
        header: report["header"]; usa ``suggestion_model`` (dict com model_label,
            weights_sha256, conf) e ``taxonomy_version``.

    Retorna:
        str com a frase do passo 2 da seção 4.2 (HTML e PDF).
    """
    model = header.get("suggestion_model") or {}
    # Sem arquivo de sugestões: não dizer que houve detecção automática.
    if not model:
        return ("Detecção automática: sem arquivo de sugestões do detector nesta revisão; as demarcações vêm da revisão "
                "humana e do registro de propostas.")
    # Nome do modelo, hash dos pesos e, se conhecido, o limiar de confiança usado.
    text = f"{model.get('model_label') or 'modelo'} (pesos SHA-256 {model.get('weights_sha256') or '—'}"
    text += f", confiança mínima {model['conf']})" if model.get("conf") is not None else ")"
    return f"Detecção automática de candidatas por categoria da taxonomia {header.get('taxonomy_version') or '—'} com o detector {text}."


def table_of_contents(report: dict) -> list[tuple[str, str, int]]:
    """(âncora, título, nível) na ordem do documento; HTML e PDF usam a mesma lista.

    Parâmetros:
        report: o laudo.json (usa ``report["items"]`` para os subitens 6.N).

    Retorna:
        lista de tuplas (id da âncora HTML, título, nível 0 = seção, 1 = subseção).

    Chamada por: render_html (Sumário). ATENCAO: o PDF NÃO usa esta lista; o sumário do PDF
    é montado pelo reportlab a partir das chamadas ``heading(...)`` em write_pdf. Ao acrescentar
    ou renomear uma seção, mude aqui, em render_html (mesmo ``id``) e em write_pdf (``heading``).
    """
    # Seções 1 a 6 fixas (com as subseções da 4).
    entries = [("s1", "1 IDENTIFICAÇÃO DO LAUDO", 0), ("s2", "2 OBJETIVO", 0), ("s3", "3 NORMAS E DOCUMENTOS APLICÁVEIS", 0),
               ("s4", "4 METODOLOGIA", 0), ("s4-1", "4.1 Levantamento de campo", 1), ("s4-2", "4.2 Processamento assistido por IA", 1),
               ("s4-3", "4.3 Critérios de classificação", 1), ("s4-4", "4.4 Limitações", 1),
               ("s5", "5 DESCRIÇÃO DO IMÓVEL / ÁREA INSPECIONADA", 0),
               ("s6", "6 REGISTRO FOTOGRÁFICO E ANÁLISE DE CONFORMIDADES", 0)]
    # Um subitem 6.N por categoria com ocorrência confirmada (mesma numeração do laudo.json).
    for item in report["items"]:
        entries.append((f"s6-{item['number']}", f"6.{item['number']} {item['label']}", 1))
    # Os dois últimos subitens da 6 vêm depois dos itens (6.N+1 e 6.N+2), depois 7–10,
    # Referências e Anexos A–E.
    # EXTENSAO: seção ou anexo novo no sumário do HTML = mais uma tupla aqui, na posição certa.
    tail = len(report["items"]) + 1
    entries += [(f"s6-{tail}", f"6.{tail} Categorias sem ocorrência confirmada", 1),
                (f"s6-{tail + 1}", f"6.{tail + 1} Itens não avaliados pela detecção automática", 1),
                ("s7", "7 QUADRO-RESUMO DAS CONSTATAÇÕES", 0), ("s8", "8 RECOMENDAÇÕES GERAIS", 0),
                ("s9", "9 CONSIDERAÇÕES FINAIS", 0), ("s10", "10 RESPONSABILIDADE TÉCNICA", 0), ("referencias", "REFERÊNCIAS", 0),
                ("anexo-a", "ANEXO A — Fotografias sem ocorrências confirmadas", 0),
                ("anexo-b", "ANEXO B — Sugestões automáticas pendentes", 0),
                ("anexo-c", "ANEXO C — Subsídios de modelos de IA", 0),
                ("anexo-d", "ANEXO D — Demarcações por fotografia", 0),
                ("anexo-e", "ANEXO E — Apoio ao responsável técnico e rastreabilidade", 0)]
    return entries


# ---------------------------------------------------------------------------
# Estilo do HTML (A4, margens ABNT: 3 cm sup./esq., 2 cm inf./dir.; Arial 12 pt)
# Classes usadas em render_html: .capa, .rosto, .banner (aviso de versão preliminar),
# .pendente (campos pendentes em itálico marrom), .sumario, .pergunta, figure/.legenda/.fonte,
# img.evidencia, .avaliacao (quadro de 5 campos por item), .ia (subsídio de IA, roxo),
# .mono (hashes), .assinatura, .referencias, footer.
# "section" começa nova página na impressão; "section.continua" não quebra página.
# EXTENSAO: classe nova = acrescente a regra dentro desta string e use-a em render_html.
# ATENCAO: é uma string Python; comentário dentro dela vai para o HTML (use /* */ de CSS).
# ---------------------------------------------------------------------------
HTML_STYLE = """
:root { color-scheme: light; }
@page { size: A4; margin: 30mm 20mm 20mm 30mm; }
body { margin: 0; background: #fff; color: #111; font: 12pt/1.5 Arial, Helvetica, sans-serif; }
main { max-width: 160mm; margin: 0 auto; padding: 12mm 0; }
section { break-before: page; page-break-before: always; }
section.continua { break-before: auto; page-break-before: auto; }
h1, h2 { font-size: 12pt; font-weight: 700; text-transform: uppercase; margin: 0 0 12pt; }
h3 { font-size: 12pt; font-weight: 700; margin: 18pt 0 8pt; }
p { margin: 0 0 8pt; text-align: justify; }
.capa { min-height: 240mm; display: flex; flex-direction: column; justify-content: space-between; text-align: center; text-transform: uppercase; font-weight: 700; }
.capa .titulo { font-size: 16pt; line-height: 1.4; }
.capa .sub { font-weight: 400; text-transform: none; }
.rosto { text-align: center; }
.rosto .natureza { margin: 18pt 0 18pt 50%; text-align: justify; font-size: 10pt; line-height: 1.2; }
.banner { border: 1.5pt solid #B45309; background: #FFF7ED; color: #7C2D12; padding: 6pt 10pt; font-weight: 700; margin: 10pt 0 14pt; text-transform: none; font-size: 10.5pt; }
table { border-collapse: collapse; width: 100%; font-size: 10pt; line-height: 1.25; margin: 6pt 0 12pt; }
th, td { border: 0.75pt solid #9CA3AF; padding: 4pt 6pt; text-align: left; vertical-align: top; }
th { background: #F3F4F6; font-weight: 700; }
.pendente { color: #92400E; font-style: italic; }
.sumario a { color: inherit; text-decoration: none; }
.sumario li { list-style: none; margin: 2pt 0; }
.sumario li.n1 { padding-left: 18pt; }
.sumario ol { padding: 0; }
.pergunta { font-weight: 700; margin-top: 4pt; }
figure { margin: 10pt 0 12pt; break-inside: avoid; page-break-inside: avoid; text-align: center; }
figure .legenda { font-size: 10pt; margin-bottom: 4pt; }
figure .fonte { font-size: 9pt; text-align: left; margin-top: 3pt; word-break: break-all; }
img.evidencia { display: block; max-width: 100%; max-height: 120mm; margin: 0 auto; height: auto; border: 0.75pt solid #9CA3AF; }
.avaliacao th { width: 28%; }
.ia { border-left: 3pt solid #7C3AED; background: #F5F3FF; padding: 6pt 10pt; font-size: 10pt; line-height: 1.3; margin: 8pt 0 10pt; }
.ia b { color: #5B21B6; }
.mono { font-family: Consolas, "Courier New", monospace; font-size: 8.5pt; word-break: break-all; }
ul.lista li, ol.lista li { margin-bottom: 4pt; text-align: justify; }
.assinatura { margin-top: 40pt; text-align: center; }
.assinatura .linha { border-top: 0.75pt solid #111; width: 70%; margin: 0 auto 4pt; }
.referencias p { text-align: left; }
footer { margin-top: 24pt; font-size: 8.5pt; color: #4B5563; border-top: 0.75pt solid #D1D5DB; padding-top: 6pt; }
"""


def _rows(pairs: list[tuple[str, str]], width: str = "34%") -> str:
    """Tabela chave → valor; valores já escapados.

    Parâmetros:
        pairs: lista de (rótulo, valor HTML). O rótulo é escapado aqui; o valor NÃO (quem
            chama escapa com ``_esc`` ou passa HTML pronto, como ``_pending_cell()``).
        width: largura CSS da coluna dos rótulos.

    Retorna:
        str com a ``<table>`` completa.
    """
    return "<table>" + "".join(f'<tr><th style="width:{width}">{_esc(key)}</th><td>{value}</td></tr>' for key, value in pairs) + "</table>"


def _pending_cell() -> str:
    """Célula ``<span class="pendente">`` com ``PENDING_TEXT`` (campo de julgamento técnico).

    Retorna:
        str HTML.
    """
    return f'<span class="pendente">{_esc(PENDING_TEXT)}</span>'


def _evaluation_html() -> str:
    """Quadro de avaliação de cada item 6.N: cinco linhas, todas "pendente de validação técnica".

    Retorna:
        str com a ``<table class="avaliacao">``.
    ATENCAO: não preencha estes campos com valor calculado; o laudo exige que fiquem pendentes.
    EXTENSAO: campo novo de avaliação = mais um rótulo na tupla abaixo E na mesma tupla do
    ``pairs([(label, pending()) ...])`` do item em write_pdf.
    """
    return ('<table class="avaliacao">' + "".join(f"<tr><th>{label}</th><td class=\"pendente\">{_esc(PENDING_TEXT)}</td></tr>"
                                                  for label in ("Conformidade", "Risco (gravidade)", "Diagnóstico", "Recomendação", "Prioridade (NBR 16747)"))
            + "</table>")


def _figure_html(figure: dict, evidences: dict[int, bytes]) -> str:
    """Uma figura numerada (legenda, imagem embutida em base64, fonte).

    Parâmetros:
        figure: item de report["figures"] (number, numero_foto, caption, source, item).
        evidences: número da foto (``numero_foto``) → bytes JPEG da evidência reduzida.

    Retorna:
        str ``<figure id="figura-N">`` autossuficiente (a imagem vai dentro do HTML).

    Erros: KeyError se a foto da figura não estiver em ``evidences``.
    ATENCAO: o tipo é declarado como image/jpeg; exports.py gera as evidências em JPEG.
    """
    encoded = base64.b64encode(evidences[figure["numero_foto"]]).decode("ascii")
    return (f'<figure id="figura-{figure["number"]}" data-foto="{figure["numero_foto"]:03d}">'
            f'<div class="legenda"><b>Figura {figure["number"]}</b> – {_esc(figure["caption"])}</div>'
            f'<img class="evidencia" src="data:image/jpeg;base64,{encoded}" alt="Figura {figure["number"]}: {_esc(figure["caption"])}">'
            f'<div class="fonte">Fonte: {_esc(figure["source"])}</div></figure>')


# ---------------------------------------------------------------------------
# render_html: o laudo inteiro em uma página HTML.
# Mapa das seções neste corpo (procure o título entre "====" para achar cada uma):
#   Capa · Folha de rosto · Sumário · Seção 1 Identificação · Seção 2 Objetivo ·
#   Seção 3 Normas · Seção 4 Metodologia (4.1 a 4.4) · Seção 5 Imóvel ·
#   Seção 6 Registro fotográfico (itens 6.N, 6.N+1, 6.N+2) · Seção 7 Quadro-resumo ·
#   Seção 8 Recomendações · Seção 9 Considerações finais · Seção 10 Responsabilidade técnica ·
#   Referências · Anexo A · Anexo B · Anexo C · Anexo D · Anexo E · Rodapé.
# Os mesmos títulos existem em write_pdf com o sufixo "(PDF)".
# ---------------------------------------------------------------------------
# ATENCAO: HTML e PDF (write_pdf, abaixo) são desenhados separadamente a partir do mesmo
# laudo.json. Mudou uma seção em um, mude no outro; tests/test_exports.py gera os dois.
# ENTRADA: exports._build_report_files chama render_html e grava o texto em laudo.html.
def render_html(report: dict, evidences: dict[int, bytes]) -> str:
    """Laudo completo em HTML A4 autossuficiente (CSS e imagens embutidos).

    Parâmetros:
        report: o laudo.json montado por exports.build_report (chaves header, details,
            summary, items, items_without_occurrence, manual_checklist, figures, images,
            occurrences, draft_boxes, pending_suggestions, ai_support, conclusion,
            technical_responsibility, normative_references, limitations, provenance).
        evidences: numero_foto → bytes JPEG das fotos que viram figura.

    Retorna:
        str com o documento HTML inteiro (``<!doctype html>`` ... ``</html>``).

    Como funciona: cada seção é acrescentada em ``parts`` na ordem do documento e no fim
    tudo é unido. Para mudar o texto de uma seção, procure o bloco "Seção N ..." abaixo;
    para acrescentar uma seção, insira um ``parts.append('<section id="...">...')`` no ponto
    certo da ordem e a âncora correspondente em ``table_of_contents``.

    Chamada por: exports._build_report_files.
    """
    # Atalhos para as partes do laudo.json mais usadas.
    header, details, summary = report["header"], report["details"], report["summary"]
    # Figura pelo número (os itens guardam só os números das figuras).
    figures = {figure["number"]: figure for figure in report["figures"]}
    # Data de emissão AAAA-MM-DD (os 10 primeiros caracteres do ISO 8601).
    issued = header["issued_at"][:10]
    # Início do documento: <head> com título e o CSS de HTML_STYLE; abre <main>.
    parts = [f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
             f"<title>{_esc(header['title'])}</title><style>{HTML_STYLE}</style></head><body><main>"]
    # =======================================================================
    # Capa (HTML)
    # Mesmo trecho no PDF: write_pdf, bloco "Capa (PDF)".
    # =======================================================================
    # Empresa e solicitante no topo, título e local no meio com o aviso de versão preliminar,
    # endereço e ano (issued[:4]) embaixo.
    parts.append(f'<section id="capa" class="capa continua"><div>{_esc(details["company"] or "DroneCamp Vision")}<br>'
                 f'<span class="sub">{_esc(details["requester"] or "")}</span></div>'
                 f'<div class="titulo">{_esc(header["title"])}<div class="sub">{_esc(header["site"] or "")}</div>'
                 f'<div class="banner">Versão {_esc(PRELIMINARY_TEXT)}. Conformidade, risco, diagnóstico, recomendação e prioridade '
                 f"não foram inferidos e ficam pendentes.</div></div>"
                 f'<div class="sub">{_esc(details["address"] or "")}<br>{_esc(issued[:4])}</div></section>')
    # =======================================================================
    # Folha de rosto (HTML)
    # Mesmo trecho no PDF: write_pdf, bloco "Folha de rosto (PDF)".
    # =======================================================================
    # "nature" = texto da natureza do trabalho (recuado à direita, como pede a NBR 10719).
    nature = (f"Laudo técnico de inspeção visual aérea da cobertura, elaborado com apoio de detecção assistida por inteligência "
              f"artificial e revisão humana das imagens, conforme ABNT NBR 16747 e ABNT NBR 10719, para "
              f"{details['requester'] or 'o solicitante'}.")
    parts.append(f'<section id="folha-rosto" class="rosto"><p style="text-align:center">{_esc(details["responsible_name"] or "")}</p>'
                 f'<h1 style="margin-top:36pt">{_esc(header["title"])}</h1><p style="text-align:center">{_esc(header["site"] or "")}</p>'
                 f'<p class="natureza">{_esc(nature)}</p>'
                 + _rows([("Número do laudo", _esc(_or_blank(details["report_number"]))), ("Inspeção", _esc(registry_label(header))),
                          ("Data de emissão", _esc(issued)), ("Situação do documento", _esc(PRELIMINARY_TEXT))])
                 + "</section>")
    # =======================================================================
    # Sumário (HTML)
    # Mesmo trecho no PDF: write_pdf, bloco "Sumário (PDF)".
    # =======================================================================
    # Lista gerada por table_of_contents; cada linha é um link para a âncora da seção.
    parts.append('<section id="sumario"><h1 style="text-align:center">SUMÁRIO</h1><ol class="sumario">'
                 + "".join(f'<li class="n{level}"><a href="#{anchor}">{_esc(title)}</a></li>' for anchor, title, level in table_of_contents(report))
                 + "</ol></section>")
    # =======================================================================
    # Seção 1 Identificação do laudo (HTML)
    # Mesmo trecho no PDF: write_pdf, bloco "Seção 1 Identificação do laudo (PDF)".
    # =======================================================================
    # Tabela rótulo → valor. EXTENSAO: linha nova na identificação = mais uma tupla nesta
    # lista E na lista "identification" de write_pdf.
    identification = [("Solicitante", _esc(_or_blank(details["requester"]))), ("Local da inspeção", _esc(_or_blank(header["site"]))),
                      ("Endereço", _esc(_or_blank(details["address"]))), ("Data da inspeção", _esc(_or_blank(details["inspection_date"]))),
                      ("Data de emissão", _esc(issued)), ("Número do laudo", _esc(_or_blank(details["report_number"]))),
                      ("Grupos de edificação no registro", _esc(", ".join(header["buildings"]) or "—")),
                      ("Revisor(es) das imagens", _esc(", ".join(header["reviewers"]) or "—")),
                      ("Responsável técnico", _esc(_or_blank(details["responsible_name"]))),
                      ("Registro profissional", _esc(_or_blank(details["responsible_registry"]))),
                      ("ART nº", _esc(details["art_number"]) if details["art_number"] else _pending_cell()),
                      ("Situação do documento", _esc(PRELIMINARY_TEXT))]
    # Observações da revisão só aparecem quando existirem.
    if header["notes"]:
        identification.append(("Observações", _esc(header["notes"])))
    parts.append('<section id="s1"><h2>1 Identificação do laudo</h2>' + _rows(identification) + "</section>")
    # =======================================================================
    # Seção 2 Objetivo (HTML)
    # Mesmo trecho no PDF: write_pdf, bloco "Seção 2 Objetivo (PDF)".
    # =======================================================================
    # Texto fixo; altere aqui e no parágrafo do objetivo em write_pdf.
    parts.append('<section id="s2" class="continua"><h2>2 Objetivo</h2><p>Avaliar as condições da cobertura, identificando '
                 "inconformidades técnicas visíveis por inspeção aérea com drone, visando subsidiar reparos e manutenção preventiva. "
                 "As constatações apresentadas resultam da revisão humana das imagens, com apoio de detecção automática, e dependem "
                 "da validação do responsável técnico para a classificação de conformidade, risco e prioridade.</p></section>")
    # =======================================================================
    # Seção 3 Normas e documentos aplicáveis (HTML)
    # Mesmo trecho no PDF: write_pdf, bloco "Seção 3 Normas (PDF)".
    # =======================================================================
    # Uma linha por norma de report["normative_references"] (= NORMATIVE_REFERENCES), com
    # ":ano" depois do código quando houver ano, mais a linha fixa "Fabricantes".
    parts.append('<section id="s3"><h2>3 Normas e documentos aplicáveis</h2><p>Este laudo observa os documentos abaixo, sem '
                 "prejuízo das especificações dos fabricantes dos componentes da cobertura. As referências completas constam ao final.</p>"
                 "<table><tr><th style=\"width:26%\">Documento</th><th>Título</th><th style=\"width:34%\">Aplicação neste laudo</th></tr>"
                 + "".join(f"<tr><td>{_esc(norm['code'])}{':' + str(norm['year']) if norm.get('year') else ''}</td><td>{_esc(norm['title'])}</td>"
                           f"<td>{_esc(norm['use'])}</td></tr>" for norm in report["normative_references"])
                 + "<tr><td>Fabricantes</td><td>Especificações técnicas e manuais de instalação dos componentes</td>"
                   "<td>Critério de conformidade de fixações, recobrimentos e acessórios.</td></tr></table></section>")
    # =======================================================================
    # Seção 4 Metodologia (HTML)
    # Mesmo trecho no PDF: write_pdf, bloco "Seção 4 Metodologia (PDF)".
    # =======================================================================
    # Textos auxiliares: model_text (detector, também usado no Anexo C e E) e vlm_text (passo 4).
    model = header.get("suggestion_model") or {}
    model_text = (f"{model.get('model_label') or 'modelo'} (pesos SHA-256 {model.get('weights_sha256') or '—'}"
                  + (f", confiança mínima {model['conf']}" if model.get("conf") is not None else "") + ")") if model else "nenhum arquivo de sugestões nesta revisão"
    # Resumo dos subsídios do VLM (ai_summary) para o passo 4 da 4.2.
    ai = report["ai_support"]
    vlm_text = (f"descrições de {ai.get('described')} recortes geradas por {ai.get('model') or 'modelo de visão e linguagem'} "
                f"em {str(ai.get('created_at') or '')[:10]}, mantidas como subsídio pendente (Anexo C)") if ai["available"] else "não utilizado nesta revisão"
    parts.append('<section id="s4"><h2>4 Metodologia</h2>'
                 # --- 4.1 Levantamento de campo: equipamento, voos e data digitados pelo usuário.
                 f'<h3 id="s4-1">4.1 Levantamento de campo</h3><p>Equipamento utilizado: {_esc(_or_blank(details["equipment"]))}. '
                 f'Voos realizados: {_esc(_or_blank(details["flight_description"]))}. Data da inspeção: {_esc(_or_blank(details["inspection_date"]))}. '
                 "A operação da aeronave não tripulada observa a regulamentação da ANAC (RBAC-E nº 94) e do DECEA.</p>"
                 # --- 4.2 Processamento assistido por IA: passos numerados 1 a 5.
                 # EXTENSAO: passo novo = mais um <li> aqui E mais um p("N. ...") na 4.2 de write_pdf.
                 '<h3 id="s4-2">4.2 Processamento assistido por IA</h3><ol class="lista">'
                 f"<li>Registro das fotografias com hash SHA-256 (inspeção {_esc(registry_label(header))}, {summary['images']} fotografias).</li>"
                 f"<li>{_esc(detector_step(header))}</li>"
                 "<li>Revisão humana de cada fotografia na plataforma: aceite, correção ou descarte das demarcações e classificação "
                 "da fotografia (positiva, negativa, ambígua ou excluída). Apenas demarcações em fotografias positivas constituem constatações.</li>"
                 f"<li>Modelo de visão e linguagem: {_esc(vlm_text)}.</li>"
                 "<li>Emissão deste documento a partir das decisões registradas, com as figuras regeneradas das fotografias originais.</li></ol>"
                 # --- 4.3 Critérios de classificação: tabelas CONFORMITY_SCALE e PRIORITY_SCALE.
                 '<h3 id="s4-3">4.3 Critérios de classificação</h3><p>A conformidade de cada item segue a escala do laudo de referência '
                 "do projeto e a prioridade das ações segue os patamares de urgência da ABNT NBR 16747. Ambas são atribuídas exclusivamente "
                 "pelo responsável técnico.</p><table><tr><th style=\"width:30%\">Conformidade</th><th>Critério</th></tr>"
                 + "".join(f"<tr><td>{_esc(name)}</td><td>{_esc(text)}</td></tr>" for name, text in CONFORMITY_SCALE)
                 + "</table><table><tr><th style=\"width:30%\">Prioridade</th><th>Critério (ABNT NBR 16747)</th></tr>"
                 + "".join(f"<tr><td>{_esc(name)}</td><td>{_esc(text)}</td></tr>" for name, text in PRIORITY_SCALE)
                 # --- 4.4 Limitações: report["limitations"] (= LIMITATIONS) numeradas.
                 + '</table><h3 id="s4-4">4.4 Limitações</h3><ol class="lista">'
                 + "".join(f"<li>{_esc(text)}</li>" for text in report["limitations"]) + "</ol></section>")
    # =======================================================================
    # Seção 5 Descrição do imóvel / área inspecionada (HTML)
    # Mesmo trecho no PDF: write_pdf, bloco "Seção 5 Imóvel (PDF)".
    # =======================================================================
    # Tabela com os campos de PROPERTY_FIELDS (rótulos de DETAIL_FIELDS); vazio → "não informado".
    parts.append('<section id="s5" class="continua"><h2>5 Descrição do imóvel / área inspecionada</h2>'
                 + _rows([(DETAIL_FIELDS[key], _esc(_or_blank(details[key]))) for key in PROPERTY_FIELDS])
                 + "<p>Dados não informados não são estimados pelo sistema.</p></section>")
    # =======================================================================
    # Seção 6 Registro fotográfico e análise de conformidades (HTML)
    # Mesmo trecho no PDF: write_pdf, bloco "Seção 6 Registro fotográfico (PDF)".
    # =======================================================================
    # Abre a seção 6 com o parágrafo introdutório; os itens vêm no laço abaixo.
    parts.append('<section id="s6"><h2>6 Registro fotográfico e análise de conformidades</h2>'
                 "<p>Cada item corresponde a uma categoria da lista de verificação com ocorrência confirmada na revisão das imagens. "
                 f'Os campos de avaliação estão preenchidos como "{_esc(PENDING_TEXT)}" até a validação do responsável técnico.</p>')
    # Sem nenhuma categoria confirmada: só uma frase.
    if not report["items"]:
        parts.append("<p>Nenhuma ocorrência confirmada nesta revisão.</p>")
    # --- Itens 6.N: um <article> por categoria com ocorrência confirmada, com pergunta da
    # CHECKLIST, constatação (statement), notas, figuras, subsídio de IA e quadro de avaliação.
    # EXTENSAO: bloco novo dentro de cada item = um parts.append(...) neste laço, antes do
    # _evaluation_html(), e o equivalente no laço "for item in report[\"items\"]" de write_pdf.
    for item in report["items"]:
        parts.append(f'<article class="item" id="s6-{item["number"]}"><h3>6.{item["number"]} {_esc(item["label"])}</h3>'
                     f'<p class="pergunta">{_esc(item["question"])}</p><p>{_esc(item["statement"])}</p>')
        # Notas da revisão (da foto e das caixas), já limitadas a 8 em exports.py.
        for note in item["notes"]:
            parts.append(f"<p>Nota da revisão: {_esc(note)}</p>")
        # Figuras que nasceram neste item (as já mostradas em itens anteriores só são citadas).
        for number in item["figures"]:
            parts.append(_figure_html(figures[number], evidences))
        # Subsídio de IA (até 6 descrições do VLM), sempre marcado como não constatação.
        if item["ai_support"]:
            parts.append('<div class="ia"><b>Subsídio de IA</b> (descrição automática dos recortes por modelo de visão e linguagem; '
                         "não constitui constatação técnica):<ul>"
                         + "".join(f"<li>Figura {row['figure']}, demarcação {row['box']}: {_esc(row['text'])}"
                                   f"{' (' + _esc(row['model']) + ')' if row.get('model') else ''}</li>" for row in item["ai_support"])
                         + "</ul></div>")
        # Quadro de avaliação com os cinco campos pendentes e fim do item.
        parts.append(_evaluation_html() + "</article>")
    # --- 6.N+1 Categorias sem ocorrência confirmada e 6.N+2 Itens não avaliados.
    # "tail" é o número do primeiro subitem depois dos itens (mesmo cálculo de table_of_contents).
    tail = len(report["items"]) + 1
    parts.append(f'<h3 id="s6-{tail}">6.{tail} Categorias sem ocorrência confirmada</h3><p>Não há demarcação confirmada destas '
                 "categorias nas fotografias revisadas. A ausência nas imagens não comprova a ausência na cobertura; a conformidade "
                 "depende do responsável técnico.</p><table><tr><th>Item</th><th>Pergunta</th><th style=\"width:28%\">Conformidade</th></tr>"
                 + "".join(f"<tr><td>{_esc(row['label'])}</td><td>{_esc(row['question'])}</td><td>{_pending_cell()}</td></tr>"
                           for row in report["items_without_occurrence"])
                 + f'</table><h3 id="s6-{tail + 1}">6.{tail + 1} Itens não avaliados pela detecção automática</h3><p>Os itens abaixo '
                 "constam da lista de verificação do laudo de referência, mas não são cobertos pelas categorias do sistema e não foram "
                 "avaliados neste documento.</p><table><tr><th>Pergunta</th><th style=\"width:28%\">Situação</th></tr>"
                 + "".join(f"<tr><td>{_esc(question)}</td><td class=\"pendente\">não avaliado pelo sistema</td></tr>" for question in report["manual_checklist"])
                 + "</table></section>")
    # =======================================================================
    # Seção 7 Quadro-resumo das constatações (HTML)
    # Mesmo trecho no PDF: write_pdf, bloco "Seção 7 Quadro-resumo (PDF)".
    # =======================================================================
    # Uma linha por item (demarcações, fotos, figuras e três colunas pendentes) e linha de total.
    # EXTENSAO: coluna nova = <th> no cabeçalho, <td> na linha do item e ajuste do colspan do
    # total; no PDF, a mesma coluna na tabela da Seção 7 e na lista de larguras.
    parts.append('<section id="s7"><h2>7 Quadro-resumo das constatações</h2><table><tr><th>Item</th><th>Categoria</th><th>Demarcações</th>'
                 "<th>Fotografias</th><th>Figuras</th><th>Conformidade</th><th>Prioridade</th><th>Recomendação</th></tr>"
                 + "".join(f"<tr><td>6.{item['number']}</td><td>{_esc(item['label'])}</td><td>{item['boxes']}</td><td>{item['images']}</td>"
                           f"<td>{_esc(', '.join(map(str, item['figure_refs'])))}</td><td>{_pending_cell()}</td><td>{_pending_cell()}</td><td>{_pending_cell()}</td></tr>"
                           for item in report["items"])
                 + f"<tr><th colspan=\"2\">Total</th><th>{summary['confirmed_boxes']}</th><th>{summary['positive']}</th><th colspan=\"4\">"
                 "Contagem de demarcações por fotografia; fotografias semelhantes podem repetir o mesmo elemento.</th></tr></table></section>")
    # =======================================================================
    # Seção 8 Recomendações gerais (HTML)
    # Mesmo trecho no PDF: write_pdf, bloco "Seção 8 Recomendações gerais (PDF)".
    # =======================================================================
    # Lista GENERAL_RECOMMENDATIONS + aviso de recomendações por item pendentes.
    parts.append('<section id="s8" class="continua"><h2>8 Recomendações gerais</h2><ul class="lista">'
                 + "".join(f"<li>{_esc(text)}</li>" for text in GENERAL_RECOMMENDATIONS)
                 + f'</ul><p>Recomendações específicas por item: <span class="pendente">{_esc(PENDING_TEXT)}</span>.</p></section>')
    # =======================================================================
    # Seção 9 Considerações finais (HTML)
    # Mesmo trecho no PDF: write_pdf, bloco "Seção 9 Considerações finais (PDF)".
    # =======================================================================
    # Fatos (conclusion["facts"], montado em exports.py) + conclusão técnica pendente.
    conclusion = report["conclusion"]
    parts.append(f'<section id="s9" class="continua"><h2>9 Considerações finais</h2><p>{_esc(conclusion["facts"])}</p>'
                 f'<p>Conclusão técnica: <span class="pendente">{_esc(PENDING_TEXT)}</span>.</p></section>')
    # =======================================================================
    # Seção 10 Responsabilidade técnica (HTML)
    # Mesmo trecho no PDF: write_pdf, bloco "Seção 10 Responsabilidade técnica (PDF)".
    # =======================================================================
    # Declaração e bloco de assinatura (linhas em branco quando nome, registro ou ART faltam).
    responsible = report["technical_responsibility"]
    parts.append('<section id="s10" class="continua"><h2>10 Responsabilidade técnica</h2>'
                 f"<p>{_esc(responsible['statement'])}</p><div class=\"assinatura\"><div class=\"linha\"></div>"
                 f"<div>{_esc(responsible['name'] or 'Responsável técnico')}</div>"
                 f"<div>{_esc(responsible['registry'] or 'Registro profissional: ____________')}</div>"
                 f"<div>{_esc(responsible['title'] or '')}</div>"
                 f"<div>ART nº {_esc(responsible['art'] or '____________')}</div></div></section>")
    # =======================================================================
    # Referências (HTML)
    # Mesmo trecho no PDF: write_pdf, bloco "Referências (PDF)".
    # =======================================================================
    # Normas no formato NBR 6023, em ordem alfabética do texto da referência.
    parts.append('<section id="referencias" class="referencias"><h2 style="text-align:center">Referências</h2>'
                 + "".join(f"<p>{_esc(reference_text(norm))}</p>" for norm in sorted(report["normative_references"], key=reference_text))
                 + "</section>")
    # =======================================================================
    # Anexo A Fotografias sem ocorrências confirmadas (HTML)
    # Mesmo trecho no PDF: write_pdf, bloco "Anexo A (PDF)".
    # =======================================================================
    # Tabela das fotos com status diferente de "positive" e as figuras sem item (figure["item"] None).
    others = [image for image in report["images"] if image["status"] != "positive"]
    parts.append('<section id="anexo-a"><h2>Anexo A — Fotografias sem ocorrências confirmadas, ambíguas, excluídas ou pendentes</h2>')
    if others:
        parts.append("<table><tr><th>Foto</th><th>Arquivo</th><th>Status</th><th>Origem</th><th>Demarcações não confirmadas</th><th>Notas</th></tr>"
                     + "".join(f"<tr><td>{image['numero_foto']:03d}</td><td>{_esc(image['filename'])}</td><td>{_esc(image['status_label'])}</td>"
                               f"<td>{_esc(image['source_label'])}</td><td>{image['boxes']}</td><td>{_esc(image['notes'] or '—')}</td></tr>" for image in others)
                     + "</table>")
        for figure in report["figures"]:
            if figure["item"] is None:
                parts.append(_figure_html(figure, evidences))
    else:
        parts.append("<p>Todas as fotografias do registro têm ocorrências confirmadas.</p>")
    parts.append("</section>")
    # =======================================================================
    # Anexo B Sugestões automáticas pendentes (HTML)
    # Mesmo trecho no PDF: write_pdf, bloco "Anexo B (PDF)".
    # =======================================================================
    # Contagem por categoria das sugestões do detector ainda sem decisão humana.
    pending = report["pending_suggestions"]
    parts.append(f'<section id="anexo-b"><h2>Anexo B — Sugestões automáticas pendentes</h2><p>{_esc(pending["note"])}</p>')
    if pending["by_class"]:
        parts.append("<table><tr><th>Categoria</th><th>Sugestões pendentes</th><th>Fotografias</th></tr>"
                     + "".join(f"<tr><td>{_esc(row['label'])}</td><td>{row['count']}</td><td>{row['images']}</td></tr>" for row in pending["by_class"])
                     + f"<tr><th>Total</th><th>{pending['total']}</th><th></th></tr></table>")
    else:
        parts.append("<p>Nenhuma sugestão automática pendente nesta revisão.</p>")
    parts.append("</section>")
    # =======================================================================
    # Anexo C Subsídios de modelos de IA (HTML)
    # Mesmo trecho no PDF: write_pdf, bloco "Anexo C (PDF)".
    # =======================================================================
    # Detector (model_text) e, se houver ai_notes.json válido, modelo VLM, contagens e
    # concordância por classe; senão o aviso de ai_summary.
    parts.append(f'<section id="anexo-c"><h2>Anexo C — Subsídios de modelos de IA</h2><p>Detector de candidatas: {_esc(model_text)}.</p>')
    if ai["available"]:
        parts.append(f"<p>{_esc(ai['warning'])} Modelo: {_esc(ai.get('model') or '—')}; recortes descritos: {_esc(ai.get('described'))} "
                     f"de {_esc(ai.get('total_boxes'))}; concordância global com a classe humana: {_pct(ai.get('agreement_rate'))} "
                     f"({_esc(ai.get('human_boxes_described'))} demarcações humanas descritas).</p>")
        if ai["per_class"]:
            parts.append("<table><tr><th>Categoria</th><th>Demarcações descritas</th><th>Concordâncias</th><th>Taxa</th></tr>"
                         + "".join(f"<tr><td>{_esc(row['label'])}</td><td>{_esc(row['boxes'])}</td><td>{_esc(row['agree'])}</td>"
                                   f"<td>{_pct(row['agreement_rate'])}</td></tr>" for row in ai["per_class"]) + "</table>")
    else:
        parts.append(f"<p>{_esc(ai['note'])}</p>")
    parts.append("</section>")
    # =======================================================================
    # Anexo D Demarcações por fotografia (HTML)
    # Mesmo trecho no PDF: write_pdf, bloco "Anexo D (PDF)".
    # =======================================================================
    # Todas as caixas (confirmadas e rascunhos), ordenadas por foto e número da caixa.
    # bbox_xyxy = [x1, y1, x2, y2] em pixels da foto original (canto superior esquerdo e inferior direito).
    rows = report["occurrences"] + report["draft_boxes"]
    parts.append('<section id="anexo-d"><h2>Anexo D — Demarcações por fotografia</h2>')
    if rows:
        parts.append("<table><tr><th>Foto</th><th>Figura</th><th>Nº</th><th>Categoria</th><th>Coordenadas (x1, y1 – x2, y2)</th><th>Status da foto</th><th>Origem</th><th>Nota</th></tr>"
                     + "".join(f"<tr><td>{row['numero_foto']:03d}</td><td>{row.get('figura') or '—'}</td><td>{row['caixa']}</td><td>{_esc(row['classe'])}</td>"
                               f"<td>{row['bbox_xyxy'][0]}, {row['bbox_xyxy'][1]} – {row['bbox_xyxy'][2]}, {row['bbox_xyxy'][3]}</td>"
                               f"<td>{_esc(row['status_foto'])}</td><td>{_esc(row['origem'])}</td><td>{_esc(row['nota'] or '—')}</td></tr>"
                               for row in sorted(rows, key=lambda value: (value["numero_foto"], value["caixa"])))
                     + "</table>")
    else:
        parts.append("<p>Sem demarcações nesta revisão.</p>")
    parts.append("</section>")
    # =======================================================================
    # Anexo E Apoio ao responsável técnico e rastreabilidade (HTML)
    # Mesmo trecho no PDF: write_pdf, bloco "Anexo E (PDF)".
    # =======================================================================
    # Referências do laudo CEASA (CHECKLIST) e gravidade da taxonomia, marcadas como NÃO
    # aplicadas, mais hashes do registro e da taxonomia, modelo e gerador.
    # ATENCAO: estas referências nunca podem ir para os campos de avaliação dos itens.
    parts.append('<section id="anexo-e"><h2>Anexo E — Apoio ao responsável técnico e rastreabilidade</h2>'
                 "<p>Referências do laudo-modelo do projeto (CEASA Minas, Pavilhão A) e gravidade de referência da taxonomia para as "
                 "categorias com ocorrência. São apoio à redação e <b>não foram aplicadas</b> às constatações deste laudo.</p>"
                 "<table><tr><th>Item</th><th>Gravidade de referência da taxonomia (não aplicada automaticamente)</th>"
                 "<th>Laudo-modelo: conformidade</th><th>Laudo-modelo: recomendação</th><th>Critério de identificação</th></tr>"
                 + "".join(f"<tr><td>6.{item['number']} {_esc(item['label'])}</td><td>{_esc(item['reference_severity'] or '—')}</td>"
                           f"<td>{_esc((item['model_reference'] or {}).get('conformidade') or '—')}</td>"
                           f"<td>{_esc((item['model_reference'] or {}).get('recomendacao') or '—')}</td><td>{_esc(item['criterion'] or '—')}</td></tr>"
                           for item in report["items"])
                 + "</table>"
                 + _rows([("Inspeção", _esc(registry_label(header))),
                          ("SHA-256 do registro", f'<span class="mono">{_esc(header["registry_sha256"])}</span>'),
                          ("Taxonomia", f'{_esc(header["taxonomy_version"] or "—")} · <span class="mono">{_esc(header["taxonomy_sha256"])}</span>'),
                          ("Modelo das sugestões automáticas", _esc(model_text)),
                          ("Gerador", _esc(report["provenance"]["generator"])), ("Gerado em", _esc(report["provenance"]["generated_at"]))])
                 + "<p>Edições das normas sem ano na seção 3: confirmar a edição vigente no Catálogo ABNT antes da emissão.</p></section>")
    # Rodapé com data de geração e fim do documento.
    parts.append(f"<footer>Gerado em {_esc(report['provenance']['generated_at'])} por DroneCamp Vision "
                 f"({_esc(report['provenance']['generator'])}). Documento {_esc(PRELIMINARY_TEXT)}.</footer></main></body></html>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# PDF (reportlab).
# ---------------------------------------------------------------------------

# ENTRADA: exports._build_report_files chama write_pdf (só com reportlab instalado; senão o laudo sai sem PDF).
def write_pdf(path: Path, report: dict, evidences: dict[int, bytes]) -> None:
    """Mesmo conteúdo do HTML em PDF A4 (margens ABNT 3/2 cm, sumário com páginas, numeração no canto superior direito).

    Fontes padrão (Helvetica, WinAnsi): acentos do pt-BR funcionam. Capa sem número.

    Parâmetros:
        path: caminho do ``laudo.pdf`` a gravar.
        report: o mesmo laudo.json usado em render_html.
        evidences: numero_foto → bytes JPEG das figuras.

    Retorna:
        None; grava o arquivo em ``path``.

    Erros:
        ImportError se reportlab não estiver instalado; qualquer erro do reportlab sobe para
        exports.py, que apaga o PDF parcial e registra o motivo em ``pdf_reason``.

    Como funciona: monta a lista ``story`` (parágrafos, tabelas, figuras, quebras de página)
    na ordem do documento e chama ``multiBuild`` (duas passagens, para o sumário saber as
    páginas). Cada seção está marcada abaixo como "Seção N ... (PDF)".

    ATENCAO: o PDF não usa table_of_contents: cada ``heading(texto, nível)`` vira uma linha do
    sumário. Seção nova = ``heading(...)`` + conteúdo aqui E a seção em render_html.
    ATENCAO: diferenças atuais em relação ao HTML: o model_text do PDF não mostra a confiança
    mínima; a 4.2 não mostra a data do VLM; o Anexo C não mostra as demarcações humanas
    descritas; o Anexo E não tem a coluna "Critério de identificação".
    """
    # Imports tardios: reportlab é opcional (pip install dronecamp-ia[reports]).
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)
    from reportlab.platypus.tableofcontents import TableOfContents

    # -------------------------------------------------------------------
    # Estilos de texto do PDF (equivalem ao HTML_STYLE do HTML)
    # base: corpo 11 pt justificado; center: centralizado; cell/cell_bold: tabelas 8,5 pt;
    # pending_style: campos pendentes (itálico marrom); h1/h2: títulos de seção/subseção;
    # caption/source: legenda e fonte das figuras; cover_title: título da capa;
    # banner: aviso de versão preliminar; ai_style: caixa roxa do subsídio de IA.
    # EXTENSAO: estilo novo = mais um ParagraphStyle aqui, usado em p(texto, estilo).
    # -------------------------------------------------------------------
    header, details, summary = report["header"], report["details"], report["summary"]
    # Data de emissão AAAA-MM-DD.
    issued = header["issued_at"][:10]
    base = ParagraphStyle("base", fontName="Helvetica", fontSize=11, leading=16.5, alignment=TA_JUSTIFY, spaceAfter=6)
    center = ParagraphStyle("center", parent=base, alignment=TA_CENTER)
    cell = ParagraphStyle("cell", parent=base, fontSize=8.5, leading=10.5, alignment=0, spaceAfter=0)
    cell_bold = ParagraphStyle("cell_bold", parent=cell, fontName="Helvetica-Bold")
    pending_style = ParagraphStyle("pending", parent=cell, fontName="Helvetica-Oblique", textColor=rl_colors.HexColor("#92400E"))
    h1 = ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=12, leading=16, spaceBefore=6, spaceAfter=10, alignment=0)
    h2 = ParagraphStyle("h2", parent=h1, spaceBefore=10, spaceAfter=6)
    caption = ParagraphStyle("caption", parent=cell, fontSize=9.5, leading=12, alignment=TA_CENTER, spaceAfter=3)
    source = ParagraphStyle("source", parent=cell, fontSize=7.5, leading=9)
    cover_title = ParagraphStyle("cover", parent=center, fontName="Helvetica-Bold", fontSize=16, leading=22)
    banner = ParagraphStyle("banner", parent=cell, fontName="Helvetica-Bold", fontSize=9.5, leading=12.5,
                            textColor=rl_colors.HexColor("#7C2D12"), backColor=rl_colors.HexColor("#FFF7ED"),
                            borderColor=rl_colors.HexColor("#B45309"), borderWidth=1, borderPadding=6, spaceBefore=10, spaceAfter=10)
    ai_style = ParagraphStyle("ai", parent=cell, fontSize=8.5, leading=11, backColor=rl_colors.HexColor("#F5F3FF"),
                              borderColor=rl_colors.HexColor("#7C3AED"), borderWidth=0.5, borderPadding=5, spaceBefore=4, spaceAfter=8)

    # -------------------------------------------------------------------
    # Auxiliares internos de write_pdf (parágrafo, tabelas, títulos, figuras)
    # -------------------------------------------------------------------
    def p(text, style=base):
        """Parágrafo com texto escapado (o reportlab interpreta tags tipo <b>, por isso o escape).

        Parâmetros: text (qualquer valor; None vira ""), style (ParagraphStyle, padrão ``base``).
        Retorna: Paragraph.
        """
        return Paragraph(html.escape("" if text is None else str(text)), style)

    # Estilos de tabela: "grid" com cabeçalho cinza na 1ª linha; "key_grid" com a 1ª coluna
    # cinza (tabelas rótulo → valor).
    grid = TableStyle([("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#9CA3AF")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                       ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#F3F4F6"))])
    key_grid = TableStyle([("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#9CA3AF")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("BACKGROUND", (0, 0), (0, -1), rl_colors.HexColor("#F3F4F6"))])

    def table(rows, widths, style=grid, head=True):
        """Tabela com grade; cada célula de texto vira Paragraph (para quebrar linha).

        Parâmetros:
            rows: lista de linhas (cada uma lista de valores ou Paragraphs prontos).
            widths: larguras das colunas em pontos (frações de ``width``).
            style: TableStyle (``grid`` ou ``key_grid``).
            head: True → 1ª linha em negrito e repetida no topo de cada página.
        Retorna: Table.
        """
        data = [[value if isinstance(value, Paragraph) else p(value, cell_bold if head and index == 0 else cell) for value in row]
                for index, row in enumerate(rows)]
        return Table(data, colWidths=widths, repeatRows=1 if head else 0, style=style)

    def pairs(rows, ratio=0.34):
        """Tabela rótulo → valor (equivale a ``_rows`` do HTML).

        Parâmetros: rows (lista de (rótulo, valor ou Paragraph)), ratio (fração da largura do rótulo).
        Retorna: Table.
        """
        return table([[p(key, cell_bold), value if isinstance(value, Paragraph) else p(value, cell)] for key, value in rows],
                     [width * ratio, width * (1 - ratio)], key_grid, head=False)

    def pending():
        """Parágrafo "pendente de validação técnica" no estilo pendente (equivale a _pending_cell)."""
        return p(PENDING_TEXT, pending_style)

    class Document(SimpleDocTemplate):
        """Documento que registra no sumário cada título marcado por ``heading``."""
        def afterFlowable(self, flowable):  # Alimenta o sumário com os títulos marcados.
            """Depois de desenhar cada bloco: se for título (``_toc_level``), cria marcador e entrada do sumário.

            Parâmetros: flowable (bloco recém-desenhado).
            """
            level = getattr(flowable, "_toc_level", None)
            if level is not None:
                key = f"toc{id(flowable)}"
                self.canv.bookmarkPage(key)
                self.notify("TOCEntry", (level, flowable.getPlainText(), self.page, key))

    def heading(text, level=0):
        """Título de seção (level 0, estilo h1) ou subseção (level 1, estilo h2) que entra no sumário.

        Parâmetros: text (título já com número, ex. "4 METODOLOGIA"), level (0 ou 1).
        Retorna: Paragraph marcado com ``_toc_level``.
        """
        para = Paragraph(html.escape(text), h1 if level == 0 else h2)
        para._toc_level = level
        return para

    # Página A4 com margens ABNT (esquerda/superior 30 mm, direita/inferior 20 mm) e metadados do PDF.
    doc = Document(str(path), pagesize=A4, leftMargin=30 * mm, rightMargin=20 * mm, topMargin=30 * mm, bottomMargin=20 * mm,
                   title=header["title"], author=details["responsible_name"] or "DroneCamp Vision", subject=PRELIMINARY_TEXT)
    # Largura útil da página (base para as larguras das colunas, em frações).
    width = doc.width
    figures = {figure["number"]: figure for figure in report["figures"]}

    def figure_block(figure):
        """Figura numerada (legenda, imagem, fonte) que não se divide entre páginas.

        Parâmetros: figure (item de report["figures"]).
        Retorna: KeepTogether. A escala ``ratio`` cabe a imagem na largura útil e em 115 mm
        de altura, sem ampliar (máximo 1.0).
        """
        data = evidences[figure["numero_foto"]]
        image_width, image_height = ImageReader(io.BytesIO(data)).getSize()
        ratio = min(width / image_width, (115 * mm) / image_height, 1.0)
        return KeepTogether([Paragraph(f"<b>Figura {figure['number']}</b> – {html.escape(figure['caption'])}", caption),
                             RLImage(io.BytesIO(data), width=image_width * ratio, height=image_height * ratio),
                             p(f"Fonte: {figure['source']}", source), Spacer(1, 8)])

    # Textos auxiliares do detector (Anexos C e E). Diferente do HTML: sem a confiança mínima.
    model = header.get("suggestion_model") or {}
    model_text = (f"{model.get('model_label') or 'modelo'} (pesos SHA-256 {model.get('weights_sha256') or '—'})") if model else "nenhum arquivo de sugestões nesta revisão"
    ai = report["ai_support"]
    # =======================================================================
    # Capa (PDF)
    # Mesmo trecho no HTML: render_html, bloco "Capa (HTML)".
    # =======================================================================
    # Empresa, solicitante, título, local, aviso preliminar, endereço e ano; termina em PageBreak.
    story = [p((details["company"] or "DroneCamp Vision").upper(), ParagraphStyle("c1", parent=center, fontName="Helvetica-Bold")),
             p(details["requester"] or "", center), Spacer(1, 70 * mm), p(header["title"].upper(), cover_title),
             p(header["site"] or "", center),
             p(f"Versão {PRELIMINARY_TEXT}. Conformidade, risco, diagnóstico, recomendação e prioridade não foram inferidos e ficam pendentes.", banner),
             Spacer(1, 60 * mm), p(details["address"] or "", center), p(issued[:4], center), PageBreak()]
    # =======================================================================
    # Folha de rosto (PDF)
    # Mesmo trecho no HTML: render_html, bloco "Folha de rosto (HTML)".
    # =======================================================================
    # A natureza do trabalho fica na metade direita (tabela de duas colunas com a esquerda vazia).
    nature = (f"Laudo técnico de inspeção visual aérea da cobertura, elaborado com apoio de detecção assistida por inteligência "
              f"artificial e revisão humana das imagens, conforme ABNT NBR 16747 e ABNT NBR 10719, para {details['requester'] or 'o solicitante'}.")
    story += [p(details["responsible_name"] or "", center), Spacer(1, 40 * mm), p(header["title"].upper(), cover_title), p(header["site"] or "", center),
              Spacer(1, 12 * mm), Table([["", p(nature, ParagraphStyle("nat", parent=cell, fontSize=9.5, leading=12, alignment=TA_JUSTIFY))]],
                                       colWidths=[width * 0.5, width * 0.5]), Spacer(1, 12 * mm),
              pairs([("Número do laudo", _or_blank(details["report_number"])), ("Inspeção", registry_label(header)),
                     ("Data de emissão", issued), ("Situação do documento", PRELIMINARY_TEXT)]), PageBreak()]
    # =======================================================================
    # Sumário (PDF)
    # Mesmo trecho no HTML: render_html, bloco "Sumário (HTML)".
    # =======================================================================
    # TableOfContents é preenchido em multiBuild pelas entradas que Document.afterFlowable envia.
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("toc0", parent=cell, fontSize=10, leading=14, fontName="Helvetica-Bold"),
                       ParagraphStyle("toc1", parent=cell, fontSize=10, leading=14, leftIndent=14)]
    story += [p("SUMÁRIO", ParagraphStyle("sum", parent=center, fontName="Helvetica-Bold")), toc, PageBreak()]
    # =======================================================================
    # Seção 1 Identificação do laudo (PDF)  e  Seção 2 Objetivo (PDF)
    # Mesmo trecho no HTML: render_html, blocos "Seção 1 Identificação do laudo (HTML)" e
    # "Seção 2 Objetivo (HTML)".
    # =======================================================================
    # EXTENSAO: linha nova na identificação = tupla aqui E na lista "identification" de render_html.
    identification = [("Solicitante", _or_blank(details["requester"])), ("Local da inspeção", _or_blank(header["site"])),
                      ("Endereço", _or_blank(details["address"])), ("Data da inspeção", _or_blank(details["inspection_date"])),
                      ("Data de emissão", issued), ("Número do laudo", _or_blank(details["report_number"])),
                      ("Grupos de edificação no registro", ", ".join(header["buildings"]) or "—"),
                      ("Revisor(es) das imagens", ", ".join(header["reviewers"]) or "—"),
                      ("Responsável técnico", _or_blank(details["responsible_name"])),
                      ("Registro profissional", _or_blank(details["responsible_registry"])),
                      ("ART nº", details["art_number"] or pending()), ("Situação do documento", PRELIMINARY_TEXT)]
    if header["notes"]:
        identification.append(("Observações", header["notes"]))
    # Seção 1 (título + tabela), Seção 2 (título + parágrafo) e, depois do PageBreak, Seção 3.
    story += [heading("1 IDENTIFICAÇÃO DO LAUDO"), pairs(identification), Spacer(1, 10), heading("2 OBJETIVO"),
              p("Avaliar as condições da cobertura, identificando inconformidades técnicas visíveis por inspeção aérea com drone, visando "
                "subsidiar reparos e manutenção preventiva. As constatações apresentadas resultam da revisão humana das imagens, com apoio "
                "de detecção automática, e dependem da validação do responsável técnico para a classificação de conformidade, risco e prioridade."),
              # =======================================================================
              # Seção 3 Normas (PDF)
              # Mesmo trecho no HTML: render_html, bloco "Seção 3 Normas e documentos aplicáveis (HTML)".
              # =======================================================================
              PageBreak(), heading("3 NORMAS E DOCUMENTOS APLICÁVEIS"),
              p("Este laudo observa os documentos abaixo, sem prejuízo das especificações dos fabricantes dos componentes da cobertura."),
              table([["Documento", "Título", "Aplicação neste laudo"]]
                    + [[f"{norm['code']}{':' + str(norm['year']) if norm.get('year') else ''}", norm["title"], norm["use"]] for norm in report["normative_references"]]
                    + [["Fabricantes", "Especificações técnicas e manuais de instalação", "Critério de conformidade de fixações, recobrimentos e acessórios."]],
                    [width * 0.24, width * 0.42, width * 0.34]), PageBreak()]
    # =======================================================================
    # Seção 4 Metodologia (PDF)
    # Mesmo trecho no HTML: render_html, bloco "Seção 4 Metodologia (HTML)".
    # =======================================================================
    # Diferente do HTML: vlm_text sem a data de criação das descrições.
    vlm_text = (f"descrições de {ai.get('described')} recortes geradas por {ai.get('model') or 'modelo de visão e linguagem'}, "
                "mantidas como subsídio pendente (Anexo C)") if ai["available"] else "não utilizado nesta revisão"
    # 4.1 Levantamento de campo.
    story += [heading("4 METODOLOGIA"), heading("4.1 Levantamento de campo", 1),
              p(f"Equipamento utilizado: {_or_blank(details['equipment'])}. Voos realizados: {_or_blank(details['flight_description'])}. "
                f"Data da inspeção: {_or_blank(details['inspection_date'])}. A operação da aeronave não tripulada observa a regulamentação "
                "da ANAC (RBAC-E nº 94) e do DECEA."),
              # 4.2 Processamento assistido por IA (passos 1 a 5).
              # EXTENSAO: passo novo = mais um p("N. ...") aqui E um <li> na 4.2 de render_html.
              heading("4.2 Processamento assistido por IA", 1),
              p(f"1. Registro das fotografias com hash SHA-256 (inspeção {registry_label(header)}, {summary['images']} fotografias)."),
              p(f"2. {detector_step(header)}"),
              p("3. Revisão humana de cada fotografia: aceite, correção ou descarte das demarcações e classificação da fotografia. "
                "Apenas demarcações em fotografias positivas constituem constatações."),
              p(f"4. Modelo de visão e linguagem: {vlm_text}."),
              p("5. Emissão deste documento a partir das decisões registradas, com as figuras regeneradas das fotografias originais."),
              # 4.3 Critérios de classificação (tabelas CONFORMITY_SCALE e PRIORITY_SCALE).
              heading("4.3 Critérios de classificação", 1),
              p("A conformidade de cada item segue a escala do laudo de referência do projeto e a prioridade das ações segue os patamares "
                "de urgência da ABNT NBR 16747. Ambas são atribuídas exclusivamente pelo responsável técnico."),
              table([["Conformidade", "Critério"]] + [list(row) for row in CONFORMITY_SCALE], [width * 0.3, width * 0.7]), Spacer(1, 6),
              table([["Prioridade", "Critério (ABNT NBR 16747)"]] + [list(row) for row in PRIORITY_SCALE], [width * 0.3, width * 0.7]),
              # 4.4 Limitações (o texto vem logo abaixo, numerado).
              heading("4.4 Limitações", 1)]
    story += [p(f"{index}. {text}") for index, text in enumerate(report["limitations"], start=1)]
    # =======================================================================
    # Seção 5 Imóvel (PDF)  e  abertura da Seção 6 Registro fotográfico (PDF)
    # Mesmo trecho no HTML: render_html, blocos "Seção 5 Descrição do imóvel / área inspecionada (HTML)"
    # e "Seção 6 Registro fotográfico e análise de conformidades (HTML)".
    # =======================================================================
    story += [Spacer(1, 6), heading("5 DESCRIÇÃO DO IMÓVEL / ÁREA INSPECIONADA"),
              pairs([(DETAIL_FIELDS[key], _or_blank(details[key])) for key in PROPERTY_FIELDS]),
              p("Dados não informados não são estimados pelo sistema."), PageBreak(),
              heading("6 REGISTRO FOTOGRÁFICO E ANÁLISE DE CONFORMIDADES"),
              p("Cada item corresponde a uma categoria da lista de verificação com ocorrência confirmada na revisão das imagens. "
                f'Os campos de avaliação estão preenchidos como "{PENDING_TEXT}" até a validação do responsável técnico.')]
    if not report["items"]:
        story.append(p("Nenhuma ocorrência confirmada nesta revisão."))
    # Itens 6.N: título, pergunta (negrito), constatação, notas, figuras, subsídio de IA e
    # quadro com os cinco campos pendentes.
    # EXTENSAO: bloco novo dentro de cada item = mais um "story += [...]" neste laço, igual ao HTML.
    for item in report["items"]:
        story += [heading(f"6.{item['number']} {item['label']}", 1), p(item["question"], ParagraphStyle("q", parent=base, fontName="Helvetica-Bold")),
                  p(item["statement"])]
        story += [p(f"Nota da revisão: {note}") for note in item["notes"]]
        story += [figure_block(figures[number]) for number in item["figures"]]
        if item["ai_support"]:
            lines = "<br/>".join(f"• Figura {row['figure']}, demarcação {row['box']}: {html.escape(row['text'])}"
                                 + (f" ({html.escape(row['model'])})" if row.get("model") else "") for row in item["ai_support"])
            story.append(Paragraph("<b>Subsídio de IA</b> (descrição automática dos recortes por modelo de visão e linguagem; não constitui "
                                   f"constatação técnica):<br/>{lines}", ai_style))
        # Quadro de avaliação: mesma tupla de rótulos de _evaluation_html (mude nos dois).
        story += [pairs([(label, pending()) for label in ("Conformidade", "Risco (gravidade)", "Diagnóstico", "Recomendação", "Prioridade (NBR 16747)")], 0.3),
                  Spacer(1, 10)]
    # 6.N+1 Categorias sem ocorrência e 6.N+2 Itens não avaliados (mesmo "tail" do HTML).
    tail = len(report["items"]) + 1
    story += [heading(f"6.{tail} Categorias sem ocorrência confirmada", 1),
              p("Não há demarcação confirmada destas categorias nas fotografias revisadas. A ausência nas imagens não comprova a ausência "
                "na cobertura; a conformidade depende do responsável técnico."),
              table([["Item", "Pergunta", "Conformidade"]] + [[row["label"], row["question"], pending()] for row in report["items_without_occurrence"]],
                    [width * 0.28, width * 0.47, width * 0.25]),
              heading(f"6.{tail + 1} Itens não avaliados pela detecção automática", 1),
              p("Os itens abaixo constam da lista de verificação do laudo de referência, mas não são cobertos pelas categorias do sistema "
                "e não foram avaliados neste documento."),
              table([["Pergunta", "Situação"]] + [[question, p("não avaliado pelo sistema", pending_style)] for question in report["manual_checklist"]],
                    [width * 0.72, width * 0.28]), PageBreak(),
              # =======================================================================
              # Seção 7 Quadro-resumo (PDF)
              # Mesmo trecho no HTML: render_html, bloco "Seção 7 Quadro-resumo das constatações (HTML)".
              # EXTENSAO: coluna nova = cabeçalho, valor da linha, célula do total e uma fração
              # nova na tupla de larguras (a soma deve continuar 1.0).
              # =======================================================================
              heading("7 QUADRO-RESUMO DAS CONSTATAÇÕES"),
              table([["Item", "Categoria", "Demarc.", "Fotos", "Figuras", "Conformidade", "Prioridade", "Recomendação"]]
                    + [[f"6.{item['number']}", item["label"], item["boxes"], item["images"], ", ".join(map(str, item["figure_refs"])),
                        pending(), pending(), pending()] for item in report["items"]]
                    + [["Total", "", summary["confirmed_boxes"], summary["positive"], "", "", "", ""]],
                    [width * w for w in (0.07, 0.2, 0.08, 0.07, 0.1, 0.16, 0.16, 0.16)]),
              p("Contagem de demarcações por fotografia; fotografias semelhantes podem repetir o mesmo elemento.", cell), Spacer(1, 10),
              # =======================================================================
              # Seção 8 Recomendações gerais (PDF) (a lista vem logo abaixo)
              # Mesmo trecho no HTML: render_html, bloco "Seção 8 Recomendações gerais (HTML)".
              # =======================================================================
              heading("8 RECOMENDAÇÕES GERAIS")]
    story += [p(f"• {text}") for text in GENERAL_RECOMMENDATIONS]
    # Fim da Seção 8 e Seção 9 Considerações finais (PDF)
    # Mesmo trecho no HTML: render_html, blocos "Seção 8 ..." e "Seção 9 Considerações finais (HTML)".
    story += [Paragraph(f"Recomendações específicas por item: <i>{PENDING_TEXT}</i>.", base), heading("9 CONSIDERAÇÕES FINAIS"),
              p(report["conclusion"]["facts"]), Paragraph(f"Conclusão técnica: <i>{PENDING_TEXT}</i>.", base)]
    # =======================================================================
    # Seção 10 Responsabilidade técnica (PDF)  e  Referências (PDF)
    # Mesmo trecho no HTML: render_html, blocos "Seção 10 Responsabilidade técnica (HTML)" e
    # "Referências (HTML)".
    # =======================================================================
    responsible = report["technical_responsibility"]
    story += [heading("10 RESPONSABILIDADE TÉCNICA"), p(responsible["statement"]), Spacer(1, 26 * mm),
              p("_" * 50, center), p(responsible["name"] or "Responsável técnico", center),
              p(responsible["registry"] or "Registro profissional: ____________", center), p(responsible["title"] or "", center),
              p(f"ART nº {responsible['art'] or '____________'}", center), PageBreak(),
              heading("REFERÊNCIAS")]
    story += [p(reference_text(norm), ParagraphStyle("ref", parent=base, alignment=0)) for norm in sorted(report["normative_references"], key=reference_text)]
    # =======================================================================
    # Anexo A (PDF)
    # Mesmo trecho no HTML: render_html, bloco "Anexo A Fotografias sem ocorrências confirmadas (HTML)".
    # =======================================================================
    story += [PageBreak(), heading("ANEXO A — Fotografias sem ocorrências confirmadas")]
    others = [image for image in report["images"] if image["status"] != "positive"]
    if others:
        story.append(table([["Foto", "Arquivo", "Status", "Origem", "Demarc. não confirmadas", "Notas"]]
                           + [[f"{image['numero_foto']:03d}", image["filename"], image["status_label"], image["source_label"], image["boxes"],
                               image["notes"] or "—"] for image in others], [width * w for w in (0.07, 0.22, 0.15, 0.2, 0.1, 0.26)]))
        story += [figure_block(figure) for figure in report["figures"] if figure["item"] is None]
    else:
        story.append(p("Todas as fotografias do registro têm ocorrências confirmadas."))
    # =======================================================================
    # Anexo B (PDF)
    # Mesmo trecho no HTML: render_html, bloco "Anexo B Sugestões automáticas pendentes (HTML)".
    # =======================================================================
    pending_suggestions = report["pending_suggestions"]
    story += [Spacer(1, 10), heading("ANEXO B — Sugestões automáticas pendentes"), p(pending_suggestions["note"])]
    if pending_suggestions["by_class"]:
        story.append(table([["Categoria", "Sugestões pendentes", "Fotografias"]]
                           + [[row["label"], row["count"], row["images"]] for row in pending_suggestions["by_class"]]
                           + [["Total", pending_suggestions["total"], ""]], [width * 0.5, width * 0.25, width * 0.25]))
    else:
        story.append(p("Nenhuma sugestão automática pendente nesta revisão."))
    # =======================================================================
    # Anexo C (PDF)
    # Mesmo trecho no HTML: render_html, bloco "Anexo C Subsídios de modelos de IA (HTML)".
    # =======================================================================
    story += [Spacer(1, 10), heading("ANEXO C — Subsídios de modelos de IA"), p(f"Detector de candidatas: {model_text}.")]
    if ai["available"]:
        story.append(p(f"{ai['warning']} Modelo: {ai.get('model') or '—'}; recortes descritos: {ai.get('described')} de {ai.get('total_boxes')}; "
                       f"concordância global com a classe humana: {_pct(ai.get('agreement_rate'))}."))
        if ai["per_class"]:
            story.append(table([["Categoria", "Descritas", "Concordâncias", "Taxa"]]
                               + [[row["label"], row["boxes"], row["agree"], _pct(row["agreement_rate"])] for row in ai["per_class"]],
                               [width * 0.46, width * 0.18, width * 0.18, width * 0.18]))
    else:
        story.append(p(ai["note"]))
    # =======================================================================
    # Anexo D (PDF)
    # Mesmo trecho no HTML: render_html, bloco "Anexo D Demarcações por fotografia (HTML)".
    # =======================================================================
    rows = sorted(report["occurrences"] + report["draft_boxes"], key=lambda value: (value["numero_foto"], value["caixa"]))
    story += [PageBreak(), heading("ANEXO D — Demarcações por fotografia")]
    if rows:
        story.append(table([["Foto", "Fig.", "Nº", "Categoria", "Coordenadas", "Status", "Origem", "Nota"]]
                           + [[f"{row['numero_foto']:03d}", row.get("figura") or "—", row["caixa"], row["classe"],
                               "{}, {} – {}, {}".format(*row["bbox_xyxy"]), row["status_foto"], row["origem"], row["nota"] or "—"] for row in rows],
                           [width * w for w in (0.07, 0.06, 0.05, 0.2, 0.2, 0.11, 0.14, 0.17)]))
    else:
        story.append(p("Sem demarcações nesta revisão."))
    # =======================================================================
    # Anexo E (PDF)
    # Mesmo trecho no HTML: render_html, bloco "Anexo E Apoio ao responsável técnico e rastreabilidade (HTML)".
    # =======================================================================
    story += [Spacer(1, 10), heading("ANEXO E — Apoio ao responsável técnico e rastreabilidade"),
              p("Referências do laudo-modelo do projeto (CEASA Minas, Pavilhão A) e gravidade de referência da taxonomia. São apoio à "
                "redação e não foram aplicadas às constatações deste laudo."),
              table([["Item", "Gravidade de referência (não aplicada)", "Laudo-modelo: conformidade", "Laudo-modelo: recomendação"]]
                    + [[f"6.{item['number']} {item['label']}", item["reference_severity"] or "—", (item["model_reference"] or {}).get("conformidade") or "—",
                        (item["model_reference"] or {}).get("recomendacao") or "—"] for item in report["items"]],
                    [width * 0.25, width * 0.2, width * 0.2, width * 0.35]), Spacer(1, 6),
              pairs([("Inspeção", registry_label(header)), ("SHA-256 do registro", header["registry_sha256"]),
                     ("Taxonomia", f"{header['taxonomy_version'] or '—'} · {header['taxonomy_sha256']}"), ("Modelo das sugestões", model_text),
                     ("Gerador", report["provenance"]["generator"]), ("Gerado em", report["provenance"]["generated_at"])]),
              p("Edições das normas sem ano na seção 3: confirmar a edição vigente no Catálogo ABNT antes da emissão.", cell)]

    # -------------------------------------------------------------------
    # Cabeçalho/rodapé das páginas e geração final
    # -------------------------------------------------------------------
    def numbered(canvas, document):
        """Desenha em cada página (exceto a capa) o número no canto superior direito e o rodapé.

        Parâmetros: canvas (tela do reportlab), document (o Document, com page e margens).
        """
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(A4[0] - document.rightMargin, A4[1] - 18 * mm, str(document.page))
        canvas.setFont("Helvetica", 7)
        canvas.drawString(document.leftMargin, 10 * mm, f"{header['title']} — {PRELIMINARY_TEXT}")
        canvas.restoreState()

    def cover(canvas, document):
        """Capa: nada é desenhado (sem número de página, como pede a NBR 10719)."""
        pass

    # multiBuild repete a montagem até o sumário estabilizar (números de página corretos).
    # A primeira página (capa) usa "cover"; as demais, "numbered".
    doc.multiBuild(story, onFirstPage=cover, onLaterPages=numbered)
