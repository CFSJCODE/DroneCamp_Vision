"""Registros de revisão: propostas, decisões humanas e migrações de taxonomia.

Função no projeto: mantém os ``registry.json`` de ``data/reviews/<versão>/``,
que dizem, para cada foto, as caixas, o status (positiva/negativa/ambígua) e
quem decidiu. O treino só aprende com o que um humano aprovou aqui.

O que faz:
- ``review_image_size`` / ``validate_boxes`` / ``boxes_to_yolo``: utilidades de
  foto e caixa usadas em todo o projeto.
- ``build_review_registry`` / ``apply_visual_audits``: montam a revisão do CEASA a
  partir de propostas e segunda leitura por IA (comando ``prepare-review``).
- ``import_human_feedback``: importa o JSON exportado pela página (``import-review``).
- ``apply_ai_proposals``: leva propostas visuais de IA como candidatas (``add-ai-proposals``).
- ``migrate_review_taxonomy``: acrescenta classes ou funde classes (mapa da taxonomia
  nova, ex.: v6 → v7) preservando caixas (``migrate-review``).

Toda saída é uma versão nova: registros anteriores nunca são sobrescritos.
Propostas visuais nunca viram aprovação humana sozinhas.

Quando mexer: para mudar o que o arquivo exportado pela página pode conter ou
as regras de aprovação (ex.: exigir ``confirmed_complete``).

Funções públicas (em ordem no arquivo):
- ``review_image_size(path)``: largura e altura da foto; recusa rotação EXIF.
- ``validate_boxes(boxes, width, height, class_count)``: valida a lista de caixas.
- ``boxes_to_yolo(boxes, width, height)``: texto de um ``.txt`` YOLO.
- ``build_review_registry(config, directory)``: revisão inicial do CEASA.
- ``write_review_summary(directory, registry)``: grava ``review_summary.json``.
- ``apply_visual_audits(config, directory)``: segunda leitura por IA.
- ``import_human_feedback(config, registry_path, feedback_path, output)``: decisões humanas.
- ``apply_ai_proposals(config, registry_path, proposals_path, output)``: propostas de IA.
- ``migrate_review_taxonomy(config, registry_path, previous_taxonomy, output)``: migração.

Quem chama:
- ``cli.py``: ``prepare-review`` (build_review_registry + apply_visual_audits),
  ``import-review``, ``add-ai-proposals`` e ``migrate-review``.
- As utilidades de foto e caixa são importadas por ``review_render``, ``review_dataset``,
  ``review_provenance``, ``suggestions``, ``pilot``, ``tiling``, ``exports``, ``ingest``,
  ``calibration``, ``model_gate`` e ``autolabel``.

O que lê e grava no disco:
- Lê ``data/reference/ceasa/manifest.json`` e as fotos originais (só em prepare-review),
  ``<revisão>/proposals/*.json`` e ``<revisão>/audits/*.json``, o ``registry.json`` de
  origem, o JSON exportado pela página e ``configs/taxonomy.json`` (para o hash).
- Grava ``registry.json`` (ou ``registry_second_pass.json`` / o caminho de ``output``)
  e ``review_summary.json`` na pasta da revisão. Nunca apaga nem sobrescreve uma versão
  anterior: quem chama escolhe uma pasta/arquivo de saída novo.

Formato do ``registry.json`` (o "contrato" lido e validado neste arquivo)
-------------------------------------------------------------------------
Nível superior (dict):
- ``schema_version`` (int): sempre ``REVIEW_SCHEMA`` (1).
- ``version`` (str): nome da pasta da revisão (ex.: ``ceasa_v9_rev001``).
- ``display_name`` (str, opcional): nome digitado no envio pela plataforma; é removido
  ao gerar uma versão derivada.
- ``taxonomy_sha256`` (str): SHA-256 de ``configs/taxonomy.json`` com que as caixas
  foram feitas. ATENCAO: quase todo comando recusa o registro se este hash não bater
  com a taxonomia atual (use migrate-review).
- ``taxonomy_version`` (str): campo ``version`` da taxonomia.
- ``scope`` (str): texto curto que diz o que a versão contém.
- ``sources`` (list): lotes de propostas de origem (``file``/``sha256``/``annotator``) ou
  pasta de fotos novas (``folder``/``files``).
- Opcionais conforme a etapa: ``source_manifest_sha256``, ``source_pdf_sha256`` (CEASA),
  ``audits`` (segunda leitura), ``human_feedback`` (import-review), ``ai_proposals``
  (add-ai-proposals), ``taxonomy_migration`` e ``taxonomy_migration_history``
  (migrate-review), ``parent_registry_version`` (versão de onde esta saiu),
  ``additional_visual_candidates`` (dados históricos do CEASA).
- ``images`` (list): uma entrada por foto ÚNICA (identificada pelo hash).

Cada foto em ``images`` (dict):
- ``image_sha256`` (str, 64 hex minúsculos): SHA-256 dos bytes do original; é a
  identidade da foto em todo o projeto (nomes de arquivo, labels, sugestões).
- ``filename`` (str): nome original da foto, só para exibição.
- ``pages`` (list[int]): páginas do PDF do relatório onde a foto aparece ([] em fotos novas).
- ``width`` / ``height`` (int): tamanho em pixels do original, sem rotação EXIF.
- ``source_path`` (str): caminho absoluto do original (reancorado por
  ``io.resolve_local_path`` quando o registro veio de outro PC). ``original_path``
  (opcional): de onde a foto foi copiada.
- ``status`` (str): um de ``IMAGE_STATUSES``. ``positive`` exige caixas; ``negative``
  proíbe caixas; ``ambiguous`` = pendente; ``excluded`` = fora do treino.
- ``boxes`` (list): caixas (ver abaixo).
- ``annotator`` (str): quem propôs as caixas (IA ou ``sem_anotacao_humana``).
- ``notes`` (str): observações da proposta.
- ``visual_review_status`` (str): ``first_pass_ai``, ``second_pass_ai``,
  ``model_suggestions_only``, ``prior_taxonomy_human_review``, ``prior_taxonomy_ai_review``.
- ``technical_status`` (str): ``pending_human_review`` ou ``human_visual_reviewed``.
- ``human_approved`` (bool): True SOMENTE após import-review com positiva/negativa
  confirmada. ATENCAO: é a única porta de entrada para o dataset de produção.
- ``human_reviewer`` / ``human_notes`` / ``human_review_at``: autor, notas e data ISO
  da decisão humana (import-review).
- ``building_group`` (str): edificação/campanha (define o split no dataset de produção);
  ``scene_group`` (opcional): agrupa cenas parecidas (usado pelo piloto).
- ``severity``: sempre ``None``. ATENCAO: gravidade nunca é inferida por este sistema.
- ``taxonomy_recheck_pending`` (bool) e ``classes_reviewed_before_migration`` (list[int]):
  pendência de rever classes acrescentadas (ou ampliadas por fusão) na migração; os IDs
  são sempre da taxonomia ATUAL do registro.
- Campos de histórico gravados pela migração: ``previous_status``,
  ``previous_visual_review_status``, ``historical_visual_review_status``,
  ``previous_human_approved``, ``historical_human_review``.
- Segunda leitura: ``second_reviewer``, ``second_review_notes``.
- Outros campos existentes nos dados (``quality_flags``, ``review_focus_note``,
  ``additional_candidate_note``) são preservados sem validação.

Cada caixa em ``boxes`` (dict):
- ``class_id`` (int): ID da classe ativa (fase 1) em ``configs/taxonomy.json``, 0..N-1.
- ``bbox_xyxy`` (list de 4 números): ``[x1, y1, x2, y2]`` em pixels do original, com
  origem no canto superior esquerdo; exige ``0 <= x1 < x2 <= width`` e
  ``0 <= y1 < y2 <= height``.
- Opcionais (não validados aqui, só preservados): ``note`` (texto),
  ``identity_verified``, ``proposed_new_class`` (slug de classe sugerida por IA;
  validado em apply_ai_proposals), ``source``/``confidence``/``acceptance_probability``
  quando a caixa veio de uma sugestão aceita na página; ``attributes`` (dict de
  detalhes opcionais da caixa, ex.: ``{"material": "telha"}``, valores definidos em
  ``attributes`` da classe na taxonomia) e ``previous_class_id``/``previous_class_slug``
  (classe da taxonomia anterior, gravados pela migração com fusão).

Onde mexer:
- Para aceitar um novo status de foto: acrescente em ``IMAGE_STATUSES`` e confira as
  regras positivo/negativo em cada função (e o select de decisão em platform.js).
- Para exigir/aceitar um campo novo no arquivo exportado pela página: veja o laço
  "2." de ``import_human_feedback`` (marcador EXTENSAO) e a função que monta a
  exportação em ``review_templates/platform.js``.
- Para um novo campo por caixa que precise ser validado: ``validate_boxes``.
- Para mudar o resumo ``review_summary.json``: ``write_review_summary``.
- Para outro tipo de migração: ``migrate_review_taxonomy`` aceita ACRÉSCIMO de classes
  ou FUSÃO declarada em ``migration.active_class_map`` da taxonomia nova
  (``config.active_class_remap``). Divisão de uma classe em duas não é automática: exige
  redesenho humano e uma função nova.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import math
import re

from PIL import Image

from .config import ProjectConfig, active_class_remap, detection_names, load_taxonomy
from .io import file_hash, resolve_local_path, write_json

# ---------------------------------------------------------------------------
# Constantes do formato dos registros
# ---------------------------------------------------------------------------

# Versão do formato dos registros e status possíveis de uma foto.
# REVIEW_SCHEMA: número gravado em "schema_version" de registry.json, dos lotes de
# propostas/auditorias e do JSON exportado pela página. ATENCAO: mudar este valor faz
# todos os registros existentes serem recusados; só aumente junto com um conversor.
REVIEW_SCHEMA = 1
# IMAGE_STATUSES: triagem de uma foto. positive = tem ocorrência (exige caixa);
# negative = sem ocorrência (proíbe caixa); ambiguous = indecisa/pendente;
# excluded = fora do uso (foto ruim, repetida etc.).
# EXTENSAO: novo status: acrescente aqui, revise as regras positivo/negativo nas funções
# deste arquivo, o filtro de aprovação em review_dataset._approved_images e o select de
# decisão em review_templates/platform.js / index.html.
IMAGE_STATUSES = {"positive", "negative", "ambiguous", "excluded"}


# ---------------------------------------------------------------------------
# Utilidades de foto e caixa (usadas também por pilot.py e suggestions.py).
# ---------------------------------------------------------------------------

def review_image_size(path: Path) -> tuple[int, int]:
    """Use coordenadas nos pixels originais, sem rotação EXIF implícita.

    Fotos de celular com rotação declarada precisam de uma cópia canonicalizada
    e de novo hash antes da anotação; nunca gire silenciosamente uma revisão.

    Parâmetros:
        path (Path): caminho de uma foto local (jpg, png...).

    Retorna:
        tuple[int, int]: ``(largura, altura)`` em pixels, como gravadas no arquivo.

    Erros:
        ValueError("Orientação EXIF exige canonicalização...") quando a tag EXIF 274
        (Orientation) é diferente de 1 (foto "deitada" que o visualizador giraria).
        Erros do Pillow (arquivo corrompido, formato desconhecido) sobem sem tratamento.

    Chamada por: praticamente todo módulo que lê fotos (registros, página, datasets,
    sugestões, provenance).

    ATENCAO: se esta checagem fosse removida, as caixas desenhadas no navegador (que
    aplica a rotação EXIF) não coincidiriam com os pixels usados no treino.
    """
    with Image.open(path) as image:
        # Tag 274 = Orientation; 1 = normal. Qualquer outro valor significa rotação/espelho.
        if image.getexif().get(274, 1) != 1:
            raise ValueError("Orientação EXIF exige canonicalização da foto antes da revisão.")
        # load() força a leitura dos pixels: arquivo truncado falha aqui, não mais tarde.
        image.load()
        return image.size


def validate_boxes(boxes: list[dict], width: int, height: int, class_count: int) -> None:
    """Rejeite caixas inválidas antes de desenhar, importar feedback ou treinar.

    Parâmetros:
        boxes (list[dict]): caixas no formato do registro (``class_id`` e ``bbox_xyxy``;
            veja o cabeçalho do módulo). Campos extras são ignorados.
        width, height (int): tamanho da foto em pixels (limites da geometria).
        class_count (int): número de classes ativas; IDs válidos são 0..class_count-1.

    Retorna:
        None. Só retorna se TODAS as caixas forem válidas.

    Erros:
        ValueError com o número da caixa (a partir de 1) quando: a lista não é lista
        ou tem mais de 2000 itens; a caixa não é dict; ``class_id`` não é int ou está
        fora do intervalo; ``bbox_xyxy`` não tem 4 números finitos; ou a caixa sai da
        imagem / tem área zero (x1 >= x2 ou y1 >= y2).

    Chamada por: este módulo, review_render, review_dataset, review_provenance,
    suggestions.detect_boxes, pilot, exports.

    EXTENSAO: nova regra por caixa (ex.: tamanho mínimo, campo obrigatório): acrescente
    um ``if`` dentro do laço, depois das checagens existentes. Lembre que ela passa a
    valer para registros antigos também (render-review e build-reviewed-data podem
    começar a recusá-los).
    """
    # Limite de 2000 caixas por foto: protege contra arquivos malformados/gigantes.
    if not isinstance(boxes, list) or len(boxes) > 2000:
        raise ValueError("boxes deve ser uma lista com no máximo 2000 ocorrências.")
    # Confere cada caixa; "position" começa em 1 para a mensagem bater com a página.
    for position, box in enumerate(boxes, start=1):
        if not isinstance(box, dict):
            raise ValueError(f"Caixa {position}: esperado um objeto com classe e coordenadas.")
        class_id = box.get("class_id")
        coordinates = box.get("bbox_xyxy")
        # type(...) is not int recusa bool (True é int em Python) e floats como 1.0.
        if type(class_id) is not int or not 0 <= class_id < class_count:
            raise ValueError(f"Caixa {position}: ID de classe inválido.")
        if not isinstance(coordinates, list) or len(coordinates) != 4:
            raise ValueError(f"Caixa {position}: esperado [x1,y1,x2,y2].")
        # Só int/float finitos (sem NaN, infinito, texto ou bool).
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in coordinates):
            raise ValueError(f"Caixa {position}: coordenadas devem ser números finitos.")
        x1, y1, x2, y2 = coordinates
        # Caixa dentro da foto e com largura/altura positivas (canto sup. esq. -> inf. dir.).
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Caixa {position}: geometria fora da imagem ou área vazia.")


def boxes_to_yolo(boxes: list[dict], width: int, height: int) -> str:
    """A conversão não aprova a anotação; a proveniência define seu uso.

    Formato YOLO: uma linha por caixa, ``classe cx cy largura altura`` normalizados (0–1).

    Parâmetros:
        boxes (list[dict]): caixas já validadas (``class_id``, ``bbox_xyxy`` em pixels).
        width, height (int): tamanho da foto em pixels (divisores da normalização).

    Retorna:
        str: conteúdo do ``.txt`` YOLO, com 8 casas decimais e "\\n" no fim de cada
        linha; texto vazio ("") para foto negativa (sem caixas).

    Chamada por: review_render (labels propostas), review_dataset (labels de produção),
    review_provenance (recalcula e compara byte a byte), pilot, tiling.

    ATENCAO: review_provenance compara o .txt gravado com o texto gerado aqui. Mudar o
    número de casas ou o separador invalida todos os datasets de produção já gerados.
    """
    lines = []
    for box in boxes:
        x1, y1, x2, y2 = box["bbox_xyxy"]
        # xyxy em pixels -> YOLO normalizado: centro = média das bordas / dimensão;
        # largura/altura = diferença das bordas / dimensão.
        lines.append(f"{box['class_id']} {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
    # Quebra de linha final só quando há caixas (arquivo vazio = negativa).
    return "\n".join(lines) + ("\n" if lines else "")


# ---------------------------------------------------------------------------
# Revisão inicial do CEASA: propostas por IA e segunda leitura (prepare-review).
# ---------------------------------------------------------------------------

# ENTRADA: comando prepare-review (cli.py) chama build_review_registry e, com --audits,
# apply_visual_audits logo em seguida.
def build_review_registry(config: ProjectConfig, directory: Path) -> dict:
    """Consolide lotes sem perder duplicatas documentais ou inventar decisões.

    Junta as fotos únicas do relatório do CEASA (``manifest.json``) com os lotes de
    propostas de IA em ``<directory>/proposals/*.json`` e grava o primeiro
    ``registry.json`` da revisão. Nenhuma foto sai aprovada.

    Parâmetros:
        config (ProjectConfig): configuração do projeto (raiz e caminho da taxonomia).
        directory (Path): pasta da revisão (ex.: ``data/reviews/ceasa_v1``); precisa
            conter ``proposals/``. O nome da pasta vira ``version``.

    Formato de cada lote em ``proposals/*.json``: ``{schema_version: 1, annotator: str,
    taxonomy_sha256?: str, images: [{image_sha256, width, height, status, boxes, notes?,
    ...}]}``. Os campos de cada imagem são copiados para o registro.

    Retorna:
        dict: o registro gravado (formato no cabeçalho do módulo).

    Erros (ValueError): schema diferente; lote de outra taxonomia; lote antigo sem
    hash com taxonomia de mais de 8 classes; foto desconhecida ou proposta duas vezes;
    original alterado; dimensões divergentes; status inválido; caixas inválidas;
    positiva sem caixa / negativa com caixa; alguma foto do manifesto sem proposta.

    Grava: ``<directory>/registry.json`` e ``<directory>/review_summary.json``.
    ATENCAO: esta função SOBRESCREVE ``registry.json`` da pasta indicada; use só em
    uma pasta nova de revisão.
    """
    # 1. Manifesto do relatório: agrupa as ocorrências (páginas) de cada foto pelo hash.
    #    Uma mesma foto pode aparecer em várias páginas do PDF; no registro ela é única.
    manifest_path = config.root / "data/reference/ceasa/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    originals = manifest_path.parent
    occurrences = defaultdict(list)
    for image in manifest["images"]:
        occurrences[image["sha256"]].append(image)
    proposals = {}
    taxonomy = load_taxonomy(config.taxonomy_path)
    class_count = len(detection_names(taxonomy))
    sources = []
    # 2. Cada lote de propostas, em ordem alfabética de arquivo (resultado determinístico).
    for path in sorted((directory / "proposals").glob("*.json")):
        batch = json.loads(path.read_text(encoding="utf-8"))
        if batch.get("schema_version") != REVIEW_SCHEMA:
            raise ValueError(f"Schema inválido em {path.name}.")
        # O lote precisa ser da taxonomia atual (hash), senão os IDs de classe mudariam de sentido.
        declared_taxonomy = batch.get("taxonomy_sha256")
        if declared_taxonomy and declared_taxonomy != file_hash(config.taxonomy_path):
            raise ValueError("Proposta pertence a outra taxonomia; faça migração explícita.")
        # ATENCAO: "8" é o número de classes da taxonomia original do CEASA. Lotes antigos
        # (sem hash) só valem nela; com mais classes é preciso migrate-review.
        if not declared_taxonomy and class_count > 8:
            raise ValueError("Lote histórico não declara as novas classes. Use migrate-review com a taxonomia anterior.")
        sources.append({"file": path.name, "sha256": file_hash(path), "annotator": batch["annotator"]})
        # 2a. Cada foto do lote: conhecida, única, original intacto, geometria e status coerentes.
        for item in batch["images"]:
            digest = item["image_sha256"]
            if digest not in occurrences or digest in proposals:
                raise ValueError(f"Imagem desconhecida ou proposta duplicada: {digest}.")
            # Usa a primeira ocorrência do manifesto como arquivo de referência.
            first = occurrences[digest][0]
            path_original = originals / first["filename"]
            if file_hash(path_original) != digest:
                raise ValueError(f"Original alterado: {first['filename']}.")
            width, height = review_image_size(path_original)
            if (item["width"], item["height"]) != (width, height):
                raise ValueError(f"Dimensões da proposta divergem da foto: {first['filename']}.")
            if item["status"] not in IMAGE_STATUSES:
                raise ValueError("Status de triagem inválido.")
            validate_boxes(item["boxes"], width, height, class_count)
            if item["status"] == "positive" and not item["boxes"]:
                raise ValueError("Imagem positiva precisa de caixas.")
            if item["status"] == "negative" and item["boxes"]:
                raise ValueError("Imagem negativa não pode conter caixas.")
            # 2b. Entrada do registro: campos do lote + campos fixos de uma proposta de IA.
            #     human_approved=False e severity=None são deliberados (IA não aprova,
            #     gravidade nunca é inferida).
            # EXTENSAO: novo campo padrão por foto do CEASA: acrescente neste dict (e, se a
            # página precisar dele, em platform.js). building_group fixo = pavilhão do CEASA.
            proposals[digest] = {
                **item, "filename": first["filename"], "width": width, "height": height,
                "pages": sorted({occurrence["page"] for occurrence in occurrences[digest]}),
                "annotator": batch["annotator"], "visual_review_status": "first_pass_ai",
                "technical_status": "pending_human_review", "human_approved": False,
                "building_group": "ceasa_pavilhao_a", "source_path": str(path_original),
                "severity": None,
            }
    # 3. Toda foto única do manifesto precisa ter recebido uma proposta.
    missing = set(occurrences) - set(proposals)
    if missing:
        raise ValueError(f"Revisão incompleta: faltam {len(missing)} imagens únicas.")
    # 4. Registro final: fotos ordenadas pela primeira página e depois pelo nome.
    registry = {
        "schema_version": REVIEW_SCHEMA, "version": directory.name,
        "taxonomy_sha256": file_hash(config.taxonomy_path), "taxonomy_version": load_taxonomy(config.taxonomy_path)["version"],
        "source_manifest_sha256": file_hash(manifest_path), "source_pdf_sha256": manifest["source_pdf_sha256"],
        "sources": sources, "images": sorted(proposals.values(), key=lambda item: (min(item["pages"]), item["filename"])),
        "scope": "propostas_visuais_por_ia_pendentes_de_validacao_tecnica",
    }
    # 5. Grava o registro e o resumo de contagens.
    write_json(directory / "registry.json", registry)
    write_review_summary(directory, registry)
    return registry


def write_review_summary(directory: Path, registry: dict) -> dict:
    """review_summary.json: contagens de fotos, caixas por classe e aprovações.

    Parâmetros:
        directory (Path): pasta da revisão onde o resumo é gravado.
        registry (dict): registro já carregado (formato no cabeçalho do módulo).

    Retorna:
        dict: o resumo gravado. Chaves principais: ``unique_images``,
        ``image_status_counts`` ({status: fotos}), ``boxes_by_class_id`` ({id: caixas}),
        ``total_proposed_boxes``, ``human_approved_images``, contagens de revisões
        históricas/segunda leitura, ``taxonomy_recheck_pending_images``,
        ``independent_buildings`` e ``ready_for_independent_evaluation`` (sempre False).

    Grava: ``<directory>/review_summary.json`` (sobrescreve o resumo anterior da pasta).

    Chamada por: build_review_registry, apply_visual_audits, apply_ai_proposals,
    migrate_review_taxonomy e review_render.render_review_package.

    EXTENSAO: nova contagem no resumo: acrescente uma chave no dict ``summary``. Se o
    relatório Markdown precisar dela, use-a em review_render.render_review_package (etapa 5).
    """
    # Fotos por status e caixas por classe (somando todas as fotos).
    counts = Counter(item["status"] for item in registry["images"])
    per_class = Counter(box["class_id"] for item in registry["images"] for box in item["boxes"])
    summary = {
        "unique_images": len(registry["images"]), "image_status_counts": dict(counts),
        "boxes_by_class_id": dict(per_class), "total_proposed_boxes": sum(per_class.values()),
        # sum() sobre booleanos conta quantos são True.
        "human_approved_images": sum(bool(item.get("human_approved")) for item in registry["images"]),
        # Revisão humana feita na taxonomia anterior (preservada pela migração).
        "historical_human_review_images": sum(
            item.get("visual_review_status") == "prior_taxonomy_human_review"
            and item.get("previous_human_approved") is True for item in registry["images"]),
        "second_visual_review_images": sum(item.get("visual_review_status") == "second_pass_ai" for item in registry["images"]),
        # Segunda leitura de IA feita antes de alguma migração (o status atual foi trocado).
        "prior_taxonomy_second_visual_review_images": sum(
            item.get("previous_visual_review_status") == "second_pass_ai"
            or item.get("historical_visual_review_status") == "second_pass_ai"
            for item in registry["images"]),
        "taxonomy_recheck_pending_images": sum(bool(item.get("taxonomy_recheck_pending")) for item in registry["images"]),
        # Quantas edificações distintas: medida de diversidade para avaliação independente.
        "independent_buildings": len({item["building_group"] for item in registry["images"]}),
        # ATENCAO: fixo em False; uma revisão sozinha nunca é declarada pronta para avaliação.
        "ready_for_independent_evaluation": False,
        "note": "Propostas visuais são revisáveis; uma única edificação não mede generalização para outros locais.",
    }
    write_json(directory / "review_summary.json", summary)
    return summary


def apply_visual_audits(config: ProjectConfig, directory: Path) -> dict:
    """Uma segunda leitura de IA melhora a proposta, sem simular responsável humano.

    Lê ``<directory>/registry.json`` e aplica as auditorias de
    ``<directory>/audits/*.json`` (um segundo revisor de IA, diferente do anotador).

    Parâmetros:
        config (ProjectConfig): configuração (caminho da taxonomia).
        directory (Path): pasta da revisão já preparada por build_review_registry.

    Formato de cada auditoria: ``{schema_version: 1, reviewer: str, images:
    [{image_sha256, notes, status?, boxes?}]}``. Sem ``status``/``boxes``, os valores
    atuais da foto são mantidos.

    Retorna:
        dict: o registro após a segunda leitura (``visual_review_status="second_pass_ai"``,
        ``audits`` com a origem de cada arquivo).

    Erros (ValueError): taxonomia mudou; foto duplicada; schema inválido; foto
    desconhecida/repetida na auditoria; revisor igual ao anotador; status incompatível
    com caixas; alguma foto sem segunda leitura.

    Grava: ``registry_second_pass.json`` (o ``registry.json`` original fica intacto) e
    ``review_summary.json``.
    """
    # 1. Registro de origem, na taxonomia atual, sem fotos repetidas.
    registry_path = directory / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("taxonomy_sha256") != file_hash(config.taxonomy_path):
        raise ValueError("Taxonomia mudou: migre a revisão antes de auditá-la.")
    images = {item["image_sha256"]: item for item in registry["images"]}
    if len(images) != len(registry["images"]):
        raise ValueError("Imagem duplicada no registro de revisão.")
    seen = set()
    count = len(detection_names(load_taxonomy(config.taxonomy_path)))
    audit_sources = []
    # 2. Cada arquivo de auditoria e cada foto auditada.
    for path in sorted((directory / "audits").glob("*.json")):
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit.get("schema_version") != REVIEW_SCHEMA:
            raise ValueError(f"Schema inválido em {path.name}.")
        audit_sources.append({"filename": path.name, "sha256": file_hash(path), "reviewer": audit["reviewer"]})
        for review in audit["images"]:
            digest = review["image_sha256"]
            if digest in seen or digest not in images:
                raise ValueError("Imagem duplicada/desconhecida na segunda revisão.")
            seen.add(digest)
            image = images[digest]
            # ATENCAO: a segunda leitura só tem valor se vier de outro revisor.
            if audit["reviewer"] == image["annotator"]:
                raise ValueError("A segunda leitura precisa de outro revisor.")
            # Caixas novas substituem as anteriores (depois de validadas).
            if "boxes" in review:
                validate_boxes(review["boxes"], image["width"], image["height"], count)
                image["boxes"] = review["boxes"]
            status = review.get("status", image["status"])
            if status not in IMAGE_STATUSES or (status == "negative" and image["boxes"]):
                raise ValueError("Decisão de revisão incompatível com as caixas.")
            if status == "positive" and not image["boxes"]:
                raise ValueError("Revisão positiva sem ocorrência.")
            # human_approved continua False: IA nunca aprova. "notes" é obrigatório no arquivo.
            image.update(status=status, visual_review_status="second_pass_ai", second_reviewer=audit["reviewer"],
                         second_review_notes=review["notes"], human_approved=False)
    # 3. Toda foto precisa ter recebido segunda leitura.
    if seen != set(images):
        raise ValueError(f"Segunda leitura incompleta: {len(seen)}/{len(images)} imagens.")
    # 4. Grava em arquivo separado e atualiza o resumo.
    registry["audits"] = audit_sources
    write_json(directory / "registry_second_pass.json", registry)
    write_review_summary(directory, registry)
    return registry


# ---------------------------------------------------------------------------
# Decisões humanas exportadas pela página (import-review).
# ---------------------------------------------------------------------------

# ENTRADA: comando import-review (decisões exportadas pela página viram registro novo).
# ATENCAO: o arquivo exportado só vale para o registro de onde saiu (registry_sha256); se o
# registro mudou depois, exporte de novo pela página em vez de editar o hash à mão.
def import_human_feedback(config: ProjectConfig, registry_path: Path, feedback_path: Path, output: Path) -> dict:
    """Valide arquivo exportado pelo revisor e preserve o registro anterior.

    É o ÚNICO caminho pelo qual uma foto passa a ``human_approved=True``.

    Parâmetros:
        config (ProjectConfig): configuração (raiz do projeto e taxonomia).
        registry_path (Path): ``registry.json`` que estava aberto na página.
        feedback_path (Path): JSON exportado pela página (botão de exportar/enviar).
        output (Path): caminho do NOVO registro (em geral
            ``data/reviews/<nova_versão>/registry.json``); não pode existir.

    Formato do feedback (montado em review_templates/platform.js)::

        {"schema_version": 1, "registry_sha256": "<hash do registry.json>",
         "reviewer_id": "nome", "exported_at": "2026-10-12T14:00:00.000Z",
         "images": [{"image_sha256": "...", "status": "positive|negative|ambiguous|excluded",
                     "boxes": [{"class_id": 0, "bbox_xyxy": [x1, y1, x2, y2], ...}],
                     "notes": "texto", "confirmed_complete": true}]}

    Retorna:
        dict: o novo registro, com ``human_feedback`` (hash do feedback, revisor, fotos
        revisadas, hash do registro de origem) e ``parent_registry_version``.

    Erros (ValueError): schema inválido; saída já existe; feedback de outro registro
    (hash); revisor vazio ou com mais de 100 caracteres; taxonomia mudou; foto
    desconhecida/duplicada; original alterado; dimensões divergentes; status inválido;
    caixas inválidas; aprovação sem ``confirmed_complete``; status incompatível com caixas.

    Grava: só ``output``. ATENCAO: diferente de apply_ai_proposals, não atualiza
    ``review_summary.json``.
    """
    # 1. O feedback precisa ter sido gerado a partir desta versão exata do registro (hash).
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or registry.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError("Schema inválido no registro de revisão.")
    if not isinstance(feedback, dict):
        raise ValueError("Feedback deve ser um objeto JSON.")
    # ATENCAO: nunca sobrescrever uma versão existente.
    if output.exists():
        raise ValueError("Saída já existe; use uma nova versão da revisão.")
    if feedback.get("schema_version") != REVIEW_SCHEMA or feedback.get("registry_sha256") != file_hash(registry_path):
        raise ValueError("Feedback não corresponde à versão original da revisão.")
    # 1a. Identificador do revisor: texto não vazio de até 100 caracteres (sem espaços nas pontas).
    reviewer_value = feedback.get("reviewer_id", "")
    reviewer = reviewer_value.strip() if isinstance(reviewer_value, str) else ""
    if not reviewer or len(reviewer) > 100:
        raise ValueError("Informe um identificador de revisor válido.")
    # 1b. Taxonomia atual e lista de fotos do registro (indexada pelo hash).
    class_count = len(detection_names(load_taxonomy(config.taxonomy_path)))
    if registry["taxonomy_sha256"] != file_hash(config.taxonomy_path):
        raise ValueError("Taxonomia mudou: migre a revisão antes de aprovar.")
    if not isinstance(registry.get("images"), list) or any(not isinstance(item, dict) for item in registry["images"]):
        raise ValueError("O registro deve conter uma lista de imagens válida.")
    by_id = {item["image_sha256"]: item for item in registry["images"]}
    if len(by_id) != len(registry["images"]):
        raise ValueError("Imagem duplicada no registro de revisão.")
    # 2. Cada decisão: foto conhecida e intacta, caixas válidas, revisão integral confirmada.
    decisions = feedback.get("images")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("Feedback deve conter uma lista não vazia de decisões.")
    seen = set()
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ValueError("Cada decisão deve ser um objeto JSON.")
        digest = decision.get("image_sha256")
        if digest not in by_id or digest in seen:
            raise ValueError("Imagem desconhecida/duplicada no feedback.")
        seen.add(digest)
        item = by_id[digest]
        # 2a. A foto original (reancorada neste PC) precisa ser a mesma que foi revisada.
        source = resolve_local_path(item["source_path"], config.root)
        if file_hash(source) != digest:
            raise ValueError("Imagem original diverge do arquivo revisado.")
        if review_image_size(source) != (item["width"], item["height"]):
            raise ValueError("Dimensões do registro divergem da foto revisada.")
        # 2b. Status e caixas decididos pelo revisor.
        status = decision.get("status")
        if status not in IMAGE_STATUSES:
            raise ValueError("Status de feedback inválido.")
        boxes = decision.get("boxes", [])
        validate_boxes(boxes, item["width"], item["height"], class_count)
        # ATENCAO: só positive/negative contam como aprovação, e apenas com a caixa
        # "revisei a foto inteira" (confirmed_complete) marcada na página.
        approved = status in {"positive", "negative"}
        if approved and decision.get("confirmed_complete") is not True:
            raise ValueError("Aprovação exige confirmação explícita da revisão integral.")
        if (status == "negative" and boxes) or (status == "positive" and not boxes):
            raise ValueError("Status positivo/negativo incompatível com ocorrências.")
        # Positiva/negativa confirmada = aprovada para treino; ambígua continua pendente.
        # EXTENSAO: novo campo por decisão exportado pela página (ex.: qualidade da foto):
        # leia-o de "decision" aqui e acrescente no update abaixo; acrescente-o também na
        # exportação de review_templates/platform.js. severity fica sempre None
        # (ATENCAO: gravidade nunca é inferida nem importada).
        item.update(status=status, boxes=boxes, human_approved=approved,
                    technical_status="human_visual_reviewed" if approved else "pending_human_review",
                    human_reviewer=reviewer, human_notes=decision.get("notes", ""),
                    human_review_at=feedback.get("exported_at"), severity=None,
                    taxonomy_recheck_pending=not approved)
    # 3. Grava a nova versão com o hash do feedback e o vínculo com o registro anterior.
    registry["human_feedback"] = {"sha256": file_hash(feedback_path), "reviewer_id": reviewer,
                                  "reviewed_images": len(seen), "source_registry_sha256": file_hash(registry_path)}
    registry["parent_registry_version"] = registry.get("version")
    registry.pop("display_name", None)  # nome digitado no envio vale só para a versão de origem
    # Nome da versão: a pasta de saída, ou o nome do arquivo quando a saída fica na mesma pasta.
    registry["version"] = output.parent.name if output.parent.resolve() != registry_path.parent.resolve() else output.stem
    write_json(output, registry)
    return registry


# ---------------------------------------------------------------------------
# Propostas visuais de IA (add-ai-proposals) e migração de taxonomia.
# ---------------------------------------------------------------------------

# Nome de classe sugerida: minúsculas, números e "_" (ex.: telha_trincada).
# Regex: começa com letra minúscula, seguida de 2 a 60 letras minúsculas, dígitos ou "_"
# (total de 3 a 61 caracteres). Usada com fullmatch (o texto inteiro precisa casar).
PROPOSED_CLASS_SLUG = re.compile(r"[a-z][a-z0-9_]{2,60}")


# ENTRADA: comando add-ai-proposals (propostas de um modelo/IA externo entram como candidatas).
def apply_ai_proposals(config: ProjectConfig, registry_path: Path, proposals_path: Path, output: Path) -> dict:
    """Propostas visuais de IA preenchem a revisão como candidatas, nunca como aprovação.

    Classes ainda inexistentes ficam em proposed_new_class, ao lado da classe
    ativa mais próxima; ativá-las exige nova taxonomia e migração explícita.

    Parâmetros:
        config (ProjectConfig): configuração (raiz e taxonomia).
        registry_path (Path): registro de origem.
        proposals_path (Path): JSON de propostas::

            {"schema_version": 1, "annotator": "nome da IA",
             "images": [{"image_sha256": "...", "status": "positive|negative|ambiguous",
                         "boxes": [{"class_id": 3, "bbox_xyxy": [...],
                                    "proposed_new_class": "telha_trincada"}], "notes": "..."}]}

        output (Path): novo ``registry.json`` (pasta nova); não pode existir.

    Retorna:
        dict: o novo registro, com ``ai_proposals`` (hash, anotador, número de fotos,
        registro de origem e lista ordenada de ``proposed_new_classes``).

    Erros (ValueError): saída existe; schema inválido; taxonomia mudou; anotador vazio
    ou longo; foto duplicada/desconhecida; foto já aprovada por humano; original
    alterado; status inválido (``excluded`` não é aceito aqui); caixas inválidas;
    status incompatível; ``proposed_new_class`` fora do padrão.

    Grava: ``output`` e ``review_summary.json`` na pasta de ``output``.
    """
    # 1. Saída nova, arquivos no schema conhecido e taxonomia atual.
    if output.exists():
        raise ValueError("Saída já existe; use uma nova versão da revisão.")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    proposals = json.loads(proposals_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or registry.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError("Schema inválido no registro de revisão.")
    if not isinstance(proposals, dict) or proposals.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError("Schema inválido no arquivo de propostas.")
    if registry.get("taxonomy_sha256") != file_hash(config.taxonomy_path):
        raise ValueError("Taxonomia mudou: migre a revisão antes de propor caixas.")
    # 1a. Anotador (IA) obrigatório, até 100 caracteres.
    annotator_value = proposals.get("annotator", "")
    annotator = annotator_value.strip() if isinstance(annotator_value, str) else ""
    if not annotator or len(annotator) > 100:
        raise ValueError("Informe o anotador (IA) das propostas.")
    class_count = len(detection_names(load_taxonomy(config.taxonomy_path)))
    by_id = {item["image_sha256"]: item for item in registry["images"]}
    if len(by_id) != len(registry["images"]):
        raise ValueError("Imagem duplicada no registro de revisão.")
    if not isinstance(proposals.get("images"), list) or not proposals["images"]:
        raise ValueError("Propostas devem conter uma lista não vazia de imagens.")
    seen = set()
    # 2. Cada foto proposta.
    for proposal in proposals["images"]:
        digest = proposal.get("image_sha256") if isinstance(proposal, dict) else None
        if digest not in by_id or digest in seen:
            raise ValueError("Imagem desconhecida/duplicada nas propostas.")
        seen.add(digest)
        item = by_id[digest]
        # ATENCAO: proposta de IA nunca substitui uma decisão humana já registrada.
        if item.get("human_approved") is True:
            raise ValueError(f"{item.get('filename')} já tem decisão humana; proposta de IA não a substitui.")
        source = resolve_local_path(item["source_path"], config.root)
        if file_hash(source) != digest:
            raise ValueError(f"A foto original mudou: {item.get('filename')}.")
        status, boxes = proposal.get("status"), proposal.get("boxes", [])
        # EXTENSAO: status que uma IA pode propor (excluded fica de fora de propósito).
        if status not in {"positive", "negative", "ambiguous"}:
            raise ValueError("Status de proposta inválido.")
        validate_boxes(boxes, item["width"], item["height"], class_count)
        if (status == "negative" and boxes) or (status == "positive" and not boxes):
            raise ValueError("Status positivo/negativo incompatível com as caixas propostas.")
        # 2a. Classe nova sugerida: só um slug válido (o class_id continua sendo a classe ativa mais próxima).
        for box in boxes:
            slug = box.get("proposed_new_class")
            if slug is not None and (not isinstance(slug, str) or not PROPOSED_CLASS_SLUG.fullmatch(slug)):
                raise ValueError("proposed_new_class deve ser um slug minúsculo (ex.: telha_trincada).")
        # 2b. Substitui status/caixas e marca como proposta de IA pendente de revisão humana.
        #     Notas vazias na proposta mantêm as notas anteriores da foto.
        item.update(status=status, boxes=boxes, annotator=annotator, notes=proposal.get("notes") or item.get("notes", ""),
                    visual_review_status="first_pass_ai", technical_status="pending_human_review", human_approved=False)
    # 3. Bloco de origem das propostas e lista de classes novas sugeridas (de todas as fotos).
    registry["ai_proposals"] = {"sha256": file_hash(proposals_path), "annotator": annotator, "images": len(seen),
                                "source_registry_sha256": file_hash(registry_path),
                                "proposed_new_classes": sorted({box["proposed_new_class"] for item in registry["images"]
                                                                for box in item["boxes"] if box.get("proposed_new_class")})}
    registry["parent_registry_version"] = registry.get("version")
    registry.pop("display_name", None)  # nome digitado no envio vale só para a versão de origem
    # ATENCAO: a versão é sempre o nome da pasta de saída; se output ficar na mesma pasta do
    # registro de origem, o review_summary.json dessa pasta é sobrescrito.
    registry["version"] = output.parent.name
    write_json(output, registry)
    write_review_summary(output.parent, registry)
    return registry


def _remap_box(box: dict, remap: list[dict], old_names: list[str]) -> dict:
    """Caixa na classe nova, com a classe antiga e os atributos do mapa.

    Parâmetros: box (caixa já validada na taxonomia antiga); remap (``active_class_remap``);
    old_names (slugs ativos antigos). Retorna: dict novo (a caixa original não muda).
    Atributos já gravados na caixa vencem os derivados do mapa.
    """
    entry = remap[box["class_id"]]
    # setdefault: numa segunda fusão, a classe mais antiga continua registrada.
    new_box = dict(box, class_id=entry["to_id"], attributes={**entry["attributes"], **(box.get("attributes") or {})})
    new_box.setdefault("previous_class_id", box["class_id"])
    new_box.setdefault("previous_class_slug", old_names[box["class_id"]])
    return new_box


def _merge_overlaps(boxes: list[dict], limit: float) -> int:
    """Pares de caixas que a fusão juntou na mesma classe e se sobrepõem (IoU >= limit).

    Antes eram classes diferentes; depois da fusão podem ser o mesmo objeto marcado duas
    vezes. Não apaga nada: o número fica na foto para o revisor decidir.
    """
    from .pilot import box_iou  # import local: pilot importa este módulo
    pairs = 0
    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            a, b = boxes[first], boxes[second]
            if (a["class_id"] == b["class_id"] and a.get("previous_class_id") != b.get("previous_class_id")
                    and box_iou(a["bbox_xyxy"], b["bbox_xyxy"]) >= limit):
                pairs += 1
    return pairs


def _remapped_reviewed_ids(old_ids: list[int], remap: list[dict], new_names: list[str], unchanged: set[str]) -> list[int]:
    """Classes novas que continuam revistas depois da fusão.

    Uma classe nova só conta como revista se o escopo dela não cresceu (está em
    ``unchanged``) e TODAS as classes antigas que a formam tinham sido revistas.
    """
    reviewed = []
    for new_id, name in enumerate(new_names):
        sources = [old_id for old_id, entry in enumerate(remap) if entry["to_id"] == new_id]
        if name in unchanged and sources and all(old_id in old_ids for old_id in sources):
            reviewed.append(new_id)
    return reviewed


# ENTRADA: comando migrate-review. Use depois de mudar configs/taxonomy.json: cria uma cópia
# da revisão na taxonomia nova; a revisão original fica intacta.
def migrate_review_taxonomy(config: ProjectConfig, registry_path: Path, previous_taxonomy: Path, output: Path) -> dict:
    """Acrescente ou funda classes sem inventar caixas ou transferir aprovações antigas.

    Dois casos: (a) expansão, com IDs ativos anteriores preservados e classes novas ao
    fim; (b) fusão, quando a taxonomia nova declara ``migration.active_class_map``
    (v6 → v7): cada caixa muda para o ID da classe que contém a antiga, guarda
    ``previous_class_id``/``previous_class_slug`` e recebe os ``attributes`` do mapa.
    Divisão de classe não é suportada. O registro anterior e a taxonomia de origem
    continuam intactos.

    Parâmetros:
        config (ProjectConfig): configuração; ``config.taxonomy_path`` é a taxonomia NOVA.
        registry_path (Path): registro feito na taxonomia anterior.
        previous_taxonomy (Path): cópia da taxonomia anterior (ex.:
            ``configs/taxonomy_ceasa_v5_reparo_rufo.json``); o hash dela precisa bater
            com ``taxonomy_sha256`` do registro.
        output (Path): novo ``registry.json`` (pasta nova); não pode existir.

    Retorna:
        dict: o registro migrado. Toda foto fica ``human_approved=False``,
        ``taxonomy_recheck_pending=True`` e com o histórico da aprovação anterior.

    Erros (ValueError): saída existe; taxonomia anterior não confere; a nova não é um
    acréscimo puro nem declara mapa de fusão a partir da anterior; foto duplicada/original alterado; dimensões divergentes; caixas
    inválidas; escopo de classes revisadas inválido; histórico de migrações malformado.

    Grava: ``output`` e ``review_summary.json`` na pasta de ``output``.

    ATENCAO: depois da migração nenhuma foto está aprovada; é preciso revisar de novo
    na página (as classes novas ou ampliadas) e importar com import-review antes de treinar.
    """
    # 1. Saída nova e taxonomias: a nova precisa começar exatamente com as classes antigas.
    if output.exists():
        raise ValueError("Saída já existe; use uma nova versão da revisão.")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    old_taxonomy = load_taxonomy(previous_taxonomy)
    new_taxonomy = load_taxonomy(config.taxonomy_path)
    old_names, new_names = detection_names(old_taxonomy), detection_names(new_taxonomy)
    if registry.get("schema_version") != REVIEW_SCHEMA or registry.get("taxonomy_sha256") != file_hash(previous_taxonomy):
        raise ValueError("Taxonomia anterior não corresponde ao registro de origem.")
    source_registry_sha256 = file_hash(registry_path)
    previous_taxonomy_sha256 = file_hash(previous_taxonomy)
    # 1a. Expansão (IDs antigos iguais, classes novas ao fim) ou fusão declarada pela taxonomia nova.
    remap = None
    if len(new_names) <= len(old_names) or new_names[:len(old_names)] != old_names:
        declared = new_taxonomy.get("migration") or {}
        # ATENCAO: o mapa só vale para a taxonomia de onde ele foi escrito (mesmo SHA-256) ou
        # para uma anterior a ela por acréscimo (classes ativas antigas = início da lista de
        # origem, ex.: revisão v5 direto para a v7).
        source_names = declared.get("from_active_classes") or []
        if (declared.get("from_taxonomy_sha256") != previous_taxonomy_sha256
                and source_names[:len(old_names)] != old_names):
            raise ValueError("Esta migração suporta acréscimo de classes ou fusão declarada a partir da taxonomia anterior.")
        remap = active_class_remap(new_taxonomy, old_names)
        if remap is None:
            raise ValueError("Taxonomia nova não declara o mapa de classes (migration.active_class_map).")
        # Classes novas cujo escopo é exatamente o das antigas: a revisão anterior continua valendo.
        unchanged = set(declared.get("unchanged_scope_classes", []))
    seen = set()
    # 2. Cada foto: confere o original e converte aprovação atual em histórico.
    for item in registry["images"]:
        digest = item["image_sha256"]
        source = resolve_local_path(item["source_path"], config.root)
        if digest in seen or file_hash(source) != digest:
            raise ValueError("Imagem duplicada ou original alterado durante a migração.")
        seen.add(digest)
        if review_image_size(source) != (item["width"], item["height"]):
            raise ValueError("Dimensões da foto divergem do registro anterior.")
        # As caixas existentes precisam ser válidas na taxonomia ANTIGA (len(old_names)).
        validate_boxes(item["boxes"], item["width"], item["height"], len(old_names))
        reviewer_value = item.get("human_reviewer")
        has_reviewer = isinstance(reviewer_value, str) and bool(reviewer_value.strip())
        # 2a. A foto tem aprovação humana válida nesta versão?
        # O indicador isolado não comprova uma revisão humana integral. O estado
        # técnico, o autor e a ausência de pendência precisam ser compatíveis.
        current_human_approval = (
            item.get("human_approved") is True
            and item.get("technical_status") == "human_visual_reviewed"
            and has_reviewer
            and item.get("status") in {"positive", "negative"}
            and not item.get("taxonomy_recheck_pending")
        )
        # 2b. ...ou já trazia uma aprovação humana de uma migração anterior?
        historical_human_approval = (
            item.get("previous_human_approved") is True
            and item.get("visual_review_status") == "prior_taxonomy_human_review"
            and has_reviewer
        )
        # 2c. Quais IDs de classe já foram revistos para esta foto.
        # Uma migração repetida não equivale a revisar classes acrescentadas.
        # A aprovação humana nova cobre a taxonomia antiga inteira; a pendência
        # conserva somente o escopo efetivamente registrado na rodada anterior.
        if item.get("taxonomy_recheck_pending"):
            reviewed_ids = item.get("classes_reviewed_before_migration", [])
            # Lista de inteiros únicos, todos IDs da taxonomia antiga.
            if (not isinstance(reviewed_ids, list)
                    or any(type(value) is not int or value not in range(len(old_names)) for value in reviewed_ids)
                    or len(set(reviewed_ids)) != len(reviewed_ids)):
                raise ValueError("Escopo de classes revisadas antes da migração é inválido.")
        else:
            reviewed_ids = list(range(len(old_names)))
        # 2c'. Fusão: troca o ID das caixas e traduz o escopo revisado para os IDs novos.
        old_reviewed_ids = reviewed_ids
        if remap is not None:
            item["boxes"] = [_remap_box(box, remap, old_names) for box in item["boxes"]]
            overlaps = _merge_overlaps(item["boxes"], config.pilot.get("duplicate_iou", 0.7))
            if overlaps:
                item["taxonomy_merge_overlaps"] = overlaps
            reviewed_ids = _remapped_reviewed_ids(old_reviewed_ids, remap, new_names, unchanged)
        # 2d. Guarda quem aprovou, quando e em que taxonomia (rastreabilidade).
        if current_human_approval:
            item["historical_human_review"] = {
                "taxonomy_sha256": previous_taxonomy_sha256,
                "classes_reviewed": old_reviewed_ids,  # IDs da taxonomia anterior
                "source_registry_sha256": source_registry_sha256,
                "reviewer": item["human_reviewer"],
                "review_at": item.get("human_review_at"),
            }
        # 2e. Campos de histórico: setdefault preserva o status mais antigo já registrado.
        item.setdefault("historical_visual_review_status",
                        item.get("previous_visual_review_status", item.get("visual_review_status")))
        item["previous_visual_review_status"] = item.get("visual_review_status")
        item["previous_human_approved"] = current_human_approval or historical_human_approval
        item["visual_review_status"] = (
            "prior_taxonomy_human_review" if item["previous_human_approved"]
            else "prior_taxonomy_ai_review"
        )
        # 2f. Volta a ficar pendente: ninguém olhou a foto procurando as classes novas.
        item["classes_reviewed_before_migration"] = reviewed_ids
        item["taxonomy_recheck_pending"] = True
        item["human_approved"] = False
        item["technical_status"] = "pending_human_review"
        # Negativo na taxonomia anterior não comprova ausência das novas classes.
        if item["status"] == "negative":
            item["previous_status"] = "negative"
            item["status"] = "ambiguous"
        # ATENCAO: gravidade nunca é inferida.
        item["severity"] = None
    # 3. Descrição desta migração (origem, taxonomias e IDs acrescentados).
    migration = {
        "source_registry_sha256": source_registry_sha256, "source_registry_path": str(registry_path.resolve()),
        "previous_taxonomy_sha256": previous_taxonomy_sha256, "previous_taxonomy_path": str(previous_taxonomy.resolve()),
        "previous_active_classes": old_names, "new_active_classes": new_names,
        "new_class_ids": list(range(len(old_names), len(new_names))),
        "note": "Caixas anteriores preservadas. Revisão integral das novas classes e aprovação humana pendentes.",
    }
    if remap is not None:
        widened = sorted(index for index, name in enumerate(new_names) if name not in unchanged)
        migration.update(
            kind="remap", new_class_ids=widened,
            class_map=[{"from_id": index, "from_slug": name, "to_id": entry["to_id"], "to_slug": entry["to_slug"],
                        "attributes": entry["attributes"]} for index, (name, entry) in enumerate(zip(old_names, remap))],
            merge_overlap_images=sum(bool(item.get("taxonomy_merge_overlaps")) for item in registry["images"]),
            note=("Classes fundidas pelo mapa da taxonomia nova; caixas preservadas com o ID novo, a classe "
                  "anterior e atributos derivados. Conferência humana das classes ampliadas pendente."))
    # 4. Histórico de migrações: a anterior entra na lista (sem duplicar) e depois esta.
    # O último vínculo continua disponível para leitores antigos. O histórico
    # mantém os hashes das etapas anteriores, inclusive das revisões humanas.
    migration_history = registry.get("taxonomy_migration_history", [])
    if not isinstance(migration_history, list) or any(not isinstance(value, dict) for value in migration_history):
        raise ValueError("Histórico de migrações deve ser uma lista de registros.")
    previous_migration = registry.get("taxonomy_migration")
    if previous_migration is not None:
        if not isinstance(previous_migration, dict):
            raise ValueError("Registro da migração anterior é inválido.")
        if not migration_history or migration_history[-1] != previous_migration:
            migration_history.append(previous_migration)
    migration_history.append(migration)
    # 5. Cabeçalho do registro na taxonomia nova e gravação da nova versão.
    registry["taxonomy_migration_history"] = migration_history
    registry["taxonomy_migration"] = migration
    registry["taxonomy_sha256"] = file_hash(config.taxonomy_path)
    registry["taxonomy_version"] = new_taxonomy["version"]
    registry["parent_registry_version"] = registry.get("version")
    registry.pop("display_name", None)  # nome digitado no envio vale só para a versão de origem
    registry["version"] = output.parent.name
    registry["scope"] = ("propostas_preservadas_fusao_de_classes_pendente_de_revisao" if remap is not None
                         else "propostas_preservadas_expansao_de_taxonomia_pendente_de_revisao")
    write_json(output, registry)
    write_review_summary(output.parent, registry)
    return registry
