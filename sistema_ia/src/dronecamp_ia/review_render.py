"""Página local de revisão (index.html), evidências e rascunho de relatório.

Função no projeto: gera a página que o revisor abre no navegador para conferir,
corrigir e aprovar caixas. É o comando ``render-review``; ``suggest`` também
chama este arquivo para mostrar as sugestões do modelo.

O que faz (``render_review_package``):
- copia as fotos para ``images/`` e desenha as caixas em ``evidence/``;
- grava labels propostas em ``labels_proposed/<hash do registro>/``;
- embute os dados (fotos, caixas, sugestões, classes) e o painel de operações
  (treinos, curvas, comparações; ``operations.py``) no modelo
  ``review_templates/index.html`` e grava ``index.html``;
- ``refresh_review_page`` regrava só o ``index.html`` (comando ``refresh-page``).
- grava ``review_results.csv``, ``review_summary.json`` e ``review_report.md``.

Quando mexer: aparência das evidências (cores, fonte) ou dados enviados à
página. Botões e comportamento da página ficam em ``review_templates/index.html``.

Funções (em ordem no arquivo):
- ``_font(size)``: fonte usada para escrever o nome da classe nas evidências.
- ``_load_suggestions(directory, registry_path, class_count)``: lê e valida ``suggestions.json``.
- ``build_page(payload, operations)``: monta o HTML final (modelo + CSS + JS + dados).
- ``empty_page_payload(config)``: dados de uma página sem inspeção (servidor local).
- ``load_page_payload(registry_path)``: lê ``browser_data.json`` conferindo o hash.
- ``refresh_review_page(config, registry_path)``: regrava só o ``index.html``.
- ``render_review_package(config, registry_path)``: gera a pasta de revisão inteira.

Quem chama:
- ``cli.py``: ``render-review`` (render_review_package) e ``refresh-page`` (refresh_review_page).
- ``suggestions.suggest_for_registry`` e ``ingest.ingest_media``: render_review_package.
- ``platform_server.py``: build_page, empty_page_payload e load_page_payload a cada
  pedido de página.
- ``exports.py`` e ``class_catalog.py`` importam ``COLORS``.

O que lê e grava (na pasta do ``registry.json``, ``data/reviews/<versão>/``):
- Lê: ``registry.json`` (formato descrito no cabeçalho de ``review_data.py``),
  ``suggestions.json`` (opcional), as fotos originais (``source_path``), a taxonomia e
  os modelos ``review_templates/index.html``, ``platform.css`` e ``platform.js``.
- Grava: ``images/<hash>.<ext>``, ``evidence/<hash>.jpg``,
  ``labels_proposed/<hash do registro>/<hash da foto>.txt``, ``browser_data.json``,
  ``index.html``, ``review_results.csv``, ``review_summary.json`` e ``review_report.md``.
  Esses arquivos são regenerados a cada execução; o ``registry.json`` não é alterado.

Onde mexer:
- Cor de uma classe ou nova classe ativa: ``COLORS`` (um item por ID).
- Fonte/espessura/posição do texto nas evidências: etapa 2 de ``render_review_package``.
- Novo dado por foto para a página: o dict ``browser_images.append`` (etapa 2) e o uso
  em ``review_templates/platform.js``.
- Novo dado geral para a página: o dict ``payload`` (etapa 4) E ``empty_page_payload``
  (mesmas chaves) e o uso em platform.js.
- Nova coluna no CSV: cabeçalho (etapa 3) e o valor na mesma posição em ``rows.append``.
- Texto do relatório Markdown: lista ``report`` (etapa 5).
"""

from collections import Counter
from pathlib import Path
import csv
import json
import shutil

from PIL import Image, ImageDraw, ImageFont

from .config import ProjectConfig, detection_names, load_taxonomy
from .io import file_hash, resolve_local_path, write_json
from .naming import review_name
from .operations import collect_operations, run_id_from_path
from .review_data import boxes_to_yolo, review_image_size, validate_boxes, write_review_summary

# ---------------------------------------------------------------------------
# Paleta de cores das classes
# ---------------------------------------------------------------------------

# Uma cor por classe (índice = ID da classe), usada nas evidências e na página (paleta controlada e harmoniosa).
# ATENCAO: a paleta tem 14 cores (IDs 0–13). render_review_package indexa COLORS[class_id] sem
# módulo; ao ativar a classe 14 (fase 1) acrescente cores aqui antes, senão o render-review
# quebra com IndexError. Cor editada pela seção Classes fica em taxonomy_overrides.json.
# EXTENSAO: nova classe ativa: acrescente uma cor hexadecimal "#RRGGBB" ao FIM da lista
# (a posição é o ID da classe), junto com a classe em configs/taxonomy.json.
COLORS = ["#FF5C70", "#E63950", "#FF9F43", "#42B8E8", "#9B6DF2", "#EC5DAA", "#52C79A", "#2BBE72", "#F4C542", "#56D1E8", "#FF7A1A", "#F5D547", "#B89AF8", "#4F7DFF"]


# ---------------------------------------------------------------------------
# Auxiliares: fonte das evidências e leitura das sugestões
# ---------------------------------------------------------------------------

def _font(size: int):
    """Fonte do Windows quando existir; senão a fonte padrão do Pillow.

    Parâmetros:
        size (int): tamanho da fonte em pixels (só vale para as fontes TrueType).

    Retorna:
        ImageFont: Arial ou Segoe UI do Windows, ou a fonte bitmap padrão do Pillow
        (que ignora ``size``) em Linux/macOS.

    EXTENSAO: outra fonte (ex.: Linux): acrescente o caminho na tupla do laço; a
    primeira que existir é usada (exports.py tem uma lista parecida).
    """
    for path in (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/segoeui.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _load_suggestions(directory: Path, registry_path: Path, class_count: int) -> dict | None:
    """Sugestões só aparecem se pertencem a este registro; nunca viram caixas sozinhas.

    Parâmetros:
        directory (Path): pasta da revisão (onde fica ``suggestions.json``).
        registry_path (Path): ``registry.json`` atual (o hash dele é comparado).
        class_count (int): número de classes ativas (IDs válidos 0..class_count-1).

    Retorna:
        dict | None: o conteúdo de ``suggestions.json`` (gravado por
        ``suggestions.suggest_for_registry``: ``images`` = {hash da foto: [caixas com
        class_id, bbox_xyxy, confidence, id...]}, mais ``model_label``, ``conf`` etc.)
        ou None quando o arquivo não existe.

    Erros (ValueError): arquivo de outra versão do registro (hash diferente) ou de
    schema diferente; sugestão com classe fora do intervalo; confiança fora de 0..1.

    ATENCAO: depois de import-review ou migrate-review o hash do registro muda e as
    sugestões antigas passam a ser recusadas aqui; rode suggest de novo.
    """
    path = directory / "suggestions.json"
    # Sem arquivo de sugestões: a página simplesmente não mostra caixas tracejadas.
    if not path.is_file():
        return None
    suggestions = json.loads(path.read_text(encoding="utf-8"))
    if suggestions.get("schema_version") != 1 or suggestions.get("registry_sha256") != file_hash(registry_path):
        raise ValueError("suggestions.json pertence a outra versão do registro; gere as sugestões novamente.")
    # Confere classe e confiança de cada sugestão de cada foto (a geometria não é revalidada aqui).
    for values in suggestions["images"].values():
        for value in values:
            if type(value.get("class_id")) is not int or not 0 <= value["class_id"] < class_count:
                raise ValueError("Sugestão com classe inválida.")
            # Confiança ausente vira -1 e é recusada.
            if not 0 <= float(value.get("confidence", -1)) <= 1:
                raise ValueError("Sugestão com confiança inválida.")
    return suggestions


def _vlm_opinions(directory: Path, registry_path: Path) -> dict:
    """Segunda opinião do VLM (``ai_notes.json``) sobre as sugestões, por foto e id.

    Lê só notas deste registro (mesmo hash), de origem ``"sugestao"`` e com resposta
    interpretada. Retorna ``{hash da foto: {id da sugestão: {"class_guess",
    "legibility", "model_used"}}}`` (vazio sem arquivo). Só informa a página: não
    aceita nem descarta nada.
    """
    path = directory / "ai_notes.json"
    if not path.is_file():
        return {}
    notes = json.loads(path.read_text(encoding="utf-8"))
    if notes.get("registry_sha256") != file_hash(registry_path):
        return {}
    opinions: dict = {}
    for digest, entries in (notes.get("images") or {}).items():
        for entry in entries:
            if entry.get("source") == "sugestao" and entry.get("parsed") and entry.get("box_id"):
                opinions.setdefault(digest, {})[entry["box_id"]] = {
                    key: entry.get(key) for key in ("class_guess", "legibility", "model_used")}
    return opinions


# ---------------------------------------------------------------------------
# Montagem do HTML e dados da página
# ---------------------------------------------------------------------------

# ATENCAO: os marcadores de substituição do template (CSS, JS e dados) são trocados por texto
# puro. Não escreva esses nomes em comentários do index.html/platform.js: o marcador de dados
# precisa aparecer uma única vez no platform.js, senão a página não é gerada.
def build_page(payload: dict, operations: dict | None) -> str:
    """Preenche o modelo da plataforma com os dados da revisão e o painel de operações.

    ``operations`` (treinos, curvas, comparações) é um instantâneo: aberta do
    disco, a página mostra o estado do momento em que foi gerada; pelo servidor
    local (``platform``) ela consulta o estado atual a cada poucos segundos.

    Parâmetros:
        payload (dict): dados da página (formato de ``browser_data.json``; veja a etapa 4
            de ``render_review_package`` ou ``empty_page_payload``).
        operations (dict | None): painel de operações de ``operations.collect_operations``.

    Retorna:
        str: HTML completo, autocontido (CSS, JS e dados embutidos).

    Erros (ValueError): CSS/JS contendo a tag de fechamento que os embute; modelo sem o
    marcador do JS; platform.js sem exatamente um marcador de dados.

    Chamada por: refresh_review_page, render_review_package e platform_server (cada GET
    de página).
    """
    # 1. Modelo HTML e dados (payload + operações) serializados em JSON.
    templates = Path(__file__).parent / "review_templates"
    template = (templates / "index.html").read_text(encoding="utf-8")
    data = {**payload, "operations": operations}
    # JSON embutido evita fetch de arquivos locais e fecha a possibilidade de </script> nos textos.
    # Trocar "<" por \u003c mantém o JSON válido e impede que um texto feche a tag <script>.
    serialized = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    # 2. CSS e JS lidos dos arquivos próprios.
    # Tema e lógica ficam em arquivos próprios (platform.css/platform.js) e são embutidos
    # aqui, então a página continua sendo um único HTML que abre do disco. O JS não pode
    # conter "</script>" (fecharia a tag); o CSS, "</style>".
    css = (templates / "platform.css").read_text(encoding="utf-8")
    script = (templates / "platform.js").read_text(encoding="utf-8")
    if "</style" in css.lower() or "</script" in script.lower():
        raise ValueError("platform.css/platform.js não podem conter a tag de fechamento que os embute.")
    # 3. Substituições: primeiro o JS (partition divide no primeiro marcador do JS),
    #    depois o CSS e, por último, os dados (o marcador de dados está dentro do JS).
    head, marker, tail = template.partition("__PLATFORM_JS__")
    if not marker or script.count("__REVIEW_DATA__") != 1:
        raise ValueError("Template sem __PLATFORM_JS__ ou platform.js sem um único __REVIEW_DATA__.")
    page = head + script + tail
    # Os dados entram por último: o marcador fica dentro do JS (``const DATA = __REVIEW_DATA__``).
    return page.replace("__PLATFORM_CSS__", css).replace("__REVIEW_DATA__", serialized)


def empty_page_payload(config: ProjectConfig) -> dict:
    """Dados de uma página sem inspeção aberta (espaço de trabalho vazio no servidor local).

    Mesmas chaves de ``browser_data.json``, com a lista de fotos vazia: a página
    mostra as classes, o painel de operações e a importação de mídia.

    Parâmetros:
        config (ProjectConfig): configuração (taxonomia e ``pilot.duplicate_iou``).

    Retorna:
        dict: ``registry_sha256``/``version`` vazios, ``images`` = [], ``classes`` =
        [{id, label, color}], ``duplicate_iou`` e ``suggestions`` = None.

    Chamada por: platform_server quando não há inspeção selecionada.

    EXTENSAO: ao acrescentar uma chave no payload de render_review_package (etapa 4),
    acrescente-a aqui também com um valor vazio, para a página não quebrar sem inspeção.
    """
    taxonomy = load_taxonomy(config.taxonomy_path)
    detection_names(taxonomy)  # valida os IDs consecutivos como no render
    # Rótulos legíveis das classes ativas (fase 1), por ID.
    labels = {item["id"]: item["label"] for item in taxonomy["classes"] if item["phase"] == 1}
    # Aqui a cor usa módulo (key % len): nunca quebra, mesmo com mais classes que cores.
    return {"registry_sha256": "", "version": "", "images": [], "taxonomy_recheck_pending": False, "classes_requiring_review": [],
            "classes": [{"id": key, "label": value, "color": COLORS[key % len(COLORS)]} for key, value in labels.items()],
            "duplicate_iou": config.pilot.get("duplicate_iou"), "suggestions": None}


def load_page_payload(registry_path: Path) -> dict:
    """Dados já gerados por ``render-review``, conferidos contra o registro atual.

    Parâmetros:
        registry_path (Path): ``registry.json`` da inspeção.

    Retorna:
        dict: conteúdo de ``browser_data.json`` da mesma pasta.

    Erros:
        ValueError se ``browser_data.json`` foi gerado para outra versão do registro;
        OSError/FileNotFoundError se o arquivo não existe (render-review nunca rodou).

    Chamada por: refresh_review_page e platform_server.
    """
    payload = json.loads((registry_path.parent / "browser_data.json").read_text(encoding="utf-8"))
    # ATENCAO: o hash garante que a página mostra exatamente o registro atual.
    if payload.get("registry_sha256") != file_hash(registry_path):
        raise ValueError("browser_data.json pertence a outra versão do registro; rode render-review novamente.")
    return payload


# ENTRADA: comando refresh-page (cli.py). Regrava index.html sem tocar em fotos/evidências.
def refresh_review_page(config: ProjectConfig, registry_path: Path) -> Path:
    """Regrava o ``index.html`` e atualiza as cores das classes com a paleta atual,
    sem copiar fotos nem redesenhar evidências.

    Útil quando muda o modelo da página ou chegam novos treinos: fotos, caixas,
    sugestões e rótulos propostos continuam exatamente os de ``browser_data.json``.

    Parâmetros:
        config (ProjectConfig): configuração (para o painel de operações).
        registry_path (Path): ``registry.json`` da inspeção.

    Retorna:
        Path: caminho do ``index.html`` regravado.

    Grava: ``browser_data.json`` (só as cores mudam) e ``index.html``.
    """
    directory = registry_path.parent
    payload = load_page_payload(registry_path)
    # Atualiza a cor de cada classe que tenha cor na paleta (IDs fora dela ficam como estão).
    for category in payload.get("classes", []):
        if 0 <= category.get("id", -1) < len(COLORS):
            category["color"] = COLORS[category["id"]]
    write_json(directory / "browser_data.json", payload)
    # Painel de operações recalculado agora (treinos novos aparecem).
    page = build_page(payload, collect_operations(config, directory))
    (directory / "index.html").write_text(page, encoding="utf-8")
    return directory / "index.html"


# ---------------------------------------------------------------------------
# Pacote completo da revisão (render-review)
# ---------------------------------------------------------------------------

# ENTRADA: comando render-review. Gera a pasta inteira da revisão; refresh-page (acima) só
# regrava o index.html. ATENCAO: data/reviews/*/index.html é gerado no PC do projeto; depois de
# mudar o template rode refresh-page lá, em cada inspeção, em vez de editar o HTML gerado.
def render_review_package(config: ProjectConfig, registry_path: Path) -> Path:
    """Copie evidências para visualização; fotos e registros anteriores permanecem.

    Parâmetros:
        config (ProjectConfig): configuração (raiz, taxonomia, ``pilot.duplicate_iou``).
        registry_path (Path): ``registry.json`` da inspeção (formato no cabeçalho de
            ``review_data.py``).

    Retorna:
        Path: caminho do ``index.html`` gerado.

    Erros (ValueError): taxonomia mudou (rode migrate-review); foto duplicada; original
    alterado (hash); dimensões divergentes; caixas inválidas; ``suggestions.json`` de
    outra versão. IndexError se uma classe não tiver cor em ``COLORS``.

    Grava: veja "O que lê e grava" no cabeçalho do módulo.

    Chamada por: cli ``render-review``, suggestions.suggest_for_registry, ingest.ingest_media.
    """
    # 1. Registro na taxonomia atual e pastas de saída ao lado dele.
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("taxonomy_sha256") != file_hash(config.taxonomy_path):
        raise ValueError("Taxonomia mudou: migre a revisão antes de gerar as evidências.")
    directory = registry_path.parent
    images_dir = directory / "images"
    overlays_dir = directory / "evidence"
    # Cada registro tem sua pasta: rótulos de uma versão anterior não entram na atual.
    labels_dir = directory / "labels_proposed" / file_hash(registry_path)
    for target in (images_dir, overlays_dir, labels_dir):
        target.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy(config.taxonomy_path)
    names = detection_names(taxonomy)
    digests = [item["image_sha256"] for item in registry["images"]]
    if len(set(digests)) != len(digests):
        raise ValueError("Imagem duplicada no registro de revisão.")
    # Rótulo legível por ID de classe ativa (ex.: 0 -> "Telha quebrada").
    labels = {item["id"]: item["label"] for item in taxonomy["classes"] if item["phase"] == 1}
    slugs = {item["id"]: item["slug"] for item in taxonomy["classes"] if item["phase"] == 1}
    suggestions = _load_suggestions(directory, registry_path, len(names))
    vlm = _vlm_opinions(directory, registry_path)
    # 2. Cada foto: confere o original, copia, desenha as caixas e prepara os dados da página.
    browser_images, rows = [], []
    for item in registry["images"]:
        # 2a. Original reancorado neste PC, com hash e tamanho iguais aos do registro.
        digest = item["image_sha256"]
        source = resolve_local_path(item["source_path"], config.root)
        if file_hash(source) != digest:
            raise ValueError(f"A foto original mudou: {item['filename']}.")
        if review_image_size(source) != (item["width"], item["height"]):
            raise ValueError("Dimensões do registro divergem da foto revisada.")
        validate_boxes(item["boxes"], item["width"], item["height"], len(names))
        # 2b. Cópia da foto em images/<hash>.<ext> (a página carrega daqui, caminho relativo).
        copied = images_dir / f"{digest}{source.suffix.lower()}"
        shutil.copy2(source, copied)
        # 2c. Evidência: foto em RGB com retângulo colorido e etiqueta "número: classe".
        with Image.open(source) as original:
            evidence = original.convert("RGB")
        draw = ImageDraw.Draw(evidence)
        # Fonte proporcional à largura da foto (1/75), no mínimo 14 px.
        font = _font(max(14, int(item["width"] / 75)))
        for number, box in enumerate(item["boxes"], start=1):
            coordinates = box["bbox_xyxy"]
            # ATENCAO: sem módulo; classe sem cor em COLORS gera IndexError (veja COLORS).
            color = COLORS[box["class_id"]]
            # EXTENSAO: espessura do contorno (width=3) e deslocamento da etiqueta (22 px acima
            # da caixa, limitado ao topo da foto) se ajustam nestas linhas.
            draw.rectangle(coordinates, outline=color, width=3)
            text = f"{number}: {labels[box['class_id']]}"
            x, y = coordinates[:2]
            bounds = draw.textbbox((x, max(0, y - 22)), text, font=font)
            draw.rectangle(bounds, fill=color)
            draw.text((x, max(0, y - 22)), text, fill="#08111F", font=font)
        evidence.save(overlays_dir / f"{digest}.jpg", quality=92)
        # 2d. Label YOLO proposta, só para fotos com decisão positiva/negativa.
        # Estes .txt são propostas, deliberadamente separados de data/dataset.
        if item["status"] in {"positive", "negative"}:
            (labels_dir / f"{digest}.txt").write_text(boxes_to_yolo(item["boxes"], item["width"], item["height"]), encoding="utf-8")
        # 2e. Dados da foto para a página: tudo do registro, menos o caminho do PC
        #     (source_path=None), mais URLs relativas e as sugestões desta foto.
        # EXTENSAO: novo dado por foto para a página: acrescente a chave neste dict e
        # leia-a em review_templates/platform.js.
        browser_images.append({**item, "source_path": None, "image_url": copied.relative_to(directory).as_posix(),
                               "evidence_url": f"evidence/{digest}.jpg",
                               "model_suggestions": [{**value, **({"vlm": vlm[digest][value.get("id")]} if value.get("id") in vlm.get(digest, {}) else {})}
                                                     for value in (suggestions["images"].get(digest, []) if suggestions else [])]})
        # 2f. Linha do CSV (mesma ordem do cabeçalho da etapa 3). Páginas separadas por ";".
        # ATENCAO: item["notes"] é obrigatório aqui (KeyError se a foto não tiver "notes").
        rows.append([digest, item["filename"], ";".join(map(str,item["pages"])), item["status"], len(item["boxes"]),
                     item["visual_review_status"], item["technical_status"], item["notes"], item.get("second_review_notes", "")])
    # 3. Planilha com a triagem de cada foto.
    # utf-8-sig (com BOM) para o Excel reconhecer os acentos.
    with (directory / "review_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        # EXTENSAO: nova coluna no CSV: acrescente o nome aqui e o valor na mesma posição
        # em rows.append (etapa 2f).
        writer.writerow(["image_sha256", "filename", "pages", "triage_status", "proposed_boxes", "visual_review_status", "technical_status", "notes", "second_review_notes"])
        writer.writerows(rows)
    # Classes ativas que alguma foto ainda não reviu depois de uma migração (para o aviso da página).
    # O aviso inclui expansões ainda não revistas de todas as migrações anteriores.
    pending_class_ids = set()
    for item in registry["images"]:
        if item.get("taxonomy_recheck_pending"):
            pending_class_ids.update(set(labels) - set(item.get("classes_reviewed_before_migration", [])))
    # 4. Dados embutidos na página (a página funciona aberta direto do disco, sem servidor).
    # EXTENSAO: novo dado geral para a página: acrescente a chave aqui, em empty_page_payload
    # (valor vazio) e leia-a em review_templates/platform.js (objeto DATA).
    payload = {"registry_sha256": file_hash(registry_path), "version": registry["version"],
               # Nome legível da inspeção (naming.review_name); a pasta continua em "version".
               "display_name": review_name(registry["version"], registry.get("display_name")), "images": browser_images,
               "taxonomy_recheck_pending": any(item.get("taxonomy_recheck_pending") for item in registry["images"]),
               "classes_requiring_review": [labels[key] for key in sorted(pending_class_ids)],
               # slug: nome técnico da classe, usado pela página para ler a opinião do VLM (class_guess).
               "classes": [{"id": key, "label": value, "color": COLORS[key], "slug": slugs[key]} for key,value in labels.items()],
               "duplicate_iou": config.pilot.get("duplicate_iou"),
               # weights_run_id: execução dos pesos das sugestões (só o id, nunca o caminho do PC),
               # para a página ligar o modelo em uso ao treino sem depender de uma comparação salva.
               # EXTENSAO: novo metadado das sugestões para a página: acrescente a chave na tupla.
               "suggestions": {**{key: suggestions.get(key) for key in ("model_label", "conf", "warning", "weights_sha256", "pilot", "calibrator", "uncovered_classes")},
                               "weights_run_id": run_id_from_path(suggestions.get("weights"))} if suggestions else None}
    write_json(directory / "browser_data.json", payload)
    (directory / "index.html").write_text(build_page(payload, collect_operations(config, directory)), encoding="utf-8")
    # 5. Resumo e rascunho de relatório em Markdown com links para fotos e evidências.
    summary = write_review_summary(directory, registry)
    # EXTENSAO: nova linha no cabeçalho do relatório: acrescente um item nesta lista.
    # ATENCAO: o texto deixa claro que não é laudo técnico; mantenha esse aviso.
    report = ["# Revisão visual do CEASA — rascunho para conferência", "",
              f"Versão: {registry['version']}. {summary['unique_images']} imagens únicas, {summary['total_proposed_boxes']} caixas propostas.",
              f"Registro de origem: `{registry_path.name}`; SHA-256: `{file_hash(registry_path)}`.",
              f"Triagem: {summary['image_status_counts']}. Aprovações humanas registradas: {summary['human_approved_images']}.",
              f"Correções humanas de taxonomia anterior preservadas: {summary['historical_human_review_images']} fotos; não equivalem a aprovação das classes atuais.",
              "", "Este registro contém propostas visuais. Não é laudo técnico e não representa aprovação humana ou qualidade de um modelo especializado.",
              "", "| Foto original | Páginas | Triagem | Ocorrências propostas | Evidência demarcada |", "| --- | --- | --- | --- | --- |"]
    # Uma linha da tabela por foto: contagem de caixas por classe e links relativos.
    for item in registry["images"]:
        counter = Counter(labels[box["class_id"]] for box in item["boxes"])
        occurrences = "; ".join(f"{name}: {count}" for name,count in counter.items()) or "Sem caixa proposta"
        digest = item["image_sha256"]
        suffix = Path(item["source_path"]).suffix.lower()
        photo_link = f"[{item['filename']}](images/{digest}{suffix})"
        report.append(f"| {photo_link} | {', '.join(map(str,item['pages']))} | {item['status']} | {occurrences} | [Ver caixas](evidence/{digest}.jpg) |")
    # Avisos finais: contagem não é número de defeitos e nada técnico foi inferido.
    report.extend(["", "Contagens referem-se a caixas por foto, não a defeitos físicos únicos. Cenas relacionadas e recortes podem repetir ocorrências.",
                   "", "Aprovação técnica, gravidade, prioridade, causa, metros lineares, áreas e diagnóstico hidráulico não foram inferidos."])
    (directory / "review_report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    return directory / "index.html"
