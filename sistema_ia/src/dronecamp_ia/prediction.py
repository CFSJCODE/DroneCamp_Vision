"""Inferência em fotos e vídeos, com evidências e achados para revisão.

Função no projeto: roda um modelo sobre uma pasta de mídia e grava, em
``runs/predict_*``, as imagens com caixas desenhadas e um ``findings.jsonl``
com cada detecção. É o comando ``predict``.

O que faz:
- ``find_media``: lista fotos e vídeos locais de uma pasta.
- ``serialize_detections``: transforma o resultado da Ultralytics em JSON
  (classe, confiança, caixa), sempre com gravidade pendente.
- ``predict``: executa, grava evidências, ``execution.json`` e ``summary.json``.

Nunca transforma ausência de detecção em conformidade: sem caixa não quer dizer
telhado sem problema.

Quando mexer: confiança/tamanho da inferência ficam em ``prediction`` no
``configs/project.yaml``; mexa aqui para mudar o formato dos achados.

Conteúdo do arquivo (funções, na ordem em que aparecem):
- ``MEDIA_EXTENSIONS``: extensões de foto e vídeo aceitas.
- ``find_media(source)``: arquivo único ou pasta (recursiva) → lista de mídias.
- ``serialize_detections(result, taxonomy, demo)``: um resultado da Ultralytics
  (uma foto ou um quadro de vídeo) → lista de dicionários de detecção.
- ``serialize_boxes(rows, names, taxonomy, demo)``: o mesmo formato a partir de
  tuplas (classe, confiança, xyxy); é aqui que o dicionário de cada detecção é
  montado (as duas rotas de inferência passam por ela).
- ``_draw_boxes(image, boxes, names)``: desenha as caixas da inferência em
  janelas numa foto (a rota normal usa ``result.save`` da Ultralytics).
- ``predict(config, source, weights, demo)``: o comando inteiro.

Quem chama:
- ``cli.py``, comando ``python -m dronecamp_ia predict --source <pasta>
  [--weights <.pt>] [--demo]`` → ``predict``.
- Nenhum outro módulo importa estas funções hoje (a página de revisão usa
  ``suggestions.detect_boxes``, não este arquivo).

O que lê do disco:
- as mídias de ``source`` (só leitura; nada é alterado);
- ``configs/taxonomy.json`` (``load_taxonomy``) e o checkpoint ``.pt``;
- ``runs/pilot_train_*/summary.json`` ao lado dos pesos, para saber se o
  modelo é piloto (``training.is_pilot_checkpoint``).

O que grava (sempre numa pasta NOVA ``runs/predict_<UTC>_<id>/``,
``predict_pilot_*`` ou ``predict_demo_*``; nunca sobrescreve):
- ``execution.json``: modo, aviso, versões, parâmetros e SHA-256 de cada mídia;
- ``evidence/<hash12>_<NNNNNN>.jpg``: foto/quadro com as caixas desenhadas;
- ``findings.jsonl``: uma linha JSON por foto/quadro (mesmo sem detecção);
- ``summary.json``: contagem de arquivos e quadros processados.

Onde mexer:
- Aceitar outro formato de foto/vídeo: ``MEDIA_EXTENSIONS`` (marcador EXTENSAO).
- Acrescentar um campo em CADA detecção (ex.: área da caixa em px²):
  ``serialize_boxes``, no dicionário ``detections.append({...})``. Vale para as
  duas rotas (foto inteira e janelas) de uma vez.
- Acrescentar um campo em CADA linha do ``findings.jsonl`` (ex.: GPS da foto):
  ``predict``, nos DOIS dicionários ``row = {...}`` (rota em janelas e rota
  normal); os dois precisam continuar com as mesmas chaves.
- Mudar o texto de aviso ou o modo gravado: ``predict``, etapa 2.
- Mudar a cor/espessura das caixas da rota em janelas: ``_draw_boxes``. A rota
  normal desenha com ``result.save`` (estilo da Ultralytics).
- Mudar confiança, ``imgsz`` ou outros argumentos do ``model.predict``:
  ``configs/project.yaml``, bloco ``prediction`` (não precisa mexer aqui).
- Mudar quando uma foto é fatiada em janelas: ``configs/project.yaml``,
  bloco ``pilot.tiling`` (lido por ``suggestions.tiling_plan``).

ATENCAO: ``severity`` sai sempre ``None`` e ``review_status`` sempre
"pendente": a gravidade NUNCA é inferida pelo modelo; é decidida na revisão
humana. A gravidade histórica do laudo de referência vai só como referência
(``historical_severity_only``).
"""

# ---------------------------------------------------------------------------
# Importações
# ---------------------------------------------------------------------------

# Biblioteca padrão: caminhos, hash do nome do arquivo (prefixo das evidências),
# JSON das linhas do findings.jsonl e cronômetro da rota em janelas.
from pathlib import Path
from hashlib import sha256
import json
import time

# Pillow: abrir a foto e desenhar as caixas na rota em janelas (_draw_boxes).
from PIL import Image, ImageDraw

# Módulos do próprio pacote:
# - backend: carrega o checkpoint, confere os nomes das classes e coleta versões;
# - config: tipo ProjectConfig, leitura da taxonomia e lista de slugs ativos;
# - dataset: extensões que contam como FOTO (vídeo nunca é fatiado);
# - io: SHA-256 de arquivo, pasta nova em runs/ e gravação de JSON.
from .backend import check_domain_names, load_detector, runtime_info
from .config import ProjectConfig, detection_names, load_taxonomy
from .dataset import IMAGE_EXTENSIONS
from .io import file_hash, make_run_directory, write_json

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Fotos e vídeos aceitos (vídeos são lidos quadro a quadro).
# Comparação sempre em minúsculas (``path.suffix.lower()``), então ".JPG" também entra.
# EXTENSAO: novo formato de mídia: acrescente a extensão aqui, com ponto e em
# minúsculas. Se for FOTO e precisar do fatiamento em janelas dos pesos piloto,
# ela também tem de estar em ``dataset.IMAGE_EXTENSIONS``. Confira antes se a
# Ultralytics/OpenCV consegue decodificar o formato.
MEDIA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv"}


# ---------------------------------------------------------------------------
# Entrada: localizar as mídias
# ---------------------------------------------------------------------------

def find_media(source: Path) -> list[Path]:
    """Aceite somente mídia local; uma pasta é percorrida recursivamente.

    Por que existe: o ``predict`` precisa de uma lista fechada e ordenada de
    arquivos antes de começar, para gravar o SHA-256 de cada um em
    ``execution.json`` e para falhar cedo se não houver nada a processar.
    URLs, streams de câmera e globs NÃO são aceitos de propósito: só arquivos
    locais, que podem ser conferidos por hash depois.

    Parâmetros:
        source (Path): um arquivo de foto/vídeo ou uma pasta. Aceita ``~``
            (ex.: ``Path("~/voos/ceasa")``) e caminho relativo ao diretório
            atual; é convertido em caminho absoluto.

    Retorna:
        list[Path]: caminhos absolutos. Arquivo único → lista com ele;
        pasta → todos os arquivos com extensão de ``MEDIA_EXTENSIONS`` em
        qualquer subpasta, em ordem alfabética do caminho completo (ordem
        estável entre execuções).

    Erros:
        ValueError("Nenhuma imagem/vídeo local suportado encontrado em ..."):
        arquivo com extensão não suportada, caminho inexistente ou pasta sem
        nenhuma mídia aceita.

    Chamada por: ``predict`` (etapa 1).
    """
    # Normaliza: expande "~" e transforma em caminho absoluto (os caminhos
    # gravados em execution.json e findings.jsonl ficam absolutos).
    source = source.expanduser().resolve()
    # Caso 1: o usuário apontou um único arquivo de mídia suportado.
    if source.is_file() and source.suffix.lower() in MEDIA_EXTENSIONS:
        return [source]
    # Caso 2: pasta. rglob("*") desce em todas as subpastas; o filtro mantém só
    # arquivos (não pastas) com extensão aceita; sorted dá ordem determinística.
    if source.is_dir():
        files = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS)
        # Pasta com mídia: devolve. Pasta vazia cai no erro abaixo.
        if files:
            return files
    # Caso 3: nada aproveitável (extensão errada, caminho inexistente, pasta vazia).
    # EXTENSAO: para aceitar outra origem (ex.: lista de arquivos num .txt),
    # acrescente um novo "if" antes desta linha devolvendo list[Path].
    raise ValueError(f"Nenhuma imagem/vídeo local suportado encontrado em {source}.")


# ---------------------------------------------------------------------------
# Serialização: resultado do modelo → dicionários do findings.jsonl
# ---------------------------------------------------------------------------

def serialize_detections(result, taxonomy: dict, demo: bool) -> list[dict]:
    """Separe a classe visual da severidade: esta será confirmada na revisão.

    Converte UM resultado da Ultralytics (uma foto ou um quadro de vídeo) na
    lista de detecções gravada em ``findings.jsonl``. Esta função só extrai
    os números dos tensores; o dicionário de cada detecção é montado por
    ``serialize_boxes`` (assim a rota em janelas e a rota normal geram
    exatamente o mesmo formato).

    Parâmetros:
        result (ultralytics.engine.results.Results): um item devolvido por
            ``model.predict(...)``. Usa ``result.boxes`` (tensores ``xyxy``,
            ``conf``, ``cls``) e ``result.names`` (dict id → slug da classe).
        taxonomy (dict): conteúdo de ``configs/taxonomy.json`` (``load_taxonomy``).
        demo (bool): True com ``predict --demo`` (pesos genéricos, ex. COCO):
            as classes não são patologias, então não se busca a categoria na
            taxonomia.

    Retorna:
        list[dict]: uma detecção por caixa (formato em ``serialize_boxes``);
        lista vazia quando o modelo não devolveu caixas.

    Chamada por: ``predict`` (rota normal, foto inteira ou quadro de vídeo).
    """
    # Sem caixas (modelo de classificação ou nada detectado acima do limiar):
    # nenhuma detecção. ATENCAO: lista vazia NÃO significa telhado sem defeito.
    if result.boxes is None:
        return []
    boxes = result.boxes
    # Os tensores podem estar na GPU: .cpu().tolist() traz para listas Python.
    # - xyxy: [x1, y1, x2, y2] em pixels da foto ORIGINAL (canto superior
    #   esquerdo e inferior direito; origem no topo à esquerda);
    # - conf: confiança 0..1; cls: id da classe como float (ex.: 3.0).
    # O zip percorre as três listas em paralelo (uma posição = uma caixa) e a
    # tupla é reordenada para (classe, confiança, coordenadas), o formato que
    # serialize_boxes espera.
    rows = [(int(class_id), float(confidence), coordinates) for coordinates, confidence, class_id in
            zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist())]
    # Monta os dicionários (mesmo código da rota em janelas).
    return serialize_boxes(rows, result.names, taxonomy, demo)


def serialize_boxes(rows: list[tuple[int, float, list[float]]], names, taxonomy: dict, demo: bool) -> list[dict]:
    """Mesmo formato de ``serialize_detections`` a partir de (classe, confiança, xyxy).

    É o ÚNICO lugar que define o formato de uma detecção no ``findings.jsonl``.
    As duas rotas de inferência chegam aqui:
    ``serialize_detections`` (foto inteira/vídeo, Ultralytics) e a rota em
    janelas de ``predict`` (``suggestions.detect_boxes`` para pesos piloto).

    Parâmetros:
        rows (list[tuple[int, float, list[float]]]): uma tupla por caixa:
            ``(class_id, confiança 0..1, [x1, y1, x2, y2] em pixels da foto
            original)``. Ex.: ``(0, 0.8731, [120.4, 88.0, 310.9, 240.2])``.
        names (dict[int, str] | list[str]): nomes das classes do modelo,
            indexados pelo id (``model.names``/``result.names``). Num modelo do
            projeto são os slugs da taxonomia (ex.: "telha_danificada"); com
            ``--demo`` são nomes genéricos (ex.: "person").
        taxonomy (dict): conteúdo de ``configs/taxonomy.json``; usa ``source``
            (documento de referência) e ``classes`` (``slug``, ``label``,
            ``pages``, ``reference_severity``).
        demo (bool): True ignora a taxonomia (classes genéricas não são
            patologias): ``category_label`` e ``report_reference`` saem None.

    Retorna:
        list[dict]: na mesma ordem de ``rows``, cada item com as chaves:
        - ``class_id`` (int) e ``class_name`` (str, slug/nome do modelo);
        - ``confidence`` (float, 6 casas);
        - ``bbox_xyxy_pixels`` (list[float], 4 valores com 2 casas);
        - ``category_label`` (str rótulo legível da taxonomia, ou None);
        - ``severity`` (sempre None) e ``review_status`` (sempre "pendente");
        - ``report_reference`` (dict com ``document``, ``pages`` e
          ``historical_severity_only``, ou None se não houver categoria).

    Erros:
        KeyError/IndexError se ``class_id`` não existir em ``names`` (modelo
        e dados incoerentes); KeyError se a classe da taxonomia não tiver
        ``pages``/``reference_severity``.

    Chamada por: ``serialize_detections`` e ``predict`` (rota em janelas).
    """
    # Índice slug → definição da classe na taxonomia, para achar rótulo,
    # páginas do laudo de referência e gravidade histórica de cada detecção.
    # Inclui todas as classes (fase 1 e 2); o modelo só emite as ativas.
    categories = {item["slug"]: item for item in taxonomy["classes"]}
    detections = []
    # Uma iteração por caixa detectada.
    for class_id, confidence, coordinates in rows:
        # Nome da classe segundo o PRÓPRIO modelo (o id sozinho não basta,
        # pois um checkpoint genérico tem outra lista de classes).
        # ATENCAO: o id da classe é contrato entre taxonomia, pesos, labels e
        # revisões; ``check_domain_names`` (em predict) garante que os nomes do
        # modelo coincidem com a taxonomia antes de chegar aqui.
        name = names[int(class_id)]
        # Categoria da taxonomia (None em modo demo ou se o slug não existir).
        category = categories.get(name) if not demo else None
        # EXTENSAO: novo campo por detecção (ex.: "area_px": (x2-x1)*(y2-y1)):
        # acrescente a chave neste dicionário. Quem lê o findings.jsonl
        # (scripts/relatórios externos) passa a ver a chave nova; não há
        # leitor dentro do pacote hoje. Mantenha valores serializáveis em JSON
        # e finitos (json.dumps usa allow_nan=False em predict).
        detections.append({
            # Identidade da classe: id numérico + nome do modelo.
            "class_id": int(class_id), "class_name": name,
            # Confiança arredondada a 6 casas (evita ruído de float no JSON).
            "confidence": round(float(confidence), 6),
            # Caixa x1, y1, x2, y2 em pixels da foto original, 2 casas decimais.
            "bbox_xyxy_pixels": [round(float(value), 2) for value in coordinates],
            # Rótulo legível (ex.: "Telha quebrada") vindo da taxonomia.
            "category_label": category["label"] if category else None,
            # ATENCAO: gravidade nunca é inferida; sempre pendente de revisão humana.
            "severity": None, "review_status": "pendente",
            # Referência ao laudo usado para montar a taxonomia: documento,
            # páginas onde a categoria aparece e a gravidade que ELE atribuiu.
            # "historical_severity_only" deixa claro que é só histórico, não
            # a gravidade desta detecção.
            "report_reference": {
                "document": taxonomy["source"], "pages": category["pages"],
                "historical_severity_only": category["reference_severity"],
            } if category else None,
        })
    return detections


# ---------------------------------------------------------------------------
# Desenho das evidências da rota em janelas
# ---------------------------------------------------------------------------

def _draw_boxes(image, boxes: list[dict], names) -> "Image.Image":
    """Evidência com as caixas (classe e confiança), para fotos inferidas em janelas.

    Na rota normal a Ultralytics desenha com ``result.save``; na rota em
    janelas (pesos piloto + foto grande) não há ``result`` único, então o
    desenho é feito aqui, com Pillow, direto na imagem recebida.

    Parâmetros:
        image (PIL.Image.Image): foto em RGB, no tamanho ORIGINAL (as
            coordenadas das caixas estão em pixels da foto original). É
            alterada no lugar.
        boxes (list[dict]): caixas de ``suggestions.detect_boxes``, cada uma
            com ``class_id`` (int), ``confidence`` (0..1) e ``bbox_xyxy``
            ([x1, y1, x2, y2] inteiros, em pixels).
        names (dict[int, str] | list[str]): nomes das classes do modelo.

    Retorna:
        PIL.Image.Image: a mesma imagem, já com retângulos e rótulos
        "slug 0.87" (pronta para ``.save``).

    Chamada por: ``predict`` (rota em janelas).
    """
    draw = ImageDraw.Draw(image)
    # Espessura proporcional ao lado maior: 2 px até ~3000 px, 3 px em ~4500 px,
    # etc. (1 px a cada 1500 px de lado, mínimo 2), para continuar visível em
    # fotos de drone de 20 MP depois de reduzidas no visualizador.
    stroke = max(2, round(max(image.size) / 1500))
    # Um retângulo e um rótulo por caixa.
    for box in boxes:
        x1, y1, x2, y2 = box["bbox_xyxy"]
        # EXTENSAO: cor das caixas (RGB laranja-avermelhado fixo). Para cor por
        # classe, use a paleta de review_render.COLORS indexada por class_id.
        draw.rectangle([x1, y1, x2, y2], outline=(255, 64, 32), width=stroke)
        # Texto "slug confiança" com 2 casas (ex.: "telha_danificada 0.87").
        label = f"{names[int(box['class_id'])]} {box['confidence']:.2f}"
        # Rótulo logo acima da caixa (12 px por unidade de espessura); max(0, ...)
        # impede que saia da imagem quando a caixa encosta no topo.
        # Fonte padrão do Pillow (pequena e sem acentos; os slugs não têm acento).
        draw.text((x1 + stroke, max(0, y1 - 12 * stroke)), label, fill=(255, 64, 32))
    return image


# ---------------------------------------------------------------------------
# Comando predict
# ---------------------------------------------------------------------------

# ENTRADA: comando "python -m dronecamp_ia predict --source <pasta> [--weights X.pt] [--demo]" (cli.py).
def predict(config: ProjectConfig, source: Path, weights: str | None = None, demo: bool = False) -> Path:
    """Gere imagens anotadas e JSONL incremental, inclusive quando não há caixas.

    Fluxo (etapas numeradas no corpo):
    1. lista as mídias, lê a taxonomia, carrega o modelo e confere as classes;
       pesos piloto passam a usar o ``imgsz`` do treino;
    2. cria ``runs/predict*_<UTC>_<id>/`` e grava ``execution.json``;
    3. para cada foto/quadro: grava a evidência em ``evidence/`` e uma linha em
       ``findings.jsonl`` (rota em janelas para fotos grandes com pesos
       piloto; rota normal da Ultralytics nos demais casos);
    4. grava ``summary.json``.

    Parâmetros:
        config (ProjectConfig): configuração carregada de
            ``configs/project.yaml`` (usa ``root``, ``taxonomy_path``,
            ``device``, ``prediction`` = argumentos do ``model.predict`` como
            ``conf``, ``imgsz``, ``nms`` e ``pilot.tiling``).
        source (Path): arquivo ou pasta de fotos/vídeos (ver ``find_media``).
        weights (str | None): caminho de um checkpoint ``.pt``; None usa o
            ``model`` de ``configs/project.yaml``.
        demo (bool): True aceita pesos genéricos (ex.: COCO) só para testar o
            fluxo; não confere classes e marca tudo como demonstração.

    Retorna:
        Path: a pasta da execução criada em ``runs/``.

    Erros:
        ValueError: sem mídia (``find_media``); checkpoint que não é ``.pt``
        ou não existe (``load_detector``); classes do modelo diferentes das
        ativas na taxonomia sem ``--demo`` (``check_domain_names``); nenhum
        quadro decodificado (etapa 4). Neste último caso a pasta da execução
        fica criada, mas sem ``summary.json`` (execução incompleta).

    Chamada por: ``cli.py`` (comando ``predict``).
    """
    # 1. Mídia, taxonomia e modelo (--demo aceita pesos COCO genéricos, só para teste).
    media = find_media(source)
    taxonomy = load_taxonomy(config.taxonomy_path)
    model = load_detector(config, weights)
    # ATENCAO: sem --demo, os nomes do modelo precisam ser exatamente os slugs
    # das classes ativas, na mesma ordem (ids). Evita mostrar classes COCO como
    # patologias ou trocar uma classe por outra.
    if not demo:
        check_domain_names(model, detection_names(taxonomy))
    # Import tardio: training importa Ultralytics/torch; só carrega aqui.
    from .training import is_pilot_checkpoint

    # Pesos piloto = checkpoint de runs/pilot_train_*/ cujo summary.json tem
    # "pilot": true. Em modo demo nunca é piloto.
    pilot = not demo and is_pilot_checkpoint(model.ckpt_path)
    if pilot:
        from .training import pilot_inference_config

        # Pesos piloto inferem no imgsz em que foram treinados (ex.: 640), não
        # no imgsz de produção do project.yaml (ex.: 1024). Devolve uma CÓPIA
        # do config com prediction["imgsz"] trocado.
        config = pilot_inference_config(config, model)
    # 2. Pasta da execução e execution.json com modo, versões e hashes das mídias.
    # Nome da pasta: predict_demo_*, predict_pilot_* ou predict_* (make_run_directory
    # acrescenta data UTC e um id aleatório; nunca reaproveita uma pasta).
    run = make_run_directory(config.root, "predict_demo" if demo else "predict_pilot" if pilot else "predict")
    evidence = run / "evidence"
    evidence.mkdir()
    # "mode" curto que vai em CADA linha do findings.jsonl.
    # ATENCAO: são dois textos diferentes de propósito: este (por linha) e o
    # "mode" do execution.json logo abaixo. Quem filtra resultados por modo
    # precisa saber qual dos dois está lendo.
    # EXTENSAO: novo modo de execução (ex.: modelo validado em campo):
    # acrescente o caso aqui, no "mode" e no "warning" do execution.json e no
    # nome da pasta acima, mantendo os três em sincronia.
    mode = "demo_pesos_genericos" if demo else "candidatos_modelo_piloto_nao_validado" if pilot else "candidatos_revisao"
    # execution.json: tudo o que é preciso para reproduzir/auditar a execução.
    write_json(run / "execution.json", {
        # Modo da execução inteira (ver ATENCAO acima sobre os dois "mode").
        "mode": "demo_pesos_genericos" if demo else "deteccao_piloto_nao_validada" if pilot else "deteccao_especializada_para_revisao",
        # Versão e hash da taxonomia: liga os ids de classe deste resultado a
        # uma taxonomia exata.
        "taxonomy_version": taxonomy["version"], "taxonomy_sha256": file_hash(config.taxonomy_path),
        # Versões de Python/Ultralytics/torch, dispositivo e hash dos pesos;
        # parâmetros efetivos da inferência (já com o imgsz do piloto, se for o caso).
        "runtime": runtime_info(model, config), "parameters": config.prediction, "pilot": pilot,
        # Aviso legível, conforme o modo. Nenhum modo emite laudo.
        "warning": "Demonstração técnica sem reconhecimento treinado de patologias." if demo
        else "Modelo piloto treinado com uma única edificação; use só como sugestão para revisão humana." if pilot
        else "Achados candidatos exigem revisão; nenhum laudo é emitido automaticamente.",
        # SHA-256 de cada mídia de entrada (lê todos os arquivos uma vez: pode
        # demorar com vídeos grandes).
        # EXTENSAO: novo metadado da execução (ex.: operador, voo): acrescente
        # uma chave neste dicionário.
        "files": [{"path": str(path), "sha256": file_hash(path)} for path in media],
    })
    # 2b. Pesos piloto: fotos de drone grandes são inferidas janela a janela
    #     (pilot.tiling, suggestions.detect_boxes), como na revisão e no gate.
    # Imports tardios: detect_boxes faz a inferência fatiada e junta as caixas;
    # review_image_size lê largura/altura sem rotação EXIF; tiling_plan decide
    # se a foto é fatiada; calculate_patch_windows só é usado para contar janelas.
    from .suggestions import detect_boxes, review_image_size, tiling_plan
    from .tiling import calculate_patch_windows

    # 3. Cada foto/quadro: imagem com caixas em evidence/ e uma linha em findings.jsonl.
    # processed conta fotos + quadros de vídeo efetivamente gravados.
    processed = 0
    # O findings.jsonl é aberto uma vez e escrito linha a linha (JSONL = um
    # objeto JSON por linha). flush() após cada linha: se a execução cair no
    # meio de um vídeo, o que já foi processado continua no arquivo.
    with (run / "findings.jsonl").open("w", encoding="utf-8") as stream:
        for path in media:
            # Prefixo das evidências: 12 primeiros caracteres hex do SHA-256 do
            # CAMINHO (não do conteúdo), para nomes curtos, sem acentos e sem
            # colisão entre arquivos homônimos em subpastas diferentes.
            key = sha256(str(path).encode("utf-8")).hexdigest()[:12]
            # plan = None → rota normal; dict → rota em janelas.
            plan = None
            # Só pesos piloto e só FOTOS (vídeo nunca é fatiado) podem ir para
            # a rota em janelas; tiling_plan devolve None se a foto for pequena
            # (abaixo de pilot.tiling.min_side) ou o fatiamento estiver desligado.
            if pilot and path.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    width, height = review_image_size(path)
                    plan = tiling_plan(config, width, height)
                except ValueError:  # orientação EXIF: a Ultralytics trata a foto inteira
                    plan = None
            # --- Rota em janelas (pesos piloto + foto grande) ---------------
            if plan:
                started = time.perf_counter()
                # Caixas já unidas entre janelas, em pixels inteiros da foto
                # original, com confiança ≥ prediction.conf.
                boxes = detect_boxes(model, path, config, float(config.prediction["conf"]))
                # Foto = um único "quadro": sufixo _000001.
                evidence_path = evidence / f"{key}_{1:06d}.jpg"
                # Desenha numa cópia RGB (a foto original não é alterada) e
                # grava JPEG qualidade 90.
                with Image.open(path) as image:
                    _draw_boxes(image.convert("RGB"), boxes, model.names).save(evidence_path, quality=90)
                # Converte as caixas no formato padrão (class_id, conf, xyxy) e
                # serializa pela mesma função da rota normal.
                detections = serialize_boxes([(box["class_id"], box["confidence"], box["bbox_xyxy"]) for box in boxes],
                                             model.names, taxonomy, demo)
                # Linha do findings.jsonl desta foto.
                # EXTENSAO: novo campo por linha: acrescente aqui E no "row" da
                # rota normal mais abaixo (as duas rotas devem ter as mesmas chaves;
                # "inference" é a única chave exclusiva desta rota).
                row = {
                    # image_size_hw = [altura, largura] (mesma ordem de result.orig_shape).
                    "source": str(path), "frame_index_processed": 1, "image_size_hw": [height, width],
                    "evidence": str(evidence_path), "mode": mode,
                    # ATENCAO: sem caixas = "sem_deteccoes_acima_limiar", nunca
                    # "conforme": ausência de detecção não prova ausência de defeito.
                    "status": "candidatos_detectados" if detections else "sem_deteccoes_acima_limiar",
                    "detections": detections, "review_status": "pendente",
                    # Como a foto foi inferida: tamanho da janela (px) e quantas
                    # janelas a cobriram (recalculado só para registro).
                    "inference": {"regime": "janelas", "patch": plan["patch"],
                                  "windows": len(calculate_patch_windows(width, height, plan["patch"], plan["overlap"]))},
                    # Tempo total da foto em milissegundos (inferência + desenho).
                    "speed_ms": {"total": round((time.perf_counter() - started) * 1000, 1)},
                }
                # ensure_ascii=False mantém acentos; allow_nan=False recusa NaN/inf.
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                processed += 1
                # Esta foto acabou; pula a rota normal.
                continue
            # --- Rota normal (Ultralytics, foto inteira ou vídeo) ------------
            # stream=True mantém o consumo de memória limitado durante vídeos.
            # rect=False: a foto é reduzida/preenchida até um quadrado de imgsz
            # (em vez do retângulo mínimo);
            # vid_stride=1: todos os quadros do vídeo; save=False: a evidência é
            # gravada abaixo, com nome controlado. **config.prediction repassa
            # conf, imgsz, nms etc. do project.yaml.
            # EXTENSAO: para processar 1 a cada N quadros de vídeo, troque
            # vid_stride (e ajuste o texto da nota em summary.json se quiser).
            results = model.predict(source=str(path), stream=True, device=config.device,
                                    save=False, verbose=False, rect=False, vid_stride=1,
                                    **config.prediction)
            # Uma iteração por foto (1 resultado) ou por quadro de vídeo;
            # ordinal começa em 1 e vira o sufixo da evidência (_000001, _000002...).
            for ordinal, result in enumerate(results, start=1):
                evidence_path = evidence / f"{key}_{ordinal:06d}.jpg"
                # Imagem com caixas no estilo da Ultralytics.
                result.save(filename=str(evidence_path))
                detections = serialize_detections(result, taxonomy, demo)
                # Linha do findings.jsonl deste quadro (mesmas chaves da rota em
                # janelas, sem "inference").
                row = {
                    "source": str(path), "frame_index_processed": ordinal,
                    # orig_shape = (altura, largura) da foto/quadro original.
                    "image_size_hw": list(result.orig_shape), "evidence": str(evidence_path),
                    "mode": mode,
                    "status": "candidatos_detectados" if detections else "sem_deteccoes_acima_limiar",
                    "detections": detections, "review_status": "pendente",
                    # Tempos da Ultralytics em ms (preprocess, inference, postprocess).
                    # O "key" desta compreensão é local a ela e não altera o
                    # "key" (prefixo da evidência) do laço externo.
                    "speed_ms": {key: float(value) for key, value in result.speed.items()},
                }
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                processed += 1
    # 4. summary.json: só "complete" se ao menos uma imagem foi processada.
    # Vídeo corrompido ou codec ausente podem render zero quadros sem erro da
    # Ultralytics; aqui isso vira erro explícito (e a pasta fica sem summary.json).
    if processed == 0:
        raise ValueError(f"Nenhum frame/imagem foi decodificado; execução incompleta em {run}.")
    # ATENCAO: "unique_physical_defects_count" fica None de propósito: a mesma
    # telha pode aparecer em vários quadros/fotos sobrepostas; somar caixas não
    # dá número de defeitos.
    # EXTENSAO: novo total da execução (ex.: quadros com detecção): acrescente
    # um contador no laço acima e a chave aqui.
    write_json(run / "summary.json", {
        "source_files": len(media), "processed_images_or_frames": processed,
        "complete": True, "unique_physical_defects_count": None,
        "note": "Caixas em frames diferentes podem mostrar a mesma ocorrência; não somar como defeitos únicos.",
    })
    return run
