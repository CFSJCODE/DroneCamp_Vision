"""Sugestões do detector na página de revisão: candidatas, nunca aprovação.

Função no projeto: usa um modelo treinado (em geral o piloto) para propor caixas
que o revisor aceita, corrige ou descarta na página. É o comando ``suggest``.

O que faz:
- ``detect_boxes``: roda o modelo numa foto e devolve caixas com classe e confiança.
- ``suggest_for_registry``: grava ``suggestions.json`` ao lado de uma revisão
  existente e regenera a página.
- ``create_review_for_new_images``: cria uma revisão nova para fotos novas
  (cópia congelada por hash em ``originals/``) já com as sugestões.

Só o feedback humano exportado e importado entra no próximo dataset; é isso que
fecha o ciclo de melhoria do modelo.

Quando mexer: confiança mínima padrão (``pilot.suggestion_conf`` em
``configs/project.yaml``) ou o que é gravado em ``suggestions.json``.

Funções (em ordem no arquivo):
- ``tiling_plan(config, width, height, tiling)``: decide se a foto é fatiada e como.
- ``_raw_detections(model, source, config, conf)``: uma inferência (foto ou recorte).
- ``_tiled_detections(model, path, config, conf, plan)``: inferência fatiada + fusão.
- ``detect_boxes(model, path, config, conf, tiling)``: caixas finais de uma foto (pública).
- ``suggest_for_registry(...)``: sugestões para uma revisão existente (pública).
- ``create_review_for_new_images(...)``: revisão nova para fotos novas (pública).

Quem chama:
- ``cli.py``, comando ``suggest`` (``--registry`` -> suggest_for_registry;
  ``--source/--output/--group`` -> create_review_for_new_images); a tarefa "suggest"
  da seção Treinamento da plataforma passa pelo mesmo comando.
- ``detect_boxes`` também é usada por ``calibration``, ``model_gate``, ``prediction`` e
  ``autolabel``; ``tiling_plan`` por ``model_gate`` e ``prediction``.

O que lê e grava:
- Lê o ``registry.json`` (formato no cabeçalho de ``review_data.py``), as fotos
  originais, os pesos (.pt ou .onnx), opcionalmente o calibrador e os prompts de busca
  aberta (``configs/zero_shot_prompts.json`` via ``autolabel``).
- Grava ``suggestions.json`` na pasta do registro (sobrescreve o anterior) e, via
  ``review_render.render_review_package``, a página e os demais arquivos da revisão.
- ``create_review_for_new_images`` grava também ``<saída>/originals/<hash>.<ext>`` e
  ``<saída>/registry.json``.

Formato de ``suggestions.json``: ``schema_version`` (1), ``registry_sha256`` (hash do
registro a que pertence), ``created_at``, ``weights``/``weights_sha256``, ``pilot``,
``model_label``, ``runtime``, ``source_checkpoint``, ``tiling``, ``conf``, ``imgsz``,
``warning``, ``calibrator``, ``zero_shot``, ``uncatalogued``, ``total_suggestions`` e
``images`` = {hash da foto: [{id: "s001", class_id, bbox_xyxy: [x1, y1, x2, y2] em
pixels inteiros, confidence (0..1, 4 casas), acceptance_probability?, source?}]}.

Onde mexer:
- Novo campo em ``suggestions.json``: dict da etapa 4 de ``suggest_for_registry``; se a
  página precisar dele, acrescente a chave na tupla do payload ``"suggestions"`` em
  ``review_render.render_review_package``.
- Novo campo por caixa sugerida: dict ``candidate`` em ``detect_boxes``.
- Novo parâmetro de fatiamento: ``tiling_plan`` (e ``pilot.tiling`` em project.yaml).
- Novo campo padrão de foto nova: dict ``images.append`` em ``create_review_for_new_images``
  (mantenha igual ao bloco equivalente em ``ingest.py``).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .backend import check_domain_names, load_inference_model
from .config import ProjectConfig, detection_names, load_taxonomy
from .dataset import IMAGE_EXTENSIONS
from .io import file_hash, resolve_local_path, write_json
from .review_data import REVIEW_SCHEMA, review_image_size, validate_boxes
from .training import is_pilot_checkpoint, pilot_inference_config

# Aviso gravado em suggestions.json e mostrado na página junto das sugestões.
# Mudar o texto só muda a mensagem (sugestões geradas antes guardam o texto antigo).
SUGGESTION_WARNING = ("Sugestões automáticas: aceite somente o que você conferir. "
                      "O modelo pode errar a classe, a caixa ou deixar de ver ocorrências.")


# ---------------------------------------------------------------------------
# Inferência numa foto.
# ---------------------------------------------------------------------------

def tiling_plan(config: ProjectConfig, width: int, height: int, tiling: dict | None = None) -> dict | None:
    """Parâmetros do fatiamento (``pilot.tiling``) para esta foto, ou None (foto inteira).

    Fotos de drone de 20 MP têm defeitos com ~40 px; reduzir a foto inteira a 640
    os deixa com ~5 px. Acima de ``min_side`` a foto é fatiada em janelas de
    ``patch`` px (sobreposição ``overlap``), cada uma reduzida ao imgsz do modelo:
    janela 1280 → 640 é a mesma escala do dataset fatiado (``tile-dataset``).

    Parâmetros:
        config (ProjectConfig): configuração (lê ``config.pilot["tiling"]``).
        width, height (int): tamanho da foto em pixels.
        tiling (dict | None): substitui ``pilot.tiling`` quando informado (ex.:
            ``{"enabled": False}`` para forçar a foto inteira).

    Retorna:
        dict | None: None quando o fatiamento está desligado/ausente ou o maior lado
        da foto é menor que ``min_side`` (padrão 2000 px). Senão ``{"patch": int (px,
        padrão 1280), "overlap": float (fração 0..1, padrão 0.2), "full_image": bool
        (também roda a foto inteira, padrão True), "merge_iou": float (IoU para fundir
        duplicatas, padrão 0.5)}``.

    Chamada por: detect_boxes, model_gate (estratos), prediction.

    EXTENSAO: novo parâmetro de fatiamento: leia-o de ``plan`` com um padrão aqui, use-o
    em _tiled_detections e documente-o em ``pilot.tiling`` de configs/project.yaml.
    """
    # O argumento "tiling" tem prioridade sobre o YAML.
    plan = tiling if tiling is not None else config.pilot.get("tiling")
    # Sem plano, desligado, ou foto pequena: inferência na foto inteira.
    if not plan or not plan.get("enabled", True) or max(width, height) < int(plan.get("min_side", 2000)):
        return None
    return {"patch": int(plan.get("patch", 1280)), "overlap": float(plan.get("overlap", 0.2)),
            "full_image": bool(plan.get("full_image", True)), "merge_iou": float(plan.get("merge_iou", 0.5))}


def _raw_detections(model, source, config: ProjectConfig, conf: float) -> list[tuple[int, float, list[float]]]:
    """(classe, confiança, xyxy) de uma foto (caminho) ou recorte (array BGR).

    Parâmetros:
        model: modelo Ultralytics (ou o adaptador ONNX de ``backend.load_inference_model``).
        source: caminho da foto (str) ou recorte numpy em BGR (altura x largura x 3).
        config (ProjectConfig): ``config.prediction`` (imgsz, iou etc.) e ``config.device``.
        conf (float): confiança mínima (0..1); substitui ``prediction.conf``.

    Retorna:
        list[tuple]: ``(class_id, confiança, [x1, y1, x2, y2])`` em pixels de ``source``
        (de um recorte, as coordenadas são relativas ao recorte). Lista vazia sem detecção.
    """
    # rect=False: entrada quadrada, igual à usada na exportação ONNX.
    parameters = {**config.prediction, "conf": conf}
    # stream=True devolve um gerador; só há um resultado (uma imagem), pego com next().
    result = next(iter(model.predict(source=source, device=config.device, save=False, verbose=False,
                                     stream=True, rect=False, **parameters)))
    if result.boxes is None:
        return []
    # zip junta, por detecção, a caixa (xyxy), a confiança e a classe (que vem como float).
    return [(int(c), float(s), b) for b, s, c in zip(result.boxes.xyxy.cpu().tolist(),
                                                     result.boxes.conf.cpu().tolist(),
                                                     result.boxes.cls.cpu().tolist())]


def _tiled_detections(model, path: Path, config: ProjectConfig, conf: float, plan: dict) -> list[tuple[int, float, list[float]]]:
    """Inferência fatiada (estilo SAHI): janelas sobrepostas + foto inteira, NMS por classe.

    Parâmetros:
        model: modelo de inferência.
        path (Path): foto original.
        config (ProjectConfig): configuração da inferência.
        conf (float): confiança mínima (0..1).
        plan (dict): resultado de ``tiling_plan`` (patch, overlap, full_image, merge_iou).

    Retorna:
        list[tuple]: ``(class_id, confiança, [x1, y1, x2, y2])`` em pixels da FOTO
        INTEIRA, já sem duplicatas das áreas de sobreposição.
    """
    # Imports tardios: numpy/PIL/tiling só são carregados quando há fatiamento.
    import numpy as np
    from PIL import Image

    from .tiling import calculate_patch_windows, merge_tiled_predictions

    # 1. Pixels da foto em BGR (o [:, :, ::-1] inverte a ordem dos canais RGB -> BGR).
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"))[:, :, ::-1]  # Ultralytics espera BGR em arrays
    height, width = pixels.shape[:2]
    candidates = []
    # 2. Cada janela (x1, y1, x2, y2) recortada e inferida separadamente. "window" acompanha
    #    a caixa para merge_tiled_predictions somar o deslocamento (coordenadas globais).
    for window in calculate_patch_windows(width, height, plan["patch"], plan["overlap"]):
        x1, y1, x2, y2 = window
        # Recorte numpy: linhas = y, colunas = x; ascontiguousarray evita cópia "fatiada" não contígua.
        crop = np.ascontiguousarray(pixels[y1:y2, x1:x2])
        for class_id, score, box in _raw_detections(model, crop, config, conf):
            candidates.append({"class_id": class_id, "score": score, "bbox_xyxy": box, "window": list(window)})
    # 3. Foto inteira como mais uma "janela" (deslocamento zero).
    if plan["full_image"]:
        # Objetos longos (rufo, calha) podem não caber numa janela: a foto inteira entra junto.
        for class_id, score, box in _raw_detections(model, str(path), config, conf):
            candidates.append({"class_id": class_id, "score": score, "bbox_xyxy": box, "window": [0, 0, width, height]})
    # 4. Translada para coordenadas globais e remove duplicatas (NMS por classe com merge_iou).
    merged = merge_tiled_predictions(candidates, plan["merge_iou"])
    return [(item["class_id"], item["score"], item["bbox_xyxy"]) for item in merged]


def detect_boxes(model, path: Path, config: ProjectConfig, conf: float, tiling: dict | None = None) -> list[dict]:
    """Inferência real; caixas são arredondadas e recortadas aos pixels da foto.

    Fotos grandes são fatiadas conforme ``pilot.tiling`` (ou ``tiling``;
    ``{"enabled": False}`` força a foto inteira).

    Parâmetros:
        model: modelo de inferência (``model.names`` define o número de classes).
        path (Path): foto original.
        config (ProjectConfig): configuração (inferência e ``pilot.tiling``).
        conf (float): confiança mínima (0..1).
        tiling (dict | None): substitui ``pilot.tiling`` (veja ``tiling_plan``).

    Retorna:
        list[dict]: caixas ``{"class_id": int, "bbox_xyxy": [x1, y1, x2, y2] (pixels
        inteiros dentro da foto), "confidence": float (4 casas), "id": "s001"...}``,
        da mais confiante para a menos confiante.

    Erros: ValueError de ``review_image_size`` (rotação EXIF). Caixas degeneradas são
    descartadas em silêncio, não geram erro.

    Chamada por: suggest_for_registry, calibration, model_gate, prediction, autolabel.
    """
    # 1. Tamanho real (sem EXIF) e escolha entre inferência fatiada ou na foto inteira.
    width, height = review_image_size(path)
    plan = tiling_plan(config, width, height, tiling)
    detections = (_tiled_detections(model, path, config, conf, plan) if plan
                  else _raw_detections(model, str(path), config, conf))
    # 2. Converte cada detecção em {classe, caixa em pixels inteiros, confiança}.
    class_count = len(model.names)
    boxes = []
    for class_id, confidence, (x1, y1, x2, y2) in detections:
        # Arredonda e limita cada coordenada ao intervalo [0, largura] ou [0, altura].
        coordinates = [max(0, min(width, round(x1))), max(0, min(height, round(y1))),
                       max(0, min(width, round(x2))), max(0, min(height, round(y2)))]
        # EXTENSAO: novo campo por caixa sugerida: acrescente neste dict (a página recebe as
        # sugestões inteiras em model_suggestions; leia o campo em platform.js).
        candidate = {"class_id": int(class_id), "bbox_xyxy": coordinates, "confidence": round(float(confidence), 4)}
        try:
            validate_boxes([candidate], width, height, class_count)
        except ValueError:
            continue  # caixa degenerada após arredondamento
        boxes.append(candidate)
    # 3. Mais confiantes primeiro; ids s001, s002... usados pela página ao descartar.
    # ATENCAO: o id depende da ordem; sugestões regeneradas podem trocar de id.
    boxes.sort(key=lambda value: -value["confidence"])
    for index, box in enumerate(boxes, start=1):
        box["id"] = f"s{index:03d}"
    return boxes


# ---------------------------------------------------------------------------
# Sugestões numa revisão existente ou numa revisão nova de fotos novas.
# ---------------------------------------------------------------------------

# ENTRADA: comando suggest --registry e a tarefa "suggest" da seção Treinamento.
# ATENCAO: suggestions.json fica preso ao hash do registro; depois de import-review ou
# migrate-review as sugestões antigas são recusadas e precisam ser geradas de novo.
def suggest_for_registry(config: ProjectConfig, registry_path: Path, weights: str, conf: float | None = None,
                         calibrator: Path | str | None = None, allow_unproven_calibrator: bool = False,
                         zero_shot: bool = False, onnx_provider: str = "directml",
                         imgsz: int | None = None, keep: set[str] | None = None) -> Path:
    """Grave suggestions.json ao lado do registro e regenere a página de revisão.

    Com ``calibrator`` (pasta de ``fit-calibrator``), cada caixa ganha
    ``acceptance_probability`` e a lista da foto é ordenada por ela. Calibrador
    sem ganho medido é recusado, salvo ``allow_unproven_calibrator`` (testes).
    Com ``zero_shot``, a busca aberta (YOLOE, ``autolabel.py``) acrescenta caixas
    procuradas por descrição em texto e registra achados fora da taxonomia.

    Parâmetros:
        config (ProjectConfig): configuração do projeto.
        registry_path (Path): ``registry.json`` da revisão.
        weights (str): caminho dos pesos (.pt ou .onnx) com as classes ativas do projeto.
        conf (float | None): confiança mínima (0 < conf <= 1); None usa
            ``pilot.suggestion_conf`` ou, na falta dele, ``prediction.conf``.
        calibrator (Path | str | None): pasta do calibrador (opcional).
        allow_unproven_calibrator (bool): aceita calibrador sem ganho medido (testes).
        zero_shot (bool): liga a busca aberta por texto (YOLOE).
        onnx_provider (str): provedor do ONNX Runtime (``"directml"`` = GPU no Windows).
        imgsz (int | None): tamanho de inferência explícito. Para pesos piloto copiados
            para fora de ``runs/`` (ex.: ``C:\\DroneCamp_runs``), sem ``summary.json``, que
            seriam lidos no imgsz de produção em vez do tamanho do treino.

        keep (set[str] | None): hashes de fotos já decididas pelo revisor (ex.: arquivo de
            revisão salvo); com as fotos ``human_approved`` do registro, elas NÃO são
            reprocessadas e mantêm as sugestões do ``suggestions.json`` anterior (mesmo
            registro), para que aceites e descartes já feitos continuem valendo.

    Pesos de uma taxonomia anterior (classes = início exato das atuais) são aceitos;
    as classes que eles não conhecem vão em ``uncovered_classes`` do JSON e na página.
    Os ids das sugestões levam os 8 primeiros caracteres do hash dos pesos (ex.:
    ``809d823e-s001``): descartes guardados no navegador para sugestões de outro
    modelo não escondem as novas por coincidência de id.

    Retorna:
        Path: o ``index.html`` regenerado por ``render_review_package``.

    Erros (ValueError): taxonomia mudou; ``conf`` fora de (0, 1]; pesos com classes
    diferentes da taxonomia (``check_domain_names``); foto original alterada; erros do
    calibrador e do render.

    Grava: ``suggestions.json`` (sobrescreve) e a página da revisão.
    """
    # Imports tardios: evitam ciclos de importação e só carregam calibração/render quando usados.
    from .calibration import load_calibrator

    from .review_render import render_review_package

    # 1. Registro na taxonomia atual e confiança mínima (padrão: pilot.suggestion_conf).
    registry_path = Path(registry_path).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("taxonomy_sha256") != file_hash(config.taxonomy_path):
        raise ValueError("Taxonomia mudou: migre a revisão antes de sugerir caixas.")
    conf = float(conf if conf is not None else config.pilot.get("suggestion_conf", config.prediction["conf"]))
    if not 0 < conf <= 1:
        raise ValueError("conf deve estar no intervalo (0, 1].")
    # 2. Modelo com as 14 classes; pesos piloto inferem no tamanho do treino (640).
    # .onnx exportado roda na GPU (DirectML) por padrão; .pt roda no PyTorch (CPU).
    model = load_inference_model(config, weights, onnx_provider)
    # ATENCAO: recusa pesos cujas classes não sejam exatamente as da taxonomia (ex.: COCO).
    uncovered = check_domain_names(model, detection_names(load_taxonomy(config.taxonomy_path)), allow_prefix=True)
    pilot = is_pilot_checkpoint(model.dronecamp_source)
    # A partir daqui "config" é a cópia ajustada (imgsz do piloto/ONNX, ou o --imgsz explícito).
    config = pilot_inference_config(config, model)
    if imgsz is not None:
        if imgsz < 32:
            raise ValueError("imgsz deve ser pelo menos 32.")
        config = replace(config, prediction={**config.prediction, "imgsz": int(imgsz)})
    calibration = load_calibrator(config, calibrator, allow_unproven_calibrator)
    # 2a. Busca aberta opcional: (prompts, modelo YOLOE, checkpoint, descrição para o JSON).
    open_search = None
    if zero_shot:
        from .autolabel import describe, load_prompts, zero_shot_detector

        spec = load_prompts(config)
        open_model, open_checkpoint = zero_shot_detector(config, spec)
        open_search = (spec, open_model, open_checkpoint, describe(spec, open_checkpoint))
    # Achados da busca aberta fora da taxonomia, por hash da foto (só registro).
    uncatalogued = {}
    # 3. Roda o modelo em cada foto do registro (conferindo o hash do original).
    # 3-. Fotos já decididas (aprovadas no registro ou em --keep-decided) não são reclassificadas:
    #     mantêm as sugestões anteriores deste mesmo registro, com os mesmos ids.
    kept_ids = set(keep or ()) | {item["image_sha256"] for item in registry["images"] if item.get("human_approved") is True}
    previous_path = registry_path.parent / "suggestions.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.is_file() else {}
    if previous.get("registry_sha256") != file_hash(registry_path):
        previous = {}
    kept = {digest: (previous.get("images") or {}).get(digest, []) for digest in kept_ids}
    images = {}
    for item in registry["images"]:
        if item["image_sha256"] in kept:
            continue
        source = resolve_local_path(item["source_path"], config.root)
        if file_hash(source) != item["image_sha256"]:
            raise ValueError(f"A foto original mudou: {item.get('filename')}.")
        boxes = detect_boxes(model, source, config, conf)
        # 3a. Junta as caixas da busca aberta (o mesmo objeto achado pelos dois vira uma só).
        if open_search:
            from .autolabel import merge_suggestions, zero_shot_boxes

            open_boxes, extra = zero_shot_boxes(open_search[1], source, open_search[0], config.device)
            boxes = merge_suggestions(boxes, open_boxes)
            if extra:
                uncatalogued[item["image_sha256"]] = extra
        # 3b. Calibrador: acrescenta acceptance_probability e reordena (não remove caixas).
        if calibration:
            width, height = review_image_size(source)
            boxes = calibration.annotate(boxes, width, height)
        images[item["image_sha256"]] = boxes
    # 4. suggestions.json fica preso a esta versão do registro (hash) e aos pesos usados.
    # Pesos: o .onnx quando a inferência foi em ONNX, senão o .pt de origem.
    checkpoint = Path(getattr(model, "dronecamp_onnx", None) or model.dronecamp_source).resolve()
    weights_sha256 = file_hash(checkpoint)
    # Ids presos ao modelo: "s001" de outro modelo, já descartado no navegador, não esconde este.
    for boxes in images.values():
        for box in boxes:
            box["id"] = f"{weights_sha256[:8]}-{box['id']}"
    # Fotos mantidas entram depois do prefixo: os ids antigos não mudam.
    images.update({digest: boxes for digest, boxes in kept.items() if digest in {item["image_sha256"] for item in registry["images"]}})
    # EXTENSAO: novo campo em suggestions.json: acrescente neste dict; para a página vê-lo,
    # inclua a chave na tupla de "suggestions" do payload em review_render.render_review_package.
    write_json(registry_path.parent / "suggestions.json", {
        "schema_version": 1, "registry_sha256": file_hash(registry_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weights": str(checkpoint), "weights_sha256": weights_sha256, "pilot": pilot,
        # Classes da taxonomia que estes pesos não propõem (pesos de taxonomia anterior).
        "uncovered_classes": uncovered, "imgsz_override": imgsz,
        # Fotos já decididas que mantiveram as sugestões anteriores (não reclassificadas).
        "kept_previous": {"images": len(kept), "weights_sha256": previous.get("weights_sha256")} if kept else None,
        # Pesos fora de runs/ não são reconhecidos como piloto; com classes a menos ou --imgsz
        # são de um treino anterior e também não foram validados para produção.
        "model_label": "Modelo piloto (não validado)" if pilot
        else "Modelo de treino anterior (não validado)" if uncovered or imgsz is not None else "Modelo especializado",
        "runtime": model.dronecamp_runtime, "source_checkpoint": str(model.dronecamp_source),
        "tiling": config.pilot.get("tiling"),
        "conf": conf, "imgsz": config.prediction["imgsz"], "warning": SUGGESTION_WARNING, "images": images,
        "calibrator": calibration.describe() if calibration else None,
        "zero_shot": open_search[3] if open_search else None,
        # Possíveis não conformidades fora da taxonomia: só registro para decisão humana.
        "uncatalogued": uncatalogued,
        "total_suggestions": sum(len(values) for values in images.values()),
    })
    # 5. Regenera index.html com as sugestões tracejadas.
    return render_review_package(config, registry_path)


# ENTRADA: comando suggest --source <fotos> --output <pasta nova> --group <edificação>.
def create_review_for_new_images(config: ProjectConfig, source: Path, output: Path, building_group: str,
                                 weights: str, conf: float | None = None, calibrator: Path | str | None = None,
                                 allow_unproven_calibrator: bool = False, zero_shot: bool = False,
                                 onnx_provider: str = "directml", imgsz: int | None = None) -> Path:
    """Fotos novas viram uma revisão própria, com originais congelados por hash.

    Parâmetros:
        config (ProjectConfig): configuração do projeto.
        source (Path): uma foto ou uma pasta (procurada recursivamente) de fotos.
        output (Path): pasta NOVA da revisão (ex.: ``data/reviews/galpao_b_v1``).
        building_group (str): edificação/campanha (1 a 100 caracteres), ex.:
            ``galpao_b_2026_10``; vira ``building_group`` de todas as fotos.
        weights, conf, calibrator, allow_unproven_calibrator, zero_shot, onnx_provider:
            repassados a ``suggest_for_registry``.

    Retorna:
        Path: o ``index.html`` da nova revisão.

    Erros (ValueError): saída já existe; grupo vazio/longo; nenhuma foto suportada;
    rotação EXIF em alguma foto; e os erros de ``suggest_for_registry``.

    Grava: ``<output>/originals/<hash>.<ext>``, ``<output>/registry.json``, depois
    ``suggestions.json`` e a página.

    ATENCAO: todas as fotos entram como ``ambiguous``, sem caixas e sem aprovação; só
    viram dados de treino depois da revisão na página e de import-review.
    """
    # 1. Pasta de saída nova, grupo da edificação e lista de fotos suportadas.
    source, output = Path(source).expanduser().resolve(), Path(output)
    if output.exists():
        raise ValueError("Saída já existe; escolha uma nova versão da revisão.")
    if not building_group or not building_group.strip() or len(building_group) > 100:
        raise ValueError("Informe o grupo da edificação/campanha (ex.: galpao_b_2026_10).")
    # Uma foto só, ou todas as fotos da pasta (subpastas incluídas), em ordem de caminho.
    files = [source] if source.is_file() else sorted(
        path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    # Filtra também o caso de uma foto só com extensão não suportada.
    files = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        raise ValueError(f"Nenhuma foto suportada em {source}.")
    # Confere todas antes de copiar qualquer uma (falha cedo, sem pasta pela metade).
    for path in files:
        # Rotação EXIF precisa de cópia canonicalizada antes da anotação.
        review_image_size(path)
    # 2. Copia cada foto para originals/<sha256> (fotos repetidas entram uma vez).
    output = output.resolve()
    originals = output / "originals"
    originals.mkdir(parents=True)
    images, seen = [], set()
    for path in files:
        digest = file_hash(path)
        if digest in seen:
            continue
        seen.add(digest)
        frozen = originals / f"{digest}{path.suffix.lower()}"
        shutil.copyfile(path, frozen)
        width, height = review_image_size(frozen)
        # Entrada do registro (formato no cabeçalho de review_data.py). source_path aponta
        # para a cópia congelada; original_path guarda de onde ela veio.
        # EXTENSAO: novo campo padrão de foto nova: acrescente neste dict e no bloco
        # equivalente de ingest.py (os dois precisam gerar registros iguais).
        # ATENCAO: human_approved=False e severity=None são deliberados.
        images.append({
            "image_sha256": digest, "filename": path.name, "pages": [], "width": width, "height": height,
            "status": "ambiguous", "boxes": [], "annotator": "sem_anotacao_humana",
            "notes": "Foto nova: nenhuma caixa revisada. Use as sugestões do modelo como ponto de partida.",
            "visual_review_status": "model_suggestions_only", "technical_status": "pending_human_review",
            "human_approved": False, "building_group": building_group.strip(),
            "scene_group": f"{building_group.strip()}_{digest[:12]}", "source_path": str(frozen),
            "original_path": str(path), "severity": None,
        })
    # 3. Registro novo (todas ambíguas, sem caixas) e, em seguida, as sugestões.
    registry = {
        "schema_version": REVIEW_SCHEMA, "version": output.name,
        "taxonomy_sha256": file_hash(config.taxonomy_path),
        "taxonomy_version": load_taxonomy(config.taxonomy_path)["version"],
        "sources": [{"folder": str(source), "files": len(images)}], "images": images,
        "scope": "fotos_novas_com_sugestoes_do_modelo_pendentes_de_revisao",
    }
    registry_path = output / "registry.json"
    write_json(registry_path, registry)
    return suggest_for_registry(config, registry_path, weights, conf, calibrator, allow_unproven_calibrator, zero_shot,
                                onnx_provider, imgsz)
