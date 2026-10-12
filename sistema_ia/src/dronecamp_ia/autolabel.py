"""Marcação automática: a IA procura e classifica não conformidades sozinha; o humano decide.

Função no projeto: o detector piloto só reconhece o que viu nas 37 fotos do CEASA
(ex.: não achou nada na foto ``detritos-calha`` de outra edificação). Este módulo
junta a ele um detector de **vocabulário aberto** (YOLOE, Ultralytics), que
procura objetos descritos em texto ("debris in roof gutter", "plants growing in
gutter") sem precisar de exemplos anotados. Assim a IA propõe caixas também para
classes raras e para edificações novas.

Como funciona:
1. ``configs/zero_shot_prompts.json`` liga cada classe da taxonomia a frases em
   inglês; ``uncatalogued`` lista possíveis não conformidades fora da taxonomia
   (ferrugem, água parada, ninho...).
2. ``zero_shot_detector``: na primeira vez, o YOLOE codifica as frases com o
   MobileCLIP e grava um checkpoint com as frases embutidas em
   ``models/zero_shot/``. Depois disso não precisa mais do codificador de texto.
3. ``zero_shot_boxes``: roda o YOLOE e converte cada frase na classe do projeto;
   caixas que cobrem mais de ``max_area_fraction`` da foto são descartadas
   (o modelo às vezes marca o telhado inteiro).
4. ``merge_suggestions``: une com as caixas do piloto. Caixa achada pelos dois
   (mesma classe, IoU >= 0,5) vira uma só, com ``source = "piloto+busca_aberta"``.

Tudo continua pendente: as caixas vão para ``suggestions.json`` e a página de
revisão mostra Aceitar / Descartar / Substituir. Achados fora da taxonomia ficam
em ``uncatalogued`` no mesmo arquivo, para o revisor decidir se viram classe nova
(``migrate-review``). Nada é aprovado sozinho.

Quem usa: ``suggest --zero-shot`` e ``evaluate-autolabel``.
(Concretamente: ``suggestions.suggest_for_registry`` chama ``load_prompts``,
``zero_shot_detector``, ``zero_shot_boxes``, ``merge_suggestions`` e ``describe``;
``cli.py`` chama ``evaluate_autolabel``.)

O que o arquivo contém:
- Constantes: ``PROMPTS_FILE`` e os rótulos de origem ``SOURCE_*``.
- ``load_prompts``: lê e valida o arquivo de frases.
- ``zero_shot_detector``: cria/reaproveita o checkpoint YOLOE com as frases.
- ``zero_shot_boxes`` e ``_dedupe``: inferência e limpeza de duplicatas.
- ``merge_suggestions``: une piloto + busca aberta.
- ``describe``: identificação do YOLOE para gravar em JSON.
- ``evaluate_autolabel`` (ENTRADA de ``evaluate-autolabel``).

Disco:
- Lê: ``configs/zero_shot_prompts.json``, ``configs/taxonomy.json``, o peso YOLOE
  base (``models/<model>`` ou download da Ultralytics) e, na avaliação, o dataset
  piloto (pilot.json, fotos e labels) e os pesos do piloto.
- Grava: ``models/zero_shot/<modelo>_<hash12>.pt`` (cache das frases, uma vez por
  versão das frases) e ``runs/autolabel_eval_<data>_<id>/report.json`` (avaliação).

Onde mexer:
- Para acrescentar ou trocar frases de uma classe, ou um achado fora da
  taxonomia: edite só ``configs/zero_shot_prompts.json`` (sem código). O cache é
  recriado sozinho porque o nome do arquivo depende do hash das frases.
- Para mudar a confiança mínima ou o limite de área: ``conf`` e
  ``max_area_fraction`` no mesmo JSON.
- Para um novo filtro de caixa (ex.: tamanho mínimo): EXTENSAO em ``zero_shot_boxes``.
- Para uma nova combinação na avaliação: EXTENSAO em ``evaluate_autolabel``.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .config import ProjectConfig, detection_names, load_taxonomy
from .io import file_hash, make_run_directory, write_json
from .matching import IOU_MATCH, image_errors, per_class_report, yolo_label_boxes
from .pilot import box_iou

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Nome do arquivo de frases; é procurado na pasta da taxonomia (configs/).
PROMPTS_FILE = "zero_shot_prompts.json"  # na mesma pasta da taxonomia (configs/)
# Valores do campo "source" de cada sugestão em suggestions.json:
#   "piloto"               caixa só do detector treinado
#   "busca_aberta"         caixa só do YOLOE
#   "piloto+busca_aberta"  os dois acharam o mesmo objeto (mesma classe, IoU >= 0,5)
# ATENCAO: esses textos ficam gravados nos suggestions.json já gerados e são
# conferidos em tests/test_e2e_ultralytics.py; mudar um deles exige procurar o
# texto antigo no projeto inteiro (testes, docs e quem ler suggestions.json antigos).
SOURCE_PILOT, SOURCE_OPEN, SOURCE_BOTH = "piloto", "busca_aberta", "piloto+busca_aberta"


# ---------------------------------------------------------------------------
# Frases e detector de vocabulário aberto (YOLOE)
# ---------------------------------------------------------------------------

def load_prompts(config: ProjectConfig) -> dict:
    """Lê e confere as frases: toda classe citada precisa existir na taxonomia.

    Parâmetros:
        config: configuração do projeto (``taxonomy_path`` diz onde fica configs/).

    Retorna:
        O conteúdo do JSON (``model``, ``conf``, ``max_area_fraction``, ``classes``,
        ``uncatalogued``...) acrescido de ``vocabulary`` (lista de tuplas
        ``(frase, class_id ou None)`` na ordem do checkpoint) e ``sha256`` (do arquivo).

    Erros:
        ValueError quando o JSON cita classe fora da taxonomia, quando
        ``max_area_fraction`` sai de (0, 1] ou quando há frase repetida;
        OSError/JSONDecodeError se o arquivo faltar ou estiver inválido.
    """
    # 1. Arquivo e nomes das classes da taxonomia.
    path = config.taxonomy_path.parent / PROMPTS_FILE
    spec = json.loads(path.read_text(encoding="utf-8"))
    names = detection_names(load_taxonomy(config.taxonomy_path))
    # 2. Validações: classes conhecidas e limite de área válido.
    unknown = sorted(set(spec["classes"]) - set(names))
    if unknown:
        raise ValueError(f"{PROMPTS_FILE} cita classes fora da taxonomia: {', '.join(unknown)}.")
    if not 0 < float(spec.get("max_area_fraction", 1)) <= 1:
        raise ValueError("max_area_fraction deve estar no intervalo (0, 1].")
    # Vocabulário na ordem do checkpoint: (frase, id da classe ou None se fora da taxonomia).
    # ATENCAO: o índice de cada frase nesta lista é o "cls" que o YOLOE devolve;
    # zero_shot_boxes usa spec["vocabulary"][índice] para voltar à classe do projeto.
    vocabulary = [(prompt, names.index(slug)) for slug, prompts in spec["classes"].items() for prompt in prompts]
    vocabulary += [(prompt, None) for prompt in spec.get("uncatalogued", [])]
    # Frase repetida tornaria ambíguo o mapeamento frase -> classe.
    if len({prompt for prompt, _ in vocabulary}) != len(vocabulary):
        raise ValueError(f"Frases repetidas em {PROMPTS_FILE}.")
    spec["vocabulary"] = vocabulary
    # Hash do arquivo: registrado em suggestions.json/report.json para auditoria.
    spec["sha256"] = file_hash(path)
    return spec


def zero_shot_detector(config: ProjectConfig, spec: dict):
    """YOLOE com as frases embutidas; criado uma vez por versão das frases e reaproveitado.

    Parâmetros:
        config: configuração do projeto (pastas de modelos e downloads).
        spec: saída de ``load_prompts``.

    Retorna:
        ``(model, cached)``: o ``ultralytics.YOLOE`` carregado do cache e o Path do
        checkpoint ``models/zero_shot/<modelo>_<hash12>.pt``.

    Erros:
        ValueError quando o checkpoint em cache não tem as mesmas frases (cache
        corrompido ou editado à mão: apague o arquivo para recriar). Na primeira
        execução pode falhar por falta de rede (download do YOLOE e do MobileCLIP).
    """
    from .backend import _ultralytics

    _ultralytics(config)  # pastas e downloads dentro de sistema_ia
    from ultralytics import YOLOE

    # 1. Nome do cache: 12 primeiros hex do SHA-256 de [modelo, frases].
    # Mudar qualquer frase, a ordem delas ou o modelo gera outro arquivo.
    prompts = [prompt for prompt, _ in spec["vocabulary"]]
    key = sha256(json.dumps([spec["model"], prompts]).encode("utf-8")).hexdigest()[:12]
    cached = config.root / "models" / "zero_shot" / f"{Path(spec['model']).stem}_{key}.pt"
    # 2. Sem cache: cria (usa models/<model> local se existir, senão baixa).
    if not cached.is_file():
        # Primeira vez: baixa o YOLOE oficial e o MobileCLIP e codifica as frases.
        base = config.root / "models" / spec["model"]
        model = YOLOE(str(base) if base.is_file() else spec["model"])
        model.set_classes(prompts)
        cached.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(cached))
    # 3. Carrega do cache e confere que as "classes" dele são exatamente as frases.
    model = YOLOE(str(cached))
    # model.names pode vir como lista ou como dict {índice: nome}; normaliza para lista.
    names = model.names if isinstance(model.names, list) else [model.names[index] for index in range(len(model.names))]
    if names != prompts:
        raise ValueError(f"Checkpoint {cached.name} não corresponde às frases de {PROMPTS_FILE}; apague-o para recriar.")
    return model, cached


def zero_shot_boxes(model, path: Path, spec: dict, device: str) -> tuple[list[dict], list[dict]]:
    """Caixas do YOLOE convertidas para as classes do projeto, e achados fora da taxonomia.

    Parâmetros:
        model: YOLOE de ``zero_shot_detector``.
        path: foto a analisar.
        spec: saída de ``load_prompts`` (``vocabulary``, ``conf``, ``max_area_fraction``).
        device: dispositivo da Ultralytics (ex.: ``"cpu"``; vem de ``config.device``).

    Retorna:
        ``(catalog, uncatalogued)``: ``catalog`` são caixas
        ``{"class_id", "bbox_xyxy" (pixels inteiros), "confidence", "source": "busca_aberta",
        "prompt"}``; ``uncatalogued`` tem o mesmo formato sem ``class_id``. Ambas
        já sem duplicatas (``_dedupe``).
    """
    from .review_data import review_image_size

    # Tamanho real da foto (para recortar e medir a área).
    width, height = review_image_size(path)
    # stream=True + next(): processa só esta foto sem guardar a lista de resultados.
    result = next(iter(model.predict(source=str(path), conf=float(spec.get("conf", 0.15)), device=device,
                                     save=False, verbose=False, stream=True)))
    catalog, uncatalogued = [], []
    if result.boxes is None:
        return catalog, uncatalogued
    # Área máxima permitida, como fração da foto (0,5 = metade).
    limit = float(spec.get("max_area_fraction", 1))
    # Cada detecção: caixa xyxy (pixels), confiança e índice da frase.
    for (x1, y1, x2, y2), confidence, index in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                                                   result.boxes.cls.cpu().tolist()):
        # Arredonda e recorta a caixa aos limites da foto (0..largura, 0..altura).
        box = [max(0, min(width, round(x1))), max(0, min(height, round(y1))),
               max(0, min(width, round(x2))), max(0, min(height, round(y2)))]
        # Caixa sem área depois do recorte: descarta.
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if (box[2] - box[0]) * (box[3] - box[1]) > limit * width * height:
            continue  # cobre quase a foto toda: não localiza uma ocorrência
        # EXTENSAO: filtro novo de caixa (ex.: área mínima em pixels): acrescente um
        # "if ...: continue" aqui e o parâmetro em configs/zero_shot_prompts.json.
        # Índice da frase -> (frase, classe do projeto ou None).
        prompt, class_id = spec["vocabulary"][int(index)]
        entry = {"bbox_xyxy": box, "confidence": round(float(confidence), 4), "source": SOURCE_OPEN, "prompt": prompt}
        # Frase "uncatalogued" (fora da taxonomia) vai para a lista separada.
        if class_id is None:
            uncatalogued.append(entry)
        else:
            catalog.append({"class_id": class_id, **entry})
    # Fora da taxonomia não tem classe: duplicatas são removidas só pela posição.
    return _dedupe(catalog), _dedupe(uncatalogued, by_class=False)


def _dedupe(boxes: list[dict], by_class: bool = True) -> list[dict]:
    """Várias frases da mesma classe acham o mesmo objeto: fica a caixa mais confiante.

    Parâmetros:
        boxes: caixas de uma foto.
        by_class: True = só caixas da MESMA classe competem; False = qualquer
            sobreposição conta (usado nos achados sem classe).

    Retorna:
        Lista sem duplicatas (IoU >= IOU_MATCH), da maior para a menor confiança.
    """
    kept = []
    # Gulosa: da mais confiante para a menos; uma caixa só entra se não duplicar
    # nenhuma já mantida.
    for box in sorted(boxes, key=lambda value: -value["confidence"]):
        # Para cada mantida: "não conflita" se for de outra classe (com by_class) ou
        # se o IoU for menor que o limiar. Com by_class=False a 1ª parte é sempre
        # falsa e o class_id nem é lido (achados sem classe não o têm).
        if all((by_class and other["class_id"] != box["class_id"]) or box_iou(other["bbox_xyxy"], box["bbox_xyxy"]) < IOU_MATCH
               for other in kept):
            kept.append(box)
    return kept


# ---------------------------------------------------------------------------
# União com as sugestões do piloto
# ---------------------------------------------------------------------------

def merge_suggestions(pilot: list[dict], open_boxes: list[dict]) -> list[dict]:
    """Une piloto e busca aberta; o mesmo objeto achado pelos dois vira uma sugestão só.

    Parâmetros:
        pilot: caixas do detector treinado (``suggestions.detect_boxes``).
        open_boxes: caixas da taxonomia vindas de ``zero_shot_boxes`` (sem os achados
            fora da taxonomia).

    Retorna:
        Lista nova, da maior para a menor confiança, com ``source`` em cada caixa e
        ``id`` sequencial ``s001``, ``s002``...
    """
    # 1. Todas as do piloto entram, marcadas com a origem.
    merged = [{**box, "source": SOURCE_PILOT} for box in pilot]
    # 2. Cada caixa aberta: procura uma "gêmea" do piloto (mesma classe, IoU >= 0,5).
    for box in open_boxes:
        twin = next((item for item in merged if item["class_id"] == box["class_id"] and item["source"] == SOURCE_PILOT
                     and box_iou(item["bbox_xyxy"], box["bbox_xyxy"]) >= IOU_MATCH), None)
        if twin:
            # Mantém a caixa do piloto, marca as duas origens e fica com a maior confiança.
            twin.update(source=SOURCE_BOTH, prompt=box["prompt"], confidence=max(twin["confidence"], box["confidence"]))
        else:
            merged.append(dict(box))
    # 3. Ordena e numera.
    merged.sort(key=lambda value: -value["confidence"])
    for index, box in enumerate(merged, start=1):
        # ATENCAO: o id muda se a lista mudar (é a posição); a página usa esse id para
        # descartar, então ele vale só para o suggestions.json em que foi gravado.
        box["id"] = f"s{index:03d}"  # ids usados pela página ao descartar
    return merged


def describe(spec: dict, checkpoint: Path) -> dict:
    """Identificação da busca aberta para gravar em suggestions.json e report.json.

    Parâmetros:
        spec: saída de ``load_prompts``.
        checkpoint: Path do cache YOLOE usado.

    Retorna:
        ``{"model", "checkpoint", "checkpoint_sha256", "prompts_sha256", "conf",
        "max_area_fraction"}``.
    """
    return {"model": spec["model"], "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
            "prompts_sha256": spec["sha256"], "conf": spec.get("conf"), "max_area_fraction": spec.get("max_area_fraction")}


# ---------------------------------------------------------------------------
# Avaliação: piloto sozinho × busca aberta sozinha × as duas juntas, por classe.
# ---------------------------------------------------------------------------

# ENTRADA: comando ``evaluate-autolabel`` (cli.py chama esta função).
def evaluate_autolabel(config: ProjectConfig, data_path: Path, weights: str, conf: float | None = None) -> Path:
    """Mede recall/precisão por classe nas fotos com caixas humanas e grava runs/autolabel_eval_*/report.json.

    Fotos do treino do piloto ficam fora das linhas "piloto" e "piloto+busca"; a
    busca aberta nunca viu o CEASA, então é medida também em todas as fotos.

    Parâmetros:
        config: configuração do projeto.
        data_path: dataset.yaml do dataset piloto (pilot.json na mesma pasta).
        weights: pesos do detector piloto.
        conf: confiança mínima das sugestões do piloto; None = ``pilot.suggestion_conf``
            ou ``prediction.conf``. (A confiança do YOLOE vem do JSON de frases.)

    Retorna:
        Path de ``runs/autolabel_eval_<data>_<id>`` (contém report.json com
        ``per_class`` para cada uma das 4 linhas de avaliação).

    Erros:
        ValueError de ``check_domain_names``, ``load_prompts`` e ``zero_shot_detector``.
    """
    from .backend import check_domain_names, load_detector
    from .calibration import trained_image_hashes
    from .review_data import review_image_size
    from .suggestions import detect_boxes
    from .training import pilot_inference_config

    # 1. Taxonomia, manifesto, confiança e os dois detectores.
    names = detection_names(load_taxonomy(config.taxonomy_path))
    root = Path(data_path).resolve().parent
    manifest = json.loads((root / "pilot.json").read_text(encoding="utf-8"))
    conf = float(conf if conf is not None else config.pilot.get("suggestion_conf", config.prediction["conf"]))
    pilot = load_detector(config, weights)
    check_domain_names(pilot, names)
    pilot_config = pilot_inference_config(config, pilot)
    # Fotos do treino do piloto (excluídas das linhas que envolvem o piloto).
    seen = trained_image_hashes(Path(pilot.ckpt_path))
    spec = load_prompts(config)
    open_model, checkpoint = zero_shot_detector(config, spec)
    # 2. Uma lista de erros por foto para cada linha de comparação.
    # EXTENSAO: nova combinação (ex.: "busca_aberta_conf_alta"): crie a chave aqui e
    # faça o append correspondente no laço; ela aparece sozinha em per_class.
    errors = {"piloto": [], "busca_aberta": [], "piloto+busca_aberta": [], "busca_aberta_todas_as_fotos": []}
    uncatalogued = 0
    for sample in manifest["images"]:
        image = root / sample["image"]
        width, height = review_image_size(image)
        truth = yolo_label_boxes(root / sample["label"], width, height)
        # Busca aberta em TODAS as fotos (ela nunca treinou com o CEASA).
        open_boxes, extra = zero_shot_boxes(open_model, image, spec, config.device)
        uncatalogued += len(extra)
        errors["busca_aberta_todas_as_fotos"].append(image_errors(open_boxes, truth))
        # Daqui para baixo só fotos fora do treino do piloto (comparação justa).
        if sample["image_sha256"] in seen:
            continue
        pilot_boxes = detect_boxes(pilot, image, pilot_config, conf)
        errors["piloto"].append(image_errors(pilot_boxes, truth))
        errors["busca_aberta"].append(image_errors(open_boxes, truth))
        errors["piloto+busca_aberta"].append(image_errors(merge_suggestions(pilot_boxes, open_boxes), truth))
    # 3. Grava o relatório numa pasta nova em runs/.
    run = make_run_directory(config.root, "autolabel_eval")
    # EXTENSAO: campo novo no report.json: acrescente neste dict.
    write_json(run / "report.json", {
        "schema_version": 1, "dataset": str(Path(data_path).resolve()), "pilot_weights": str(Path(pilot.ckpt_path)),
        "pilot_conf": conf, "zero_shot": describe(spec, checkpoint), "iou_match": IOU_MATCH,
        "images_outside_pilot_training": len(errors["piloto"]), "images_total": len(manifest["images"]),
        "uncatalogued_findings": uncatalogued,
        # Relatório por classe (TP/FP/FN, precisão, recall) de cada linha.
        "per_class": {key: per_class_report(value, names) for key, value in errors.items()},
        "note": "Uma única edificação; mede se a busca aberta acrescenta acertos, não prontidão para produção.",
    })
    return run
