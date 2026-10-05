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
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .config import ProjectConfig, detection_names, load_taxonomy
from .io import file_hash, make_run_directory, write_json
from .matching import IOU_MATCH, image_errors, per_class_report, yolo_label_boxes
from .pilot import box_iou

PROMPTS_FILE = "zero_shot_prompts.json"  # na mesma pasta da taxonomia (configs/)
SOURCE_PILOT, SOURCE_OPEN, SOURCE_BOTH = "piloto", "busca_aberta", "piloto+busca_aberta"


def load_prompts(config: ProjectConfig) -> dict:
    """Lê e confere as frases: toda classe citada precisa existir na taxonomia."""
    path = config.taxonomy_path.parent / PROMPTS_FILE
    spec = json.loads(path.read_text(encoding="utf-8"))
    names = detection_names(load_taxonomy(config.taxonomy_path))
    unknown = sorted(set(spec["classes"]) - set(names))
    if unknown:
        raise ValueError(f"{PROMPTS_FILE} cita classes fora da taxonomia: {', '.join(unknown)}.")
    if not 0 < float(spec.get("max_area_fraction", 1)) <= 1:
        raise ValueError("max_area_fraction deve estar no intervalo (0, 1].")
    # Vocabulário na ordem do checkpoint: (frase, id da classe ou None se fora da taxonomia).
    vocabulary = [(prompt, names.index(slug)) for slug, prompts in spec["classes"].items() for prompt in prompts]
    vocabulary += [(prompt, None) for prompt in spec.get("uncatalogued", [])]
    if len({prompt for prompt, _ in vocabulary}) != len(vocabulary):
        raise ValueError(f"Frases repetidas em {PROMPTS_FILE}.")
    spec["vocabulary"] = vocabulary
    spec["sha256"] = file_hash(path)
    return spec


def zero_shot_detector(config: ProjectConfig, spec: dict):
    """YOLOE com as frases embutidas; criado uma vez por versão das frases e reaproveitado."""
    from .backend import _ultralytics

    _ultralytics(config)  # pastas e downloads dentro de sistema_ia
    from ultralytics import YOLOE

    prompts = [prompt for prompt, _ in spec["vocabulary"]]
    key = sha256(json.dumps([spec["model"], prompts]).encode("utf-8")).hexdigest()[:12]
    cached = config.root / "models" / "zero_shot" / f"{Path(spec['model']).stem}_{key}.pt"
    if not cached.is_file():
        # Primeira vez: baixa o YOLOE oficial e o MobileCLIP e codifica as frases.
        base = config.root / "models" / spec["model"]
        model = YOLOE(str(base) if base.is_file() else spec["model"])
        model.set_classes(prompts)
        cached.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(cached))
    model = YOLOE(str(cached))
    names = model.names if isinstance(model.names, list) else [model.names[index] for index in range(len(model.names))]
    if names != prompts:
        raise ValueError(f"Checkpoint {cached.name} não corresponde às frases de {PROMPTS_FILE}; apague-o para recriar.")
    return model, cached


def zero_shot_boxes(model, path: Path, spec: dict, device: str) -> tuple[list[dict], list[dict]]:
    """Caixas do YOLOE convertidas para as classes do projeto, e achados fora da taxonomia."""
    from .review_data import review_image_size

    width, height = review_image_size(path)
    result = next(iter(model.predict(source=str(path), conf=float(spec.get("conf", 0.15)), device=device,
                                     save=False, verbose=False, stream=True)))
    catalog, uncatalogued = [], []
    if result.boxes is None:
        return catalog, uncatalogued
    limit = float(spec.get("max_area_fraction", 1))
    for (x1, y1, x2, y2), confidence, index in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                                                   result.boxes.cls.cpu().tolist()):
        box = [max(0, min(width, round(x1))), max(0, min(height, round(y1))),
               max(0, min(width, round(x2))), max(0, min(height, round(y2)))]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if (box[2] - box[0]) * (box[3] - box[1]) > limit * width * height:
            continue  # cobre quase a foto toda: não localiza uma ocorrência
        prompt, class_id = spec["vocabulary"][int(index)]
        entry = {"bbox_xyxy": box, "confidence": round(float(confidence), 4), "source": SOURCE_OPEN, "prompt": prompt}
        if class_id is None:
            uncatalogued.append(entry)
        else:
            catalog.append({"class_id": class_id, **entry})
    return _dedupe(catalog), _dedupe(uncatalogued, by_class=False)


def _dedupe(boxes: list[dict], by_class: bool = True) -> list[dict]:
    """Várias frases da mesma classe acham o mesmo objeto: fica a caixa mais confiante."""
    kept = []
    for box in sorted(boxes, key=lambda value: -value["confidence"]):
        if all((by_class and other["class_id"] != box["class_id"]) or box_iou(other["bbox_xyxy"], box["bbox_xyxy"]) < IOU_MATCH
               for other in kept):
            kept.append(box)
    return kept


def merge_suggestions(pilot: list[dict], open_boxes: list[dict]) -> list[dict]:
    """Une piloto e busca aberta; o mesmo objeto achado pelos dois vira uma sugestão só."""
    merged = [{**box, "source": SOURCE_PILOT} for box in pilot]
    for box in open_boxes:
        twin = next((item for item in merged if item["class_id"] == box["class_id"] and item["source"] == SOURCE_PILOT
                     and box_iou(item["bbox_xyxy"], box["bbox_xyxy"]) >= IOU_MATCH), None)
        if twin:
            twin.update(source=SOURCE_BOTH, prompt=box["prompt"], confidence=max(twin["confidence"], box["confidence"]))
        else:
            merged.append(dict(box))
    merged.sort(key=lambda value: -value["confidence"])
    for index, box in enumerate(merged, start=1):
        box["id"] = f"s{index:03d}"  # ids usados pela página ao descartar
    return merged


def describe(spec: dict, checkpoint: Path) -> dict:
    return {"model": spec["model"], "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
            "prompts_sha256": spec["sha256"], "conf": spec.get("conf"), "max_area_fraction": spec.get("max_area_fraction")}


# ---------------------------------------------------------------------------
# Avaliação: piloto sozinho × busca aberta sozinha × as duas juntas, por classe.
# ---------------------------------------------------------------------------

def evaluate_autolabel(config: ProjectConfig, data_path: Path, weights: str, conf: float | None = None) -> Path:
    """Mede recall/precisão por classe nas fotos com caixas humanas e grava runs/autolabel_eval_*/report.json.

    Fotos do treino do piloto ficam fora das linhas "piloto" e "piloto+busca"; a
    busca aberta nunca viu o CEASA, então é medida também em todas as fotos.
    """
    from .backend import check_domain_names, load_detector
    from .calibration import trained_image_hashes
    from .review_data import review_image_size
    from .suggestions import detect_boxes
    from .training import pilot_inference_config

    names = detection_names(load_taxonomy(config.taxonomy_path))
    root = Path(data_path).resolve().parent
    manifest = json.loads((root / "pilot.json").read_text(encoding="utf-8"))
    conf = float(conf if conf is not None else config.pilot.get("suggestion_conf", config.prediction["conf"]))
    pilot = load_detector(config, weights)
    check_domain_names(pilot, names)
    pilot_config = pilot_inference_config(config, pilot)
    seen = trained_image_hashes(Path(pilot.ckpt_path))
    spec = load_prompts(config)
    open_model, checkpoint = zero_shot_detector(config, spec)
    errors = {"piloto": [], "busca_aberta": [], "piloto+busca_aberta": [], "busca_aberta_todas_as_fotos": []}
    uncatalogued = 0
    for sample in manifest["images"]:
        image = root / sample["image"]
        width, height = review_image_size(image)
        truth = yolo_label_boxes(root / sample["label"], width, height)
        open_boxes, extra = zero_shot_boxes(open_model, image, spec, config.device)
        uncatalogued += len(extra)
        errors["busca_aberta_todas_as_fotos"].append(image_errors(open_boxes, truth))
        if sample["image_sha256"] in seen:
            continue
        pilot_boxes = detect_boxes(pilot, image, pilot_config, conf)
        errors["piloto"].append(image_errors(pilot_boxes, truth))
        errors["busca_aberta"].append(image_errors(open_boxes, truth))
        errors["piloto+busca_aberta"].append(image_errors(merge_suggestions(pilot_boxes, open_boxes), truth))
    run = make_run_directory(config.root, "autolabel_eval")
    write_json(run / "report.json", {
        "schema_version": 1, "dataset": str(Path(data_path).resolve()), "pilot_weights": str(Path(pilot.ckpt_path)),
        "pilot_conf": conf, "zero_shot": describe(spec, checkpoint), "iou_match": IOU_MATCH,
        "images_outside_pilot_training": len(errors["piloto"]), "images_total": len(manifest["images"]),
        "uncatalogued_findings": uncatalogued,
        "per_class": {key: per_class_report(value, names) for key, value in errors.items()},
        "note": "Uma única edificação; mede se a busca aberta acrescenta acertos, não prontidão para produção.",
    })
    return run
