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
"""

from pathlib import Path
from hashlib import sha256
import json

from .backend import check_domain_names, load_detector, runtime_info
from .config import ProjectConfig, detection_names, load_taxonomy
from .io import file_hash, make_run_directory, write_json

# Fotos e vídeos aceitos (vídeos são lidos quadro a quadro).
MEDIA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv"}


def find_media(source: Path) -> list[Path]:
    """Aceite somente mídia local; uma pasta é percorrida recursivamente."""
    source = source.expanduser().resolve()
    if source.is_file() and source.suffix.lower() in MEDIA_EXTENSIONS:
        return [source]
    if source.is_dir():
        files = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS)
        if files:
            return files
    raise ValueError(f"Nenhuma imagem/vídeo local suportado encontrado em {source}.")


def serialize_detections(result, taxonomy: dict, demo: bool) -> list[dict]:
    """Separe a classe visual da severidade: esta será confirmada na revisão."""
    if result.boxes is None:
        return []
    categories = {item["slug"]: item for item in taxonomy["classes"]}
    boxes = result.boxes
    detections = []
    for coordinates, confidence, class_id in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
        name = result.names[int(class_id)]
        category = categories.get(name) if not demo else None
        detections.append({
            "class_id": int(class_id), "class_name": name,
            "confidence": round(float(confidence), 6),
            "bbox_xyxy_pixels": [round(float(value), 2) for value in coordinates],
            "category_label": category["label"] if category else None,
            "severity": None, "review_status": "pendente",
            "report_reference": {
                "document": taxonomy["source"], "pages": category["pages"],
                "historical_severity_only": category["reference_severity"],
            } if category else None,
        })
    return detections


def predict(config: ProjectConfig, source: Path, weights: str | None = None, demo: bool = False) -> Path:
    """Gere imagens anotadas e JSONL incremental, inclusive quando não há caixas."""
    # 1. Mídia, taxonomia e modelo (--demo aceita pesos COCO genéricos, só para teste).
    media = find_media(source)
    taxonomy = load_taxonomy(config.taxonomy_path)
    model = load_detector(config, weights)
    if not demo:
        check_domain_names(model, detection_names(taxonomy))
    from .training import is_pilot_checkpoint

    pilot = not demo and is_pilot_checkpoint(model.ckpt_path)
    if pilot:
        from .training import pilot_inference_config

        config = pilot_inference_config(config, model)
    # 2. Pasta da execução e execution.json com modo, versões e hashes das mídias.
    run = make_run_directory(config.root, "predict_demo" if demo else "predict_pilot" if pilot else "predict")
    evidence = run / "evidence"
    evidence.mkdir()
    mode = "demo_pesos_genericos" if demo else "candidatos_modelo_piloto_nao_validado" if pilot else "candidatos_revisao"
    write_json(run / "execution.json", {
        "mode": "demo_pesos_genericos" if demo else "deteccao_piloto_nao_validada" if pilot else "deteccao_especializada_para_revisao",
        "taxonomy_version": taxonomy["version"], "taxonomy_sha256": file_hash(config.taxonomy_path),
        "runtime": runtime_info(model, config), "parameters": config.prediction, "pilot": pilot,
        "warning": "Demonstração técnica sem reconhecimento treinado de patologias." if demo
        else "Modelo piloto treinado com uma única edificação; use só como sugestão para revisão humana." if pilot
        else "Achados candidatos exigem revisão; nenhum laudo é emitido automaticamente.",
        "files": [{"path": str(path), "sha256": file_hash(path)} for path in media],
    })
    # 3. Cada foto/quadro: imagem com caixas em evidence/ e uma linha em findings.jsonl.
    processed = 0
    with (run / "findings.jsonl").open("w", encoding="utf-8") as stream:
        for path in media:
            key = sha256(str(path).encode("utf-8")).hexdigest()[:12]
            # stream=True mantém o consumo de memória limitado durante vídeos.
            results = model.predict(source=str(path), stream=True, device=config.device,
                                    save=False, verbose=False, rect=False, vid_stride=1,
                                    **config.prediction)
            for ordinal, result in enumerate(results, start=1):
                evidence_path = evidence / f"{key}_{ordinal:06d}.jpg"
                result.save(filename=str(evidence_path))
                detections = serialize_detections(result, taxonomy, demo)
                row = {
                    "source": str(path), "frame_index_processed": ordinal,
                    "image_size_hw": list(result.orig_shape), "evidence": str(evidence_path),
                    "mode": mode,
                    "status": "candidatos_detectados" if detections else "sem_deteccoes_acima_limiar",
                    "detections": detections, "review_status": "pendente",
                    "speed_ms": {key: float(value) for key, value in result.speed.items()},
                }
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                processed += 1
    # 4. summary.json: só "complete" se ao menos uma imagem foi processada.
    if processed == 0:
        raise ValueError(f"Nenhum frame/imagem foi decodificado; execução incompleta em {run}.")
    write_json(run / "summary.json", {
        "source_files": len(media), "processed_images_or_frames": processed,
        "complete": True, "unique_physical_defects_count": None,
        "note": "Caixas em frames diferentes podem mostrar a mesma ocorrência; não somar como defeitos únicos.",
    })
    return run
