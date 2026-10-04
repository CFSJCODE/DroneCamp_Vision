"""Exportação para ONNX e conferência de paridade com o .pt.

Função no projeto: gera ``model.onnx`` para rodar fora do Python/Ultralytics
(ex.: ONNX Runtime com DirectML na APU) e prova que ele dá as mesmas caixas
que o checkpoint PyTorch. É o comando ``export``.

O que faz:
- ``export_onnx``: copia o ``.pt`` para ``runs/export_*``, exporta FP32 com
  entrada fixa e grava ``export.json``.
- ``verify_onnx_parity`` / ``compare_outputs``: roda ``.pt`` e ``.onnx`` nas
  mesmas fotos e pareia as caixas (IoU ≥ 0,9 ou ≤ 1 px; Δconfiança ≤ 0,02).

Criar um ONNX não certifica a qualidade do modelo; só a equivalência numérica.

Quando mexer: tolerâncias da paridade (constantes abaixo) ou opções de
exportação (tamanho, FP16/INT8, NMS embutido).
"""

from dataclasses import replace
from pathlib import Path
import importlib.util
import shutil

from .backend import check_domain_names, load_detector, load_exported_detector, runtime_info
from .config import ProjectConfig, detection_names, load_taxonomy
from .dataset import IMAGE_EXTENSIONS
from .io import file_hash, make_run_directory, write_json

# Tolerâncias para considerar duas caixas (.pt × .onnx) iguais.
PARITY_IOU = 0.9
PARITY_CONFIDENCE = 0.02
# Caixas finíssimas mudam muito de IoU com décimos de pixel; aceite até 1 px.
PARITY_PIXELS = 1.0


# ---------------------------------------------------------------------------
# Comparação de saídas: caixas do .pt contra caixas do .onnx.
# ---------------------------------------------------------------------------

def _iou(a: list[float], b: list[float]) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def _boxes(model, image: Path, config: ProjectConfig, conf: float) -> list[tuple[int, float, list[float]]]:
    """Caixas (classe, confiança, xyxy) de um modelo .pt ou .onnx numa foto."""
    parameters = {**config.prediction, "conf": conf}
    # rect=False: o .pt recebe a mesma entrada quadrada fixa do ONNX exportado.
    result = next(iter(model.predict(source=str(image), device="cpu", save=False, verbose=False, stream=True,
                                     rect=False, **parameters)))
    if result.boxes is None:
        return []
    return [(int(c), float(s), [float(v) for v in xyxy]) for xyxy, s, c in
            zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist())]


def compare_outputs(reference: list, candidate: list, conf: float) -> dict:
    """Pareie caixas pelo melhor IoU global; tolere só oscilações junto ao limiar."""
    def similarity(a: list[float], b: list[float]) -> float:
        close = max(abs(x - y) for x, y in zip(a, b)) <= PARITY_PIXELS
        return max(_iou(a, b), 1.0 if close else 0.0)

    # Candidatos a par: mesma classe e confiança próxima; os mais parecidos primeiro.
    pairs = sorted(((similarity(a[2], b[2]), i, j) for i, a in enumerate(reference) for j, b in enumerate(candidate)
                    if a[0] == b[0] and abs(a[1] - b[1]) <= PARITY_CONFIDENCE), reverse=True)
    used_reference, used_candidate, worst_iou, worst_confidence = set(), set(), 1.0, 0.0
    for iou, i, j in pairs:
        if iou < PARITY_IOU:
            break
        if i in used_reference or j in used_candidate:
            continue
        used_reference.add(i)
        used_candidate.add(j)
        worst_iou = min(worst_iou, _iou(reference[i][2], candidate[j][2]) if iou < 1.0 else iou)
        worst_confidence = max(worst_confidence, abs(reference[i][1] - candidate[j][1]))
    # Caixas sem par só contam se estiverem claramente acima do limiar.
    missing = [{"class_id": c, "confidence": round(s, 4)} for index, (c, s, _) in enumerate(reference)
               if index not in used_reference and s - conf > PARITY_CONFIDENCE]
    extra = [{"class_id": c, "confidence": round(s, 4)} for index, (c, s, _) in enumerate(candidate)
             if index not in used_candidate and s - conf > PARITY_CONFIDENCE]
    return {"reference_boxes": len(reference), "exported_boxes": len(candidate), "matched": len(used_reference),
            "missing_in_export": missing, "extra_in_export": extra,
            "worst_matched_iou": round(worst_iou, 4), "worst_confidence_delta": round(worst_confidence, 4),
            "equivalent": not missing and not extra}


# ---------------------------------------------------------------------------
# Paridade e exportação.
# ---------------------------------------------------------------------------

def default_parity_images(config: ProjectConfig, limit: int = 5) -> list[Path]:
    """Sem --parity-source, compara nas 5 primeiras fotos do CEASA."""
    reference = config.root / "data" / "reference" / "ceasa"
    return sorted(path for path in reference.glob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)[:limit]


def _onnx_imgsz(path: Path) -> int | None:
    """Tamanho de entrada gravado nos metadados do ONNX (ex.: 640)."""
    import ast
    import onnxruntime

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    value = session.get_modelmeta().custom_metadata_map.get("imgsz")
    try:
        size = ast.literal_eval(value) if value else None
    except (ValueError, SyntaxError):
        return None
    return int(size[0]) if isinstance(size, (list, tuple)) and size else None


def verify_onnx_parity(config: ProjectConfig, checkpoint: Path, onnx_path: Path, images: list[Path], conf: float) -> dict:
    """Rode o .pt e o .onnx nas mesmas fotos: só assim a exportação é comparável."""
    if not images:
        raise ValueError("Informe ao menos uma foto para verificar a paridade da exportação.")
    reference = load_detector(config, str(checkpoint))
    exported = load_exported_detector(config, onnx_path)
    # O ONNX tem entrada fixa: compare sempre no tamanho gravado nos metadados.
    imgsz = _onnx_imgsz(onnx_path)
    if imgsz:
        config = replace(config, prediction={**config.prediction, "imgsz": imgsz})
    per_image = []
    for image in images:
        comparison = compare_outputs(_boxes(reference, image, config, conf), _boxes(exported, image, config, conf), conf)
        per_image.append({"image": str(image), "sha256": file_hash(image), **comparison})
    return {"conf": conf, "iou_required": PARITY_IOU, "confidence_tolerance": PARITY_CONFIDENCE, "pixel_tolerance": PARITY_PIXELS,
            "images": per_image, "runtime_parity_validated": all(item["equivalent"] for item in per_image),
            "scope": "Paridade numérica PyTorch × ONNX Runtime na CPU deste computador; não avalia qualidade nem outro hardware."}


def export_onnx(config: ProjectConfig, weights: str, demo: bool = False, parity_images: list[Path] | None = None) -> Path:
    """Exporte ONNX FP32, batch 1 e tamanho fixo; confira a paridade com o .pt."""
    if importlib.util.find_spec("onnx") is None:
        raise ValueError("Dependência ONNX ausente. Instale o extra export documentado no README antes de exportar.")
    # 1. Modelo com as 13 classes; pesos piloto exportam no tamanho de treino.
    model = load_detector(config, weights)
    if not demo:
        from .training import pilot_inference_config

        check_domain_names(model, detection_names(load_taxonomy(config.taxonomy_path)))
        config = pilot_inference_config(config, model)
    # 2. Cópia do .pt na pasta da exportação e exportação FP32, batch 1, tamanho fixo.
    run = make_run_directory(config.root, "export_demo" if demo else "export")
    checkpoint = run / "model.pt"
    shutil.copy2(model.ckpt_path, checkpoint)
    # O exportador grava ao lado do checkpoint: uma cópia evita tocar o original.
    model = load_detector(config, str(checkpoint))
    output = model.export(format="onnx", device="cpu", imgsz=config.prediction["imgsz"],
                          batch=1, dynamic=False, simplify=False, nms=config.prediction["nms"], quantize=32)
    # 3. Paridade .pt × .onnx (parity.json) e resumo da exportação (export.json).
    images = parity_images if parity_images is not None else default_parity_images(config)
    parity = None
    if importlib.util.find_spec("onnxruntime") is not None and images:
        parity = verify_onnx_parity(config, checkpoint, Path(output), images, min(0.1, float(config.prediction["conf"])))
        write_json(run / "parity.json", parity)
    write_json(run / "export.json", {
        "runtime": runtime_info(model, config), "artifact": str(output),
        "sha256": file_hash(Path(output)), "format": "onnx", "quantize": 32,
        "nms": config.prediction["nms"], "imgsz": config.prediction["imgsz"], "demo": demo, "source_weights": str(Path(weights)),
        "runtime_parity_validated": bool(parity and parity["runtime_parity_validated"]),
        "parity_images": len(images) if parity else 0, "human_acceptance": "pendente",
    })
    return run
