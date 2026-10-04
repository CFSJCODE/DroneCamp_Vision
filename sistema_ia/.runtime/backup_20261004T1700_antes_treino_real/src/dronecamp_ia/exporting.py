"""Exportação separada: criar um ONNX não certifica a qualidade do modelo."""

from pathlib import Path
import importlib.util
import shutil

from .backend import check_domain_names, load_detector, runtime_info
from .config import ProjectConfig, detection_names, load_taxonomy
from .io import file_hash, make_run_directory, write_json


def export_onnx(config: ProjectConfig, weights: str, demo: bool = False) -> Path:
    """Exporte ONNX FP32, batch 1 e tamanho fixo para iniciar testes de destino."""
    if importlib.util.find_spec("onnx") is None:
        raise ValueError("Dependência ONNX ausente. Instale o extra export documentado no README antes de exportar.")
    model = load_detector(config, weights)
    if not demo:
        check_domain_names(model, detection_names(load_taxonomy(config.taxonomy_path)))
    run = make_run_directory(config.root, "export_demo" if demo else "export")
    checkpoint = run / "model.pt"
    shutil.copy2(model.ckpt_path, checkpoint)
    # O exportador grava ao lado do checkpoint: uma cópia evita tocar o original.
    model = load_detector(config, str(checkpoint))
    output = model.export(format="onnx", device="cpu", imgsz=config.prediction["imgsz"],
                          batch=1, dynamic=False, simplify=False, nms=config.prediction["nms"], quantize=32)
    write_json(run / "export.json", {
        "runtime": runtime_info(model, config), "artifact": str(output),
        "sha256": file_hash(Path(output)), "format": "onnx", "quantize": 32,
        "nms": config.prediction["nms"], "demo": demo,
        "runtime_parity_validated": False, "human_acceptance": "pendente",
    })
    return run
