"""Único ponto dependente de Ultralytics: simplifica revisão e troca de modelo."""

from pathlib import Path
import os
import platform
import re

from .config import ProjectConfig
from .io import file_hash


def load_detector(config: ProjectConfig, weights: str | None = None):
    """Carregue checkpoint local ou peso oficial YOLO26; sem fallback silencioso."""
    selected = weights or config.model
    if Path(selected).suffix.lower() != ".pt":
        raise ValueError("Este pipeline usa checkpoints PyTorch .pt. Para ONNX, valide o runtime de destino separadamente.")
    local = Path(selected).expanduser()
    if not local.is_absolute():
        local = config.root / local
    if not local.is_file():
        if re.fullmatch(r"yolo27[nslm]\.pt", selected, re.IGNORECASE):
            raise ValueError("YOLO27 ainda não foi publicado. Use YOLO26 ou um checkpoint local validado; consulte docs/pesquisa_ultralytics.md.")
        if not re.fullmatch(r"yolo26[nslmx]\.pt", selected):
            raise ValueError(f"Checkpoint local não encontrado: {local}. Downloads automáticos limitados aos pesos oficiais YOLO26 de detecção.")
        local = config.root / "models" / selected
        local.parent.mkdir(parents=True, exist_ok=True)
    YOLO = _ultralytics(config)
    model = YOLO(str(local), task="detect")
    if model.task != "detect":
        raise ValueError(f"Checkpoint é da tarefa {model.task!r}; esperado detect.")
    return model


def _ultralytics(config: ProjectConfig):
    """Restrinja ajustes/cache à pasta do projeto antes de importar a biblioteca."""
    config_directory = config.root / ".runtime"
    config_directory.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_directory)
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO
    from ultralytics import settings

    settings.update({"sync": False, "datasets_dir": str(config.root / "data"),
                     "weights_dir": str(config.root / "models"), "runs_dir": str(config.root / "runs")})
    return YOLO


def load_exported_detector(config: ProjectConfig, path: Path):
    """Carregue um ONNX local já exportado, somente para conferir paridade."""
    path = Path(path).resolve()
    if path.suffix.lower() != ".onnx" or not path.is_file():
        raise ValueError(f"Artefato ONNX local não encontrado: {path}.")
    return _ultralytics(config)(str(path), task="detect")


def check_domain_names(model, expected: list[str]) -> None:
    """Evite apresentar classes COCO como patologias: nomes precisam coincidir."""
    names = model.names
    actual = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    if actual != expected:
        raise ValueError(f"Checkpoint não corresponde às {len(expected)} classes ativas do projeto. Treine com o dataset revisado; para testar pesos gerais use predict --demo explicitamente.")


def runtime_info(model, config: ProjectConfig) -> dict:
    import torch
    import ultralytics

    checkpoint = Path(model.ckpt_path)
    return {
        "python": platform.python_version(), "ultralytics": ultralytics.__version__,
        "torch": torch.__version__, "device_requested": config.device,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_hash(checkpoint),
        "requested_future_model": config.requested_model,
    }
