"""Ponte com a Ultralytics: carrega o modelo YOLO e confere suas classes.

Função no projeto: é o único arquivo que importa a biblioteca Ultralytics. Treino,
predição, sugestões e exportação pedem o modelo por aqui.

O que faz:
- ``load_detector``: abre um checkpoint ``.pt`` local (ou baixa um YOLO26 oficial).
- ``_ultralytics``: aponta configurações e caches da biblioteca para dentro do projeto.
- ``load_exported_detector``: abre um ``.onnx`` exportado, só para conferir paridade.
- ``check_domain_names``: recusa pesos cujas classes não são as 13 do projeto.
- ``runtime_info``: versões de Python/torch/Ultralytics e hash do checkpoint.

Quando mexer: para trocar de família de modelo (outro YOLO) ou de biblioteca.
Para trocar só o tamanho do modelo (n/s/m/l/x), basta mudar ``model`` e
``pilot.model`` em ``configs/project.yaml``.
"""

from pathlib import Path
import os
import platform
import re

from .config import ProjectConfig
from .io import file_hash


def load_detector(config: ProjectConfig, weights: str | None = None):
    """Carregue checkpoint local ou peso oficial YOLO26; sem fallback silencioso."""
    # Só checkpoints PyTorch: o ONNX é conferido à parte (exporting.py).
    selected = weights or config.model
    if Path(selected).suffix.lower() != ".pt":
        raise ValueError("Este pipeline usa checkpoints PyTorch .pt. Para ONNX, valide o runtime de destino separadamente.")
    # Caminho relativo é resolvido a partir da pasta sistema_ia.
    local = Path(selected).expanduser()
    if not local.is_absolute():
        local = config.root / local
    # Sem arquivo local, só aceita baixar os pesos oficiais YOLO26 para models/.
    if not local.is_file():
        if re.fullmatch(r"yolo27[nslm]\.pt", selected, re.IGNORECASE):
            raise ValueError("YOLO27 ainda não foi publicado. Use YOLO26 ou um checkpoint local validado; consulte docs/pesquisa_ultralytics.md.")
        if not re.fullmatch(r"yolo26[nslmx]\.pt", selected):
            raise ValueError(f"Checkpoint local não encontrado: {local}. Downloads automáticos limitados aos pesos oficiais YOLO26 de detecção.")
        local = config.root / "models" / selected
        local.parent.mkdir(parents=True, exist_ok=True)
    # Abre o modelo pela Ultralytics e garante que é de detecção (caixas).
    YOLO = _ultralytics(config)
    model = YOLO(str(local), task="detect")
    if model.task != "detect":
        raise ValueError(f"Checkpoint é da tarefa {model.task!r}; esperado detect.")
    return model


def _ultralytics(config: ProjectConfig):
    """Restrinja ajustes/cache à pasta do projeto antes de importar a biblioteca."""
    # Configurações da Ultralytics em sistema_ia/.runtime; sem instalar pacotes sozinha.
    config_directory = config.root / ".runtime"
    config_directory.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_directory)
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO
    from ultralytics import settings

    # Datasets, pesos e execuções sempre dentro de sistema_ia; sem telemetria (sync).
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
    """Versões e hash do checkpoint gravados em execution.json (reprodutibilidade)."""
    import torch
    import ultralytics

    checkpoint = Path(model.ckpt_path)
    return {
        "python": platform.python_version(), "ultralytics": ultralytics.__version__,
        "torch": torch.__version__, "device_requested": config.device,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_hash(checkpoint),
        "requested_future_model": config.requested_model,
    }
