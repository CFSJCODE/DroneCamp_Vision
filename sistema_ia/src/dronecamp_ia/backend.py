"""Ponte com a Ultralytics: carrega o modelo YOLO e confere suas classes.

Função no projeto: é o único arquivo que importa a biblioteca Ultralytics. Treino,
predição, sugestões e exportação pedem o modelo por aqui.

O que faz:
- ``load_detector``: abre um checkpoint ``.pt`` local (ou baixa um YOLO26 oficial).
- ``_ultralytics``: aponta configurações e caches da biblioteca para dentro do projeto.
- ``load_exported_detector``: abre um ``.onnx`` exportado, só para conferir paridade.
- ``load_inference_model``: abre ``.pt`` ou ``.onnx`` para sugerir/avaliar; o ONNX
  roda por padrão na GPU via DirectML (gancho em ``ONNXBackend.load_model``).
- ``check_domain_names``: recusa pesos cujas classes não são as 13 do projeto.
- ``runtime_info``: versões de Python/torch/Ultralytics e hash do checkpoint.

Quando mexer: para trocar de família de modelo (outro YOLO) ou de biblioteca.
Para trocar só o tamanho do modelo (n/s/m/l/x), basta mudar ``model`` e
``pilot.model`` em ``configs/project.yaml``.
"""

from pathlib import Path
import json
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


# ---------------------------------------------------------------------------
# Inferência na GPU (DirectML): ONNX exportado + ONNX Runtime com DmlExecutionProvider.
# A Ultralytics só cria sessões ONNX com CPU/CUDA/CoreML e recria a sessão sempre
# que o predictor é refeito; por isso o gancho fica em ONNXBackend.load_model.
# Medido na APU Ryzen 5 4600G (Vega): 115 ms por inferência 640×640 no DirectML,
# 356 ms no ONNX Runtime CPU e 535 ms no PyTorch CPU; caixas iguais às do .pt.
# ---------------------------------------------------------------------------

_DIRECTML_WEIGHTS: set[str] = set()
ONNX_PROVIDERS = ("directml", "cpu")


def _directml_session(weight: str):
    """Sessão DirectML; falha alto se o provedor não estiver ativo (nada de CPU silenciosa)."""
    import onnxruntime

    if "DmlExecutionProvider" not in onnxruntime.get_available_providers():
        raise RuntimeError("DirectML indisponível: instale onnxruntime-directml (não o onnxruntime comum) "
                           "ou use --onnx-provider cpu.")
    options = onnxruntime.SessionOptions()
    # Exigências do DirectML: sem padrão de memória e execução sequencial.
    options.enable_mem_pattern = False
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session = onnxruntime.InferenceSession(weight, options, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    if session.get_providers()[0] != "DmlExecutionProvider":
        raise RuntimeError(f"ONNX Runtime não ativou o DirectML para {weight}.")
    return session


def _install_directml_hook() -> None:
    from ultralytics.nn.backends import onnx as onnx_backend

    if getattr(onnx_backend.ONNXBackend.load_model, "_dronecamp_directml", False):
        return
    original = onnx_backend.ONNXBackend.load_model

    def load_model(self, weight):
        original(self, weight)
        if str(Path(weight).resolve()) in _DIRECTML_WEIGHTS:
            self.session = _directml_session(str(weight))
            # A linha "Using ONNX Runtime ... CPUExecutionProvider" acima é da Ultralytics, antes da troca.
            onnx_backend.LOGGER.info(f"DroneCamp: sessão ONNX trocada para {self.session.get_providers()[0]} (GPU).")

    load_model._dronecamp_directml = True
    onnx_backend.ONNXBackend.load_model = load_model


def onnx_source_checkpoint(config: ProjectConfig, onnx_path: Path) -> Path:
    """O .pt que gerou o ONNX (export.json), conferido pelo SHA-256 gravado na exportação."""
    manifest = Path(onnx_path).resolve().parent / "export.json"
    try:
        export = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"ONNX sem export.json ao lado ({manifest}); exporte com o comando export.") from error
    source = Path(str(export.get("source_weights", "")).replace("\\", "/"))
    source = source if source.is_absolute() else config.root / source
    if not source.is_file() or file_hash(source) != export.get("runtime", {}).get("checkpoint_sha256"):
        raise ValueError(f"O .pt de origem do ONNX não confere com export.json: {source}")
    if file_hash(Path(onnx_path)) != export.get("sha256"):
        raise ValueError(f"O ONNX mudou desde a exportação: {onnx_path}")
    return source.resolve()


def load_inference_model(config: ProjectConfig, weights: str, onnx_provider: str = "directml"):
    """Abre .pt (PyTorch) ou .onnx (ONNX Runtime; DirectML por padrão) para sugerir e avaliar.

    O modelo devolvido ganha ``dronecamp_source`` (o .pt de origem, para identificar
    piloto e fotos de treino) e ``dronecamp_runtime`` (registrado nas saídas).
    """
    path = Path(weights).expanduser()
    if path.suffix.lower() != ".onnx":
        model = load_detector(config, weights)
        model.dronecamp_source = Path(model.ckpt_path).resolve()
        model.dronecamp_runtime = "pytorch-cpu"
        return model
    if onnx_provider not in ONNX_PROVIDERS:
        raise ValueError(f"onnx_provider deve ser um de {ONNX_PROVIDERS}.")
    path = (path if path.is_absolute() else config.root / path).resolve()
    source = onnx_source_checkpoint(config, path)
    model = load_exported_detector(config, path)
    if onnx_provider == "directml":
        _directml_session(str(path))  # falha cedo, antes de qualquer foto
        _install_directml_hook()
        _DIRECTML_WEIGHTS.add(str(path))
    model.dronecamp_source = source
    model.dronecamp_onnx = path
    model.dronecamp_runtime = f"onnx-{onnx_provider}"
    return model


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
