"""Configuração central: altere parâmetros no YAML, sem espalhá-los pelo código."""

from dataclasses import dataclass
from pathlib import Path
import json

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    model: str
    requested_model: str
    taxonomy_path: Path
    dataset_path: Path
    device: str
    prediction: dict
    training: dict


def load_config(path: Path | None = None) -> ProjectConfig:
    """Resolva caminhos no projeto e rejeite configurações de outra tarefa."""
    config_path = (path or PROJECT_ROOT / "configs/project.yaml").resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Configuração deve ser um mapa com schema_version: 1.")
    if raw.get("task") != "detect":
        raise ValueError("Este estágio implementa apenas detecção (task: detect).")
    # O arquivo de configuração fica em configs/; o seu pai define o projeto.
    root = config_path.parent.parent
    prediction = raw["prediction"]
    if not isinstance(prediction.get("nms"), bool):
        raise ValueError("prediction.nms deve ser booleano.")
    if not 0 < float(prediction["conf"]) <= 1:
        raise ValueError("prediction.conf deve estar no intervalo (0, 1].")
    if not 0 < float(prediction["iou"]) <= 1:
        raise ValueError("prediction.iou deve estar no intervalo (0, 1].")
    if int(prediction["imgsz"]) < 32:
        raise ValueError("prediction.imgsz deve ser pelo menos 32.")
    return ProjectConfig(
        root=root, model=str(raw["model"]), requested_model=str(raw["requested_model"]),
        taxonomy_path=(root / raw["taxonomy"]).resolve(),
        dataset_path=(root / raw["dataset"]).resolve(), device=str(raw["device"]),
        prediction=dict(prediction), training=dict(raw["training"]),
    )


def load_taxonomy(path: Path) -> dict:
    """IDs e slugs são contrato do dataset, checkpoint e saída."""
    taxonomy = json.loads(path.read_text(encoding="utf-8"))
    classes = taxonomy["classes"]
    if [item["id"] for item in classes] != list(range(len(classes))):
        raise ValueError("IDs da taxonomia devem ser consecutivos a partir de zero.")
    if len({item["slug"] for item in classes}) != len(classes):
        raise ValueError("Slugs duplicados na taxonomia.")
    return taxonomy


def detection_names(taxonomy: dict) -> list[str]:
    """O MVP utiliza só fase 1; ampliar classes exige nova versão e novo treino."""
    active = [item for item in taxonomy["classes"] if item["phase"] == 1]
    if not active or [item["id"] for item in active] != list(range(len(active))):
        raise ValueError("IDs das classes ativas devem ser consecutivos a partir de zero.")
    return [item["slug"] for item in active]
