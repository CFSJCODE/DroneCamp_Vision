"""Configuração central: lê configs/project.yaml e a taxonomia de classes.

Função no projeto: todo comando começa aqui. Os parâmetros ficam no YAML, não
espalhados pelo código.

O que faz:
- ``ProjectConfig``: os valores lidos (modelo, dispositivo, predição, treino, piloto).
- ``load_config``: lê e valida ``configs/project.yaml``.
- ``load_taxonomy``: lê ``configs/taxonomy.json`` (IDs e nomes das classes).
- ``detection_names``: lista das classes ativas (fase 1), na ordem dos IDs.

Quando mexer: quase nunca. Para mudar valores, edite ``configs/project.yaml``.
Mexa aqui só para criar uma seção nova no YAML ou validar um parâmetro novo.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json

import yaml

# Pasta sistema_ia (dois níveis acima de src/dronecamp_ia/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectConfig:
    """Valores do project.yaml já validados; caminhos absolutos a partir de root."""
    root: Path               # pasta sistema_ia
    model: str               # peso base (ex.: yolo26l.pt)
    requested_model: str     # modelo desejado no futuro (só registrado)
    taxonomy_path: Path      # configs/taxonomy.json
    dataset_path: Path       # configs/dataset.yaml (produção)
    device: str              # "cpu" (Ultralytics só treina em CUDA/MPS ou CPU)
    prediction: dict         # imgsz, conf, iou, nms, max_det da inferência
    training: dict           # hiperparâmetros do treino de produção
    # Piloto: treino exploratório com uma só edificação, nunca aprovado para uso.
    pilot: dict = field(default_factory=dict)


def load_config(path: Path | None = None) -> ProjectConfig:
    """Resolva caminhos no projeto e rejeite configurações de outra tarefa."""
    # Usa configs/project.yaml do projeto, ou o arquivo passado em --config.
    config_path = (path or PROJECT_ROOT / "configs/project.yaml").resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Configuração deve ser um mapa com schema_version: 1.")
    if raw.get("task") != "detect":
        raise ValueError("Este estágio implementa apenas detecção (task: detect).")
    # O arquivo de configuração fica em configs/; o seu pai define o projeto.
    root = config_path.parent.parent
    # Limites básicos dos parâmetros de inferência.
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
        pilot=dict(raw.get("pilot") or {}),
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
