"""Família e versão do detector: YOLO26 hoje, YOLO27 quando for publicado.

Função no projeto: concentra tudo o que depende de "qual YOLO" está em uso, para
que trocar de versão seja uma mudança de ``configs/project.yaml`` (``model:``),
não de código. ``backend.py`` pergunta aqui se um nome de peso pode ser baixado,
``training.py`` pergunta quais colunas de perda esperar e ``exporting.py`` como
exportar. Nada aqui carrega a Ultralytics por padrão.

O que faz:
- ``parse_model_name``: ``yolo26l.pt`` → família 26, tamanho l, tarefa detect.
- ``FAMILIES``: o que muda entre YOLO26 e YOLO27 (cabeça sem NMS, perdas,
  tamanhos publicados, situação do lançamento).
- ``family_support``: confere na Ultralytics instalada se a família já existe
  (arquitetura e pesos oficiais), sem baixar nada.
- ``describe_checkpoint``: família/versão registrada em ``execution.json`` e
  ``summary.json`` (procedência do modelo).
- ``migration_checklist``: etapas obrigatórias para adotar uma versão nova
  (mesmo teste, ``compare-models``, modelo anterior preservado).

Quando mexer: quando a Ultralytics publicar o YOLO27 (confirmar os dados de
``FAMILIES[27]`` com a documentação oficial) ou para incluir outra família.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import importlib.util
import json
import re

# Nome oficial de peso: yolo<versão><tamanho>[-tarefa].pt (ex.: yolo26l.pt, yolo26l-seg.pt).
_OFFICIAL = re.compile(r"^yolo(?P<version>\d{1,2})(?P<size>[nslmx])(?:-(?P<task>seg|sem|depth|cls|pose|obb))?\.pt$",
                       re.IGNORECASE)

# O que o pipeline precisa saber de cada família. Dados do YOLO27 vêm da página oficial
# consultada em docs/pesquisa_ultralytics.md (prévia, sem pesos públicos): confirmar no lançamento.
# EXTENSAO: versão nova do YOLO = entrada nova aqui; depois troque ``model:`` em
# configs/project.yaml e rode model-info. Adoção só depois do compare-models.
FAMILIES: dict[int, dict] = {
    26: {
        "name": "YOLO26",
        "sizes": ("n", "s", "m", "l", "x"),
        "end_to_end": True,                      # cabeça um-para-um: inferência com nms=False
        "distance_loss": ("l1_loss",),           # coluna de results.csv (sem DFL no YOLO26)
        "released": True,
        "min_ultralytics": "8.4.0",
        "notes": "Base atual do projeto; treino na CPU, inferência ONNX na GPU (DirectML).",
    },
    27: {
        "name": "YOLO27",
        "sizes": ("n", "s", "m", "l"),
        "end_to_end": True,                      # decodificador por consultas, sem NMS (prévia oficial)
        "distance_loss": ("l1_loss", "dfl_loss"),  # desconhecida até o lançamento: aceita as duas
        "released": False,
        "min_ultralytics": None,                 # definida pela versão que publicar os pesos
        "notes": "Alvo futuro (requested_model). Sem pesos públicos na consulta de 03/10/2026.",
    },
}

# Colunas de perda aceitas quando a família é desconhecida (checkpoint de terceiros).
GENERIC_DISTANCE_LOSS = ("l1_loss", "dfl_loss")


@dataclass(frozen=True)
class ModelSpec:
    """Nome oficial decomposto; ``family`` é ``None`` para checkpoints fora do padrão."""
    name: str
    family: int | None
    size: str | None
    task: str
    official: bool          # nome oficial da Ultralytics (pode ser baixado)
    released: bool          # a família já tem pesos públicos
    end_to_end: bool        # inferência sem NMS externo
    distance_loss: tuple[str, ...]

    @property
    def label(self) -> str:
        if self.family is None:
            return self.name
        return f"{FAMILIES.get(self.family, {}).get('name', f'YOLO{self.family}')}{(self.size or '').upper()}"


def parse_model_name(name: str) -> ModelSpec:
    """Decomponha ``yolo26l.pt``; caminhos de checkpoint locais viram família desconhecida."""
    base = Path(str(name)).name
    match = _OFFICIAL.match(base)
    if not match:
        return ModelSpec(name=base, family=None, size=None, task="detect", official=False, released=True,
                         end_to_end=True, distance_loss=GENERIC_DISTANCE_LOSS)
    version = int(match.group("version"))
    size = match.group("size").lower()
    family = FAMILIES.get(version)
    task = match.group("task") or "detect"
    if family is None:
        # Versão que este projeto não conhece (ex.: yolo11): nome oficial, sem garantias.
        return ModelSpec(name=base, family=version, size=size, task=task, official=True, released=True,
                         end_to_end=False, distance_loss=GENERIC_DISTANCE_LOSS)
    return ModelSpec(name=base, family=version, size=size, task=task, official=size in family["sizes"],
                     released=bool(family["released"]), end_to_end=bool(family["end_to_end"]),
                     distance_loss=tuple(family["distance_loss"]))


def family_support(spec: ModelSpec) -> dict:
    """Confira na Ultralytics instalada (sem baixar) se a família do peso já existe.

    Dois sinais: a pasta de arquiteturas ``ultralytics/cfg/models/<versão>`` e o
    nome do peso na lista de ativos oficiais (``GITHUB_ASSETS_NAMES``). Quando a
    biblioteca publicar o YOLO27, os dois passam a ser verdadeiros sem mudar
    código aqui; só ``FAMILIES[27]["released"]`` precisa ser confirmado.
    """
    report = {"model": spec.name, "family": spec.family, "ultralytics": None,
              "architecture_available": None, "official_weights_listed": None, "downloadable": False}
    if importlib.util.find_spec("ultralytics") is None:
        report["reason"] = "Ultralytics não instalada neste ambiente."
        return report
    import ultralytics
    from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

    report["ultralytics"] = ultralytics.__version__
    if spec.family is not None:
        models_dir = Path(ultralytics.__file__).resolve().parent / "cfg" / "models" / str(spec.family)
        report["architecture_available"] = models_dir.is_dir() and any(models_dir.glob("*.yaml"))
    report["official_weights_listed"] = spec.official and spec.name.lower() in {name.lower() for name in GITHUB_ASSETS_NAMES}
    report["downloadable"] = bool(report["official_weights_listed"])
    if spec.family in FAMILIES and not FAMILIES[spec.family]["released"] and not report["downloadable"]:
        report["reason"] = (f"{FAMILIES[spec.family]['name']} ainda não foi publicado na Ultralytics "
                            f"{ultralytics.__version__} instalada.")
    return report


def ensure_downloadable(spec: ModelSpec, local: Path) -> None:
    """Sem arquivo local, só nomes oficiais de uma família publicada podem ser baixados."""
    if not spec.official:
        raise ValueError(f"Checkpoint local não encontrado: {local}. Downloads automáticos limitados aos pesos "
                         f"oficiais de detecção das famílias conhecidas ({', '.join(f['name'] for f in FAMILIES.values())}).")
    if spec.task != "detect":
        raise ValueError(f"{spec.name} é um peso da tarefa {spec.task!r}; o pipeline usa detecção (caixas).")
    if not spec.released:
        support = family_support(spec)
        if not support["downloadable"]:
            family = FAMILIES[spec.family]["name"]
            version = support.get("ultralytics") or "não instalada"
            raise ValueError(f"{family} ainda não foi publicado (Ultralytics {version}). Use YOLO26 ou um checkpoint "
                             f"local validado; consulte docs/modelos_e_ia_gratuita.md e rode `model-info`.")


def describe_checkpoint(path: Path | str, train_args: dict | None = None) -> dict:
    """Família/versão para registrar junto do hash do checkpoint (procedência)."""
    spec = parse_model_name(Path(path).name)
    origin = None
    # Checkpoints treinados aqui guardam em train_args o peso de partida (ex.: yolo26l.pt).
    base = (train_args or {}).get("model") or (train_args or {}).get("pretrained")
    if isinstance(base, str) and _OFFICIAL.match(Path(base).name):
        origin = parse_model_name(Path(base).name)
        if spec.family is None:
            spec = origin
    return {"model": Path(path).name, "family": spec.family, "label": spec.label, "size": spec.size,
            "end_to_end": spec.end_to_end, "base_weights": origin.name if origin else None}


def distance_loss_columns(model=None, weights: str | Path | None = None) -> tuple[str, ...]:
    """Colunas de perda de distância que ``results.csv`` deve ter para esta família."""
    train_args = (getattr(model, "ckpt", None) or {}).get("train_args", {}) if model is not None else {}
    name = weights or getattr(model, "ckpt_path", None) or ""
    info = describe_checkpoint(name, train_args)
    family = FAMILIES.get(info["family"]) if info["family"] is not None else None
    return tuple(family["distance_loss"]) if family else GENERIC_DISTANCE_LOSS


def migration_checklist(current: ModelSpec, target: ModelSpec) -> list[dict]:
    """Etapas para trocar de versão sem perder o que já foi medido (docs/modelos_e_ia_gratuita.md)."""
    target_support = family_support(target)
    return [
        {"step": "pesos publicados", "done": bool(target_support["downloadable"]),
         "how": f"`model-info --model {target.name}` deve mostrar downloadable: true (atualize a Ultralytics quando sair)."},
        {"step": "treino piloto com o mesmo dataset", "done": False,
         "how": f"`retrain-cycle --weights {target.name} ...` ou `train-pilot --weights {target.name}` no mesmo data/pilot/<versão>."},
        {"step": "exportação ONNX com paridade", "done": False,
         "how": "`export --weights runs/<treino>/fit/weights/best.pt`; parity.json precisa de runtime_parity_validated: true."},
        {"step": "comparação por classe com o modelo atual", "done": False,
         "how": f"`compare-models --baseline <{current.label} atual> --candidate <{target.label}>` (adopt: true + regressões lidas)."},
        {"step": "modelo anterior preservado", "done": True,
         "how": "Pesos ficam em runs/; nenhum comando apaga ou sobrescreve o piloto anterior."},
        {"step": "configuração", "done": False,
         "how": f"Só depois do gate: `model: {target.name}` e `pilot.model: {target.name}` em configs/project.yaml."},
    ]


def _spec_report(spec: ModelSpec) -> dict:
    return {**asdict(spec), "label": spec.label, "support": family_support(spec)}


def model_info(config, model: str | None = None) -> dict:
    """Relatório do comando ``model-info``: configurado, alvo, suporte instalado e migração.

    ``model`` (opção ``--model``) é só o peso inspecionado: o configurado continua
    sendo ``config.model`` e a lista de migração continua indo dele para o alvo.
    """
    current = parse_model_name(config.model)
    requested = parse_model_name(config.requested_model)
    report = {
        "configured": _spec_report(current), "requested": _spec_report(requested),
        "families": {version: {key: (list(value) if isinstance(value, tuple) else value) for key, value in family.items()}
                     for version, family in FAMILIES.items()},
        "local_weights": sorted(path.name for path in (config.root / "models").glob("*.pt")) if (config.root / "models").is_dir() else [],
    }
    if requested.name.lower() != current.name.lower():
        report["migration"] = migration_checklist(current, requested)
    if model:
        inspected = parse_model_name(model)
        report["inspected"] = _spec_report(inspected)
        if inspected.name.lower() not in (current.name.lower(), requested.name.lower()):
            report["migration_to_inspected"] = migration_checklist(current, inspected)
    return report


def format_model_info(report: dict) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2)
