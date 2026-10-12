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
- ``ensure_downloadable``: recusa baixar pesos não oficiais ou de família não publicada.
- ``distance_loss_columns``: colunas de perda de distância esperadas no ``results.csv``.
- ``model_info`` / ``format_model_info``: relatório do comando ``model-info``.
- ``ModelSpec``: o nome do peso decomposto (família, tamanho, tarefa...).

Quando mexer: quando a Ultralytics publicar o YOLO27 (confirmar os dados de
``FAMILIES[27]`` com a documentação oficial) ou para incluir outra família.

Quem chama:
- ``backend.load_detector`` → ``parse_model_name`` + ``ensure_downloadable``.
- ``backend.runtime_info`` → ``describe_checkpoint``.
- ``training.review_training_result`` → ``distance_loss_columns`` e
  ``GENERIC_DISTANCE_LOSS``.
- ``cli.py`` (comando ``model-info``) → ``model_info`` + ``format_model_info``.

O que lê e grava no disco: só lê. Consulta a pasta da Ultralytics instalada
(``ultralytics/cfg/models/<versão>``) e lista ``sistema_ia/models/*.pt``. Não
baixa nem grava nada.

Onde mexer (casos concretos):
- YOLO27 publicado: em ``FAMILIES[27]`` mude ``released`` para ``True``,
  confirme ``sizes``, ``distance_loss`` e ``min_ultralytics``; atualize a
  Ultralytics; rode ``model-info``; siga ``migration_checklist``.
- Família nova (ex.: YOLO28): entrada nova em ``FAMILIES`` (ver EXTENSAO:).
- Outra tarefa aceita no nome do peso (ex.: ``-xyz``): regex ``_OFFICIAL``.
- Etapa nova no processo de migração: lista em ``migration_checklist``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import importlib.util
import json
import re

# ---------------------------------------------------------------------------
# Tabela das famílias conhecidas e padrão do nome oficial dos pesos
# ---------------------------------------------------------------------------

# Nome oficial de peso: yolo<versão><tamanho>[-tarefa].pt (ex.: yolo26l.pt, yolo26l-seg.pt).
# Grupos: version = 1 ou 2 dígitos (26), size = uma letra n/s/l/m/x, task = sufixo opcional
# depois do hífen. IGNORECASE aceita "YOLO26L.PT".
# EXTENSAO: sufixo de tarefa novo publicado pela Ultralytics = acrescente na alternativa (seg|sem|...).
_OFFICIAL = re.compile(r"^yolo(?P<version>\d{1,2})(?P<size>[nslmx])(?:-(?P<task>seg|sem|depth|cls|pose|obb))?\.pt$",
                       re.IGNORECASE)

# O que o pipeline precisa saber de cada família. Dados do YOLO27 vêm da página oficial
# consultada em docs/pesquisa_ultralytics.md (prévia, sem pesos públicos): confirmar no lançamento.
# Chaves de cada família:
#   name           nome legível (usado em label e mensagens);
#   sizes          tamanhos com pesos oficiais (só esses contam como "official");
#   end_to_end     True = cabeça sem NMS externo (inferência com nms=False);
#   distance_loss  colunas de perda de distância aceitas no results.csv (sem prefixo train/ val/);
#   released       False = ainda sem pesos públicos (download recusado);
#   min_ultralytics  versão mínima da biblioteca (informativo, mostrado no model-info);
#   notes          texto livre para o model-info.
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
    # ATENCAO: ao publicar o YOLO27, mude released para True só depois de confirmar os dados
    # abaixo na documentação oficial; até lá o download é recusado de propósito.
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
# Basta uma delas existir em train/ e val/ para a auditoria do treino aceitar o results.csv.
GENERIC_DISTANCE_LOSS = ("l1_loss", "dfl_loss")


# ---------------------------------------------------------------------------
# Decomposição do nome do peso
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """Nome oficial decomposto; ``family`` é ``None`` para checkpoints fora do padrão.

    Atributos:
        name (str): nome do arquivo, sem pasta (ex.: ``"yolo26l.pt"``, ``"best.pt"``).
        family (int | None): versão do YOLO (26, 27...) ou ``None`` se o nome não
            segue o padrão oficial (ex.: ``best.pt`` treinado aqui).
        size (str | None): ``"n"``, ``"s"``, ``"m"``, ``"l"`` ou ``"x"``.
        task (str): ``"detect"`` ou o sufixo do nome (``"seg"``, ``"pose"``...).
        official (bool): nome oficial da Ultralytics, com tamanho publicado (pode ser baixado).
        released (bool): a família já tem pesos públicos.
        end_to_end (bool): inferência sem NMS externo.
        distance_loss (tuple[str, ...]): perdas de distância esperadas no results.csv.
    """
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
        """Nome legível do modelo, ex.: ``"YOLO26L"``.

        Retorna:
            str: ``name`` quando a família é desconhecida; senão o nome da
            família (de ``FAMILIES`` ou ``"YOLO<versão>"``) seguido do tamanho em maiúscula.
        """
        if self.family is None:
            return self.name
        return f"{FAMILIES.get(self.family, {}).get('name', f'YOLO{self.family}')}{(self.size or '').upper()}"


def parse_model_name(name: str) -> ModelSpec:
    """Decomponha ``yolo26l.pt``; caminhos de checkpoint locais viram família desconhecida.

    Parâmetros:
        name (str): nome ou caminho do peso (só o nome do arquivo é analisado).

    Retorna:
        ModelSpec com três casos possíveis:
        - nome fora do padrão (``best.pt``): ``family=None``, ``official=False``,
          ``released=True``, perdas genéricas;
        - padrão oficial de versão desconhecida (``yolo11n.pt``): ``official=True``,
          ``end_to_end=False``, perdas genéricas;
        - família de ``FAMILIES``: dados da tabela; ``official`` só se o tamanho
          estiver em ``sizes``.

    Chamada por: ``backend.load_detector``, ``describe_checkpoint``, ``model_info``.
    """
    base = Path(str(name)).name
    match = _OFFICIAL.match(base)
    # Caso 1: nome fora do padrão oficial (checkpoint treinado aqui, ex.: best.pt).
    if not match:
        return ModelSpec(name=base, family=None, size=None, task="detect", official=False, released=True,
                         end_to_end=True, distance_loss=GENERIC_DISTANCE_LOSS)
    version = int(match.group("version"))
    size = match.group("size").lower()
    family = FAMILIES.get(version)
    # Sem sufixo de tarefa no nome = detecção.
    task = match.group("task") or "detect"
    if family is None:
        # Caso 2 (ver docstring).
        # Versão que este projeto não conhece (ex.: yolo11): nome oficial, sem garantias.
        return ModelSpec(name=base, family=version, size=size, task=task, official=True, released=True,
                         end_to_end=False, distance_loss=GENERIC_DISTANCE_LOSS)
    # Caso 3: família conhecida; os dados vêm de FAMILIES.
    return ModelSpec(name=base, family=version, size=size, task=task, official=size in family["sizes"],
                     released=bool(family["released"]), end_to_end=bool(family["end_to_end"]),
                     distance_loss=tuple(family["distance_loss"]))


# ---------------------------------------------------------------------------
# Suporte na Ultralytics instalada e permissão de download
# ---------------------------------------------------------------------------

def family_support(spec: ModelSpec) -> dict:
    """Confira na Ultralytics instalada (sem baixar) se a família do peso já existe.

    Dois sinais: a pasta de arquiteturas ``ultralytics/cfg/models/<versão>`` e o
    nome do peso na lista de ativos oficiais (``GITHUB_ASSETS_NAMES``). Quando a
    biblioteca publicar o YOLO27, os dois passam a ser verdadeiros sem mudar
    código aqui; só ``FAMILIES[27]["released"]`` precisa ser confirmado.

    Parâmetros:
        spec (ModelSpec): resultado de ``parse_model_name``.

    Retorna:
        dict com ``model``, ``family``, ``ultralytics`` (versão instalada ou None),
        ``architecture_available`` (bool ou None), ``official_weights_listed``
        (bool ou None), ``downloadable`` (bool) e, quando há explicação,
        ``reason`` (texto).

    Chamada por: ``ensure_downloadable``, ``migration_checklist``, ``_spec_report``.
    """
    # Começa com tudo "desconhecido" (None) e downloadable False.
    report = {"model": spec.name, "family": spec.family, "ultralytics": None,
              "architecture_available": None, "official_weights_listed": None, "downloadable": False}
    # find_spec só verifica se o pacote existe, sem importá-lo.
    if importlib.util.find_spec("ultralytics") is None:
        report["reason"] = "Ultralytics não instalada neste ambiente."
        return report
    import ultralytics
    from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

    report["ultralytics"] = ultralytics.__version__
    # Sinal 1: a biblioteca tem os YAMLs de arquitetura desta versão (cfg/models/26, cfg/models/27...).
    if spec.family is not None:
        models_dir = Path(ultralytics.__file__).resolve().parent / "cfg" / "models" / str(spec.family)
        report["architecture_available"] = models_dir.is_dir() and any(models_dir.glob("*.yaml"))
    # Sinal 2: o nome do peso está na lista oficial de downloads (comparação sem maiúsculas).
    report["official_weights_listed"] = spec.official and spec.name.lower() in {name.lower() for name in GITHUB_ASSETS_NAMES}
    report["downloadable"] = bool(report["official_weights_listed"])
    # Explicação para família conhecida ainda não publicada.
    if spec.family in FAMILIES and not FAMILIES[spec.family]["released"] and not report["downloadable"]:
        report["reason"] = (f"{FAMILIES[spec.family]['name']} ainda não foi publicado na Ultralytics "
                            f"{ultralytics.__version__} instalada.")
    return report


# ATENCAO: esta é a única barreira contra baixar pesos inesperados (ex.: um nome digitado
# errado que por acaso é oficial de outra tarefa). Não relaxe as regras sem motivo forte.
def ensure_downloadable(spec: ModelSpec, local: Path) -> None:
    """Sem arquivo local, só nomes oficiais de uma família publicada podem ser baixados.

    Parâmetros:
        spec (ModelSpec): o nome pedido, decomposto.
        local (Path): caminho local que não existe (só aparece na mensagem de erro).

    Retorna:
        None quando o download é permitido.

    Erros:
        ValueError: nome não oficial ("Checkpoint local não encontrado..."); tarefa
            diferente de detect; família ainda não publicada e não listada para
            download na Ultralytics instalada.

    Chamada por: ``backend.load_detector``.
    """
    # Regra 1: só nomes oficiais (yolo<versão><tamanho>.pt com tamanho publicado).
    if not spec.official:
        raise ValueError(f"Checkpoint local não encontrado: {local}. Downloads automáticos limitados aos pesos "
                         f"oficiais de detecção das famílias conhecidas ({', '.join(f['name'] for f in FAMILIES.values())}).")
    # Regra 2: só pesos de detecção (o pipeline inteiro trabalha com caixas).
    if spec.task != "detect":
        raise ValueError(f"{spec.name} é um peso da tarefa {spec.task!r}; o pipeline usa detecção (caixas).")
    # Regra 3: família ainda não publicada só passa se a Ultralytics instalada já listar o peso.
    if not spec.released:
        support = family_support(spec)
        if not support["downloadable"]:
            family = FAMILIES[spec.family]["name"]
            version = support.get("ultralytics") or "não instalada"
            raise ValueError(f"{family} ainda não foi publicado (Ultralytics {version}). Use YOLO26 ou um checkpoint "
                             f"local validado; consulte docs/modelos_e_ia_gratuita.md e rode `model-info`.")


# ---------------------------------------------------------------------------
# Procedência do checkpoint e colunas esperadas no results.csv
# ---------------------------------------------------------------------------

def describe_checkpoint(path: Path | str, train_args: dict | None = None) -> dict:
    """Família/versão para registrar junto do hash do checkpoint (procedência).

    Parâmetros:
        path (Path | str): caminho do peso (só o nome do arquivo é analisado).
        train_args (dict | None): ``ckpt["train_args"]`` do checkpoint, se houver;
            de lá sai o peso de partida (``model`` ou ``pretrained``).

    Retorna:
        dict com ``model`` (nome do arquivo), ``family``, ``label``, ``size``,
        ``end_to_end`` e ``base_weights`` (nome oficial do peso de partida ou None).
        Para ``best.pt`` treinado a partir de ``yolo26l.pt``, a família vem do
        peso de partida (26, ``"YOLO26L"``).

    Chamada por: ``backend.runtime_info`` e ``distance_loss_columns``.
    """
    spec = parse_model_name(Path(path).name)
    origin = None
    # Checkpoints treinados aqui guardam em train_args o peso de partida (ex.: yolo26l.pt).
    base = (train_args or {}).get("model") or (train_args or {}).get("pretrained")
    if isinstance(base, str) and _OFFICIAL.match(Path(base).name):
        origin = parse_model_name(Path(base).name)
        # Nome do arquivo fora do padrão (best.pt): herda a família do peso de partida.
        if spec.family is None:
            spec = origin
    # EXTENSAO: dado novo de procedência no execution.json (ex.: data do peso base) entra neste dict.
    return {"model": Path(path).name, "family": spec.family, "label": spec.label, "size": spec.size,
            "end_to_end": spec.end_to_end, "base_weights": origin.name if origin else None}


def distance_loss_columns(model=None, weights: str | Path | None = None) -> tuple[str, ...]:
    """Colunas de perda de distância que ``results.csv`` deve ter para esta família.

    Parâmetros:
        model: modelo Ultralytics carregado (usa ``ckpt["train_args"]`` e ``ckpt_path``); opcional.
        weights (str | Path | None): caminho do peso, quando não há ``model``
            (tem prioridade sobre ``model.ckpt_path``).

    Retorna:
        tuple[str, ...]: nomes sem prefixo, ex.: ``("l1_loss",)`` no YOLO26;
        ``GENERIC_DISTANCE_LOSS`` (``l1_loss`` ou ``dfl_loss``) se a família for desconhecida.

    Chamada por: ``training.review_training_result``.
    """
    # train_args do checkpoint (vazio sem modelo ou sem ckpt).
    train_args = (getattr(model, "ckpt", None) or {}).get("train_args", {}) if model is not None else {}
    name = weights or getattr(model, "ckpt_path", None) or ""
    info = describe_checkpoint(name, train_args)
    family = FAMILIES.get(info["family"]) if info["family"] is not None else None
    return tuple(family["distance_loss"]) if family else GENERIC_DISTANCE_LOSS


# ---------------------------------------------------------------------------
# Migração de versão e relatório do comando model-info
# ---------------------------------------------------------------------------

def migration_checklist(current: ModelSpec, target: ModelSpec) -> list[dict]:
    """Etapas para trocar de versão sem perder o que já foi medido (docs/modelos_e_ia_gratuita.md).

    Parâmetros:
        current (ModelSpec): modelo configurado hoje (``model:``).
        target (ModelSpec): modelo para o qual se quer migrar.

    Retorna:
        list[dict]: uma entrada por etapa, com ``step`` (nome), ``done`` (bool;
        só "pesos publicados" é conferido de verdade, "modelo anterior
        preservado" é sempre True e as demais ficam False para o humano conferir)
        e ``how`` (comando ou instrução).

    Chamada por: ``model_info``.
    """
    target_support = family_support(target)
    # EXTENSAO: etapa nova obrigatória da migração = dict novo nesta lista, na posição da ordem de execução.
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
    """Junta os campos de ``ModelSpec``, o ``label`` e o suporte instalado num só dict.

    Parâmetros:
        spec (ModelSpec): o peso analisado.

    Retorna:
        dict: campos do dataclass (``asdict``) + ``label`` + ``support``
        (resultado de ``family_support``).
    """
    return {**asdict(spec), "label": spec.label, "support": family_support(spec)}


# ENTRADA: comando model-info (cli.py).
def model_info(config, model: str | None = None) -> dict:
    """Relatório do comando ``model-info``: configurado, alvo, suporte instalado e migração.

    ``model`` (opção ``--model``) é só o peso inspecionado: o configurado continua
    sendo ``config.model`` e a lista de migração continua indo dele para o alvo.

    Parâmetros:
        config (ProjectConfig): usa ``model``, ``requested_model`` e ``root``.
        model (str | None): nome ou caminho de um peso extra a inspecionar.

    Retorna:
        dict com ``configured`` e ``requested`` (``_spec_report``), ``families``
        (``FAMILIES`` com tuplas viradas listas, para o JSON), ``local_weights``
        (nomes dos ``.pt`` em ``models/``) e, quando aplicável, ``migration``,
        ``inspected`` e ``migration_to_inspected``.
    """
    current = parse_model_name(config.model)
    requested = parse_model_name(config.requested_model)
    # FAMILIES vai para o relatório com tuplas convertidas em listas (formato JSON).
    report = {
        "configured": _spec_report(current), "requested": _spec_report(requested),
        "families": {version: {key: (list(value) if isinstance(value, tuple) else value) for key, value in family.items()}
                     for version, family in FAMILIES.items()},
        "local_weights": sorted(path.name for path in (config.root / "models").glob("*.pt")) if (config.root / "models").is_dir() else [],
    }
    # Só há migração a mostrar quando o alvo é diferente do configurado.
    if requested.name.lower() != current.name.lower():
        report["migration"] = migration_checklist(current, requested)
    # Peso extra pedido em --model: relatório próprio e, se for um terceiro peso, a migração até ele.
    if model:
        inspected = parse_model_name(model)
        report["inspected"] = _spec_report(inspected)
        if inspected.name.lower() not in (current.name.lower(), requested.name.lower()):
            report["migration_to_inspected"] = migration_checklist(current, inspected)
    return report


def format_model_info(report: dict) -> str:
    """Formata o relatório de ``model_info`` como JSON legível para o terminal.

    Parâmetros:
        report (dict): resultado de ``model_info``.

    Retorna:
        str: JSON indentado (2 espaços), com acentos preservados.
    """
    return json.dumps(report, ensure_ascii=False, indent=2)
