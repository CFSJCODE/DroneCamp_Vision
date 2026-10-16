"""Adaptadores que põem YOLO26, RF-DETR e D-FINE atrás de uma mesma interface de treino.

Função no projeto: o aprendizado mútuo (``mutual_learning.py``) precisa, a cada
lote, de duas coisas de cada detector: a perda contra as anotações (calculada
pela própria biblioteca do modelo) e as previsões cruas, com gradiente, num
formato comum para comparar com as previsões dos outros. As três bibliotecas
falam formatos diferentes:

| Família | Biblioteca | Entrada | Saída crua usada |
| --- | --- | --- | --- |
| ``yolo26`` | Ultralytics 8.4 | imagem 0-1 | ``one2many`` e ``one2one``: caixas por âncora (8400 em 640) + logits sigmoides |
| ``rfdetr`` | ``rfdetr`` (Roboflow) | imagem normalizada ImageNet | 300 queries do 1.º grupo: ``pred_boxes`` cxcywh 0-1 + ``pred_logits`` |
| ``dfine`` | ``transformers`` (``DFineForObjectDetection``) | imagem 0-1 | 300 queries (sem as de denoising): ``pred_boxes`` + ``logits`` |

O formato comum é ``RawPredictions``: caixas ``cxcywh`` normalizadas (0-1) e
logits por classe ``[B, Q, K]``. As três cabeças usam sigmoide por classe
(BCE/varifocal no YOLO, focal/VFL nos DETRs), então ``sigmoid(logit)`` é uma
probabilidade de Bernoulli por classe nas três; é isso que torna a consistência
comparável entre elas sem softmax.

Conteúdo do arquivo:
- ``RawPredictions``: caixas + logits em formato comum.
- ``DetectorAdapter``: interface (``forward_train``, ``predict_raw``, ``save``...).
- ``YOLO26Adapter``, ``RFDETRAdapter``, ``DFINEAdapter``: uma por família.
- ``FAMILIES`` e ``build_adapter(spec, names, imgsz)``: fábrica a partir do JSON.
- ``dependency_report()``: quais bibliotecas estão instaladas (para ``dml-plan``).

Quem chama: ``mutual_learning.py`` (treino, avaliação, cache de pares),
``dml_lan.py`` (trabalhador da rede local) e os testes ``tests/test_mutual_learning.py``.

O que lê e grava no disco:
- Lê os pesos iniciais (``weights`` do spec): ``.pt`` da Ultralytics, ``.pth`` do
  RF-DETR ou pasta/ID do Hugging Face do D-FINE. Sem ``weights`` cada biblioteca
  baixa os pesos COCO oficiais na primeira vez (precisa de internet).
- Grava (``save``): ``yolo26`` → ``.pt`` que a Ultralytics abre (``YOLO(caminho)``),
  então ``compare-models``/``suggest`` funcionam com ele; ``rfdetr`` → ``.pth`` com
  ``model`` (state_dict) + ``model_config``; ``dfine`` → pasta ``save_pretrained``.

Onde mexer:
- Nova família (ex.: YOLO27, RT-DETRv3): crie uma subclasse de ``DetectorAdapter``
  e registre em ``FAMILIES``. Veja o EXTENSAO em ``FAMILIES``.
- Variante de tamanho: ``variant`` no ``configs/dml.json`` (``yolo26n..x``,
  RF-DETR ``nano..large``, D-FINE ``ustc-community/dfine-<n|s|m|l|x>-coco``).

ATENCAO: as bibliotecas pesadas (torch, ultralytics, rfdetr, transformers) são
importadas dentro das classes; importar este módulo não exige nenhuma delas.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

# Média e desvio ImageNet: normalização esperada pelo backbone DINOv2 do RF-DETR.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class RawPredictions:
    """Previsões cruas de um detector num lote, no formato comum.

    Atributos:
        boxes: tensor ``[B, Q, 4]`` em ``cx, cy, w, h`` normalizados (0-1), com gradiente.
        logits: tensor ``[B, Q, K]``; ``sigmoid(logits)`` = probabilidade por classe.
        head: nome da cabeça (ex.: ``"one2one"``), só para o relatório.
    """

    boxes: object
    logits: object
    head: str = "main"

    def detach(self) -> "RawPredictions":
        """Cópia sem gradiente (alvo de consistência dos pares)."""
        return RawPredictions(self.boxes.detach(), self.logits.detach(), self.head)


class DetectorAdapter:
    """Interface comum dos três detectores.

    Atributos que toda subclasse define:
        name: nome do modelo no experimento (chave do ``configs/dml.json``).
        family: ``"yolo26"``, ``"rfdetr"`` ou ``"dfine"``.
        num_classes: K, igual para os três (classes da taxonomia ativa).
        imgsz: lado da imagem quadrada de entrada (px).
        module: o ``torch.nn.Module`` treinável.

    Chamada por: ``mutual_learning.train_mutual`` e ``mutual_learning.evaluate_adapter``.
    """

    name: str
    family: str
    num_classes: int
    imgsz: int
    module: object

    def parameters(self):
        """Parâmetros treináveis (para o otimizador)."""
        return [parameter for parameter in self.module.parameters() if parameter.requires_grad]

    def to(self, device) -> "DetectorAdapter":
        """Move o modelo para ``device`` (``"cpu"`` ou ``"cuda"``)."""
        self.module.to(device)
        self.device = device
        return self

    def train(self) -> None:
        """Modo treino (dropout, BatchNorm e grupos/denoising dos DETRs ativos)."""
        self.module.train()

    def eval(self) -> None:
        """Modo avaliação."""
        self.module.eval()

    def forward_train(self, images, targets) -> tuple[object, list[RawPredictions]]:
        """Perda de detecção e previsões cruas COM gradiente.

        Parâmetros:
            images: tensor ``[B, 3, S, S]`` com pixels 0-1 (cada adaptador normaliza).
            targets: lista de B dicts ``{"labels": LongTensor[n], "boxes": FloatTensor[n, 4]}``
                com caixas ``cxcywh`` normalizadas.

        Retorna:
            ``(perda_escalar, [RawPredictions, ...])``; a perda já é a soma ponderada
            que a própria biblioteca usa no treino oficial dela, por imagem.
        """
        raise NotImplementedError

    def predict_raw(self, images) -> RawPredictions:
        """Previsões cruas SEM gradiente, em modo avaliação (pares congelados e validação)."""
        raise NotImplementedError

    def on_epoch_end(self) -> None:
        """Gancho de fim de época (o YOLO26 decai o peso da cabeça one2many aqui)."""

    def save(self, path: Path, names: list[str]) -> Path:
        """Grava os pesos no formato nativo da família; devolve o caminho gravado."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# YOLO26 (Ultralytics)
# ---------------------------------------------------------------------------

class YOLO26Adapter(DetectorAdapter):
    """YOLO26 da Ultralytics treinado fora do ``model.train()``, num laço próprio.

    Parâmetros:
        name: nome do modelo no experimento.
        names: classes (a posição é o class_id).
        imgsz: lado da entrada (múltiplo de 32).
        weights: ``.pt`` (COCO ``yolo26x.pt`` ou um checkpoint do projeto) ou
            ``.yaml`` (``yolo26n.yaml``: pesos aleatórios, só para testes).
        epochs: total de épocas (a perda E2E da Ultralytics decai o peso one2many com ele).
        heads: cabeças usadas na consistência; ``one2many`` treina o backbone
            (a ``one2one`` recebe features destacadas), ``one2one`` é a usada na inferência.

    ATENCAO: o laço próprio não tem o mosaico, o EMA nem o otimizador MuSGD do
    ``train-pilot``. Por isso o controle justo do DML é o braço λ = 0 deste mesmo
    laço, não o treino oficial; ver docs/aprendizado_mutuo_dml_rl.md.
    """

    family = "yolo26"

    def __init__(self, name: str, names: list[str], imgsz: int = 640, weights: str = "yolo26x.pt",
                 epochs: int = 100, heads: tuple[str, ...] = ("one2many", "one2one")):
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import DEFAULT_CFG

        self.name, self.names, self.num_classes, self.imgsz = name, list(names), len(names), int(imgsz)
        self.heads = tuple(heads)
        self.device = "cpu"
        # 1. Carrega o checkpoint ou o yaml. Classes diferentes (COCO 80 → K):
        #    reconstrói a rede com nc=K e transfere as camadas de mesmo formato,
        #    como o trainer da Ultralytics faz em get_model().
        source = YOLO(weights)
        base = source.model
        if str(weights).endswith((".yaml", ".yml")) or int(base.yaml.get("nc", -1)) != self.num_classes:
            model = DetectionModel(deepcopy(base.yaml), nc=self.num_classes, verbose=False)
            if not str(weights).endswith((".yaml", ".yml")):
                model.load(base, verbose=False)
        else:
            model = base
        model.float()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        # 2. Hiperparâmetros da perda (box/cls/dfl e epochs do decaimento E2E).
        args = get_cfg(DEFAULT_CFG)
        args.epochs = int(epochs)
        model.args = args
        model.names = {index: value for index, value in enumerate(self.names)}
        model.criterion = None
        self.module = model
        self.weights = str(weights)

    def _batch(self, images, targets) -> dict:
        """Lote no formato da Ultralytics: ``batch_idx``, ``cls [n,1]`` e ``bboxes`` cxcywh 0-1."""
        import torch

        indexes, classes, boxes = [], [], []
        for position, target in enumerate(targets):
            count = int(target["labels"].numel())
            indexes.append(torch.full((count,), float(position)))
            classes.append(target["labels"].float().view(-1, 1))
            boxes.append(target["boxes"].float().view(-1, 4))
        device = images.device
        return {"img": images, "batch_idx": torch.cat(indexes).to(device), "cls": torch.cat(classes).to(device),
                "bboxes": torch.cat(boxes).to(device)}

    def _decode(self, raw: dict, head_name: str) -> RawPredictions:
        """Caixas por âncora (distâncias ltrb) → cxcywh normalizado; logits ``[B, A, K]``."""
        from ultralytics.utils.tal import dist2bbox, make_anchors

        head = self.module.model[-1]
        anchors, strides = (value.transpose(0, 1) for value in make_anchors(raw["feats"], head.stride, 0.5))
        # ATENCAO: decodifica sempre em xyxy (xywh=False) e converte aqui; o atributo
        # end2end da cabeça muda o formato de _get_decode_boxes conforme o checkpoint.
        xyxy = dist2bbox(head.dfl(raw["boxes"]), anchors.unsqueeze(0), xywh=False, dim=1) * strides
        xyxy = xyxy.permute(0, 2, 1) / float(self.imgsz)
        cxcywh = (xyxy[..., :2] + xyxy[..., 2:]) / 2
        wh = (xyxy[..., 2:] - xyxy[..., :2]).clamp(min=0)
        import torch

        return RawPredictions(torch.cat([cxcywh, wh], dim=-1), raw["scores"].permute(0, 2, 1), head_name)

    def forward_train(self, images, targets):
        self.module.train()
        raw = self.module(images)
        loss, _ = self.module.loss(self._batch(images, targets), raw)
        # A Ultralytics devolve [box, cls, dfl] já multiplicado pelo tamanho do lote;
        # dividir deixa a perda "por imagem", na mesma escala dos DETRs.
        det = loss.sum() / max(len(targets), 1)
        heads = raw if isinstance(raw, dict) and "one2one" in raw else {"one2many": raw}
        return det, [self._decode(heads[key], key) for key in self.heads if key in heads]

    def predict_raw(self, images):
        import torch

        self.module.eval()
        with torch.no_grad():
            output = self.module(images)
        raw = output[1] if isinstance(output, tuple) else output
        key = "one2one" if isinstance(raw, dict) and "one2one" in raw else None
        return self._decode(raw[key] if key else raw, key or "one2many")

    def on_epoch_end(self) -> None:
        criterion = getattr(self.module, "criterion", None)
        if criterion is not None and hasattr(criterion, "update"):
            criterion.update()

    def save(self, path: Path, names: list[str]) -> Path:
        import torch
        from ultralytics import __version__

        path = Path(path).with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        model = deepcopy(self.module).cpu()
        model.criterion = None
        # Checkpoint mínimo que ultralytics.YOLO(path) abre (mesmas chaves do trainer).
        torch.save({"date": datetime.now(timezone.utc).isoformat(), "version": __version__, "epoch": -1,
                    "model": model, "ema": None, "train_args": {"imgsz": self.imgsz, "source": "dml"},
                    "dronecamp_dml": {"family": self.family, "names": list(names)}}, path)
        return path


# ---------------------------------------------------------------------------
# RF-DETR (Roboflow)
# ---------------------------------------------------------------------------

# Variantes do pacote rfdetr (nome do spec → classe do pacote).
RFDETR_VARIANTS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
                   "base": "RFDETRBase", "large": "RFDETRLarge"}


class RFDETRAdapter(DetectorAdapter):
    """RF-DETR com o critério de Hungarian matching do próprio pacote.

    Parâmetros:
        name, names, imgsz: como no YOLO26Adapter; ``imgsz`` precisa ser múltiplo
            de ``patch_size × num_windows`` (32 nas variantes nano/small/medium).
        variant: ``nano``, ``small``, ``medium``, ``base`` ou ``large``.
        weights: ``None`` = pesos COCO oficiais (baixados no 1.º uso); ``"none"`` =
            aleatórios (testes); caminho = checkpoint ``.pth``.

    ATENCAO: no treino o RF-DETR usa 13 grupos de queries (Group DETR); a
    consistência usa só o 1.º grupo (as ``num_queries`` primeiras), que é o
    único usado na inferência.
    """

    family = "rfdetr"

    def __init__(self, name: str, names: list[str], imgsz: int = 640, variant: str = "medium",
                 weights: str | None = None, lr: float | None = None):
        import rfdetr
        from rfdetr.config import TrainConfig
        from rfdetr.models.lwdetr import build_criterion_from_config

        self.name, self.names, self.num_classes, self.imgsz = name, list(names), len(names), int(imgsz)
        self.device = "cpu"
        if variant not in RFDETR_VARIANTS:
            raise ValueError(f"Variante RF-DETR desconhecida: {variant} (use {', '.join(RFDETR_VARIANTS)}).")
        # ATENCAO: o DINOv2 com janelas exige lado múltiplo de patch_size × num_windows
        # (32 em nano/small/medium/large, 56 no base); 640 não serve para o base.
        fields = getattr(rfdetr, RFDETR_VARIANTS[variant])._model_config_class.model_fields
        patch, windows = fields["patch_size"].default, fields["num_windows"].default
        block = patch * windows if isinstance(patch, int) and isinstance(windows, int) else 0
        if block and self.imgsz % block:
            raise ValueError(f"RF-DETR {variant} precisa de imgsz múltiplo de {block} (recebido {self.imgsz}); "
                             f"use {self.imgsz // block * block} ou {(self.imgsz // block + 1) * block}.")
        options = {"num_classes": self.num_classes, "resolution": self.imgsz, "device": "cpu"}
        if weights is not None:
            options["pretrain_weights"] = None if str(weights).lower() == "none" else str(weights)
        wrapper = getattr(rfdetr, RFDETR_VARIANTS[variant])(**options)
        self.model_config = wrapper.model_config
        self.module = wrapper.model.model
        self.num_queries = int(self.module.num_queries)
        self.criterion, _ = build_criterion_from_config(self.model_config, TrainConfig(dataset_dir=".", output_dir="."))
        self.weights = str(weights)

    def _normalize(self, images):
        import torch

        mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
        return (images - mean) / std

    def _raw(self, output: dict) -> RawPredictions:
        # Só o 1.º grupo de queries e só as K primeiras colunas (a última é "sem objeto").
        boxes = output["pred_boxes"][:, : self.num_queries]
        logits = output["pred_logits"][:, : self.num_queries, : self.num_classes]
        return RawPredictions(boxes, logits, "decoder")

    def forward_train(self, images, targets):
        self.module.train()
        output = self.module(self._normalize(images))
        device = images.device
        losses = self.criterion(output, [{"labels": item["labels"].to(device), "boxes": item["boxes"].to(device)}
                                         for item in targets])
        weights = self.criterion.weight_dict
        det = sum(losses[key] * weights[key] for key in losses if key in weights)
        return det, [self._raw(output)]

    def predict_raw(self, images):
        import torch

        self.module.eval()
        with torch.no_grad():
            output = self.module(self._normalize(images))
        return self._raw(output)

    def save(self, path: Path, names: list[str]) -> Path:
        import torch

        path = Path(path).with_suffix(".pth")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {key: value.cpu() for key, value in self.module.state_dict().items()},
                    "model_config": self.model_config.model_dump(mode="json"),
                    "class_names": list(names), "dronecamp_dml": {"family": self.family}}, path)
        return path


# ---------------------------------------------------------------------------
# D-FINE (Hugging Face transformers)
# ---------------------------------------------------------------------------

class DFINEAdapter(DetectorAdapter):
    """D-FINE (refinamento fino de distribuição) via ``transformers``.

    Parâmetros:
        name, names, imgsz: como acima (múltiplo de 32).
        variant: ID do Hugging Face (ex.: ``ustc-community/dfine-medium-coco``) ou
            ``"random-small"`` para pesos aleatórios (testes, sem internet).
        weights: pasta ``save_pretrained`` de um treino anterior (tem prioridade).

    ATENCAO: no treino com ``labels`` o D-FINE acrescenta queries de denoising ANTES
    das queries normais; a consistência usa só as ``num_queries`` últimas.
    """

    family = "dfine"

    def __init__(self, name: str, names: list[str], imgsz: int = 640,
                 variant: str = "ustc-community/dfine-medium-coco", weights: str | None = None):
        from transformers import DFineConfig, DFineForObjectDetection

        self.name, self.names, self.num_classes, self.imgsz = name, list(names), len(names), int(imgsz)
        self.device = "cpu"
        labels = {index: value for index, value in enumerate(self.names)}
        source = weights or variant
        if source == "random-small":
            # Configuração reduzida (2 camadas de decoder, 100 queries) para testes rápidos.
            config = DFineConfig(num_labels=self.num_classes, id2label=labels,
                                 label2id={v: k for k, v in labels.items()}, decoder_layers=2, num_queries=100)
            self.module = DFineForObjectDetection(config)
        else:
            self.module = DFineForObjectDetection.from_pretrained(
                source, num_labels=self.num_classes, id2label=labels, label2id={v: k for k, v in labels.items()},
                ignore_mismatched_sizes=True)
        self.num_queries = int(self.module.config.num_queries)
        self.weights = str(source)

    def _raw(self, output) -> RawPredictions:
        return RawPredictions(output.pred_boxes[:, -self.num_queries:], output.logits[:, -self.num_queries:], "decoder")

    def forward_train(self, images, targets):
        self.module.train()
        labels = [{"class_labels": item["labels"].to(images.device), "boxes": item["boxes"].to(images.device)}
                  for item in targets]
        output = self.module(pixel_values=images, labels=labels)
        return output.loss, [self._raw(output)]

    def predict_raw(self, images):
        import torch

        self.module.eval()
        with torch.no_grad():
            output = self.module(pixel_values=images)
        return self._raw(output)

    def save(self, path: Path, names: list[str]) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.module.save_pretrained(path)
        (path / "dronecamp_dml.json").write_text(json.dumps({"family": self.family, "names": list(names)}),
                                                 encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Fábrica e dependências
# ---------------------------------------------------------------------------

# EXTENSAO: família nova = subclasse de DetectorAdapter + uma linha aqui + o pacote
# em DEPENDENCIES; o configs/dml.json passa a aceitar "family": "<nome>".
FAMILIES = {"yolo26": YOLO26Adapter, "rfdetr": RFDETRAdapter, "dfine": DFINEAdapter}
# Pacote Python exigido por família (conferido por dependency_report).
DEPENDENCIES = {"yolo26": "ultralytics", "rfdetr": "rfdetr", "dfine": "transformers"}


def build_adapter(spec: dict, names: list[str], imgsz: int, epochs: int = 100) -> DetectorAdapter:
    """Cria o adaptador descrito por um item de ``models`` do ``configs/dml.json``.

    Parâmetros:
        spec: ``{"name", "family", "variant"?, "weights"?}``.
        names: classes da taxonomia ativa.
        imgsz: lado da entrada (o mesmo para os três: todos veem a mesma imagem).
        epochs: total de épocas (só o YOLO26 usa).

    Erros:
        ValueError: família desconhecida.

    Chamada por: ``mutual_learning`` e ``dml_lan``.
    """
    family = spec.get("family")
    if family not in FAMILIES:
        raise ValueError(f"Família desconhecida: {family!r} (use {', '.join(FAMILIES)}).")
    if family == "yolo26":
        return YOLO26Adapter(spec["name"], names, imgsz, weights=spec.get("weights") or f"{spec.get('variant', 'yolo26x')}.pt",
                             epochs=epochs, heads=tuple(spec.get("heads", ("one2many", "one2one"))))
    if family == "rfdetr":
        return RFDETRAdapter(spec["name"], names, imgsz, variant=spec.get("variant", "medium"), weights=spec.get("weights"))
    return DFINEAdapter(spec["name"], names, imgsz, variant=spec.get("variant", "ustc-community/dfine-medium-coco"),
                        weights=spec.get("weights"))


def dependency_report() -> dict:
    """``{família: {"package", "installed"}}`` mais o torch e se há CUDA.

    Chamada por: ``mutual_learning.plan_dml`` (comando ``dml-plan``).
    """
    report = {family: {"package": package, "installed": importlib.util.find_spec(package) is not None}
              for family, package in DEPENDENCIES.items()}
    torch_ok = importlib.util.find_spec("torch") is not None
    cuda = False
    if torch_ok:
        import torch

        cuda = bool(torch.cuda.is_available())
    report["torch"] = {"package": "torch", "installed": torch_ok, "cuda": cuda}
    return report
