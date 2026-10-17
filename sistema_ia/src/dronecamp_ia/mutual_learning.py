"""Aprendizado mútuo (Deep Mutual Learning) entre YOLO26, RF-DETR e D-FINE.

Função no projeto: treinar os três detectores para que cada um aprenda com as
anotações humanas e, ao mesmo tempo, com as previsões dos outros dois, e medir
se isso ajuda. Formulação completa e justificativas em
``docs/aprendizado_mutuo_dml_rl.md``; resumo:

    L_i = L_detecção,i  +  λ(t) · a_i(t) · s_i · L_consistência,i

- ``L_detecção,i``: a perda oficial da biblioteca do modelo i (Ultralytics E2E,
  SetCriterion do RF-DETR, perda do D-FINE no transformers).
- ``λ(t)``: rampa gaussiana ``λ_max·exp(−5(1 − t/T)²)`` nas ``ramp_epochs``
  primeiras épocas (Laine & Aila 2017): no começo os pares ainda erram muito.
- ``a_i(t)``: multiplicador escolhido por época pelo controlador por reforço
  (``dml_controller.DiscountedUCB``); ``a = 0`` é o controle sem consistência.
- ``s_i = EMA(L_det,i) / EMA(L_cons,i)`` (sem gradiente): deixa λ sem unidade. As
  três perdas têm escalas muito diferentes (D-FINE começa perto de 1000, RF-DETR
  perto de 10); com ``s_i``, λ = 0,5 significa "a consistência pesa metade da
  detecção" nos três.
- ``L_consistência,i``: (1) as previsões dos pares j ≠ i (sem gradiente) com
  confiança ≥ τ são fundidas por sobreposição (estilo WBF): caixa = média
  ponderada, probabilidade por classe = média sobre os pares (par que não viu
  conta zero), peso = confiança × fração de pares que concordam; (2) no modo
  ``anchor = "gt"`` só ficam os alvos fundidos que coincidem (IoU ≥ 0,5) com
  alguma caixa humana: nas fotos aprovadas tudo que não é caixa humana é falso
  positivo, então o que vale aprender dos pares é a distribuição entre classes e
  o ajuste fino da caixa, não as detecções extras; (3) Hungarian matching
  entre as queries/âncoras do modelo i e os alvos (custo = BCE de classe + 5·L1
  + 2·(1 − GIoU), como no DETR); (4) perda nos pares casados:
  ``Σ_c KL_Bernoulli(p̄_c ‖ σ(z_c/T))·T² + β·(L1 + 1 − GIoU)``, ponderada.

Modos (``mode``):
- ``online``: os três na memória, todos aprendem a cada lote (DML original).
- ``distill``: um aluno aprende; os outros são professores congelados
  (destilação do ensemble num modelo só, ex.: os três → YOLO26 para a borda).
- ``solo``: um aluno sem pares (λ = 0); é a rodada 0 da rede local e o controle justo.
- ``cached``: um aluno aprende com previsões dos pares gravadas em disco por
  ``build_peer_cache`` (DML em rodadas: cada rodada usa os pares da rodada
  anterior). É o modo que cabe na CPU do 4600G (um modelo por vez na memória) e
  o que ``dml_lan.py`` distribui entre máquinas da rede local.

Conteúdo do arquivo (seções):
- Dados: ``load_dataset_spec``, ``split_images``, ``TileDataset``, ``collate``.
- Geometria e fusão: ``cxcywh_to_xyxy``, ``fuse_peer_predictions``, ``anchor_to_truth``.
- Perda: ``bernoulli_kl``, ``consistency_loss``, ``ramp_weight``.
- Avaliação: ``detections_from_raw``, ``evaluate_adapter``, ``evaluate_ensemble``,
  ``complementarity``.
- Cache de pares: ``build_peer_cache``, ``PeerCache``.
- Treino: ``DEFAULT_SETTINGS``, ``load_dml_config``, ``train_mutual``.
- Planejamento: ``plan_dml`` (dependências + tempo medido por passo).
- ENTRADAS da CLI: ``dml-plan`` (plan_dml), ``dml-train`` (train_mutual),
  ``dml-cache`` (build_peer_cache), ``dml-evaluate`` (evaluate_models).

O que lê e grava no disco:
- Lê: o ``dataset.yaml`` (formato Ultralytics: ``path``, ``train``, ``val``,
  ``test``, ``names``; pastas ou listas .txt), as fotos e labels YOLO, o
  ``configs/dml.json`` e, no modo ``cached``, ``<cache>/<modelo>.npz``.
- Grava em ``runs/dml_<data>_<id>/``: ``settings.json``, ``history.jsonl`` (uma linha
  por época e modelo), ``controller.json`` (estado do RL), ``weights/<modelo>/last*``
  e ``best*`` (formato nativo de cada família) e ``summary.json``.
- ``dml-cache`` grava ``<saída>/<modelo>.npz`` + ``<modelo>.json``.
- ``dml-evaluate`` grava ``runs/dml_eval_*/evaluation.json``.

Regras do projeto que este módulo respeita:
- Nada aqui altera revisões, caixas humanas ou o modelo em uso. Os pesos gerados
  são candidatos: adoção só depois de ``compare-models`` (YOLO26) ou
  ``dml-evaluate`` no teste contra o modelo atual, por classe.
- O split ``test`` nunca é usado no treino nem na escolha do melhor checkpoint.

Onde mexer:
- Pesos da perda de consistência: seção ``consistency`` do ``configs/dml.json``.
- Fusão dos pares (IoU, limite por par): ``fuse_peer_predictions``.
- Outro critério de "melhor época": ``train_mutual``, etapa 6 (hoje mAP50 de val).
- Aumento de dados: ``TileDataset.__getitem__`` (hoje espelho horizontal).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time

from .io import make_run_directory, write_json
from .matching import image_errors, match_predictions, per_class_report

# Extensões de imagem aceitas nas pastas do dataset (as mesmas da Ultralytics que usamos).
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Padrões do experimento. O configs/dml.json sobrescreve por seção; a CLI sobrescreve
# epochs/batch/mode. Cada valor está explicado em docs/aprendizado_mutuo_dml_rl.md.
DEFAULT_SETTINGS = {
    "imgsz": 640,
    "train": {"epochs": 60, "batch": 4, "weight_decay": 0.0001, "seed": 42, "eval_conf": 0.25,
              "hflip": 0.5, "vflip": 0.0, "threads": None, "device": "cpu"},
    "consistency": {"lambda": 0.5, "ramp_epochs": 5, "tau": 0.3, "fuse_iou": 0.55, "anchor": "gt",
                    "anchor_iou": 0.5, "ramp_rounds": 2, "temperature": 1.0, "box_weight": 1.0, "negative_weight": 0.0,
                    "max_per_peer": 100, "ema": 0.9},
    "rl": {"enabled": True, "arms": [0.5, 0.0, 1.0, 0.25], "gamma": 0.9, "exploration": 0.05},
}
# Taxa de aprendizado e corte de gradiente padrão por família (AdamW). Os DETRs
# usam 1e-4 e corte 0,1 como nos treinos oficiais deles; o YOLO tolera mais.
FAMILY_DEFAULTS = {"yolo26": {"lr": 0.001, "grad_clip": 10.0},
                   "rfdetr": {"lr": 0.0001, "grad_clip": 0.1},
                   "dfine": {"lr": 0.0001, "grad_clip": 0.1}}


# ---------------------------------------------------------------------------
# Dados: dataset.yaml da Ultralytics → tensores comuns aos três modelos
# ---------------------------------------------------------------------------

def load_dataset_spec(data_path: Path) -> tuple[dict, Path, list[str]]:
    """Lê o ``dataset.yaml`` e devolve ``(spec, raiz, nomes)``.

    Parâmetros:
        data_path: caminho do ``dataset.yaml`` (o mesmo que o ``train-pilot`` usa).

    Retorna:
        ``spec`` (dict do YAML), ``raiz`` (``path`` do YAML, relativo à pasta do
        arquivo quando não absoluto) e a lista de nomes na ordem dos IDs.

    Erros:
        FileNotFoundError: arquivo ausente. ValueError: sem ``names``.
    """
    import yaml

    data_path = Path(data_path)
    spec = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    # Sem "path" a raiz é a própria pasta do YAML (não juntar a pasta duas vezes).
    root = Path(spec["path"]) if spec.get("path") else data_path.parent
    if not root.is_absolute():
        root = (data_path.parent / root) if spec.get("path") else root
    root = root.resolve()
    names = spec.get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=int)]
    if not names:
        raise ValueError(f"{data_path} não tem 'names'.")
    return spec, root, list(names)


def split_images(spec: dict, root: Path, split: str) -> list[Path]:
    """Fotos de um split: pasta (busca recursiva), lista ``.txt`` ou lista das duas coisas.

    Retorna lista ordenada (ordem estável = mesmo lote em todas as máquinas).
    Erros: ValueError se o split não existir no YAML.
    """
    entry = spec.get(split)
    if entry is None:
        raise ValueError(f"O dataset não tem o split '{split}'.")
    images: list[Path] = []
    for item in entry if isinstance(entry, list) else [entry]:
        path = Path(item)
        path = path if path.is_absolute() else root / path
        if path.is_file() and path.suffix == ".txt":
            # Lista de fotos (ex.: train_repeat_factor.txt do RFS): uma por linha, repetições valem.
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    candidate = Path(line.strip())
                    images.append(candidate if candidate.is_absolute() else (root / candidate))
        elif path.is_dir():
            images.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES))
    return images


def label_path(image: Path) -> Path:
    """Label YOLO da foto: troca a última pasta ``images`` por ``labels`` e a extensão por ``.txt``."""
    parts = list(image.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def image_key(image: Path, root: Path) -> str:
    """Chave estável da foto no cache de pares (caminho relativo à raiz, com ``/``)."""
    try:
        return image.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return image.name


def read_labels(path: Path) -> tuple[list[int], list[list[float]]]:
    """Classes e caixas ``cxcywh`` normalizadas de uma label YOLO (arquivo ausente = foto negativa)."""
    classes, boxes = [], []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) >= 5:
                classes.append(int(float(values[0])))
                boxes.append([float(value) for value in values[1:5]])
    return classes, boxes


class TileDataset:
    """Recortes (ou fotos) do dataset YOLO como tensores 0-1 quadrados de lado ``imgsz``.

    Parâmetros:
        data_path: ``dataset.yaml``.
        split: ``train``, ``val`` ou ``test``.
        imgsz: lado da saída (px). Os recortes de 640 do treino v7 não são redimensionados.
        augment: liga o espelhamento aleatório (só no treino).
        hflip, vflip: probabilidade de espelhar na horizontal/vertical.
        seed: semente do sorteio dos espelhos (mesmo seed = mesmos espelhos).

    ATENCAO: o MESMO espelho vale para os três modelos no lote (eles precisam ver a
    mesma imagem para que as previsões sejam comparáveis), e o cache de pares é
    espelhado junto (``PeerCache.lookup``).
    ATENCAO: foto não quadrada é esticada para ``imgsz`` × ``imgsz`` (as labels são
    normalizadas, então continuam certas); os recortes do treino já são quadrados.
    """

    def __init__(self, data_path: Path, split: str, imgsz: int = 640, augment: bool = False,
                 hflip: float = 0.5, vflip: float = 0.0, seed: int = 42):
        self.spec, self.root, self.names = load_dataset_spec(data_path)
        self.images = split_images(self.spec, self.root, split)
        self.split, self.imgsz, self.augment = split, int(imgsz), bool(augment)
        self.hflip, self.vflip = float(hflip), float(vflip)
        self.random = random.Random(seed)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        import numpy as np
        from PIL import Image
        import torch

        path = self.images[index]
        # 1. Foto em RGB, redimensionada ao lado comum.
        with Image.open(path) as source:
            width, height = source.size
            picture = source.convert("RGB")
            if picture.size != (self.imgsz, self.imgsz):
                picture = picture.resize((self.imgsz, self.imgsz), Image.BILINEAR)
            array = np.asarray(picture, dtype=np.float32) / 255.0
        classes, boxes = read_labels(label_path(path))
        # 2. Espelhos (mesma transformação na imagem e nas caixas cxcywh).
        flip_h = self.augment and self.random.random() < self.hflip
        flip_v = self.augment and self.random.random() < self.vflip
        if flip_h:
            array = array[:, ::-1]
            boxes = [[1.0 - cx, cy, w, h] for cx, cy, w, h in boxes]
        if flip_v:
            array = array[::-1, :]
            boxes = [[cx, 1.0 - cy, w, h] for cx, cy, w, h in boxes]
        image = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
        target = {"labels": torch.tensor(classes, dtype=torch.long),
                  "boxes": torch.tensor(boxes, dtype=torch.float32).view(-1, 4)}
        meta = {"key": image_key(path, self.root), "path": str(path), "size": (width, height),
                "hflip": flip_h, "vflip": flip_v}
        return image, target, meta


def collate(items):
    """Junta itens do ``TileDataset`` em ``(imagens [B,3,S,S], alvos, metas)``."""
    import torch

    images, targets, metas = zip(*items)
    return torch.stack(images), list(targets), list(metas)


def iterate_batches(dataset: TileDataset, batch: int, shuffle: bool, seed: int, max_batches: int | None = None):
    """Gera lotes na ordem do dataset ou embaralhada (semente fixa por época).

    Chamada por: ``train_mutual``, ``evaluate_adapter``, ``build_peer_cache``, ``plan_dml``.
    """
    order = list(range(len(dataset)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for count, start in enumerate(range(0, len(order), batch)):
        if max_batches is not None and count >= max_batches:
            return
        yield collate([dataset[index] for index in order[start:start + batch]])


# ---------------------------------------------------------------------------
# Geometria e fusão das previsões dos pares
# ---------------------------------------------------------------------------

def cxcywh_to_xyxy(boxes):
    """``[..., 4]`` cx, cy, w, h → x1, y1, x2, y2 (mesma escala)."""
    import torch

    center, size = boxes[..., :2], boxes[..., 2:].clamp(min=0)
    return torch.cat([center - size / 2, center + size / 2], dim=-1)


def peer_view(raws: list, image: int):
    """Caixas cxcywh e probabilidades de UMA imagem, da cabeça que o par usa na inferência.

    Usa a última cabeça da lista (``one2one`` no YOLO26, decoder nos DETRs).
    Retorna ``(boxes [Q,4], probs [Q,K])`` sem gradiente.
    """
    raw = raws[-1] if isinstance(raws, list) else raws
    return raw.boxes[image].detach(), raw.logits[image].detach().sigmoid()


def fuse_peer_predictions(peers: list[tuple], tau: float, fuse_iou: float = 0.55, max_per_peer: int = 100,
                          peer_nms_iou: float = 0.7, details: bool = False):
    """Funde as previsões dos pares de uma imagem em alvos de consistência.

    Parâmetros:
        peers: uma tupla ``(boxes cxcywh [Q,4], probs [Q,K])`` por par (sem gradiente).
        tau: confiança mínima (máximo das probabilidades) para entrar na fusão.
        fuse_iou: sobreposição mínima entre previsões de pares diferentes para
            virarem o mesmo alvo.
        max_per_peer: no máximo N previsões por par (as mais confiantes).
        peer_nms_iou: NMS sem classe dentro de cada par (remove as duplicatas da
            cabeça one2many/denoising antes de fundir).
        details: devolve também quem contribuiu para cada alvo (registro de
            cobertura da cooperação; não muda a fusão).

    Retorna:
        ``(boxes xyxy [T,4], probs [T,K], weights [T])``; T pode ser 0. Com
        ``details=True``, um 4.º item ``{"received": [[classe, ...] por par],
        "members": [[(par, classe), ...] por alvo]}``: ``received`` são as previsões
        de cada par que passaram por τ e pelo NMS interno (classe = maior probabilidade).

    Algoritmo: candidatos de todos os pares em ordem de confiança; cada um entra
    no primeiro grupo com IoU ≥ ``fuse_iou`` que ainda não tem previsão do MESMO
    par, ou abre um grupo novo. Caixa do grupo = média ponderada pela confiança;
    probabilidade = soma das probabilidades / nº de pares (par ausente conta 0);
    peso = confiança média × (pares no grupo / nº de pares).

    Chamada por: ``train_mutual`` e ``evaluate_ensemble``.
    """
    import torch
    from torchvision.ops import box_iou, nms

    info = {"received": [[] for _ in peers], "members": []}
    if not peers:
        empty = (torch.zeros((0, 4)), torch.zeros((0, 0)), torch.zeros((0,)))
        return (*empty, info) if details else empty
    num_classes = peers[0][1].shape[-1]
    device = peers[0][0].device
    candidates = []
    # 1. Filtra por confiança e remove duplicatas dentro de cada par.
    for peer_index, (boxes, probs) in enumerate(peers):
        scores = probs.max(dim=-1).values
        keep = torch.nonzero(scores >= tau).flatten()
        if keep.numel() == 0:
            continue
        keep = keep[scores[keep].argsort(descending=True)[:max_per_peer]]
        xyxy = cxcywh_to_xyxy(boxes[keep]).clamp(0, 1)
        kept = nms(xyxy.float(), scores[keep].float(), peer_nms_iou)
        for position in kept.tolist():
            candidates.append((float(scores[keep][position]), peer_index, xyxy[position], probs[keep][position]))
            info["received"][peer_index].append(int(probs[keep][position].argmax()))
    if not candidates:
        empty = (torch.zeros((0, 4), device=device), torch.zeros((0, num_classes), device=device),
                 torch.zeros((0,), device=device))
        return (*empty, info) if details else empty
    # 2. Agrupamento guloso entre pares, do mais confiante para o menos.
    candidates.sort(key=lambda item: -item[0])
    groups: list[dict] = []
    for score, peer_index, box, prob in candidates:
        chosen = None
        if groups:
            centers = torch.stack([group["box"] for group in groups])
            overlaps = box_iou(box.view(1, 4), centers).flatten()
            for position in overlaps.argsort(descending=True).tolist():
                if overlaps[position] < fuse_iou:
                    break
                if peer_index not in groups[position]["peers"]:
                    chosen = groups[position]
                    break
        if chosen is None:
            groups.append({"peers": {peer_index}, "scores": [score], "boxes": [box], "probs": [prob], "box": box,
                           "members": [(peer_index, int(prob.argmax()))]})
        else:
            chosen["peers"].add(peer_index)
            chosen["members"].append((peer_index, int(prob.argmax())))
            chosen["scores"].append(score)
            chosen["boxes"].append(box)
            chosen["probs"].append(prob)
            weights = torch.tensor(chosen["scores"], device=device).view(-1, 1)
            chosen["box"] = (torch.stack(chosen["boxes"]) * weights).sum(0) / weights.sum()
    # 3. Alvo final por grupo.
    total = float(len(peers))
    boxes = torch.stack([group["box"] for group in groups])
    probs = torch.stack([torch.stack(group["probs"]).sum(0) / total for group in groups])
    weights = torch.tensor([sum(group["scores"]) / len(group["scores"]) * len(group["peers"]) / total
                            for group in groups], device=device)
    if details:
        info["members"] = [group["members"] for group in groups]
        return boxes, probs, weights, info
    return boxes, probs, weights


def anchor_to_truth(boxes, probs, weights, truth_xyxy, anchor_iou: float = 0.5, return_mask: bool = False):
    """Mantém só os alvos fundidos que coincidem com alguma caixa humana (modo ``anchor = "gt"``).

    Parâmetros: alvos da fusão e caixas humanas xyxy normalizadas ``[n,4]``;
    ``return_mask`` devolve também a máscara booleana ``[T]`` dos alvos mantidos.
    Retorna os mesmos três tensores filtrados (mais a máscara, se pedida).
    """
    import torch
    from torchvision.ops import box_iou

    if boxes.shape[0] == 0 or truth_xyxy.shape[0] == 0:
        keep = torch.zeros((boxes.shape[0],), dtype=torch.bool, device=boxes.device)
    else:
        keep = box_iou(boxes, truth_xyxy.to(boxes.device)).max(dim=1).values >= anchor_iou
    result = (boxes[keep], probs[keep], weights[keep])
    return (*result, keep) if return_mask else result


# ---------------------------------------------------------------------------
# Perda de consistência
# ---------------------------------------------------------------------------

def bernoulli_kl(target, logits, temperature: float = 1.0):
    """KL(p̄ ‖ σ(z/T)) por classe, somado nas classes, vezes T² (escala de Hinton et al. 2015).

    Parâmetros: ``target`` probabilidades ``[N,K]`` (alvo, sem gradiente); ``logits`` ``[N,K]``.
    Retorna tensor ``[N]``. Igual a BCE(z/T, p̄) − H(p̄): mesmo gradiente da BCE com
    alvo suave, mas vale 0 quando o aluno já concorda com os pares.
    """
    import torch
    import torch.nn.functional as F

    target = target.clamp(1e-6, 1 - 1e-6)
    bce = F.binary_cross_entropy_with_logits(logits / temperature, target, reduction="none")
    entropy = -(target * torch.log(target) + (1 - target) * torch.log(1 - target))
    return (bce - entropy).sum(-1) * temperature ** 2


def consistency_loss(boxes, logits, target_boxes, target_probs, target_weights, temperature: float = 1.0,
                     box_weight: float = 1.0, negative_weight: float = 0.0, min_iou: float = 0.1,
                     negatives: int = 50):
    """Perda de consistência de UMA imagem do aluno contra os alvos fundidos dos pares.

    Parâmetros:
        boxes: ``[Q,4]`` cxcywh 0-1 do aluno (com gradiente).
        logits: ``[Q,K]`` do aluno (com gradiente).
        target_boxes: ``[T,4]`` xyxy 0-1; ``target_probs`` ``[T,K]``; ``target_weights`` ``[T]``.
        temperature: T do KL.
        box_weight: β do termo de caixa (L1 + 1 − GIoU).
        negative_weight: peso de empurrar para 0 as queries confiantes do aluno que
            não casaram com nenhum alvo (0 = desligado; ver ATENCAO).
        min_iou: par casado com IoU menor que isso é descartado (casamento absurdo
            no começo do treino).
        negatives: quantas queries não casadas entram no termo negativo.

    Retorna:
        ``(perda escalar, {"matched": n, "targets": T, "matched_targets": [j, ...]})``;
        ``matched_targets`` são as posições dos alvos casados (cada alvo no máximo
        uma vez nesta cabeça). Perda 0 ligada ao grafo quando não há alvo (o
        ``backward`` continua válido).

    ATENCAO: o termo negativo pune o aluno por achar algo que os pares não acharam;
    num par fraco isso apaga acertos. Fica desligado por padrão e só deve ser ligado
    com o modo ``anchor = "gt"`` desligado e medição no teste.

    Chamada por: ``train_mutual``.
    """
    import torch
    import torch.nn.functional as F
    from scipy.optimize import linear_sum_assignment
    from torchvision.ops import box_iou, generalized_box_iou

    zero = logits.sum() * 0.0
    if target_boxes.shape[0] == 0:
        return zero, {"matched": 0, "targets": 0, "matched_targets": []}
    student_xyxy = cxcywh_to_xyxy(boxes).clamp(0, 1)
    target_boxes = target_boxes.to(boxes.device)
    target_probs = target_probs.to(boxes.device)
    target_weights = target_weights.to(boxes.device)
    # 1. Custo do casamento (sem gradiente): classe + caixa, como no matcher do DETR.
    with torch.no_grad():
        probs = logits.sigmoid()
        class_cost = -(target_probs.unsqueeze(0) * torch.log(probs.unsqueeze(1).clamp(min=1e-6))
                       + (1 - target_probs.unsqueeze(0)) * torch.log((1 - probs.unsqueeze(1)).clamp(min=1e-6))).mean(-1)
        l1_cost = torch.cdist(student_xyxy, target_boxes, p=1)
        giou = generalized_box_iou(student_xyxy, target_boxes)
        cost = class_cost + 5.0 * l1_cost + 2.0 * (1.0 - giou)
        rows, cols = linear_sum_assignment(cost.cpu().numpy())
        rows_t = torch.as_tensor(rows, dtype=torch.long, device=boxes.device)
        cols_t = torch.as_tensor(cols, dtype=torch.long, device=boxes.device)
        # IoU real do par casado (GIoU pode ser > 0 mesmo com caixas disjuntas).
        valid = torch.diag(box_iou(student_xyxy[rows_t], target_boxes[cols_t])) >= min_iou
        rows_t, cols_t = rows_t[valid], cols_t[valid]
    if rows_t.numel() == 0:
        return zero, {"matched": 0, "targets": int(target_boxes.shape[0]), "matched_targets": []}
    # 2. Classe (KL de Bernoulli) e caixa (L1 + 1 − GIoU) nos pares casados.
    weights = target_weights[cols_t]
    class_term = bernoulli_kl(target_probs[cols_t], logits[rows_t], temperature)
    matched_boxes = student_xyxy[rows_t]
    box_term = (matched_boxes - target_boxes[cols_t]).abs().sum(-1) \
        + (1.0 - torch.diag(generalized_box_iou(matched_boxes, target_boxes[cols_t])))
    loss = (weights * (class_term + box_weight * box_term)).sum() / weights.sum().clamp(min=1e-6)
    # 3. Termo negativo opcional (desligado por padrão).
    if negative_weight > 0:
        mask = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
        mask[rows_t] = False
        rest = logits[mask]
        if rest.numel():
            top = rest.max(-1).values.topk(min(negatives, rest.shape[0])).indices
            loss = loss + negative_weight * F.binary_cross_entropy_with_logits(
                rest[top], torch.zeros_like(rest[top]), reduction="none").sum(-1).mean()
    return loss, {"matched": int(rows_t.numel()), "targets": int(target_boxes.shape[0]),
                  "matched_targets": cols_t.tolist()}


def ramp_weight(epoch: int, ramp_epochs: int, maximum: float) -> float:
    """λ(t) = λ_max·exp(−5(1 − t/T)²) nas T primeiras épocas, depois λ_max (época começa em 0)."""
    if ramp_epochs <= 0:
        return float(maximum)
    progress = min(1.0, (epoch + 1) / float(ramp_epochs))
    return float(maximum) * math.exp(-5.0 * (1.0 - progress) ** 2)


# ---------------------------------------------------------------------------
# Avaliação no formato do projeto (matching.py / model_gate.average_precision)
# ---------------------------------------------------------------------------

def detections_from_raw(boxes, logits, size: tuple[int, int], min_conf: float = 0.001, max_det: int = 100,
                        nms_iou: float = 0.7, probabilities: bool = False) -> list[dict]:
    """Previsões cruas de UMA imagem → sugestões ``{"class_id", "bbox_xyxy" px, "confidence"}``.

    Classe = maior probabilidade; NMS por classe (inofensivo nos modelos sem NMS,
    necessário na fusão do ensemble); caixas na escala da foto original ``size``.
    ``probabilities=True`` quando ``logits`` já são probabilidades (saída da fusão).
    """
    from torchvision.ops import batched_nms

    probs = logits if probabilities else logits.sigmoid()
    scores, classes = probs.max(-1)
    keep = scores >= min_conf
    if keep.sum() == 0:
        return []
    xyxy = cxcywh_to_xyxy(boxes[keep]).clamp(0, 1)
    scores, classes = scores[keep], classes[keep]
    kept = batched_nms(xyxy.float(), scores.float(), classes, nms_iou)[:max_det]
    width, height = size
    result = []
    for index in kept.tolist():
        x1, y1, x2, y2 = xyxy[index].tolist()
        result.append({"class_id": int(classes[index]), "confidence": float(scores[index]),
                       "bbox_xyxy": [x1 * width, y1 * height, x2 * width, y2 * height]})
    return result


def truth_from_target(target: dict, size: tuple[int, int]) -> list[dict]:
    """Caixas humanas do alvo (cxcywh 0-1) no formato do ``matching.py`` (xyxy px)."""
    width, height = size
    result = []
    for label, (cx, cy, w, h) in zip(target["labels"].tolist(), target["boxes"].tolist()):
        result.append({"class_id": int(label),
                       "bbox_xyxy": [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height]})
    return result


def summarize_detections(per_image: list[tuple[list[dict], list[dict]]], names: list[str], eval_conf: float) -> dict:
    """mAP50 por classe (todas as confianças) + TP/FP/FN/precisão/recall em ``eval_conf``.

    Parâmetros: lista de ``(sugestões, caixas humanas)`` por foto.
    Retorna ``{"map50", "per_class": {...}, "images"}``; mAP50 = média das classes
    com caixa humana (classe sem caixa = não medida, fora da média). Cada classe
    traz também ``f1``, ``support_images`` (recortes com a classe) e ``status``
    (``nao_avaliada`` quando não há caixa humana: suporte zero, não nota zero).
    """
    from .detection_metrics import add_f1_and_status
    from .model_gate import average_precision

    scored = {index: [] for index in range(len(names))}
    positives = {index: 0 for index in range(len(names))}
    support_images: dict[int, int] = {}
    errors = []
    for predictions, truth in per_image:
        for class_id in {box["class_id"] for box in truth}:
            support_images[class_id] = support_images.get(class_id, 0) + 1
        hits, _ = match_predictions(predictions, truth)
        for prediction, hit in zip(predictions, hits):
            scored.setdefault(prediction["class_id"], []).append((prediction["confidence"], hit))
        for box in truth:
            positives[box["class_id"]] = positives.get(box["class_id"], 0) + 1
        errors.append(image_errors([p for p in predictions if p["confidence"] >= eval_conf], truth))
    report = per_class_report(errors, names)
    aps = []
    for index, name in enumerate(names):
        ap = average_precision(scored.get(index, []), positives.get(index, 0))
        report[name]["ap50"] = ap
        if ap is not None:
            aps.append(ap)
    add_f1_and_status(report, support_images, names)
    return {"map50": round(sum(aps) / len(aps), 4) if aps else None, "per_class": report, "images": len(per_image),
            "eval_conf": eval_conf}


def evaluate_adapter(adapter, dataset: TileDataset, batch: int = 4, eval_conf: float = 0.25,
                     max_batches: int | None = None, keep_detections: bool = False) -> dict:
    """Avalia um modelo num split, com a mesma régua do ``compare-models`` (IoU 0,5, por classe).

    Retorna o dict de ``summarize_detections``; com ``keep_detections`` também
    ``"_detections"`` (lista por foto, usada por ``complementarity``).

    ATENCAO: avalia os recortes do dataset como estão (não refaz o fatiamento da
    foto inteira). Para comparar com o modelo em uso nas fotos de 20 MP, use o
    ``compare-models`` com o ``.pt`` do YOLO26 gerado aqui.
    """
    per_image, kept = [], []
    for images, targets, metas in iterate_batches(dataset, batch, shuffle=False, seed=0, max_batches=max_batches):
        raw = adapter.predict_raw(images.to(adapter.device))
        for position, (target, meta) in enumerate(zip(targets, metas)):
            predictions = detections_from_raw(raw.boxes[position].cpu(), raw.logits[position].cpu(), meta["size"])
            truth = truth_from_target(target, meta["size"])
            per_image.append((predictions, truth))
            kept.append({"key": meta["key"], "boxes": raw.boxes[position].cpu(),
                         "probs": raw.logits[position].cpu().sigmoid(), "size": meta["size"]})
    result = summarize_detections(per_image, dataset.names, eval_conf)
    if keep_detections:
        result["_detections"] = per_image
        result["_raw"] = kept
    return result


def complementarity(results: dict[str, dict], eval_conf: float) -> dict:
    """Quanto os modelos erram coisas diferentes (vale a pena juntá-los?).

    Parâmetros: ``{modelo: saída de evaluate_adapter(..., keep_detections=True)}``.
    Retorna por modelo o recall em ``eval_conf``; o recall da união (oráculo: caixa
    humana achada por pelo menos um) e a concordância de Jaccard entre os conjuntos
    de caixas achadas, por par de modelos. União bem acima do melhor modelo e
    Jaccard baixo = erros complementares, o caso em que DML/ensemble podem ajudar.
    """
    found: dict[str, set] = {}
    total = 0
    for name, result in results.items():
        hits_set = set()
        total = 0
        for image_index, (predictions, truth) in enumerate(result["_detections"]):
            confident = [p for p in predictions if p["confidence"] >= eval_conf]
            _, used = match_predictions(confident, truth)
            hits_set.update((image_index, position) for position in used)
            total += len(truth)
        found[name] = hits_set
    names = list(found)
    union = set().union(*found.values()) if found else set()
    pairs = {}
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            a, b = found[names[first]], found[names[second]]
            pairs[f"{names[first]}|{names[second]}"] = round(len(a & b) / len(a | b), 4) if a | b else None
    return {"human_boxes": total,
            "recall": {name: round(len(value) / total, 4) if total else None for name, value in found.items()},
            "union_recall": round(len(union) / total, 4) if total else None,
            "jaccard": pairs, "eval_conf": eval_conf}


def evaluate_ensemble(results: dict[str, dict], names: list[str], eval_conf: float, fuse_iou: float = 0.55) -> dict:
    """Ensemble por fusão das previsões dos modelos (mesma fusão da consistência, τ = 0,001).

    É o "professor" do modo ``distill``: se o ensemble não supera o melhor modelo
    sozinho, destilar não tem o que transferir.
    """
    import torch

    models = list(results)
    count = len(results[models[0]]["_raw"])
    per_image = []
    for index in range(count):
        peers = [(results[name]["_raw"][index]["boxes"], results[name]["_raw"][index]["probs"]) for name in models]
        boxes, probs, weights = fuse_peer_predictions(peers, tau=0.001, fuse_iou=fuse_iou, max_per_peer=100)
        size = results[models[0]]["_raw"][index]["size"]
        if boxes.shape[0]:
            # Volta a cxcywh para reutilizar detections_from_raw.
            cxcywh = torch.cat([(boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]], dim=-1)
            predictions = detections_from_raw(cxcywh, probs, size, probabilities=True)
        else:
            predictions = []
        per_image.append((predictions, results[models[0]]["_detections"][index][1]))
    return summarize_detections(per_image, names, eval_conf)


# ---------------------------------------------------------------------------
# Cache de previsões dos pares (modo cached e rede local)
# ---------------------------------------------------------------------------

def build_peer_cache(adapter, dataset: TileDataset, output: Path, batch: int = 4, max_per_image: int = 100,
                     min_conf: float = 0.05, max_batches: int | None = None) -> Path:
    """Grava as previsões de um modelo em todas as fotos de um split (sem espelho).

    Parâmetros:
        adapter: modelo já treinado (rodada anterior).
        dataset: normalmente o split ``train`` SEM ``augment``.
        output: pasta do cache; grava ``<modelo>.npz`` e ``<modelo>.json``.
        max_per_image: previsões mais confiantes guardadas por foto.
        min_conf: confiança mínima guardada (abaixo de τ não serve para a fusão).

    Formato do ``.npz`` (sem pickle): ``keys`` (texto), ``offsets`` (int64, N+1),
    ``boxes`` (float16 [M,4] cxcywh 0-1), ``probs`` (float16 [M,K]).
    Tamanho típico: 1.234 recortes × 100 × (4 + K) × 2 bytes ≈ 3 MB com K = 9.

    Chamada por: comando ``dml-cache`` e ``dml_lan`` (cada trabalhador ao fim da rodada).
    """
    import numpy as np

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    keys, offsets, boxes_all, probs_all = [], [0], [], []
    for images, _, metas in iterate_batches(dataset, batch, shuffle=False, seed=0, max_batches=max_batches):
        raw = adapter.predict_raw(images.to(adapter.device))
        for position, meta in enumerate(metas):
            probs = raw.logits[position].float().sigmoid().cpu()
            scores = probs.max(-1).values
            order = scores.argsort(descending=True)[:max_per_image]
            order = order[scores[order] >= min_conf]
            keys.append(meta["key"])
            boxes_all.append(raw.boxes[position][order].float().cpu().numpy().astype(np.float16))
            probs_all.append(probs[order].numpy().astype(np.float16))
            offsets.append(offsets[-1] + int(order.numel()))
    path = output / f"{adapter.name}.npz"
    np.savez_compressed(path, keys=np.array(keys), offsets=np.array(offsets, dtype=np.int64),
                        boxes=np.concatenate(boxes_all) if boxes_all else np.zeros((0, 4), np.float16),
                        probs=np.concatenate(probs_all) if probs_all else np.zeros((0, adapter.num_classes), np.float16))
    write_json(output / f"{adapter.name}.json", {"model": adapter.name, "family": adapter.family, "images": len(keys),
                                                 "predictions": offsets[-1], "split": dataset.split,
                                                 "created_at": datetime.now(timezone.utc).isoformat()})
    return path


class PeerCache:
    """Previsões gravadas de um par, consultadas por chave de foto (com o espelho do lote).

    Parâmetros: ``path`` do ``.npz`` gerado por ``build_peer_cache``.
    Erros: ValueError se o arquivo tiver objetos pickle (``allow_pickle=False``).
    """

    def __init__(self, path: Path):
        import numpy as np

        data = np.load(Path(path), allow_pickle=False)
        self.name = Path(path).stem
        self.offsets = data["offsets"]
        self.boxes = data["boxes"]
        self.probs = data["probs"]
        self.index = {str(key): position for position, key in enumerate(data["keys"].tolist())}

    def lookup(self, key: str, hflip: bool = False, vflip: bool = False):
        """``(boxes cxcywh, probs)`` da foto, espelhados como o lote; ``None`` se a foto não está no cache."""
        import torch

        position = self.index.get(key)
        if position is None:
            return None
        start, end = int(self.offsets[position]), int(self.offsets[position + 1])
        boxes = torch.from_numpy(self.boxes[start:end].astype("float32"))
        probs = torch.from_numpy(self.probs[start:end].astype("float32"))
        if hflip and boxes.numel():
            boxes[:, 0] = 1.0 - boxes[:, 0]
        if vflip and boxes.numel():
            boxes[:, 1] = 1.0 - boxes[:, 1]
        return boxes, probs


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def _merge(base: dict, extra: dict | None) -> dict:
    """Mescla raso por seção (dicts de dicts), sem alterar ``base``."""
    result = json.loads(json.dumps(base))
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key].update(value)
        else:
            result[key] = value
    return result


def load_dml_config(path: Path | None) -> dict:
    """Lê o ``configs/dml.json`` e completa com ``DEFAULT_SETTINGS`` e ``FAMILY_DEFAULTS``.

    Erros: ValueError se não houver ``models`` ou houver nome repetido.
    """
    # utf-8-sig: o PowerShell 5.1 grava JSON com BOM, e json.loads recusa o BOM.
    data = json.loads(Path(path).read_text(encoding="utf-8-sig")) if path else {}
    settings = _merge(DEFAULT_SETTINGS, data)
    models = data.get("models") or []
    if not models:
        raise ValueError("O configs/dml.json precisa de uma lista 'models'.")
    names = [model["name"] for model in models]
    if len(set(names)) != len(names):
        raise ValueError("Nomes de modelo repetidos no configs/dml.json.")
    settings["models"] = [{**FAMILY_DEFAULTS.get(model.get("family"), {}), **model} for model in models]
    return settings


# ---------------------------------------------------------------------------
# Registro por época do DML (influência dos pares, cooperação, tempos, decisões)
# ---------------------------------------------------------------------------

# Motivos de uma imagem não receber orientação útil dos pares (chaves de "no_guidance").
NO_GUIDANCE_REASONS = ("peso_zero_na_epoca", "sem_pares", "pares_sem_previsao_acima_de_tau",
                       "nada_coincide_com_caixa_humana", "sem_casamento_com_o_aluno")


def _new_epoch_stats(peers: list[str]) -> dict:
    """Acumuladores de UM aluno numa época (ver ``_epoch_line`` para o significado de cada um).

    Parâmetros: ``peers`` = nomes dos pares que podem orientar este aluno.
    """
    from collections import Counter

    return {"images": 0, "images_with_peers": 0, "images_with_useful_guidance": 0,
            "no_guidance": {reason: 0 for reason in NO_GUIDANCE_REASONS},
            "fused_targets": 0, "anchored_targets": 0, "matched_targets_final_head": 0,
            "matched_targets_any_head": 0, "head_matches": 0, "heads": 0,
            "peers": {peer: {"received": Counter(), "used": Counter()} for peer in peers},
            "det": 0.0, "cons_raw": 0.0, "cons_applied": 0.0, "scale": 0.0, "batches": 0,
            "optimizer_steps": 0, "skipped_nonfinite": 0, "grad_norms": [],
            "seconds": {"forward_loss": 0.0, "consistency": 0.0, "backward_step": 0.0,
                        "validation": 0.0, "checkpoint": 0.0}}


def _consistency_for_image(name: str, index: int, raws: list, peer_views: dict, truth_xyxy, cons: dict,
                           record: dict) -> list:
    """Perdas de consistência de UMA imagem do aluno ``name`` e o registro da cooperação.

    Denominadores (corrigem a ambiguidade do registro antigo, que somava casamentos
    por imagem E por cabeça): ``fused_targets``/``anchored_targets`` contam cada alvo
    uma vez por imagem; ``matched_targets_final_head`` conta alvos distintos casados
    pela cabeça de inferência (a última); ``matched_targets_any_head`` conta alvos
    distintos casados por pelo menos uma cabeça; ``head_matches`` mantém a soma antiga
    (alvo × cabeça), só para referência.

    Retorna: lista de perdas (uma por cabeça) para a média do lote; vazia sem alvo.
    """
    from collections import Counter

    record["images"] += 1
    named = [(other, views[index]) for other, views in peer_views.items()
             if other != name and views[index] is not None]
    if not named:
        record["no_guidance"]["sem_pares"] += 1
        return []
    record["images_with_peers"] += 1
    boxes, probs, weights, info = fuse_peer_predictions([view for _, view in named], float(cons["tau"]),
                                                        float(cons["fuse_iou"]), int(cons["max_per_peer"]),
                                                        details=True)
    # Sugestões recebidas: previsões de cada par acima de τ (depois do NMS interno), por classe.
    for (peer, _), classes in zip(named, info["received"]):
        record["peers"].setdefault(peer, {"received": Counter(), "used": Counter()})
        for class_id in classes:
            record["peers"][peer]["received"][class_id] += 1
    record["fused_targets"] += int(boxes.shape[0])
    if boxes.shape[0] == 0:
        record["no_guidance"]["pares_sem_previsao_acima_de_tau"] += 1
        return []
    members = info["members"]
    if cons["anchor"] == "gt":
        boxes, probs, weights, keep = anchor_to_truth(boxes, probs, weights, truth_xyxy, float(cons["anchor_iou"]),
                                                      return_mask=True)
        members = [member for member, kept in zip(members, keep.tolist()) if kept]
    record["anchored_targets"] += int(boxes.shape[0])
    if boxes.shape[0] == 0:
        record["no_guidance"]["nada_coincide_com_caixa_humana"] += 1
        return []
    losses, matched_any, matched_final = [], set(), set()
    for head, raw in enumerate(raws):
        loss, stats = consistency_loss(raw.boxes[index], raw.logits[index], boxes, probs, weights,
                                       float(cons["temperature"]), float(cons["box_weight"]),
                                       float(cons["negative_weight"]))
        losses.append(loss)
        record["head_matches"] += stats["matched"]
        matched_any.update(stats["matched_targets"])
        if head == len(raws) - 1:
            matched_final = set(stats["matched_targets"])
    record["matched_targets_final_head"] += len(matched_final)
    record["matched_targets_any_head"] += len(matched_any)
    if matched_any:
        record["images_with_useful_guidance"] += 1
    else:
        record["no_guidance"]["sem_casamento_com_o_aluno"] += 1
    # Aproveitadas: previsões de par que formaram um alvo casado pela cabeça de inferência.
    for position, member in enumerate(members):
        if position in matched_final:
            for peer_index, class_id in member:
                record["peers"][named[peer_index][0]]["used"][class_id] += 1
    return losses


def _epoch_line(epoch: int, name: str, mode: str, ramp: float, choice: dict | None, record: dict, metrics: dict,
                state, decision: dict, shared: dict, train_wall: float, students: int) -> dict:
    """Linha do ``history.jsonl`` de um aluno numa época (registros pedidos para o DML).

    Blocos:
    - ``influence``: perda de detecção, consistência bruta e termo REALMENTE aplicado
      (λ_rampa × multiplicador × s_i × consistência), com s_i médio. ATENCAO: as perdas
      são as nativas de cada família; não compare ``det_loss`` entre arquiteturas.
    - ``cooperation``: sugestões recebidas/aproveitadas/descartadas por par e classe,
      imagens sem orientação útil e o motivo (ver ``_consistency_for_image``).
    - ``val_per_class``: qualidade por classe na validação (classe sem caixa = não avaliada).
    - ``rl``: decisão do UCB (``dml_controller.decision_record``).
    - ``optimizer``: passos efetivos, passos pulados por valor não finito, norma do gradiente.
    - ``seconds.model``: tempo PRÓPRIO deste modelo; ``seconds.shared``: leitura dos dados,
      professores e cache dos pares (comuns); ``seconds.train_wall_shared``: relógio do
      laço de treino da época, comum a todos os alunos (no modo online inclui os outros).
    """
    from .detection_metrics import _quantiles

    batches = max(record["batches"], 1)
    per_peer = {}
    for peer, counts in record["peers"].items():
        received, used = counts["received"], counts["used"]
        per_peer[peer] = {"received": {metrics_name(metrics, k): v for k, v in sorted(received.items())},
                          "used": {metrics_name(metrics, k): v for k, v in sorted(used.items())},
                          "discarded": {metrics_name(metrics, k): received[k] - used.get(k, 0)
                                        for k in sorted(received)},
                          "received_total": sum(received.values()), "used_total": sum(used.values())}
    own = record["seconds"]
    per_class = {key: {field: value.get(field) for field in ("status", "support_images", "human_boxes", "tp", "fp",
                                                             "fn", "precision", "recall", "f1", "ap50")}
                 for key, value in metrics["per_class"].items()}
    return {
        "record": "epoch", "epoch": epoch, "model": name, "mode": mode,
        "lambda_ramp": round(ramp, 6), "multiplier": choice["multiplier"] if choice else 1.0,
        "det_loss": round(record["det"] / batches, 6), "cons_loss": round(record["cons_raw"] / batches, 6),
        "influence": {"det_loss": round(record["det"] / batches, 6),
                      "consistency_raw": round(record["cons_raw"] / batches, 6),
                      "consistency_applied": round(record["cons_applied"] / batches, 6),
                      "weight": round(ramp * (choice["multiplier"] if choice else 1.0), 6),
                      "scale_s_mean": round(record["scale"] / batches, 6),
                      "note": "perdas nativas da família; não comparar entre arquiteturas"},
        # Compatível com o registro antigo, agora sem dupla contagem: alvos distintos por imagem.
        "matched": record["matched_targets_final_head"], "targets": record["anchored_targets"],
        "cooperation": {key: record[key] for key in ("images", "images_with_peers", "images_with_useful_guidance",
                                                     "no_guidance", "fused_targets", "anchored_targets",
                                                     "matched_targets_final_head", "matched_targets_any_head",
                                                     "head_matches", "heads")} | {"per_peer": per_peer},
        "val_map50": metrics["map50"], "best_val_map50": state.best_map, "best_epoch": state.best_epoch,
        "epochs_since_best": None if state.best_epoch is None else epoch - state.best_epoch,
        "val_per_class": per_class, "reward": decision.get("reward"), "rl": decision,
        "optimizer": {"steps": record["optimizer_steps"], "skipped_nonfinite": record["skipped_nonfinite"],
                      "lr": [round(group["lr"], 10) for group in state.optimizer.param_groups][:4],
                      "grad_norm_before_clip": _quantiles([v for v in record["grad_norms"] if math.isfinite(v)])},
        "seconds": {"model": {key: round(value, 2) for key, value in own.items()} | {
                        "total_own": round(sum(own.values()), 2)},
                    "shared": {key: round(value, 2) for key, value in shared.items()},
                    "train_wall_shared": round(train_wall, 2), "students_in_process": students},
    }


def metrics_name(metrics: dict, class_id: int) -> str:
    """Nome da classe ``class_id`` pela ordem do ``per_class`` da validação (ou o número)."""
    names = [key for key in metrics.get("per_class", {}) if key != "_total"]
    return names[class_id] if 0 <= class_id < len(names) else str(class_id)


@dataclass
class _Student:
    """Estado de treino de um aluno: adaptador, otimizador, médias móveis e controlador RL."""

    spec: dict
    adapter: object
    optimizer: object
    controller: object | None
    ema_det: float | None = None
    ema_cons: float | None = None
    best_map: float | None = None
    last_map: float | None = None
    best_epoch: int | None = None


def train_mutual(config_root: Path, data_path: Path, settings: dict, mode: str = "online",
                 student: str | None = None, peer_cache: Path | None = None, max_batches: int | None = None,
                 val_max_batches: int | None = None, run: Path | None = None) -> Path:
    """ENTRADA de ``dml-train``: treina por aprendizado mútuo e grava a pasta do experimento.

    Parâmetros:
        config_root: pasta ``sistema_ia`` (onde fica ``runs/``).
        data_path: ``dataset.yaml`` (recortes de treino/val; o test não é usado).
        settings: saída de ``load_dml_config`` (já com overrides da CLI).
        mode: ``online``, ``distill`` ou ``cached`` (ver cabeçalho).
        student: modelo que aprende nos modos ``distill``/``cached`` (obrigatório neles).
        peer_cache: pasta com ``<par>.npz`` (modo ``cached``).
        max_batches, val_max_batches: limites por época (testes rápidos).
        run: pasta já criada (usada pelo ``dml_lan``); ``None`` cria ``runs/dml_*``.

    Retorna: a pasta do experimento.

    Erros:
        ValueError: modo desconhecido, aluno ausente, cache faltando.

    Etapas: 1. dados e modelos; 2. otimizadores e controladores; 3. por época,
    o controlador escolhe ``a_i``; 4. por lote, perda de detecção + consistência
    contra os pares e passo do otimizador; 5. validação, recompensa do RL;
    6. checkpoint ``last`` e ``best`` (mAP50 de val); 7. resumo.
    """
    import torch

    from .detector_adapters import build_adapter
    from .dml_controller import DiscountedUCB, decision_record

    if mode not in {"online", "distill", "cached", "solo"}:
        raise ValueError(f"Modo desconhecido: {mode}.")
    train_cfg, cons = settings["train"], settings["consistency"]
    rl_cfg = settings["rl"]
    epochs, batch = int(train_cfg["epochs"]), int(train_cfg["batch"])
    imgsz, seed = int(settings["imgsz"]), int(train_cfg["seed"])
    device = train_cfg.get("device") or "cpu"
    if train_cfg.get("threads"):
        torch.set_num_threads(int(train_cfg["threads"]))
    torch.manual_seed(seed)

    # 1. Dados e modelos.
    train_set = TileDataset(data_path, "train", imgsz, augment=True, hflip=train_cfg["hflip"],
                            vflip=train_cfg["vflip"], seed=seed)
    val_set = TileDataset(data_path, "val", imgsz)
    names = train_set.names
    specs = {model["name"]: model for model in settings["models"]}
    if mode != "online":
        if student not in specs:
            raise ValueError(f"Informe --student com um destes modelos: {', '.join(specs)}.")
        student_names = [student]
    else:
        student_names = list(specs)
    run = run or make_run_directory(Path(config_root), "dml")
    write_json(run / "settings.json", {"mode": mode, "student": student, "data": str(data_path), **settings,
                                       "peer_cache": str(peer_cache) if peer_cache else None,
                                       "started_at": datetime.now(timezone.utc).isoformat()})
    students: dict[str, _Student] = {}
    for name in student_names:
        spec = specs[name]
        adapter = build_adapter(spec, names, imgsz, epochs=int(train_cfg.get("schedule_epochs") or epochs)).to(device)
        # 2. AdamW por modelo (cada família com sua taxa) e um controlador RL por aluno.
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(spec["lr"]), weight_decay=float(train_cfg["weight_decay"]))
        controller = DiscountedUCB(rl_cfg["arms"], rl_cfg["gamma"], rl_cfg["exploration"]) if rl_cfg.get("enabled") else None
        students[name] = _Student(spec, adapter, optimizer, controller)
        # Medida antes do treino: a recompensa da 1.ª época já é um Δ real, não 0 por falta de base.
        students[name].last_map = evaluate_adapter(adapter, val_set, batch, float(train_cfg["eval_conf"]),
                                                   max_batches=val_max_batches)["map50"]
    teachers = {}
    if mode == "distill":
        for name, spec in specs.items():
            if name != student:
                teachers[name] = build_adapter(spec, names, imgsz, epochs=epochs).to(device)
    caches = []
    if mode == "cached":
        if not peer_cache:
            raise ValueError("O modo cached precisa de --peer-cache (pasta gerada pelo dml-cache).")
        # Só os pares configurados: um .npz antigo ou de outro experimento na pasta não entra na fusão.
        expected = [name for name in specs if name != student]
        missing = [name for name in expected if not (Path(peer_cache) / f"{name}.npz").exists()]
        if missing:
            raise ValueError(f"Faltam caches de par em {peer_cache}: {', '.join(m + '.npz' for m in missing)}.")
        caches = [PeerCache(Path(peer_cache) / f"{name}.npz") for name in expected]

    history_path = run / "history.jsonl"
    with history_path.open("a", encoding="utf-8") as handle:
        for name, state in students.items():
            handle.write(json.dumps({"record": "baseline", "epoch": -1, "model": name, "mode": mode,
                                     "val_map50": state.last_map}) + "\n")
    clock = time.perf_counter
    for epoch in range(epochs):
        # 3. Peso da época: rampa × multiplicador escolhido pelo RL (1.0 sem RL). select_details
        #    devolve a mesma escolha de select() com o raciocínio de cada braço (registro do UCB).
        ramp = ramp_weight(epoch, int(cons["ramp_epochs"]), float(cons["lambda"]))
        choices = {name: state.controller.select_details() if state.controller else None
                   for name, state in students.items()}
        peer_names = [name for name in students] + list(teachers) + [cache.name for cache in caches]
        stats = {name: _new_epoch_stats([other for other in peer_names if other != name]) for name in students}
        # Tempos comuns a todos os alunos (não pertencem a nenhum modelo sozinho).
        shared = {"data": 0.0, "teachers": 0.0, "peer_cache": 0.0}
        epoch_start = fetch_start = clock()
        for images, targets, metas in iterate_batches(train_set, batch, shuffle=True, seed=seed + epoch,
                                                      max_batches=max_batches):
            shared["data"] += clock() - fetch_start
            images = images.to(device)
            truth_xyxy = [cxcywh_to_xyxy(target["boxes"]).clamp(0, 1) for target in targets]
            # 4a. Passo para frente de cada aluno (mesmo lote, mesmo espelho), cronometrado por modelo.
            outputs = {}
            for name, state in students.items():
                started = clock()
                outputs[name] = state.adapter.forward_train(images, targets)
                stats[name]["seconds"]["forward_loss"] += clock() - started
            # 4b. Visão dos pares por imagem, sem gradiente.
            peer_views: dict[str, list] = {}
            for name, (_, raws) in outputs.items():
                peer_views[name] = [peer_view(raws, index) for index in range(len(targets))]
            started = clock()
            for name, adapter in teachers.items():
                raw = adapter.predict_raw(images)
                peer_views[name] = [(raw.boxes[index], raw.logits[index].sigmoid()) for index in range(len(targets))]
            shared["teachers"] += clock() - started
            started = clock()
            for cache in caches:
                views = []
                for meta in metas:
                    found = cache.lookup(meta["key"], meta["hflip"], meta["vflip"])
                    views.append(tuple(value.to(device) for value in found) if found else None)
                peer_views[cache.name] = views
            shared["peer_cache"] += clock() - started
            for name, state in students.items():
                record = stats[name]
                det, raws = outputs[name]
                multiplier = choices[name]["multiplier"] if choices[name] else 1.0
                weight = ramp * multiplier
                cons_total = det * 0.0
                record["heads"] = len(raws)
                started = clock()
                if weight > 0:
                    # 4c. Consistência por imagem contra os outros. Cada alvo fundido conta UMA vez
                    #     por imagem (não por cabeça): ver _consistency_for_image.
                    losses = []
                    for index in range(len(targets)):
                        losses.extend(_consistency_for_image(
                            name, index, raws, peer_views, truth_xyxy[index].to(device), cons, record))
                    if losses:
                        cons_total = torch.stack(losses).mean()
                else:
                    record["images"] += len(targets)
                    record["no_guidance"]["peso_zero_na_epoca"] += len(targets)
                record["seconds"]["consistency"] += clock() - started
                # 4d. Escala s_i = EMA(det)/EMA(cons): λ sem unidade (ver cabeçalho).
                decay = float(cons["ema"])
                det_value, cons_value = float(det.detach()), float(cons_total.detach())
                state.ema_det = det_value if state.ema_det is None else decay * state.ema_det + (1 - decay) * det_value
                if cons_value > 0:
                    state.ema_cons = cons_value if state.ema_cons is None else decay * state.ema_cons + (1 - decay) * cons_value
                scale = (state.ema_det / state.ema_cons) if state.ema_cons else 0.0
                total = det + weight * scale * cons_total
                # 4e. Retropropagação e passo, cronometrados; gradiente não finito não atualiza os pesos.
                started = clock()
                state.optimizer.zero_grad(set_to_none=True)
                total.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(state.adapter.parameters(), float(state.spec["grad_clip"])))
                if math.isfinite(grad_norm) and math.isfinite(float(total.detach())):
                    state.optimizer.step()
                    record["optimizer_steps"] += 1
                else:
                    # ATENCAO: passo pulado de propósito (NaN/Inf apagaria os pesos); fica contado.
                    record["skipped_nonfinite"] += 1
                    state.optimizer.zero_grad(set_to_none=True)
                record["seconds"]["backward_step"] += clock() - started
                record["grad_norms"].append(grad_norm)
                record["det"] += det_value
                record["cons_raw"] += cons_value
                record["cons_applied"] += weight * scale * cons_value
                record["scale"] += scale
                record["batches"] += 1
            fetch_start = clock()
        train_wall = clock() - epoch_start
        # 5. Validação, decisão do RL e registro da época (um bloco por aluno).
        for name, state in students.items():
            record = stats[name]
            state.adapter.on_epoch_end()
            started = clock()
            metrics = evaluate_adapter(state.adapter, val_set, batch, float(train_cfg["eval_conf"]),
                                       max_batches=val_max_batches)
            record["seconds"]["validation"] = clock() - started
            current = metrics["map50"]
            weight = ramp * (choices[name]["multiplier"] if choices[name] else 1.0)
            decision = decision_record(state.controller, choices[name], state.last_map, current, weight, step=epoch)
            state.last_map = current
            # 6. Checkpoints no formato nativo (o YOLO26 .pt abre no compare-models).
            started = clock()
            weights_dir = run / "weights" / name
            state.adapter.save(weights_dir / "last", names)
            improved = current is not None and (state.best_map is None or current > state.best_map)
            if improved:
                state.best_map, state.best_epoch = current, epoch
                state.adapter.save(weights_dir / "best", names)
                write_json(weights_dir / "best_metrics.json", {"epoch": epoch, "criterion": "maior val_map50", **metrics})
            record["seconds"]["checkpoint"] = clock() - started
            line = _epoch_line(epoch, name, mode, ramp, choices[name], record, metrics, state, decision, shared,
                               train_wall, len(students))
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")
        write_json(run / "controller.json", {name: state.controller.to_dict() for name, state in students.items()
                                             if state.controller})
    # 7. Resumo do experimento.
    summary = {"mode": mode, "epochs": epochs, "models": {}, "finished_at": datetime.now(timezone.utc).isoformat(),
               "next_step": "Comparar no split test com dml-evaluate (e compare-models para o .pt do YOLO26) "
                            "contra o modelo em uso; adotar só com ganho por classe."}
    for name, state in students.items():
        summary["models"][name] = {"best_val_map50": state.best_map, "best_epoch": state.best_epoch,
                                   "last_val_map50": state.last_map,
                                   "rl": state.controller.summary() if state.controller else None}
    write_json(run / "summary.json", summary)
    return run


def evaluate_models(config_root: Path, data_path: Path, settings: dict, split: str = "test",
                    max_batches: int | None = None) -> Path:
    """ENTRADA de ``dml-evaluate``: mede cada modelo, a complementaridade e o ensemble num split.

    Usa os ``weights`` de cada modelo do ``configs/dml.json`` (aponte para os
    ``best`` do treino). Grava ``runs/dml_eval_*/evaluation.json``.
    """
    from .detector_adapters import build_adapter

    imgsz = int(settings["imgsz"])
    dataset = TileDataset(data_path, split, imgsz)
    eval_conf = float(settings["train"]["eval_conf"])
    device = settings["train"].get("device") or "cpu"
    results = {}
    for spec in settings["models"]:
        adapter = build_adapter(spec, dataset.names, imgsz).to(device)
        results[spec["name"]] = evaluate_adapter(adapter, dataset, int(settings["train"]["batch"]), eval_conf,
                                                 max_batches=max_batches, keep_detections=True)
    report = {"split": split, "data": str(data_path), "models": {},
              "complementarity": complementarity(results, eval_conf),
              "ensemble": evaluate_ensemble(results, dataset.names, eval_conf, float(settings["consistency"]["fuse_iou"]))}
    for name, result in results.items():
        report["models"][name] = {key: value for key, value in result.items() if not key.startswith("_")}
    run = make_run_directory(Path(config_root), "dml_eval")
    write_json(run / "evaluation.json", report)
    return run


# ---------------------------------------------------------------------------
# Planejamento: dependências e tempo medido
# ---------------------------------------------------------------------------

def plan_dml(data_path: Path, settings: dict, batches: int = 3) -> dict:
    """ENTRADA de ``dml-plan``: mede o tempo real de um passo de treino de cada modelo nesta máquina.

    Para cada modelo: monta o adaptador, roda ``batches`` passos (frente + trás +
    otimizador) em lotes reais do treino e extrapola para uma época e para o
    treino inteiro. Também estima o modo ``online`` (os três por lote, soma) e o
    ``cached`` por rodada (um por vez + inferência do cache).

    Retorna dict com ``dependencies``, ``per_model`` e ``estimates`` (horas).
    O primeiro passo é descartado (aquecimento) quando ``batches`` ≥ 2.
    """
    import torch

    from .detector_adapters import build_adapter, dependency_report

    deps = dependency_report()
    train_cfg = settings["train"]
    imgsz, batch, epochs = int(settings["imgsz"]), int(train_cfg["batch"]), int(train_cfg["epochs"])
    if train_cfg.get("threads"):
        torch.set_num_threads(int(train_cfg["threads"]))
    dataset = TileDataset(data_path, "train", imgsz, augment=True)
    steps_per_epoch = math.ceil(len(dataset) / batch)
    per_model = {}
    for spec in settings["models"]:
        family = spec.get("family")
        if not deps.get(family, {}).get("installed"):
            per_model[spec["name"]] = {"error": f"pacote {deps.get(family, {}).get('package')} não instalado"}
            continue
        try:
            adapter = build_adapter(spec, dataset.names, imgsz, epochs=epochs).to(train_cfg.get("device") or "cpu")
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(spec["lr"]))
            timings, infer = [], []
            for images, targets, _ in iterate_batches(dataset, batch, shuffle=True, seed=0, max_batches=batches):
                images = images.to(adapter.device)
                start = time.time()
                det, _ = adapter.forward_train(images, targets)
                optimizer.zero_grad(set_to_none=True)
                det.backward()
                optimizer.step()
                timings.append(time.time() - start)
                start = time.time()
                adapter.predict_raw(images)
                infer.append(time.time() - start)
            used = timings[1:] if len(timings) > 1 else timings
            step = sum(used) / len(used)
            inference = sum(infer[1:] if len(infer) > 1 else infer) / max(len(infer), 1)
            per_model[spec["name"]] = {
                "family": family, "parameters_m": round(sum(p.numel() for p in adapter.parameters()) / 1e6, 2),
                "seconds_per_step": round(step, 3), "seconds_per_inference_batch": round(inference, 3),
                "hours_per_epoch": round(step * steps_per_epoch / 3600, 3),
                "hours_total": round(step * steps_per_epoch * epochs / 3600, 2)}
        except Exception as error:  # noqa: BLE001 - o plano relata o erro de cada modelo e segue
            per_model[spec["name"]] = {"error": f"{type(error).__name__}: {error}"}
    measured = [value for value in per_model.values() if "hours_per_epoch" in value]
    estimates = {}
    if measured:
        online_epoch = sum(value["hours_per_epoch"] for value in measured)
        cache_hours = sum(value["seconds_per_inference_batch"] for value in measured) * steps_per_epoch / 3600
        estimates = {"steps_per_epoch": steps_per_epoch, "epochs": epochs,
                     "online_hours": round(online_epoch * epochs, 2),
                     "sequential_individual_hours": round(sum(value["hours_total"] for value in measured), 2),
                     "lan_parallel_hours": round(max(value["hours_total"] for value in measured) + cache_hours, 2),
                     "note": "online = os três no mesmo processo; lan_parallel = um modelo por máquina iguais a esta, "
                             "trocando previsões por rodada (dml_lan). Não inclui a validação de cada época."}
    return {"dependencies": deps, "train_images": len(dataset), "batch": batch, "imgsz": imgsz,
            "per_model": per_model, "estimates": estimates, "measured_at": datetime.now(timezone.utc).isoformat()}
