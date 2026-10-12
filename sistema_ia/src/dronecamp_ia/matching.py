"""Pareamento entre caixas do modelo e caixas humanas, e contagem por classe.

Função no projeto: é a régua comum dos módulos de aprendizado. Diz, para cada
sugestão do modelo, se ela reencontra uma caixa aprovada por um humano (mesma
classe e IoU >= 0,5) e conta acertos (TP), falsos positivos (FP) e omissões
(FN) por classe.

Quem usa:
- ``calibration.py``: rótulo 1/0 de cada sugestão para treinar o calibrador.
- ``active_learning.py``: recompensa do bandit (quanto o modelo errou na foto).
- ``model_gate.py``: precisão, recall, FP e omissões por classe na comparação.
- ``autolabel.py``: mede a busca aberta YOLOE (``evaluate-autolabel``) com a mesma régua.

Quando mexer: o limiar de IoU (``IOU_MATCH``) ou a regra de pareamento.

O que o arquivo contém (todas públicas):
- ``yolo_label_boxes``: lê um .txt YOLO e devolve caixas em pixels [x1, y1, x2, y2].
- ``match_predictions``: pareamento guloso sugestão x caixa humana (quem é TP).
- ``image_errors``: TP, FP e FN de UMA foto, no total e por classe.
- ``per_class_report``: soma várias fotos e calcula precisão e recall por classe.

Disco: só LÊ arquivos de label YOLO (``yolo_label_boxes``). Não grava nada.

Fórmulas usadas (a IoU vem de ``pilot.box_iou``):
    interseção = max(0, min(x2a, x2b) - max(x1a, x1b)) * max(0, min(y2a, y2b) - max(y1a, y1b))
    união      = área(a) + área(b) - interseção
    IoU        = interseção / união          (0 = disjuntas, 1 = idênticas)
    precisão   = TP / (TP + FP)              (das sugestões, quantas estavam certas)
    recall     = TP / (TP + FN)              (das caixas humanas, quantas o modelo achou)

Onde mexer:
- Para mudar a régua de "acerto" (ex.: IoU 0,3 para defeitos pequenos): altere
  ``IOU_MATCH`` abaixo. ATENCAO: isso muda ao mesmo tempo o calibrador, o bandit,
  a comparação de modelos e a avaliação do YOLOE; compare só números medidos com a
  mesma régua.
- Para aceitar pareamento entre classes diferentes (ex.: contar "localização"
  sem olhar classe): veja o ``continue`` dentro de ``match_predictions``.
- Para acrescentar uma métrica por classe (ex.: F1): veja o ponto EXTENSAO em
  ``per_class_report``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

# Mesma função de IoU usada no dataset piloto; manter UMA implementação evita
# que módulos diferentes meçam a sobreposição de jeitos diferentes.
from .pilot import box_iou

# ---------------------------------------------------------------------------
# Constantes da régua
# ---------------------------------------------------------------------------

# Mesma régua do scripts/compare_pilot_models.py e do mAP50.
# IOU_MATCH: sobreposição mínima (0 a 1, adimensional) para uma sugestão contar
# como a mesma ocorrência de uma caixa humana. Subir (ex.: 0.7) exige caixas mais
# justas e reduz TP; descer (ex.: 0.3) tolera caixas frouxas e aumenta TP.
# ATENCAO: valor compartilhado por calibration.py, active_learning.py,
# model_gate.py e autolabel.py; mudar aqui muda todos eles e torna números antigos
# (runs/gate_*, runs/measure_*, calibradores já treinados) incomparáveis com os novos.
IOU_MATCH = 0.5


# ---------------------------------------------------------------------------
# Leitura de labels YOLO
# ---------------------------------------------------------------------------

def yolo_label_boxes(label_path: Path, width: int, height: int) -> list[dict]:
    """Converte uma label YOLO (cx cy w h normalizados) em caixas [x1, y1, x2, y2] em pixels.

    Serve para comparar a label "verdade" de um dataset com as caixas do modelo,
    que já vêm em pixels (xyxy).

    Parâmetros:
        label_path: caminho do .txt YOLO; cada linha é ``class_id cx cy w h`` com
            cx, cy, w, h normalizados entre 0 e 1 (relativos à largura/altura da foto).
            Linhas vazias são ignoradas.
        width: largura da foto em pixels (ex.: 5472 numa foto de 20 MP).
        height: altura da foto em pixels (ex.: 3648).

    Retorna:
        Lista de dicts ``{"class_id": int, "bbox_xyxy": [x1, y1, x2, y2]}`` em pixels,
        na ordem das linhas do arquivo. Lista vazia para foto negativa (arquivo vazio).

    Erros:
        OSError/FileNotFoundError se o arquivo não existir; ValueError se uma linha
        não tiver exatamente 5 campos numéricos (ex.: label de segmentação/polígono).

    Chamada por: calibration.py, model_gate.py e autolabel.py.
    """
    boxes = []
    # Uma linha por caixa. ATENCAO: lido como "utf-8" (não "utf-8-sig"); um .txt
    # salvo com BOM no Windows quebraria o int(class_id) da primeira linha.
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        # Linha em branco (comum no fim do arquivo) não é caixa.
        if not line.strip():
            continue
        # Formato YOLO de detecção: exatamente 5 campos separados por espaço.
        class_id, cx, cy, w, h = line.split()
        # Desnormaliza: centro e tamanho multiplicados pela largura (x) e altura (y).
        cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
        # Centro/tamanho -> cantos: x1 = cx - w/2, y1 = cy - h/2, x2 = cx + w/2, y2 = cy + h/2.
        boxes.append({"class_id": int(class_id), "bbox_xyxy": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]})
    return boxes


# ---------------------------------------------------------------------------
# Pareamento sugestão x caixa humana (quem é acerto)
# ---------------------------------------------------------------------------

def match_predictions(predictions: list[dict], truth: list[dict], iou: float = IOU_MATCH) -> tuple[list[bool], set[int]]:
    """Pareamento guloso por confiança: cada caixa humana é reencontrada no máximo uma vez.

    Devolve ``(acertos, usadas)``: ``acertos[i]`` diz se ``predictions[i]`` (na ordem
    recebida) é um TP; ``usadas`` são os índices das caixas humanas reencontradas.

    É o mesmo critério do mAP50: as sugestões mais confiantes escolhem primeiro; uma
    caixa humana já reencontrada não pode ser contada de novo (a segunda sugestão
    sobre o mesmo alvo vira FP, ou seja, duplicata).

    Parâmetros:
        predictions: sugestões do modelo, cada uma ``{"class_id": int,
            "bbox_xyxy": [x1, y1, x2, y2] em pixels, "confidence": float 0-1}``.
            Sem ``confidence`` conta como 0.0.
        truth: caixas humanas/aprovadas, ``{"class_id": int, "bbox_xyxy": [...]}``
            no MESMO sistema de coordenadas (pixels da foto original).
        iou: sobreposição mínima para contar como acerto (padrão ``IOU_MATCH`` = 0,5).

    Retorna:
        ``(hits, used)``: ``hits`` é uma lista de bool alinhada com ``predictions``;
        ``used`` é o conjunto de posições de ``truth`` que foram reencontradas
        (as que ficaram fora são as omissões, FN).

    Chamada por: image_errors (aqui), calibration.py e model_gate.py.
    """
    # Começa com tudo "errado"; só vira True quem achar par.
    hits = [False] * len(predictions)
    used: set[int] = set()
    # Índices das sugestões da MAIOR para a MENOR confiança (o sinal de menos
    # inverte a ordenação). O resultado continua indexado pela ordem original.
    order = sorted(range(len(predictions)), key=lambda index: -float(predictions[index].get("confidence", 0.0)))
    for index in order:
        prediction = predictions[index]
        # best = posição da melhor caixa humana até agora; best_iou começa no
        # limiar, então só caixas com IoU >= limiar podem ser escolhidas.
        best, best_iou = None, iou
        # Procura, entre as caixas humanas livres, a de maior IoU com esta sugestão.
        for position, box in enumerate(truth):
            # Pula caixa já usada e caixa de outra classe.
            # EXTENSAO: para parear sem olhar a classe (medir só localização),
            # retire a comparação de class_id desta condição (crie uma função nova
            # para não mudar a régua dos outros módulos).
            if position in used or box["class_id"] != prediction["class_id"]:
                continue
            overlap = box_iou(box["bbox_xyxy"], prediction["bbox_xyxy"])
            # ">=" faz um empate posterior substituir o anterior; na prática só
            # importa em caixas idênticas.
            if overlap >= best_iou:
                best, best_iou = position, overlap
        # Achou par: reserva a caixa humana e marca a sugestão como TP.
        if best is not None:
            used.add(best)
            hits[index] = True
    return hits, used


# ---------------------------------------------------------------------------
# Contagem de erros por foto e relatório por classe
# ---------------------------------------------------------------------------

def image_errors(predictions: list[dict], truth: list[dict]) -> dict:
    """TP, FP e FN de uma foto, no total e por classe.

    Parâmetros:
        predictions: sugestões do modelo na foto (formato de ``match_predictions``).
        truth: caixas humanas da mesma foto.

    Retorna:
        ``{"tp": int, "fp": int, "fn": int, "by_class": {"tp": Counter, "fp": Counter,
        "fn": Counter}}``; cada Counter tem ``class_id -> quantidade``. É o formato
        que ``per_class_report`` soma.

    Chamada por: active_learning.py (recompensa), model_gate.py e autolabel.py.
    """
    # Pareamento com a régua padrão (IOU_MATCH).
    hits, used = match_predictions(predictions, truth)
    tp, fp, fn = Counter(), Counter(), Counter()
    # Cada sugestão é TP (achou par) ou FP (alarme falso ou duplicata), contada na
    # classe que o MODELO previu.
    for prediction, hit in zip(predictions, hits):
        (tp if hit else fp)[prediction["class_id"]] += 1
    # Cada caixa humana sem par é uma omissão (FN), contada na classe humana.
    for position, box in enumerate(truth):
        if position not in used:
            fn[box["class_id"]] += 1
    # EXTENSAO: para devolver outra contagem por foto (ex.: lista das posições
    # omitidas), acrescente a chave aqui; quem consome é per_class_report e
    # active_learning.review_reward.
    return {"tp": sum(tp.values()), "fp": sum(fp.values()), "fn": sum(fn.values()),
            "by_class": {"tp": tp, "fp": fp, "fn": fn}}


def per_class_report(errors: list[dict], names: list[str]) -> dict:
    """Soma os erros de várias fotos e calcula precisão e recall por classe.

    Classe sem nenhuma caixa humana nem sugestão fica com ``None``: não medida,
    o que é diferente de zero.

    Parâmetros:
        errors: lista de saídas de ``image_errors`` (uma por foto).
        names: nomes das classes na ordem dos IDs da taxonomia (``names[0]`` é a
            classe 0). ATENCAO: a posição na lista É o class_id; uma lista em
            outra ordem troca os números entre classes.

    Retorna:
        Dict ``{nome_da_classe: {"tp", "fp", "fn", "human_boxes", "precision",
        "recall"}, ..., "_total": {...}}``. ``human_boxes`` = TP + FN (caixas
        humanas). ``precision``/``recall`` com 4 casas ou ``None`` quando o
        denominador é zero.

    Chamada por: model_gate.py e autolabel.py.
    """
    # 1. Soma os Counters por classe de todas as fotos.
    totals = {key: Counter() for key in ("tp", "fp", "fn")}
    for item in errors:
        for key in totals:
            totals[key].update(item["by_class"][key])
    # 2. Uma linha por classe da taxonomia (classe ausente nos dados fica com 0/None).
    # ATENCAO: IDs de class_id fora de range(len(names)) são somados em totals mas
    # NÃO aparecem por classe; só entram no "_total".
    report = {}
    for class_id, name in enumerate(names):
        tp, fp, fn = (totals[key][class_id] for key in ("tp", "fp", "fn"))
        # precisão = TP/(TP+FP); recall = TP/(TP+FN); None quando não há base.
        # EXTENSAO: nova métrica por classe (ex.: "f1": 2*p*r/(p+r)): acrescente a
        # chave aqui e no "_total" logo abaixo; quem exibe é model_gate.py/model_report.py.
        report[name] = {
            "tp": tp, "fp": fp, "fn": fn, "human_boxes": tp + fn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        }
    # 3. Total geral (micro-média: soma de todas as classes antes de dividir).
    tp, fp, fn = (sum(totals[key].values()) for key in ("tp", "fp", "fn"))
    report["_total"] = {"tp": tp, "fp": fp, "fn": fn, "human_boxes": tp + fn,
                        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                        "recall": round(tp / (tp + fn), 4) if tp + fn else None}
    return report
