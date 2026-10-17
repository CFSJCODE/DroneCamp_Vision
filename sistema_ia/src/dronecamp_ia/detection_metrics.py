"""Métricas detalhadas de detecção: por classe, por tamanho, por condição e por confiança.

Função no projeto: responder "onde o modelo erra" com números que sempre dizem
quantos exemplos os sustentam. ``model_gate.py`` decide adoção (precisão, recall
e AP50 por classe); este módulo acrescenta, com a MESMA régua de pareamento
(``matching.match_predictions``: mesma classe, guloso por confiança):

- por classe: suporte (fotos e caixas humanas), TP/FP/FN, precisão, recall, F1,
  AP50, AP75 e AP50-95 (média em IoU 0,50:0,05:0,95, como no COCO);
- classe sem caixa humana aparece com ``status = "nao_avaliada"`` e suporte 0,
  nunca como zero nem escondida numa média;
- localização: distribuição da IoU das caixas pareadas e o tipo de cada falso
  positivo (fundo, classe trocada, caixa imprecisa, duplicata);
- tamanho do objeto em pixels da foto original (faixas COCO por padrão);
- operação: falsos positivos por foto original, recall com um limite de falsos
  alarmes por foto, curvas precisão-recall e F1 por confiança;
- condições de captura: as mesmas métricas por edificação, material, ângulo,
  iluminação ou período, com o percentual de fotos sem a etiqueta;
- exemplos de erro (imagem, caixa humana, previsão, classe, confiança, IoU) para
  ``quality_report.py`` recortar e gravar.

Quem chama: ``quality_report.py`` (comando ``evaluate-detailed``) e
``mutual_learning.summarize_detections`` (F1 e status por classe no DML).

O que lê e grava no disco: nada. Recebe listas em memória e devolve dicts.

Unidade de avaliação: a FOTO ORIGINAL. As previsões precisam chegar já
consolidadas na foto inteira (``suggestions.detect_boxes`` junta as janelas
sobrepostas); contar falsos alarmes por recorte dobraria os alarmes que caem
na sobreposição de duas janelas.

Fórmulas (P = caixas humanas da classe; k = sugestões ordenadas pela confiança):
    precisão = TP / (TP + FP)        recall = TP / (TP + FN)
    F1       = 2·precisão·recall / (precisão + recall)
    AP@t     = área sob a curva precisão × recall com pareamento em IoU ≥ t
               (interpolação em todos os pontos: ``model_gate.average_precision``)
    AP50-95  = média de AP@t para t = 0,50; 0,55; ...; 0,95
    mAP      = média simples das AP das classes AVALIADAS
    FP/foto  = falsos positivos ÷ fotos avaliadas (inclusive as negativas)

ATENCAO: o COCO usa interpolação em 101 pontos; aqui é a interpolação em todos os
pontos já usada pelo projeto (``model_gate``), para os números baterem com o
``compare-models``. A diferença costuma ficar abaixo de 0,01.

Onde mexer:
- Faixas de tamanho: ``DEFAULT_SIZE_EDGES`` (ou ``size_edges`` na chamada).
- Limites de falsos alarmes por foto: ``DEFAULT_FP_BUDGETS``.
- Novo tipo de falso positivo: ``classify_false_positive``.
- Nova métrica por classe: ``_class_block`` (EXTENSAO lá).
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math

from .matching import IOU_MATCH, match_predictions
from .pilot import box_iou

# ---------------------------------------------------------------------------
# Constantes do protocolo
# ---------------------------------------------------------------------------

# Limiares de IoU do AP50-95 (COCO): 0,50 até 0,95 de 0,05 em 0,05.
COCO_IOUS = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))
# Faixas de tamanho pela ÁREA da caixa em pixels² da foto original (COCO: 32² e 96²).
# ATENCAO: nas fotos de 20 MP do drone quase todo defeito passa de 96² px; as faixas
# COCO ficam como padrão por serem a referência citada, mas ``--size-edges`` permite
# faixas do domínio (ex.: 64 256 para lados de 64 e 256 px).
DEFAULT_SIZE_EDGES = (32, 96)
# Limites de falsos alarmes por foto em que o recall é informado (trabalho de revisão).
DEFAULT_FP_BUDGETS = (0.5, 1.0, 2.0, 5.0)
# Confianças em que as curvas são amostradas (0,00 a 0,95 de 0,05 em 0,05).
CURVE_THRESHOLDS = tuple(round(0.05 * step, 2) for step in range(20))
# Exemplos de erro guardados por classe e por tipo (os mais confiantes primeiro).
DEFAULT_EXAMPLES = 8
# Texto padrão de etiqueta ausente nas condições de captura.
UNKNOWN = "desconhecido"


def _ratio(numerator: float, denominator: float) -> float | None:
    """Divisão com 4 casas; ``None`` quando o denominador é zero (não medido ≠ zero)."""
    return round(numerator / denominator, 4) if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    """F1 = 2PR/(P+R); ``None`` se faltar P ou R; 0 se ambos forem 0."""
    if precision is None or recall is None:
        return None
    return round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0


def _quantiles(values: list[float]) -> dict:
    """Resumo de uma lista: n, média, p25, mediana, p75, mínimo (vazio = só n=0)."""
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        # Interpolação linear entre as duas posições vizinhas (mesma regra do numpy).
        position = fraction * (len(ordered) - 1)
        low, high = math.floor(position), math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {"n": len(ordered), "mean": round(sum(ordered) / len(ordered), 4), "p25": round(at(0.25), 4),
            "median": round(at(0.5), 4), "p75": round(at(0.75), 4), "min": round(ordered[0], 4)}


def box_area(box: list[float]) -> float:
    """Área em pixels² de ``[x1, y1, x2, y2]`` (caixa invertida conta 0)."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def size_bucket(area: float, edges: tuple[float, ...] = DEFAULT_SIZE_EDGES) -> str:
    """Nome da faixa de tamanho de uma área: ``pequeno``, ``medio``, ``grande`` (ou ``faixa_N``).

    Parâmetros:
        area: área da caixa em pixels².
        edges: LADOS (px) que separam as faixas; a faixa i vai de edges[i-1]² a edges[i]².
    """
    names = ("pequeno", "medio", "grande") if len(edges) == 2 else tuple(f"faixa_{i}" for i in range(len(edges) + 1))
    for index, edge in enumerate(edges):
        if area < float(edge) ** 2:
            return names[index]
    return names[len(edges)]


def size_bucket_names(edges: tuple[float, ...] = DEFAULT_SIZE_EDGES) -> tuple[str, ...]:
    """Nomes das faixas, na ordem (mesma regra de ``size_bucket``)."""
    return ("pequeno", "medio", "grande") if len(edges) == 2 else tuple(f"faixa_{i}" for i in range(len(edges) + 1))


# ---------------------------------------------------------------------------
# Tipo de cada falso positivo (localização × confusão de classe)
# ---------------------------------------------------------------------------

def classify_false_positive(prediction: dict, truth: list[dict], used: set[int], iou: float = IOU_MATCH,
                            loose_iou: float = 0.1) -> tuple[str, float, int | None]:
    """Explica um falso positivo comparando-o com as caixas humanas da foto.

    Ordem de teste (o primeiro que valer):
    1. ``duplicata``: IoU ≥ ``iou`` com caixa humana da MESMA classe já usada por
       uma sugestão mais confiante;
    2. ``classe_trocada``: IoU ≥ ``iou`` com caixa humana de OUTRA classe
       (o modelo achou a região, errou o nome);
    3. ``caixa_imprecisa``: mesma classe com ``loose_iou`` ≤ IoU < ``iou``
       (achou o defeito, desenhou a caixa errada);
    4. ``fundo``: nenhuma sobreposição relevante (alarme no telhado sem defeito
       anotado; pode ser também um defeito que o revisor não marcou).

    Retorna: ``(tipo, maior IoU encontrada, posição da caixa humana mais próxima ou None)``.
    """
    best_same, best_same_pos, best_other, best_other_pos = 0.0, None, 0.0, None
    for position, box in enumerate(truth):
        overlap = box_iou(box["bbox_xyxy"], prediction["bbox_xyxy"])
        if box["class_id"] == prediction["class_id"]:
            if overlap > best_same:
                best_same, best_same_pos = overlap, position
        elif overlap > best_other:
            best_other, best_other_pos = overlap, position
    if best_same >= iou and best_same_pos in used:
        return "duplicata", round(best_same, 4), best_same_pos
    if best_other >= iou:
        return "classe_trocada", round(best_other, 4), best_other_pos
    if best_same >= loose_iou:
        return "caixa_imprecisa", round(best_same, 4), best_same_pos
    nearest = max((best_same, best_same_pos), (best_other, best_other_pos), key=lambda item: item[0])
    return "fundo", round(nearest[0], 4), nearest[1] if nearest[0] > 0 else None


FP_TYPES = ("fundo", "classe_trocada", "caixa_imprecisa", "duplicata")


# ---------------------------------------------------------------------------
# Avaliação de um conjunto de fotos
# ---------------------------------------------------------------------------

def _ap(scored: list[tuple[float, bool]], positives: int) -> float | None:
    """AP com a implementação do projeto (importada aqui para evitar ciclo de import)."""
    from .model_gate import average_precision

    return average_precision(scored, positives)


def _class_block(class_id: int, name: str, samples: list[dict], matches: dict, conf: float,
                 size_edges: tuple[float, ...]) -> dict:
    """Métricas de UMA classe sobre todas as fotos (ver ``evaluate_detections``).

    ``matches[iou][image_index]`` = ``(hits, used)`` de ``match_predictions`` naquela foto.
    """
    support_boxes = sum(1 for sample in samples for box in sample["truth"] if box["class_id"] == class_id)
    support_images = sum(1 for sample in samples if any(box["class_id"] == class_id for box in sample["truth"]))
    block = {"class_id": class_id, "support_images": support_images, "support_boxes": support_boxes,
             "status": "avaliada" if support_boxes else "nao_avaliada"}
    # 1. AP em cada limiar de IoU (todas as confianças ≥ ap_conf).
    aps = {}
    for iou in COCO_IOUS:
        scored = []
        for index, sample in enumerate(samples):
            hits, _ = matches[iou][index]
            scored.extend((p["confidence"], hit) for p, hit in zip(sample["predictions"], hits) if p["class_id"] == class_id)
        aps[iou] = _ap(scored, support_boxes)
    measured = [value for value in aps.values() if value is not None]
    block.update({"ap50": aps[0.5], "ap75": aps[0.75],
                  "ap50_95": round(sum(measured) / len(measured), 4) if len(measured) == len(COCO_IOUS) else None})
    # 2. Contagens operacionais na confiança das sugestões, com o tipo de cada FP e a IoU dos acertos.
    tp = fp = fn = predicted = 0
    fp_types, ious = Counter(), []
    sizes = {bucket: {"support_boxes": 0, "tp": 0, "fn": 0, "predictions": 0, "fp": 0}
             for bucket in size_bucket_names(size_edges)}
    for index, sample in enumerate(samples):
        hits, used = matches[0.5][index]
        for prediction, hit in zip(sample["predictions"], hits):
            if prediction["class_id"] != class_id or prediction["confidence"] < conf:
                continue
            predicted += 1
            bucket = sizes[size_bucket(box_area(prediction["bbox_xyxy"]), size_edges)]
            bucket["predictions"] += 1
            if hit:
                tp += 1
                # IoU do par: a caixa humana da mesma classe com maior sobreposição.
                ious.append(max((box_iou(box["bbox_xyxy"], prediction["bbox_xyxy"]) for box in sample["truth"]
                                 if box["class_id"] == class_id), default=0.0))
            else:
                fp += 1
                bucket["fp"] += 1
                fp_types[sample["_fp_types"].get(id(prediction), "fundo")] += 1
        # FN e TP por faixa contados pela caixa HUMANA (suporte de cada faixa).
        confident_used = sample["_used_conf"]
        for position, box in enumerate(sample["truth"]):
            if box["class_id"] != class_id:
                continue
            bucket = sizes[size_bucket(box_area(box["bbox_xyxy"]), size_edges)]
            bucket["support_boxes"] += 1
            if position in confident_used:
                bucket["tp"] += 1
            else:
                fn += 1
                bucket["fn"] += 1
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    for bucket in sizes.values():
        bucket["recall"] = _ratio(bucket["tp"], bucket["support_boxes"])
        bucket["precision"] = _ratio(bucket["predictions"] - bucket["fp"], bucket["predictions"])
    # EXTENSAO: métrica nova por classe = chave nova neste update (e coluna no quality_report._markdown).
    block.update({"predictions": predicted, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
                  "f1": _f1(precision, recall), "fp_types": {key: fp_types.get(key, 0) for key in FP_TYPES},
                  "iou_matched": _quantiles(ious), "by_size": sizes})
    return block


def _operating_curves(samples: list[dict], matches: dict, class_ids: list[int], images: int,
                      fp_budgets: tuple[float, ...]) -> dict:
    """Curvas por confiança (precisão, recall, F1, FP/foto) e recall em limites de FP por foto.

    ``class_ids``: classes incluídas (uma classe ou todas as avaliadas).
    """
    positives = sum(1 for sample in samples for box in sample["truth"] if box["class_id"] in class_ids)
    scored = []
    for index, sample in enumerate(samples):
        hits, _ = matches[0.5][index]
        scored.extend((p["confidence"], hit) for p, hit in zip(sample["predictions"], hits) if p["class_id"] in class_ids)
    scored.sort(key=lambda item: -item[0])
    curve = []
    for threshold in CURVE_THRESHOLDS:
        tp = sum(1 for confidence, hit in scored if confidence >= threshold and hit)
        fp = sum(1 for confidence, hit in scored if confidence >= threshold and not hit)
        precision, recall = _ratio(tp, tp + fp), _ratio(tp, positives)
        curve.append({"conf": threshold, "tp": tp, "fp": fp, "precision": precision, "recall": recall,
                      "f1": _f1(precision, recall), "fp_per_image": _ratio(fp, images)})
    best = max((point for point in curve if point["f1"] is not None), key=lambda point: point["f1"], default=None)
    # Recall no limite de falsos alarmes: desce a confiança enquanto FP/foto ≤ limite.
    budgets = {}
    for budget in fp_budgets:
        tp = fp = 0
        best_recall, best_conf = 0.0 if positives else None, None
        index = 0
        # ATENCAO: previsões com a mesma confiança entram juntas (um limiar real inclui todas).
        while index < len(scored):
            confidence = scored[index][0]
            while index < len(scored) and scored[index][0] == confidence:
                tp, fp = tp + scored[index][1], fp + (not scored[index][1])
                index += 1
            if images and fp / images > budget:
                break
            if positives:
                best_recall, best_conf = tp / positives, confidence
        budgets[str(budget)] = {"recall": round(best_recall, 4) if best_recall is not None else None,
                                "conf": round(best_conf, 4) if best_conf is not None else None}
    return {"positives": positives, "curve": curve, "best_f1": best, "recall_at_fp_per_image": budgets}


def evaluate_detections(samples: list[dict], names: list[str], conf: float = 0.25, ap_conf: float = 0.001,
                        size_edges: tuple[float, ...] = DEFAULT_SIZE_EDGES,
                        fp_budgets: tuple[float, ...] = DEFAULT_FP_BUDGETS, condition_fields: tuple[str, ...] = (),
                        examples: int = DEFAULT_EXAMPLES, protocol: dict | None = None) -> dict:
    """Relatório completo de um modelo num conjunto de fotos originais.

    Parâmetros:
        samples: uma entrada por FOTO ORIGINAL: ``{"image": texto (caminho ou id),
            "truth": [{"class_id", "bbox_xyxy"}], "predictions": [{"class_id",
            "bbox_xyxy", "confidence"}] (já consolidadas na foto, confiança ≥ ap_conf),
            "conditions": {campo: valor}}`` (``conditions`` opcional).
        names: classes na ordem dos IDs.
        conf: confiança das contagens operacionais (o que o revisor vê).
        ap_conf: menor confiança das previsões recebidas (registrada no protocolo).
        size_edges: lados (px) que separam as faixas de tamanho.
        fp_budgets: limites de falsos positivos por foto para o recall operacional.
        condition_fields: campos de ``conditions`` a estratificar (ex.: ``("edificacao",)``).
        examples: exemplos de erro guardados por classe e tipo.
        protocol: dados do protocolo vindos de quem chama (versão dos dados, checkpoint...).

    Retorna:
        ``{"protocol", "per_class", "total", "operating", "conditions", "errors"}``
        (ver ``docs/metricas_treino_avaliacao.md``).

    Chamada por: ``quality_report.evaluate_weights`` e testes.
    """
    # 1. Pareamento uma vez por limiar de IoU e por foto (reaproveitado por todas as métricas).
    matches = {iou: [match_predictions(sample["predictions"], sample["truth"], iou) for sample in samples]
               for iou in COCO_IOUS}
    errors = {"fp": defaultdict(list), "fn": defaultdict(list)}
    for index, sample in enumerate(samples):
        hits, used = matches[0.5][index]
        confident = [(p, hit) for p, hit in zip(sample["predictions"], hits) if p["confidence"] >= conf]
        # Caixas humanas reencontradas por uma sugestão na confiança operacional (o que o revisor veria).
        _, sample["_used_conf"] = match_predictions([p for p, _ in confident], sample["truth"], IOU_MATCH)
        sample["_fp_types"] = {}
        for prediction, hit in confident:
            if hit:
                continue
            kind, overlap, nearest = classify_false_positive(prediction, sample["truth"], used)
            sample["_fp_types"][id(prediction)] = kind
            errors["fp"][prediction["class_id"]].append({
                "type": "falso_positivo", "fp_type": kind, "image": sample["image"],
                "class_id": prediction["class_id"], "class": names[prediction["class_id"]]
                if 0 <= prediction["class_id"] < len(names) else str(prediction["class_id"]),
                "confidence": prediction["confidence"], "bbox_xyxy": prediction["bbox_xyxy"], "iou": overlap,
                "truth_bbox_xyxy": sample["truth"][nearest]["bbox_xyxy"] if nearest is not None else None,
                "truth_class_id": sample["truth"][nearest]["class_id"] if nearest is not None else None})
        for position, box in enumerate(sample["truth"]):
            if position in sample["_used_conf"]:
                continue
            # Omissão: a melhor previsão (qualquer classe, qualquer confiança) explica o motivo.
            nearest = max(sample["predictions"], key=lambda p: box_iou(box["bbox_xyxy"], p["bbox_xyxy"]), default=None)
            overlap = box_iou(box["bbox_xyxy"], nearest["bbox_xyxy"]) if nearest else 0.0
            if nearest and overlap >= IOU_MATCH and nearest["class_id"] != box["class_id"]:
                reason = "classe_trocada"
            elif nearest and overlap >= IOU_MATCH:
                reason = "confianca_baixa"
            elif nearest and overlap >= 0.1:
                reason = "caixa_imprecisa"
            else:
                reason = "nao_detectado"
            errors["fn"][box["class_id"]].append({
                "type": "omissao", "fn_reason": reason, "image": sample["image"], "class_id": box["class_id"],
                "class": names[box["class_id"]] if 0 <= box["class_id"] < len(names) else str(box["class_id"]),
                "truth_bbox_xyxy": box["bbox_xyxy"], "iou": round(overlap, 4),
                "confidence": nearest["confidence"] if nearest else None,
                "bbox_xyxy": nearest["bbox_xyxy"] if nearest else None,
                "predicted_class_id": nearest["class_id"] if nearest else None})
    # 2. Uma linha por classe da taxonomia (inclusive as não avaliadas).
    per_class = {name: _class_block(class_id, name, samples, matches, conf, size_edges)
                 for class_id, name in enumerate(names)}
    evaluated = [name for name in names if per_class[name]["status"] == "avaliada"]
    # 3. Total (micro nas contagens; mAP = média simples das classes avaliadas).
    images = len(samples)
    tp, fp, fn = (sum(per_class[name][key] for name in names) for key in ("tp", "fp", "fn"))
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)

    def mean_of(key: str) -> float | None:
        values = [per_class[name][key] for name in evaluated if per_class[name][key] is not None]
        return round(sum(values) / len(values), 4) if values else None

    fp_types = Counter()
    for name in names:
        fp_types.update(per_class[name]["fp_types"])
    total = {"images": images, "negative_images": sum(1 for sample in samples if not sample["truth"]),
             "support_boxes": sum(len(sample["truth"]) for sample in samples), "tp": tp, "fp": fp, "fn": fn,
             "precision": precision, "recall": recall, "f1": _f1(precision, recall),
             "map50": mean_of("ap50"), "map75": mean_of("ap75"), "map50_95": mean_of("ap50_95"),
             "classes_evaluated": len(evaluated),
             "classes_not_evaluated": [name for name in names if name not in evaluated],
             "fp_per_image": _ratio(fp, images), "fp_types": dict(fp_types)}
    # 4. Operação: curvas e recall com limite de alarmes, no total e por classe avaliada.
    class_index = {name: index for index, name in enumerate(names)}
    # Todas as classes no total: alarme de classe sem caixa humana nesta divisão é falso alarme
    # e conta no FP por foto; o recall só tem positivos das classes avaliadas.
    operating = {"_total": _operating_curves(samples, matches, list(range(len(names))), images, fp_budgets)}
    for name in evaluated:
        operating[name] = _operating_curves(samples, matches, [class_index[name]], images, fp_budgets)
    # 5. Condições de captura: o mesmo total por valor de cada campo, com o percentual desconhecido.
    conditions = {}
    for field in condition_fields:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            value = (sample.get("conditions") or {}).get(field)
            groups[str(value) if value not in (None, "") else UNKNOWN].append(index)
        conditions[field] = {"unknown_share": _ratio(len(groups.get(UNKNOWN, [])), images), "values": {}}
        for value, indices in sorted(groups.items()):
            subset = [samples[index] for index in indices]
            sub_matches = {iou: [matches[iou][index] for index in indices] for iou in COCO_IOUS}
            blocks = [_class_block(class_id, name, subset, sub_matches, conf, size_edges)
                      for class_id, name in enumerate(names)]
            stp, sfp, sfn = (sum(block[key] for block in blocks) for key in ("tp", "fp", "fn"))
            aps = [block["ap50"] for block in blocks if block["ap50"] is not None]
            sp, sr = _ratio(stp, stp + sfp), _ratio(stp, stp + sfn)
            conditions[field]["values"][value] = {
                "images": len(subset), "support_boxes": sum(len(s["truth"]) for s in subset),
                "tp": stp, "fp": sfp, "fn": sfn, "precision": sp, "recall": sr, "f1": _f1(sp, sr),
                "map50": round(sum(aps) / len(aps), 4) if aps else None, "classes_evaluated": len(aps),
                "fp_per_image": _ratio(sfp, len(subset))}
    # 6. Exemplos de erro: os falsos positivos mais confiantes e as omissões, por classe.
    kept = []
    for kind, subtype in (("fp", "fp_type"), ("fn", "fn_reason")):
        for class_id, items in errors[kind].items():
            groups = defaultdict(list)
            for item in items:
                groups[item[subtype]].append(item)
            for group in groups.values():
                group.sort(key=lambda item: -(item.get("confidence") or 0.0))
                kept.extend(group[:examples])
    for sample in samples:
        sample.pop("_used_conf", None)
        sample.pop("_fp_types", None)
    return {"protocol": {**(protocol or {}), "unit": "foto_original", "conf": conf, "ap_conf": ap_conf,
                         "iou_match": IOU_MATCH, "ap_ious": list(COCO_IOUS), "ap_interpolation": "todos_os_pontos",
                         "size_edges_px": list(size_edges), "size_measure": "area_px2_foto_original",
                         "fp_budgets_per_image": list(fp_budgets), "examples_per_class_and_type": examples},
            "per_class": per_class, "total": total, "operating": operating, "conditions": conditions,
            "errors": kept}


def add_f1_and_status(report: dict, support_images: dict[int, int] | None = None, names: list[str] | None = None) -> dict:
    """Acrescenta ``f1`` e ``status`` (avaliada/nao_avaliada) a um ``matching.per_class_report``.

    Usada onde o relatório antigo continua (DML), para que classe sem caixa
    humana apareça como não avaliada e não suma da leitura.

    Parâmetros:
        report: saída de ``per_class_report`` (é alterado e devolvido).
        support_images: ``{class_id: nº de fotos com a classe}`` (opcional).
        names: classes na ordem dos IDs (necessário com ``support_images``).
    """
    for index, name in enumerate(names or []):
        if name in report and support_images is not None:
            report[name]["support_images"] = int(support_images.get(index, 0))
    for name, row in report.items():
        if not isinstance(row, dict) or "tp" not in row:
            continue
        row["f1"] = _f1(row.get("precision"), row.get("recall"))
        if name != "_total":
            row["status"] = "avaliada" if row.get("human_boxes") else "nao_avaliada"
    return report
