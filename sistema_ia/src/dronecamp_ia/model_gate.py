"""Comparação candidato × modelo atual por classe, com regra explícita de adoção.

Função no projeto: um treino que termina ou um arquivo de pesos novo não prova
melhora. Este módulo roda os dois modelos nas mesmas fotos com caixas humanas e
mede, por classe: acertos (TP), falsos positivos (FP), omissões (FN), precisão,
recall e AP50. Depois aplica a regra de adoção abaixo e grava tudo em
``runs/gate_*/gate.json``. Não troca pesos, não apaga o modelo anterior.

Fotos que estiveram no treino de QUALQUER um dos dois modelos são excluídas:
nelas o modelo pode só ter decorado as caixas.

Regra de adoção (``adopt``), todas obrigatórias:
1. mAP50 do candidato maior que o do atual por pelo menos ``min_map_gain``;
2. recall total do candidato não menor que o do atual;
3. nenhuma classe com ``min_class_boxes`` ou mais caixas humanas perde mais que
   ``max_class_recall_drop`` de recall (análise de regressão por classe).
Mesmo com ``adopt`` verdadeiro, ``production_ready`` continua falso enquanto a
avaliação não usar edificações independentes e revisão técnica.

Quem usa: comando ``compare-models`` e ``hparam_bandit.py`` (recomendação final).
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from .config import ProjectConfig, detection_names, load_taxonomy
from .io import file_hash, make_run_directory, write_json
from .matching import image_errors, match_predictions, per_class_report, yolo_label_boxes

AP_CONF = 0.001  # AP precisa das caixas de baixa confiança, como no model.val


def average_precision(scored: list[tuple[float, bool]], positives: int) -> float | None:
    """AP com interpolação em todos os pontos (VOC 2010+/COCO, um limiar de IoU)."""
    if positives == 0:
        return None
    scored = sorted(scored, key=lambda value: -value[0])
    tp = fp = 0
    recalls, precisions = [0.0], [1.0]
    for _, hit in scored:
        tp, fp = tp + hit, fp + (not hit)
        recalls.append(tp / positives)
        precisions.append(tp / (tp + fp))
    # Envelope monotônico da precisão e soma das áreas onde o recall cresce.
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    return round(sum((recalls[index] - recalls[index - 1]) * precisions[index] for index in range(1, len(recalls))), 4)


STRATA = ("fotos_fatiadas", "fotos_inteiras")
REGIMES = ("janelas", "inteira")


def _summarize(errors: list[dict], scored: dict, positives: dict, names: list[str]) -> dict:
    report = per_class_report(errors, names)
    aps = []
    for class_id, name in enumerate(names):
        report[name]["ap50"] = average_precision(scored[class_id], positives[class_id])
        if report[name]["ap50"] is not None:
            aps.append(report[name]["ap50"])
    report["_total"]["map50"] = round(sum(aps) / len(aps), 4) if aps else None
    report["_total"]["classes_measured"] = len(aps)
    report["_total"]["images"] = len(errors)
    return report


def evaluate_model(model, model_config: ProjectConfig, root: Path, samples: list[dict], names: list[str],
                   conf: float, regime: str = "janelas") -> dict:
    """Métricas por classe de um modelo nas amostras dadas (``pilot.json``).

    ``regime``: ``"janelas"`` usa ``pilot.tiling`` (o regime das sugestões; fotos
    pequenas continuam inteiras), ``"inteira"`` força a foto inteira (regime dos
    modelos antigos). O relatório traz também ``_strata``: as mesmas métricas
    separadas em fotos fatiadas (drone, 20 MP) e fotos inteiras (laudo), porque o
    pedido é confiabilidade nas fotos de drone, e ``_regime``.
    """
    from .review_data import review_image_size
    from .suggestions import detect_boxes, tiling_plan

    if regime not in REGIMES:
        raise ValueError(f"regime deve ser um de {REGIMES}.")
    tiling = None if regime == "janelas" else {"enabled": False}
    buckets = {key: ([], defaultdict(list), defaultdict(int)) for key in ("_all", *STRATA)}
    for sample in samples:
        image = root / sample["image"]
        width, height = review_image_size(image)
        truth = yolo_label_boxes(root / sample["label"], width, height)
        predictions = detect_boxes(model, image, model_config, AP_CONF, tiling)
        hits, _ = match_predictions(predictions, truth)
        stratum = STRATA[0] if tiling_plan(model_config, width, height) else STRATA[1]
        for key in ("_all", stratum):
            errors, scored, positives = buckets[key]
            for prediction, hit in zip(predictions, hits):
                scored[prediction["class_id"]].append((prediction["confidence"], hit))
            for box in truth:
                positives[box["class_id"]] += 1
            # Contagens operacionais na confiança das sugestões (o que o revisor vê).
            errors.append(image_errors([value for value in predictions if value["confidence"] >= conf], truth))
    report = _summarize(*buckets["_all"], names)
    report["_strata"] = {key: _summarize(*buckets[key], names) for key in STRATA if buckets[key][0]}
    report["_regime"] = regime
    return report


def decide(baseline: dict, candidate: dict, names: list[str], min_map_gain: float = 0.01,
           max_class_recall_drop: float = 0.1, min_class_boxes: int = 3) -> dict:
    """Aplica a regra de adoção e lista cada motivo de recusa e cada regressão."""
    reasons, regressions = [], []
    base_map, cand_map = baseline["_total"]["map50"], candidate["_total"]["map50"]
    if base_map is None or cand_map is None:
        reasons.append("Sem caixas humanas avaliáveis nas fotos fora do treino.")
    elif cand_map < base_map + min_map_gain:
        reasons.append(f"mAP50 {cand_map:.3f} não supera o atual {base_map:.3f} em {min_map_gain:.3f}.")
    base_recall, cand_recall = baseline["_total"]["recall"] or 0.0, candidate["_total"]["recall"] or 0.0
    if cand_recall < base_recall:
        reasons.append(f"Recall total caiu de {base_recall:.3f} para {cand_recall:.3f}.")
    for name in names:
        before, after = baseline[name], candidate[name]
        if before["human_boxes"] < min_class_boxes or before["recall"] is None:
            continue
        drop = before["recall"] - (after["recall"] or 0.0)
        if drop > max_class_recall_drop:
            regressions.append({"class": name, "recall_before": before["recall"], "recall_after": after["recall"],
                                "human_boxes": before["human_boxes"]})
    if regressions:
        reasons.append("Regressão de recall em: " + ", ".join(item["class"] for item in regressions) + ".")
    return {"adopt": not reasons, "reasons": reasons, "regressions": regressions,
            "rule": {"min_map_gain": min_map_gain, "max_class_recall_drop": max_class_recall_drop,
                     "min_class_boxes": min_class_boxes}}


def _trained(weights: Path) -> set[str]:
    from .calibration import trained_image_hashes

    return trained_image_hashes(weights)


def _load_models(config: ProjectConfig, weights: list[str], names: list[str], onnx_provider: str) -> tuple[list, set]:
    """Abre cada peso (.pt ou .onnx) e junta as fotos vistas no treino de qualquer um."""
    from .backend import check_domain_names, load_inference_model
    from .training import pilot_inference_config

    loaded, excluded = [], set()
    for path in weights:
        model = load_inference_model(config, path, onnx_provider)
        check_domain_names(model, names)
        source = Path(model.dronecamp_source).resolve()
        used = Path(getattr(model, "dronecamp_onnx", None) or source).resolve()
        loaded.append({"model": model, "config": pilot_inference_config(config, model), "source": source, "used": used})
        excluded |= _trained(source)
    return loaded, excluded


def _series(entry: dict, report: dict) -> dict:
    strata = report.pop("_strata", {})
    regime = report.pop("_regime", None)
    return {"weights": str(entry["used"]), "sha256": file_hash(entry["used"]), "source_checkpoint": str(entry["source"]),
            "runtime": entry["model"].dronecamp_runtime, "imgsz": entry["config"].prediction["imgsz"],
            "regime": regime, "metrics": report, "strata": strata}


def _strata_counts(strata: dict) -> dict:
    return {key: value["_total"]["images"] for key, value in (strata or {}).items()}


def compare_models(config: ProjectConfig, data_path: Path, baseline: str, candidate: str,
                   splits: tuple[str, ...] = ("test",), conf: float | None = None, onnx_provider: str = "directml",
                   output: Path | None = None, **rule) -> Path:
    """Roda os dois modelos nas mesmas fotos fora do treino de ambos e grava gate.json.

    Pesos ``.onnx`` rodam no ONNX Runtime (``onnx_provider``: GPU DirectML ou CPU);
    a decisão usa o regime das sugestões (janelas); ``metrics_whole`` registra a
    foto inteira para comparar com os modelos antigos.
    """
    names = detection_names(load_taxonomy(config.taxonomy_path))
    root = Path(data_path).resolve().parent
    manifest = json.loads((root / "pilot.json").read_text(encoding="utf-8"))
    conf = float(conf if conf is not None else config.pilot.get("suggestion_conf", config.prediction["conf"]))
    loaded, excluded = _load_models(config, [baseline, candidate], names, onnx_provider)
    models = dict(zip(("baseline", "candidate"), loaded))
    samples = [sample for sample in manifest["images"]
               if sample["split"] in splits and sample["image_sha256"] not in excluded]
    if not samples:
        raise ValueError("Nenhuma foto do split escolhido ficou fora do treino dos dois modelos.")
    series = {role: _series(entry, evaluate_model(entry["model"], entry["config"], root, samples, names, conf))
              for role, entry in models.items()}
    whole = {role: evaluate_model(entry["model"], entry["config"], root, samples, names, conf, regime="inteira")
             for role, entry in models.items()}
    decision = decide(series["baseline"]["metrics"], series["candidate"]["metrics"], names, **rule)
    run = output if output is not None else make_run_directory(config.root, "gate")
    write_json(run / "gate.json", {
        "schema_version": 1, "dataset": str(Path(data_path).resolve()), "pilot_sha256": file_hash(root / "pilot.json"),
        "splits": list(splits), "conf": conf, "ap_conf": AP_CONF, "iou_match": 0.5,
        "tiling": config.pilot.get("tiling"), "strata": _strata_counts(series["baseline"]["strata"]),
        "images_evaluated": len(samples), "images_excluded_seen_in_training": len(excluded),
        "buildings": sorted({str(sample.get("building_group")) for sample in samples}),
        "independent_evaluation": len({sample.get("building_group") for sample in samples}) >= 3,
        **{role: {**series[role], "metrics_whole": {key: value for key, value in whole[role].items()
                                                     if key not in ("_strata", "_regime")}}
           for role in ("baseline", "candidate")},
        **decision, "production_ready": False,
        "note": "adopt=true recomenda o candidato para sugestões; o modelo anterior continua preservado.",
    })
    return run


def measure_models(config: ProjectConfig, data_path: Path, weights: list[str], splits: tuple[str, ...] = ("test",),
                   conf: float | None = None, onnx_provider: str = "directml", regimes: tuple[str, ...] | None = None,
                   output: Path | None = None) -> Path:
    """Mede N modelos nas MESMAS fotos (fora do treino de todos) e grava ``measurement.json``.

    É a entrada do ``report-models``: gráficos comparáveis exigem o mesmo conjunto
    de fotos e a mesma régua. ``regimes``: ``("janelas", "inteira")`` por padrão
    quando ``pilot.tiling`` está ligado (cada modelo vira uma série por regime),
    senão só ``inteira``. Nada aqui decide adoção; isso é do ``compare-models``.
    """
    names = detection_names(load_taxonomy(config.taxonomy_path))
    root = Path(data_path).resolve().parent
    manifest = json.loads((root / "pilot.json").read_text(encoding="utf-8"))
    conf = float(conf if conf is not None else config.pilot.get("suggestion_conf", config.prediction["conf"]))
    tiling = config.pilot.get("tiling") or {}
    tiled = bool(tiling) and tiling.get("enabled", True)
    regimes = tuple(regimes or (REGIMES if tiled else ("inteira",)))
    if any(regime not in REGIMES for regime in regimes) or ("janelas" in regimes and not tiled):
        raise ValueError("regimes aceita 'janelas' (exige pilot.tiling) e 'inteira'.")
    loaded, excluded = _load_models(config, list(weights), names, onnx_provider)
    samples = [sample for sample in manifest["images"]
               if sample["split"] in splits and sample["image_sha256"] not in excluded]
    if not samples:
        raise ValueError("Nenhuma foto do split escolhido ficou fora do treino de todos os modelos.")
    series = [_series(entry, evaluate_model(entry["model"], entry["config"], root, samples, names, conf, regime))
              for entry in loaded for regime in regimes]
    run = output if output is not None else make_run_directory(config.root, "measure")
    write_json(run / "measurement.json", {
        "schema_version": 1, "dataset": str(Path(data_path).resolve()), "pilot_sha256": file_hash(root / "pilot.json"),
        "splits": list(splits), "conf": conf, "ap_conf": AP_CONF, "iou_match": 0.5, "tiling": tiling or None,
        "images_evaluated": len(samples), "images_excluded_seen_in_training": len(excluded),
        "images": [{"image_sha256": sample["image_sha256"], "split": sample["split"]} for sample in samples],
        "strata": _strata_counts(series[0]["strata"]) if series else {},
        "buildings": sorted({str(sample.get("building_group")) for sample in samples}),
        "models": series, "production_ready": False,
        "note": "Mesmas fotos e mesma régua para todos os modelos; nenhuma adoção é decidida aqui.",
    })
    return run
