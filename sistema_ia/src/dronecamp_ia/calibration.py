"""Calibração das sugestões com scikit-learn: probabilidade de o revisor aceitar.

Função no projeto: a confiança do YOLO piloto não é probabilidade. Uma sugestão
com 0,30 de ``residuos_telha`` e outra com 0,30 de ``fixador_telha_frouxo`` não
acertam na mesma taxa. Este módulo aprende, com as decisões humanas já
registradas, P(sugestão reencontra uma caixa aprovada | características da
sugestão) e usa essa probabilidade para ordenar as sugestões na revisão.

Modelagem:
- Exemplo = uma sugestão do modelo numa foto com decisão humana.
- Rótulo y = 1 se a sugestão reencontra caixa humana (mesma classe, IoU >= 0,5;
  ``matching.py``), senão 0.
- Características x (``FEATURES``): confiança e logit, classe (one-hot), área e
  proporção da caixa, posição, posição no ranking da foto, quantidade de
  sugestões na foto e sobreposição com outras sugestões (mesma classe e outra).
- Modelo: ``LogisticRegression`` (com padronização) sobre x. É deliberadamente
  simples: com dezenas de exemplos, modelos maiores só decorariam.
- Avaliação honesta: validação cruzada agrupada por cena (``GroupKFold``), para
  que recortes da mesma cena não fiquem em treino e teste ao mesmo tempo.
  Compara Brier, AUC-ROC e precisão média contra a confiança crua do YOLO.
  ``improves_over_raw_confidence`` só é verdadeiro se o calibrador ganhar nas
  duas primeiras métricas.

O calibrador nunca aprova nem apaga sugestões: grava ``acceptance_probability``
em cada caixa e reordena a lista. A decisão continua sendo do revisor.

Fotos que estiveram no treino do modelo são excluídas por padrão: nelas o modelo
decorou as respostas e a confiança parece melhor do que é.

Quem usa: comandos ``fit-calibrator`` e ``suggest --calibrator``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from .config import ProjectConfig, detection_names, load_taxonomy
from .io import file_hash, make_run_directory, resolve_local_path, write_json
from .matching import match_predictions, yolo_label_boxes
from .pilot import box_iou

SCHEMA = 1
BASE_FEATURES = ["confidence", "logit_confidence", "log_area_fraction", "log_aspect_ratio", "center_x", "center_y",
                 "rank_fraction", "log_suggestions_in_image", "max_iou_same_class", "max_iou_other_class"]


def feature_names(class_count: int) -> list[str]:
    return BASE_FEATURES + [f"class_{index}" for index in range(class_count)]


def suggestion_features(boxes: list[dict], width: int, height: int, class_count: int) -> np.ndarray:
    """Matriz (sugestões × características) de uma foto; ordem das linhas = ordem de ``boxes``."""
    rows = []
    ranked = sorted(range(len(boxes)), key=lambda index: -boxes[index]["confidence"])
    rank = {index: position for position, index in enumerate(ranked)}
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box["bbox_xyxy"]
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        confidence = min(max(float(box["confidence"]), 1e-4), 1 - 1e-4)
        same = [box_iou(box["bbox_xyxy"], other["bbox_xyxy"]) for position, other in enumerate(boxes)
                if position != index and other["class_id"] == box["class_id"]]
        other = [box_iou(box["bbox_xyxy"], value["bbox_xyxy"]) for position, value in enumerate(boxes)
                 if position != index and value["class_id"] != box["class_id"]]
        one_hot = [0.0] * class_count
        one_hot[int(box["class_id"])] = 1.0
        rows.append([
            confidence, math.log(confidence / (1 - confidence)),
            math.log(w * h / (width * height)), math.log(w / h),
            (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            rank[index] / max(1, len(boxes) - 1), math.log1p(len(boxes)),
            max(same, default=0.0), max(other, default=0.0), *one_hot,
        ])
    return np.asarray(rows, dtype=float).reshape(len(boxes), len(BASE_FEATURES) + class_count)


# ---------------------------------------------------------------------------
# Exemplos rotulados a partir de um dataset piloto (fotos com caixas aprovadas).
# ---------------------------------------------------------------------------

def trained_image_hashes(weights: Path) -> set[str]:
    """SHA-256 das fotos que estiveram no treino destes pesos (dataset_snapshot.json da execução)."""
    snapshot = Path(weights).resolve().parent.parent.parent / "dataset_snapshot.json"
    if not snapshot.is_file():
        return set()
    files = json.loads(snapshot.read_text(encoding="utf-8"))["files"]
    return {Path(item["path"]).stem for item in files if item["path"].startswith("images/train/")}


def collect_examples(config: ProjectConfig, data_path: Path, weights: str, conf: float = 0.01,
                     include_trained: bool = False) -> dict:
    """Roda o detector nas fotos do dataset piloto e rotula cada sugestão contra as caixas humanas."""
    from .backend import check_domain_names, load_detector
    from .review_data import review_image_size
    from .suggestions import detect_boxes
    from .training import pilot_inference_config

    names = detection_names(load_taxonomy(config.taxonomy_path))
    root = Path(data_path).resolve().parent
    manifest = json.loads((root / "pilot.json").read_text(encoding="utf-8"))
    model = load_detector(config, weights)
    check_domain_names(model, names)
    model_config = pilot_inference_config(config, model)
    seen = set() if include_trained else trained_image_hashes(Path(model.ckpt_path))
    features, labels, groups, confidences, used, skipped = [], [], [], [], [], 0
    for sample in manifest["images"]:
        if sample["image_sha256"] in seen:
            skipped += 1
            continue
        image = root / sample["image"]
        width, height = review_image_size(image)
        truth = yolo_label_boxes(root / sample["label"], width, height)
        boxes = detect_boxes(model, image, model_config, conf)
        if not boxes:
            used.append({"image_sha256": sample["image_sha256"], "suggestions": 0, "human_boxes": len(truth)})
            continue
        hits, _ = match_predictions(boxes, truth)
        features.append(suggestion_features(boxes, width, height, len(names)))
        labels.extend(int(hit) for hit in hits)
        groups.extend([sample["scene_group"]] * len(boxes))
        confidences.extend(box["confidence"] for box in boxes)
        used.append({"image_sha256": sample["image_sha256"], "suggestions": len(boxes), "human_boxes": len(truth),
                     "matched": int(sum(hits))})
    return {
        "X": np.vstack(features) if features else np.zeros((0, len(feature_names(len(names))))),
        "y": np.asarray(labels, dtype=int), "groups": np.asarray(groups), "raw_confidence": np.asarray(confidences),
        "images": used, "skipped_trained_images": skipped, "names": names,
        "weights": str(Path(model.ckpt_path).resolve()), "conf": conf,
    }


# ---------------------------------------------------------------------------
# Treino e avaliação do calibrador.
# ---------------------------------------------------------------------------

def _pipeline(seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # C pequeno = regularização forte. Sem class_weight="balanced": ele desloca as
    # probabilidades para cima e destrói a calibração, que é o objetivo aqui.
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, random_state=seed))


def cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, raw: np.ndarray, seed: int = 42) -> dict:
    """Probabilidades fora da dobra (GroupKFold por cena) comparadas com a confiança crua."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold

    folds = min(5, len(set(groups.tolist())))
    if folds < 2:
        raise ValueError("Calibração exige sugestões em pelo menos 2 cenas diferentes.")
    out_of_fold = np.full(len(y), np.nan)
    for train_index, test_index in GroupKFold(n_splits=folds).split(X, y, groups):
        if len(set(y[train_index].tolist())) < 2:
            out_of_fold[test_index] = y[train_index].mean()  # dobra sem as duas classes: prevê a taxa base
            continue
        model = _pipeline(seed).fit(X[train_index], y[train_index])
        out_of_fold[test_index] = model.predict_proba(X[test_index])[:, 1]

    def scores(probability):
        return {"brier": round(float(brier_score_loss(y, probability)), 4),
                "roc_auc": round(float(roc_auc_score(y, probability)), 4),
                "average_precision": round(float(average_precision_score(y, probability)), 4)}

    calibrated, baseline = scores(out_of_fold), scores(raw)
    return {"folds": folds, "group": "scene_group", "calibrated": calibrated, "raw_confidence": baseline,
            "improves_over_raw_confidence": calibrated["brier"] < baseline["brier"]
            and calibrated["roc_auc"] > baseline["roc_auc"]}


def fit_calibrator(config: ProjectConfig, data_path: Path, weights: str, conf: float = 0.01,
                   include_trained: bool = False, seed: int = 42) -> Path:
    """Coleta exemplos, avalia por validação cruzada, treina com tudo e grava em runs/calibrator_*."""
    import joblib
    import sklearn

    examples = collect_examples(config, data_path, weights, conf, include_trained)
    y = examples["y"]
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    # Mínimo para não fingir aprendizado: acertos e erros em quantidade e cenas distintas.
    if positives < 5 or negatives < 5:
        raise ValueError(f"Exemplos insuficientes para calibrar: {positives} acertos e {negatives} erros "
                         f"(mínimo 5 de cada). Revise mais fotos fora do treino do modelo.")
    report = cross_validate(examples["X"], y, examples["groups"], examples["raw_confidence"], seed)
    model = _pipeline(seed).fit(examples["X"], y)
    run = make_run_directory(config.root, "calibrator")
    model_path = run / "calibrator.joblib"
    joblib.dump(model, model_path)
    metadata = {
        "schema_version": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "acceptance_probability_logistic_regression", "sklearn_version": sklearn.__version__,
        "model_file": model_path.name, "model_sha256": file_hash(model_path),
        "features": feature_names(len(examples["names"])), "class_names": examples["names"],
        "taxonomy_sha256": file_hash(config.taxonomy_path),
        "detector_weights": examples["weights"], "detector_weights_sha256": file_hash(Path(examples["weights"])),
        "dataset": str(Path(data_path).resolve()), "collection_conf": conf,
        "excluded_training_images": not include_trained, "skipped_trained_images": examples["skipped_trained_images"],
        "examples": len(y), "positives": positives, "negatives": negatives, "images": examples["images"],
        "cross_validation": report,
        "warning": "Probabilidade aprendida com poucas decisões humanas de uma edificação; ordena sugestões, "
                   "não aprova nem descarta caixas.",
    }
    write_json(run / "calibrator.json", metadata)
    return run


# ---------------------------------------------------------------------------
# Uso do calibrador nas sugestões.
# ---------------------------------------------------------------------------

class Calibrator:
    """Calibrador carregado e conferido (hash do arquivo e taxonomia) antes do uso."""

    def __init__(self, config: ProjectConfig, path: Path, allow_unproven: bool = False):
        import joblib

        directory = Path(path).resolve()
        directory = directory if directory.is_dir() else directory.parent
        self.metadata = json.loads((directory / "calibrator.json").read_text(encoding="utf-8"))
        model_path = directory / self.metadata["model_file"]
        # joblib usa pickle: só carregue arquivos gerados aqui; o hash impede troca silenciosa.
        if file_hash(model_path) != self.metadata["model_sha256"]:
            raise ValueError("calibrator.joblib não confere com o hash registrado em calibrator.json.")
        if self.metadata["taxonomy_sha256"] != file_hash(config.taxonomy_path):
            raise ValueError("Calibrador de outra taxonomia; gere um novo com fit-calibrator.")
        # Regra do projeto: só usa o que mostrou ganho medido sobre a confiança crua.
        if not allow_unproven and not self.metadata["cross_validation"].get("improves_over_raw_confidence"):
            raise ValueError("Calibrador sem ganho medido sobre a confiança crua do YOLO (veja cross_validation "
                             "em calibrator.json). Revise mais fotos fora do treino e gere outro.")
        self.model = joblib.load(model_path)
        self.class_count = len(self.metadata["class_names"])
        self.directory = directory

    def annotate(self, boxes: list[dict], width: int, height: int) -> list[dict]:
        """Acrescenta ``acceptance_probability`` e reordena; não remove nenhuma caixa."""
        if not boxes:
            return boxes
        probability = self.model.predict_proba(suggestion_features(boxes, width, height, self.class_count))[:, 1]
        for box, value in zip(boxes, probability):
            box["acceptance_probability"] = round(float(value), 4)
        return sorted(boxes, key=lambda box: -box["acceptance_probability"])

    def describe(self) -> dict:
        return {"path": str(self.directory), "model_sha256": self.metadata["model_sha256"],
                "cross_validation": self.metadata["cross_validation"],
                "proven": bool(self.metadata["cross_validation"].get("improves_over_raw_confidence"))}


def load_calibrator(config: ProjectConfig, path: Path | str | None, allow_unproven: bool = False) -> Calibrator | None:
    return Calibrator(config, resolve_local_path(str(path), config.root), allow_unproven) if path else None
