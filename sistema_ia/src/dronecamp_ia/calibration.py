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

Fórmulas (para conferir os números do calibrator.json):
- Regressão logística: P(y=1 | x) = 1 / (1 + exp(-(w·z + w₀))), com z = x
  padronizado (média 0, desvio 1 por coluna, ``StandardScaler``).
- logit da confiança: ln(p / (1 - p)), com p limitado a [0,0001; 0,9999].
- IoU (sobreposição): interseção / união das duas caixas (``pilot.box_iou``).
- Brier = média de (probabilidade - y)²; menor é melhor (0 = perfeito).
- AUC-ROC: chance de um acerto receber probabilidade maior que um erro
  (0,5 = sorteio, 1 = separa perfeitamente).
- Precisão média (AP): área sob a curva precisão × recall, como no mAP.

O que o arquivo contém:
- ``SCHEMA``, ``BASE_FEATURES``, ``feature_names``, ``suggestion_features``: características.
- ``trained_image_hashes``, ``collect_examples``: exemplos rotulados (y) a partir de um dataset piloto.
- ``_pipeline``, ``cross_validate``, ``fit_calibrator`` (ENTRADA de ``fit-calibrator``).
- ``Calibrator`` e ``load_calibrator``: uso nas sugestões (``suggestions.py``).

Disco:
- Lê: ``dataset.yaml`` + ``pilot.json`` do dataset piloto, fotos e labels, os pesos
  do detector e o ``dataset_snapshot.json`` da execução de treino desses pesos.
- Grava: ``runs/calibrator_<data>_<id>/calibrator.joblib`` (modelo) e
  ``calibrator.json`` (metadados, hashes e validação cruzada).

Onde mexer:
- Nova característica da sugestão: EXTENSAO em ``BASE_FEATURES`` e em
  ``suggestion_features`` (mesma posição). Calibradores antigos deixam de servir.
- Trocar o modelo (ex.: outro C ou outro algoritmo): ``_pipeline``.
- Mudar o critério "o calibrador ganhou": ``cross_validate`` (chave
  ``improves_over_raw_confidence``). ATENCAO: é a trava que impede usar um
  calibrador pior que a confiança crua.
- Mudar o mínimo de exemplos: ``fit_calibrator`` (5 acertos e 5 erros).
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

# ---------------------------------------------------------------------------
# Características de cada sugestão (entrada do calibrador)
# ---------------------------------------------------------------------------

# Versão do formato de calibrator.json (subir ao mudar o significado dos campos).
SCHEMA = 1
# Características numéricas de cada sugestão, NA ORDEM das colunas de
# suggestion_features (depois delas vêm as colunas one-hot class_0, class_1, ...):
#   confidence               confiança crua do YOLO (0-1)
#   logit_confidence         ln(p / (1 - p)) da confiança
#   log_area_fraction        ln(área da caixa / área da foto)
#   log_aspect_ratio         ln(largura / altura) da caixa
#   center_x, center_y       centro da caixa normalizado (0-1)
#   rank_fraction            posição da sugestão no ranking de confiança da foto (0 = a mais confiante, 1 = a menos)
#   log_suggestions_in_image ln(1 + nº de sugestões na foto)
#   max_iou_same_class       maior IoU com outra sugestão da MESMA classe (duplicata?)
#   max_iou_other_class      maior IoU com sugestão de OUTRA classe (dúvida entre classes?)
# EXTENSAO: característica nova = nome aqui (antes das one-hot, ou seja, no fim
# desta lista) + valor na MESMA posição em suggestion_features (antes de *one_hot).
# ATENCAO: calibradores já gravados usam a lista antiga e passam a dar erro de
# dimensão no predict_proba; gere outro com fit-calibrator.
BASE_FEATURES = ["confidence", "logit_confidence", "log_area_fraction", "log_aspect_ratio", "center_x", "center_y",
                 "rank_fraction", "log_suggestions_in_image", "max_iou_same_class", "max_iou_other_class"]


def feature_names(class_count: int) -> list[str]:
    """Nomes de todas as colunas: ``BASE_FEATURES`` + ``class_0`` ... ``class_{n-1}``.

    Parâmetros:
        class_count: nº de classes da taxonomia.

    Retorna:
        Lista de nomes na ordem das colunas da matriz de características.
    """
    return BASE_FEATURES + [f"class_{index}" for index in range(class_count)]


def suggestion_features(boxes: list[dict], width: int, height: int, class_count: int) -> np.ndarray:
    """Matriz (sugestões × características) de uma foto; ordem das linhas = ordem de ``boxes``.

    Parâmetros:
        boxes: sugestões do modelo na foto (``class_id``, ``bbox_xyxy`` em pixels,
            ``confidence`` 0-1).
        width, height: tamanho da foto em pixels (normaliza área e centro).
        class_count: nº de classes (tamanho do one-hot).

    Retorna:
        ``np.ndarray`` float de forma ``(len(boxes), len(BASE_FEATURES) + class_count)``;
        forma ``(0, n)`` quando não há sugestões.

    Erros:
        IndexError se algum class_id >= class_count (sugestão de outra taxonomia).
    """
    rows = []
    # Ranking por confiança (maior primeiro): rank[índice da caixa] = posição 0, 1, 2...
    ranked = sorted(range(len(boxes)), key=lambda index: -boxes[index]["confidence"])
    rank = {index: position for position, index in enumerate(ranked)}
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box["bbox_xyxy"]
        # Largura/altura com mínimo de 1 px: evita log(0) em caixas degeneradas.
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        # Confiança limitada a [1e-4, 1 - 1e-4] para o logit não ir a ±infinito.
        confidence = min(max(float(box["confidence"]), 1e-4), 1 - 1e-4)
        # IoU com as outras sugestões da mesma classe (sinal de caixa duplicada).
        same = [box_iou(box["bbox_xyxy"], other["bbox_xyxy"]) for position, other in enumerate(boxes)
                if position != index and other["class_id"] == box["class_id"]]
        # IoU com sugestões de outra classe (sinal de confusão entre classes).
        other = [box_iou(box["bbox_xyxy"], value["bbox_xyxy"]) for position, value in enumerate(boxes)
                 if position != index and value["class_id"] != box["class_id"]]
        # One-hot da classe: 1.0 na posição do class_id, 0.0 nas demais.
        one_hot = [0.0] * class_count
        one_hot[int(box["class_id"])] = 1.0
        # ATENCAO: a ordem desta linha TEM de ser a de BASE_FEATURES + one-hot.
        rows.append([
            # confidence, logit_confidence = ln(p / (1 - p))
            confidence, math.log(confidence / (1 - confidence)),
            # log_area_fraction = ln(w*h / (W*H)), log_aspect_ratio = ln(w / h)
            math.log(w * h / (width * height)), math.log(w / h),
            # center_x = (x1 + x2) / 2W, center_y = (y1 + y2) / 2H
            (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            # rank_fraction (0 a 1), log_suggestions_in_image = ln(1 + n)
            rank[index] / max(1, len(boxes) - 1), math.log1p(len(boxes)),
            # max_iou_same_class, max_iou_other_class (0 se não há outra caixa) e one-hot
            # EXTENSAO: valor da característica nova aqui, antes de *one_hot.
            max(same, default=0.0), max(other, default=0.0), *one_hot,
        ])
    # reshape garante a forma (0, n) quando a foto não tem sugestões.
    return np.asarray(rows, dtype=float).reshape(len(boxes), len(BASE_FEATURES) + class_count)


# ---------------------------------------------------------------------------
# Exemplos rotulados a partir de um dataset piloto (fotos com caixas aprovadas).
# ---------------------------------------------------------------------------

def trained_image_hashes(weights: Path) -> set[str]:
    """SHA-256 das fotos que estiveram no treino destes pesos (dataset_snapshot.json da execução).

    Parâmetros:
        weights: caminho do .pt treinado (ex.: ``runs/pilot_train_X/fit/weights/best.pt``).

    Retorna:
        Conjunto de hashes (str); vazio se não houver dataset_snapshot.json (ex.:
        pesos oficiais da Ultralytics, que não treinaram nas nossas fotos).

    Chamada por: collect_examples (aqui), model_gate._trained e autolabel.evaluate_autolabel.
    """
    # Três níveis acima: weights/ -> fit/ -> runs/pilot_train_X/ (onde o treino grava o snapshot).
    # ATENCAO: depende dessa estrutura de pastas; pesos copiados para outro lugar
    # perdem a exclusão silenciosamente (conjunto vazio).
    snapshot = Path(weights).resolve().parent.parent.parent / "dataset_snapshot.json"
    if not snapshot.is_file():
        return set()
    files = json.loads(snapshot.read_text(encoding="utf-8"))["files"]
    # ATENCAO: o hash é o NOME do arquivo (stem), porque pilot.py grava cada foto
    # como images/<split>/<sha256>.<ext>. Datasets com outro padrão de nome
    # (ex.: janelas do tile-dataset) não casam com image_sha256 e nada é excluído.
    return {Path(item["path"]).stem for item in files if item["path"].startswith("images/train/")}


def collect_examples(config: ProjectConfig, data_path: Path, weights: str, conf: float = 0.01,
                     include_trained: bool = False) -> dict:
    """Roda o detector nas fotos do dataset piloto e rotula cada sugestão contra as caixas humanas.

    Parâmetros:
        config: configuração do projeto.
        data_path: ``data/pilot/<versão>/dataset.yaml`` (o pilot.json fica na mesma pasta).
        weights: pesos do detector (.pt) a calibrar.
        conf: confiança mínima das sugestões coletadas (baixa, 0.01, para ter
            exemplos de erro e de acerto em toda a faixa).
        include_trained: True usa também fotos do treino desses pesos (enviesado).

    Retorna:
        Dict com ``X`` (matriz de características), ``y`` (0/1), ``groups``
        (scene_group de cada sugestão, para o GroupKFold), ``raw_confidence``,
        ``images`` (resumo por foto), ``skipped_trained_images``, ``names``,
        ``weights`` (caminho absoluto) e ``conf``.

    Erros:
        ValueError de ``check_domain_names`` se as classes do peso não forem as da
        taxonomia; OSError se pilot.json/fotos/labels faltarem.
    """
    # Imports pesados aqui dentro: só carregam a Ultralytics quando o comando roda.
    from .backend import check_domain_names, load_detector
    from .review_data import review_image_size
    from .suggestions import detect_boxes
    from .training import pilot_inference_config

    # 1. Taxonomia, manifesto do dataset e detector (conferido contra a taxonomia).
    names = detection_names(load_taxonomy(config.taxonomy_path))
    root = Path(data_path).resolve().parent
    manifest = json.loads((root / "pilot.json").read_text(encoding="utf-8"))
    model = load_detector(config, weights)
    check_domain_names(model, names)
    # Mesmo imgsz/fatiamento que as sugestões reais usam (o calibrador aprende o que o revisor vê).
    model_config = pilot_inference_config(config, model)
    # 2. Fotos a pular: as do treino desses pesos (a menos que include_trained).
    seen = set() if include_trained else trained_image_hashes(Path(model.ckpt_path))
    features, labels, groups, confidences, used, skipped = [], [], [], [], [], 0
    # 3. Para cada foto: sugestões do modelo, verdade humana e rótulo de cada sugestão.
    for sample in manifest["images"]:
        if sample["image_sha256"] in seen:
            skipped += 1
            continue
        image = root / sample["image"]
        width, height = review_image_size(image)
        truth = yolo_label_boxes(root / sample["label"], width, height)
        boxes = detect_boxes(model, image, model_config, conf)
        # Foto sem sugestão não gera exemplo (nada para rotular), só entra no resumo.
        if not boxes:
            used.append({"image_sha256": sample["image_sha256"], "suggestions": 0, "human_boxes": len(truth)})
            continue
        # y = 1 se a sugestão reencontra caixa humana (mesma classe, IoU >= 0,5).
        hits, _ = match_predictions(boxes, truth)
        features.append(suggestion_features(boxes, width, height, len(names)))
        labels.extend(int(hit) for hit in hits)
        # Grupo = cena: sugestões da mesma cena ficam sempre na mesma dobra da validação.
        groups.extend([sample["scene_group"]] * len(boxes))
        confidences.extend(box["confidence"] for box in boxes)
        used.append({"image_sha256": sample["image_sha256"], "suggestions": len(boxes), "human_boxes": len(truth),
                     "matched": int(sum(hits))})
    # 4. Junta tudo em arrays (matriz vazia com o nº certo de colunas se não houve sugestão).
    return {
        "X": np.vstack(features) if features else np.zeros((0, len(feature_names(len(names))))),
        "y": np.asarray(labels, dtype=int), "groups": np.asarray(groups), "raw_confidence": np.asarray(confidences),
        "images": used, "skipped_trained_images": skipped, "names": names,
        "weights": str(Path(model.ckpt_path).resolve()), "conf": conf,
    }


# ---------------------------------------------------------------------------
# Treino e avaliação do calibrador.
# ---------------------------------------------------------------------------

# Algoritmos aceitos pelo fit-calibrator --algorithm. "lightgbm" exige o pacote lightgbm.
ALGORITHMS = ("logistic", "lightgbm")


def _pipeline(seed: int, algorithm: str = "logistic"):
    """Modelo do calibrador: padronização + regressão logística (scikit-learn) ou LightGBM.

    Parâmetros:
        seed: semente do solver (resultado reproduzível).
        algorithm: "logistic" (padrão) ou "lightgbm" (árvores com gradient boosting;
            capta interações classe × tamanho × confiança que a regressão não vê). Com
            poucas decisões as árvores ficam rasas e muito regularizadas; a validação
            cruzada decide se ganhou da confiança crua, como no logístico.

    Retorna:
        ``sklearn.pipeline.Pipeline`` ainda não treinado.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if algorithm == "lightgbm":
        from lightgbm import LGBMClassifier

        # Poucos exemplos: árvores de até 7 folhas, 20+ exemplos por folha e subamostragem.
        return LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=7, min_child_samples=20,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=seed, n_jobs=1, verbose=-1)
    if algorithm != "logistic":
        raise ValueError(f"Algoritmo de calibração desconhecido: {algorithm} (use {', '.join(ALGORITHMS)}).")

    # C pequeno = regularização forte. Sem class_weight="balanced": ele desloca as
    # probabilidades para cima e destrói a calibração, que é o objetivo aqui.
    # EXTENSAO: para testar outro modelo (ex.: C=1.0 ou outro classificador com
    # predict_proba), troque aqui; a validação cruzada mede se ganhou.
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, random_state=seed))


def cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, raw: np.ndarray, seed: int = 42,
                   algorithm: str = "logistic") -> dict:
    """Probabilidades fora da dobra (GroupKFold por cena) comparadas com a confiança crua.

    Parâmetros:
        X: matriz de características (n_exemplos × n_colunas).
        y: rótulos 0/1.
        groups: cena de cada exemplo (mesmo tamanho de y).
        raw: confiança crua do YOLO de cada exemplo (a referência a vencer).
        seed: semente do modelo.

    Retorna:
        ``{"folds", "group", "calibrated": {brier, roc_auc, average_precision},
        "raw_confidence": {...}, "improves_over_raw_confidence": bool}``.

    Erros:
        ValueError "Calibração exige sugestões em pelo menos 2 cenas diferentes."
        e ValueError do scikit-learn se y tiver uma só classe (fit_calibrator já
        impede isso exigindo 5 acertos e 5 erros).
    """
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold

    # Até 5 dobras, limitado ao nº de cenas distintas (cada dobra precisa de cena própria).
    folds = min(5, len(set(groups.tolist())))
    if folds < 2:
        raise ValueError("Calibração exige sugestões em pelo menos 2 cenas diferentes.")
    # Probabilidade de cada exemplo prevista por um modelo que NÃO o viu no treino.
    out_of_fold = np.full(len(y), np.nan)
    for train_index, test_index in GroupKFold(n_splits=folds).split(X, y, groups):
        if len(set(y[train_index].tolist())) < 2:
            out_of_fold[test_index] = y[train_index].mean()  # dobra sem as duas classes: prevê a taxa base
            continue
        model = _pipeline(seed, algorithm).fit(X[train_index], y[train_index])
        # Coluna 1 do predict_proba = P(y = 1), a probabilidade de aceitação.
        out_of_fold[test_index] = model.predict_proba(X[test_index])[:, 1]

    def scores(probability):
        """Brier, AUC-ROC e precisão média de um vetor de probabilidades contra y.

        Parâmetros:
            probability: np.ndarray com uma probabilidade (0-1) por exemplo.

        Retorna:
            ``{"brier", "roc_auc", "average_precision"}`` com 4 casas.
        """
        # EXTENSAO: métrica nova da validação (ex.: log_loss): acrescente aqui; ela
        # aparece em calibrator.json em "calibrated" e "raw_confidence".
        return {"brier": round(float(brier_score_loss(y, probability)), 4),
                "roc_auc": round(float(roc_auc_score(y, probability)), 4),
                "average_precision": round(float(average_precision_score(y, probability)), 4)}

    # Mesmas métricas para o calibrador (fora da dobra) e para a confiança crua.
    calibrated, baseline = scores(out_of_fold), scores(raw)
    # ATENCAO: trava do projeto. Só é "melhor" se ganhar em Brier (menor) E em AUC (maior);
    # Calibrator recusa calibrador sem esse ganho (a menos de --allow-unproven-calibrator).
    return {"folds": folds, "group": "scene_group", "calibrated": calibrated, "raw_confidence": baseline,
            "improves_over_raw_confidence": calibrated["brier"] < baseline["brier"]
            and calibrated["roc_auc"] > baseline["roc_auc"]}


# ENTRADA: comando ``fit-calibrator`` (cli.py chama esta função).
def fit_calibrator(config: ProjectConfig, data_path: Path, weights: str, conf: float = 0.01,
                   include_trained: bool = False, seed: int = 42, algorithm: str = "logistic") -> Path:
    """Coleta exemplos, avalia por validação cruzada, treina com tudo e grava em runs/calibrator_*.

    Parâmetros:
        config: configuração do projeto.
        data_path: dataset.yaml do dataset piloto com caixas humanas.
        weights: pesos do detector cujas sugestões serão calibradas.
        conf: confiança mínima das sugestões coletadas (padrão 0.01).
        include_trained: usar também fotos do treino (enviesado; só para testes).
        seed: semente da regressão logística.

    Retorna:
        Path da pasta nova ``runs/calibrator_<data>_<id>`` (com calibrator.joblib e
        calibrator.json). Usar depois com ``suggest --calibrator <pasta>``.

    Erros:
        ValueError "Exemplos insuficientes para calibrar: ..." com menos de 5 acertos
        ou 5 erros; os de ``collect_examples`` e ``cross_validate``.
    """
    import joblib
    import sklearn

    # 1. Exemplos rotulados.
    examples = collect_examples(config, data_path, weights, conf, include_trained)
    y = examples["y"]
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    # Mínimo para não fingir aprendizado: acertos e erros em quantidade e cenas distintas.
    if positives < 5 or negatives < 5:
        raise ValueError(f"Exemplos insuficientes para calibrar: {positives} acertos e {negatives} erros "
                         f"(mínimo 5 de cada). Revise mais fotos fora do treino do modelo.")
    # 2. Avaliação honesta (validação cruzada por cena) ANTES de treinar com tudo.
    report = cross_validate(examples["X"], y, examples["groups"], examples["raw_confidence"], seed, algorithm)
    # 3. Modelo final treinado com todos os exemplos.
    model = _pipeline(seed, algorithm).fit(examples["X"], y)
    # 4. Pasta nova em runs/ (nunca sobrescreve) e o modelo serializado.
    run = make_run_directory(config.root, "calibrator")
    model_path = run / "calibrator.joblib"
    joblib.dump(model, model_path)
    # 5. Metadados: hashes (modelo, taxonomia, pesos), características e validação.
    # EXTENSAO: campo novo no calibrator.json: acrescente aqui; para usar na hora de
    # carregar, leia em Calibrator.__init__ (self.metadata).
    metadata = {
        "schema_version": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "acceptance_probability_logistic_regression" if algorithm == "logistic" else f"acceptance_probability_{algorithm}",
        "algorithm": algorithm, "sklearn_version": sklearn.__version__,
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
    """Calibrador carregado e conferido (hash do arquivo e taxonomia) antes do uso.

    Atributos:
        metadata (dict): conteúdo do calibrator.json.
        model: Pipeline do scikit-learn (carregado com joblib).
        class_count (int): nº de classes com que foi treinado.
        directory (Path): pasta runs/calibrator_* de origem.

    Usado por: ``suggestions.py`` (opção ``suggest --calibrator``), via ``load_calibrator``.
    """

    def __init__(self, config: ProjectConfig, path: Path, allow_unproven: bool = False):
        """Carrega e confere o calibrador.

        Parâmetros:
            config: configuração do projeto (taxonomia atual).
            path: pasta runs/calibrator_* ou qualquer arquivo dentro dela.
            allow_unproven: True aceita calibrador sem ganho medido (só testes).

        Erros:
            ValueError quando o hash do .joblib não confere, quando a taxonomia é
            outra ou quando não houve ganho sobre a confiança crua (sem allow_unproven).
        """
        import joblib

        # Aceita a pasta ou um arquivo dela (ex.: o calibrator.json).
        directory = Path(path).resolve()
        directory = directory if directory.is_dir() else directory.parent
        self.metadata = json.loads((directory / "calibrator.json").read_text(encoding="utf-8"))
        model_path = directory / self.metadata["model_file"]
        # joblib usa pickle: só carregue arquivos gerados aqui; o hash impede troca silenciosa.
        if file_hash(model_path) != self.metadata["model_sha256"]:
            raise ValueError("calibrator.joblib não confere com o hash registrado em calibrator.json.")
        # IDs de classe do one-hot só valem com a mesma taxonomia.
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
        """Acrescenta ``acceptance_probability`` e reordena; não remove nenhuma caixa.

        Parâmetros:
            boxes: sugestões de uma foto (são alteradas no lugar, ganhando a chave nova).
            width, height: tamanho da foto em pixels.

        Retorna:
            Nova lista com as mesmas caixas, da maior para a menor probabilidade.
        """
        if not boxes:
            return boxes
        # P(revisor aceita) de cada caixa = coluna 1 do predict_proba.
        probability = self.model.predict_proba(suggestion_features(boxes, width, height, self.class_count))[:, 1]
        for box, value in zip(boxes, probability):
            box["acceptance_probability"] = round(float(value), 4)
        return sorted(boxes, key=lambda box: -box["acceptance_probability"])

    def describe(self) -> dict:
        """Resumo do calibrador para registrar em suggestions.json.

        Retorna:
            ``{"path", "model_sha256", "cross_validation", "proven"}``.
        """
        return {"path": str(self.directory), "model_sha256": self.metadata["model_sha256"],
                "cross_validation": self.metadata["cross_validation"],
                "proven": bool(self.metadata["cross_validation"].get("improves_over_raw_confidence"))}


def load_calibrator(config: ProjectConfig, path: Path | str | None, allow_unproven: bool = False) -> Calibrator | None:
    """Abre o calibrador da opção ``--calibrator``, ou None se ela não foi usada.

    Parâmetros:
        config: configuração do projeto.
        path: caminho informado (relativo à raiz do projeto ou de outra máquina,
            resolvido por ``resolve_local_path``); None/vazio = sem calibrador.
        allow_unproven: repassado a ``Calibrator``.

    Retorna:
        ``Calibrator`` ou None.
    """
    return Calibrator(config, resolve_local_path(str(path), config.root), allow_unproven) if path else None
