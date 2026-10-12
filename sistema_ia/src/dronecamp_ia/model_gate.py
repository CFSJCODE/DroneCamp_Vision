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
Na versão atual do código, quem importa este módulo é: ``cli.py``
(``compare-models`` e ``measure-models``), ``cycle.py`` (``retrain-cycle``, passos
gate e medição) e ``gpu_eval.py`` (``evaluate_model`` durante o treino). O
``hparam_bandit.py`` não o importa diretamente.

Fórmulas:
- Pareamento e IoU: ``matching.py`` (mesma classe e IoU >= 0,5).
- Precisão = TP / (TP + FP); recall = TP / (TP + FN).
- AP (precisão média, uma classe, um limiar de IoU), interpolação em todos os pontos:
      1. ordene as sugestões da classe pela confiança, da maior para a menor;
      2. após cada sugestão k: recall_k = TP_k / P  e  precisão_k = TP_k / k
         (P = nº de caixas humanas da classe);
      3. envelope: precisão_k ← max(precisão_j, j >= k) (curva sempre descendo);
      4. AP = Σ (recall_k - recall_{k-1}) · precisão_k.
- mAP50 = média simples das AP50 das classes MEDIDAS (classe sem caixa humana
  fica de fora, não conta como zero).

O que o arquivo contém:
- ``AP_CONF``, ``STRATA``, ``REGIMES``: constantes.
- ``average_precision``, ``_summarize``, ``evaluate_model``: métricas de um modelo.
- ``decide``: regra de adoção.
- ``_trained``, ``_load_models``, ``_series``, ``_strata_counts``: auxiliares.
- ``compare_models`` (ENTRADA de ``compare-models``) e ``measure_models``
  (ENTRADA de ``measure-models``).

Disco:
- Lê: ``pilot.json`` e fotos/labels do dataset piloto; pesos (.pt ou .onnx);
  ``dataset_snapshot.json`` das execuções de treino (fotos a excluir).
- Grava: ``runs/gate_<data>_<id>/gate.json`` ou ``runs/measure_<data>_<id>/measurement.json``
  (ou a pasta ``output`` indicada).

Onde mexer:
- Regra de adoção: ``decide`` (ATENCAO: registre o motivo em docs/decisoes.md).
- Nova métrica por classe: ``matching.per_class_report`` (contagens) ou
  ``_summarize`` (métricas que dependem das confianças, como AP).
- Campo novo no gate.json / measurement.json: os EXTENSAO em ``compare_models``
  e ``measure_models``; quem lê é ``model_report.py``.
- Novo regime de inferência: ``REGIMES`` + ``evaluate_model``.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from .config import ProjectConfig, detection_names, load_taxonomy
from .io import file_hash, make_run_directory, write_json
from .matching import image_errors, match_predictions, per_class_report, yolo_label_boxes

# ---------------------------------------------------------------------------
# Constantes e precisão média (AP)
# ---------------------------------------------------------------------------

# Confiança mínima das caixas usadas no cálculo de AP (0-1). Bem baixa de propósito:
# AP é a área da curva inteira, e cortar em 0,25 esconderia o fim da curva.
# As contagens operacionais (TP/FP/FN, precisão, recall) usam ``conf`` (a confiança
# das sugestões), não este valor.
AP_CONF = 0.001  # AP precisa das caixas de baixa confiança, como no model.val


def average_precision(scored: list[tuple[float, bool]], positives: int) -> float | None:
    """AP com interpolação em todos os pontos (VOC 2010+/COCO, um limiar de IoU).

    Parâmetros:
        scored: uma tupla ``(confiança, acertou)`` por sugestão da classe, em
            qualquer ordem (é ordenada aqui).
        positives: nº de caixas humanas da classe (P).

    Retorna:
        AP entre 0 e 1 com 4 casas, ou None se a classe não tem caixa humana
        (não medida; diferente de zero).
    """
    if positives == 0:
        return None
    # 1. Da maior para a menor confiança.
    scored = sorted(scored, key=lambda value: -value[0])
    tp = fp = 0
    # Ponto inicial da curva: recall 0, precisão 1.
    recalls, precisions = [0.0], [1.0]
    # 2. Acumula TP/FP sugestão a sugestão (hit é bool: soma como 1/0).
    for _, hit in scored:
        tp, fp = tp + hit, fp + (not hit)
        recalls.append(tp / positives)
        precisions.append(tp / (tp + fp))
    # Envelope monotônico da precisão e soma das áreas onde o recall cresce.
    # 3. Percorre de trás para frente: cada precisão vira o máximo das seguintes.
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    # 4. AP = Σ (Δrecall · precisão); FP não aumenta o recall, então não soma área.
    return round(sum((recalls[index] - recalls[index - 1]) * precisions[index] for index in range(1, len(recalls))), 4)


# Estratos do relatório: fotos que o fatiamento divide em janelas (drone, 20 MP)
# e fotos pequenas avaliadas inteiras (ex.: fotos extraídas do laudo).
# EXTENSAO: estrato novo exige a regra de classificação em evaluate_model (linha "stratum = ...").
STRATA = ("fotos_fatiadas", "fotos_inteiras")
# Regimes de inferência: "janelas" usa pilot.tiling (como as sugestões reais);
# "inteira" força a foto inteira (como os modelos antigos eram usados).
# ATENCAO: estes nomes são aceitos na CLI (measure-models --regime) e gravados
# em measurement.json; mudar o texto quebra a leitura do model_report.py.
REGIMES = ("janelas", "inteira")


def _summarize(errors: list[dict], scored: dict, positives: dict, names: list[str]) -> dict:
    """Relatório por classe (TP/FP/FN, precisão, recall) mais AP50 e mAP50.

    Parâmetros:
        errors: lista de ``matching.image_errors`` (uma por foto).
        scored: ``{class_id: [(confiança, acertou), ...]}`` para o AP.
        positives: ``{class_id: nº de caixas humanas}``.
        names: nomes das classes na ordem dos IDs.

    Retorna:
        O dict de ``per_class_report`` com ``ap50`` em cada classe e, em ``_total``,
        ``map50``, ``classes_measured`` e ``images``.
    """
    report = per_class_report(errors, names)
    aps = []
    # AP50 por classe; só classes com caixa humana entram na média.
    for class_id, name in enumerate(names):
        report[name]["ap50"] = average_precision(scored[class_id], positives[class_id])
        if report[name]["ap50"] is not None:
            aps.append(report[name]["ap50"])
    # mAP50 = média das AP50 medidas.
    # EXTENSAO: métrica agregada nova (ex.: mAP só das classes raras): acrescente em _total aqui.
    report["_total"]["map50"] = round(sum(aps) / len(aps), 4) if aps else None
    report["_total"]["classes_measured"] = len(aps)
    report["_total"]["images"] = len(errors)
    return report


# ---------------------------------------------------------------------------
# Avaliação de um modelo nas fotos com caixas humanas
# ---------------------------------------------------------------------------

def evaluate_model(model, model_config: ProjectConfig, root: Path, samples: list[dict], names: list[str],
                   conf: float, regime: str = "janelas") -> dict:
    """Métricas por classe de um modelo nas amostras dadas (``pilot.json``).

    ``regime``: ``"janelas"`` usa ``pilot.tiling`` (o regime das sugestões; fotos
    pequenas continuam inteiras), ``"inteira"`` força a foto inteira (regime dos
    modelos antigos). O relatório traz também ``_strata``: as mesmas métricas
    separadas em fotos fatiadas (drone, 20 MP) e fotos inteiras (laudo), porque o
    pedido é confiabilidade nas fotos de drone, e ``_regime``.

    Parâmetros:
        model: detector carregado (``backend.load_inference_model``; .pt ou .onnx).
        model_config: configuração ajustada ao modelo (``pilot_inference_config``: imgsz etc.).
        root: pasta do dataset piloto (onde está o pilot.json).
        samples: entradas de ``pilot.json["images"]`` a avaliar (``image``, ``label``...).
        names: nomes das classes na ordem dos IDs.
        conf: confiança mínima para as contagens operacionais (o que o revisor vê).
        regime: ``"janelas"`` ou ``"inteira"``.

    Retorna:
        Relatório de ``_summarize`` (todas as fotos) + ``_strata`` (por estrato,
        só os que têm fotos) + ``_regime``.

    Erros:
        ValueError "regime deve ser um de ..." para regime desconhecido.

    Chamada por: compare_models, measure_models e gpu_eval.py.
    """
    from .review_data import review_image_size
    from .suggestions import detect_boxes, tiling_plan

    if regime not in REGIMES:
        raise ValueError(f"regime deve ser um de {REGIMES}.")
    # None = usar pilot.tiling da configuração; {"enabled": False} = foto inteira.
    tiling = None if regime == "janelas" else {"enabled": False}
    # Um "balde" por recorte do relatório: "_all" e um por estrato. Cada balde é
    # (lista de erros por foto, {classe: [(conf, acerto)]}, {classe: nº caixas humanas}).
    buckets = {key: ([], defaultdict(list), defaultdict(int)) for key in ("_all", *STRATA)}
    for sample in samples:
        # 1. Foto, tamanho real e caixas humanas em pixels.
        image = root / sample["image"]
        width, height = review_image_size(image)
        truth = yolo_label_boxes(root / sample["label"], width, height)
        # 2. Inferência com confiança mínima AP_CONF (curva completa para o AP).
        predictions = detect_boxes(model, image, model_config, AP_CONF, tiling)
        hits, _ = match_predictions(predictions, truth)
        # 3. Estrato pelo TAMANHO da foto (se a configuração a fatiaria), mesmo no regime "inteira".
        stratum = STRATA[0] if tiling_plan(model_config, width, height) else STRATA[1]
        # 4. A mesma foto soma no balde geral e no do seu estrato.
        for key in ("_all", stratum):
            errors, scored, positives = buckets[key]
            for prediction, hit in zip(predictions, hits):
                scored[prediction["class_id"]].append((prediction["confidence"], hit))
            for box in truth:
                positives[box["class_id"]] += 1
            # Contagens operacionais na confiança das sugestões (o que o revisor vê).
            errors.append(image_errors([value for value in predictions if value["confidence"] >= conf], truth))
    # 5. Resumo geral e por estrato (estrato sem fotos fica de fora).
    report = _summarize(*buckets["_all"], names)
    report["_strata"] = {key: _summarize(*buckets[key], names) for key in STRATA if buckets[key][0]}
    report["_regime"] = regime
    return report


# ---------------------------------------------------------------------------
# Regra de adoção
# ---------------------------------------------------------------------------

# ATENCAO: esta é a regra de adoção de modelo. Afrouxar os limites abaixo faz um modelo pior
# parecer melhor; mude só com justificativa registrada em docs/decisoes.md.
def decide(baseline: dict, candidate: dict, names: list[str], min_map_gain: float = 0.01,
           max_class_recall_drop: float = 0.1, min_class_boxes: int = 3) -> dict:
    """Aplica a regra de adoção e lista cada motivo de recusa e cada regressão.

    Parâmetros:
        baseline: métricas do modelo atual (saída de ``evaluate_model``).
        candidate: métricas do candidato, nas mesmas fotos.
        names: nomes das classes.
        min_map_gain: ganho mínimo de mAP50 (absoluto; 0.01 = 1 ponto percentual).
        max_class_recall_drop: queda máxima de recall por classe (absoluta; 0.1 = 10 p.p.).
        min_class_boxes: só classes com pelo menos isso de caixas humanas entram na
            análise de regressão (com 1 ou 2 caixas o recall oscila demais).

    Retorna:
        ``{"adopt": bool, "reasons": [texto], "regressions": [{class, recall_before,
        recall_after, human_boxes}], "rule": {os três limites}}``. ``adopt`` só é
        True sem nenhum motivo de recusa.
    """
    reasons, regressions = [], []
    # Critério 1: ganho de mAP50.
    base_map, cand_map = baseline["_total"]["map50"], candidate["_total"]["map50"]
    if base_map is None or cand_map is None:
        reasons.append("Sem caixas humanas avaliáveis nas fotos fora do treino.")
    elif cand_map < base_map + min_map_gain:
        reasons.append(f"mAP50 {cand_map:.3f} não supera o atual {base_map:.3f} em {min_map_gain:.3f}.")
    # Critério 2: recall total não pode cair (None = sem caixa, tratado como 0).
    base_recall, cand_recall = baseline["_total"]["recall"] or 0.0, candidate["_total"]["recall"] or 0.0
    if cand_recall < base_recall:
        reasons.append(f"Recall total caiu de {base_recall:.3f} para {cand_recall:.3f}.")
    # Critério 3: regressão por classe.
    for name in names:
        before, after = baseline[name], candidate[name]
        # Classe com poucas caixas ou sem recall medido no modelo atual não é julgada.
        if before["human_boxes"] < min_class_boxes or before["recall"] is None:
            continue
        drop = before["recall"] - (after["recall"] or 0.0)
        if drop > max_class_recall_drop:
            regressions.append({"class": name, "recall_before": before["recall"], "recall_after": after["recall"],
                                "human_boxes": before["human_boxes"]})
    if regressions:
        reasons.append("Regressão de recall em: " + ", ".join(item["class"] for item in regressions) + ".")
    # EXTENSAO: critério novo (ex.: FP total não pode subir mais que X%): acrescente
    # um bloco acima que faça reasons.append(...) e registre o limite em "rule".
    return {"adopt": not reasons, "reasons": reasons, "regressions": regressions,
            "rule": {"min_map_gain": min_map_gain, "max_class_recall_drop": max_class_recall_drop,
                     "min_class_boxes": min_class_boxes}}


# ---------------------------------------------------------------------------
# Auxiliares: carregar modelos e montar as séries
# ---------------------------------------------------------------------------

def _trained(weights: Path) -> set[str]:
    """Hashes das fotos de treino destes pesos (delegado a ``calibration.trained_image_hashes``).

    Parâmetros:
        weights: caminho do .pt de origem.

    Retorna:
        Conjunto de SHA-256 (vazio se a execução não tiver dataset_snapshot.json).
    """
    from .calibration import trained_image_hashes

    return trained_image_hashes(weights)


def _load_models(config: ProjectConfig, weights: list[str], names: list[str], onnx_provider: str) -> tuple[list, set]:
    """Abre cada peso (.pt ou .onnx) e junta as fotos vistas no treino de qualquer um.

    Parâmetros:
        config: configuração do projeto.
        weights: caminhos dos pesos, na ordem desejada.
        names: nomes das classes (cada peso é conferido contra eles).
        onnx_provider: ``"directml"`` (GPU) ou ``"cpu"`` para pesos .onnx.

    Retorna:
        ``(loaded, excluded)``: ``loaded`` é uma lista de dicts ``{"model", "config",
        "source" (o .pt de origem), "used" (o arquivo realmente executado)}``;
        ``excluded`` é a UNIÃO dos hashes de treino de todos os pesos.

    Erros:
        ValueError de ``check_domain_names`` (classes do peso ≠ taxonomia).
    """
    from .backend import check_domain_names, load_inference_model
    from .training import pilot_inference_config

    loaded, excluded = [], set()
    for path in weights:
        model = load_inference_model(config, path, onnx_provider)
        check_domain_names(model, names)
        # source = .pt de origem (achar o snapshot de treino); used = .onnx se for ONNX.
        source = Path(model.dronecamp_source).resolve()
        used = Path(getattr(model, "dronecamp_onnx", None) or source).resolve()
        loaded.append({"model": model, "config": pilot_inference_config(config, model), "source": source, "used": used})
        # União: foto de treino de QUALQUER modelo sai da avaliação de todos.
        excluded |= _trained(source)
    return loaded, excluded


def _series(entry: dict, report: dict) -> dict:
    """Monta a "série" de um modelo para o JSON (identificação + métricas).

    Parâmetros:
        entry: item de ``_load_models``.
        report: saída de ``evaluate_model`` (é alterado: ``_strata`` e ``_regime`` são retirados).

    Retorna:
        ``{"weights", "sha256", "source_checkpoint", "runtime", "imgsz", "regime",
        "metrics", "strata"}``.
    """
    # pop: tira os campos auxiliares de "metrics" e os guarda em chaves próprias.
    strata = report.pop("_strata", {})
    regime = report.pop("_regime", None)
    # EXTENSAO: identificação nova por modelo (ex.: data do treino): acrescente aqui;
    # vale para gate.json e measurement.json ao mesmo tempo.
    return {"weights": str(entry["used"]), "sha256": file_hash(entry["used"]), "source_checkpoint": str(entry["source"]),
            "runtime": entry["model"].dronecamp_runtime, "imgsz": entry["config"].prediction["imgsz"],
            "regime": regime, "metrics": report, "strata": strata}


def _strata_counts(strata: dict) -> dict:
    """Nº de fotos de cada estrato.

    Parâmetros:
        strata: ``{estrato: relatório}`` (de ``_series``) ou None.

    Retorna:
        ``{estrato: nº de fotos}``.
    """
    return {key: value["_total"]["images"] for key, value in (strata or {}).items()}


# ---------------------------------------------------------------------------
# Comandos: compare-models (gate de adoção) e measure-models (N modelos)
# ---------------------------------------------------------------------------

# ENTRADA: comando ``compare-models`` (cli.py) e passo "gate" do ``retrain-cycle`` (cycle.py).
def compare_models(config: ProjectConfig, data_path: Path, baseline: str, candidate: str,
                   splits: tuple[str, ...] = ("test",), conf: float | None = None, onnx_provider: str = "directml",
                   output: Path | None = None, **rule) -> Path:
    """Roda os dois modelos nas mesmas fotos fora do treino de ambos e grava gate.json.

    Pesos ``.onnx`` rodam no ONNX Runtime (``onnx_provider``: GPU DirectML ou CPU);
    a decisão usa o regime das sugestões (janelas); ``metrics_whole`` registra a
    foto inteira para comparar com os modelos antigos.

    Parâmetros:
        config: configuração do projeto.
        data_path: dataset.yaml do dataset piloto (pilot.json na mesma pasta).
        baseline: pesos do modelo atual (.pt ou .onnx).
        candidate: pesos do candidato.
        splits: splits do pilot.json avaliados (padrão só ``test``).
        conf: confiança das contagens operacionais; None = ``pilot.suggestion_conf``
            ou ``prediction.conf`` do project.yaml.
        onnx_provider: ``"directml"`` ou ``"cpu"``.
        output: pasta já criada para o gate.json; None = ``runs/gate_<data>_<id>``.
        **rule: limites repassados a ``decide`` (min_map_gain, max_class_recall_drop,
            min_class_boxes). A CLI não os expõe hoje.

    Retorna:
        Path da pasta com ``gate.json``.

    Erros:
        ValueError "Nenhuma foto do split escolhido ficou fora do treino dos dois modelos."
        e os de ``_load_models``/``evaluate_model``.
    """
    # 1. Taxonomia, manifesto, confiança e modelos.
    names = detection_names(load_taxonomy(config.taxonomy_path))
    root = Path(data_path).resolve().parent
    manifest = json.loads((root / "pilot.json").read_text(encoding="utf-8"))
    conf = float(conf if conf is not None else config.pilot.get("suggestion_conf", config.prediction["conf"]))
    loaded, excluded = _load_models(config, [baseline, candidate], names, onnx_provider)
    models = dict(zip(("baseline", "candidate"), loaded))
    # 2. Fotos avaliadas: do split pedido e fora do treino de ambos.
    samples = [sample for sample in manifest["images"]
               if sample["split"] in splits and sample["image_sha256"] not in excluded]
    if not samples:
        raise ValueError("Nenhuma foto do split escolhido ficou fora do treino dos dois modelos.")
    # 3. Métricas no regime das sugestões (janelas) e, só para registro, na foto inteira.
    series = {role: _series(entry, evaluate_model(entry["model"], entry["config"], root, samples, names, conf))
              for role, entry in models.items()}
    whole = {role: evaluate_model(entry["model"], entry["config"], root, samples, names, conf, regime="inteira")
             for role, entry in models.items()}
    # 4. Decisão de adoção (só com as métricas em janelas).
    decision = decide(series["baseline"]["metrics"], series["candidate"]["metrics"], names, **rule)
    # 5. Grava gate.json (pasta nova em runs/ quando output não é dado).
    run = output if output is not None else make_run_directory(config.root, "gate")
    # EXTENSAO: campo novo no gate.json: acrescente neste dict; quem lê é
    # model_report.py (report-models) e cycle.py (decisão do ciclo).
    # ATENCAO: "iou_match" está fixo em 0.5 aqui (não usa matching.IOU_MATCH);
    # se mudar IOU_MATCH, atualize este valor também, senão o JSON mente.
    # ATENCAO: "production_ready" é sempre False; a adoção só recomenda o candidato.
    write_json(run / "gate.json", {
        "schema_version": 1, "dataset": str(Path(data_path).resolve()), "pilot_sha256": file_hash(root / "pilot.json"),
        "splits": list(splits), "conf": conf, "ap_conf": AP_CONF, "iou_match": 0.5,
        "tiling": config.pilot.get("tiling"), "strata": _strata_counts(series["baseline"]["strata"]),
        "images_evaluated": len(samples), "images_excluded_seen_in_training": len(excluded),
        "buildings": sorted({str(sample.get("building_group")) for sample in samples}),
        # Avaliação independente = pelo menos 3 edificações diferentes nas fotos medidas.
        "independent_evaluation": len({sample.get("building_group") for sample in samples}) >= 3,
        # baseline e candidate: série + métricas na foto inteira (sem os campos auxiliares).
        **{role: {**series[role], "metrics_whole": {key: value for key, value in whole[role].items()
                                                     if key not in ("_strata", "_regime")}}
           for role in ("baseline", "candidate")},
        **decision, "production_ready": False,
        "note": "adopt=true recomenda o candidato para sugestões; o modelo anterior continua preservado.",
    })
    return run


# ENTRADA: comando ``measure-models`` (cli.py) e passo "medição" do ``retrain-cycle`` (cycle.py).
def measure_models(config: ProjectConfig, data_path: Path, weights: list[str], splits: tuple[str, ...] = ("test",),
                   conf: float | None = None, onnx_provider: str = "directml", regimes: tuple[str, ...] | None = None,
                   output: Path | None = None) -> Path:
    """Mede N modelos nas MESMAS fotos (fora do treino de todos) e grava ``measurement.json``.

    É a entrada do ``report-models``: gráficos comparáveis exigem o mesmo conjunto
    de fotos e a mesma régua. ``regimes``: ``("janelas", "inteira")`` por padrão
    quando ``pilot.tiling`` está ligado (cada modelo vira uma série por regime),
    senão só ``inteira``. Nada aqui decide adoção; isso é do ``compare-models``.

    Parâmetros:
        config: configuração do projeto.
        data_path: dataset.yaml do dataset piloto.
        weights: lista de pesos (.pt/.onnx), um por modelo.
        splits: splits avaliados (padrão ``test``).
        conf: confiança das contagens operacionais (None = a das sugestões).
        onnx_provider: ``"directml"`` ou ``"cpu"``.
        regimes: regimes a medir; None = padrão descrito acima.
        output: pasta já criada; None = ``runs/measure_<data>_<id>``.

    Retorna:
        Path da pasta com ``measurement.json``.

    Erros:
        ValueError "regimes aceita 'janelas' (exige pilot.tiling) e 'inteira'." e
        "Nenhuma foto do split escolhido ficou fora do treino de todos os modelos."
    """
    # 1. Taxonomia, manifesto, confiança e regimes.
    names = detection_names(load_taxonomy(config.taxonomy_path))
    root = Path(data_path).resolve().parent
    manifest = json.loads((root / "pilot.json").read_text(encoding="utf-8"))
    conf = float(conf if conf is not None else config.pilot.get("suggestion_conf", config.prediction["conf"]))
    tiling = config.pilot.get("tiling") or {}
    # Fatiamento ligado = existe pilot.tiling e ele não diz enabled: false.
    tiled = bool(tiling) and tiling.get("enabled", True)
    regimes = tuple(regimes or (REGIMES if tiled else ("inteira",)))
    # "janelas" sem pilot.tiling seria igual a "inteira" com outro nome: recusa.
    if any(regime not in REGIMES for regime in regimes) or ("janelas" in regimes and not tiled):
        raise ValueError("regimes aceita 'janelas' (exige pilot.tiling) e 'inteira'.")
    # 2. Modelos e fotos fora do treino de TODOS.
    loaded, excluded = _load_models(config, list(weights), names, onnx_provider)
    samples = [sample for sample in manifest["images"]
               if sample["split"] in splits and sample["image_sha256"] not in excluded]
    if not samples:
        raise ValueError("Nenhuma foto do split escolhido ficou fora do treino de todos os modelos.")
    # 3. Uma série por (modelo, regime), nessa ordem.
    series = [_series(entry, evaluate_model(entry["model"], entry["config"], root, samples, names, conf, regime))
              for entry in loaded for regime in regimes]
    # 4. Grava measurement.json.
    run = output if output is not None else make_run_directory(config.root, "measure")
    # EXTENSAO: campo novo no measurement.json: acrescente neste dict; quem lê é
    # model_report.py (report-models --measurement).
    # ATENCAO: "iou_match" também está fixo em 0.5 aqui (ver compare_models).
    write_json(run / "measurement.json", {
        "schema_version": 1, "dataset": str(Path(data_path).resolve()), "pilot_sha256": file_hash(root / "pilot.json"),
        "splits": list(splits), "conf": conf, "ap_conf": AP_CONF, "iou_match": 0.5, "tiling": tiling or None,
        "images_evaluated": len(samples), "images_excluded_seen_in_training": len(excluded),
        # Lista das fotos medidas: permite provar que todas as séries usaram as mesmas fotos.
        "images": [{"image_sha256": sample["image_sha256"], "split": sample["split"]} for sample in samples],
        "strata": _strata_counts(series[0]["strata"]) if series else {},
        "buildings": sorted({str(sample.get("building_group")) for sample in samples}),
        "models": series, "production_ready": False,
        "note": "Mesmas fotos e mesma régua para todos os modelos; nenhuma adoção é decidida aqui.",
    })
    return run
