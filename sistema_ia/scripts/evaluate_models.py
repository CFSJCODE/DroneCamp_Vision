"""Avaliação comparativa por classe, no mesmo pipeline das sugestões, com gráficos.

Para cada modelo (.pt ou .onnx exportado) e cada conjunto de fotos com gabarito
humano, roda ``detect_boxes`` (fatiamento de ``pilot.tiling`` nas fotos grandes) e
pareia caixas da mesma classe com IoU >= 0,5. Mede por classe:

- na confiança das sugestões (``pilot.suggestion_conf``): acertos, falsos
  positivos, omissões, precisão, recall e F1;
- AP50 (interpolação de todos os pontos, como no VOC) com as predições a partir
  de ``--ap-conf`` (padrão 0,01). Calculado aqui porque o ``val`` da Ultralytics
  avalia a foto inteira reduzida e não reproduz a inferência fatiada.

Conjuntos de fotos:
- ``--data data/pilot/V/dataset.yaml``: splits val e test do dataset piloto (fotos
  vistas no treino de um modelo são marcadas e ficam fora da média daquele modelo);
- ``--review REGISTRO.json FEEDBACK.json``: fotos confirmadas num arquivo de revisão
  (ex.: Lado A em andamento), gabarito independente para modelos que não as viram.

Uso:
  python scripts/evaluate_models.py --data data/pilot/ceasa_v9_piloto_s42/dataset.yaml \
      --review data/reviews/ceasa_lado_a_v1/registry.json data/reviews/ceasa_lado_a_v1/feedback_inbox/X.json \
      --weights runs/export_A/model.onnx --label A --weights runs/export_B/model.onnx --label B \
      --output runs/relatorio_modelos_X

Gera ``report.json``, gráficos PNG e ``index.html`` na pasta de saída. Nenhum
modelo é aprovado por este relatório: ele só organiza a evidência.

Função no projeto:
    Relatório de evidência mais detalhado que ``compare_pilot_models.py``: métricas POR
    CLASSE (acertos, falsos positivos, omissões, precisão, recall, F1 e AP50), em vários
    conjuntos de fotos de uma vez, com gráficos e uma página HTML simples para ler fora
    da plataforma. Também é a biblioteca de métricas do ``scripts/gpu_eval_watcher.py``
    (que importa ``evaluate``, ``pilot_sets`` e ``trained_on``).

O que contém:
    Constantes: ``IOU_MATCH``, ``DEDUP_IOU``, ``COLORS``.
    Gabaritos:  ``_dedup``, ``pilot_sets``, ``review_set``.
    Métricas:   ``_match``, ``average_precision``, ``evaluate``, ``trained_on``.
    Saída:      ``_charts`` (PNGs com matplotlib), ``_page`` (index.html).
    Programa:   ``main``.

Quem chama:
    Rodado à mão pelo terminal (ver ``Uso``). ``gpu_eval_watcher.py`` importa funções daqui.

O que lê e grava no disco:
    Lê:    dataset.yaml/pilot.json + fotos e labels do dataset piloto; registry.json e o
           arquivo de revisão exportado (feedback); as fotos originais apontadas pelo
           registro; os pesos e o ``dataset_snapshot.json`` da execução de origem;
           configs/project.yaml e configs/taxonomy.json.
    Grava: na pasta ``--output`` (que NÃO pode existir antes): ``report.json``,
           ``totais_<conjunto>.png``, ``classes_<conjunto>.png`` e ``index.html``.

Onde mexer:
    - Para mudar o critério de acerto: ``IOU_MATCH`` (mesmo valor em compare_pilot_models.py).
    - Para outra fonte de gabarito (ex.: outro formato de revisão): crie uma função como
      ``review_set`` que devolva ``{nome_do_conjunto: [{"image", "sha256", "truth"}]}`` e
      chame-a em ``main`` junto com ``pilot_sets``/``review_set``, com um argumento novo.
    - Para uma métrica nova por classe: calcule em ``evaluate`` (dicionário ``result[name]``)
      e, se for total, em ``totals``; para mostrar, acrescente coluna em ``_page`` e/ou
      gráfico em ``_charts``.
    - Para mais de seis modelos com cores distintas: acrescente cores em ``COLORS``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys
import time

# Raiz do sistema_ia (pasta acima de scripts/); usada para achar src/ e reancorar caminhos.
ROOT = Path(__file__).resolve().parents[1]
# Permite importar o pacote sem instalá-lo (src/) e o compare_pilot_models.py vizinho (scripts/).
# ATENCAO: precisa vir antes dos imports de dronecamp_ia abaixo (por isso os "noqa: E402").
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dronecamp_ia.backend import check_domain_names, load_inference_model  # noqa: E402
from dronecamp_ia.config import detection_names, load_config, load_taxonomy  # noqa: E402
from dronecamp_ia.io import file_hash, resolve_local_path, write_json  # noqa: E402
from dronecamp_ia.pilot import box_iou  # noqa: E402
from dronecamp_ia.suggestions import detect_boxes  # noqa: E402
from dronecamp_ia.training import pilot_inference_config  # noqa: E402

# IoU mínima (0 a 1) entre sugestão e caixa humana da mesma classe para contar acerto.
# ATENCAO: mesmo valor de compare_pilot_models.IOU_MATCH; mudar só aqui torna os relatórios incomparáveis.
IOU_MATCH = 0.5
# IoU a partir da qual duas caixas humanas da mesma classe são tratadas como duplicadas
# (mesma política do build-pilot-data). Baixar o valor junta mais caixas; subir junta menos.
DEDUP_IOU = 0.7


# ---------------------------------------------------------------------------
# Gabaritos: fotos com caixas humanas.
# ---------------------------------------------------------------------------

def _dedup(boxes: list[dict]) -> list[dict]:
    """Mesma classe com IoU >= 0,7 conta uma vez (mesma política do build-pilot-data).

    Evita que duas caixas humanas quase iguais (marcadas duas vezes) virem uma omissão
    garantida, já que uma sugestão só pode parear com uma delas.

    Parâmetros:
        boxes (list[dict]): caixas com ``"class_id"`` e ``"bbox_xyxy"`` ([x1, y1, x2, y2] em pixels).

    Retorna:
        list[dict]: caixas mantidas (a primeira de cada grupo de duplicadas), com
        ``class_id`` convertido para int e coordenadas para float.

    Chamada por: ``pilot_sets`` e ``review_set``.
    """
    kept = []
    # Mantém a caixa só se nenhuma já mantida for da mesma classe com IoU >= DEDUP_IOU.
    for box in boxes:
        if not any(k["class_id"] == box["class_id"] and box_iou(k["bbox_xyxy"], box["bbox_xyxy"]) >= DEDUP_IOU for k in kept):
            kept.append({"class_id": int(box["class_id"]), "bbox_xyxy": [float(v) for v in box["bbox_xyxy"]]})
    return kept


def pilot_sets(data: Path) -> dict[str, list[dict]]:
    """Fotos val/test do dataset piloto com caixas do label YOLO.

    Parâmetros:
        data (Path): caminho do dataset.yaml do dataset piloto (a pasta dele tem o pilot.json).

    Retorna:
        dict: ``{"<versão>:val": [foto, ...], "<versão>:test": [...]}``, onde cada foto é
        ``{"image": Path, "sha256": str, "truth": [caixas em pixels, sem duplicadas]}``.
        Conjuntos sem nenhuma foto não aparecem.

    Erros:
        OSError se faltar o pilot.json, uma foto ou uma label.

    Chamada por: ``main`` (opção --data) e ``gpu_eval_watcher.evaluate_snapshot``.
    """
    # Import tardio: o compare_pilot_models.py está em scripts/ (posto no sys.path acima).
    from compare_pilot_models import ground_truth

    root = data.resolve().parent
    manifest = json.loads((root / "pilot.json").read_text(encoding="utf-8"))
    sets = defaultdict(list)
    for sample in manifest["images"]:
        # Só validação e teste: as fotos de treino não medem generalização.
        # EXTENSAO: para avaliar também o treino, acrescente "train" neste conjunto.
        if sample["split"] in {"val", "test"}:
            sets[f"{manifest['version']}:{sample['split']}"].append({
                "image": root / sample["image"], "sha256": sample["image_sha256"], "truth": _dedup(ground_truth(root, sample))})
    return dict(sets)


def review_set(registry: Path, feedback: Path) -> dict[str, list[dict]]:
    """Fotos confirmadas (confirmed_complete) de um arquivo de revisão exportado.

    Usa como gabarito as decisões humanas de um arquivo de revisão (o JSON baixado pelo
    botão "Salvar revisão" da plataforma), mesmo antes de ele ser importado.

    Parâmetros:
        registry (Path): ``data/reviews/<inspeção>/registry.json``.
        feedback (Path): arquivo de revisão exportado (``feedback_inbox/*.json`` ou o baixado).

    Retorna:
        dict: ``{"<versão>:revisado": [{"image": Path, "sha256": str, "truth": [caixas]}]}``.
        Entram só fotos com ``confirmed_complete`` e situação ``positive`` ou ``negative``
        (ambíguas, excluídas e pendentes ficam de fora).

    Erros:
        ValueError("Arquivo de revisão de outra versão do registro.") quando o
        ``registry_sha256`` do arquivo não bate com o hash do registry.json.
        KeyError se o arquivo de revisão citar uma foto que não está no registro.

    Chamada por: ``main`` (opção --review).
    """
    registry_data = json.loads(registry.read_text(encoding="utf-8"))
    # Índice do registro por hash da foto (para achar o caminho da foto original).
    sources = {item["image_sha256"]: item for item in registry_data["images"]}
    exported = json.loads(feedback.read_text(encoding="utf-8"))
    # ATENCAO: o hash vincula as decisões ao registro exato; com outro registro as caixas
    # poderiam apontar para fotos diferentes.
    if exported.get("registry_sha256") != file_hash(registry):
        raise ValueError("Arquivo de revisão de outra versão do registro.")
    photos = []
    for item in exported["images"]:
        # Só decisões fechadas e confirmadas valem como gabarito.
        # EXTENSAO: para aceitar outra situação como gabarito, acrescente-a neste conjunto.
        if not item.get("confirmed_complete") or item.get("status") not in {"positive", "negative"}:
            continue
        # O caminho gravado pode ser de outra máquina (ex.: E:\...); reancora neste clone.
        source = resolve_local_path(sources[item["image_sha256"]]["source_path"], ROOT)
        photos.append({"image": source, "sha256": item["image_sha256"], "truth": _dedup(item["boxes"])})
    return {f"{registry_data['version']}:revisado": photos}


# ---------------------------------------------------------------------------
# Métricas.
# ---------------------------------------------------------------------------

def _match(predictions: list[dict], truth: list[dict]) -> list[tuple[dict, bool]]:
    """Pareamento guloso por confiança: cada caixa humana conta uma vez.

    Mesma regra de ``compare_pilot_models.match``, mas devolve cada sugestão com o
    resultado (acerto ou não), necessário para o AP e para contar falsos positivos.

    Parâmetros:
        predictions (list[dict]): sugestões com ``"class_id"``, ``"confidence"`` e ``"bbox_xyxy"``.
        truth (list[dict]): caixas humanas com ``"class_id"`` e ``"bbox_xyxy"``.

    Retorna:
        list[tuple[dict, bool]]: ``(sugestão, acertou)`` para cada sugestão, em ordem
        decrescente de confiança.

    Chamada por: ``evaluate`` (duas vezes por foto: para o AP e para o ponto de operação).
    """
    used, out = set(), []
    # Maior confiança escolhe primeiro.
    for prediction in sorted(predictions, key=lambda value: -value["confidence"]):
        # Começa no limiar: só aceita IoU >= IOU_MATCH.
        best, best_iou = None, IOU_MATCH
        for index, box in enumerate(truth):
            # Caixa humana já usada ou de outra classe não serve.
            if index in used or box["class_id"] != prediction["class_id"]:
                continue
            overlap = box_iou(box["bbox_xyxy"], prediction["bbox_xyxy"])
            if overlap >= best_iou:
                best, best_iou = index, overlap
        if best is not None:
            used.add(best)
        out.append((prediction, best is not None))
    return out


def average_precision(scored: list[tuple[float, bool]], positives: int) -> float | None:
    """AP com interpolação de todos os pontos (envelope da curva precisão × recall).

    Fórmula: AP = soma, para cada sugestão em ordem decrescente de confiança, de
    (recall_i - recall_anterior) x (maior precisão dali em diante). O "maior dali em
    diante" é o envelope que torna a curva precisão x recall monotônica (padrão VOC).

    Parâmetros:
        scored (list[tuple[float, bool]]): ``(confiança, acertou)`` de cada sugestão da classe.
        positives (int): total de caixas humanas da classe (denominador do recall).

    Retorna:
        float | None: AP entre 0 e 1, arredondado a 4 casas; None quando a classe não
        tem nenhuma caixa humana (AP indefinido, fica fora da média mAP50).

    Chamada por: ``evaluate``.
    """
    if positives == 0:
        return None
    scored = sorted(scored, key=lambda value: -value[0])
    tp = fp = 0
    # points: (recall, precisão) acumulados depois de cada sugestão.
    points = []
    for _, hit in scored:
        # bool soma como 0/1: acerto incrementa tp, erro incrementa fp.
        tp, fp = tp + hit, fp + (not hit)
        points.append((tp / positives, tp / (tp + fp)))
    ap, previous_recall = 0.0, 0.0
    for index, (recall, _) in enumerate(points):
        # Envelope: maior precisão a partir deste ponto (O(n²), suficiente para poucas fotos).
        precision = max(p for _, p in points[index:])
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return round(ap, 4)


def evaluate(model, config, photos: list[dict], names: list[str], conf: float, ap_conf: float,
             excluded: set[str]) -> dict:
    """Métricas por classe numa lista de fotos; fotos em ``excluded`` (vistas no treino) ficam fora.

    Roda a inferência UMA vez por foto com a confiança baixa ``ap_conf`` e usa o mesmo
    resultado para duas contas: o AP50 (todas as predições) e o ponto de operação das
    sugestões (só as predições com confiança >= ``conf``).

    Parâmetros:
        model: modelo de ``load_inference_model`` (.pt ou .onnx).
        config: ``ProjectConfig`` já ajustado por ``pilot_inference_config`` (imgsz, fatiamento).
        photos (list[dict]): fotos de ``pilot_sets``/``review_set`` (``image``, ``sha256``, ``truth``).
        names (list[str]): slugs das classes na ordem dos IDs (``detection_names``).
        conf (float): confiança das sugestões (0 a 1), ex.: 0,15.
        ap_conf (float): confiança mínima usada para o AP (0 a 1), ex.: 0,01.
        excluded (set[str]): hashes das fotos vistas no treino deste modelo (ignoradas).

    Retorna:
        dict: ``{"totals": {...}, "per_class": {slug: {...}}}``.
        Por classe: ``truth`` (caixas humanas), ``tp`` (acertos), ``fp`` (falsos positivos),
        ``fn`` (omissões = truth - tp), ``precision``, ``recall``, ``f1`` e ``ap50``
        (None quando não há como calcular). Totais: as mesmas somas, ``precision``,
        ``recall``, ``map50`` (média dos AP50 das classes com gabarito), ``photos``
        (fotos contadas) e ``seconds_per_photo``.

    Erros:
        IndexError se uma caixa ou predição tiver ``class_id`` fora de ``names``
        (ex.: caixa de uma classe de fase 2 num arquivo de revisão).

    Chamada por: ``main`` e ``gpu_eval_watcher.evaluate_snapshot``.
    """
    # Acumuladores por classe; "scored" guarda (confiança, acertou) para o AP.
    # EXTENSAO: contagem nova por classe = nova chave aqui, a soma no laço e o cálculo em result[name].
    per_class = {name: {"truth": 0, "tp": 0, "fp": 0, "scored": []} for name in names}
    started, counted = time.perf_counter(), 0
    for photo in photos:
        # ATENCAO: fotos vistas no treino deste modelo nunca entram na conta dele (memorização).
        if photo["sha256"] in excluded:
            continue
        counted += 1
        # Uma inferência por foto, na confiança mínima do AP.
        predictions = detect_boxes(model, photo["image"], config, ap_conf)
        # Total de caixas humanas por classe (denominador do recall).
        for box in photo["truth"]:
            per_class[names[box["class_id"]]]["truth"] += 1
        # AP: todas as predições >= ap_conf, pareadas uma vez.
        for prediction, hit in _match(predictions, photo["truth"]):
            per_class[names[prediction["class_id"]]]["scored"].append((prediction["confidence"], hit))
        # Ponto de operação das sugestões: só caixas >= conf, pareadas de novo.
        for prediction, hit in _match([p for p in predictions if p["confidence"] >= conf], photo["truth"]):
            per_class[names[prediction["class_id"]]]["tp" if hit else "fp"] += 1
    # Métricas por classe a partir dos acumuladores.
    result = {}
    for name, entry in per_class.items():
        tp, fp, truth = entry["tp"], entry["fp"], entry["truth"]
        # Precisão = acertos / sugestões; None se a classe não teve sugestão.
        precision = tp / (tp + fp) if tp + fp else None
        # Recall = acertos / caixas humanas; None se a classe não tem gabarito.
        recall = tp / truth if truth else None
        # F1 = média harmônica de precisão e recall; 0 se uma delas for 0/None; None sem gabarito.
        f1 = (2 * precision * recall / (precision + recall) if precision and recall else 0.0) if truth else None
        # EXTENSAO: métrica nova por classe = nova chave neste dicionário (aparece no report.json).
        result[name] = {"truth": truth, "tp": tp, "fp": fp, "fn": truth - tp,
                        "precision": None if precision is None else round(precision, 4),
                        "recall": None if recall is None else round(recall, 4),
                        "f1": None if f1 is None else round(f1, 4),
                        "ap50": average_precision(entry["scored"], truth)}
    # Totais: somas das contagens de todas as classes.
    totals = {key: sum(value[key] for value in result.values()) for key in ("truth", "tp", "fp", "fn")}
    # mAP50 = média simples dos AP50 das classes que têm gabarito (classes sem caixa ficam fora).
    measured = [value["ap50"] for value in result.values() if value["ap50"] is not None]
    totals["precision"] = round(totals["tp"] / (totals["tp"] + totals["fp"]), 4) if totals["tp"] + totals["fp"] else None
    totals["recall"] = round(totals["tp"] / totals["truth"], 4) if totals["truth"] else None
    totals["map50"] = round(sum(measured) / len(measured), 4) if measured else None
    totals["photos"] = counted
    # Tempo médio por foto (inclui leitura da foto e fatiamento), em segundos.
    totals["seconds_per_photo"] = round((time.perf_counter() - started) / counted, 2) if counted else None
    return {"totals": totals, "per_class": result}


def trained_on(model) -> set[str]:
    """SHA-256 das fotos de treino do modelo (dataset_snapshot do run de origem).

    Os pesos de origem ficam em ``runs/<execução>/fit/weights/<arquivo>.pt``; o snapshot
    fica na pasta da execução (terceiro pai). Fotos do dataset piloto se chamam
    ``<sha256>.<ext>``, então o nome sem extensão é o hash.

    Parâmetros:
        model: modelo de ``load_inference_model``; usa ``model.dronecamp_source`` (o .pt de
            origem, também para ONNX).

    Retorna:
        set[str]: hashes das fotos de ``images/train/``; conjunto vazio se o snapshot não
        existir ou estiver ilegível (então nenhuma foto é excluída).

    Chamada por: ``main`` e ``gpu_eval_watcher.evaluate_snapshot``.
    """
    # parent = weights/, parent.parent = fit/, parent.parent.parent = runs/<execução>/.
    snapshot = Path(model.dronecamp_source).resolve().parent.parent.parent / "dataset_snapshot.json"
    try:
        files = json.loads(snapshot.read_text(encoding="utf-8"))["files"]
    except (OSError, ValueError, KeyError):
        # Sem snapshot (peso externo, cópia avulsa): não exclui nada.
        return set()
    return {Path(item["path"]).stem for item in files if item["path"].startswith("images/train/")}


# ---------------------------------------------------------------------------
# Gráficos e página.
# ---------------------------------------------------------------------------

# Cores das barras, uma por modelo na ordem de --weights (repetem a partir do sétimo).
# EXTENSAO: mais modelos com cores distintas = acrescente cores hexadecimais aqui.
COLORS = ["#2a78d6", "#e3893b", "#4c9a5f", "#b5577a", "#7a5fc4", "#8a8f2e"]


def _charts(report: dict, output: Path) -> list[tuple[str, str]]:
    """Desenha os gráficos PNG do relatório com matplotlib (sem janela, backend "Agg").

    Para cada conjunto de fotos gera:
      1. ``totais_<conjunto>.png``: barras de precisão, recall e mAP50 por modelo;
      2. ``classes_<conjunto>.png``: recall e falsos positivos por classe (só classes com
         gabarito ou com algum erro em algum modelo).

    Parâmetros:
        report (dict): relatório montado em ``main`` (chaves ``models``, ``sets``,
            ``classes``, ``conf``, ``ap_conf``).
        output (Path): pasta onde os PNGs são gravados (já criada).

    Retorna:
        list[tuple[str, str]]: ``(nome_do_arquivo, legenda)`` de cada gráfico, na ordem
        em que aparecem na página.

    Chamada por: ``main``.
    EXTENSAO: gráfico novo = mais um bloco "3." dentro do laço, terminando com
    ``fig.savefig(output / nome)``, ``plt.close(fig)`` e ``charts.append((nome, legenda))``.
    """
    # Import tardio: matplotlib só é necessário aqui. "Agg" desenha em arquivo, sem tela.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [model["label"] for model in report["models"]]
    charts = []
    for set_name in report["sets"]:
        # Nome de arquivo seguro: "versão:val" vira "versão_val" (":" é proibido no Windows).
        safe = set_name.replace(":", "_")
        # 1. Totais: precisão, recall e mAP50 por modelo.
        fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=130)
        metrics = [("precision", "Precisão"), ("recall", "Recall"), ("map50", "mAP50")]
        # Barras agrupadas: largura total 0,8 dividida entre os modelos.
        width = 0.8 / len(labels)
        for index, model in enumerate(report["models"]):
            totals = model["sets"][set_name]["totals"]
            # None (sem dado) vira 0 no gráfico.
            values = [totals[key] or 0 for key, _ in metrics]
            # Cada modelo deslocado index * width dentro do grupo de cada métrica.
            bars = ax.bar([m + index * width for m in range(len(metrics))], values, width, label=model["label"],
                          color=COLORS[index % len(COLORS)])
            ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], fontsize=7, padding=2)
        # Rótulo no centro de cada grupo de barras.
        ax.set_xticks([m + width * (len(labels) - 1) / 2 for m in range(len(metrics))], [n for _, n in metrics])
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{set_name} — totais (confiança {report['conf']:.2f}; AP desde {report['ap_conf']:.2f})", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        name = f"totais_{safe}.png"
        fig.savefig(output / name)
        plt.close(fig)
        charts.append((name, f"{set_name}: totais"))
        # 2. Por classe: recall e falsos positivos (só classes com gabarito ou com erros).
        classes = [c for c in report["classes"]
                   if any(m["sets"][set_name]["per_class"][c]["truth"] or m["sets"][set_name]["per_class"][c]["fp"]
                          for m in report["models"])]
        if not classes:
            continue
        # Altura cresce com o número de classes (0,42 polegada por classe + margem).
        fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(classes) + 1.6), dpi=130, sharey=True)
        height = 0.8 / len(labels)
        for index, model in enumerate(report["models"]):
            per_class = model["sets"][set_name]["per_class"]
            positions = [c + index * height for c in range(len(classes))]
            recall = [per_class[c]["recall"] or 0 for c in classes]
            fps = [per_class[c]["fp"] for c in classes]
            # Esquerda: recall por classe; direita: falsos positivos por classe.
            axes[0].barh(positions, recall, height, color=COLORS[index % len(COLORS)], label=model["label"])
            axes[1].barh(positions, fps, height, color=COLORS[index % len(COLORS)])
        # n= número de caixas humanas da classe (igual para todos os modelos, salvo fotos excluídas).
        truth = [report["models"][0]["sets"][set_name]["per_class"][c]["truth"] for c in classes]
        axes[0].set_yticks([c + height * (len(labels) - 1) / 2 for c in range(len(classes))],
                           [f"{c} (n={n})" for c, n in zip(classes, truth)], fontsize=7)
        # Primeira classe no topo.
        axes[0].invert_yaxis()
        axes[0].set_xlim(0, 1)
        axes[0].set_title("Recall por classe (ocorrências reencontradas)", fontsize=9)
        axes[1].set_title("Falsos positivos por classe", fontsize=9)
        axes[0].legend(fontsize=7, frameon=False, loc="lower right")
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="x", labelsize=7)
        fig.suptitle(set_name, fontsize=9)
        fig.tight_layout()
        name = f"classes_{safe}.png"
        fig.savefig(output / name)
        plt.close(fig)
        charts.append((name, f"{set_name}: por classe"))
    return charts


def _page(report: dict, charts: list[tuple[str, str]]) -> str:
    """Monta o ``index.html`` do relatório (página única, sem dependências externas).

    Parâmetros:
        report (dict): relatório montado em ``main``.
        charts (list[tuple[str, str]]): ``(arquivo PNG, legenda)`` de ``_charts``.

    Retorna:
        str: HTML completo com cabeçalho, avisos, lista de modelos, tabela de totais
        (uma linha por conjunto x modelo) e as figuras.

    Chamada por: ``main``.
    ATENCAO: todo texto vindo de arquivos (nomes, rótulos, caminhos) passa por
    ``html.escape``; mantenha isso em campos novos.
    EXTENSAO: coluna nova na tabela = um ``<th>`` no cabeçalho (fim da função) e um
    ``<td>`` na mesma posição dentro de ``rows.append``.
    """
    rows = []
    # Uma linha da tabela por (conjunto, modelo).
    for set_name in report["sets"]:
        for model in report["models"]:
            t = model["sets"][set_name]["totals"]
            # Formata número com 3 casas; None vira travessão.
            fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
            rows.append(f"<tr><td>{html.escape(set_name)}</td><td>{html.escape(model['label'])}</td><td>{t['photos']}</td>"
                        f"<td>{t['truth']}</td><td>{t['tp']}</td><td>{t['fp']}</td><td>{t['fn']}</td>"
                        f"<td>{fmt(t['precision'])}</td><td>{fmt(t['recall'])}</td><td>{fmt(t['map50'])}</td>"
                        f"<td>{t['seconds_per_photo']}</td></tr>")
    # Figuras com os PNGs gravados na mesma pasta (caminho relativo).
    images = "".join(f"<figure><img src='{name}' alt='{html.escape(caption)}'><figcaption>{html.escape(caption)}</figcaption></figure>"
                     for name, caption in charts)
    # Lista dos modelos avaliados (rótulo, caminho e runtime).
    models = "".join(f"<li><b>{html.escape(m['label'])}</b>: {html.escape(m['weights'])} ({html.escape(m['runtime'])})</li>"
                     for m in report["models"])
    # Página final; as chaves duplas {{ }} são chaves literais do CSS dentro da f-string.
    return f"""<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Comparação de modelos</title>
<style>body{{font:14px/1.5 system-ui,sans-serif;margin:24px auto;max-width:1100px;padding:0 16px;color:#1d2733;background:#fff}}
table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{border-bottom:1px solid #dde3ea;padding:4px 6px;text-align:right}}
td:first-child,td:nth-child(2),th:first-child,th:nth-child(2){{text-align:left}}figure{{margin:16px 0}}img{{max-width:100%}}
figcaption{{font-size:12px;color:#5b6773}}.note{{background:#f3f6fa;padding:8px 12px;border-radius:6px}}</style>
<h1>Comparação de modelos</h1><p>{html.escape(report['created_at'])} · IoU {IOU_MATCH} · confiança das sugestões {report['conf']}
· fatiamento: {html.escape(json.dumps(report['tiling'], ensure_ascii=False))}</p>
<p class="note">Métricas no mesmo pipeline das sugestões. Fotos vistas no treino de um modelo ficam fora da conta daquele modelo.
Uma única edificação não mede generalização; nenhum modelo é aprovado por este relatório.</p>
<ul>{models}</ul>
<table><tr><th>Conjunto</th><th>Modelo</th><th>Fotos</th><th>Gabarito</th><th>Acertos</th><th>FP</th><th>Omissões</th>
<th>Precisão</th><th>Recall</th><th>mAP50</th><th>s/foto</th></tr>{''.join(rows)}</table>{images}</html>"""


# ---------------------------------------------------------------------------
# Programa principal.
# ---------------------------------------------------------------------------

def main() -> int:
    """ENTRADA: lê os argumentos, monta os conjuntos de fotos, avalia cada modelo e grava o relatório.

    Parâmetros (linha de comando):
        --data PATH                 dataset.yaml de um dataset piloto (repetível).
        --review REGISTRO FEEDBACK  registry.json + arquivo de revisão exportado (repetível).
        --weights PATH              peso .pt ou .onnx (repetível; obrigatório).
        --label TEXTO               rótulo de cada peso, na mesma ordem de --weights; sem
                                    rótulo, usa o nome da pasta da execução de origem.
        --onnx-provider             ``directml`` (GPU, padrão) ou ``cpu``.
        --no-tiling                 foto inteira reduzida (sem fatiar fotos grandes).
        --ap-conf FLOAT             confiança mínima das predições usadas no AP (padrão 0,01).
        --output PASTA              pasta de saída; NÃO pode existir.

    Retorna:
        int: 0 quando termina.

    Erros:
        SystemExit("Saída já existe; relatórios não são sobrescritos.") se a pasta existir;
        ValueError de ``review_set`` ou de ``check_domain_names``.
    """
    # 1. Argumentos.
    # EXTENSAO: opção nova = parser.add_argument(...) aqui e o uso nas etapas seguintes.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, action="append", default=[], help="dataset.yaml de um dataset piloto.")
    parser.add_argument("--review", type=Path, nargs=2, action="append", default=[], metavar=("REGISTRO", "FEEDBACK"))
    parser.add_argument("--weights", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--onnx-provider", choices=["directml", "cpu"], default="directml")
    parser.add_argument("--no-tiling", action="store_true")
    parser.add_argument("--ap-conf", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # ATENCAO: relatórios nunca são sobrescritos; escolha uma pasta nova a cada execução.
    if args.output.exists():
        raise SystemExit("Saída já existe; relatórios não são sobrescritos.")
    # 2. Configuração, classes e confiança das sugestões (pilot.suggestion_conf, padrão 0,15).
    config = load_config()
    names = detection_names(load_taxonomy(config.taxonomy_path))
    conf = float(config.pilot.get("suggestion_conf", 0.15))
    # Fatiamento registrado no relatório (None = desligado por --no-tiling).
    tiling = None if args.no_tiling else config.pilot.get("tiling")
    # --no-tiling: troca pilot.tiling na configuração para detect_boxes usar a foto inteira.
    if args.no_tiling:
        config = replace(config, pilot={**config.pilot, "tiling": {"enabled": False}})
    # 3. Conjuntos de fotos com gabarito: {nome: [fotos]}.
    # EXTENSAO: fonte nova de gabarito = sets.update(sua_funcao(...)) aqui, com um argumento novo.
    sets: dict[str, list[dict]] = {}
    for data in args.data:
        sets.update(pilot_sets(data))
    for registry, feedback in args.review:
        sets.update(review_set(registry, feedback))
    # 4. Cabeçalho do relatório (report.json). "sets" guarda só o número de fotos de cada conjunto.
    # EXTENSAO: campo novo no topo do report.json = nova chave aqui.
    report = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "conf": conf,
              "ap_conf": args.ap_conf, "iou_match": IOU_MATCH, "tiling": tiling, "classes": names,
              "sets": {name: len(photos) for name, photos in sets.items()}, "models": []}
    # 5. Avalia cada modelo em todos os conjuntos.
    for index, weights in enumerate(args.weights):
        model = load_inference_model(config, str(weights), args.onnx_provider)
        # ATENCAO: recusa modelos cujas classes não batem com a taxonomia (ex.: pesos COCO).
        check_domain_names(model, names)
        model_config = pilot_inference_config(config, model)
        # Fotos vistas no treino deste modelo (ficam fora da conta dele).
        seen = trained_on(model)
        # Rótulo: o --label na mesma posição; senão parts[-4] de runs/<execução>/fit/weights/best.pt = <execução>.
        label = args.label[index] if index < len(args.label) else Path(model.dronecamp_source).parts[-4]
        entry = {"label": label, "weights": str(weights), "weights_sha256": file_hash(weights),
                 "source_checkpoint": str(model.dronecamp_source), "runtime": model.dronecamp_runtime, "sets": {}}
        for set_name, photos in sets.items():
            entry["sets"][set_name] = evaluate(model, model_config, photos, names, conf, args.ap_conf, seen)
            t = entry["sets"][set_name]["totals"]
            # Resumo de uma linha no terminal por modelo x conjunto.
            print(f"{label:24s} {set_name:36s} fotos {t['photos']:3d} | acertos {t['tp']:3d}/{t['truth']:3d}"
                  f" | FP {t['fp']:4d} | P {t['precision']} R {t['recall']} mAP50 {t['map50']} | {t['seconds_per_photo']} s/foto")
        report["models"].append(entry)
    # 6. Grava report.json, os gráficos e a página (a pasta é criada só agora, depois das contas).
    args.output.mkdir(parents=True)
    write_json(args.output / "report.json", report)
    charts = _charts(report, args.output)
    (args.output / "index.html").write_text(_page(report, charts), encoding="utf-8")
    print(f"Relatório: {args.output / 'index.html'}")
    return 0


# ENTRADA: execução direta pelo terminal; o código de saída é o retorno de main().
if __name__ == "__main__":
    raise SystemExit(main())
