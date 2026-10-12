"""Compara pesos piloto nas mesmas fotos de um dataset piloto, com inferência real.

Para cada split: caixas humanas reencontradas (mesma classe, IoU >= 0,5) e
sugestões corretas na confiança das sugestões. Fotos que estiveram no treino de
um modelo são contadas à parte, para não confundir memorização com melhora.

Uso: python scripts/compare_pilot_models.py --data data/pilot/V/dataset.yaml \
        --weights A.pt --weights B.pt --output runs/compare_X.json

Função no projeto:
    Etapa 5 do ciclo de fine-tuning ("comparar antes de adotar"). Roda cada modelo nas
    MESMAS fotos de um dataset piloto, com o mesmo pipeline das sugestões
    (``suggestions.detect_boxes``, com fatiamento de fotos grandes), e conta acertos por
    divisão (train/val/test). O JSON gerado é lido pela plataforma (seção Modelos,
    "Comparação entre modelos") via ``operations.py`` e ``model_report.py``.

O que contém:
    - ``IOU_MATCH``: limiar de sobreposição para uma sugestão contar como acerto.
    - ``ground_truth``: caixas humanas de uma foto, lidas da label YOLO e convertidas para pixels.
    - ``match``: pareamento guloso sugestão x caixa humana; devolve o número de acertos.
    - ``trained_images``: SHA-256 das fotos que estiveram no treino de um peso.
    - ``main``: lê os argumentos, roda os modelos e grava o JSON.
    ``ground_truth`` também é importada por ``scripts/evaluate_models.py``.

Quem chama:
    Rodado à mão pelo terminal (ver ``Uso`` acima e o README). A plataforma só LÊ o
    resultado (``runs/compare_*.json``); não inicia este script.

O que lê e grava no disco:
    Lê:    o dataset.yaml (só para achar a pasta), ``pilot.json`` do dataset piloto, as fotos
           e labels listadas nele, os pesos (.pt ou .onnx com export.json) e o
           ``dataset_snapshot.json`` da execução de origem de cada peso; configs/project.yaml
           e configs/taxonomy.json (via ``load_config``/``load_taxonomy``).
    Grava: o JSON de ``--output`` (com ``write_json``; um arquivo existente é sobrescrito).

Formato do JSON gravado:
    {"dataset", "pilot_sha256", "conf", "iou_match", "tiling",
     "models": [{"weights", "weights_sha256", "runtime", "source_checkpoint", "imgsz",
                 "splits": {"val": {"images", "truth", "found", "suggestions", "correct"},
                            "train (visto no treino)": {...}, ...}}]}
    ATENCAO: a plataforma (renderCompare no platform.js) lê exatamente estas chaves;
    renomear quebra a tabela da seção Modelos.

Onde mexer:
    - Para mudar o critério de acerto: ``IOU_MATCH`` (e o mesmo valor em evaluate_models.py
      e matching.py, que usam a mesma régua).
    - Para acrescentar uma contagem por divisão: crie a chave no ``defaultdict`` de ``totals``
      em ``main``, some o valor no laço das fotos e mostre na tabela de renderCompare (platform.js).
    - Para uma opção nova de linha de comando: ``parser.add_argument`` em ``main``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

# Permite rodar o script sem instalar o pacote: acrescenta sistema_ia/src ao caminho de import.
# ATENCAO: precisa vir antes dos imports de dronecamp_ia abaixo (por isso os "noqa: E402").
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronecamp_ia.backend import check_domain_names, load_inference_model  # noqa: E402
from dronecamp_ia.config import detection_names, load_config, load_taxonomy  # noqa: E402
from dronecamp_ia.io import file_hash, write_json  # noqa: E402
from dronecamp_ia.pilot import box_iou  # noqa: E402
from dronecamp_ia.review_data import review_image_size  # noqa: E402
from dronecamp_ia.suggestions import detect_boxes  # noqa: E402
from dronecamp_ia.training import pilot_inference_config  # noqa: E402

# ---------------------------------------------------------------------------
# Parâmetros fixos da comparação
# ---------------------------------------------------------------------------

# IoU mínima (0 a 1) entre sugestão e caixa humana da mesma classe para contar acerto.
# Subir o valor exige caixas mais justas (menos acertos); baixar aceita caixas frouxas.
# ATENCAO: mudar aqui torna o JSON novo incomparável com os antigos (o valor fica gravado em "iou_match").
IOU_MATCH = 0.5


# ---------------------------------------------------------------------------
# Gabarito e pareamento
# ---------------------------------------------------------------------------

def ground_truth(root: Path, sample: dict) -> list[dict]:
    """Lê as caixas humanas de uma foto do dataset piloto e as converte para pixels.

    A label YOLO guarda ``classe cx cy w h`` normalizados (0 a 1, centro e tamanho);
    aqui eles viram ``[x1, y1, x2, y2]`` em pixels da foto original, o mesmo formato
    que ``detect_boxes`` devolve, para os dois serem comparados por IoU.

    Parâmetros:
        root (Path): pasta do dataset piloto (onde está o pilot.json).
        sample (dict): item de ``pilot.json["images"]``; usa ``sample["image"]`` e
            ``sample["label"]`` (caminhos relativos a ``root``, ex.: "images/val/<sha>.jpg").

    Retorna:
        list[dict]: uma entrada por caixa, ``{"class_id": int, "bbox_xyxy": [x1, y1, x2, y2]}``
        (floats, em pixels). Lista vazia para foto negativa (label vazia).

    Erros:
        OSError se a foto ou a label não existir; ValueError se uma linha da label não
        tiver exatamente 5 valores.

    Chamada por: ``main`` (aqui) e ``scripts/evaluate_models.pilot_sets``.
    """
    # Tamanho real da foto em pixels (sem rotação EXIF implícita), para desnormalizar.
    width, height = review_image_size(root / sample["image"])
    boxes = []
    # Uma linha por caixa; linhas em branco são ignoradas.
    for line in (root / sample["label"]).read_text(encoding="utf-8").splitlines():
        if line.strip():
            class_id, cx, cy, w, h = line.split()
            # Desnormaliza: centro (cx, cy) e tamanho (w, h) multiplicados pela largura/altura.
            cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
            # Centro + tamanho -> cantos: x1 = cx - w/2, y1 = cy - h/2, x2 = cx + w/2, y2 = cy + h/2.
            boxes.append({"class_id": int(class_id), "bbox_xyxy": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]})
    return boxes


def match(predictions: list[dict], truth: list[dict]) -> int:
    """Pareamento guloso por confiança: cada caixa humana conta uma vez.

    As sugestões são percorridas da maior para a menor confiança; cada uma fica com a
    caixa humana livre, da mesma classe, de maior IoU (no mínimo ``IOU_MATCH``).
    Uma caixa humana já usada não pode ser reaproveitada por outra sugestão.

    Parâmetros:
        predictions (list[dict]): sugestões de ``detect_boxes``, cada uma com
            ``"class_id"``, ``"confidence"`` (0 a 1) e ``"bbox_xyxy"`` em pixels.
        truth (list[dict]): caixas humanas de ``ground_truth`` (mesmo formato de bbox).

    Retorna:
        int: número de pares formados (acertos). Como o pareamento é um para um, é ao
        mesmo tempo "caixas humanas reencontradas" e "sugestões corretas".

    Chamada por: ``main``. Há uma cópia que devolve os pares em ``evaluate_models._match``.
    """
    # used: índices das caixas humanas já pareadas; hits: total de acertos.
    used, hits = set(), 0
    # Maior confiança primeiro: a sugestão mais confiável escolhe antes.
    for prediction in sorted(predictions, key=lambda value: -value["confidence"]):
        # best_iou começa no limiar: só aceita IoU >= IOU_MATCH.
        best, best_iou = None, IOU_MATCH
        for index, box in enumerate(truth):
            # Pula caixas já usadas e de outra classe (classe errada nunca é acerto).
            if index in used or box["class_id"] != prediction["class_id"]:
                continue
            overlap = box_iou(box["bbox_xyxy"], prediction["bbox_xyxy"])
            if overlap >= best_iou:
                best, best_iou = index, overlap
        # Achou par: marca a caixa humana como usada e conta o acerto.
        if best is not None:
            used.add(best)
            hits += 1
    return hits


def trained_images(weights: Path) -> set[str]:
    """SHA-256 das fotos que estiveram no treino destes pesos.

    Lê o ``dataset_snapshot.json`` gravado pelo treino na pasta da execução. Os pesos
    ficam em ``runs/<execução>/fit/weights/<arquivo>.pt``, então a execução é o
    terceiro pai do arquivo. No dataset piloto, cada foto se chama ``<sha256>.<ext>``;
    por isso o nome sem extensão (``stem``) já é o hash.

    Parâmetros:
        weights (Path): o .pt de origem (para ONNX, ``model.dronecamp_source``).

    Retorna:
        set[str]: hashes das fotos de ``images/train/`` do snapshot.

    Erros:
        OSError (FileNotFoundError) se o snapshot não existir, por exemplo para pesos
        fora de ``runs/<execução>/fit/weights``.
        ATENCAO: diferente de ``evaluate_models.trained_on`` e de
        ``calibration.trained_image_hashes``, aqui a falta do snapshot interrompe o script.

    Chamada por: ``main``.
    """
    # parent = weights/, parent.parent = fit/, parent.parent.parent = runs/<execução>/.
    snapshot = weights.resolve().parent.parent.parent / "dataset_snapshot.json"
    files = json.loads(snapshot.read_text(encoding="utf-8"))["files"]
    # Só as fotos do split de treino (labels e val/test ficam de fora).
    return {Path(item["path"]).stem for item in files if item["path"].startswith("images/train/")}


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main() -> int:
    """ENTRADA: lê os argumentos, roda cada peso nas fotos do dataset e grava o JSON.

    Parâmetros (linha de comando):
        --data PATH        dataset.yaml do dataset piloto (a pasta dele tem o pilot.json).
        --weights PATH     peso a comparar (.pt ou .onnx); repita para vários modelos.
        --conf FLOAT       confiança mínima das sugestões (0 a 1); padrão:
                           ``pilot.suggestion_conf`` do project.yaml (0,15 se ausente).
        --no-tiling        foto inteira reduzida ao imgsz (sem fatiar fotos grandes).
        --onnx-provider    ``directml`` (GPU, padrão) ou ``cpu``, só para pesos .onnx.
        --output PATH      JSON de saída (ex.: runs/compare_X.json).

    Retorna:
        int: 0 (código de saída do processo) quando termina.

    Erros:
        ValueError de ``check_domain_names`` quando as classes do modelo não batem com a
        taxonomia; OSError se faltar pilot.json, foto, label ou dataset_snapshot.json.
    """
    # 1. Argumentos da linha de comando.
    # EXTENSAO: opção nova = parser.add_argument(...) aqui e o uso logo abaixo.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, action="append", required=True)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--no-tiling", action="store_true",
                        help="Foto inteira reduzida ao imgsz (como antes); padrão: pilot.tiling para fotos grandes.")
    parser.add_argument("--onnx-provider", choices=["directml", "cpu"], default="directml",
                        help="Para --weights .onnx (comando export): directml usa a GPU.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # 2. Configuração do projeto e nomes das classes na ordem dos IDs (contrato com os pesos).
    config = load_config()
    names = detection_names(load_taxonomy(config.taxonomy_path))
    # 3. Manifesto do dataset piloto: lista das fotos com split, hash, foto e label.
    root = args.data.resolve().parent
    manifest = json.loads((root / "pilot.json").read_text(encoding="utf-8"))
    # Confiança: --conf tem prioridade; senão pilot.suggestion_conf; senão 0,15.
    conf = float(args.conf if args.conf is not None else config.pilot.get("suggestion_conf", 0.15))
    # 4. Cabeçalho do relatório (o hash do pilot.json prova em quais fotos a conta foi feita).
    # EXTENSAO: campo novo no topo do JSON = acrescente a chave aqui.
    report = {"dataset": str(args.data), "pilot_sha256": file_hash(root / "pilot.json"), "conf": conf,
              "iou_match": IOU_MATCH, "tiling": None if args.no_tiling else config.pilot.get("tiling"), "models": []}
    # 5. Um modelo por vez, sempre nas mesmas fotos.
    for weights in args.weights:
        # .onnx roda na GPU (DirectML); fotos de treino vêm do .pt de origem (export.json).
        model = load_inference_model(config, str(weights), args.onnx_provider)
        # ATENCAO: recusa modelos cujas classes não batem com a taxonomia (ex.: pesos COCO).
        check_domain_names(model, names)
        # Pesos piloto inferem no imgsz em que foram treinados (não no da produção).
        model_config = pilot_inference_config(config, model)
        seen_in_training = trained_images(model.dronecamp_source)
        # Contagens por grupo (split, ou split + "(visto no treino)"):
        # images = fotos, truth = caixas humanas, found = humanas reencontradas,
        # suggestions = sugestões feitas, correct = sugestões que acertaram.
        # EXTENSAO: contagem nova = nova chave neste dicionário + a soma no laço abaixo.
        totals = defaultdict(lambda: {"images": 0, "truth": 0, "found": 0, "suggestions": 0, "correct": 0})
        for sample in manifest["images"]:
            truth = ground_truth(root, sample)
            # Inferência real; {"enabled": False} desliga o fatiamento, None usa pilot.tiling.
            predictions = detect_boxes(model, root / sample["image"], model_config, conf,
                                       {"enabled": False} if args.no_tiling else None)
            hits = match(predictions, truth)
            # Fotos vistas no treino deste modelo vão para um grupo separado (memorização, não melhora).
            group = sample["split"] + (" (visto no treino)" if sample["image_sha256"] in seen_in_training else "")
            entry = totals[group]
            entry["images"] += 1
            entry["truth"] += len(truth)
            # Pareamento um para um: found e correct recebem o mesmo número de acertos;
            # a diferença está no denominador (truth para recall, suggestions para precisão).
            entry["found"] += hits
            entry["suggestions"] += len(predictions)
            entry["correct"] += hits
        # 6. Resultado do modelo: identificação (hash, runtime, origem) e as contagens por grupo.
        report["models"].append({"weights": str(weights), "weights_sha256": file_hash(weights),
                                 "runtime": model.dronecamp_runtime, "source_checkpoint": str(model.dronecamp_source),
                                 "imgsz": model_config.prediction["imgsz"], "splits": dict(sorted(totals.items()))})
        # Resumo no terminal: uma linha por grupo.
        print(f"\n{weights}")
        for group, entry in sorted(totals.items()):
            print(f"  {group:28s} fotos {entry['images']:2d} | humanas reencontradas {entry['found']:3d}/{entry['truth']:3d}"
                  f" | sugestões corretas {entry['correct']:3d}/{entry['suggestions']:3d}")
    # 7. Grava o JSON (sobrescreve se já existir).
    write_json(args.output, report)
    return 0


# ENTRADA: execução direta pelo terminal; o código de saída é o retorno de main().
if __name__ == "__main__":
    raise SystemExit(main())
