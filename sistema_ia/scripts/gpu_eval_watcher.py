"""Avaliação na GPU em paralelo ao treino na CPU (CPU + GPU juntas).

O treino (``train-pilot``) ocupa a CPU; a Radeon integrada não treina o YOLO
(``torch-directml`` não implementa operações da perda; ver README). Este vigia
roda ao lado do treino e, a cada ``--every`` épocas:

1. copia o ``last.pt`` do run (área de gravação, se houver) para
   ``runs/<run>/gpu_eval/epoch_NNN/model.pt``;
2. exporta ONNX (CPU, poucos segundos, com 2 threads para não frear o treino) e
   grava um ``export.json`` com os hashes, como o comando ``export``;
3. avalia na GPU (ONNX Runtime DirectML) as fotos de validação e teste do
   dataset piloto inteiras, com o mesmo fatiamento das sugestões — a métrica que
   a validação da Ultralytics (feita nas janelas) não mostra;
4. acrescenta a linha em ``gpu_eval/curve.jsonl`` e redesenha ``curva_gpu.png``.

Quando o treino termina, avalia também o ``best.pt`` final e sai. Só lê o run:
nunca altera pesos, summary ou métricas do treino.

Uso (num segundo terminal, depois de iniciar o treino):
  python scripts/gpu_eval_watcher.py --run runs/pilot_train_X --data data/pilot/V/dataset.yaml --every 5

Função no projeto:
    Versão manual (script avulso) da avaliação paralela na GPU. A versão integrada ao
    treino é ``src/dronecamp_ia/gpu_eval.py`` (``train-pilot --gpu-eval`` e a caixa
    "Avaliar na GPU durante o treino" da plataforma), que grava ``gpu_eval/eval_curve.jsonl``.
    Este vigia grava ``gpu_eval/curve.jsonl``; ``operations.read_gpu_curve`` lê os dois
    formatos para a plataforma (seção Treinamento/Monitoramento).

O que contém:
    - ``weights_directory``: pasta onde o treino grava os pesos agora.
    - ``epochs_done``: épocas concluídas (linhas do results.csv).
    - ``training_state``: estado do treino no summary.json ("running", "completed"...).
    - ``snapshot_and_export``: copia o checkpoint e exporta ONNX com export.json.
    - ``evaluate_snapshot``: avalia o ONNX nas fotos val/test (usa scripts/evaluate_models.py).
    - ``plot_curve``: desenha ``curva_gpu.png``.
    - ``main``: laço de vigia.

Quem chama:
    Rodado à mão num segundo terminal (ver ``Uso``). ``tests/test_gpu_eval_watcher.py``
    testa ``weights_directory``, ``epochs_done`` e ``training_state``.

O que lê e grava no disco:
    Lê (sem alterar): ``runs/<run>/execution.json``, ``summary.json``, ``fit/results.csv``,
    ``last.pt``/``best.pt``; o dataset piloto (pilot.json, fotos, labels); configs/.
    Grava só dentro de ``runs/<run>/gpu_eval/``: ``epoch_NNN/`` ou ``final_best/`` (model.pt,
    model.onnx, export.json), ``curve.jsonl`` (uma linha JSON por avaliação, só acrescenta)
    e ``curva_gpu.png``. Apaga pastas ``epoch_*`` antigas (mantém as ``--keep`` mais novas).

Onde mexer:
    - Para registrar mais um dado por avaliação: acrescente a chave em ``row`` dentro de ``main``
      (e, se a plataforma deve mostrar, leia em ``operations.read_gpu_curve``).
    - Para avaliar outros conjuntos (ex.: o treino): mude ``evaluate_models.pilot_sets``.
    - Para desenhar outra métrica na curva: acrescente o par na tupla de ``plot_curve``.
    - Para mudar o tamanho de entrada do ONNX: ``imgsz=640`` em ``snapshot_and_export``
      (nos dois lugares: no ``export`` e no ``export.json``).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time

# Raiz do sistema_ia (pasta acima de scripts/).
ROOT = Path(__file__).resolve().parents[1]
# Permite importar o pacote sem instalá-lo (src/) e o evaluate_models.py vizinho (scripts/).
# ATENCAO: precisa vir antes dos imports de dronecamp_ia abaixo (por isso o "noqa: E402").
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dronecamp_ia.io import file_hash, write_json  # noqa: E402


# ---------------------------------------------------------------------------
# Leitura do estado do treino (só lê arquivos do run)
# ---------------------------------------------------------------------------

def weights_directory(run: Path) -> Path:
    """Onde o Ultralytics grava os pesos agora: área de gravação ou run/fit/weights.

    Quando o treino usa uma área de gravação temporária (``fit_staging`` no
    execution.json, ver ``training._fit_staging``), o ``last.pt`` fica lá durante o
    treino e só é copiado para ``run/fit/weights`` no fim.

    Parâmetros:
        run (Path): pasta da execução (``runs/pilot_train_<data>_<id>``).

    Retorna:
        Path: ``<fit_staging>/weights`` se declarado; senão ``run/fit/weights``.
        execution.json ausente ou inválido conta como "sem área de gravação".

    Chamada por: ``main`` (a cada volta do laço).
    """
    try:
        execution = json.loads((run / "execution.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        execution = {}
    staging = execution.get("fit_staging")
    return Path(staging) / "weights" if staging else run / "fit" / "weights"


def epochs_done(run: Path) -> int:
    """Épocas concluídas segundo o results.csv (espelhado em run/fit a cada época).

    Parâmetros:
        run (Path): pasta da execução.

    Retorna:
        int: número de linhas de dados do ``run/fit/results.csv`` (uma por época
        concluída); 0 se o arquivo ainda não existe.

    Chamada por: ``main``.
    """
    results = run / "fit" / "results.csv"
    try:
        # DictReader pula o cabeçalho; "if row" ignora linhas vazias.
        with results.open(encoding="utf-8", newline="") as stream:
            return sum(1 for row in csv.DictReader(stream) if row)
    except OSError:
        return 0


def training_state(run: Path) -> str:
    """Estado do treino gravado em ``run/summary.json``.

    Parâmetros:
        run (Path): pasta da execução.

    Retorna:
        str: o campo ``state`` (ex.: "running", "completed", "failed"); "running" quando
        o summary não existe, está ilegível ou não tem o campo (assume que continua).

    Chamada por: ``main``.
    """
    try:
        return json.loads((run / "summary.json").read_text(encoding="utf-8")).get("state", "running")
    except (OSError, ValueError):
        return "running"


# ---------------------------------------------------------------------------
# Cópia, exportação e avaliação de um checkpoint
# ---------------------------------------------------------------------------

def snapshot_and_export(config, source: Path, target: Path) -> Path:
    """Copia o checkpoint e exporta ONNX com export.json (mesmo contrato do comando export).

    O ONNX só é aceito por ``load_inference_model`` com um ``export.json`` ao lado que
    confira os hashes; por isso ele é gravado aqui no mesmo formato do comando ``export``.

    Parâmetros:
        config: ``ProjectConfig`` (usa ``config.root`` para o caminho relativo do .pt).
        source (Path): ``last.pt`` (ou ``best.pt``) do treino em andamento.
        target (Path): pasta de destino, ex.: ``runs/<run>/gpu_eval/epoch_010``.

    Retorna:
        Path: caminho do ``model.onnx`` gerado em ``target``.

    Erros:
        ValueError em ``relative_to`` se a pasta do run estiver fora de ``config.root``;
        erros da exportação da Ultralytics/ONNX são repassados.

    Chamada por: ``main``.
    """
    import torch

    from dronecamp_ia.backend import load_detector

    target.mkdir(parents=True, exist_ok=True)
    # 1. Cópia própria: o treino continua sobrescrevendo last.pt enquanto exportamos.
    checkpoint = target / "model.pt"
    shutil.copy2(source, checkpoint)
    torch.set_num_threads(2)  # o treino continua com o resto da CPU
    # 2. Exporta para ONNX na CPU, entrada fixa 640 x 640, lote 1 (o DirectML exige forma fixa).
    model = load_detector(config, str(checkpoint))
    output = Path(model.export(format="onnx", device="cpu", imgsz=640, batch=1, dynamic=False, simplify=False,
                               verbose=False))
    # A Ultralytics grava o .onnx ao lado do .pt com outro nome; move para model.onnx.
    onnx = target / "model.onnx"
    if output.resolve() != onnx.resolve():
        shutil.move(str(output), onnx)
    # 3. export.json: hashes do .pt e do .onnx, que identificam o piloto e as fotos de treino.
    # ATENCAO: as chaves seguem o contrato do comando export (exporting.py); não renomeie.
    write_json(target / "export.json", {
        "source_weights": checkpoint.resolve().relative_to(config.root.resolve()).as_posix(),
        "runtime": {"checkpoint_sha256": file_hash(checkpoint)}, "sha256": file_hash(onnx), "format": "onnx",
        "imgsz": 640, "purpose": "avaliacao_gpu_durante_treino", "copied_from": str(source)})
    return onnx


def evaluate_snapshot(config, onnx: Path, data: Path, provider: str) -> dict:
    """Avalia um ONNX nas fotos val/test do dataset piloto (métricas de evaluate_models.py).

    Parâmetros:
        config: ``ProjectConfig``.
        onnx (Path): ``model.onnx`` de ``snapshot_and_export``.
        data (Path): dataset.yaml do dataset piloto do treino.
        provider (str): ``"directml"`` (GPU) ou ``"cpu"``.

    Retorna:
        dict: ``{"<versão>:val": {"totals": {...}, "per_class": {...}}, "<versão>:test": ...}``,
        no formato de ``evaluate_models.evaluate``.

    Chamada por: ``main``.
    ATENCAO: o ``model.pt`` copiado fica em ``gpu_eval/epoch_NNN/``, e não em
    ``runs/<run>/fit/weights``; por isso ``trained_on`` procura o dataset_snapshot.json
    três pastas acima (``runs/<run>/``), que existe e lista as fotos de treino. Se mudar a
    profundidade das pastas de ``gpu_eval``, a exclusão das fotos de treino para de funcionar.
    """
    # Imports tardios: evaluate_models (scripts/) e o backend pesam e só são usados aqui.
    from evaluate_models import evaluate, pilot_sets, trained_on

    from dronecamp_ia.backend import load_inference_model
    from dronecamp_ia.config import detection_names, load_taxonomy
    from dronecamp_ia.training import pilot_inference_config

    names = detection_names(load_taxonomy(config.taxonomy_path))
    model = load_inference_model(config, str(onnx), provider)
    model_config = pilot_inference_config(config, model)
    # Fotos de treino deste modelo ficam fora da conta.
    seen = trained_on(model)
    conf = float(config.pilot.get("suggestion_conf", 0.15))
    # AP com predições a partir de 0,01 (mesmo padrão de --ap-conf do evaluate_models).
    return {set_name: evaluate(model, model_config, photos, names, conf, 0.01, seen)
            for set_name, photos in pilot_sets(data).items()}


# ---------------------------------------------------------------------------
# Gráfico da curva
# ---------------------------------------------------------------------------

def plot_curve(rows: list[dict], output: Path) -> None:
    """Desenha ``curva_gpu.png``: mAP50 (linha cheia) e recall (tracejada) por época.

    Parâmetros:
        rows (list[dict]): linhas do curve.jsonl (``{"epoch": int, "sets": {...}}``).
        output (Path): arquivo PNG a gravar (sobrescrito a cada avaliação).

    Retorna:
        None.

    Chamada por: ``main``.
    EXTENSAO: outra métrica na curva = acrescente ``("precision", ":")``, por exemplo,
    na tupla de métricas do laço (a chave precisa existir em ``totals``).
    """
    # Import tardio; "Agg" desenha em arquivo, sem janela.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Todos os conjuntos que aparecem em alguma linha (ex.: "v9:test", "v9:val").
    sets = sorted({name for row in rows for name in row["sets"]})
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=130)
    for index, set_name in enumerate(sets):
        epochs = [row["epoch"] for row in rows if set_name in row["sets"]]
        # Mesma cor (C0, C1...) por conjunto; estilo diferencia a métrica.
        for metric, style in (("map50", "-"), ("recall", "--")):
            # None (sem dado) vira 0.
            values = [row["sets"][set_name]["totals"][metric] or 0 for row in rows if set_name in row["sets"]]
            ax.plot(epochs, values, style, marker="o", markersize=3, color=f"C{index}", label=f"{set_name} {metric}")
    ax.set_xlabel("época")
    ax.set_ylim(0, 1)
    ax.set_title("Avaliação na GPU (fotos inteiras, fatiamento das sugestões)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Laço de vigia
# ---------------------------------------------------------------------------

def main() -> int:
    """ENTRADA: vigia o treino e avalia um checkpoint a cada ``--every`` épocas.

    Parâmetros (linha de comando):
        --run PATH          pasta da execução (relativa ao sistema_ia ou absoluta).
        --data PATH         dataset.yaml piloto do treino (fotos val/test).
        --every N           avaliar a cada N épocas (padrão 5).
        --poll SEGUNDOS     intervalo entre verificações (padrão 30 s).
        --onnx-provider     ``directml`` (GPU, padrão) ou ``cpu``.
        --keep N            quantas pastas ``epoch_*`` (com pesos) manter (padrão 2).

    Retorna:
        int: 0 quando o treino terminou e o ``best.pt`` final foi avaliado; 1 quando o
        treino terminou sem ``best.pt`` (nada a avaliar).

    Retomada: se ``curve.jsonl`` já existir (vigia reiniciado), as épocas já avaliadas
    são puladas e a próxima é calculada a partir da última.
    """
    # 1. Argumentos.
    # EXTENSAO: opção nova = parser.add_argument(...) aqui e o uso no laço.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="dataset.yaml piloto do treino (val/test).")
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--poll", type=float, default=30.0)
    parser.add_argument("--onnx-provider", choices=["directml", "cpu"], default="directml")
    parser.add_argument("--keep", type=int, default=2, help="Quantas cópias de pesos de época manter.")
    args = parser.parse_args()
    # Impede a Ultralytics de instalar pacotes sozinha durante a exportação.
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from dronecamp_ia.config import load_config

    config = load_config()
    # 2. Pasta do run (relativa ao sistema_ia) e pasta de saída gpu_eval/.
    run = (args.run if args.run.is_absolute() else ROOT / args.run).resolve()
    out = run / "gpu_eval"
    out.mkdir(exist_ok=True)
    # 3. Retomada: lê as avaliações já gravadas (uma linha JSON por avaliação).
    curve = out / "curve.jsonl"
    rows = [json.loads(line) for line in curve.read_text(encoding="utf-8").splitlines()] if curve.is_file() else []
    # done: rótulos já avaliados (números de época ou "final_best").
    done = {row["epoch"] for row in rows}
    # Próxima época a avaliar: --every, ou a última avaliada + --every (ignora "final_best").
    next_epoch = max([args.every] + [e + args.every for e in done if isinstance(e, int)])
    print(f"Vigia GPU em {run.name}: a cada {args.every} épocas ({args.onnx_provider}).", flush=True)
    # 4. Laço até o treino terminar.
    while True:
        state, epoch = training_state(run), epochs_done(run)
        final = state != "running"
        # Durante o treino: last.pt da área de gravação atual.
        source = weights_directory(run) / "last.pt"
        if final:
            # Treino encerrado: avalia o best.pt definitivo (sempre em run/fit/weights).
            source = run / "fit" / "weights" / "best.pt"
            label = "final_best"
            if not source.is_file():
                print(f"Treino terminou como {state!r} sem best.pt; nada a avaliar.", flush=True)
                return 1
        else:
            label = epoch
        # Avalia quando chegou a época alvo (ou no fim), o peso existe e o rótulo é novo.
        if (final or epoch >= next_epoch) and source.is_file() and label not in done:
            started = time.perf_counter()
            # Espera o arquivo parar de mudar (o treino grava last.pt no fim da época).
            size = -1
            while size != source.stat().st_size:
                size = source.stat().st_size
                time.sleep(3)
            # Pasta da avaliação: epoch_NNN (3 dígitos, ordena por nome) ou final_best.
            target = out / (f"epoch_{epoch:03d}" if not final else "final_best")
            onnx = snapshot_and_export(config, source, target)
            sets = evaluate_snapshot(config, onnx, args.data.resolve(), args.onnx_provider)
            # Linha gravada no curve.jsonl (lida por operations.read_gpu_curve).
            # EXTENSAO: dado novo por avaliação = nova chave aqui (e a leitura em operations.read_gpu_curve).
            row = {"epoch": label, "epochs_done": epoch, "state": state, "seconds": round(time.perf_counter() - started, 1),
                   "sets": sets, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            rows.append(row)
            done.add(label)
            # Só acrescenta (modo "a"): avaliações antigas nunca são reescritas.
            with curve.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            # Curva com as épocas numéricas; se só houver a final, desenha só ela.
            plot_curve([r for r in rows if isinstance(r["epoch"], int)] or rows[-1:], out / "curva_gpu.png")
            summary = " | ".join(f"{name}: R {value['totals']['recall']} mAP50 {value['totals']['map50']}"
                                 for name, value in sets.items())
            print(f"[{label}] {summary} ({row['seconds']} s)", flush=True)
            # Próximo alvo conta a partir da época realmente avaliada (pode ter passado do alvo).
            next_epoch = epoch + args.every
            # Mantém só as últimas cópias de pesos de época (o JSONL guarda as métricas).
            # ATENCAO: com --keep 0, epochs_dirs[:-0] é uma lista vazia e NADA é apagado.
            epochs_dirs = sorted(p for p in out.glob("epoch_*") if p.is_dir())
            for old in epochs_dirs[:-args.keep]:
                shutil.rmtree(old, ignore_errors=True)
        # Fim: treino encerrado e best.pt final já avaliado (agora ou numa execução anterior).
        if final and "final_best" in done:
            print("Treino encerrado; avaliação final gravada.", flush=True)
            return 0
        time.sleep(args.poll)


# ENTRADA: execução direta pelo terminal; o código de saída é o retorno de main().
if __name__ == "__main__":
    raise SystemExit(main())
