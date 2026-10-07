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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dronecamp_ia.io import file_hash, write_json  # noqa: E402


def weights_directory(run: Path) -> Path:
    """Onde o Ultralytics grava os pesos agora: área de gravação ou run/fit/weights."""
    try:
        execution = json.loads((run / "execution.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        execution = {}
    staging = execution.get("fit_staging")
    return Path(staging) / "weights" if staging else run / "fit" / "weights"


def epochs_done(run: Path) -> int:
    """Épocas concluídas segundo o results.csv (espelhado em run/fit a cada época)."""
    results = run / "fit" / "results.csv"
    try:
        with results.open(encoding="utf-8", newline="") as stream:
            return sum(1 for row in csv.DictReader(stream) if row)
    except OSError:
        return 0


def training_state(run: Path) -> str:
    try:
        return json.loads((run / "summary.json").read_text(encoding="utf-8")).get("state", "running")
    except (OSError, ValueError):
        return "running"


def snapshot_and_export(config, source: Path, target: Path) -> Path:
    """Copia o checkpoint e exporta ONNX com export.json (mesmo contrato do comando export)."""
    import torch

    from dronecamp_ia.backend import load_detector

    target.mkdir(parents=True, exist_ok=True)
    checkpoint = target / "model.pt"
    shutil.copy2(source, checkpoint)
    torch.set_num_threads(2)  # o treino continua com o resto da CPU
    model = load_detector(config, str(checkpoint))
    output = Path(model.export(format="onnx", device="cpu", imgsz=640, batch=1, dynamic=False, simplify=False,
                               verbose=False))
    onnx = target / "model.onnx"
    if output.resolve() != onnx.resolve():
        shutil.move(str(output), onnx)
    write_json(target / "export.json", {
        "source_weights": checkpoint.resolve().relative_to(config.root.resolve()).as_posix(),
        "runtime": {"checkpoint_sha256": file_hash(checkpoint)}, "sha256": file_hash(onnx), "format": "onnx",
        "imgsz": 640, "purpose": "avaliacao_gpu_durante_treino", "copied_from": str(source)})
    return onnx


def evaluate_snapshot(config, onnx: Path, data: Path, provider: str) -> dict:
    from evaluate_models import evaluate, pilot_sets, trained_on

    from dronecamp_ia.backend import load_inference_model
    from dronecamp_ia.config import detection_names, load_taxonomy
    from dronecamp_ia.training import pilot_inference_config

    names = detection_names(load_taxonomy(config.taxonomy_path))
    model = load_inference_model(config, str(onnx), provider)
    model_config = pilot_inference_config(config, model)
    seen = trained_on(model)
    conf = float(config.pilot.get("suggestion_conf", 0.15))
    return {set_name: evaluate(model, model_config, photos, names, conf, 0.01, seen)
            for set_name, photos in pilot_sets(data).items()}


def plot_curve(rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = sorted({name for row in rows for name in row["sets"]})
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=130)
    for index, set_name in enumerate(sets):
        epochs = [row["epoch"] for row in rows if set_name in row["sets"]]
        for metric, style in (("map50", "-"), ("recall", "--")):
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="dataset.yaml piloto do treino (val/test).")
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--poll", type=float, default=30.0)
    parser.add_argument("--onnx-provider", choices=["directml", "cpu"], default="directml")
    parser.add_argument("--keep", type=int, default=2, help="Quantas cópias de pesos de época manter.")
    args = parser.parse_args()
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from dronecamp_ia.config import load_config

    config = load_config()
    run = (args.run if args.run.is_absolute() else ROOT / args.run).resolve()
    out = run / "gpu_eval"
    out.mkdir(exist_ok=True)
    curve = out / "curve.jsonl"
    rows = [json.loads(line) for line in curve.read_text(encoding="utf-8").splitlines()] if curve.is_file() else []
    done = {row["epoch"] for row in rows}
    next_epoch = max([args.every] + [e + args.every for e in done if isinstance(e, int)])
    print(f"Vigia GPU em {run.name}: a cada {args.every} épocas ({args.onnx_provider}).", flush=True)
    while True:
        state, epoch = training_state(run), epochs_done(run)
        final = state != "running"
        source = weights_directory(run) / "last.pt"
        if final:
            source = run / "fit" / "weights" / "best.pt"
            label = "final_best"
            if not source.is_file():
                print(f"Treino terminou como {state!r} sem best.pt; nada a avaliar.", flush=True)
                return 1
        else:
            label = epoch
        if (final or epoch >= next_epoch) and source.is_file() and label not in done:
            started = time.perf_counter()
            # Espera o arquivo parar de mudar (o treino grava last.pt no fim da época).
            size = -1
            while size != source.stat().st_size:
                size = source.stat().st_size
                time.sleep(3)
            target = out / (f"epoch_{epoch:03d}" if not final else "final_best")
            onnx = snapshot_and_export(config, source, target)
            sets = evaluate_snapshot(config, onnx, args.data.resolve(), args.onnx_provider)
            row = {"epoch": label, "epochs_done": epoch, "state": state, "seconds": round(time.perf_counter() - started, 1),
                   "sets": sets, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            rows.append(row)
            done.add(label)
            with curve.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            plot_curve([r for r in rows if isinstance(r["epoch"], int)] or rows[-1:], out / "curva_gpu.png")
            summary = " | ".join(f"{name}: R {value['totals']['recall']} mAP50 {value['totals']['map50']}"
                                 for name, value in sets.items())
            print(f"[{label}] {summary} ({row['seconds']} s)", flush=True)
            next_epoch = epoch + args.every
            # Mantém só as últimas cópias de pesos de época (o JSONL guarda as métricas).
            epochs_dirs = sorted(p for p in out.glob("epoch_*") if p.is_dir())
            for old in epochs_dirs[:-args.keep]:
                shutil.rmtree(old, ignore_errors=True)
        if final and "final_best" in done:
            print("Treino encerrado; avaliação final gravada.", flush=True)
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
