"""Avaliação na GPU em paralelo ao treino na CPU (``train-pilot --gpu-eval``).

Função no projeto: no PC do projeto a Ultralytics treina só na CPU (Ryzen) e a
GPU (APU Radeon) fica parada; ela só executa ONNX via DirectML. Este módulo usa
as duas ao mesmo tempo: a cada N épocas o treino copia ``last.pt`` para uma
fila; um processo separado (worker) exporta o checkpoint para ONNX, avalia o
split pedido na GPU com ``model_gate.evaluate_model`` (fotos inteiras,
inferência fatiada) e grava uma linha em ``gpu_eval/eval_curve.jsonl``. O
resultado é a curva "mAP50 por época nas fotos inteiras", que o ``results.csv``
(validação da Ultralytics em janelas) não mostra, mais a medida do que foi
feito em paralelo (segundos de avaliação na GPU durante o treino).

O que faz:
- ``gpu_eval_settings``: lê e valida ``pilot.gpu_eval`` do ``configs/project.yaml``.
- ``work_directory``: onde ficam fila e exportações (área de gravação, quando há).
- ``enqueue_checkpoint``: cópia atômica de ``last.pt`` para ``queue/epoch_NNNN.pt``.
- ``start_worker`` / ``finish_worker``: inicia o subprocesso e o encerra no fim do treino.
- ``run_worker``: laço do worker (comando ``gpu-eval-worker``).
- ``evaluate_checkpoint``: exporta um checkpoint e o avalia num split do dataset piloto.
- ``read_curve``: linhas válidas da curva (para o resumo do treino e o relatório).

A curva NÃO escolhe pesos nem aprova nada: ``best.pt`` continua sendo o da
Ultralytics e a adoção continua passando pelo ``compare-models``.

Quando mexer: frequência, provedor, split ou tempo de espera ficam em
``pilot.gpu_eval``; mexa aqui para mudar o protocolo da fila (arquivos
``epoch_NNNN.pt`` + marcador ``DONE``) ou o que cada linha registra.
"""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from .config import ProjectConfig
from .io import file_hash, write_json

DEFAULTS = {"enabled": False, "every": 5, "provider": "directml", "split": "val", "timeout": 1800}
PROVIDERS = ("directml", "cpu")
SPLITS = ("val", "test")
FOLDER = "gpu_eval"
CURVE = "eval_curve.jsonl"
DONE = "DONE"
# Threads de CPU do worker (exportação ONNX e pré/pós-processamento): o treino fica com o resto.
WORKER_THREADS = 2
# Erros seguidos que encerram o worker (ex.: DirectML indisponível).
MAX_CONSECUTIVE_ERRORS = 3
# Espaço livre mínimo para copiar um checkpoint (~200 MB) para a fila.
MIN_FREE_BYTES = 2 * 1024 ** 3
GITIGNORE = "# Checkpoints e ONNX intermediários (centenas de MB) ficam fora do git.\nqueue/\nexport_*/\n*.pt\n*.onnx\n"


# ---------------------------------------------------------------------------
# Configuração e caminhos.
# ---------------------------------------------------------------------------

def gpu_eval_settings(config: ProjectConfig | dict | None, overrides: dict | None = None) -> dict | None:
    """``pilot.gpu_eval`` + opções da linha de comando; ``None`` quando desligado."""
    raw = config if isinstance(config, dict) or config is None else (config.pilot or {}).get("gpu_eval")
    settings = {**DEFAULTS, **(raw or {}), **{key: value for key, value in (overrides or {}).items() if value is not None}}
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Chaves desconhecidas em pilot.gpu_eval: {', '.join(sorted(unknown))}.")
    if not isinstance(settings["enabled"], bool):
        raise ValueError("pilot.gpu_eval.enabled deve ser booleano.")
    if type(settings["every"]) is not int or settings["every"] < 1:
        raise ValueError("pilot.gpu_eval.every deve ser um inteiro positivo (épocas entre avaliações).")
    if settings["provider"] not in PROVIDERS:
        raise ValueError(f"pilot.gpu_eval.provider deve ser um de {PROVIDERS}.")
    if settings["split"] not in SPLITS:
        raise ValueError(f"pilot.gpu_eval.split deve ser um de {SPLITS}.")
    if isinstance(settings["timeout"], bool) or not isinstance(settings["timeout"], (int, float)) or settings["timeout"] < 0:
        raise ValueError("pilot.gpu_eval.timeout deve ser um número de segundos não negativo.")
    return settings if settings["enabled"] else None


def folder(run: Path) -> Path:
    """Arquivos pequenos (curva, log, DONE, worker.json) ficam na execução."""
    return Path(run) / FOLDER


def work_directory(run: Path) -> Path:
    """Fila e exportações (centenas de MB): na área de gravação DRONECAMP_FIT_STAGING, se houver.

    No PC do projeto a execução fica num HD USB onde gravações grandes já
    falharam (OSError 22); ``training._fit_staging`` manda o ``fit/`` para o
    disco interno e a fila da GPU vai para o mesmo lugar.
    """
    base = os.environ.get("DRONECAMP_FIT_STAGING")
    if base:
        return Path(base).expanduser().resolve() / Path(run).name / FOLDER
    return folder(run)


def curve_path(run: Path) -> Path:
    return folder(run) / CURVE


def _epoch_of(checkpoint: Path) -> int:
    return int(checkpoint.stem.split("_")[-1])


def _retry(action, attempts: int = 3, pause: float = 2.0):
    """Repete operações de arquivo que o Windows bloqueia por instantes (antivírus, indexação)."""
    for attempt in range(attempts):
        try:
            return action()
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(pause)


def directml_available() -> bool:
    try:
        import onnxruntime
    except ImportError:
        return False
    return "DmlExecutionProvider" in onnxruntime.get_available_providers()


# ---------------------------------------------------------------------------
# Lado do treino: fila de checkpoints e ciclo de vida do worker.
# ---------------------------------------------------------------------------

def enqueue_checkpoint(work: Path, checkpoint: Path, epoch: int) -> Path:
    """Copia ``last.pt`` para a fila sem que o worker veja um arquivo pela metade.

    A cópia vai para ``.tmp`` (nome que o worker não lista) e só depois é
    renomeada. Recusa quando o disco está quase cheio: o próprio ``last.pt``
    precisa desse espaço a cada época.
    """
    queue = Path(work) / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(queue).free < MIN_FREE_BYTES:
        raise OSError(f"menos de {MIN_FREE_BYTES // 1024 ** 3} GB livres em {queue}")
    target = queue / f"epoch_{epoch:04d}.pt"
    temporary = target.with_suffix(".tmp")
    _retry(lambda: shutil.copyfile(checkpoint, temporary))
    _retry(lambda: os.replace(temporary, target))
    return target


def pending_checkpoints(work: Path) -> list[Path]:
    queue = Path(work) / "queue"
    return sorted(queue.glob("epoch_*.pt"), key=_epoch_of) if queue.is_dir() else []


def start_worker(config: ProjectConfig, run: Path, data_path: Path, settings: dict, imgsz: int,
                 config_path: Path | None = None) -> subprocess.Popen | None:
    """Inicia ``python -m dronecamp_ia gpu-eval-worker``; ``None`` quando o DirectML não existe.

    O worker recebe um pipe em stdin: quando o treino morre (inclusive por
    ``TerminateProcess`` no Windows), o pipe fecha e o worker se encerra
    sozinho — nenhum processo órfão segurando a GPU.
    """
    run, work = Path(run).resolve(), work_directory(run)
    for directory in {folder(run), work}:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    if settings["provider"] == "directml" and not directml_available():
        write_json(folder(run) / "worker.json", {"state": "skipped", "reason": "directml_unavailable",
                                                 "settings": settings})
        return None
    log = (folder(run) / "worker.log").open("ab")
    environment = {**os.environ, "OMP_NUM_THREADS": str(WORKER_THREADS), "MKL_NUM_THREADS": str(WORKER_THREADS),
                   "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    config_path = config_path or os.environ.get("DRONECAMP_CONFIG") or (config.root / "configs" / "project.yaml")
    command = [sys.executable, "-m", "dronecamp_ia", "--config", str(config_path),
               "gpu-eval-worker", "--run", str(run), "--work", str(work), "--data", str(Path(data_path).resolve()),
               "--provider", settings["provider"], "--split", settings["split"], "--imgsz", str(imgsz), "--watch-stdin"]
    extra = {}
    if os.name == "nt":
        extra["creationflags"] = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    process = subprocess.Popen(command, cwd=str(config.root), env=environment, stdin=subprocess.PIPE,
                               stdout=log, stderr=subprocess.STDOUT, **extra)
    process.dronecamp_log = log  # fechado em finish_worker
    write_json(folder(run) / "worker.json", {"state": "running", "pid": process.pid, "command": command,
                                             "settings": settings, "work": str(work), "imgsz": imgsz,
                                             "started_at": datetime.now(timezone.utc).isoformat()})
    return process


def _training_seconds(run: Path) -> float | None:
    """Duração do treino pelo ``time`` acumulado do results.csv (fit/ ou área de gravação)."""
    for candidate in (Path(run) / "fit" / "results.csv",
                      *( [Path(os.environ["DRONECAMP_FIT_STAGING"]).expanduser() / Path(run).name / "fit" / "results.csv"]
                         if os.environ.get("DRONECAMP_FIT_STAGING") else [])):
        try:
            with candidate.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            values = [float({key.strip(): value for key, value in row.items() if key}["time"]) for row in rows]
            return max(values) if values else None
        except (OSError, KeyError, ValueError):
            continue
    return None


def finish_worker(run: Path, process: subprocess.Popen | None, timeout: float) -> dict:
    """Marca DONE, espera o worker esvaziar a fila e resume a curva; nunca derruba o treino."""
    run = Path(run).resolve()
    summary: dict = {"folder": str(folder(run)), "curve": str(curve_path(run)), "entries": 0}
    try:
        folder(run).mkdir(parents=True, exist_ok=True)
        (folder(run) / DONE).write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
        if process is not None:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                summary["error"] = f"worker encerrado após {timeout:.0f} s sem esvaziar a fila."
            summary["worker_returncode"] = process.returncode
            for handle in (getattr(process, "dronecamp_log", None), process.stdin):
                if handle is not None:
                    handle.close()
            if process.returncode not in (None, 0) and "error" not in summary:
                summary["error"] = f"worker terminou com código {process.returncode}; veja gpu_eval/worker.log."
    except OSError as error:
        summary["error"] = f"{type(error).__name__}: {error}"
    entries = read_curve(run)
    summary["entries"] = len(entries)
    summary["pending"] = [path.name for path in pending_checkpoints(work_directory(run))]
    scored = [entry for entry in entries if entry.get("map50") is not None]
    if scored:
        best = max(scored, key=lambda entry: entry["map50"])
        summary["best_epoch_by_map50"] = {"epoch": best["epoch"], "map50": best["map50"], "split": best.get("split")}
    errors = [entry for entry in read_curve(run, include_errors=True) if "error" in entry]
    if errors:
        summary["evaluation_errors"] = len(errors)
    # O que foi feito em paralelo: segundos de GPU/exportação executados durante o treino.
    eval_seconds = round(sum(float(entry.get("seconds") or 0) for entry in entries), 1)
    training_seconds = _training_seconds(run)
    summary["parallel"] = {
        "evaluations": len(entries), "eval_total_seconds": eval_seconds,
        "export_seconds": round(sum(float(entry.get("export_seconds") or 0) for entry in entries), 1),
        "inference_seconds": round(sum(float(entry.get("inference_seconds") or 0) for entry in entries), 1),
        "training_seconds": training_seconds,
        "gpu_busy_fraction": round(eval_seconds / training_seconds, 3) if training_seconds else None,
        "providers": sorted({str(entry.get("provider_active") or entry.get("provider")) for entry in entries}),
        "note": "Sem o worker, eval_total_seconds seriam somados ao fim do treino; a curva mede fotos inteiras.",
    }
    return summary


def read_curve(run: Path, include_errors: bool = False) -> list[dict]:
    """Linhas válidas de ``eval_curve.jsonl`` em ordem de época (linhas corrompidas são ignoradas)."""
    path = curve_path(run)
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and isinstance(entry.get("epoch"), int) and (include_errors or "error" not in entry):
            entries.append(entry)
    return sorted(entries, key=lambda entry: entry["epoch"])


# ---------------------------------------------------------------------------
# Lado do worker: exportar, avaliar e registrar.
# ---------------------------------------------------------------------------

def limit_worker_threads(threads: int = WORKER_THREADS) -> None:
    """A Ultralytics devolve o torch a 8 threads em todo predict/export; fixa o limite do worker."""
    import torch

    torch.set_num_threads(threads)
    try:
        import ultralytics.utils as utils
        import ultralytics.utils.torch_utils as torch_utils

        utils.NUM_THREADS = threads
        torch_utils.NUM_THREADS = threads
    except ImportError:
        pass


def evaluate_checkpoint(config: ProjectConfig, run: Path, work: Path, data_path: Path, checkpoint: Path,
                        provider: str, split: str, imgsz: int) -> dict:
    """Exporta o checkpoint para ONNX em ``<work>/export_NNNN`` e avalia o split nas fotos inteiras."""
    from dataclasses import replace

    from .backend import check_domain_names, load_inference_model
    from .config import detection_names, load_taxonomy
    from .exporting import export_onnx
    from .model_gate import evaluate_model
    from .training import pilot_inference_config

    started, started_at = time.monotonic(), datetime.now(timezone.utc).isoformat()
    epoch = _epoch_of(checkpoint)
    export_dir = Path(work) / f"export_{epoch:04d}"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    # imgsz explícito: fora da pasta da execução o checkpoint não é reconhecido como piloto.
    export_config = replace(config, prediction={**config.prediction, "imgsz": int(imgsz)})
    # Sem paridade (parity_images=[]): a conferência .pt × .onnx custa tanto quanto a avaliação.
    export_onnx(export_config, str(checkpoint), demo=False, parity_images=[], output=export_dir)
    export_seconds = round(time.monotonic() - started, 1)
    onnx_path = export_dir / "model.onnx"
    onnx_sha256 = file_hash(onnx_path)
    names = detection_names(load_taxonomy(config.taxonomy_path))
    model = load_inference_model(config, str(onnx_path), provider)
    check_domain_names(model, names)
    model_config = pilot_inference_config(config, model)
    root = Path(data_path).resolve().parent
    manifest = json.loads((root / "pilot.json").read_text(encoding="utf-8"))
    samples = [sample for sample in manifest["images"] if sample["split"] == split]
    if not samples:
        raise ValueError(f"Nenhuma foto no split {split} do dataset piloto.")
    conf = float(config.pilot.get("suggestion_conf", config.prediction["conf"]))
    inference_started = time.monotonic()
    report = evaluate_model(model, model_config, root, samples, names, conf)
    inference_seconds = round(time.monotonic() - inference_started, 1)
    total = report.pop("_total")
    strata = {key: value["_total"] for key, value in (report.pop("_strata", None) or {}).items()}
    regime = report.pop("_regime", None)
    provider_active = None
    try:  # provedor que a sessão ONNX de fato ativou (prova de GPU)
        provider_active = model.predictor.model.session.get_providers()[0]
    except AttributeError:
        pass
    # O .pt e o ONNX intermediários não ficam: a curva guarda os hashes; best.pt/last.pt ficam no fit.
    shutil.rmtree(export_dir, ignore_errors=True)
    return {"epoch": epoch, "split": split, "provider": provider, "provider_active": provider_active,
            "runtime": model.dronecamp_runtime, "images": len(samples), "conf": conf,
            "map50": total.get("map50"), "precision": total.get("precision"), "recall": total.get("recall"),
            "tp": total["tp"], "fp": total["fp"], "fn": total["fn"], "classes_measured": total.get("classes_measured"),
            "strata": strata, "regime": regime, "per_class": report,
            "checkpoint_sha256": file_hash(checkpoint), "onnx_sha256": onnx_sha256, "imgsz": model_config.prediction["imgsz"],
            "export_seconds": export_seconds, "inference_seconds": inference_seconds,
            "seconds": round(time.monotonic() - started, 1), "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat()}


def _append(run: Path, entry: dict) -> None:
    path = curve_path(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")


def _watch_stdin() -> None:
    """Encerra o worker quando o processo do treino some (o pipe de stdin fecha)."""
    def wait_for_parent():
        try:
            sys.stdin.buffer.read()
        except (OSError, ValueError):
            pass
        print("gpu-eval-worker: treino encerrado; saindo.", flush=True)
        os._exit(3)

    threading.Thread(target=wait_for_parent, daemon=True).start()


def run_worker(config: ProjectConfig, run: Path, data_path: Path, provider: str, split: str, imgsz: int,
               work: Path | None = None, poll_seconds: float = 5.0, evaluator=evaluate_checkpoint,
               max_idle_seconds: float | None = None, watch_stdin: bool = False) -> dict:
    """Laço do worker: avalia cada checkpoint da fila até DONE com a fila vazia.

    Sempre pega o checkpoint mais recente; os mais antigos ainda na fila são
    descartados como "superseded" (a curva registra quais), para a GPU não
    ficar atrasada em relação ao treino. ``evaluator`` é trocável nos testes;
    ``max_idle_seconds`` encerra um worker sem checkpoint novo.
    """
    run = Path(run).resolve()
    work = Path(work).resolve() if work else work_directory(run)
    (work / "queue").mkdir(parents=True, exist_ok=True)
    if watch_stdin:
        _watch_stdin()
    if evaluator is evaluate_checkpoint:
        limit_worker_threads()
    processed, consecutive_errors, last_activity = 0, 0, time.monotonic()
    print(f"gpu-eval-worker: fila {work / 'queue'} (provider={provider}, split={split}, imgsz={imgsz})", flush=True)
    while True:
        pending = pending_checkpoints(work)
        if pending:
            checkpoint = pending[-1]
            for stale in pending[:-1]:
                _append(run, {"epoch": _epoch_of(stale), "split": split, "superseded_by": _epoch_of(checkpoint),
                              "error": "superseded: o worker estava atrasado e pulou esta época"})
                _retry(lambda path=stale: path.unlink(missing_ok=True))
            try:
                entry = evaluator(config, run, work, data_path, checkpoint, provider, split, imgsz)
                consecutive_errors = 0
                print(f"gpu-eval-worker: época {entry['epoch']} mAP50={entry.get('map50')} em {entry.get('seconds')} s "
                      f"({entry.get('provider_active') or provider})", flush=True)
            except Exception as error:  # a curva registra o erro; o treino segue
                consecutive_errors += 1
                entry = {"epoch": _epoch_of(checkpoint), "split": split, "provider": provider,
                         "error": f"{type(error).__name__}: {error}", "finished_at": datetime.now(timezone.utc).isoformat()}
                print(f"gpu-eval-worker: falha na época {entry['epoch']}: {entry['error']}", flush=True)
            _append(run, entry)
            _retry(lambda: checkpoint.unlink(missing_ok=True))
            processed += 1
            last_activity = time.monotonic()
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print("gpu-eval-worker: erros seguidos demais; encerrando.", flush=True)
                return {"processed": processed, "status": "failed"}
            continue
        if (folder(run) / DONE).is_file():
            if pending_checkpoints(work):  # enfileirado entre a varredura e o DONE
                continue
            return {"processed": processed, "status": "done"}
        if max_idle_seconds is not None and time.monotonic() - last_activity > max_idle_seconds:
            print("gpu-eval-worker: sem checkpoints novos; encerrando.", flush=True)
            return {"processed": processed, "status": "idle_timeout"}
        time.sleep(poll_seconds)
