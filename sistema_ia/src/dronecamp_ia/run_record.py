"""Ficha imutável de cada treino: quem, onde, com que código, dados e pesos, e cada tentativa.

Função no projeto: permitir reproduzir e auditar um resultado. Cada execução de
``train-pilot`` ou ``resume-pilot`` abre uma TENTATIVA e grava arquivos que nunca
são reescritos (criação exclusiva, modo ``"x"``) na pasta ``<run>/ficha/``:

- ``attempt_NNN_start.json``: ID do experimento (nome da pasta do run) e da
  tentativa, horário, máquina, processo, versão do código e alterações locais,
  versões das bibliotecas, hashes dos dados (imagens, anotações, classes, divisão
  treino/val/teste), pesos iniciais ou checkpoint pai com hash, configuração
  completa, sementes, e as tentativas anteriores que terminaram sem registro
  (queda de energia, reinício do Windows, processo morto).
- ``attempt_NNN_trainer.json``: o que a Ultralytics realmente usou ao começar
  (argumentos efetivos, época inicial, melhor fitness herdado, lotes por época,
  imagens de treino e validação, otimizador).
- ``attempt_NNN_end.json``: como terminou (concluída, falhou, interrompida),
  motivo, época global alcançada, passos do otimizador, lotes e imagens consumidos
  nesta tentativa, tempo efetivo e picos de recursos.
- ``evaluation_<data>.json``: avaliação feita depois com um checkpoint deste run
  (``quality_report.py``): qual peso, hash, época e por que é o "melhor".
- ``index.jsonl``: uma linha por arquivo gravado com o SHA-256; ``verify_run_card``
  confere que nenhum foi alterado depois.

Quem chama: ``training.train_pilot`` e ``training.resume_pilot`` (via
``training_monitor.TrainingMonitor``), ``quality_report.py`` e os comandos
``run-card`` e ``training-progress``.

O que lê e grava no disco: lê o repositório git (``git rev-parse``, ``git status``,
``git diff``), os arquivos de auditoria do run (``dataset_snapshot.json`` etc.) e os
pesos (hash). Grava só dentro de ``<run>/ficha/``.

ATENCAO: "imutável" aqui é uma regra do código (criação exclusiva + hash no
índice), não uma proteção do sistema de arquivos; ``run-card`` acusa qualquer
arquivo alterado ou apagado à mão.

Onde mexer:
- Biblioteca nova na lista de versões: ``LIBRARIES``.
- Arquivo de auditoria novo do run com hash na ficha: ``DATA_FILES``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

from .io import file_hash

# Pasta da ficha dentro do run.
FOLDER = "ficha"
# Bibliotecas cuja versão instalada é registrada (as que mudam resultado ou desempenho).
LIBRARIES = ("dronecamp-ia", "ultralytics", "torch", "torchvision", "numpy", "opencv-python", "pillow", "pyyaml",
             "scipy", "rfdetr", "transformers", "onnxruntime", "onnxruntime-directml", "psutil")
# Arquivos de auditoria do run cujo hash entra na ficha (os que existirem).
DATA_FILES = ("execution.json", "dataset_snapshot.json", "dataset_resolved.yaml", "dataset_tiled_resolved.yaml",
              "tiling_manifest.json", "train_repeat_factor.txt", "sampling.json", "pilot_provenance.json",
              "dataset_validation.json", "training_provenance.json")


def _now() -> str:
    """Hora UTC em ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def _git(root: Path, *args: str) -> str | None:
    """Saída de um comando git na pasta ``root``; ``None`` sem git ou fora de um repositório."""
    try:
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.decode("utf-8", "replace") if result.returncode == 0 else None


def code_version(root: Path) -> dict:
    """Commit, ramo e alterações locais do código (hash do ``git diff`` para identificá-las).

    Retorna ``{"commit", "branch", "dirty", "changed_files", "diff_sha256"}``; sem git,
    ``{"commit": None, "package_version"}``.
    """
    commit = _git(root, "rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "package_version": _version("dronecamp-ia"),
                "note": "pasta sem git: só a versão do pacote identifica o código"}
    status = _git(root, "status", "--porcelain", "--untracked-files=normal", "--", ".") or ""
    diff = _git(root, "diff", "HEAD", "--", ".") or ""
    changed = [line[3:] for line in status.splitlines() if line.strip()]
    return {"commit": commit.strip(), "branch": (_git(root, "rev-parse", "--abbrev-ref", "HEAD") or "").strip(),
            "dirty": bool(changed), "changed_files": changed[:200],
            "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else None}


def _version(name: str) -> str | None:
    """Versão instalada de um pacote (``None`` se não instalado)."""
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def library_versions() -> dict:
    """Versões efetivamente instaladas de ``LIBRARIES`` + Python, e o build do torch quando importado."""
    versions = {name: _version(name) for name in LIBRARIES}
    versions["python"] = platform.python_version()
    torch = sys.modules.get("torch")
    if torch is not None:
        versions["torch_build"] = {"cuda": getattr(torch.version, "cuda", None), "threads": torch.get_num_threads()}
    return versions


def host_info() -> dict:
    """Máquina e processo: nome, sistema, CPU, RAM total e PID."""
    info = {"host": platform.node(), "system": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "pid": os.getpid(), "python_executable": sys.executable}
    try:
        import psutil

        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
        info["cpu_physical"] = psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001 - psutil é opcional aqui
        pass
    return info


def data_identity(run: Path) -> dict:
    """Hashes dos arquivos de auditoria do run e um hash por divisão (treino/val/teste).

    O hash da divisão é o SHA-256 da lista ordenada ``caminho:sha256`` daquela pasta no
    ``dataset_snapshot.json``: mudou uma foto, uma anotação ou a divisão, mudou o hash.
    """
    run = Path(run)
    files = {name: file_hash(run / name) for name in DATA_FILES if (run / name).is_file()}
    identity = {"files": files}
    snapshot = run / "dataset_snapshot.json"
    if snapshot.is_file():
        data = json.loads(snapshot.read_text(encoding="utf-8"))
        splits = {}
        for split in ("train", "val", "test"):
            entries = sorted(f"{item['path']}:{item['sha256']}" for item in data.get("files", [])
                             if f"/{split}/" in f"/{item['path']}")
            images = sum(1 for entry in entries if entry.startswith("images/"))
            splits[split] = {"files": len(entries), "images": images,
                             "sha256": hashlib.sha256("\n".join(entries).encode()).hexdigest()}
        identity.update({"taxonomy_sha256": data.get("taxonomy_sha256"), "groups_sha256": data.get("groups_sha256"),
                         "source_yaml_sha256": data.get("source_yaml_sha256"), "splits": splits})
    return identity


def _folder(run: Path) -> Path:
    """``<run>/ficha`` (criada se faltar)."""
    folder = Path(run) / FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write_once(run: Path, name: str, data: dict) -> str:
    """Grava ``<run>/ficha/<name>`` uma única vez e registra o hash no ``index.jsonl``.

    Erros: FileExistsError se o arquivo já existir (a ficha nunca é reescrita).
    Retorna o SHA-256 do arquivo gravado.
    """
    folder = _folder(run)
    body = json.dumps(data, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    with (folder / name).open("xb") as handle:
        handle.write(body)
    digest = hashlib.sha256(body).hexdigest()
    with (folder / "index.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"file": name, "sha256": digest, "written_at": _now()}) + "\n")
    return digest


def _attempts(run: Path) -> list[int]:
    """Números das tentativas já abertas neste run."""
    folder = Path(run) / FOLDER
    if not folder.is_dir():
        return []
    return sorted(int(path.name.split("_")[1]) for path in folder.glob("attempt_*_start.json"))


def unfinished_attempts(run: Path) -> list[dict]:
    """Tentativas com ``start`` e sem ``end``: terminaram sem registro (queda, reinício, processo morto).

    Para cada uma devolve o último sinal de vida conhecido (última amostra de recursos ou
    batimento), que é a melhor estimativa de quando o processo parou.
    """
    folder = Path(run) / FOLDER
    result = []
    last_sample = None
    resources = Path(run) / "instrumentation" / "resources.jsonl"
    if resources.is_file():
        lines = [line for line in resources.read_text(encoding="utf-8").splitlines() if line.strip()]
        for line in reversed(lines):
            try:
                last_sample = json.loads(line).get("t")
                break
            except json.JSONDecodeError:
                continue
    heartbeat = Path(run) / "fit" / "heartbeat"
    last_beat = heartbeat.read_text(encoding="utf-8").strip() if heartbeat.is_file() else None
    for number in _attempts(run):
        if not (folder / f"attempt_{number:03d}_end.json").exists():
            start = json.loads((folder / f"attempt_{number:03d}_start.json").read_text(encoding="utf-8"))
            result.append({"attempt": number, "attempt_id": start.get("attempt_id"), "started_at": start.get("started_at"),
                           "host": start.get("host", {}).get("host"), "last_resource_sample": last_sample,
                           "last_heartbeat": last_beat,
                           "probable_reason": "encerrada_sem_registro (queda de energia, reinício ou processo morto)"})
    return result


def start_attempt(run: Path, kind: str, parameters: dict | None = None, weights: str | Path | None = None,
                  parent_checkpoint: Path | None = None, extra: dict | None = None) -> dict:
    """Abre uma tentativa: grava ``attempt_NNN_start.json`` e devolve o descritor da tentativa.

    Parâmetros:
        run: pasta do treino.
        kind: ``"novo"`` ou ``"retomada"``.
        parameters: parâmetros efetivos entregues à Ultralytics (configuração completa).
        weights: peso inicial (treino novo).
        parent_checkpoint: ``last.pt`` de onde a retomada parte (hash registrado).
        extra: dados adicionais (ex.: ``pilot_details``).

    Retorna: ``{"run", "experiment_id", "attempt", "attempt_id", "started_at"}``.
    """
    run = Path(run)
    number = (max(_attempts(run)) + 1) if _attempts(run) else 1
    attempt = {"run": str(run), "experiment_id": run.name, "attempt": number, "attempt_id": uuid.uuid4().hex,
               "started_at": _now()}
    initial = None
    if weights is not None:
        path = Path(str(weights))
        initial = {"name": str(weights), "sha256": file_hash(path) if path.is_file() else None,
                   "note": None if path.is_file() else "nome de peso da Ultralytics (baixado ou em cache); hash na ficha do treinador"}
    parent = None
    if parent_checkpoint is not None:
        parent = {"path": str(parent_checkpoint), "sha256": file_hash(Path(parent_checkpoint)),
                  "bytes": Path(parent_checkpoint).stat().st_size}
    parameters = dict(parameters or {})
    write_once(run, f"attempt_{number:03d}_start.json", {
        **attempt, "kind": kind, "host": host_info(), "code": code_version(Path(__file__).resolve().parents[2]),
        "libraries": library_versions(), "data": data_identity(run), "initial_weights": initial,
        "parent_checkpoint": parent, "parameters": parameters,
        "seeds": {"seed": parameters.get("seed"), "deterministic": parameters.get("deterministic")},
        "previous_unfinished": unfinished_attempts(run), **(extra or {})})
    return attempt


def trainer_snapshot(attempt: dict, trainer) -> None:
    """Grava ``attempt_NNN_trainer.json`` com o que a Ultralytics usou de fato (callback ``on_train_start``).

    Nunca lança exceção: falha aqui não pode parar o treino.
    """
    try:
        args = {key: (value if isinstance(value, (int, float, str, bool, type(None), list, dict)) else str(value))
                for key, value in vars(trainer.args).items()}
        optimizer = trainer.optimizer
        groups = [{"lr": group.get("lr"), "initial_lr": group.get("initial_lr"), "weight_decay": group.get("weight_decay"),
                   "params": len(group.get("params", []))} for group in optimizer.param_groups]
        weights = Path(str(getattr(trainer.args, "model", "") or ""))
        write_once(Path(attempt["run"]), f"attempt_{attempt['attempt']:03d}_trainer.json", {
            "attempt_id": attempt["attempt_id"], "recorded_at": _now(), "effective_args": args,
            "start_epoch": int(trainer.start_epoch), "epochs": int(trainer.epochs),
            "best_fitness_inherited": None if trainer.best_fitness is None else float(trainer.best_fitness),
            "batch_size": int(trainer.batch_size), "batches_per_epoch": len(trainer.train_loader),
            "train_images": len(trainer.train_loader.dataset), "val_images": len(trainer.test_loader.dataset),
            "accumulate": int(getattr(trainer, "accumulate", 1)), "optimizer": type(optimizer).__name__,
            "param_groups": groups, "device": str(trainer.device), "amp": bool(getattr(trainer, "amp", False)),
            "model_weights": {"path": str(weights), "sha256": file_hash(weights) if weights.is_file() else None},
            "save_dir": str(trainer.save_dir)})
    except Exception as error:  # noqa: BLE001 - registro nunca interrompe o treino
        print(f"ficha: não foi possível gravar o retrato do treinador: {error}")


def end_attempt(attempt: dict, status: str, reason: str | None, counters: dict | None = None) -> None:
    """Fecha a tentativa: grava ``attempt_NNN_end.json`` (uma única vez). Nunca lança exceção."""
    try:
        write_once(Path(attempt["run"]), f"attempt_{attempt['attempt']:03d}_end.json", {
            "attempt_id": attempt["attempt_id"], "ended_at": _now(), "status": status, "reason": reason,
            **(counters or {})})
    except FileExistsError:
        pass
    except Exception as error:  # noqa: BLE001
        print(f"ficha: não foi possível fechar a tentativa: {error}")


def record_evaluation(run: Path, evaluation: dict) -> str | None:
    """Grava ``evaluation_<data>.json`` na ficha de um run que já tem ficha (treinos novos).

    Retorna o nome do arquivo, ou ``None`` se o run não tem ficha (treinos antigos não são tocados).
    """
    folder = Path(run) / FOLDER
    if not folder.is_dir():
        return None
    name = f"evaluation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}.json"
    write_once(Path(run), name, evaluation)
    return name


def verify_run_card(run: Path) -> dict:
    """Confere a ficha: cada arquivo do índice existe com o mesmo hash; acusa arquivos fora do índice.

    Retorna ``{"integrity": "ok"|"alterada"|"sem_ficha", "changed", "missing", "unindexed", "files"}``.
    """
    folder = Path(run) / FOLDER
    index = folder / "index.jsonl"
    if not index.is_file():
        return {"integrity": "sem_ficha", "changed": [], "missing": [], "unindexed": [], "files": 0}
    entries = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed, missing = [], []
    for entry in entries:
        path = folder / entry["file"]
        if not path.is_file():
            missing.append(entry["file"])
        elif file_hash(path) != entry["sha256"]:
            changed.append(entry["file"])
    indexed = {entry["file"] for entry in entries}
    unindexed = sorted(path.name for path in folder.glob("*.json") if path.name not in indexed)
    return {"integrity": "ok" if not (changed or missing or unindexed) else "alterada", "changed": changed,
            "missing": missing, "unindexed": unindexed, "files": len(entries)}


def run_card_summary(run: Path) -> dict:
    """Resumo da ficha para relatórios: tentativas, como terminaram e integridade."""
    check = verify_run_card(run)
    folder = Path(run) / FOLDER
    attempts = []
    for number in _attempts(run):
        start = json.loads((folder / f"attempt_{number:03d}_start.json").read_text(encoding="utf-8"))
        end_path = folder / f"attempt_{number:03d}_end.json"
        end = json.loads(end_path.read_text(encoding="utf-8")) if end_path.is_file() else None
        attempts.append({"attempt": number, "kind": start.get("kind"), "host": start.get("host", {}).get("host"),
                         "started_at": start.get("started_at"), "commit": start.get("code", {}).get("commit"),
                         "dirty": start.get("code", {}).get("dirty"),
                         "status": end.get("status") if end else "sem_registro_de_fim",
                         "ended_at": end.get("ended_at") if end else None,
                         "global_epoch_reached": end.get("global_epoch_reached") if end else None,
                         "effective_seconds": end.get("effective_seconds") if end else None})
    return {"integrity": check["integrity"], "attempts": len(attempts), "attempt_list": attempts, "check": check,
            "effective_seconds_total": round(sum(item["effective_seconds"] or 0 for item in attempts), 1)}
