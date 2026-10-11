"""Servidor local da plataforma: inspeções, importação de mídia, laudos e tarefas.

Função no projeto: comando ``platform``. Aberta direto do disco, a página de
revisão funciona sozinha, mas o navegador não deixa um arquivo local ler
``runs/`` enquanto um treino escreve nele, receber fotos nem iniciar comandos.
Este servidor, só da biblioteca padrão e preso a 127.0.0.1, faz isso:

- ``GET /`` e ``GET /r/<pasta>/``: a página de cada inspeção (``review_render.build_page``)
  com o modelo atual e dados frescos; sem ``--registry`` o servidor descobre as
  inspeções em ``data/reviews/`` (modo espaço de trabalho) e abre a mais recente;
- ``GET /api/workspace``: lista das inspeções com fotos, decisões e sugestões, e o
  que este PC consegue gerar (PDF, quadros de vídeo);
- ``GET /api/operations``: treinos, curvas e comparações (``operations.py``) a cada consulta;
- ``GET /api/jobs`` / ``POST /api/jobs`` / ``POST /api/jobs/stop``: uma tarefa por vez
  (``train-pilot`` ou ``suggest``) rodando ``python -m dronecamp_ia`` com log em ``.runtime/jobs/``;
- ``POST /api/uploads`` → ``PUT /api/uploads/<lote>/files`` → ``POST /api/uploads/<lote>/ingest``:
  fotos e vídeos enviados pela página viram uma inspeção nova (``ingest.ingest_media``),
  com sugestões do modelo em seguida se o usuário pedir; ``GET /api/uploads/<lote>`` acompanha;
- ``POST /api/export/evidence`` e ``POST /api/report``: pacote ZIP de fotos demarcadas e
  laudo preliminar (``exports.py``) gravados em ``exportacoes/`` e ``laudos/`` da inspeção;
- ``GET /api/reviews/<pasta>/files``: laudos e pacotes já gerados para baixar;
- ``POST /api/feedback``: grava o arquivo de revisão exportado na pasta da revisão.

O que não faz: aprovar dados, adotar modelos, apagar inspeções ou inferir
gravidade. Toda requisição que altera algo exige o token sorteado na partida
(impede que outro site aberto no navegador dispare treinos ou envie arquivos).

Quando mexer: para liberar um comando novo na página, acrescente-o em
``_job_arguments`` com validação de cada opção; para servir um tipo novo de
arquivo, ajuste ``REVIEW_SUFFIXES``/``DOWNLOAD_SUFFIXES``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import html
import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
from urllib.parse import parse_qs, unquote, urlsplit
import webbrowser

from .config import ProjectConfig, detection_names, load_taxonomy
from .io import file_hash
from .operations import collect_operations, collect_review_files
from .review_render import build_page, empty_page_payload, load_page_payload

MAX_BODY_BYTES = 16 * 1024 * 1024
# Envio de mídia: um arquivo por requisição, direto para o disco (vídeos de drone passam de 1 GB).
MAX_UPLOAD_FILE_BYTES = 8 * 1024 ** 3
MAX_UPLOAD_BATCH_BYTES = 64 * 1024 ** 3
LOG_TAIL_LINES = 80
UPLOAD_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
# Arquivos da revisão que a página pede (fotos, evidências, dados) e os gerados para baixar.
REVIEW_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".json", ".csv"}
DOWNLOAD_FOLDERS = ("laudos", "exportacoes")
DOWNLOAD_SUFFIXES = {".zip", ".pdf", ".html", ".json", ".csv", ".txt", ".md"}
CONTENT_TYPES = {".webp": "image/webp", ".zip": "application/zip", ".pdf": "application/pdf", ".md": "text/markdown; charset=utf-8",
                 ".txt": "text/plain; charset=utf-8", ".html": "text/html; charset=utf-8", ".json": "application/json; charset=utf-8",
                 ".csv": "text/csv; charset=utf-8"}


class JobRunner:
    """Uma tarefa por vez; o processo filho é o mesmo CLI usado no terminal."""

    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.Lock()
        self.process: subprocess.Popen | None = None
        self.job: dict | None = None

    def start(self, kind: str, arguments: list[str]) -> dict:
        with self.lock:
            if self.process and self.process.poll() is None:
                raise RuntimeError("Já existe uma tarefa em andamento; aguarde ou interrompa antes de iniciar outra.")
            logs = self.root / ".runtime" / "jobs"
            logs.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            log_path = logs / f"{stamp}_{kind}.log"
            command = [sys.executable, "-m", "dronecamp_ia", kind, *arguments]
            stream = log_path.open("w", encoding="utf-8")
            # O filho herda o ambiente (o mesmo .venv) e roda na pasta sistema_ia.
            self.process = subprocess.Popen(command, cwd=self.root, stdout=stream, stderr=subprocess.STDOUT,
                                            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
            stream.close()
            self.job = {"kind": kind, "arguments": arguments, "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "log": log_path.relative_to(self.root).as_posix(), "pid": self.process.pid}
            return self.status()

    def stop(self) -> dict:
        with self.lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()
            return self.status()

    def status(self) -> dict:
        if not self.job:
            return {"state": "idle"}
        code = self.process.poll() if self.process else None
        state = "running" if code is None else ("finished" if code == 0 else "failed")
        log = self.root / self.job["log"]
        try:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-LOG_TAIL_LINES:]
        except OSError:
            tail = []
        return {**self.job, "state": state, "exit_code": code, "log_tail": tail}


def _inside(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _project_file(root: Path, value: str, folders: tuple[str, ...], suffix: str) -> str:
    """Caminho relativo existente dentro das pastas permitidas (nada fora de sistema_ia)."""
    if not isinstance(value, str) or not value:
        raise ValueError("Caminho ausente.")
    path = (root / value.replace("\\", "/")).resolve()
    if path.suffix.lower() != suffix or not path.is_file() or not any(_inside(path, root / folder) for folder in folders):
        raise ValueError(f"Arquivo não permitido: {value}")
    return path.relative_to(root.resolve()).as_posix()


def _integer(options: dict, key: str, low: int, high: int) -> list[str]:
    value = options.get(key)
    if value in (None, ""):
        return []
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{key} deve ser inteiro entre {low} e {high}.")
    return [f"--{key}", str(value)]


def _job_arguments(root: Path, registry_path: Path | None, kind: str, options: dict) -> list[str]:
    """Lista fechada de comandos e opções que a página pode iniciar; só ``suggest`` precisa de ``registry_path``."""
    if kind == "train-pilot":
        arguments = ["--data", _project_file(root, options.get("data"), ("data/pilot",), ".yaml")]
        if options.get("weights"):
            arguments += ["--weights", _project_file(root, options["weights"], ("models", "runs"), ".pt")]
        for key, low, high in (("epochs", 1, 1000), ("imgsz", 64, 2048), ("batch", 1, 64)):
            arguments += _integer(options, key, low, high)
        if "--imgsz" in arguments and int(arguments[arguments.index("--imgsz") + 1]) % 32:
            raise ValueError("imgsz deve ser múltiplo de 32.")
        if options.get("tile") is True:
            arguments.append("--tile")
        if options.get("gpu_eval") is True:
            arguments.append("--gpu-eval")
        return arguments
    if kind == "suggest":
        arguments = ["--weights", _project_file(root, options.get("weights"), ("models", "runs"), ".pt"),
                     "--registry", registry_path.resolve().relative_to(root.resolve()).as_posix()]
        conf = options.get("conf")
        if conf not in (None, ""):
            if not isinstance(conf, (int, float)) or not 0 < conf <= 1:
                raise ValueError("conf deve estar entre 0 e 1.")
            arguments += ["--conf", str(conf)]
        return arguments
    raise ValueError(f"Tarefa não permitida pela página: {kind}")


def _text(options: dict, key: str, limit: int, default: str = "") -> str:
    value = options.get(key)
    if value in (None, ""):
        return default
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{key} deve ser texto com até {limit} caracteres.")
    return value


def _report_details(value) -> dict | None:
    """Dados técnicos opcionais do laudo (data da inspeção, cobertura, responsável, ART...): só texto curto."""
    from .laudo_abnt import DETAIL_FIELDS
    if value in (None, {}):
        return None
    if not isinstance(value, dict):
        raise ValueError("details deve ser um objeto.")
    return {key: _text(value, key, 500) for key in DETAIL_FIELDS if key in value}


def _decisions_map(value) -> dict | None:
    """Decisões como a página exporta (lista com ``image_sha256``) viram o mapa que ``exports`` lê."""
    if value in (None, [], {}):
        return None
    if isinstance(value, dict):
        return value
    if not isinstance(value, list) or any(not isinstance(item, dict) or not item.get("image_sha256") for item in value):
        raise ValueError("decisions deve ser a lista de decisões exportada pela página.")
    decisions = {}
    for item in value:
        decisions[item["image_sha256"]] = {key: entry for key, entry in item.items() if key != "image_sha256"}
    return decisions


def capabilities() -> dict:
    """O que este PC consegue gerar; a página desliga o que falta em vez de falhar depois."""
    try:
        import reportlab  # noqa: F401
        pdf = True
    except ImportError:
        pdf = False
    try:
        import cv2  # noqa: F401
        frames = "opencv"
    except ImportError:
        frames = "ffmpeg" if shutil.which("ffmpeg") else None
    return {"pdf": pdf, "video_frames": frames}


class Workspace:
    """Inspeções abertas: lista fixa (``--registry``) ou tudo em ``data/reviews/`` (modo espaço de trabalho).

    Só lê. No modo espaço de trabalho a lista é relida a cada consulta, então
    uma inspeção importada ou criada pelo terminal aparece sem reiniciar.
    """

    def __init__(self, config: ProjectConfig, registries: list[Path] | None):
        self.config = config
        self.root = config.root.resolve()
        self.reviews_dir = self.root / "data" / "reviews"
        self.fixed = {path.parent.name: path.resolve() for path in registries} if registries else None
        self._cache: dict[str, tuple[tuple, dict]] = {}
        self._lock = threading.Lock()

    def reviews(self) -> dict[str, Path]:
        if self.fixed is not None:
            with self._lock:
                return dict(self.fixed)
        if not self.reviews_dir.is_dir():
            return {}
        return {path.parent.name: path for path in sorted(self.reviews_dir.glob("*/registry.json")) if path.is_file()}

    @staticmethod
    def page_ready(registry_path: Path) -> bool:
        try:
            load_page_payload(registry_path)
            return True
        except (OSError, ValueError):
            return False

    def register(self, name: str, path: Path) -> None:
        """Inspeção recém-importada: entra na lista fixa (no espaço de trabalho ela já é descoberta)."""
        with self._lock:
            self._cache.pop(name, None)
            if self.fixed is not None:
                self.fixed[name] = path.resolve()

    def default(self) -> str | None:
        """Primeira da lista fixa; no espaço de trabalho, a inspeção com página mais recente."""
        if self.fixed is not None:
            return next(iter(self.fixed))
        ready = [(path.stat().st_mtime, name) for name, path in self.reviews().items() if self.page_ready(path)]
        return max(ready)[1] if ready else None

    def _summary(self, name: str, path: Path) -> dict:
        """Contagens de uma inspeção, em cache enquanto registro, sugestões e caixa de entrada não mudam."""
        directory = path.parent
        inbox = sorted(directory.glob("feedback_inbox/*.json"))
        latest = inbox[-1] if inbox else None
        stamp = lambda file: (file.stat().st_mtime_ns, file.stat().st_size) if file and file.is_file() else None  # noqa: E731
        key = (stamp(path), stamp(directory / "browser_data.json"), stamp(directory / "suggestions.json"), stamp(latest))
        with self._lock:
            cached = self._cache.get(name)
            if cached and cached[0] == key:
                return dict(cached[1])
        registry = json.loads(path.read_text(encoding="utf-8"))
        images = registry.get("images") or []
        decided, decisions = 0, None
        if latest is not None:
            try:
                feedback = json.loads(latest.read_text(encoding="utf-8"))
                if feedback.get("registry_sha256") == file_hash(path):
                    decided = sum(1 for item in feedback.get("images") or [] if item.get("status") not in (None, "pending"))
                    decisions = _decisions_map(feedback.get("images"))
            except (OSError, ValueError):
                decided, decisions = 0, None
        if not decided:
            decided = sum(1 for item in images if item.get("human_approved") is True)
        pending = 0
        suggestions_path = directory / "suggestions.json"
        if suggestions_path.is_file():
            try:
                suggestions = json.loads(suggestions_path.read_text(encoding="utf-8"))
                if suggestions.get("registry_sha256") == file_hash(path):
                    pending = self._pending_suggestions(registry, decisions, suggestions)
            except (OSError, ValueError):
                pending = 0
        updated = max([path.stat().st_mtime] + [file.stat().st_mtime for file in inbox])
        summary = {"name": name, "version": registry.get("version") or name, "href": f"/r/{name}/", "images": len(images),
                   "decided": decided, "suggestions_pending": pending,
                   "buildings": sorted({item.get("building_group") for item in images if item.get("building_group")}),
                   "updated_at": datetime.fromtimestamp(updated, timezone.utc).isoformat(timespec="seconds"),
                   "page_ready": self.page_ready(path),
                   "scope_label": "importada pela plataforma" if registry.get("ingest") else None}
        with self._lock:
            self._cache[name] = (key, summary)
        return dict(summary)

    def _pending_suggestions(self, registry: dict, decisions: dict | None, suggestions: dict) -> int:
        """Sugestões que o último arquivo de revisão válido ainda não aceitou nem descartou (mesma regra da página e do laudo)."""
        from .exports import resolve_decisions
        try:
            class_count = len(detection_names(load_taxonomy(self.config.taxonomy_path)))
            return sum(entry["pending_suggestions"] for entry in resolve_decisions(registry, decisions, class_count, suggestions))
        except (OSError, ValueError, KeyError, TypeError):
            # Decisões ou sugestões fora do contrato: conta todas, como antes de qualquer revisão.
            return sum(len(values) for values in (suggestions.get("images") or {}).values() if isinstance(values, list))

    def describe(self, current: str | None) -> dict:
        reviews = []
        for name, path in self.reviews().items():
            try:
                summary = self._summary(name, path)
            except (OSError, ValueError) as error:
                summary = {"name": name, "href": f"/r/{name}/", "images": None, "page_ready": False, "error": str(error)}
            reviews.append({**summary, "current": name == current})
        reviews.sort(key=lambda item: (item.get("current", False), item.get("updated_at") or ""), reverse=True)
        return {"mode": "fixed" if self.fixed is not None else "workspace", "default": self.default(),
                "reviews_dir": self.reviews_dir.relative_to(self.root).as_posix(), "reviews": reviews, "capabilities": capabilities()}


class UploadManager:
    """Lotes de envio: arquivos ficam em ``.runtime/uploads/<lote>/`` até a importação, que cria a inspeção."""

    def __init__(self, root: Path, workspace: Workspace):
        self.root = root.resolve()
        self.workspace = workspace
        self.staging = self.root / ".runtime" / "uploads"
        self.batches: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.running: str | None = None

    def create(self, name) -> dict:
        from .ingest import slugify_review_name
        slug = slugify_review_name(name)
        if (self.workspace.reviews_dir / slug).exists():
            raise ValueError(f"Já existe uma inspeção chamada {slug}; escolha outro nome.")
        batch = secrets.token_urlsafe(12)
        directory = self.staging / batch
        directory.mkdir(parents=True)
        with self.lock:
            self.batches[batch] = {"name": slug, "directory": directory, "files": [], "bytes": 0, "state": "open",
                                   "message": "Aguardando arquivos.", "result": None, "error": None}
        return {"batch": batch, "name": slug}

    def _batch(self, batch: str, state: str | None = "open") -> dict:
        if not isinstance(batch, str) or not UPLOAD_ID.match(batch) or batch not in self.batches:
            raise ValueError("Lote de envio desconhecido.")
        entry = self.batches[batch]
        if state and entry["state"] != state:
            raise ValueError("Este lote já foi enviado para importação.")
        return entry

    def store(self, batch: str, filename: str, stream, length: int) -> dict:
        """Grava um arquivo do lote direto do corpo da requisição, em blocos."""
        from .ingest import media_kind
        entry = self._batch(batch)
        name = Path(unquote(filename or "")).name
        kind = media_kind(Path(name))
        if not name or kind is None:
            raise ValueError(f"Tipo de arquivo não suportado: {name or '(sem nome)'}. Envie fotos (JPG, PNG, WEBP, TIFF, BMP) ou vídeos (MP4, MOV, AVI, MKV).")
        if length <= 0 or length > MAX_UPLOAD_FILE_BYTES:
            raise ValueError("Arquivo vazio ou maior que o limite de envio.")
        if entry["bytes"] + length > MAX_UPLOAD_BATCH_BYTES:
            raise ValueError("Lote maior que o limite de envio; importe em partes.")
        target = entry["directory"] / name
        counter = 1
        while target.exists():  # dois arquivos com o mesmo nome (pastas diferentes) ficam ambos
            counter += 1
            target = entry["directory"] / f"{Path(name).stem}_{counter}{Path(name).suffix}"
        remaining = length
        with target.open("wb") as output:
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                output.write(chunk)
                remaining -= len(chunk)
        if remaining:
            target.unlink(missing_ok=True)
            raise ValueError("Envio interrompido antes do fim do arquivo.")
        entry["files"].append(target.name)
        entry["bytes"] += length
        entry["message"] = f"{len(entry['files'])} arquivo(s) recebidos."
        return {"stored": target.name, "bytes": length, "kind": kind, "files": len(entry["files"])}

    def start_ingest(self, batch: str, config: ProjectConfig, options: dict, jobs: JobRunner) -> dict:
        from .ingest import slugify_review_name
        entry = self._batch(batch)
        if not entry["files"]:
            raise ValueError("Nenhum arquivo recebido neste lote.")
        name = slugify_review_name(options.get("name") or entry["name"])
        group = _text(options, "group", 100)
        if not group.strip():
            raise ValueError("Informe a edificação ou campanha.")
        every = options.get("every_seconds", 2.0)
        frames = options.get("max_frames", 200)
        if isinstance(frames, float) and frames.is_integer():
            frames = int(frames)
        parameters = {"building_group": group.strip(), "every_seconds": every, "max_frames": frames,
                      "uploaded_by": _text(options, "uploaded_by", 100) or None, "campaign_notes": _text(options, "notes", 5000)}
        weights = options.get("weights") or None
        if weights:
            _project_file(self.root, weights, ("models", "runs"), ".pt")  # valida antes de importar
        with self.lock:
            if self.running and self.batches[self.running]["state"] == "running":
                raise RuntimeError("Outra importação está em andamento; aguarde terminar.")
            self.running = batch
            entry.update(state="running", message="Congelando fotos por hash e extraindo quadros dos vídeos…", name=name)
        worker = threading.Thread(target=self._run, args=(entry, config, parameters, weights, jobs), daemon=True)
        worker.start()
        return self.status(batch)

    def _run(self, entry: dict, config: ProjectConfig, parameters: dict, weights: str | None, jobs: JobRunner) -> None:
        from .ingest import ingest_media
        output = self.workspace.reviews_dir / entry["name"]
        try:
            inputs = [entry["directory"] / name for name in entry["files"]]
            # keep_sources: os arquivos enviados passam a morar em <inspeção>/midia_enviada/ (o lote temporário some).
            result = ingest_media(config, inputs, output, parameters.pop("building_group"), keep_sources=True, **parameters)
            result = {key: (str(value) if isinstance(value, Path) else value) for key, value in result.items()}
            result["href"] = f"/r/{entry['name']}/"
            self.workspace.register(entry["name"], output / "registry.json")
            result["suggest_job"] = None
            if weights:
                try:
                    status = jobs.start("suggest", _job_arguments(self.root, output / "registry.json", "suggest", {"weights": weights}))
                    result["suggest_job"] = {"state": status["state"], "log": status["log"]}
                except (RuntimeError, ValueError, OSError) as error:
                    result["suggest_job"] = {"state": "not_started", "error": str(error)}
            entry.update(state="finished", result=result, message="Inspeção criada.")
        except Exception as error:  # noqa: BLE001 - o erro vai para a página, não para o terminal
            entry.update(state="failed", error=str(error), message="A importação falhou.")
        finally:
            shutil.rmtree(entry["directory"], ignore_errors=True)
            self.workspace._cache.pop(entry["name"], None)

    def status(self, batch: str) -> dict:
        entry = self._batch(batch, state=None)
        return {"batch": batch, "name": entry["name"], "state": entry["state"], "files": len(entry["files"]), "bytes": entry["bytes"],
                "message": entry["message"], "result": entry["result"], "error": entry["error"]}


def _reviews_meta(reviews: dict[str, Path], current: str | None) -> str:
    """Lista das revisões abertas pelo servidor (seletor da fila); vazia com uma só revisão."""
    if len(reviews) < 2:
        return ""
    items = []
    for name, path in reviews.items():
        try:
            images = len(load_page_payload(path).get("images") or [])
        except (OSError, ValueError):
            images = None
        items.append({"name": name, "href": f"/r/{name}/", "images": images})
    return html.escape(json.dumps({"current": current, "items": items}, ensure_ascii=False), quote=True)


def make_handler(config: ProjectConfig, registry_path: Path | list[Path] | None, token: str, jobs: JobRunner):
    """Handler HTTP ligado a uma, várias ou todas as revisões; arquivos servidos só da revisão e de runs/.

    Com ``registry_path`` a primeira revisão abre em ``/``; sem ele (modo espaço
    de trabalho) abre a mais recente de ``data/reviews/`` e a lista é relida a
    cada consulta. Cada revisão também abre em ``/r/<pasta>/``: como a página usa
    caminhos relativos para as fotos, o prefixo faz cada foto vir da sua pasta.
    """
    root = config.root.resolve()
    if registry_path is None:
        paths = None
    else:
        paths = registry_path if isinstance(registry_path, list) else [registry_path]
    workspace = Workspace(config, paths)
    uploads = UploadManager(root, workspace)

    def split_review(path: str, reviews: dict[str, Path]) -> tuple[str | None, str]:
        """``/r/<pasta>/resto`` → (pasta, ``/resto``); demais caminhos → revisão padrão."""
        match = re.match(r"^/r/([^/]+)(/.*)?$", path)
        if not match:
            return workspace.default(), path
        name = unquote(match[1])
        return (name if name in reviews else None), (match[2] or "")

    def registry_for(options: dict, reviews: dict[str, Path]) -> Path:
        name = options.pop("review", None) or workspace.default()
        if name not in reviews:
            raise ValueError(f"Revisão não aberta neste servidor: {name}")
        return reviews[name]

    class Handler(BaseHTTPRequestHandler):
        server_version = "DroneCampPlatform/2"

        def log_message(self, format, *args):  # noqa: A002 - assinatura da biblioteca
            pass  # o terminal fica livre para o log do treino

        def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, target: Path, attachment: bool) -> None:
            """Arquivo em blocos (pacotes ZIP e PDFs grandes não passam inteiros pela memória)."""
            suffix = target.suffix.lower()
            kind = CONTENT_TYPES.get(suffix) or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(target.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if attachment:
                safe = re.sub(r"[^A-Za-z0-9._-]+", "_", target.name)
                self.send_header("Content-Disposition", f'attachment; filename="{safe}"')
            self.end_headers()
            with target.open("rb") as stream:
                shutil.copyfileobj(stream, self.wfile, 1024 * 1024)

        def _json(self, value: dict, status: int = 200) -> None:
            self._send(status, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _not_found(self, text: str = "Nao encontrado") -> None:
            self._send(404, text.encode("utf-8"), "text/plain; charset=utf-8")

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                raise ValueError("Conteúdo grande demais.")
            value = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(value, dict):
                raise ValueError("Envie um objeto JSON.")
            return value

        def _authorized(self) -> bool:
            return secrets.compare_digest(self.headers.get("X-DroneCamp-Token", ""), token)

        def _page(self, reviews: dict[str, Path], name: str | None) -> None:
            if name is None:
                payload, operations = empty_page_payload(config), collect_operations(config, None)
            else:
                payload = load_page_payload(reviews[name])
                operations = collect_operations(config, reviews[name].parent.resolve())
            page = build_page(payload, operations)
            # O token só existe na página servida; a cópia em disco nunca o contém.
            page = page.replace('<meta name="dronecamp-server" content="">', f'<meta name="dronecamp-server" content="{token}">', 1)
            page = page.replace('<meta name="dronecamp-reviews" content="">',
                                f'<meta name="dronecamp-reviews" content="{_reviews_meta(reviews, name)}">', 1)
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802 - nome exigido pela biblioteca
            parts = urlsplit(self.path)
            full, query = parts.path, parse_qs(parts.query)
            reviews = workspace.reviews()
            asked = (query.get("review") or [None])[0]
            if full == "/api/operations":
                name = asked if asked in reviews else workspace.default()
                return self._json(collect_operations(config, reviews[name].parent.resolve() if name else None))
            if full == "/api/jobs":
                return self._json(jobs.status())
            if full == "/api/workspace":
                return self._json(workspace.describe(asked if asked in reviews else None))
            match = re.match(r"^/api/reviews/([^/]+)/files$", full)
            if match:
                name = unquote(match[1])
                if name not in reviews:
                    return self._json({"error": "Revisão não aberta neste servidor."}, HTTPStatus.NOT_FOUND)
                return self._json({"review": name, "files": collect_review_files(reviews[name].parent)})
            match = re.match(r"^/api/uploads/([^/]+)$", full)
            if match:
                try:
                    return self._json(uploads.status(match[1]))
                except ValueError as error:
                    return self._json({"error": str(error)}, HTTPStatus.NOT_FOUND)
            name, path = split_review(full, reviews)
            if name is None and not full.startswith("/r/") and path in {"/", "/index.html"}:
                return self._page(reviews, None)  # espaço de trabalho vazio: a página abre sem inspeção
            if name is None:
                return self._not_found("Revisao nao aberta neste servidor")
            if path in {"/", "/index.html"}:
                try:
                    return self._page(reviews, name)
                except (OSError, ValueError) as error:
                    text = f"A página desta revisão ainda não foi gerada (render-review): {error}"
                    return self._send(HTTPStatus.CONFLICT, text.encode("utf-8"), "text/plain; charset=utf-8")
            if path == "":  # /r/<pasta> sem barra: relativos precisam da barra final
                self.send_response(HTTPStatus.MOVED_PERMANENTLY)
                self.send_header("Location", full + "/")
                self.end_headers()
                return None
            directory = reviews[name].parent.resolve()
            # Arquivos: /runs/... vem de sistema_ia/runs; o resto, da pasta da revisão.
            relative = re.sub(r"^/+", "", unquote(path))
            base = root if relative.startswith("runs/") else directory
            target = (base / relative).resolve()
            suffix = target.suffix.lower()
            if not _inside(target, base) or not target.is_file():
                return self._not_found()
            download = base is directory and relative.split("/", 1)[0] in DOWNLOAD_FOLDERS and suffix in DOWNLOAD_SUFFIXES
            if download:
                return self._send_file(target, attachment=suffix in {".zip", ".pdf"})
            if suffix not in REVIEW_SUFFIXES:
                return self._not_found()
            return self._send_file(target, attachment=False)

        def do_PUT(self) -> None:  # noqa: N802
            if not self._authorized():
                return self._json({"error": "Token ausente ou inválido."}, HTTPStatus.FORBIDDEN)
            match = re.match(r"^/api/uploads/([^/]+)/files$", self.path.split("?", 1)[0])
            if not match:
                return self._json({"error": "Rota desconhecida."}, HTTPStatus.NOT_FOUND)
            try:
                length = int(self.headers.get("Content-Length") or 0)
                return self._json(uploads.store(match[1], self.headers.get("X-File-Name", ""), self.rfile, length))
            except (ValueError, OSError) as error:
                self.close_connection = True  # o corpo pode não ter sido lido inteiro
                return self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                return self._json({"error": "Token ausente ou inválido."}, HTTPStatus.FORBIDDEN)
            try:
                body = self._body()
                reviews = workspace.reviews()
                path = self.path.split("?", 1)[0]
                if path == "/api/jobs":
                    kind = body.get("kind")
                    options = dict(body.get("options") or {})
                    # Só as sugestões precisam de uma revisão (a aberta na página); o treino roda com o espaço vazio.
                    selected = registry_for(options, reviews) if kind == "suggest" else None
                    return self._json(jobs.start(kind, _job_arguments(root, selected, kind, options)))
                if path == "/api/jobs/stop":
                    return self._json(jobs.stop())
                if path == "/api/feedback":
                    # O arquivo exportado já diz de qual registro veio (SHA-256).
                    digest = body.get("registry_sha256")
                    selected = next((value for value in reviews.values() if file_hash(value) == digest), None)
                    if selected is None:
                        raise ValueError("Arquivo de revisão de outra versão do registro.")
                    return self._json(_save_feedback(root, selected, body))
                if path == "/api/uploads":
                    return self._json(uploads.create(body.get("name")))
                match = re.match(r"^/api/uploads/([^/]+)/ingest$", path)
                if match:
                    return self._json(uploads.start_ingest(match[1], config, body, jobs))
                if path == "/api/export/evidence":
                    return self._json(_export_evidence(config, root, registry_for(body, reviews), body))
                if path == "/api/report":
                    return self._json(_build_report(config, root, registry_for(body, reviews), body))
                return self._json({"error": "Rota desconhecida."}, HTTPStatus.NOT_FOUND)
            except RuntimeError as error:
                return self._json({"error": str(error)}, HTTPStatus.CONFLICT)
            except (ValueError, OSError) as error:
                return self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    return Handler


def _export_evidence(config: ProjectConfig, root: Path, registry_path: Path, body: dict) -> dict:
    """ZIP de fotos demarcadas em ``exportacoes/`` da revisão (``exports.build_evidence_package``)."""
    from .exports import build_evidence_package
    max_side = body.get("max_side")
    if max_side is not None and (type(max_side) is not int or not 256 <= max_side <= 8192):
        raise ValueError("max_side deve ser vazio (original) ou um inteiro entre 256 e 8192.")
    only = body.get("only") or "all"
    if only not in {"all", "decided", "positive"}:
        raise ValueError("only deve ser all, decided ou positive.")
    result = build_evidence_package(config, registry_path, registry_path.parent / "exportacoes", decisions=_decisions_map(body.get("decisions")),
                                    max_side=max_side, only=only, include_suggestions=body.get("include_suggestions") is True)
    archive = Path(result["zip"])
    relative = archive.relative_to(registry_path.parent).as_posix()
    return {"images": result["images"], "boxes": result["boxes"], "bytes": result["bytes"], "sha256": result["sha256"],
            "file": relative, "href": relative, "saved": archive.relative_to(root).as_posix()}


def _build_report(config: ProjectConfig, root: Path, registry_path: Path, body: dict) -> dict:
    """Laudo técnico (estrutura ABNT, ``laudo_abnt``) em ``laudos/`` da revisão (``exports.build_report``); PDF só com reportlab."""
    from .exports import build_report
    result = build_report(config, registry_path, registry_path.parent / "laudos", decisions=_decisions_map(body.get("decisions")),
                          reviewer=_text(body, "reviewer", 100), title=_text(body, "title", 200) or None, client=_text(body, "client", 200),
                          site=_text(body, "site", 200), inspector=_text(body, "inspector", 200), notes=_text(body, "notes", 5000),
                          details=_report_details(body.get("details")),
                          pdf=body.get("pdf") is not False, only_positive_images=body.get("all_images") is not True)
    directory = registry_path.parent

    def href(value) -> str | None:
        return Path(value).relative_to(directory).as_posix() if value else None

    return {"images": result["images"], "occurrences": result["occurrences"], "pending_suggestions": result.get("pending_suggestions"),
            "html": href(result["html"]), "pdf": href(result.get("pdf")), "pdf_reason": result.get("pdf_reason"),
            "json": href(result.get("json")), "directory": Path(result["directory"]).relative_to(root).as_posix()}


def _save_feedback(root: Path, registry_path: Path, feedback: dict) -> dict:
    """Grava o arquivo exportado ao lado do registro; não importa nem aprova nada."""
    if feedback.get("schema_version") != 1 or feedback.get("registry_sha256") != file_hash(registry_path):
        raise ValueError("Arquivo de revisão de outra versão do registro.")
    if not isinstance(feedback.get("images"), list) or not feedback["images"]:
        raise ValueError("Arquivo de revisão sem decisões.")
    reviewer = re.sub(r"[^A-Za-z0-9_-]+", "_", str(feedback.get("reviewer_id") or "revisor"))[:40]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = registry_path.parent / "feedback_inbox" / f"{stamp}_{reviewer}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:  # "x": nunca sobrescreve
        json.dump(feedback, stream, ensure_ascii=False, indent=2)
    return {"saved": target.resolve().relative_to(root).as_posix(), "decisions": len(feedback["images"])}


def serve_platform(config: ProjectConfig, registry: Path | list[Path] | None = None, port: int = 8765, open_browser: bool = True) -> None:
    """Inicia o servidor em 127.0.0.1 até Ctrl+C.

    Com ``registry`` (um ou vários), só essas revisões ficam abertas; sem ele, o
    servidor trabalha com tudo o que houver em ``data/reviews/`` e com as
    inspeções importadas pela página.
    """
    paths: list[Path] | None = None
    if registry:
        paths = []
        for value in (registry if isinstance(registry, list) else [registry]):
            # Caminho relativo vale a partir do terminal ou, se não existir, da pasta sistema_ia.
            path = (value if value.is_absolute() or value.exists() else config.root / value).resolve()
            load_page_payload(path)  # falha cedo se a página não foi gerada para este registro
            paths.append(path)
        if len({path.parent.name for path in paths}) != len(paths):
            raise ValueError("Duas revisões com o mesmo nome de pasta; o endereço /r/<pasta>/ ficaria ambíguo.")
    token = secrets.token_urlsafe(24)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(config, paths, token, JobRunner(config.root)))
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    reviews = Workspace(config, paths).reviews()
    print(f"Plataforma DroneCamp em {url} (Ctrl+C para encerrar). Inspeções abertas: {len(reviews)}.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
