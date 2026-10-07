"""Servidor local da plataforma: monitoramento ao vivo e tarefas iniciadas pela página.

Função no projeto: comando ``platform``. Aberta direto do disco, a página de
revisão funciona sozinha, mas o navegador não deixa um arquivo local ler
``runs/`` enquanto um treino escreve nele nem iniciar comandos. Este servidor,
só da biblioteca padrão e preso a 127.0.0.1, faz essas duas coisas:

- ``GET /``: a página com o modelo atual e dados frescos (``review_render.build_page``);
- ``GET /api/operations``: treinos, curvas e comparações (``operations.py``) a cada consulta;
- ``GET /api/jobs`` / ``POST /api/jobs`` / ``POST /api/jobs/stop``: uma tarefa por vez
  (``train-pilot`` ou ``suggest``) rodando ``python -m dronecamp_ia`` com log em ``.runtime/jobs/``;
- ``POST /api/feedback``: grava o arquivo de revisão exportado na pasta da revisão.

O que não faz: importar revisões, aprovar dados, adotar modelos ou apagar
arquivos. Toda requisição que altera algo exige o token sorteado na partida
(impede que outro site aberto no navegador dispare treinos).

Quando mexer: para liberar um comando novo na página, acrescente-o em
``_job_arguments`` com validação de cada opção.
"""

from __future__ import annotations

from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import mimetypes
import os
import re
import secrets
import subprocess
import sys
import threading
from urllib.parse import unquote
import html
import webbrowser

from .config import ProjectConfig
from .io import file_hash
from .operations import collect_operations
from .review_render import build_page, load_page_payload

MAX_BODY_BYTES = 5 * 1024 * 1024
LOG_TAIL_LINES = 80


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


def _job_arguments(root: Path, registry_path: Path, kind: str, options: dict) -> list[str]:
    """Lista fechada de comandos e opções que a página pode iniciar."""
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
                     "--registry", registry_path.relative_to(root.resolve()).as_posix()]
        conf = options.get("conf")
        if conf not in (None, ""):
            if not isinstance(conf, (int, float)) or not 0 < conf <= 1:
                raise ValueError("conf deve estar entre 0 e 1.")
            arguments += ["--conf", str(conf)]
        return arguments
    raise ValueError(f"Tarefa não permitida pela página: {kind}")


def _reviews_meta(reviews: dict[str, Path], current: str) -> str:
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


def make_handler(config: ProjectConfig, registry_path: Path | list[Path], token: str, jobs: JobRunner):
    """Handler HTTP ligado a uma ou mais revisões; arquivos servidos só da revisão e de runs/.

    A primeira revisão abre em ``/``; cada uma também abre em ``/r/<pasta>/``. Como a
    página usa caminhos relativos para as fotos, o prefixo faz cada foto vir da
    pasta da sua própria revisão.
    """
    root = config.root.resolve()
    paths = registry_path if isinstance(registry_path, list) else [registry_path]
    reviews = {path.parent.name: path for path in paths}
    default = paths[0].parent.name

    def split_review(path: str) -> tuple[str | None, str]:
        """``/r/<pasta>/resto`` → (pasta, ``/resto``); demais caminhos → revisão padrão."""
        match = re.match(r"^/r/([^/]+)(/.*)?$", path)
        if not match:
            return default, path
        name = unquote(match[1])
        return (name if name in reviews else None), (match[2] or "")

    def registry_for(options: dict) -> Path:
        name = options.pop("review", None) or default
        if name not in reviews:
            raise ValueError(f"Revisão não aberta neste servidor: {name}")
        return reviews[name]

    class Handler(BaseHTTPRequestHandler):
        server_version = "DroneCampPlatform/1"

        def log_message(self, format, *args):  # noqa: A002 - assinatura da biblioteca
            pass  # o terminal fica livre para o log do treino

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value: dict, status: int = 200) -> None:
            self._send(status, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                raise ValueError("Conteúdo grande demais.")
            value = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(value, dict):
                raise ValueError("Envie um objeto JSON.")
            return value

        def do_GET(self) -> None:  # noqa: N802 - nome exigido pela biblioteca
            full = self.path.split("?", 1)[0]
            if full == "/api/operations":
                return self._json(collect_operations(config, reviews[default].parent.resolve()))
            if full == "/api/jobs":
                return self._json(jobs.status())
            name, path = split_review(full)
            if name is None:
                return self._send(404, b"Revisao nao aberta neste servidor", "text/plain; charset=utf-8")
            if path == "":  # /r/<pasta> sem barra: relativos precisam da barra final
                self.send_response(HTTPStatus.MOVED_PERMANENTLY)
                self.send_header("Location", full + "/")
                self.end_headers()
                return None
            directory = reviews[name].parent.resolve()
            if path in {"/", "/index.html"}:
                payload = load_page_payload(reviews[name])
                page = build_page(payload, collect_operations(config, directory))
                # O token só existe na página servida; a cópia em disco nunca o contém.
                marker = '<meta name="dronecamp-server" content="">'
                page = page.replace(marker, f'<meta name="dronecamp-server" content="{token}">', 1)
                page = page.replace('<meta name="dronecamp-reviews" content="">',
                                    f'<meta name="dronecamp-reviews" content="{_reviews_meta(reviews, name)}">', 1)
                return self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            # Arquivos: /runs/... vem de sistema_ia/runs; o resto, da pasta da revisão.
            relative = re.sub(r"^/+", "", unquote(path))
            base = root if relative.startswith("runs/") else directory
            target = (base / relative).resolve()
            if not _inside(target, base) or not target.is_file() or target.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".json", ".csv"}:
                return self._send(404, b"Nao encontrado", "text/plain; charset=utf-8")
            # .webp não está no registro de tipos de todo Windows; com nosniff, octet-stream não abre a foto.
            kind = {".webp": "image/webp"}.get(target.suffix.lower()) or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            return self._send(200, target.read_bytes(), kind)

        def do_POST(self) -> None:  # noqa: N802
            if not secrets.compare_digest(self.headers.get("X-DroneCamp-Token", ""), token):
                return self._json({"error": "Token ausente ou inválido."}, HTTPStatus.FORBIDDEN)
            try:
                body = self._body()
                if self.path == "/api/jobs":
                    kind = body.get("kind")
                    options = dict(body.get("options") or {})
                    selected = registry_for(options)  # sugestões vão para a revisão aberta na página
                    return self._json(jobs.start(kind, _job_arguments(root, selected, kind, options)))
                if self.path == "/api/jobs/stop":
                    return self._json(jobs.stop())
                if self.path == "/api/feedback":
                    # O arquivo exportado já diz de qual registro veio (SHA-256).
                    digest = body.get("registry_sha256")
                    selected = next((path for path in reviews.values() if file_hash(path) == digest), reviews[default])
                    return self._json(_save_feedback(root, selected, body))
                return self._json({"error": "Rota desconhecida."}, HTTPStatus.NOT_FOUND)
            except RuntimeError as error:
                return self._json({"error": str(error)}, HTTPStatus.CONFLICT)
            except (ValueError, OSError) as error:
                return self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    return Handler


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


def serve_platform(config: ProjectConfig, registry: Path | list[Path], port: int = 8765, open_browser: bool = True) -> None:
    """Inicia o servidor em 127.0.0.1 até Ctrl+C; várias revisões viram um seletor na fila."""
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
    print(f"Plataforma DroneCamp em {url} (Ctrl+C para encerrar).", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
