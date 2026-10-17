"""Aprendizado mútuo distribuído na rede local: um coordenador e trabalhadores por máquina.

Função no projeto: treinar YOLO26, RF-DETR e D-FINE em máquinas diferentes da
rede local (ex.: HP-Z230 e 6006U, depois o 4600G) sem nenhuma delas precisar
segurar os três modelos. Cada trabalhador é dono de um ou mais modelos; a cada
rodada as máquinas trocam só as PREVISÕES dos modelos nas fotos de treino
(``mutual_learning.build_peer_cache``, ~3 MB por modelo), não gradientes.

Por que previsões e não gradientes (DDP): sincronizar os gradientes do yolo26x
(59 M parâmetros × 4 bytes ≈ 236 MB) a cada passo, numa rede de 1 Gbit/s
(≈ 110 MB/s na prática), custaria uns 2 s de rede por passo e por máquina, e as
máquinas mais lentas seguram as outras. Aqui a rede trabalha uma vez por rodada.

Protocolo (rodadas):
- Rodada 0: cada modelo treina sozinho (modo ``solo``, λ = 0). É também a linha
  de base individual que a recomendação "treinar os três separados primeiro" pede.
- Fim de cada rodada: o trabalhador grava o cache das previsões do modelo no
  split de treino e envia ao coordenador junto com o mAP50 de validação.
- Rodada r ≥ 1: quando os caches de todos os modelos da rodada r−1 chegaram, o
  coordenador abre a rodada r. Cada trabalhador baixa os caches dos OUTROS
  modelos e treina no modo ``cached`` (DML com pares defasados uma rodada),
  continuando dos próprios pesos ``last`` da rodada anterior.
- O coordenador escolhe, por modelo e por rodada, o multiplicador ``a`` do λ com
  o mesmo bandit UCB descontado do treino local (``dml_controller``); a
  recompensa é o Δ mAP50 de validação que o trabalhador informou.

Segurança e robustez:
- Só JSON e ``.npz`` sem pickle (``numpy.load(allow_pickle=False)``) atravessam a
  rede; nenhum peso de modelo é enviado. Toda requisição exige o cabeçalho
  ``X-DroneCamp-Token`` (segredo gerado pelo coordenador; não grave em repositório).
- Nomes de modelo e números de rodada são conferidos contra o ``configs/dml.json``
  antes de virar caminho de arquivo (sem ``..``).
- O dataset precisa ser o MESMO em todas as máquinas: o coordenador recusa um
  trabalhador cuja impressão digital (``dataset_fingerprint``) for diferente.
- Estado do coordenador em ``state.json`` (sobrevive a reinício do PC); o
  trabalhador retoma a rodada em que estava.

Conteúdo do arquivo:
- ``dataset_fingerprint``: impressão digital dos splits train/val.
- ``CoordinatorState`` + ``serve_coordinator`` (ENTRADA de ``dml-lan-coordinator``).
- ``CoordinatorClient`` + ``run_worker`` (ENTRADA de ``dml-lan-worker``).

O que lê e grava no disco:
- Coordenador: ``runs/dml_lan_<data>_<id>/state.json``, ``rounds/<r>/<modelo>.npz``,
  ``rounds/<r>/<modelo>.json`` (métricas enviadas) e ``rounds/<r>/joint.json`` (ficha da
  rodada fechada: versão de cada modelo, máquina, hashes e decisão do RL).
- Trabalhador: ``runs/dml_lan_worker_<data>_<id>/<modelo>/round_<r>/`` (uma pasta de
  ``train_mutual`` por rodada, com pesos ``last``/``best``), ``peers/<r>/`` e
  ``worker_log.jsonl`` (uma linha por rodada e modelo: espera, transferências, tentativas).

Registro da coordenação (``state.json``, chave ``coordination``): por rodada e modelo,
quando o trabalho foi atribuído e concluído, por qual máquina, bytes enviados e
baixados, tempo de espera e tentativas informados pelo trabalhador, e as falhas
(envio recusado, rodada errada) com o motivo.

Onde mexer:
- Porta padrão: ``DEFAULT_PORT``. Tempo entre consultas do trabalhador: ``POLL_SECONDS``.
- Regra de abertura da rodada (hoje: todos os modelos enviaram): ``CoordinatorState.task``.
- Liberar a porta no Windows do coordenador é feito pelo usuário (comando em
  docs/aprendizado_mutuo_dml_rl.md); este módulo não mexe no firewall.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import secrets
import threading
import time
from urllib import error as urlerror
from urllib import request as urlrequest

from .io import file_hash, make_run_directory, write_json

# Porta TCP padrão do coordenador (fora das usadas pela plataforma, 8765).
DEFAULT_PORT = 8790
# Segundos entre consultas do trabalhador enquanto espera os pares.
POLL_SECONDS = 20
# Tamanho máximo aceito por upload de cache (bytes). 1.234 recortes ≈ 3 MB; folga grande.
MAX_UPLOAD = 256 * 1024 * 1024
# Cabeçalho do segredo compartilhado.
TOKEN_HEADER = "X-DroneCamp-Token"
_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def dataset_fingerprint(data_path: Path, imgsz: int) -> str:
    """SHA-256 dos nomes, tamanhos e conteúdo das labels dos splits train e val.

    Duas máquinas com a mesma cópia do dataset têm a mesma impressão; foto ou
    label diferente muda o valor (lê o conteúdo de cada foto: ~1 s por 100 MB).
    """
    from .mutual_learning import TileDataset, label_path

    from .mutual_learning import image_key

    digest = hashlib.sha256()
    for split in ("train", "val"):
        dataset = TileDataset(data_path, split, imgsz)
        digest.update(f"{split}:{len(dataset)}".encode())
        for image in dataset.images:
            # Caminho relativo (a chave do cache) + conteúdo da foto + label.
            digest.update(image_key(Path(image), dataset.root).encode())
            digest.update(hashlib.sha256(Path(image).read_bytes()).digest())
            label = label_path(Path(image))
            digest.update(label.read_bytes() if label.exists() else b"-")
    return digest.hexdigest()


def _weights_hash(path: Path) -> str | None:
    """SHA-256 dos pesos: do arquivo, ou de uma pasta (D-FINE) pelos hashes dos arquivos em ordem."""
    path = Path(path)
    if path.is_file():
        return file_hash(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for item in sorted(path.rglob("*")):
            if item.is_file():
                digest.update(item.relative_to(path).as_posix().encode())
                digest.update(file_hash(item).encode())
        return digest.hexdigest()
    return None


# ---------------------------------------------------------------------------
# Coordenador
# ---------------------------------------------------------------------------

class CoordinatorState:
    """Estado das rodadas, persistido em ``state.json``.

    Parâmetros:
        directory: pasta do experimento (``runs/dml_lan_*``).
        settings: saída de ``load_dml_config``.
        rounds: total de rodadas (a 0 é a individual).
        epochs_per_round: épocas de cada rodada.

    Chamada por: ``serve_coordinator`` e testes.
    """

    def __init__(self, directory: Path, settings: dict, rounds: int, epochs_per_round: int):
        from .dml_controller import DiscountedUCB

        self.directory = Path(directory)
        self.lock = threading.Lock()
        self.models = [model["name"] for model in settings["models"]]
        state_path = self.directory / "state.json"
        if state_path.exists():
            self.data = json.loads(state_path.read_text(encoding="utf-8"))
            self.controllers = {name: DiscountedUCB.from_dict(value) for name, value in self.data["controllers"].items()}
            # state.json de versão anterior (sem o registro da coordenação): completa as chaves.
            self.data.setdefault("decisions", {})
            self.data.setdefault("coordination", {"assignments": {}, "downloads": {}, "failures": []})
            return
        rl = settings["rl"]
        self.controllers = {name: DiscountedUCB(rl["arms"], rl["gamma"], rl["exploration"]) for name in self.models}
        self.data = {"rounds": int(rounds), "epochs_per_round": int(epochs_per_round), "round": 0,
                     "settings": settings, "fingerprint": None, "workers": {}, "uploads": {"0": {}},
                     "multipliers": {"0": {name: 0.0 for name in self.models}}, "arms": {"0": {}},
                     "decisions": {}, "coordination": {"assignments": {}, "downloads": {}, "failures": []},
                     "created_at": datetime.now(timezone.utc).isoformat(), "complete": False}
        self.save()

    def save(self) -> None:
        """Grava ``state.json`` (estado + controladores)."""
        self.data["controllers"] = {name: policy.to_dict() for name, policy in self.controllers.items()}
        write_json(self.directory / "state.json", self.data)

    def register(self, payload: dict) -> dict:
        """Registra um trabalhador; confere modelos e impressão digital do dataset.

        Erros: ValueError com a mensagem devolvida ao trabalhador (HTTP 400).
        """
        with self.lock:
            models = payload.get("models") or []
            unknown = [name for name in models if name not in self.models]
            if not models or unknown:
                raise ValueError(f"Modelos desconhecidos: {unknown or models}; o coordenador conhece {self.models}.")
            fingerprint = payload.get("fingerprint")
            if self.data["fingerprint"] is None:
                self.data["fingerprint"] = fingerprint
            elif fingerprint != self.data["fingerprint"]:
                raise ValueError("Dataset diferente do das outras máquinas (impressão digital não confere). "
                                 "Copie a mesma pasta do dataset para esta máquina.")
            for name in models:
                self.data["workers"][name] = {"host": payload.get("host"), "device": payload.get("device"),
                                              "registered_at": datetime.now(timezone.utc).isoformat()}
            self.save()
            return {"rounds": self.data["rounds"], "epochs_per_round": self.data["epochs_per_round"],
                    "round": self.data["round"], "models": self.models}

    def task(self, model: str) -> dict:
        """O que ``model`` deve fazer agora: treinar a rodada r, esperar ou parar.

        Retorna ``{"action": "train"|"wait"|"done", "round", "lambda", "peers"}``.
        A rodada r está aberta quando todos os modelos enviaram a r−1; o modelo
        espera se já enviou a rodada atual.
        """
        from .mutual_learning import ramp_weight

        with self.lock:
            if self.data["complete"]:
                return {"action": "done", "round": self.data["round"]}
            current = self.data["round"]
            uploads = self.data["uploads"].get(str(current), {})
            if model in uploads:
                return {"action": "wait", "round": current}
            peers = [name for name in self.models if name != model] if current > 0 else []
            multiplier = self.data["multipliers"].get(str(current), {}).get(model, 0.0)
            # Registro: a primeira entrega da tarefa marca a atribuição (o trabalhador consulta várias vezes).
            assigned = self.data["coordination"]["assignments"].setdefault(str(current), {})
            if model not in assigned:
                assigned[model] = {"host": self.data["workers"].get(model, {}).get("host"),
                                   "assigned_at": datetime.now(timezone.utc).isoformat(), "completed_at": None}
                self.save()
            cons = self.data["settings"]["consistency"]
            weight = 0.0 if current == 0 else ramp_weight(current - 1, int(cons.get("ramp_rounds", 2)),
                                                          float(cons["lambda"])) * multiplier
            return {"action": "train", "round": current, "lambda": weight, "multiplier": multiplier,
                    "peers": peers, "epochs": self.data["epochs_per_round"]}

    def receive(self, round_index: int, model: str, cache: bytes, metrics: dict) -> dict:
        """Guarda o cache e as métricas de ``model`` na rodada; abre a próxima se todos chegaram.

        Erros: ValueError para rodada errada, modelo desconhecido ou ``.npz`` inválido.
        """
        import numpy as np

        with self.lock:
            try:
                if model not in self.models:
                    raise ValueError(f"Modelo desconhecido: {model}.")
                if round_index != self.data["round"]:
                    raise ValueError(f"Rodada {round_index} não está aberta (atual: {self.data['round']}).")
                # Valida o arquivo antes de gravar: sem pickle e com as chaves esperadas.
                with np.load(io.BytesIO(cache), allow_pickle=False) as data:
                    missing = {"keys", "offsets", "boxes", "probs"} - set(data.files)
                    if missing:
                        raise ValueError(f"Cache sem {sorted(missing)}.")
            except ValueError as error:
                # Falha registrada antes de virar HTTP 400 (o trabalhador vê a mesma mensagem).
                self.data["coordination"]["failures"].append({
                    "at": datetime.now(timezone.utc).isoformat(), "event": "upload_recusado", "model": model,
                    "round": round_index, "reason": str(error)})
                self.save()
                raise
            folder = self.directory / "rounds" / str(round_index)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{model}.npz").write_bytes(cache)
            write_json(folder / f"{model}.json", metrics)
            received_at = datetime.now(timezone.utc).isoformat()
            self.data["uploads"].setdefault(str(round_index), {})[model] = {
                "val_map50": metrics.get("val_map50"), "received_at": received_at, "bytes": len(cache),
                "sha256": hashlib.sha256(cache).hexdigest(), "host": self.data["workers"].get(model, {}).get("host"),
                "weights_sha256": metrics.get("weights_sha256"), "worker": metrics.get("worker")}
            assigned = self.data["coordination"]["assignments"].setdefault(str(round_index), {}).setdefault(
                model, {"host": self.data["workers"].get(model, {}).get("host"), "assigned_at": None})
            assigned["completed_at"] = received_at
            self._maybe_advance()
            self.save()
            return {"round": self.data["round"], "complete": self.data["complete"]}

    def _maybe_advance(self) -> None:
        """Fecha a rodada quando todos enviaram: recompensa do RL e multiplicadores da próxima."""
        from .dml_controller import decision_record

        current = self.data["round"]
        uploads = self.data["uploads"].get(str(current), {})
        if set(uploads) != set(self.models):
            return
        # Recompensa: Δ mAP50 de validação em relação à rodada anterior (rodada 0 não pontua).
        decisions = {}
        if current > 0:
            previous = self.data["uploads"].get(str(current - 1), {})
            choices = self.data.get("choices", {}).get(str(current), {})
            for name in self.models:
                arm = self.data["arms"].get(str(current), {}).get(name)
                if arm is not None:
                    choice = choices.get(name) or {"arm": arm, "multiplier": self.controllers[name].arms[arm]}
                    decisions[name] = decision_record(
                        self.controllers[name], choice, previous.get(name, {}).get("val_map50"),
                        uploads[name].get("val_map50"), self.data["multipliers"][str(current)][name], step=current)
            self.data["decisions"][str(current)] = decisions
        # Ficha da rodada fechada: qual versão de cada modelo entrou no "checkpoint conjunto".
        write_json(self.directory / "rounds" / str(current) / "joint.json", {
            "round": current, "closed_at": datetime.now(timezone.utc).isoformat(),
            "models": {name: {key: uploads[name].get(key) for key in ("host", "sha256", "bytes", "weights_sha256",
                                                                     "val_map50", "received_at")}
                       for name in self.models},
            "multipliers": self.data["multipliers"].get(str(current), {}), "rl_decisions": decisions,
            "fingerprint": self.data.get("fingerprint")})
        if current + 1 >= self.data["rounds"]:
            self.data["complete"] = True
            return
        nxt = current + 1
        self.data["round"] = nxt
        self.data["uploads"][str(nxt)] = {}
        if not self.data["settings"]["rl"].get("enabled", True):
            # RL desligado no configs/dml.json: multiplicador fixo 1 (λ da configuração com a rampa).
            self.data["arms"][str(nxt)] = {}
            self.data["multipliers"][str(nxt)] = {name: 1.0 for name in self.models}
            return
        choices = {name: self.controllers[name].select_details() for name in self.models}
        arms = {name: choice["arm"] for name, choice in choices.items()}
        self.data.setdefault("choices", {})[str(nxt)] = choices
        self.data["arms"][str(nxt)] = arms
        self.data["multipliers"][str(nxt)] = {name: self.controllers[name].arms[arm] for name, arm in arms.items()}

    def peer_file(self, round_index: int, model: str) -> Path:
        """Caminho do cache de ``model`` na rodada (conferido; nunca sai da pasta do experimento)."""
        if model not in self.models or round_index < 0:
            raise ValueError("Pedido de cache inválido.")
        return self.directory / "rounds" / str(int(round_index)) / f"{model}.npz"

    def note_download(self, round_index: int, model: str, size: int) -> None:
        """Conta um cache servido a um trabalhador (bytes baixados por rodada e modelo)."""
        with self.lock:
            bucket = self.data["coordination"]["downloads"].setdefault(str(round_index), {}).setdefault(
                model, {"count": 0, "bytes": 0})
            bucket["count"] += 1
            bucket["bytes"] += int(size)
            self.save()

    def public(self) -> dict:
        """Resumo para ``GET /status`` (sem o token, sem caminhos locais)."""
        with self.lock:
            return {key: self.data[key] for key in ("round", "rounds", "epochs_per_round", "complete", "workers",
                                                    "uploads", "multipliers", "decisions", "coordination")}


def _handler(state: CoordinatorState, token: str):
    """Classe de handler HTTP ligada ao estado e ao token (fechamento)."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "DroneCampDML/1"

        def log_message(self, format, *args):  # noqa: A002 - assinatura da biblioteca padrão
            return

        def _reply(self, code: int, payload: dict | bytes, content_type: str = "application/json") -> None:
            body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            # ATENCAO: compare_digest evita medir o token pelo tempo de resposta.
            if not secrets.compare_digest(self.headers.get(TOKEN_HEADER, ""), token):
                self._reply(401, {"error": "token inválido"})
                return False
            return True

        def _body(self) -> bytes:
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_UPLOAD:
                raise ValueError("Arquivo grande demais.")
            return self.rfile.read(length)

        def do_GET(self):  # noqa: N802 - nome exigido pela biblioteca padrão
            if not self._authorized():
                return
            try:
                parts = [part for part in self.path.split("?")[0].split("/") if part]
                if parts == ["status"]:
                    return self._reply(200, state.public())
                if len(parts) == 2 and parts[0] == "task" and _NAME.match(parts[1]):
                    return self._reply(200, state.task(parts[1]))
                if len(parts) == 3 and parts[0] == "peer" and parts[1].isdigit() and _NAME.match(parts[2]):
                    path = state.peer_file(int(parts[1]), parts[2])
                    if not path.exists():
                        return self._reply(404, {"error": "cache ainda não recebido"})
                    body = path.read_bytes()
                    state.note_download(int(parts[1]), parts[2], len(body))
                    return self._reply(200, body, "application/octet-stream")
                return self._reply(404, {"error": "rota desconhecida"})
            except ValueError as error:
                return self._reply(400, {"error": str(error)})

        def do_POST(self):  # noqa: N802
            if not self._authorized():
                return
            try:
                parts = [part for part in self.path.split("?")[0].split("/") if part]
                if parts == ["register"]:
                    return self._reply(200, state.register(json.loads(self._body() or b"{}")))
                if len(parts) == 3 and parts[0] == "upload" and parts[1].isdigit() and _NAME.match(parts[2]):
                    metrics = json.loads(self.headers.get("X-DroneCamp-Metrics", "{}"))
                    return self._reply(200, state.receive(int(parts[1]), parts[2], self._body(), metrics))
                return self._reply(404, {"error": "rota desconhecida"})
            except (ValueError, json.JSONDecodeError, OSError) as error:
                return self._reply(400, {"error": str(error)})

    return Handler


def serve_coordinator(config_root: Path, settings: dict, host: str, port: int = DEFAULT_PORT, rounds: int = 4,
                      epochs_per_round: int = 5, token: str | None = None, resume: Path | None = None,
                      ready: threading.Event | None = None, stop: threading.Event | None = None) -> Path:
    """ENTRADA de ``dml-lan-coordinator``: serve as rodadas até todas terminarem (ou Ctrl+C).

    Parâmetros:
        config_root: pasta ``sistema_ia`` (para ``runs/``).
        settings: ``load_dml_config``.
        host: IP desta máquina na rede local (ex.: ``192.168.0.82``). ATENCAO: não use
            ``0.0.0.0`` numa rede com convidados; o token é a única proteção.
        port, rounds, epochs_per_round: ver ``docs/aprendizado_mutuo_dml_rl.md``.
        token: segredo compartilhado; ``None`` gera um e imprime (copie para os trabalhadores).
        resume: pasta de um coordenador anterior (retoma do ``state.json``).
        ready, stop: eventos para testes (servidor em thread).

    Retorna: a pasta do experimento.
    """
    directory = Path(resume) if resume else make_run_directory(Path(config_root), "dml_lan")
    state = CoordinatorState(directory, settings, rounds, epochs_per_round)
    token = token or secrets.token_urlsafe(18)
    server = ThreadingHTTPServer((host, int(port)), _handler(state, token))
    server.timeout = 1.0
    if ready is None:
        print(f"Coordenador DML em http://{host}:{port}  pasta: {directory}")
        print(f"Token (passe com --token nos trabalhadores): {token}")
    else:
        ready.set()
    try:
        while not state.data["complete"] and not (stop is not None and stop.is_set()):
            server.handle_request()
        # Mantém o servidor alguns segundos depois do fim para os trabalhadores lerem "done".
        deadline = time.time() + (2 if stop is not None else 120)
        while time.time() < deadline and not (stop is not None and stop.is_set()):
            server.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.save()
    return directory


# ---------------------------------------------------------------------------
# Trabalhador
# ---------------------------------------------------------------------------

class CoordinatorClient:
    """Cliente HTTP do coordenador (biblioteca padrão, com novas tentativas).

    Parâmetros: ``url`` (``http://IP:porta``), ``token``, ``retries`` (tentativas com espera
    2, 4, 8... s em falha de rede; erro 4xx do coordenador não é repetido).
    """

    def __init__(self, url: str, token: str, retries: int = 5):
        self.url, self.token, self.retries = url.rstrip("/"), token, int(retries)
        # Contadores do registro da coordenação (lidos e zerados pelo run_worker a cada rodada).
        self.attempts_retried = 0
        self.network_failures: list[str] = []

    def _call(self, method: str, path: str, data: bytes | None = None, headers: dict | None = None,
              raw: bool = False):
        headers = {TOKEN_HEADER: self.token, **(headers or {})}
        for attempt in range(self.retries + 1):
            try:
                request = urlrequest.Request(self.url + path, data=data, method=method, headers=headers)
                with urlrequest.urlopen(request, timeout=120) as response:
                    body = response.read()
                return body if raw else json.loads(body or b"{}")
            except urlerror.HTTPError as error:
                detail = error.read().decode("utf-8", "replace")
                raise RuntimeError(f"Coordenador recusou {path}: {error.code} {detail}") from error
            except (urlerror.URLError, TimeoutError, ConnectionError) as error:
                self.network_failures.append(f"{path}: {type(error).__name__}")
                if attempt >= self.retries:
                    raise
                self.attempts_retried += 1
                time.sleep(2 ** (attempt + 1))
        raise RuntimeError("inalcançável")

    def register(self, payload: dict) -> dict:
        return self._call("POST", "/register", json.dumps(payload).encode(), {"Content-Type": "application/json"})

    def task(self, model: str) -> dict:
        return self._call("GET", f"/task/{model}")

    def peer(self, round_index: int, model: str) -> bytes:
        return self._call("GET", f"/peer/{round_index}/{model}", raw=True)

    def upload(self, round_index: int, model: str, cache: bytes, metrics: dict) -> dict:
        return self._call("POST", f"/upload/{round_index}/{model}", cache,
                          {"Content-Type": "application/octet-stream", "X-DroneCamp-Metrics": json.dumps(metrics)})


def run_worker(config_root: Path, data_path: Path, settings: dict, models: list[str], coordinator: str, token: str,
               host_label: str | None = None, max_batches: int | None = None, val_max_batches: int | None = None,
               poll_seconds: float = POLL_SECONDS, workdir: Path | None = None) -> Path:
    """ENTRADA de ``dml-lan-worker``: treina os ``models`` desta máquina, rodada a rodada.

    Parâmetros:
        config_root: pasta ``sistema_ia``.
        data_path: ``dataset.yaml`` (cópia idêntica à das outras máquinas).
        settings: ``load_dml_config`` (o mesmo ``configs/dml.json`` das outras máquinas).
        models: nomes dos modelos que esta máquina treina (um ou mais; mais de um
            treina em sequência dentro da rodada).
        coordinator: ``http://IP:porta`` do coordenador; ``token``: o segredo dele.
        host_label: nome desta máquina no ``state.json`` (ex.: ``HP-Z230``).
        max_batches, val_max_batches: limites para o teste de fumaça.
        workdir: pasta de trabalho. ``None`` cria ``runs/dml_lan_worker_*`` nova; para
            retomar depois de reiniciar a máquina, passe a MESMA pasta (``--workdir``):
            a rodada seguinte precisa dos pesos ``last`` da anterior, que ficam nela.

    Retorna a pasta de trabalho. Etapas por rodada e modelo: 1. pede a tarefa;
    2. baixa os caches dos pares; 3. ``train_mutual`` (``solo`` na rodada 0,
    ``cached`` depois) continuando dos pesos ``last`` da rodada anterior;
    4. grava e envia o próprio cache + mAP50 de validação.
    """
    import platform

    from .mutual_learning import TileDataset, build_peer_cache, train_mutual
    from .detector_adapters import build_adapter

    client = CoordinatorClient(coordinator, token)
    imgsz = int(settings["imgsz"])
    info = client.register({"models": models, "fingerprint": dataset_fingerprint(data_path, imgsz),
                            "host": host_label or platform.node(), "device": settings["train"].get("device", "cpu")})
    workdir = Path(workdir) if workdir else make_run_directory(Path(config_root), "dml_lan_worker")
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"Trabalhador DML: pasta {workdir} (use --workdir com ela para retomar)")
    specs = {model["name"]: dict(model) for model in settings["models"]}
    total_epochs = int(info["rounds"]) * int(info["epochs_per_round"])
    pending = list(models)
    # Espera acumulada desde o último trabalho de cada modelo (registro da coordenação).
    waiting = {name: 0.0 for name in models}
    log_path = workdir / "worker_log.jsonl"
    while pending:
        progressed = False
        for name in list(pending):
            task = client.task(name)
            if task["action"] == "done":
                pending.remove(name)
                continue
            if task["action"] == "wait":
                continue
            round_index = int(task["round"])
            assigned_at = time.time()
            client.attempts_retried, client.network_failures = 0, []
            transfer = {"downloaded_bytes": 0, "download_seconds": 0.0}
            # 1-2. Pesos da rodada anterior e caches dos pares desta rodada.
            spec = dict(specs[name])
            spec["start_epoch"] = round_index * int(info["epochs_per_round"])
            previous = workdir / name / f"round_{round_index - 1}" / "weights" / name / "last"
            if round_index > 0:
                candidates = [previous.with_suffix(".pt"), previous.with_suffix(".pth"), previous]
                found = next((path for path in candidates if path.exists()), None)
                if found is None:
                    raise RuntimeError(f"Pesos da rodada {round_index - 1} de {name} não estão em {previous.parent}.")
                spec["weights"] = str(found)
            peer_dir = workdir / "peers" / str(round_index - 1)
            for peer in task.get("peers", []):
                target = peer_dir / f"{peer}.npz"
                if not target.exists():
                    peer_dir.mkdir(parents=True, exist_ok=True)
                    started = time.time()
                    body = client.peer(round_index - 1, peer)
                    target.write_bytes(body)
                    transfer["downloaded_bytes"] += len(body)
                    transfer["download_seconds"] += time.time() - started
            # 3. Treino da rodada: λ e multiplicador já vêm do coordenador (RL lá, não aqui).
            local = json.loads(json.dumps(settings))
            local["models"] = [spec if model["name"] == name else model for model in local["models"]]
            local["train"].update({"epochs": int(task["epochs"]), "schedule_epochs": total_epochs})
            local["consistency"].update({"lambda": float(task.get("lambda", 0.0)), "ramp_epochs": 0})
            local["rl"]["enabled"] = False
            run = workdir / name / f"round_{round_index}"
            run.mkdir(parents=True, exist_ok=True)
            mode = "cached" if task.get("peers") and float(task.get("lambda", 0.0)) > 0 else "solo"
            train_started = time.time()
            train_mutual(Path(config_root), data_path, local, mode=mode, student=name,
                         peer_cache=peer_dir if mode == "cached" else None, max_batches=max_batches,
                         val_max_batches=val_max_batches, run=run)
            train_seconds = time.time() - train_started
            summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))["models"][name]
            # 4. Cache das previsões do modelo treinado (pesos last) no split de treino.
            last = run / "weights" / name / "last"
            last = next((path for path in (last.with_suffix(".pt"), last.with_suffix(".pth"), last) if path.exists()))
            adapter = build_adapter({**spec, "weights": str(last)}, TileDataset(data_path, "val", imgsz).names,
                                    imgsz).to(local["train"].get("device") or "cpu")
            cache_dir = run / "cache"
            cache_path = build_peer_cache(adapter, TileDataset(data_path, "train", imgsz), cache_dir,
                                          batch=int(local["train"]["batch"]), max_batches=max_batches)
            cache_bytes = cache_path.read_bytes()
            worker_record = {"host": host_label or platform.node(), "round": round_index, "model": name,
                             "mode": mode, "wait_seconds_before": round(waiting[name], 1),
                             "train_seconds": round(train_seconds, 1), **{k: round(v, 2) if isinstance(v, float) else v
                                                                         for k, v in transfer.items()},
                             "uploaded_bytes": len(cache_bytes), "retries": client.attempts_retried,
                             "network_failures": client.network_failures[-5:]}
            weights_sha = _weights_hash(last)
            started = time.time()
            client.upload(round_index, name, cache_bytes,
                          {"val_map50": summary["last_val_map50"], "best_val_map50": summary["best_val_map50"],
                           "mode": mode, "lambda": task.get("lambda"), "weights_sha256": weights_sha,
                           "worker": worker_record})
            worker_record.update({"upload_seconds": round(time.time() - started, 2),
                                  "total_seconds": round(time.time() - assigned_at, 1), "weights_sha256": weights_sha})
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(worker_record, ensure_ascii=False) + "\n")
            waiting[name] = 0.0
            progressed = True
        if pending and not progressed:
            time.sleep(poll_seconds)
            for name in pending:
                waiting[name] += poll_seconds
    return workdir
