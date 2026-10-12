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
- ``POST /api/feedback``: grava o arquivo de revisão exportado na pasta da revisão;
- ``GET/POST /api/classes``, ``PUT/DELETE /api/classes/<id>``: cadastro das classes
  (``class_catalog``): nome de exibição, cor, ativo/inativo e classes propostas,
  gravados em ``configs/taxonomy_overrides.json`` sem tocar em ``taxonomy.json``.

O que não faz: aprovar dados, adotar modelos, apagar inspeções ou classes,
reescrever a taxonomia ou inferir gravidade. Toda requisição que altera algo exige o token sorteado na partida
(impede que outro site aberto no navegador dispare treinos ou envie arquivos).

Quando mexer: para liberar um comando novo na página, acrescente-o em
``_job_arguments`` com validação de cada opção; para servir um tipo novo de
arquivo, ajuste ``REVIEW_SUFFIXES``/``DOWNLOAD_SUFFIXES``.

Conteúdo do arquivo:
- Constantes de limites, tipos de arquivo e cabeçalhos (``MAX_*``, ``*_SUFFIXES``, ``CONTENT_TYPES``);
- ``JobRunner``: roda UMA tarefa do CLI por vez em processo filho (treino, sugestões);
- validação das opções vindas da página: ``_inside``, ``_project_file``, ``_integer``,
  ``_job_arguments``, ``_text``, ``_report_details``, ``_decisions_map``;
- ``capabilities``: o que este PC consegue gerar (PDF, quadros de vídeo);
- ``Workspace``: quais inspeções estão abertas e o resumo de cada uma;
- ``UploadManager``: lotes de envio de fotos/vídeos e a importação em segundo plano;
- ``_reviews_meta``: lista de revisões embutida na página (seletor);
- ``make_handler``: monta a classe ``Handler`` com TODAS as rotas HTTP
  (``do_GET``, ``do_POST``, ``do_PUT``, ``do_DELETE``);
- ações chamadas pelas rotas: ``_export_evidence``, ``_build_report``, ``_save_feedback``;
- ``_require_current_taxonomy``: recusa revisões feitas com taxonomia antiga;
- ``serve_platform``: ENTRADA do comando ``platform`` (cli.py).

Quem chama: ``cli.py`` (comando ``python -m dronecamp_ia platform``, via
``serve_platform``) e os testes ``tests/test_platform.py`` e ``tests/test_class_catalog.py``
(que usam ``make_handler``/``JobRunner`` direto). A página (``review_templates/platform.js``)
fala com as rotas por ``api(...)``, ``classRequest(...)`` e ``putFile(...)``.

Disco:
- LÊ: ``data/reviews/*/registry.json``, ``browser_data.json``, ``suggestions.json``,
  ``feedback_inbox/*.json``, ``configs/taxonomy.json``, fotos da revisão, ``runs/...``;
- GRAVA: ``.runtime/jobs/<data>_<comando>.log`` (log das tarefas),
  ``.runtime/uploads/<lote>/`` (envio temporário; apagado depois da importação),
  ``data/reviews/<nova inspeção>/`` (via ``ingest.ingest_media``),
  ``<revisão>/feedback_inbox/<data>_<revisor>.json``, ``<revisão>/exportacoes/*.zip``,
  ``<revisão>/laudos/<pasta>/`` e ``configs/taxonomy_overrides.json`` (via ``class_catalog``).

Onde mexer (casos concretos):
- Rota nova de LEITURA (GET): ``do_GET``, num ``if full == "/api/..."`` antes do trecho
  que serve arquivos; responda com ``self._json(...)``. Na página, ``api('/api/...')``.
- Rota nova que ALTERA algo (POST): ``do_POST``, num ``if path == "/api/..."``; o token
  já foi conferido. Na página, ``api('/api/...', corpo)``.
- Comando novo do CLI que a página pode iniciar: ramo novo em ``_job_arguments``
  (e o botão/``startJob`` em platform.js).
- Tipo novo de arquivo servido: ``REVIEW_SUFFIXES`` (fotos/dados da revisão) ou
  ``DOWNLOAD_SUFFIXES``/``DOWNLOAD_FOLDERS`` (gerados para baixar); tipo MIME em ``CONTENT_TYPES``.
- Campo novo no laudo: ``laudo_abnt.DETAIL_FIELDS`` (aceito por ``_report_details``)
  ou argumento novo em ``_build_report``.
- Opção nova na importação de mídia: ``UploadManager.start_ingest`` (dict ``parameters``)
  e o argumento correspondente em ``ingest.ingest_media``.
- Informação nova na lista de inspeções: ``Workspace._summary``.
- Limites de tamanho: ``MAX_BODY_BYTES``, ``MAX_UPLOAD_FILE_BYTES``, ``MAX_UPLOAD_BATCH_BYTES``.
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

from . import class_catalog
from .config import ProjectConfig, detection_names, load_taxonomy
from .io import file_hash
from .operations import collect_operations, collect_review_files
from .naming import review_name
from .review_render import build_page, empty_page_payload, load_page_payload

# ---------------------------------------------------------------------------
# Constantes: limites de tamanho, formato dos lotes, tipos de arquivo servidos
# ---------------------------------------------------------------------------

# Tamanho máximo (bytes) de um corpo JSON em POST/PUT de classes (16 MiB). O arquivo de
# revisão exportado (POST /api/feedback) também passa por aqui; aumente se ele crescer além disso.
MAX_BODY_BYTES = 16 * 1024 * 1024
# Envio de mídia: um arquivo por requisição, direto para o disco (vídeos de drone passam de 1 GB).
# Limite por arquivo: 8 GiB.
MAX_UPLOAD_FILE_BYTES = 8 * 1024 ** 3
# Limite da soma dos arquivos de um lote: 64 GiB (acima disso, a página pede para importar em partes).
MAX_UPLOAD_BATCH_BYTES = 64 * 1024 ** 3
# Quantas linhas finais do log da tarefa vão para a página (painel de log do Monitoramento).
LOG_TAIL_LINES = 80
# Formato do identificador de lote (gerado por secrets.token_urlsafe(12) = 16 caracteres):
# só letras, números, "_" e "-", de 8 a 64 caracteres. Impede "../" e outros caminhos na URL.
UPLOAD_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
# Arquivos da revisão que a página pede (fotos, evidências, dados) e os gerados para baixar.
# EXTENSAO: tipo novo de arquivo que a página exibe (ex.: ".tif") = extensão aqui (e, se o
# navegador não souber o tipo, em CONTENT_TYPES).
# ATENCAO: não acrescente ".py", ".yaml" ou similares: tudo da pasta da revisão com essa
# extensão passaria a ser entregue a quem abrir a página.
REVIEW_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".json", ".csv"}
# Pastas da revisão cujos arquivos podem ser baixados (laudos e pacotes de evidências).
# EXTENSAO: pasta nova de arquivos gerados para baixar = nome aqui (e listagem em
# operations.collect_review_files).
DOWNLOAD_FOLDERS = ("laudos", "exportacoes")
# Extensões aceitas dentro de DOWNLOAD_FOLDERS. ".zip" e ".pdf" vão como anexo (download);
# as demais abrem no navegador (ver do_GET).
DOWNLOAD_SUFFIXES = {".zip", ".pdf", ".html", ".json", ".csv", ".txt", ".md"}
# Tipo MIME (cabeçalho Content-Type) por extensão; o que não está aqui usa mimetypes.guess_type.
# Os textos levam "charset=utf-8" para acentos aparecerem certos.
CONTENT_TYPES = {".webp": "image/webp", ".zip": "application/zip", ".pdf": "application/pdf", ".md": "text/markdown; charset=utf-8",
                 ".txt": "text/plain; charset=utf-8", ".html": "text/html; charset=utf-8", ".json": "application/json; charset=utf-8",
                 ".csv": "text/csv; charset=utf-8"}


# ---------------------------------------------------------------------------
# Tarefas do CLI iniciadas pela página (treino piloto, sugestões)
# ---------------------------------------------------------------------------

# ATENCAO: uma tarefa por vez. O log de cada tarefa fica em .runtime/jobs/<data>_<comando>.log;
# é o primeiro lugar para olhar quando um treino ou sugestão iniciado pela página falha.
class JobRunner:
    """Uma tarefa por vez; o processo filho é o mesmo CLI usado no terminal.

    Atributos:
        root: pasta ``sistema_ia`` (diretório de trabalho do filho e base do log).
        lock: trava que impede duas partidas/paradas simultâneas.
        process: o processo filho atual (ou o último), ou None antes da primeira tarefa.
        job: descrição da tarefa atual/última (``kind``, ``arguments``, ``started_at``,
            ``log``, ``pid``), ou None.

    Usada por: rotas ``/api/jobs`` e ``/api/jobs/stop`` e por ``UploadManager._run``
    (sugestões logo depois de importar mídia).
    """

    def __init__(self, root: Path):
        """Cria o executor sem nenhuma tarefa.

        Parâmetros:
            root: pasta ``sistema_ia`` (``config.root``).
        """
        self.root = root
        self.lock = threading.Lock()
        self.process: subprocess.Popen | None = None
        self.job: dict | None = None

    def start(self, kind: str, arguments: list[str]) -> dict:
        """Inicia ``python -m dronecamp_ia <kind> <arguments>`` em segundo plano, com log em arquivo.

        Parâmetros:
            kind: comando do CLI (ex.: "train-pilot", "suggest"); já validado por ``_job_arguments``.
            arguments: lista de argumentos já validada (ex.: ``["--data", "data/pilot/x/dataset.yaml"]``).

        Retorna:
            ``status()`` logo após a partida (estado "running" na maioria dos casos).

        Erros:
            RuntimeError "Já existe uma tarefa em andamento..." se a anterior não terminou
            (as rotas transformam em HTTP 409); OSError se não puder criar o log.

        ATENCAO: ``arguments`` NUNCA deve conter texto livre vindo do navegador sem validação;
        a lista vai direto para o processo filho.
        """
        with self.lock:
            # 1. Só uma tarefa por vez: poll() None = o filho ainda está vivo.
            if self.process and self.process.poll() is None:
                raise RuntimeError("Já existe uma tarefa em andamento; aguarde ou interrompa antes de iniciar outra.")
            # 2. Arquivo de log .runtime/jobs/<AAAAMMDDTHHMMSSZ>_<comando>.log.
            logs = self.root / ".runtime" / "jobs"
            logs.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            log_path = logs / f"{stamp}_{kind}.log"
            # 3. Mesmo Python do servidor (sys.executable), mesmo pacote: idêntico ao terminal.
            command = [sys.executable, "-m", "dronecamp_ia", kind, *arguments]
            stream = log_path.open("w", encoding="utf-8")
            # O filho herda o ambiente (o mesmo .venv) e roda na pasta sistema_ia.
            # PYTHONUNBUFFERED: o log aparece linha a linha (sem esperar buffer encher);
            # PYTHONIOENCODING: acentos no log no Windows. stderr vai junto para o mesmo log.
            self.process = subprocess.Popen(command, cwd=self.root, stdout=stream, stderr=subprocess.STDOUT,
                                            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
            # O filho tem a própria cópia do descritor; o servidor pode fechar a dele.
            stream.close()
            # 4. Guarda a descrição da tarefa (mostrada no painel de tarefa da página).
            self.job = {"kind": kind, "arguments": arguments, "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "log": log_path.relative_to(self.root).as_posix(), "pid": self.process.pid}
            return self.status()

    def stop(self) -> dict:
        """Pede o fim da tarefa em andamento (``terminate``: SIGTERM no Linux, TerminateProcess no Windows).

        Parâmetros: nenhum.

        Retorna: ``status()`` (o estado pode ainda aparecer "running" por um instante,
        até o processo de fato terminar).
        """
        with self.lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()
            return self.status()

    def status(self) -> dict:
        """Estado da tarefa atual/última, com o final do log.

        Parâmetros: nenhum.

        Retorna:
            ``{"state": "idle"}`` se nenhuma tarefa foi iniciada; senão o dict de ``self.job``
            mais ``state`` ("running", "finished" se saiu com código 0, "failed" se outro código),
            ``exit_code`` (None enquanto roda) e ``log_tail`` (últimas ``LOG_TAIL_LINES`` linhas).
        """
        if not self.job:
            return {"state": "idle"}
        # poll(): None enquanto roda; o código de saída quando terminou.
        code = self.process.poll() if self.process else None
        state = "running" if code is None else ("finished" if code == 0 else "failed")
        log = self.root / self.job["log"]
        # Lê o log inteiro e guarda só as últimas linhas (errors="replace": byte inválido não quebra).
        try:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-LOG_TAIL_LINES:]
        except OSError:
            tail = []
        return {**self.job, "state": state, "exit_code": code, "log_tail": tail}


# ---------------------------------------------------------------------------
# Validação das opções enviadas pela página (caminhos, números, textos)
# ---------------------------------------------------------------------------

def _inside(path: Path, base: Path) -> bool:
    """Diz se ``path`` fica dentro de ``base`` depois de resolver ".." e links simbólicos.

    Parâmetros:
        path: caminho a conferir.
        base: pasta que deve contê-lo.

    Retorna: True se ``path`` está em ``base`` (ou é a própria ``base``); False caso contrário.

    ATENCAO: é a proteção contra "../" em URLs e opções; toda leitura de arquivo pedida
    pelo navegador precisa passar por aqui.
    """
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _project_file(root: Path, value: str, folders: tuple[str, ...], suffix: str) -> str:
    """Caminho relativo existente dentro das pastas permitidas (nada fora de sistema_ia).

    Parâmetros:
        root: pasta ``sistema_ia``.
        value: caminho enviado pela página, relativo a ``root`` (ex.: "runs/<id>/fit/weights/best.pt");
            "\\" é aceito e trocado por "/".
        folders: pastas permitidas, relativas a ``root`` (ex.: ``("models", "runs")``).
        suffix: extensão exigida, em minúsculas (ex.: ".pt", ".yaml").

    Retorna:
        O caminho relativo a ``root`` com "/" (ex.: "models/yolo26l.pt").

    Erros:
        ValueError "Caminho ausente." se vazio; "Arquivo não permitido: <valor>" se a
        extensão não bate, o arquivo não existe ou está fora das pastas permitidas.
    """
    if not isinstance(value, str) or not value:
        raise ValueError("Caminho ausente.")
    path = (root / value.replace("\\", "/")).resolve()
    # Três condições: extensão certa, arquivo existe e está dentro de alguma pasta permitida.
    if path.suffix.lower() != suffix or not path.is_file() or not any(_inside(path, root / folder) for folder in folders):
        raise ValueError(f"Arquivo não permitido: {value}")
    return path.relative_to(root.resolve()).as_posix()


def _integer(options: dict, key: str, low: int, high: int) -> list[str]:
    """Opção inteira da página convertida em argumento de linha de comando.

    Parâmetros:
        options: opções enviadas pela página (dict).
        key: nome da opção, igual ao nome do argumento do CLI (ex.: "epochs" → ``--epochs``).
        low, high: limites aceitos (inclusive).

    Retorna:
        ``["--<key>", "<valor>"]`` ou ``[]`` se a opção não veio (o CLI usa o padrão).

    Erros:
        ValueError "<key> deve ser inteiro entre <low> e <high>." para valor fora da faixa,
        texto, float ou booleano (``type(...) is not int`` recusa True/False).
    """
    value = options.get(key)
    if value in (None, ""):
        return []
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{key} deve ser inteiro entre {low} e {high}.")
    return [f"--{key}", str(value)]


# EXTENSAO: para a página iniciar outro comando do CLI, acrescente um ramo ``if kind == ...``
# validando CADA opção (nunca repasse texto livre do navegador para a linha de comando).
def _job_arguments(root: Path, registry_path: Path | None, kind: str, options: dict) -> list[str]:
    """Lista fechada de comandos e opções que a página pode iniciar; só ``suggest`` precisa de ``registry_path``.

    Parâmetros:
        root: pasta ``sistema_ia``.
        registry_path: ``registry.json`` da revisão (usado só por "suggest"); None para "train-pilot".
        kind: comando pedido pela página ("train-pilot" ou "suggest").
        options: opções do formulário. "train-pilot": ``data`` (dataset.yaml em data/pilot),
            ``weights`` (.pt em models/ ou runs/, opcional), ``epochs`` (1–1000), ``imgsz``
            (64–2048, múltiplo de 32), ``batch`` (1–64), ``tile`` e ``gpu_eval`` (True liga).
            "suggest": ``weights`` (obrigatório) e ``conf`` (0 < conf ≤ 1, opcional).

    Retorna:
        Lista de argumentos para ``JobRunner.start`` (sem o nome do comando).

    Erros:
        ValueError com a mensagem de cada validação, ou "Tarefa não permitida pela página: <kind>".

    Chamada por: ``do_POST`` (``/api/jobs``) e ``UploadManager._run`` (sugestões após importar).
    """
    # Treino piloto: seção Treinamento da página (botão "startTrain").
    if kind == "train-pilot":
        arguments = ["--data", _project_file(root, options.get("data"), ("data/pilot",), ".yaml")]
        if options.get("weights"):
            arguments += ["--weights", _project_file(root, options["weights"], ("models", "runs"), ".pt")]
        # Opções inteiras com faixa (nome do CLI, mínimo, máximo).
        # EXTENSAO: opção inteira nova do train-pilot = tupla nova aqui (o nome tem de existir no CLI).
        for key, low, high in (("epochs", 1, 1000), ("imgsz", 64, 2048), ("batch", 1, 64)):
            arguments += _integer(options, key, low, high)
        # imgsz precisa ser múltiplo de 32 (exigência do YOLO); o valor está logo após "--imgsz".
        if "--imgsz" in arguments and int(arguments[arguments.index("--imgsz") + 1]) % 32:
            raise ValueError("imgsz deve ser múltiplo de 32.")
        # Opções liga/desliga: só True explícito liga (qualquer outro valor é ignorado).
        # EXTENSAO: opção booleana nova = um if igual a estes.
        if options.get("tile") is True:
            arguments.append("--tile")
        if options.get("gpu_eval") is True:
            arguments.append("--gpu-eval")
        return arguments
    # Sugestões do modelo para a revisão aberta: seção Monitoramento (botão "startSuggest")
    # e importação de mídia com pesos escolhidos.
    if kind == "suggest":
        arguments = ["--weights", _project_file(root, options.get("weights"), ("models", "runs"), ".pt"),
                     "--registry", registry_path.resolve().relative_to(root.resolve()).as_posix()]
        conf = options.get("conf")
        if conf not in (None, ""):
            # ATENCAO: bool é subclasse de int em Python, então conf=True passa nesta checagem
            # (True vale 1) e vira o texto "True" na linha de comando.
            if not isinstance(conf, (int, float)) or not 0 < conf <= 1:
                raise ValueError("conf deve estar entre 0 e 1.")
            arguments += ["--conf", str(conf)]
        return arguments
    # EXTENSAO: novo comando permitido = novo "if kind == ..." ACIMA desta linha.
    raise ValueError(f"Tarefa não permitida pela página: {kind}")


def _text(options: dict, key: str, limit: int, default: str = "") -> str:
    """Campo de texto opcional de um corpo JSON, com limite de tamanho.

    Parâmetros:
        options: corpo/opções enviados pela página.
        key: nome do campo (ex.: "reviewer", "notes", "author").
        limit: tamanho máximo em caracteres.
        default: valor devolvido quando o campo falta ou está vazio.

    Retorna: o texto como veio (sem normalizar espaços) ou ``default``.

    Erros: ValueError "<key> deve ser texto com até <limit> caracteres." se não for texto
    ou for longo demais.
    """
    value = options.get(key)
    if value in (None, ""):
        return default
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{key} deve ser texto com até {limit} caracteres.")
    return value


def _report_details(value) -> dict | None:
    """Dados técnicos opcionais do laudo (data da inspeção, cobertura, responsável, ART...): só texto curto.

    Parâmetros:
        value: objeto ``details`` do corpo de ``POST /api/report`` (dict de campo → texto), ou None/{}.

    Retorna:
        dict só com os campos de ``laudo_abnt.DETAIL_FIELDS`` presentes (cada um até 500
        caracteres), ou None se nada veio.

    Erros:
        ValueError "details deve ser um objeto." ou "Campo desconhecido nos dados do laudo: <campo>.".

    EXTENSAO: campo técnico novo no laudo = nome em ``laudo_abnt.DETAIL_FIELDS`` (aceito aqui
    automaticamente) + campo no formulário da seção Laudos (index.html/platform.js ``reportDetails``).
    """
    from .laudo_abnt import DETAIL_FIELDS
    if value in (None, {}):
        return None
    if not isinstance(value, dict):
        raise ValueError("details deve ser um objeto.")
    unknown = set(value) - set(DETAIL_FIELDS)
    if unknown:  # Campo com nome errado não pode sumir em silêncio do laudo (ex.: ART).
        raise ValueError(f"Campo desconhecido nos dados do laudo: {sorted(unknown)[0]}.")
    return {key: _text(value, key, 500) for key in DETAIL_FIELDS if key in value}


def _decisions_map(value) -> dict | None:
    """Decisões como a página exporta (lista com ``image_sha256``) viram o mapa que ``exports`` lê.

    Parâmetros:
        value: lista ``[{"image_sha256": "...", "status": ..., "boxes": ...}, ...]`` (formato da
            página/arquivo de revisão), um dict já no formato de mapa, ou vazio/None.

    Retorna:
        ``{"<image_sha256>": {demais campos da decisão}}``, o próprio dict se já veio como mapa,
        ou None se não há decisões (``exports`` usa então o que está no registro).

    Erros:
        ValueError "decisions deve ser a lista de decisões exportada pela página." se algum
        item não for dict ou não tiver ``image_sha256``.

    Chamada por: ``_export_evidence``, ``_build_report`` e ``Workspace._summary``.
    """
    if value in (None, [], {}):
        return None
    if isinstance(value, dict):
        return value
    if not isinstance(value, list) or any(not isinstance(item, dict) or not item.get("image_sha256") for item in value):
        raise ValueError("decisions deve ser a lista de decisões exportada pela página.")
    decisions = {}
    # A chave do mapa é o hash da foto; o resto da decisão vai como valor (sem repetir o hash).
    for item in value:
        decisions[item["image_sha256"]] = {key: entry for key, entry in item.items() if key != "image_sha256"}
    return decisions


def capabilities() -> dict:
    """O que este PC consegue gerar; a página desliga o que falta em vez de falhar depois.

    Parâmetros: nenhum.

    Retorna:
        ``{"pdf": bool, "video_frames": "opencv" | "ffmpeg" | None}``: PDF exige o pacote
        ``reportlab``; quadros de vídeo usam OpenCV (``cv2``) ou, sem ele, o programa ``ffmpeg``.

    Usada em ``/api/workspace`` (seções Importar mídia e Laudos e downloads).

    EXTENSAO: recurso opcional novo (ex.: outro gerador) = um ``try: import ...`` aqui e uma
    chave nova no dict; a página lê em ``WORKSPACE.capabilities``.
    """
    # Testa se o reportlab está instalado (só para saber; não usa aqui).
    try:
        import reportlab  # noqa: F401
        pdf = True
    except ImportError:
        pdf = False
    # Extração de quadros: OpenCV primeiro; senão, ffmpeg no PATH; senão, nenhum.
    try:
        import cv2  # noqa: F401
        frames = "opencv"
    except ImportError:
        frames = "ffmpeg" if shutil.which("ffmpeg") else None
    return {"pdf": pdf, "video_frames": frames}


# ---------------------------------------------------------------------------
# Inspeções abertas no servidor (lista fixa ou espaço de trabalho)
# ---------------------------------------------------------------------------

class Workspace:
    """Inspeções abertas: lista fixa (``--registry``) ou tudo em ``data/reviews/`` (modo espaço de trabalho).

    Só lê. No modo espaço de trabalho a lista é relida a cada consulta, então
    uma inspeção importada ou criada pelo terminal aparece sem reiniciar.

    Atributos:
        config: configuração do projeto.
        root: pasta ``sistema_ia`` resolvida.
        reviews_dir: ``sistema_ia/data/reviews``.
        fixed: ``{nome_da_pasta: registry.json}`` no modo lista fixa; None no espaço de trabalho.
        _cache: resumos já calculados por inspeção (``nome → (chave_de_validade, resumo)``).
        _lock: trava do cache e da lista fixa (o servidor atende várias requisições ao mesmo tempo).
    """

    def __init__(self, config: ProjectConfig, registries: list[Path] | None):
        """Prepara o espaço.

        Parâmetros:
            config: configuração do projeto.
            registries: lista de ``registry.json`` (modo lista fixa) ou None (modo espaço de trabalho).
        """
        self.config = config
        self.root = config.root.resolve()
        self.reviews_dir = self.root / "data" / "reviews"
        # Nome de cada inspeção = nome da pasta que contém o registry.json.
        self.fixed = {path.parent.name: path.resolve() for path in registries} if registries else None
        self._cache: dict[str, tuple[tuple, dict]] = {}
        self._lock = threading.Lock()

    def reviews(self) -> dict[str, Path]:
        """Inspeções abertas agora.

        Parâmetros: nenhum.

        Retorna:
            ``{nome_da_pasta: caminho do registry.json}``. Lista fixa: cópia de ``fixed``.
            Espaço de trabalho: toda pasta de ``data/reviews/`` com ``registry.json`` (ordem alfabética).
        """
        if self.fixed is not None:
            with self._lock:
                return dict(self.fixed)
        if not self.reviews_dir.is_dir():
            return {}
        return {path.parent.name: path for path in sorted(self.reviews_dir.glob("*/registry.json")) if path.is_file()}

    @staticmethod
    def page_ready(registry_path: Path) -> bool:
        """Diz se a página da inspeção já foi gerada e confere com o registro atual.

        Parâmetros:
            registry_path: caminho do ``registry.json``.

        Retorna:
            True se ``browser_data.json`` existe e pertence a este registro (``render-review``
            já rodou); False se falta ou está desatualizado.
        """
        try:
            load_page_payload(registry_path)
            return True
        except (OSError, ValueError):
            return False

    def register(self, name: str, path: Path) -> None:
        """Inspeção recém-importada: entra na lista fixa (no espaço de trabalho ela já é descoberta).

        Parâmetros:
            name: nome da pasta da inspeção.
            path: caminho do ``registry.json`` dela.

        Retorna: None. Também descarta o resumo em cache com esse nome.
        """
        with self._lock:
            self._cache.pop(name, None)
            if self.fixed is not None:
                self.fixed[name] = path.resolve()

    def default(self) -> str | None:
        """Primeira da lista fixa; no espaço de trabalho, a inspeção com página mais recente.

        Parâmetros: nenhum.

        Retorna:
            Nome da inspeção aberta em ``/`` ou None (espaço vazio ou nenhuma página gerada).
            "Mais recente" = maior data de modificação do ``registry.json`` entre as que têm página.
        """
        if self.fixed is not None:
            return next(iter(self.fixed))
        # Tuplas (data, nome): max() escolhe a data maior.
        ready = [(path.stat().st_mtime, name) for name, path in self.reviews().items() if self.page_ready(path)]
        return max(ready)[1] if ready else None

    def _summary(self, name: str, path: Path) -> dict:
        """Contagens de uma inspeção, em cache enquanto registro, sugestões e caixa de entrada não mudam.

        Parâmetros:
            name: nome da pasta da inspeção.
            path: caminho do ``registry.json``.

        Retorna:
            dict com ``name``, ``version``, ``href`` ("/r/<nome>/"), ``images`` (quantidade de
            fotos), ``display_name``, ``decided`` (fotos já decididas), ``suggestions_pending``,
            ``buildings``, ``updated_at``, ``page_ready`` e ``scope_label``.

        Erros: OSError/ValueError se o ``registry.json`` não puder ser lido (``describe`` trata).

        EXTENSAO: informação nova por inspeção na seção Inspeções = chave nova no dict ``summary``
        (e, se depender de outro arquivo, inclua o carimbo dele na chave do cache ``key``).
        """
        directory = path.parent
        # 1. Arquivo de revisão mais recente na caixa de entrada (nomes começam com a data, então
        # a ordem alfabética é a ordem cronológica).
        inbox = sorted(directory.glob("feedback_inbox/*.json"))
        latest = inbox[-1] if inbox else None
        # 2. Chave do cache: (data em ns, tamanho) de cada arquivo que influencia o resumo.
        # Se nenhum mudou, devolve o resumo guardado sem reler nada.
        stamp = lambda file: (file.stat().st_mtime_ns, file.stat().st_size) if file and file.is_file() else None  # noqa: E731
        key = (stamp(path), stamp(directory / "browser_data.json"), stamp(directory / "suggestions.json"), stamp(latest))
        with self._lock:
            cached = self._cache.get(name)
            if cached and cached[0] == key:
                return dict(cached[1])
        registry = json.loads(path.read_text(encoding="utf-8"))
        images = registry.get("images") or []
        # 3. Fotos decididas: pelo último arquivo de revisão, se for deste registro (mesmo SHA-256).
        decided, decisions = 0, None
        if latest is not None:
            try:
                feedback = json.loads(latest.read_text(encoding="utf-8"))
                if feedback.get("registry_sha256") == file_hash(path):
                    decided = sum(1 for item in feedback.get("images") or [] if item.get("status") not in (None, "pending"))
                    decisions = _decisions_map(feedback.get("images"))
            except (OSError, ValueError):
                decided, decisions = 0, None
        # Sem arquivo de revisão válido: conta as fotos já aprovadas no próprio registro.
        if not decided:
            decided = sum(1 for item in images if item.get("human_approved") is True)
        # 4. Sugestões do modelo ainda pendentes (só se suggestions.json é deste registro).
        pending = 0
        suggestions_path = directory / "suggestions.json"
        if suggestions_path.is_file():
            try:
                suggestions = json.loads(suggestions_path.read_text(encoding="utf-8"))
                if suggestions.get("registry_sha256") == file_hash(path):
                    pending = self._pending_suggestions(registry, decisions, suggestions)
            except (OSError, ValueError):
                pending = 0
        # 5. Última atualização: o mais recente entre registro e arquivos de revisão.
        updated = max([path.stat().st_mtime] + [file.stat().st_mtime for file in inbox])
        summary = {"name": name, "version": registry.get("version") or name, "href": f"/r/{name}/", "images": len(images),
                   "display_name": review_name(name, registry.get("display_name")),
                   "decided": decided, "suggestions_pending": pending,
                   "buildings": sorted({item.get("building_group") for item in images if item.get("building_group")}),
                   "updated_at": datetime.fromtimestamp(updated, timezone.utc).isoformat(timespec="seconds"),
                   "page_ready": self.page_ready(path),
                   "scope_label": "importada pela plataforma" if registry.get("ingest") else None}
        # 6. Guarda no cache e devolve uma cópia (quem chama pode alterar sem estragar o cache).
        with self._lock:
            self._cache[name] = (key, summary)
        return dict(summary)

    def _pending_suggestions(self, registry: dict, decisions: dict | None, suggestions: dict) -> int:
        """Sugestões que o último arquivo de revisão válido ainda não aceitou nem descartou (mesma regra da página e do laudo).

        Parâmetros:
            registry: conteúdo do ``registry.json``.
            decisions: mapa de decisões (``_decisions_map``) ou None.
            suggestions: conteúdo do ``suggestions.json`` (``images``: hash → lista de sugestões).

        Retorna:
            Número de sugestões pendentes. Se decisões/sugestões estiverem fora do formato,
            conta todas as sugestões (como antes de qualquer revisão).

        ATENCAO: a regra vem de ``exports.resolve_decisions`` para bater com a página e o laudo;
        não reimplemente a contagem aqui.
        """
        from .exports import resolve_decisions
        try:
            class_count = len(detection_names(load_taxonomy(self.config.taxonomy_path)))
            return sum(entry["pending_suggestions"] for entry in resolve_decisions(registry, decisions, class_count, suggestions))
        except (OSError, ValueError, KeyError, TypeError):
            # Decisões ou sugestões fora do contrato: conta todas, como antes de qualquer revisão.
            return sum(len(values) for values in (suggestions.get("images") or {}).values() if isinstance(values, list))

    def describe(self, current: str | None) -> dict:
        """Resposta de ``GET /api/workspace``: todas as inspeções com resumo e as capacidades do PC.

        Parâmetros:
            current: nome da inspeção aberta na página (fica marcada e vai para o topo), ou None.

        Retorna:
            ``{"mode": "fixed"|"workspace", "default": nome|None, "reviews_dir": "data/reviews",
            "reviews": [resumos de _summary + "current"], "capabilities": capabilities()}``.
            Inspeção com registro ilegível entra com ``error`` e ``page_ready`` False.
        """
        reviews = []
        for name, path in self.reviews().items():
            try:
                summary = self._summary(name, path)
            except (OSError, ValueError) as error:
                summary = {"name": name, "display_name": review_name(name), "href": f"/r/{name}/", "images": None, "page_ready": False, "error": str(error)}
            reviews.append({**summary, "current": name == current})
        # Ordem: a atual primeiro; depois da atualização mais recente para a mais antiga.
        reviews.sort(key=lambda item: (item.get("current", False), item.get("updated_at") or ""), reverse=True)
        return {"mode": "fixed" if self.fixed is not None else "workspace", "default": self.default(),
                "reviews_dir": self.reviews_dir.relative_to(self.root).as_posix(), "reviews": reviews, "capabilities": capabilities()}


# ---------------------------------------------------------------------------
# Envio de fotos e vídeos pela página (lotes) e importação como nova inspeção
# ---------------------------------------------------------------------------

class UploadManager:
    """Lotes de envio: arquivos ficam em ``.runtime/uploads/<lote>/`` até a importação, que cria a inspeção.

    Fluxo (seção Importar mídia da página):
        1. ``create`` (``POST /api/uploads``) reserva o lote e o nome da inspeção;
        2. ``store`` (``PUT /api/uploads/<lote>/files``) recebe um arquivo por requisição;
        3. ``start_ingest`` (``POST /api/uploads/<lote>/ingest``) dispara ``_run`` numa thread;
        4. ``status`` (``GET /api/uploads/<lote>``) é consultado pela página até terminar.

    Atributos:
        root: pasta ``sistema_ia``; workspace: o ``Workspace`` do servidor;
        staging: ``.runtime/uploads``; batches: ``{lote: estado do lote}``;
        lock: trava de ``batches``/``running``; running: lote em importação (ou None).

    ATENCAO: os lotes vivem só na memória; reiniciar o servidor perde os lotes abertos
    (os arquivos já enviados ficam em ``.runtime/uploads/`` até alguém apagar).
    """

    def __init__(self, root: Path, workspace: Workspace):
        """Prepara o gerenciador sem lotes.

        Parâmetros:
            root: pasta ``sistema_ia``.
            workspace: lista de inspeções do servidor (para conferir nomes e registrar a nova).
        """
        self.root = root.resolve()
        self.workspace = workspace
        self.staging = self.root / ".runtime" / "uploads"
        self.batches: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.running: str | None = None

    def create(self, name) -> dict:
        """Abre um lote de envio para uma inspeção nova.

        Parâmetros:
            name: nome da inspeção como a pessoa digitou (ex.: "Galpão B outubro").

        Retorna: ``{"batch": "<id do lote>", "name": "<slug da pasta>"}``.

        Erros:
            ValueError de ``ingest.slugify_review_name`` (nome vazio/inválido) ou
            "Já existe uma inspeção chamada <slug>; escolha outro nome.".
        """
        from .ingest import slugify_review_name
        # 1. Nome da pasta (minúsculas, sem acento) e checagem de que não existe.
        slug = slugify_review_name(name)
        if (self.workspace.reviews_dir / slug).exists():
            raise ValueError(f"Já existe uma inspeção chamada {slug}; escolha outro nome.")
        # 2. Id aleatório do lote (16 caracteres URL-safe) e pasta temporária.
        batch = secrets.token_urlsafe(12)
        directory = self.staging / batch
        directory.mkdir(parents=True)
        with self.lock:
            # display_name: o nome como foi digitado, para o registro; a pasta usa o slug.
            # Estados do lote: "open" (recebendo) → "running" (importando) → "finished"/"failed".
            self.batches[batch] = {"name": slug, "display_name": str(name).strip()[:80], "directory": directory, "files": [], "bytes": 0, "state": "open",
                                   "message": "Aguardando arquivos.", "result": None, "error": None}
        return {"batch": batch, "name": slug}

    def _batch(self, batch: str, state: str | None = "open") -> dict:
        """Estado de um lote, conferindo o id e (opcionalmente) a fase.

        Parâmetros:
            batch: id do lote vindo da URL.
            state: fase exigida ("open" por padrão); None aceita qualquer fase.

        Retorna: o dict do lote (o MESMO objeto guardado, para ser alterado no lugar).

        Erros:
            ValueError "Lote de envio desconhecido." (id inválido ou inexistente) ou
            "Este lote já foi enviado para importação." (fase diferente da exigida).
        """
        if not isinstance(batch, str) or not UPLOAD_ID.match(batch) or batch not in self.batches:
            raise ValueError("Lote de envio desconhecido.")
        entry = self.batches[batch]
        if state and entry["state"] != state:
            raise ValueError("Este lote já foi enviado para importação.")
        return entry

    def store(self, batch: str, filename: str, stream, length: int) -> dict:
        """Grava um arquivo do lote direto do corpo da requisição, em blocos.

        Parâmetros:
            batch: id do lote (precisa estar "open").
            filename: nome do arquivo (cabeçalho ``X-File-Name``, codificado para URL pela página);
                só o nome final é usado (pastas são descartadas).
            stream: corpo da requisição (``self.rfile``), lido em blocos de 1 MiB.
            length: tamanho em bytes (cabeçalho ``Content-Length``).

        Retorna:
            ``{"stored": nome gravado, "bytes": tamanho, "kind": "photo"|"video", "files": total no lote}``.

        Erros:
            ValueError para tipo não suportado, arquivo vazio/grande demais, lote acima do
            limite ou envio interrompido (o arquivo parcial é apagado).

        EXTENSAO: tipos de mídia aceitos ficam em ``ingest.media_kind`` (IMAGE_/VIDEO_EXTENSIONS);
        se mudar, atualize a mensagem de erro abaixo.
        """
        from .ingest import media_kind
        entry = self._batch(batch)
        # 1. Nome seguro: decodifica %XX e fica só com o nome final (Path(...).name tira "../").
        name = Path(unquote(filename or "")).name
        kind = media_kind(Path(name))
        if not name or kind is None:
            raise ValueError(f"Tipo de arquivo não suportado: {name or '(sem nome)'}. Envie fotos (JPG, PNG, WEBP, TIFF, BMP) ou vídeos (MP4, MOV, AVI, MKV).")
        # 2. Limites de tamanho por arquivo e por lote.
        if length <= 0 or length > MAX_UPLOAD_FILE_BYTES:
            raise ValueError("Arquivo vazio ou maior que o limite de envio.")
        if entry["bytes"] + length > MAX_UPLOAD_BATCH_BYTES:
            raise ValueError("Lote maior que o limite de envio; importe em partes.")
        # 3. Nome livre na pasta do lote: "foto.jpg", "foto_2.jpg", "foto_3.jpg"...
        target = entry["directory"] / name
        counter = 1
        while target.exists():  # dois arquivos com o mesmo nome (pastas diferentes) ficam ambos
            counter += 1
            target = entry["directory"] / f"{Path(name).stem}_{counter}{Path(name).suffix}"
        # 4. Copia o corpo em blocos de até 1 MiB (vídeo de vários GB não passa inteiro pela memória).
        remaining = length
        with target.open("wb") as output:
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                # Bloco vazio = a conexão fechou antes do fim.
                if not chunk:
                    break
                output.write(chunk)
                remaining -= len(chunk)
        # 5. Faltaram bytes: apaga o arquivo parcial e avisa.
        if remaining:
            target.unlink(missing_ok=True)
            raise ValueError("Envio interrompido antes do fim do arquivo.")
        # 6. Atualiza o estado do lote.
        entry["files"].append(target.name)
        entry["bytes"] += length
        entry["message"] = f"{len(entry['files'])} arquivo(s) recebidos."
        return {"stored": target.name, "bytes": length, "kind": kind, "files": len(entry["files"])}

    def start_ingest(self, batch: str, config: ProjectConfig, options: dict, jobs: JobRunner) -> dict:
        """Valida as opções e começa a importação do lote numa thread (a resposta volta na hora).

        Parâmetros:
            batch: id do lote (precisa estar "open" e ter pelo menos um arquivo).
            config: configuração do projeto (repassada a ``ingest_media``).
            options: corpo enviado pela página: ``name`` (nome da inspeção), ``group``
                (edificação/campanha, obrigatório), ``every_seconds`` (intervalo entre quadros
                de vídeo, padrão 2.0), ``max_frames`` (máximo de quadros por vídeo, padrão 200),
                ``notes``, ``uploaded_by`` e ``weights`` (.pt para gerar sugestões depois; opcional).
            jobs: executor de tarefas (para iniciar ``suggest`` ao fim, se houver ``weights``).

        Retorna: ``status(batch)`` já com estado "running".

        Erros:
            ValueError (lote vazio, grupo ausente, textos longos, pesos não permitidos);
            RuntimeError "Outra importação está em andamento..." (vira HTTP 409).

        ATENCAO: ``every_seconds`` e ``max_frames`` são validados por ``ingest_media``
        (``_validate_sampling``), não aqui; o erro aparece no estado "failed" do lote.
        """
        from .ingest import slugify_review_name
        entry = self._batch(batch)
        if not entry["files"]:
            raise ValueError("Nenhum arquivo recebido neste lote.")
        # 1. Nome final da pasta (a página pode ter mudado o nome desde create).
        name = slugify_review_name(options.get("name") or entry["name"])
        group = _text(options, "group", 100)
        if not group.strip():
            raise ValueError("Informe a edificação ou campanha.")
        every = options.get("every_seconds", 2.0)
        frames = options.get("max_frames", 200)
        # JSON pode mandar 200.0; converte para int quando não tem parte decimal.
        if isinstance(frames, float) and frames.is_integer():
            frames = int(frames)
        # 2. Argumentos para ingest.ingest_media (building_group sai do dict em _run).
        # EXTENSAO: opção nova da importação = chave nova aqui (mesmo nome do argumento de
        # ingest_media) + campo no formulário da seção Importar mídia.
        parameters = {"building_group": group.strip(), "every_seconds": every, "max_frames": frames,
                      # Nome como a pessoa digitou (com acentos); a pasta usa a forma normalizada.
                      "display_name": _text(options, "name", 80).strip() or entry.get("display_name") or None,
                      "uploaded_by": _text(options, "uploaded_by", 100) or None, "campaign_notes": _text(options, "notes", 5000)}
        weights = options.get("weights") or None
        if weights:
            _project_file(self.root, weights, ("models", "runs"), ".pt")  # valida antes de importar
        # 3. Uma importação por vez em todo o servidor.
        with self.lock:
            if self.running and self.batches[self.running]["state"] == "running":
                raise RuntimeError("Outra importação está em andamento; aguarde terminar.")
            self.running = batch
            entry.update(state="running", message="Congelando fotos por hash e extraindo quadros dos vídeos…", name=name)
        # 4. Thread daemon: não impede o servidor de fechar com Ctrl+C.
        worker = threading.Thread(target=self._run, args=(entry, config, parameters, weights, jobs), daemon=True)
        worker.start()
        return self.status(batch)

    def _run(self, entry: dict, config: ProjectConfig, parameters: dict, weights: str | None, jobs: JobRunner) -> None:
        """Importação em segundo plano: cria a inspeção e, se pedido, inicia as sugestões.

        Parâmetros:
            entry: estado do lote (alterado no lugar: ``state``, ``result``, ``message``, ``error``).
            config: configuração do projeto.
            parameters: argumentos de ``ingest_media`` montados em ``start_ingest``.
            weights: pesos para ``suggest`` (caminho relativo validado) ou None.
            jobs: executor de tarefas.

        Retorna: None. O resultado fica em ``entry["result"]`` (caminhos como texto, ``href``
        da nova inspeção e ``suggest_job``); um erro fica em ``entry["error"]``.

        ATENCAO: a pasta temporária do lote é apagada SEMPRE no fim (sucesso ou falha);
        com ``keep_sources=True`` os originais já foram copiados para ``<inspeção>/midia_enviada/``.
        """
        from .ingest import ingest_media
        output = self.workspace.reviews_dir / entry["name"]
        try:
            # 1. Importa: congela fotos por hash, extrai quadros e grava registry.json.
            inputs = [entry["directory"] / name for name in entry["files"]]
            # keep_sources: os arquivos enviados passam a morar em <inspeção>/midia_enviada/ (o lote temporário some).
            result = ingest_media(config, inputs, output, parameters.pop("building_group"), keep_sources=True, **parameters)
            # 2. Path não vai para JSON: converte para texto.
            result = {key: (str(value) if isinstance(value, Path) else value) for key, value in result.items()}
            result["href"] = f"/r/{entry['name']}/"
            self.workspace.register(entry["name"], output / "registry.json")
            # 3. Sugestões do modelo logo em seguida, se a pessoa escolheu pesos.
            result["suggest_job"] = None
            if weights:
                try:
                    status = jobs.start("suggest", _job_arguments(self.root, output / "registry.json", "suggest", {"weights": weights}))
                    result["suggest_job"] = {"state": status["state"], "log": status["log"]}
                except (RuntimeError, ValueError, OSError) as error:
                    # Inspeção criada mesmo assim; a página mostra por que as sugestões não começaram.
                    result["suggest_job"] = {"state": "not_started", "error": str(error)}
            entry.update(state="finished", result=result, message="Inspeção criada.")
        except Exception as error:  # noqa: BLE001 - o erro vai para a página, não para o terminal
            entry.update(state="failed", error=str(error), message="A importação falhou.")
        finally:
            # 4. Limpeza: apaga o lote temporário e o resumo em cache da nova inspeção.
            shutil.rmtree(entry["directory"], ignore_errors=True)
            self.workspace._cache.pop(entry["name"], None)

    def status(self, batch: str) -> dict:
        """Estado de um lote para a página acompanhar (``GET /api/uploads/<lote>``).

        Parâmetros:
            batch: id do lote (qualquer fase).

        Retorna:
            ``{"batch", "name", "state" ("open"|"running"|"finished"|"failed"), "files" (quantidade),
            "bytes", "message", "result", "error"}``.

        Erros: ValueError "Lote de envio desconhecido." (vira HTTP 404).
        """
        entry = self._batch(batch, state=None)
        return {"batch": batch, "name": entry["name"], "state": entry["state"], "files": len(entry["files"]), "bytes": entry["bytes"],
                "message": entry["message"], "result": entry["result"], "error": entry["error"]}


def _reviews_meta(reviews: dict[str, Path], current: str | None) -> str:
    """Lista das revisões abertas pelo servidor (seletor da fila); vazia com uma só revisão.

    Parâmetros:
        reviews: ``{nome: registry.json}`` abertos.
        current: nome da revisão sendo servida.

    Retorna:
        Texto JSON ``{"current": nome, "items": [{"name", "display_name", "href", "images"}]}``
        já escapado para caber no atributo ``content`` de uma tag ``<meta>`` (aspas viram
        ``&quot;``); texto vazio quando há menos de duas revisões.

    Usada por ``Handler._page`` (meta ``dronecamp-reviews``, lida por ``REVIEWS`` no platform.js).
    """
    if len(reviews) < 2:
        return ""
    items = []
    for name, path in reviews.items():
        # Página não gerada: entra sem número de fotos e com o nome derivado da pasta.
        try:
            payload = load_page_payload(path)
            images, display = len(payload.get("images") or []), payload.get("display_name")
        except (OSError, ValueError):
            images, display = None, None
        items.append({"name": name, "display_name": display or review_name(name), "href": f"/r/{name}/", "images": images})
    return html.escape(json.dumps({"current": current, "items": items}, ensure_ascii=False), quote=True)


# ---------------------------------------------------------------------------
# Rotas HTTP (Handler criado por make_handler)
# ---------------------------------------------------------------------------

def make_handler(config: ProjectConfig, registry_path: Path | list[Path] | None, token: str, jobs: JobRunner):
    """Handler HTTP ligado a uma, várias ou todas as revisões; arquivos servidos só da revisão e de runs/.

    Com ``registry_path`` a primeira revisão abre em ``/``; sem ele (modo espaço
    de trabalho) abre a mais recente de ``data/reviews/`` e a lista é relida a
    cada consulta. Cada revisão também abre em ``/r/<pasta>/``: como a página usa
    caminhos relativos para as fotos, o prefixo faz cada foto vir da sua pasta.

    Parâmetros:
        config: configuração do projeto.
        registry_path: um ``registry.json``, uma lista deles (lista fixa) ou None (espaço de trabalho).
        token: segredo sorteado na partida; exigido no cabeçalho ``X-DroneCamp-Token`` de
            toda requisição que altera algo.
        jobs: executor de tarefas compartilhado por todas as requisições.

    Retorna:
        A CLASSE ``Handler`` (subclasse de ``BaseHTTPRequestHandler``) para ``ThreadingHTTPServer``.
        As variáveis abaixo (``root``, ``workspace``, ``uploads``...) ficam no fechamento
        (closure) e são compartilhadas por todas as requisições.

    Chamada por: ``serve_platform`` e pelos testes.
    """
    # Estado compartilhado do servidor (criado uma vez).
    root = config.root.resolve()
    if registry_path is None:
        paths = None
    else:
        paths = registry_path if isinstance(registry_path, list) else [registry_path]
    workspace = Workspace(config, paths)
    uploads = UploadManager(root, workspace)

    def split_review(path: str, reviews: dict[str, Path]) -> tuple[str | None, str]:
        """``/r/<pasta>/resto`` → (pasta, ``/resto``); demais caminhos → revisão padrão.

        Parâmetros:
            path: caminho da URL (sem a query).
            reviews: inspeções abertas.

        Retorna:
            ``(nome, resto)``: nome da inspeção (None se ``/r/<pasta>`` não está aberta) e o
            resto do caminho ("" quando a URL é ``/r/<pasta>`` sem barra final). Fora de
            ``/r/``, o nome é ``workspace.default()`` e o resto é o caminho inteiro.
        """
        # Regex: "/r/" + nome (sem "/") + resto opcional começando com "/".
        match = re.match(r"^/r/([^/]+)(/.*)?$", path)
        if not match:
            return workspace.default(), path
        name = unquote(match[1])
        return (name if name in reviews else None), (match[2] or "")

    def registry_for(options: dict, reviews: dict[str, Path]) -> Path:
        """Registro da revisão indicada no corpo (``review``) ou da revisão padrão.

        Parâmetros:
            options: corpo da requisição; a chave ``review`` é REMOVIDA dele (``pop``) para não
                ser confundida com uma opção do comando.
            reviews: inspeções abertas.

        Retorna: caminho do ``registry.json``.

        Erros: ValueError "Revisão não aberta neste servidor: <nome>".
        """
        name = options.pop("review", None) or workspace.default()
        if name not in reviews:
            raise ValueError(f"Revisão não aberta neste servidor: {name}")
        return reviews[name]

    class Handler(BaseHTTPRequestHandler):
        """Atende cada requisição HTTP (uma instância por requisição, numa thread própria).

        Métodos de rota: ``do_GET`` (página, consultas e arquivos), ``do_POST`` (ações),
        ``do_PUT`` (envio de arquivo e edição de classe), ``do_DELETE`` (desativar classe).
        Auxiliares: ``_send``, ``_send_file``, ``_json``, ``_not_found``, ``_body``,
        ``_authorized``, ``_page``, ``_class_change``.
        """

        # Nome do servidor no cabeçalho "Server" das respostas.
        server_version = "DroneCampPlatform/2"

        def log_message(self, format, *args):  # noqa: A002 - assinatura da biblioteca
            """Silencia o log de cada requisição que a biblioteca imprime no terminal.

            Parâmetros: ``format`` e ``args`` da biblioteca (ignorados).
            Retorna: None.
            """
            pass  # o terminal fica livre para o log do treino

        def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
            """Envia uma resposta completa já em memória.

            Parâmetros:
                status: código HTTP (ex.: 200, 404).
                body: corpo em bytes.
                content_type: valor do cabeçalho Content-Type.
                extra: cabeçalhos adicionais ``{nome: valor}``, opcional.

            Retorna: None.

            Cabeçalhos fixos: ``Cache-Control: no-store`` (a página sempre vê dados frescos) e
            ``X-Content-Type-Options: nosniff`` (o navegador não "adivinha" outro tipo).
            """
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
            """Arquivo em blocos (pacotes ZIP e PDFs grandes não passam inteiros pela memória).

            Parâmetros:
                target: arquivo já validado (dentro da pasta permitida).
                attachment: True envia ``Content-Disposition: attachment`` (o navegador baixa
                    em vez de abrir).

            Retorna: None.
            """
            # Tipo: tabela própria, senão o palpite da biblioteca, senão binário genérico.
            suffix = target.suffix.lower()
            kind = CONTENT_TYPES.get(suffix) or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(target.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if attachment:
                # Nome do arquivo só com caracteres seguros (o resto vira "_") para o cabeçalho.
                safe = re.sub(r"[^A-Za-z0-9._-]+", "_", target.name)
                self.send_header("Content-Disposition", f'attachment; filename="{safe}"')
            self.end_headers()
            # Copia em blocos de 1 MiB.
            with target.open("rb") as stream:
                shutil.copyfileobj(stream, self.wfile, 1024 * 1024)

        def _json(self, value: dict, status: int = 200) -> None:
            """Responde com JSON em UTF-8 (acentos sem escape).

            Parâmetros:
                value: dict a serializar.
                status: código HTTP (padrão 200).

            Retorna: None.
            """
            self._send(status, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _not_found(self, text: str = "Nao encontrado") -> None:
            """Responde 404 com texto simples.

            Parâmetros:
                text: mensagem (sem acento no padrão, por ser texto puro).

            Retorna: None.
            """
            self._send(404, text.encode("utf-8"), "text/plain; charset=utf-8")

        def _body(self) -> dict:
            """Lê e decodifica o corpo JSON da requisição.

            Parâmetros: nenhum (usa ``Content-Length`` e ``self.rfile``).

            Retorna: o objeto JSON (dict); corpo vazio vira ``{}``.

            Erros:
                ValueError "Conteúdo grande demais." (acima de ``MAX_BODY_BYTES``),
                "Envie um objeto JSON." (não é objeto) ou erro de JSON inválido.
            """
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                raise ValueError("Conteúdo grande demais.")
            value = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(value, dict):
                raise ValueError("Envie um objeto JSON.")
            return value

        # ATENCAO: toda rota que altera algo (POST/PUT/DELETE) passa por aqui. A página manda o token
        # no cabeçalho X-DroneCamp-Token; sem ele a resposta é 403 (proteção contra outros sites).
        def _authorized(self) -> bool:
            """Confere o token da requisição.

            Parâmetros: nenhum (lê o cabeçalho ``X-DroneCamp-Token``).

            Retorna: True se o cabeçalho é igual ao token da partida.
            ``secrets.compare_digest`` compara em tempo constante (não revela o token por tempo de resposta).
            """
            return secrets.compare_digest(self.headers.get("X-DroneCamp-Token", ""), token)

        def _page(self, reviews: dict[str, Path], name: str | None) -> None:
            """Monta e envia o ``index.html`` da inspeção com dados frescos e o token.

            Parâmetros:
                reviews: inspeções abertas.
                name: inspeção a mostrar, ou None (espaço de trabalho vazio: página sem fotos).

            Retorna: None.

            Erros: ValueError de ``_require_current_taxonomy`` (taxonomia antiga) ou de
            ``load_page_payload`` (página não gerada); OSError de leitura. ``do_GET`` responde 409.
            """
            # 1. Dados da página e instantâneo de operações.
            if name is None:
                payload, operations = empty_page_payload(config), collect_operations(config, None)
            else:
                _require_current_taxonomy(config, reviews[name])  # modo espaço de trabalho: recusa ao abrir
                payload = load_page_payload(reviews[name])
                operations = collect_operations(config, reviews[name].parent.resolve())
            # Nome de exibição, cor e ativo/inativo da seção Classes valem na página servida.
            payload = {**payload, "classes": class_catalog.apply_to_page_classes(config, payload.get("classes") or [])}
            # 2. HTML completo (modelo + CSS + JS + dados embutidos).
            page = build_page(payload, operations)
            # O token só existe na página servida; a cópia em disco nunca o contém.
            # ATENCAO: estas substituições dependem do texto EXATO das tags <meta> em index.html;
            # se mudar a tag lá, mude aqui também (senão a página não acha o servidor).
            page = page.replace('<meta name="dronecamp-server" content="">', f'<meta name="dronecamp-server" content="{token}">', 1)
            page = page.replace('<meta name="dronecamp-reviews" content="">',
                                f'<meta name="dronecamp-reviews" content="{_reviews_meta(reviews, name)}">', 1)
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")

        # EXTENSAO: rota de leitura nova = um ``if full == "/api/..."`` aqui, antes do trecho que
        # serve arquivos da revisão; devolva com self._json(...). Do lado da página, use api(path).
        # ENTRADA: toda requisição GET do navegador (página, consultas /api/... e arquivos).
        def do_GET(self) -> None:  # noqa: N802 - nome exigido pela biblioteca
            """Rotas de leitura (não exigem token). Ordem: rotas ``/api/...``, depois a página e,
            por fim, arquivos da revisão ou de ``runs/``.

            Parâmetros: nenhum (usa ``self.path``; ``?review=<pasta>`` escolhe a inspeção nas consultas).
            Retorna: None (a resposta é escrita no socket).
            """
            # Separa caminho e query; "asked" = inspeção pedida em ?review=.
            parts = urlsplit(self.path)
            full, query = parts.path, parse_qs(parts.query)
            reviews = workspace.reviews()
            asked = (query.get("review") or [None])[0]
            # GET /api/operations: treinos, curvas, comparações, datasets e pesos
            # (operations.collect_operations). Página: refreshOperations a cada 5 s, para as
            # seções Início, Treinamento, Monitoramento e Modelos.
            if full == "/api/operations":
                name = asked if asked in reviews else workspace.default()
                return self._json(collect_operations(config, reviews[name].parent.resolve() if name else None))
            # GET /api/jobs: estado da tarefa e final do log (JobRunner.status). Página:
            # refreshOperations → renderJob (painel de tarefa no Treinamento/Monitoramento).
            if full == "/api/jobs":
                return self._json(jobs.status())
            # GET /api/workspace: lista de inspeções com resumo e capacidades do PC
            # (Workspace.describe). Página: refreshWorkspace (seções Início e Inspeções e o seletor).
            if full == "/api/workspace":
                return self._json(workspace.describe(asked if asked in reviews else None))
            # GET /api/classes: cadastro de classes (class_catalog.list_classes). Página:
            # loadClasses → seção Classes. Arquivo de edições inválido = 500 com a mensagem.
            if full == "/api/classes":
                try:
                    return self._json(class_catalog.list_classes(config))
                except (OSError, ValueError) as error:
                    return self._json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            # GET /api/reviews/<pasta>/files: laudos e ZIPs já gerados (operations.collect_review_files).
            # Página: renderFileHistory na seção Laudos e downloads.
            match = re.match(r"^/api/reviews/([^/]+)/files$", full)
            if match:
                name = unquote(match[1])
                if name not in reviews:
                    return self._json({"error": "Revisão não aberta neste servidor."}, HTTPStatus.NOT_FOUND)
                return self._json({"review": name, "files": collect_review_files(reviews[name].parent)})
            # GET /api/uploads/<lote>: andamento de um envio/importação (UploadManager.status).
            # Página: consulta repetida na seção Importar mídia até "finished"/"failed".
            match = re.match(r"^/api/uploads/([^/]+)$", full)
            if match:
                try:
                    return self._json(uploads.status(match[1]))
                except ValueError as error:
                    return self._json({"error": str(error)}, HTTPStatus.NOT_FOUND)
            # Daqui para baixo: página e arquivos. Descobre de qual inspeção é o caminho.
            name, path = split_review(full, reviews)
            # GET / sem nenhuma inspeção pronta: página vazia (só classes, operações e importação).
            if name is None and not full.startswith("/r/") and path in {"/", "/index.html"}:
                return self._page(reviews, None)  # espaço de trabalho vazio: a página abre sem inspeção
            # /r/<pasta> de uma inspeção não aberta (ou outro caminho sem inspeção padrão): 404.
            if name is None:
                return self._not_found("Revisao nao aberta neste servidor")
            # GET / ou /r/<pasta>/: a página da inspeção (Handler._page → review_render.build_page).
            # Página não gerada ou taxonomia antiga: 409 com a explicação.
            if path in {"/", "/index.html"}:
                try:
                    return self._page(reviews, name)
                except (OSError, ValueError) as error:
                    text = f"A página desta revisão ainda não foi gerada (render-review): {error}"
                    return self._send(HTTPStatus.CONFLICT, text.encode("utf-8"), "text/plain; charset=utf-8")
            # GET /r/<pasta> sem barra final: redireciona (301) para /r/<pasta>/.
            if path == "":  # /r/<pasta> sem barra: relativos precisam da barra final
                self.send_response(HTTPStatus.MOVED_PERMANENTLY)
                self.send_header("Location", full + "/")
                self.end_headers()
                return None
            # GET de arquivo: fotos e dados (seção Revisão), imagens de treino em runs/
            # (seções Treinamento/Monitoramento/Modelos) e downloads (seção Laudos e downloads).
            directory = reviews[name].parent.resolve()
            # Arquivos: /runs/... vem de sistema_ia/runs; o resto, da pasta da revisão.
            relative = re.sub(r"^/+", "", unquote(path))
            base = root if relative.startswith("runs/") else directory
            target = (base / relative).resolve()
            suffix = target.suffix.lower()
            # ATENCAO: _inside impede "../" de sair da pasta base; nunca sirva arquivo sem esta checagem.
            if not _inside(target, base) or not target.is_file():
                return self._not_found()
            # Downloads: só em laudos/ ou exportacoes/ da revisão e com extensão de DOWNLOAD_SUFFIXES.
            download = base is directory and relative.split("/", 1)[0] in DOWNLOAD_FOLDERS and suffix in DOWNLOAD_SUFFIXES
            if download:
                # ZIP e PDF são baixados; HTML/JSON/CSV/TXT/MD abrem no navegador.
                return self._send_file(target, attachment=suffix in {".zip", ".pdf"})
            # Demais arquivos: só os tipos de REVIEW_SUFFIXES.
            if suffix not in REVIEW_SUFFIXES:
                return self._not_found()
            return self._send_file(target, attachment=False)

        def _class_change(self, method: str) -> None:
            """PUT/DELETE em ``/api/classes/<id>``: edita ou desativa (exclusão lógica).

            Parâmetros:
                method: "PUT" (edita com o corpo JSON) ou "DELETE" (desativa; autor em ``?author=``).

            Retorna: None. Responde 200 com a classe atualizada, 404 (rota/ID inexistente) ou
            400 (validação).

            ATENCAO: o token já foi conferido por do_PUT/do_DELETE antes de chegar aqui.
            """
            # ID: 1 a 6 dígitos no fim do caminho (sem query).
            match = re.match(r"^/api/classes/(\d{1,6})$", self.path.split("?", 1)[0])
            if not match:
                return self._json({"error": "Rota desconhecida."}, HTTPStatus.NOT_FOUND)
            try:
                # DELETE /api/classes/<id>: class_catalog.deactivate_class.
                if method == "DELETE":
                    query = parse_qs(urlsplit(self.path).query)  # DELETE sem corpo: autor opcional em ?author=
                    author = _text({"author": (query.get("author") or [None])[0]}, "author", 100) or None
                    return self._json(class_catalog.deactivate_class(config, int(match[1]), author))
                # PUT /api/classes/<id>: class_catalog.update_class. "author" sai do corpo para não
                # ser recusado como campo não editável.
                body = self._body()
                author = _text(body, "author", 100) or None
                body.pop("author", None)
                return self._json(class_catalog.update_class(config, int(match[1]), body, author))
            except class_catalog.ClassNotFound as error:
                return self._json({"error": str(error)}, HTTPStatus.NOT_FOUND)
            except (ValueError, OSError) as error:
                return self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        # ENTRADA: requisições DELETE (só /api/classes/<id>).
        def do_DELETE(self) -> None:  # noqa: N802
            """DELETE ``/api/classes/<id>``: desativa a classe (``class_catalog.deactivate_class``).

            Página: botão "Desativar" da seção Classes (``classRequest('DELETE', ...)``).
            Sem token válido: 403.

            Parâmetros: nenhum. Retorna: None.
            """
            if not self._authorized():
                return self._json({"error": "Token ausente ou inválido."}, HTTPStatus.FORBIDDEN)
            return self._class_change("DELETE")

        # ENTRADA: requisições PUT (edição de classe e envio de arquivo de mídia).
        def do_PUT(self) -> None:  # noqa: N802
            """Rotas PUT (exigem token; sem ele, 403).

            - PUT ``/api/classes/<id>``: edita a classe (``_class_change`` → ``class_catalog.update_class``);
              página: salvar edição na seção Classes.
            - PUT ``/api/uploads/<lote>/files``: corpo = bytes do arquivo, nome no cabeçalho
              ``X-File-Name`` (``UploadManager.store``); página: ``putFile`` na seção Importar mídia.

            Parâmetros: nenhum. Retorna: None.

            EXTENSAO: rota PUT nova = um ``if``/``re.match`` antes do ``if not match`` final.
            """
            if not self._authorized():
                return self._json({"error": "Token ausente ou inválido."}, HTTPStatus.FORBIDDEN)
            # PUT /api/classes/<id>.
            if self.path.startswith("/api/classes/"):
                return self._class_change("PUT")
            # PUT /api/uploads/<lote>/files.
            match = re.match(r"^/api/uploads/([^/]+)/files$", self.path.split("?", 1)[0])
            if not match:
                return self._json({"error": "Rota desconhecida."}, HTTPStatus.NOT_FOUND)
            try:
                length = int(self.headers.get("Content-Length") or 0)
                return self._json(uploads.store(match[1], self.headers.get("X-File-Name", ""), self.rfile, length))
            except (ValueError, OSError) as error:
                self.close_connection = True  # o corpo pode não ter sido lido inteiro
                return self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        # EXTENSAO: rota que altera algo = um ``if path == "/api/..."`` aqui (o token já foi conferido).
        # ValueError/OSError viram 400 e RuntimeError vira 409, com a mensagem mostrada na página.
        # ENTRADA: requisições POST (tarefas, revisão, importação, classes, laudos e evidências).
        def do_POST(self) -> None:  # noqa: N802
            """Rotas de ação (exigem token; sem ele, 403). O corpo é sempre um objeto JSON.

            Parâmetros: nenhum. Retorna: None.
            Erros viram resposta: RuntimeError → 409; ValueError/OSError → 400; rota
            desconhecida → 404.
            """
            if not self._authorized():
                return self._json({"error": "Token ausente ou inválido."}, HTTPStatus.FORBIDDEN)
            try:
                body = self._body()
                reviews = workspace.reviews()
                path = self.path.split("?", 1)[0]
                # POST /api/jobs: inicia train-pilot ou suggest (_job_arguments → JobRunner.start).
                # Página: startJob, botões "startTrain" (seção Treinamento) e "startSuggest"
                # (seção Monitoramento). Corpo: {"kind": ..., "options": {...}}.
                if path == "/api/jobs":
                    kind = body.get("kind")
                    options = dict(body.get("options") or {})
                    # Só as sugestões precisam de uma revisão (a aberta na página); o treino roda com o espaço vazio.
                    selected = registry_for(options, reviews) if kind == "suggest" else None
                    return self._json(jobs.start(kind, _job_arguments(root, selected, kind, options)))
                # POST /api/jobs/stop: interrompe a tarefa (JobRunner.stop). Página: botão "stopJob"
                # (seção Monitoramento).
                if path == "/api/jobs/stop":
                    return self._json(jobs.stop())
                # POST /api/feedback: grava o arquivo de revisão em feedback_inbox/ (_save_feedback).
                # Página: botão "export" (exportar revisão), usado na seção Revisão.
                if path == "/api/feedback":
                    # O arquivo exportado já diz de qual registro veio (SHA-256).
                    # Procura, entre as inspeções abertas, a que tem esse hash.
                    digest = body.get("registry_sha256")
                    selected = next((value for value in reviews.values() if file_hash(value) == digest), None)
                    if selected is None:
                        raise ValueError("Arquivo de revisão de outra versão do registro.")
                    return self._json(_save_feedback(root, selected, body))
                # POST /api/uploads: abre um lote de envio (UploadManager.create). Página: início
                # do envio na seção Importar mídia. Corpo: {"name": ...}.
                if path == "/api/uploads":
                    return self._json(uploads.create(body.get("name")))
                # POST /api/classes: cria classe proposta (class_catalog.create_class), resposta 201.
                # Página: formulário de nova classe na seção Classes.
                if path == "/api/classes":
                    author = _text(body, "author", 100) or None
                    body.pop("author", None)
                    return self._json(class_catalog.create_class(config, body, author), HTTPStatus.CREATED)
                # POST /api/uploads/<lote>/ingest: começa a importação (UploadManager.start_ingest).
                # Página: último passo do envio na seção Importar mídia.
                match = re.match(r"^/api/uploads/([^/]+)/ingest$", path)
                if match:
                    return self._json(uploads.start_ingest(match[1], config, body, jobs))
                # POST /api/export/evidence: pacote ZIP de fotos demarcadas (_export_evidence).
                # Página: botão "makeEvidence" na seção Laudos e downloads.
                if path == "/api/export/evidence":
                    return self._json(_export_evidence(config, root, registry_for(body, reviews), body))
                # POST /api/report: laudo técnico HTML/PDF/JSON (_build_report).
                # Página: botão "makeReport" na seção Laudos e downloads.
                if path == "/api/report":
                    return self._json(_build_report(config, root, registry_for(body, reviews), body))
                # EXTENSAO: rota POST nova = um "if path == ..." ACIMA desta linha.
                return self._json({"error": "Rota desconhecida."}, HTTPStatus.NOT_FOUND)
            except RuntimeError as error:
                return self._json({"error": str(error)}, HTTPStatus.CONFLICT)
            except (ValueError, OSError) as error:
                return self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    return Handler


# ---------------------------------------------------------------------------
# Ações chamadas pelas rotas POST: evidências, laudo e arquivo de revisão
# ---------------------------------------------------------------------------

def _export_evidence(config: ProjectConfig, root: Path, registry_path: Path, body: dict) -> dict:
    """ZIP de fotos demarcadas em ``exportacoes/`` da revisão (``exports.build_evidence_package``).

    Parâmetros:
        config: configuração do projeto.
        root: pasta ``sistema_ia``.
        registry_path: ``registry.json`` da revisão.
        body: corpo de ``POST /api/export/evidence``: ``max_side`` (None = tamanho original, ou
            inteiro 256–8192 em pixels do maior lado), ``only`` ("all", "decided" ou "positive"),
            ``include_suggestions`` (True inclui sugestões do modelo) e ``decisions`` (rascunho
            atual da página).

    Retorna:
        ``{"images", "boxes", "bytes", "sha256", "file"/"href" (relativos à revisão),
        "saved" (relativo a sistema_ia)}``.

    Erros: ValueError para ``max_side``/``only`` inválidos ou vindos de ``exports``.

    EXTENSAO: filtro novo de fotos = valor novo no conjunto de ``only`` (e suporte em
    ``exports.build_evidence_package`` e no seletor ``evidenceOnly`` da página).
    """
    from .exports import build_evidence_package
    # 1. Validação das opções.
    max_side = body.get("max_side")
    if max_side is not None and (type(max_side) is not int or not 256 <= max_side <= 8192):
        raise ValueError("max_side deve ser vazio (original) ou um inteiro entre 256 e 8192.")
    only = body.get("only") or "all"
    if only not in {"all", "decided", "positive"}:
        raise ValueError("only deve ser all, decided ou positive.")
    # 2. Gera o pacote em <revisão>/exportacoes/.
    result = build_evidence_package(config, registry_path, registry_path.parent / "exportacoes", decisions=_decisions_map(body.get("decisions")),
                                    max_side=max_side, only=only, include_suggestions=body.get("include_suggestions") is True)
    # 3. Caminhos relativos para o link da página.
    archive = Path(result["zip"])
    relative = archive.relative_to(registry_path.parent).as_posix()
    return {"images": result["images"], "boxes": result["boxes"], "bytes": result["bytes"], "sha256": result["sha256"],
            "file": relative, "href": relative, "saved": archive.relative_to(root).as_posix()}


def _build_report(config: ProjectConfig, root: Path, registry_path: Path, body: dict) -> dict:
    """Laudo técnico (estrutura ABNT, ``laudo_abnt``) em ``laudos/`` da revisão (``exports.build_report``); PDF só com reportlab.

    Parâmetros:
        config: configuração do projeto.
        root: pasta ``sistema_ia``.
        registry_path: ``registry.json`` da revisão.
        body: corpo de ``POST /api/report``: ``decisions``, ``reviewer`` (até 100), ``title``,
            ``client``, ``site``, ``inspector`` (até 200 cada), ``notes`` (até 5000),
            ``details`` (ver ``_report_details``), ``pdf`` (False desliga o PDF) e
            ``all_images`` (True inclui fotos sem ocorrência).

    Retorna:
        ``{"images", "occurrences", "pending_suggestions", "html", "pdf", "pdf_reason", "json"
        (relativos à revisão; None se não gerado), "directory" (relativo a sistema_ia)}``.

    Erros: ValueError de validação dos textos/detalhes ou vindos de ``exports.build_report``.

    ATENCAO: o laudo nunca infere gravidade nem aprova dados; só descreve as decisões humanas.

    EXTENSAO: campo novo do formulário do laudo = argumento novo de ``exports.build_report``
    passado aqui (validado com ``_text``) + campo na seção Laudos da página (``makeReport``).
    """
    from .exports import build_report
    # Gera HTML/JSON (e PDF se reportlab estiver instalado) em <revisão>/laudos/<pasta>/.
    result = build_report(config, registry_path, registry_path.parent / "laudos", decisions=_decisions_map(body.get("decisions")),
                          reviewer=_text(body, "reviewer", 100), title=_text(body, "title", 200) or None, client=_text(body, "client", 200),
                          site=_text(body, "site", 200), inspector=_text(body, "inspector", 200), notes=_text(body, "notes", 5000),
                          details=_report_details(body.get("details")),
                          pdf=body.get("pdf") is not False, only_positive_images=body.get("all_images") is not True)
    directory = registry_path.parent

    def href(value) -> str | None:
        """Caminho relativo à pasta da revisão (link da página), ou None se o arquivo não foi gerado.

        Parâmetros: value: caminho absoluto (texto/Path) ou None.
        Retorna: texto com "/" ou None.
        """
        return Path(value).relative_to(directory).as_posix() if value else None

    return {"images": result["images"], "occurrences": result["occurrences"], "pending_suggestions": result.get("pending_suggestions"),
            "html": href(result["html"]), "pdf": href(result.get("pdf")), "pdf_reason": result.get("pdf_reason"),
            "json": href(result.get("json")), "directory": Path(result["directory"]).relative_to(root).as_posix()}


def _save_feedback(root: Path, registry_path: Path, feedback: dict) -> dict:
    """Grava o arquivo exportado ao lado do registro; não importa nem aprova nada.

    Parâmetros:
        root: pasta ``sistema_ia``.
        registry_path: ``registry.json`` da revisão a que o arquivo pertence.
        feedback: arquivo de revisão exportado pela página (``schema_version`` 1,
            ``registry_sha256``, ``reviewer_id``, ``images``: lista de decisões).

    Retorna:
        ``{"saved": caminho relativo a sistema_ia, "decisions": quantidade de decisões}``.

    Erros:
        ValueError "Arquivo de revisão de outra versão do registro." (versão/hash não batem)
        ou "Arquivo de revisão sem decisões."; FileExistsError (OSError) se já existir arquivo
        com o mesmo nome (mesmo segundo e revisor).

    ATENCAO: a importação de verdade é o comando ``import-review`` (etapa 2 do ciclo); a IA
    nunca aprova dados. Este arquivo só fica guardado em ``feedback_inbox/``.
    """
    # 1. O arquivo precisa ser deste registro (mesmo SHA-256) e ter decisões.
    if feedback.get("schema_version") != 1 or feedback.get("registry_sha256") != file_hash(registry_path):
        raise ValueError("Arquivo de revisão de outra versão do registro.")
    if not isinstance(feedback.get("images"), list) or not feedback["images"]:
        raise ValueError("Arquivo de revisão sem decisões.")
    # 2. Nome do arquivo: <AAAAMMDDTHHMMSSZ>_<revisor>.json; o revisor só com caracteres
    # seguros (o resto vira "_") e no máximo 40 caracteres.
    reviewer = re.sub(r"[^A-Za-z0-9_-]+", "_", str(feedback.get("reviewer_id") or "revisor"))[:40]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = registry_path.parent / "feedback_inbox" / f"{stamp}_{reviewer}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    # 3. Grava sem nunca sobrescrever.
    with target.open("x", encoding="utf-8") as stream:  # "x": nunca sobrescreve
        json.dump(feedback, stream, ensure_ascii=False, indent=2)
    return {"saved": target.resolve().relative_to(root).as_posix(), "decisions": len(feedback["images"])}


# ---------------------------------------------------------------------------
# Conferência de taxonomia e partida do servidor
# ---------------------------------------------------------------------------

# ATENCAO: uma inspeção feita com taxonomia anterior não abre na plataforma; o erro pede
# migrate-review + render-review. Não contorne isso: as classes da página ficariam erradas.
def _require_current_taxonomy(config: ProjectConfig, registry_path: Path) -> None:
    """Revisão de taxonomia anterior mostraria classes a menos e geraria arquivos recusados.

    Parâmetros:
        config: configuração do projeto (``taxonomy_path``).
        registry_path: ``registry.json`` da revisão.

    Retorna: None se o ``taxonomy_sha256`` do registro bate com o hash atual de ``taxonomy.json``.

    Erros: ValueError explicando que é preciso rodar ``migrate-review`` e ``render-review``.

    Chamada por: ``Handler._page`` (ao abrir) e ``serve_platform`` (na partida, lista fixa).
    """
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("taxonomy_sha256") != file_hash(config.taxonomy_path):
        raise ValueError(
            f"A revisão {registry_path.parent.name} usa uma taxonomia anterior à atual "
            "(classes diferentes). Migre-a com migrate-review e gere a página com render-review "
            "antes de abri-la na plataforma.")


# ENTRADA: comando platform (e o atalho "Abrir plataforma DroneCamp.cmd").
def serve_platform(config: ProjectConfig, registry: Path | list[Path] | None = None, port: int = 8765, open_browser: bool = True) -> None:
    """Inicia o servidor em 127.0.0.1 até Ctrl+C.

    Com ``registry`` (um ou vários), só essas revisões ficam abertas; sem ele, o
    servidor trabalha com tudo o que houver em ``data/reviews/`` e com as
    inspeções importadas pela página.

    Parâmetros:
        config: configuração do projeto.
        registry: um ou vários ``registry.json`` (``--registry``), ou None (espaço de trabalho).
            Caminho relativo vale a partir do terminal ou, se não existir, de ``sistema_ia``.
        port: porta TCP (padrão 8765; 0 escolhe uma livre).
        open_browser: True abre o navegador na URL (``--no-browser`` desliga).

    Retorna: None (bloqueia até Ctrl+C).

    Erros:
        ValueError/OSError se uma revisão da lista não tem página gerada, usa taxonomia antiga
        ou se duas revisões têm o mesmo nome de pasta; OSError se a porta estiver ocupada.

    Chamada por: ``cli.py`` (comando ``platform``).

    ATENCAO: o endereço é sempre 127.0.0.1 (só este PC). Não troque por "0.0.0.0": a
    plataforma não tem login e serviria fotos e comandos para a rede inteira.
    """
    # 1. Lista fixa: confere cada revisão ANTES de abrir a porta (falha cedo, no terminal).
    paths: list[Path] | None = None
    if registry:
        paths = []
        for value in (registry if isinstance(registry, list) else [registry]):
            # Caminho relativo vale a partir do terminal ou, se não existir, da pasta sistema_ia.
            path = (value if value.is_absolute() or value.exists() else config.root / value).resolve()
            load_page_payload(path)  # falha cedo se a página não foi gerada para este registro
            _require_current_taxonomy(config, path)
            paths.append(path)
        # O endereço /r/<pasta>/ usa o nome da pasta; dois iguais seriam ambíguos.
        if len({path.parent.name for path in paths}) != len(paths):
            raise ValueError("Duas revisões com o mesmo nome de pasta; o endereço /r/<pasta>/ ficaria ambíguo.")
    # 2. Token aleatório desta execução (vai só para a página servida).
    token = secrets.token_urlsafe(24)
    # 3. Servidor multi-thread (uma thread por requisição) preso a 127.0.0.1.
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(config, paths, token, JobRunner(config.root)))
    # Porta real (útil quando port=0).
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    reviews = Workspace(config, paths).reviews()
    print(f"Plataforma DroneCamp em {url} (Ctrl+C para encerrar). Inspeções abertas: {len(reviews)}.", flush=True)
    if open_browser:
        webbrowser.open(url)
    # 4. Atende até Ctrl+C; fecha a porta ao sair.
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
