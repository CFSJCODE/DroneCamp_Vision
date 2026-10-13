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

Demais funções: ``folder``, ``curve_path``, ``pending_checkpoints``, ``directml_available``,
``limit_worker_threads`` (públicas auxiliares) e ``_epoch_of``, ``_retry``,
``_training_seconds``, ``_append``, ``_watch_stdin`` (privadas).

Dois processos, dois lados do arquivo:
- Lado do TREINO (processo do ``train-pilot``, em ``training.train_pilot``):
  ``gpu_eval_settings`` -> ``start_worker`` -> a cada época ``enqueue_checkpoint``
  -> no fim ``finish_worker``.
- Lado do WORKER (subprocesso ``python -m dronecamp_ia gpu-eval-worker``, em ``cli.py``):
  ``run_worker`` -> ``evaluate_checkpoint`` para cada checkpoint -> ``_append`` na curva.
  Os dois só conversam por ARQUIVOS (fila, DONE, curva) e pelo pipe de stdin.

Disco (``<run>`` = ``runs/<treino>``):
- ``<run>/gpu_eval/``: ``eval_curve.jsonl`` (uma linha JSON por avaliação), ``worker.log``,
  ``worker.json`` (estado do worker), ``DONE`` (marcador de fim) e ``.gitignore``.
- ``<work>/queue/epoch_NNNN.pt`` (checkpoints na fila) e ``<work>/export_NNNN/`` (ONNX
  temporário, apagado após cada avaliação). ``<work>`` = ``<run>/gpu_eval`` ou
  ``$DRONECAMP_FIT_STAGING/<nome do run>/gpu_eval``.
- Lê: ``pilot.json`` do dataset piloto (lista de fotos por split) e ``results.csv`` do treino.

Onde mexer:
- Opção nova em ``pilot.gpu_eval``: acrescente a chave e o padrão em ``DEFAULTS``, valide em
  ``gpu_eval_settings`` e, se for opção de linha de comando, em ``cli.py`` (``--gpu-eval-*``)
  e no dict ``gpu_eval`` montado no ramo ``train-pilot`` do ``cli.py``.
- Provedor novo do ONNX Runtime (ex.: CUDA): ``PROVIDERS`` + ``backend.load_inference_model``
  + ``choices`` de ``--provider`` em ``cli.py`` (+ checagem de disponibilidade em ``start_worker``).
- Campo novo em cada linha da curva: dict de retorno de ``evaluate_checkpoint``.
- Campo novo no resumo do fim do treino: ``finish_worker`` (``summary``).
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

# ---------------------------------------------------------------------------
# Constantes: padrões da configuração, nomes de arquivos e limites do worker
# ---------------------------------------------------------------------------

# Valores padrão de pilot.gpu_eval (project.yaml), sobrescritos pelo YAML e pela CLI.
# enabled: liga o worker; every: épocas entre avaliações; provider: "directml" (GPU) ou
# "cpu"; split: "val" ou "test"; timeout: segundos que o fim do treino espera o worker.
# EXTENSAO: opção nova em pilot.gpu_eval: acrescente a chave e o padrão aqui
# (chaves fora deste dict são recusadas por gpu_eval_settings) e valide-a lá.
DEFAULTS = {"enabled": False, "every": 5, "provider": "directml", "split": "val", "timeout": 1800}
# Provedores aceitos. EXTENSAO: provedor novo: acrescente aqui, em backend.load_inference_model
# e nos choices de --provider/--gpu-eval-provider do cli.py.
PROVIDERS = ("directml", "cpu")
# Splits que podem ser avaliados (o treino nunca é avaliado pelo worker).
SPLITS = ("val", "test")
# Nome da subpasta dentro do run.
FOLDER = "gpu_eval"
# Arquivo da curva (JSON Lines: um objeto JSON por linha).
CURVE = "eval_curve.jsonl"
# Marcador criado pelo treino ao terminar: o worker esvazia a fila e sai.
# ATENCAO: protocolo entre processos; trocar o nome exige mudar os dois lados (este arquivo inteiro).
DONE = "DONE"
# Threads de CPU do worker (exportação ONNX e pré/pós-processamento): o treino fica com o resto.
WORKER_THREADS = 2
# Erros seguidos que encerram o worker (ex.: DirectML indisponível).
MAX_CONSECUTIVE_ERRORS = 3
# Espaço livre mínimo para copiar um checkpoint (~200 MB) para a fila.
# Valor em bytes: 2 GiB. Abaixo disso a cópia é recusada para não faltar espaço ao last.pt.
MIN_FREE_BYTES = 2 * 1024 ** 3
# Conteúdo do .gitignore gravado nas pastas do worker.
GITIGNORE = "# Checkpoints e ONNX intermediários (centenas de MB) ficam fora do git.\nqueue/\nexport_*/\n*.pt\n*.onnx\n"


# ---------------------------------------------------------------------------
# Configuração e caminhos.
# ---------------------------------------------------------------------------

def gpu_eval_settings(config: ProjectConfig | dict | None, overrides: dict | None = None) -> dict | None:
    """``pilot.gpu_eval`` + opções da linha de comando; ``None`` quando desligado.

    Parâmetros:
        config: ``ProjectConfig`` (lê ``config.pilot["gpu_eval"]``), ou diretamente o dict
            da seção gpu_eval (usado nos testes), ou None (só padrões + overrides).
        overrides: opções da CLI (``enabled``, ``every``, ``provider``, ``split``);
            valores None são ignorados (opção não informada).

    Retorna:
        dict completo com as chaves de ``DEFAULTS`` quando ``enabled`` é True; senão None.
        Precedência: DEFAULTS < project.yaml < linha de comando.

    Erros:
        ValueError com mensagem "pilot.gpu_eval...." para chave desconhecida ou valor inválido.

    Chamada por: ``training.train_pilot``.
    """
    # Seção crua: o próprio dict, None, ou pilot.gpu_eval do ProjectConfig.
    raw = config if isinstance(config, dict) or config is None else (config.pilot or {}).get("gpu_eval")
    # Mescla em camadas; o dict mais à direita vence.
    settings = {**DEFAULTS, **(raw or {}), **{key: value for key, value in (overrides or {}).items() if value is not None}}
    # Chave digitada errada no YAML (ex.: "evry") vira erro, não é ignorada em silêncio.
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Chaves desconhecidas em pilot.gpu_eval: {', '.join(sorted(unknown))}.")
    # Validação de cada opção. "type(...) is not int" recusa True/False em every.
    # EXTENSAO: validação da opção nova acrescentada em DEFAULTS: ponha um if aqui.
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
    """Arquivos pequenos (curva, log, DONE, worker.json) ficam na execução.

    Parâmetros:
        run: pasta do treino (``runs/<treino>``).

    Retorna:
        ``<run>/gpu_eval``.
    """
    return Path(run) / FOLDER


def work_directory(run: Path) -> Path:
    """Fila e exportações (centenas de MB): na área de gravação DRONECAMP_FIT_STAGING, se houver.

    No PC do projeto a execução fica num HD USB onde gravações grandes já
    falharam (OSError 22); ``training._fit_staging`` manda o ``fit/`` para o
    disco interno e a fila da GPU vai para o mesmo lugar.

    Parâmetros:
        run: pasta do treino.

    Retorna:
        ``$DRONECAMP_FIT_STAGING/<nome do run>/gpu_eval`` se a variável de ambiente
        existir; senão ``<run>/gpu_eval`` (mesma pasta de ``folder``).
    """
    base = os.environ.get("DRONECAMP_FIT_STAGING")
    if base:
        return Path(base).expanduser().resolve() / Path(run).name / FOLDER
    return folder(run)


def curve_path(run: Path) -> Path:
    """Caminho da curva.

    Parâmetros:
        run: pasta do treino.

    Retorna:
        ``<run>/gpu_eval/eval_curve.jsonl`` (sempre no run, nunca na área de gravação).
    """
    return folder(run) / CURVE


def _epoch_of(checkpoint: Path) -> int:
    """Época codificada no nome do checkpoint da fila.

    Parâmetros:
        checkpoint: arquivo ``epoch_NNNN.pt`` (ex.: ``epoch_0015.pt``).

    Retorna:
        o número depois do último ``_`` do nome sem extensão (``15``).

    ATENCAO: depende do formato ``epoch_{epoch:04d}.pt`` de ``enqueue_checkpoint``.
    """
    return int(checkpoint.stem.split("_")[-1])


def _retry(action, attempts: int = 3, pause: float = 2.0):
    """Repete operações de arquivo que o Windows bloqueia por instantes (antivírus, indexação).

    Parâmetros:
        action: função sem argumentos com a operação (ex.: ``lambda: os.replace(a, b)``).
        attempts: tentativas no total (padrão 3).
        pause: segundos de espera entre tentativas.

    Retorna:
        o que ``action`` devolver.

    Erros:
        repassa o último OSError se todas as tentativas falharem; outros erros não são repetidos.
    """
    for attempt in range(attempts):
        try:
            return action()
        except OSError:
            # Última tentativa: desiste e propaga o erro.
            if attempt == attempts - 1:
                raise
            time.sleep(pause)


def directml_available() -> bool:
    """Diz se o ONNX Runtime instalado tem o provedor DirectML (GPU no Windows).

    Parâmetros: nenhum.

    Retorna:
        True se ``onnxruntime`` importa e lista ``DmlExecutionProvider``; False caso contrário
        (pacote ausente, ou ``onnxruntime`` comum em vez de ``onnxruntime-directml``).
    """
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

    Parâmetros:
        work: pasta de trabalho (``work_directory(run)``).
        checkpoint: ``last.pt`` do treino.
        epoch: número da época (1, 2, ...).

    Retorna:
        caminho final ``<work>/queue/epoch_NNNN.pt`` (época com 4 dígitos).

    Erros:
        OSError quando há menos de ``MIN_FREE_BYTES`` livres, ou se a cópia/renomeação
        falhar nas 3 tentativas de ``_retry``.

    Chamada por: ``training.train_pilot`` (callback de fim de época e no fim do treino).
    """
    queue = Path(work) / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    # 1. Espaço livre no disco da fila.
    if shutil.disk_usage(queue).free < MIN_FREE_BYTES:
        raise OSError(f"menos de {MIN_FREE_BYTES // 1024 ** 3} GB livres em {queue}")
    # 2. Nome com 4 dígitos (epoch_0005.pt): ordena certo e _epoch_of lê o número.
    target = queue / f"epoch_{epoch:04d}.pt"
    temporary = target.with_suffix(".tmp")
    # 3. Copia para .tmp e renomeia: os.replace é atômico no mesmo disco, então o worker
    #    (que só lista epoch_*.pt) nunca abre um checkpoint incompleto.
    # ATENCAO: não copie direto para .pt; o worker poderia exportar um arquivo truncado.
    _retry(lambda: shutil.copyfile(checkpoint, temporary))
    _retry(lambda: os.replace(temporary, target))
    return target


def pending_checkpoints(work: Path) -> list[Path]:
    """Checkpoints aguardando avaliação na fila.

    Parâmetros:
        work: pasta de trabalho.

    Retorna:
        lista de ``<work>/queue/epoch_*.pt`` em ordem crescente de época
        (lista vazia se a fila ainda não existe). Arquivos ``.tmp`` não entram.
    """
    queue = Path(work) / "queue"
    return sorted(queue.glob("epoch_*.pt"), key=_epoch_of) if queue.is_dir() else []


# ENTRADA: lado do treino; training.train_pilot inicia o worker quando pilot.gpu_eval está ligado.
def start_worker(config: ProjectConfig, run: Path, data_path: Path, settings: dict, imgsz: int,
                 config_path: Path | None = None) -> subprocess.Popen | None:
    """Inicia ``python -m dronecamp_ia gpu-eval-worker``; ``None`` quando o DirectML não existe.

    O worker recebe um pipe em stdin: quando o treino morre (inclusive por
    ``TerminateProcess`` no Windows), o pipe fecha e o worker se encerra
    sozinho — nenhum processo órfão segurando a GPU.

    Parâmetros:
        config: configuração do projeto (``config.root`` vira o diretório do subprocesso).
        run: pasta do treino.
        data_path: ``dataset.yaml`` do dataset piloto (o worker lê o ``pilot.json`` ao lado).
        settings: resultado de ``gpu_eval_settings`` (usa ``provider`` e ``split``).
        imgsz: tamanho de entrada do treino (px), repassado à exportação ONNX.
        config_path: ``project.yaml`` a passar ao worker; padrão: ``$DRONECAMP_CONFIG``
            ou ``<root>/configs/project.yaml``.

    Retorna:
        o ``subprocess.Popen`` do worker (com o atributo extra ``dronecamp_log``, o
        arquivo de log aberto), ou None quando o provedor é directml e ele não existe
        (``worker.json`` registra ``state: skipped``).
    """
    # 1. Cria as pastas (run e área de trabalho; o set evita repetir quando são a mesma)
    #    e grava o .gitignore que mantém checkpoints e ONNX fora do git.
    run, work = Path(run).resolve(), work_directory(run)
    for directory in {folder(run), work}:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    # 2. Sem DirectML não há GPU para usar: registra e segue o treino sem worker.
    # EXTENSAO: provedor novo que dependa de instalação específica: confira a disponibilidade aqui.
    if settings["provider"] == "directml" and not directml_available():
        write_json(folder(run) / "worker.json", {"state": "skipped", "reason": "directml_unavailable",
                                                 "settings": settings})
        return None
    # 3. Log do worker em modo "append binário" (stdout e stderr juntos).
    log = (folder(run) / "worker.log").open("ab")
    # 4. Ambiente do subprocesso: poucas threads (o treino fica com a CPU) e UTF-8 sem buffer
    #    (as mensagens aparecem no log na hora).
    environment = {**os.environ, "OMP_NUM_THREADS": str(WORKER_THREADS), "MKL_NUM_THREADS": str(WORKER_THREADS),
                   "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    config_path = config_path or os.environ.get("DRONECAMP_CONFIG") or (config.root / "configs" / "project.yaml")
    # 5. Linha de comando do worker.
    # ATENCAO: estes argumentos precisam existir no subcomando gpu-eval-worker do cli.py.
    # EXTENSAO: argumento novo para o worker: acrescente aqui, no parser gpu-eval-worker do
    # cli.py e no parâmetro correspondente de run_worker.
    command = [sys.executable, "-m", "dronecamp_ia", "--config", str(config_path),
               "gpu-eval-worker", "--run", str(run), "--work", str(work), "--data", str(Path(data_path).resolve()),
               "--provider", settings["provider"], "--split", settings["split"], "--imgsz", str(imgsz), "--watch-stdin",
               # No Windows o worker observa este pid em vez do stdin (ver _watch_parent).
               "--parent-pid", str(os.getpid())]
    # 6. No Windows o worker roda com prioridade abaixo do normal (o treino tem preferência).
    extra = {}
    if os.name == "nt":
        extra["creationflags"] = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    # 7. Inicia o processo; stdin=PIPE é o "cordão" que o _watch_stdin do worker observa.
    process = subprocess.Popen(command, cwd=str(config.root), env=environment, stdin=subprocess.PIPE,
                               stdout=log, stderr=subprocess.STDOUT, **extra)
    process.dronecamp_log = log  # fechado em finish_worker
    # 8. Estado do worker para diagnóstico (pid, comando, opções).
    write_json(folder(run) / "worker.json", {"state": "running", "pid": process.pid, "command": command,
                                             "settings": settings, "work": str(work), "imgsz": imgsz,
                                             "started_at": datetime.now(timezone.utc).isoformat()})
    return process


def _training_seconds(run: Path) -> float | None:
    """Duração do treino pelo ``time`` acumulado do results.csv (fit/ ou área de gravação).

    Parâmetros:
        run: pasta do treino.

    Retorna:
        o maior valor da coluna ``time`` (segundos acumulados que a Ultralytics grava),
        do primeiro ``results.csv`` legível; None se nenhum existir ou servir.
    """
    # Candidatos: <run>/fit/results.csv e, se houver área de gravação, a cópia de lá.
    # A expressão *( [...] if ... else [] ) acrescenta o segundo caminho só quando a variável existe.
    for candidate in (Path(run) / "fit" / "results.csv",
                      *( [Path(os.environ["DRONECAMP_FIT_STAGING"]).expanduser() / Path(run).name / "fit" / "results.csv"]
                         if os.environ.get("DRONECAMP_FIT_STAGING") else [])):
        try:
            with candidate.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            # A Ultralytics alinha o cabeçalho com espaços (ex.: "      time"); key.strip() limpa.
            # "if key" descarta a coluna sem nome (None) que aparece em linhas com campos a mais.
            values = [float({key.strip(): value for key, value in row.items() if key}["time"]) for row in rows]
            return max(values) if values else None
        except (OSError, KeyError, ValueError):
            # Arquivo ausente, sem coluna time ou valor não numérico: tenta o próximo.
            continue
    return None


def finish_worker(run: Path, process: subprocess.Popen | None, timeout: float) -> dict:
    """Marca DONE, espera o worker esvaziar a fila e resume a curva; nunca derruba o treino.

    Parâmetros:
        run: pasta do treino.
        process: o ``Popen`` de ``start_worker`` (ou None se não houve worker).
        timeout: segundos de espera pelo worker (``pilot.gpu_eval.timeout``; 0 = encerra já).

    Retorna:
        dict de resumo gravado no run do treino: ``folder``, ``curve``, ``entries``
        (avaliações bem-sucedidas), ``pending`` (checkpoints não avaliados),
        ``worker_returncode``, ``best_epoch_by_map50`` (se houver), ``evaluation_errors``,
        ``parallel`` (segundos de avaliação, exportação, inferência, treino, fração de
        ocupação da GPU, provedores) e ``error`` (texto) quando algo falhou.

    Chamada por: ``training.train_pilot`` no fim do treino (e em falha, com timeout 0).

    ATENCAO: "melhor época por mAP50" é só informativo; não troca o best.pt da Ultralytics.
    """
    run = Path(run).resolve()
    summary: dict = {"folder": str(folder(run)), "curve": str(curve_path(run)), "entries": 0}
    try:
        # 1. Cria o marcador DONE: o worker termina quando a fila esvaziar.
        folder(run).mkdir(parents=True, exist_ok=True)
        (folder(run) / DONE).write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
        if process is not None:
            # 2. Espera o worker; se estourar o tempo, pede para encerrar (terminate),
            #    espera 30 s e, se ainda vivo, força (kill).
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
            # 3. Fecha o log e o stdin (fechar o stdin também avisa o worker, se ainda estiver vivo).
            for handle in (getattr(process, "dronecamp_log", None), process.stdin):
                if handle is not None:
                    handle.close()
            if process.returncode not in (None, 0) and "error" not in summary:
                summary["error"] = f"worker terminou com código {process.returncode}; veja gpu_eval/worker.log."
    except OSError as error:
        summary["error"] = f"{type(error).__name__}: {error}"
    # 4. Resumo da curva: avaliações válidas, fila restante e melhor época por mAP50.
    entries = read_curve(run)
    summary["entries"] = len(entries)
    summary["pending"] = [path.name for path in pending_checkpoints(work_directory(run))]
    scored = [entry for entry in entries if entry.get("map50") is not None]
    if scored:
        best = max(scored, key=lambda entry: entry["map50"])
        summary["best_epoch_by_map50"] = {"epoch": best["epoch"], "map50": best["map50"], "split": best.get("split")}
    # Linhas com "error" (falhas e épocas puladas como "superseded").
    errors = [entry for entry in read_curve(run, include_errors=True) if "error" in entry]
    if errors:
        summary["evaluation_errors"] = len(errors)
    # O que foi feito em paralelo: segundos de GPU/exportação executados durante o treino.
    # gpu_busy_fraction = segundos de avaliação ÷ segundos de treino (ex.: 0.25 = GPU ocupada 1/4 do tempo).
    # EXTENSAO: indicador novo no resumo do paralelismo: acrescente neste dict.
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
    """Linhas válidas de ``eval_curve.jsonl`` em ordem de época (linhas corrompidas são ignoradas).

    Parâmetros:
        run: pasta do treino.
        include_errors: True inclui também as linhas com ``error`` (falhas e "superseded").

    Retorna:
        lista de dicts (uma por linha), só os que têm ``epoch`` inteiro, ordenada por época.
        Lista vazia se a curva não existe.

    Chamada por: ``finish_worker``, ``model_report.py``, ``operations.py`` e os testes.
    """
    path = curve_path(run)
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        # Uma linha cortada (worker morto no meio da gravação) não derruba a leitura.
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
    """A Ultralytics devolve o torch a 8 threads em todo predict/export; fixa o limite do worker.

    Parâmetros:
        threads: número de threads de CPU (padrão ``WORKER_THREADS``).

    Retorna: None. Ajusta ``torch.set_num_threads`` e a constante ``NUM_THREADS`` da
    Ultralytics (que ela usa ao reconfigurar o torch).
    """
    import torch

    torch.set_num_threads(threads)
    try:
        import ultralytics.utils as utils
        import ultralytics.utils.torch_utils as torch_utils

        # ATENCAO: atributo interno da Ultralytics; se uma versão nova renomear, o limite deixa de valer.
        utils.NUM_THREADS = threads
        torch_utils.NUM_THREADS = threads
    except ImportError:
        pass


def evaluate_checkpoint(config: ProjectConfig, run: Path, work: Path, data_path: Path, checkpoint: Path,
                        provider: str, split: str, imgsz: int) -> dict:
    """Exporta o checkpoint para ONNX em ``<work>/export_NNNN`` e avalia o split nas fotos inteiras.

    Parâmetros:
        config: configuração do projeto.
        run: pasta do treino (não usada aqui; faz parte da assinatura comum de ``evaluator``).
        work: pasta de trabalho (onde é criada ``export_NNNN``).
        data_path: ``dataset.yaml`` do dataset piloto; ``pilot.json`` precisa estar na mesma pasta.
        checkpoint: ``<work>/queue/epoch_NNNN.pt``.
        provider: ``"directml"`` ou ``"cpu"``.
        split: ``"val"`` ou ``"test"`` (fotos do ``pilot.json`` com esse split).
        imgsz: tamanho de entrada do ONNX em pixels.

    Retorna:
        dict de uma linha da curva: ``epoch``, ``split``, ``provider``, ``provider_active``,
        ``runtime``, ``images``, ``conf``, ``map50``, ``precision``, ``recall``, ``tp``, ``fp``,
        ``fn``, ``classes_measured``, ``strata``, ``regime``, ``per_class``, hashes do
        checkpoint e do ONNX, ``imgsz``, tempos (``export_seconds``, ``inference_seconds``,
        ``seconds``) e horários de início/fim.

    Erros:
        qualquer exceção da exportação, carga do modelo ou avaliação é propagada
        (``run_worker`` a registra na curva); ValueError se o split não tem fotos.

    EXTENSAO: campo novo em cada linha da curva: acrescente no dict do ``return``
    (e, se for numérico e somável, no resumo ``parallel`` de ``finish_worker``).
    """
    # Imports pesados só no worker (torch, Ultralytics, ONNX Runtime).
    from dataclasses import replace

    from .backend import check_domain_names, load_inference_model
    from .config import detection_names, load_taxonomy
    from .exporting import export_onnx
    from .model_gate import evaluate_model
    from .training import pilot_inference_config

    # monotonic para medir duração (não muda com ajuste de relógio); ISO UTC para registro.
    started, started_at = time.monotonic(), datetime.now(timezone.utc).isoformat()
    epoch = _epoch_of(checkpoint)
    # 1. Pasta de exportação limpa (sobra de uma falha anterior é apagada).
    export_dir = Path(work) / f"export_{epoch:04d}"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    # imgsz explícito: fora da pasta da execução o checkpoint não é reconhecido como piloto.
    try:
        # 2. Exportação .pt -> ONNX com o imgsz do treino (cópia da config com prediction.imgsz trocado).
        export_config = replace(config, prediction={**config.prediction, "imgsz": int(imgsz)})
        # Sem paridade (parity_images=[]): a conferência .pt × .onnx custa tanto quanto a avaliação.
        export_onnx(export_config, str(checkpoint), demo=False, parity_images=[], output=export_dir)
        export_seconds = round(time.monotonic() - started, 1)
        onnx_path = export_dir / "model.onnx"
        onnx_sha256 = file_hash(onnx_path)
        # 3. Carrega o ONNX no provedor pedido e confere que as classes do modelo batem com a taxonomia.
        # ATENCAO: IDs de classe são contrato; check_domain_names barra pesos de outra taxonomia.
        names = detection_names(load_taxonomy(config.taxonomy_path))
        model = load_inference_model(config, str(onnx_path), provider)
        check_domain_names(model, names)
        model_config = pilot_inference_config(config, model)
        # 4. Fotos do split pedido, lidas do pilot.json ao lado do dataset.yaml.
        root = Path(data_path).resolve().parent
        manifest = json.loads((root / "pilot.json").read_text(encoding="utf-8"))
        samples = [sample for sample in manifest["images"] if sample["split"] == split]
        if not samples:
            raise ValueError(f"Nenhuma foto no split {split} do dataset piloto.")
        # 5. Mesma confiança mínima das sugestões (pilot.suggestion_conf), ou a de prediction.conf.
        conf = float(config.pilot.get("suggestion_conf", config.prediction["conf"]))
        inference_started = time.monotonic()
        # 6. Avaliação nas fotos inteiras (com inferência fatiada quando configurada).
        report = evaluate_model(model, model_config, root, samples, names, conf)
        inference_seconds = round(time.monotonic() - inference_started, 1)
        # O relatório mistura métricas por classe com chaves especiais "_..."; separa-as.
        total = report.pop("_total")
        strata = {key: value["_total"] for key, value in (report.pop("_strata", None) or {}).items()}
        regime = report.pop("_regime", None)
        provider_active = None
        try:  # provedor que a sessão ONNX de fato ativou (prova de GPU)
            provider_active = model.predictor.model.session.get_providers()[0]
        except AttributeError:
            pass
    finally:
        # O .pt e o ONNX intermediários não ficam, nem numa falha: a curva guarda os hashes.
        shutil.rmtree(export_dir, ignore_errors=True)
    # 7. Linha da curva. Depois do "report.pop", report contém só as métricas por classe.
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
    """Acrescenta uma linha JSON ao fim de ``eval_curve.jsonl``.

    Parâmetros:
        run: pasta do treino.
        entry: dict da avaliação (ou do erro) a gravar.

    Retorna: None.

    Erros:
        ValueError se ``entry`` contiver NaN/infinito (``allow_nan=False``), para o
        arquivo continuar JSON válido.
    """
    path = curve_path(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Modo "a" (append): nunca reescreve as linhas anteriores.
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")


def _watch_stdin() -> None:
    """Encerra o worker quando o processo do treino some (o pipe de stdin fecha).

    Parâmetros: nenhum.

    Retorna: None. Inicia uma thread daemon que bloqueia lendo o stdin; quando a
    leitura termina (pipe fechado), encerra o processo inteiro com código 3.
    """
    def wait_for_parent():
        """Bloqueia até o stdin fechar e então mata o processo (os._exit(3), sem limpeza)."""
        try:
            # read() só retorna no fim do arquivo, ou seja, quando o treino fecha o pipe.
            sys.stdin.buffer.read()
        except (OSError, ValueError):
            pass
        print("gpu-eval-worker: treino encerrado; saindo.", flush=True)
        # os._exit encerra na hora, mesmo com a thread principal ocupada numa avaliação.
        os._exit(3)

    threading.Thread(target=wait_for_parent, daemon=True).start()


def _watch_parent(pid: int) -> None:
    """Windows: encerra o worker quando o processo do treino (``pid``) termina.

    ATENCAO: substitui ``_watch_stdin`` no Windows. Uma thread parada em
    ``sys.stdin.buffer.read()`` sobre um pipe síncrono trava outras chamadas no mesmo
    handle (o Windows serializa a E/S do arquivo), e os imports de torch/numpy consultam
    o stdin: o worker do treino de 13/10 ficou travado na partida e a GPU parada. Aqui a
    thread espera o próprio processo do treino (``WaitForSingleObject``), sem tocar no stdin.

    Parâmetros: pid do processo do treino (``os.getpid()`` em ``start_worker``).
    Retorna: None. Sem acesso ao processo (já terminou), encerra na hora com código 3.
    """
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    handle = kernel32.OpenProcess(0x00100000, False, int(pid))  # SYNCHRONIZE
    if not handle:
        print("gpu-eval-worker: treino não encontrado; saindo.", flush=True)
        os._exit(3)

    def wait_for_parent():
        """Bloqueia até o treino terminar e então encerra o worker (os._exit(3))."""
        kernel32.WaitForSingleObject(ctypes.c_void_p(handle), 0xFFFFFFFF)  # INFINITE
        print("gpu-eval-worker: treino encerrado; saindo.", flush=True)
        os._exit(3)

    threading.Thread(target=wait_for_parent, daemon=True).start()


# ENTRADA: comando interno gpu-eval-worker (cli.py), iniciado por start_worker.
def run_worker(config: ProjectConfig, run: Path, data_path: Path, provider: str, split: str, imgsz: int,
               work: Path | None = None, poll_seconds: float = 5.0, evaluator=evaluate_checkpoint,
               max_idle_seconds: float | None = None, watch_stdin: bool = False,
               parent_pid: int | None = None) -> dict:
    """Laço do worker: avalia cada checkpoint da fila até DONE com a fila vazia.

    Sempre pega o checkpoint mais recente; os mais antigos ainda na fila são
    descartados como "superseded" (a curva registra quais), para a GPU não
    ficar atrasada em relação ao treino. ``evaluator`` é trocável nos testes;
    ``max_idle_seconds`` encerra um worker sem checkpoint novo.

    Parâmetros:
        config: configuração do projeto.
        run: pasta do treino.
        data_path: ``dataset.yaml`` do dataset piloto.
        provider: ``"directml"`` ou ``"cpu"``.
        split: ``"val"`` ou ``"test"``.
        imgsz: tamanho de entrada do ONNX (px).
        work: pasta da fila; padrão ``work_directory(run)``.
        poll_seconds: espera entre consultas à fila vazia (s).
        evaluator: função com a assinatura de ``evaluate_checkpoint``.
        max_idle_seconds: encerra após esse tempo sem checkpoint (None = espera para sempre).
        watch_stdin: True liga ``_watch_stdin`` (usado quando iniciado por ``start_worker``).
        parent_pid: pid do treino; no Windows, com ele o worker observa o processo
            (``_watch_parent``) em vez do stdin, que travava a partida do worker.

    Retorna:
        ``{"processed": n, "status": "done" | "failed" | "idle_timeout"}``.
    """
    run = Path(run).resolve()
    work = Path(work).resolve() if work else work_directory(run)
    (work / "queue").mkdir(parents=True, exist_ok=True)
    if parent_pid and os.name == "nt":
        _watch_parent(parent_pid)
    elif watch_stdin:
        _watch_stdin()
    # Só limita threads quando o avaliador é o real (os testes usam um falso, sem torch).
    if evaluator is evaluate_checkpoint:
        limit_worker_threads()
    processed, consecutive_errors, last_activity = 0, 0, time.monotonic()
    print(f"gpu-eval-worker: fila {work / 'queue'} (provider={provider}, split={split}, imgsz={imgsz})", flush=True)
    while True:
        pending = pending_checkpoints(work)
        # 1. Há checkpoints: avalia só o MAIS RECENTE e descarta os anteriores.
        if pending:
            checkpoint = pending[-1]
            # 1a. Os mais antigos viram linhas "superseded" na curva e são apagados.
            #     "path=stale" fixa o valor no lambda (senão todos apagariam o último stale).
            for stale in pending[:-1]:
                _append(run, {"epoch": _epoch_of(stale), "split": split, "superseded_by": _epoch_of(checkpoint),
                              "error": "superseded: o worker estava atrasado e pulou esta época"})
                _retry(lambda path=stale: path.unlink(missing_ok=True))
            # 1b. Avalia; uma falha vira linha de erro na curva, sem derrubar o worker.
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
            # 1c. Registra, apaga o checkpoint avaliado e confere o limite de erros seguidos.
            _append(run, entry)
            _retry(lambda: checkpoint.unlink(missing_ok=True))
            processed += 1
            last_activity = time.monotonic()
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print("gpu-eval-worker: erros seguidos demais; encerrando.", flush=True)
                return {"processed": processed, "status": "failed"}
            continue
        # 2. Fila vazia e o treino marcou DONE: fim normal.
        if (folder(run) / DONE).is_file():
            if pending_checkpoints(work):  # enfileirado entre a varredura e o DONE
                continue
            return {"processed": processed, "status": "done"}
        # 3. Fila vazia por tempo demais: encerra (worker esquecido).
        if max_idle_seconds is not None and time.monotonic() - last_activity > max_idle_seconds:
            print("gpu-eval-worker: sem checkpoints novos; encerrando.", flush=True)
            return {"processed": processed, "status": "idle_timeout"}
        # 4. Espera e consulta a fila de novo.
        time.sleep(poll_seconds)
