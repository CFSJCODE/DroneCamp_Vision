"""Instrumentação leve do treino da Ultralytics: tempo por etapa, otimizador e recursos da máquina.

Função no projeto: explicar por que uma época passou de uma para duas horas e se
alguma atualização falhou, registrando DURANTE o treino o que o ``results.csv`` não
guarda. Custo proporcional: contadores e relógios por lote, um resumo por época e
uma amostra de recursos a cada ``resource_seconds`` (padrão 30 s) numa thread.

O que registra (``<run>/instrumentation/``):
- ``epochs.jsonl``, uma linha por época global: para cada etapa do lote
  (``data_wait`` = espera pelo carregador de dados, ``preprocess``, ``forward_loss``,
  ``backward``, ``optimizer_step``, ``batch``) a mediana, o p95, a média e a soma em
  segundos; tempo de validação e de gravação do checkpoint; lotes, imagens e passos
  efetivos do otimizador nesta época e acumulados no estado do otimizador; norma do
  gradiente antes do corte; perdas e gradientes não finitos; passos pulados pelo
  GradScaler; taxa de aprendizagem; métricas de validação e fitness.
- ``resources.jsonl``, uma linha por amostra: CPU do sistema e do processo, RAM
  usada/disponível, memória do processo (RSS e VMS; no Windows o VMS é o "commit"
  do processo), swap/arquivo de paginação, leitura/escrita de disco, espaço livre,
  rede e, com CUDA, memória da GPU.
- A ficha da tentativa (``run_record.py``): início, retrato do treinador e fim com
  motivo e contadores.

Como mede cada etapa (ordem do laço da Ultralytics 8.4.172, ``BaseTrainer._do_train``):
    [busca do lote] → on_train_batch_start → preprocess_batch → forward+perda →
    scaler.scale(perda).backward() → optimizer_step (a cada ``accumulate`` lotes) →
    on_train_batch_end
  ``data_wait`` = do fim do lote anterior até ``on_train_batch_start``;
  ``forward_loss`` termina quando ``scaler.scale`` é chamado (logo antes do backward);
  ``backward`` vai daí até o passo do otimizador (ou o fim do lote).
  Para isso, em ``on_train_start``, os métodos ``preprocess_batch``, ``optimizer_step``,
  ``validate``, ``save_model`` e ``scaler.scale`` do treinador são envolvidos por
  funções que medem o tempo e chamam o original.

ATENCAO: esses pontos dependem dos internos da Ultralytics fixada no pyproject
(8.4.172). Se uma versão nova mudar o laço, os envoltórios continuam chamando o
original (o treino não muda), mas os tempos podem sair errados: confira com um
treino curto. Qualquer erro da instrumentação é engolido e registrado; ela nunca
interrompe o treino.

Quem chama: ``training.train_pilot`` e ``training.resume_pilot``.

Onde mexer:
- Ligar/desligar e intervalo: ``pilot.instrumentation`` no ``configs/project.yaml``
  (``enabled``, ``resource_seconds``, ``grad_norm``).
- Etapa nova: acrescente o nome em ``STAGES`` e meça no envoltório correspondente.
- Profiler detalhado (``torch.profiler``): EXTENSAO em ``TrainingMonitor.on_train_batch_start``;
  use só em janelas curtas (poucos lotes), pois o próprio profiler aumenta tempo e memória.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import threading
import time

from .run_record import end_attempt, trainer_snapshot

# Etapas de um lote medidas a cada iteração (segundos).
STAGES = ("data_wait", "preprocess", "forward_loss", "backward", "optimizer_step", "batch")
# Padrões de pilot.instrumentation.
DEFAULTS = {"enabled": True, "resource_seconds": 30, "grad_norm": True}


def _stats(values: list[float]) -> dict:
    """Mediana, p95, média e soma de uma lista de durações (vazia → só n=0)."""
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        low, high = math.floor(position), math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {"n": len(ordered), "median": round(at(0.5), 4), "p95": round(at(0.95), 4),
            "mean": round(sum(ordered) / len(ordered), 4), "sum": round(sum(ordered), 2)}


def _append(path: Path, row: dict) -> None:
    """Acrescenta uma linha JSON (falha de disco é ignorada: registro não para o treino)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass


class ResourceSampler(threading.Thread):
    """Thread que grava uma amostra de recursos a cada ``seconds`` e guarda os picos.

    Parâmetros: ``path`` (resources.jsonl), ``seconds`` (intervalo), ``disk_path`` (pasta
    cujo espaço livre é medido), ``attempt`` (número da tentativa, gravado em cada linha).
    """

    def __init__(self, path: Path, seconds: float, disk_path: Path, attempt: int):
        super().__init__(daemon=True, name="dronecamp-resource-sampler")
        self.path, self.seconds, self.disk_path, self.attempt = Path(path), float(seconds), Path(disk_path), attempt
        self.stop_event = threading.Event()
        self.peaks: dict[str, float] = {}
        self.samples = 0

    def sample(self, previous: dict | None) -> tuple[dict, dict]:
        """Uma amostra; ``previous`` guarda os contadores de disco/rede para calcular taxas."""
        import psutil

        now = time.time()
        process = psutil.Process()
        memory, swap = psutil.virtual_memory(), psutil.swap_memory()
        info = process.memory_info()
        row = {"t": datetime.now(timezone.utc).isoformat(), "attempt": self.attempt,
               "cpu_percent": psutil.cpu_percent(interval=None),
               "process_cpu_percent": process.cpu_percent(interval=None),
               "ram_used_gb": round((memory.total - memory.available) / 1e9, 3),
               "ram_available_gb": round(memory.available / 1e9, 3), "ram_percent": memory.percent,
               "process_rss_gb": round(info.rss / 1e9, 3), "process_vms_gb": round(info.vms / 1e9, 3),
               "swap_used_gb": round(swap.used / 1e9, 3), "threads": process.num_threads()}
        try:
            row["disk_free_gb"] = round(shutil.disk_usage(self.disk_path).free / 1e9, 2)
        except OSError:
            pass
        counters = {"t": now}
        try:
            disk = psutil.disk_io_counters()
            counters.update({"read": disk.read_bytes, "write": disk.write_bytes})
        except Exception:  # noqa: BLE001 - alguns sistemas não expõem contadores de disco
            pass
        try:
            net = psutil.net_io_counters()
            counters.update({"sent": net.bytes_sent, "recv": net.bytes_recv})
        except Exception:  # noqa: BLE001
            pass
        if previous:
            elapsed = max(now - previous["t"], 1e-6)
            for key, name in (("read", "disk_read_mb_s"), ("write", "disk_write_mb_s"),
                              ("sent", "net_sent_mb_s"), ("recv", "net_recv_mb_s")):
                if key in counters and key in previous:
                    row[name] = round((counters[key] - previous[key]) / elapsed / 1e6, 3)
        try:
            import torch

            if torch.cuda.is_available():
                row["gpu_mem_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
                row["gpu_mem_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 3)
                row["gpu_mem_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        except Exception:  # noqa: BLE001
            pass
        return row, counters

    def run(self) -> None:
        """Laço da thread: amostra, grava, atualiza picos, espera."""
        previous = None
        while not self.stop_event.is_set():
            try:
                row, previous = self.sample(previous)
                _append(self.path, row)
                self.samples += 1
                for key, value in row.items():
                    if isinstance(value, (int, float)) and key != "attempt":
                        self.peaks[key] = max(self.peaks.get(key, value), value)
            except Exception:  # noqa: BLE001 - sem psutil ou erro do sistema: para de amostrar
                return
            self.stop_event.wait(self.seconds)


class TrainingMonitor:
    """Coleta por lote e por época durante ``model.train`` (callbacks da Ultralytics).

    Parâmetros:
        run: pasta do treino.
        attempt: descritor de ``run_record.start_attempt``.
        settings: ``pilot.instrumentation`` (``DEFAULTS`` completa o que faltar).

    Uso: ``monitor.attach(model)`` antes de ``model.train`` e ``monitor.close(status, motivo)``
    num ``finally``.
    """

    def __init__(self, run: Path, attempt: dict, settings: dict | None = None):
        self.run, self.attempt = Path(run), attempt
        self.settings = {**DEFAULTS, **(settings or {})}
        self.folder = self.run / "instrumentation"
        self.sampler: ResourceSampler | None = None
        self.errors: list[str] = []
        self.totals = {"batches": 0, "images": 0, "optimizer_steps": 0, "nonfinite_loss": 0, "nonfinite_grad": 0,
                       "skipped_steps": 0, "effective_seconds": 0.0, "epochs_completed": 0}
        self.global_epoch = None
        self._reset_epoch()

    # -- estado por época e por lote ------------------------------------------------
    def _reset_epoch(self) -> None:
        """Zera os acumuladores da época."""
        self.stage = {name: [] for name in STAGES}
        self.epoch_counts = {"batches": 0, "images": 0, "optimizer_steps": 0, "nonfinite_loss": 0,
                             "nonfinite_grad": 0, "skipped_steps": 0}
        self.grad_norms: list[float] = []
        self.validation_seconds = 0.0
        self.checkpoint_seconds = 0.0
        self.epoch_start = time.perf_counter()
        self.last_batch_end = self.epoch_start
        self.marks: dict[str, float] = {}
        self.epoch_open = True

    def _safe(self, function, *args) -> None:
        """Roda um callback de registro sem deixar exceção chegar à Ultralytics."""
        try:
            function(*args)
        except Exception as error:  # noqa: BLE001
            if len(self.errors) < 20:
                self.errors.append(f"{getattr(function, '__name__', 'callback')}: {type(error).__name__}: {error}")

    def attach(self, model) -> None:
        """Registra os callbacks no modelo da Ultralytics (antes de ``model.train``)."""
        for event, function in (("on_train_start", self.on_train_start),
                                ("on_train_epoch_start", self.on_train_epoch_start),
                                ("on_train_batch_start", self.on_train_batch_start),
                                ("on_train_batch_end", self.on_train_batch_end),
                                ("on_fit_epoch_end", self.on_fit_epoch_end)):
            model.add_callback(event, lambda trainer, function=function: self._safe(function, trainer))

    # -- envoltórios dos métodos do treinador --------------------------------------
    def _wrap(self, trainer) -> None:
        """Envolve preprocess_batch, scaler.scale, optimizer_step, validate e save_model (ver cabeçalho)."""
        monitor = self
        clock = time.perf_counter

        original_preprocess = trainer.preprocess_batch

        def preprocess_batch(batch):
            started = clock()
            result = original_preprocess(batch)
            monitor.marks["preprocess"] = clock() - started
            monitor.marks["preprocess_end"] = clock()
            try:
                monitor.marks["images"] = int(result["img"].shape[0])
            except Exception:  # noqa: BLE001
                monitor.marks["images"] = 0
            return result

        trainer.preprocess_batch = preprocess_batch

        scaler = getattr(trainer, "scaler", None)
        if scaler is not None:
            original_scale = scaler.scale

            def scale(outputs):
                monitor.marks["forward_end"] = clock()
                return original_scale(outputs)

            scaler.scale = scale

        original_step = trainer.optimizer_step

        def optimizer_step():
            started = clock()
            monitor.marks["optimizer_start"] = started
            if monitor.settings["grad_norm"]:
                monitor._safe(monitor._grad_norm, trainer)
            before = scaler.get_scale() if scaler is not None and scaler.is_enabled() else None
            original_step()
            if before is not None and scaler.get_scale() < before:
                # GradScaler reduziu a escala: gradiente inf/NaN, o passo foi pulado.
                monitor.epoch_counts["skipped_steps"] += 1
            else:
                monitor.epoch_counts["optimizer_steps"] += 1
            monitor.marks["optimizer_step"] = clock() - started

        trainer.optimizer_step = optimizer_step

        original_validate = trainer.validate

        def validate():
            started = clock()
            try:
                return original_validate()
            finally:
                monitor.validation_seconds += clock() - started

        trainer.validate = validate

        original_save = trainer.save_model

        def save_model():
            started = clock()
            try:
                return original_save()
            finally:
                monitor.checkpoint_seconds += clock() - started

        trainer.save_model = save_model

    def _grad_norm(self, trainer) -> None:
        """Norma L2 total do gradiente antes do corte (desfeita a escala do GradScaler)."""
        import torch

        grads = [param.grad.detach() for param in trainer.model.parameters() if param.grad is not None]
        if not grads:
            return
        norm = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float(), 2) for g in grads]), 2))
        scaler = getattr(trainer, "scaler", None)
        if scaler is not None and scaler.is_enabled():
            norm /= max(float(scaler.get_scale()), 1e-12)
        if math.isfinite(norm):
            self.grad_norms.append(norm)
        else:
            self.epoch_counts["nonfinite_grad"] += 1

    # -- callbacks -----------------------------------------------------------------
    def on_train_start(self, trainer) -> None:
        """Envolve os métodos, grava o retrato do treinador e inicia a amostragem de recursos."""
        self._wrap(trainer)
        trainer_snapshot(self.attempt, trainer)
        if self.settings["resource_seconds"] and float(self.settings["resource_seconds"]) > 0:
            self.sampler = ResourceSampler(self.folder / "resources.jsonl", float(self.settings["resource_seconds"]),
                                           self.run, int(self.attempt["attempt"]))
            self.sampler.start()

    def on_train_epoch_start(self, trainer) -> None:
        """Começo de época: zera os acumuladores."""
        self._reset_epoch()

    def on_train_batch_start(self, trainer) -> None:
        """Começo de lote: o intervalo desde o fim do anterior é a espera pelos dados."""
        # EXTENSAO: profiler em janela curta = iniciar torch.profiler aqui nos N primeiros lotes da época.
        now = time.perf_counter()
        self.stage["data_wait"].append(now - self.last_batch_end)
        self.marks = {"batch_start": now}

    def on_train_batch_end(self, trainer) -> None:
        """Fim de lote: divide o tempo entre as etapas e confere a perda."""
        now = time.perf_counter()
        marks = self.marks
        start = marks.get("batch_start", now)
        preprocess_end = marks.get("preprocess_end", start)
        forward_end = marks.get("forward_end", preprocess_end)
        backward_end = marks.get("optimizer_start", now)
        self.stage["preprocess"].append(marks.get("preprocess", 0.0))
        self.stage["forward_loss"].append(max(0.0, forward_end - preprocess_end))
        self.stage["backward"].append(max(0.0, backward_end - forward_end))
        if "optimizer_step" in marks:
            self.stage["optimizer_step"].append(marks["optimizer_step"])
        self.stage["batch"].append(now - start)
        self.epoch_counts["batches"] += 1
        self.epoch_counts["images"] += int(marks.get("images", 0))
        loss = getattr(trainer, "loss", None)
        if loss is not None:
            try:
                if not bool(loss.isfinite().all()):
                    self.epoch_counts["nonfinite_loss"] += 1
            except Exception:  # noqa: BLE001
                pass
        self.last_batch_end = now

    def on_fit_epoch_end(self, trainer) -> None:
        """Fim da época (depois da validação e do checkpoint): grava a linha de ``epochs.jsonl``."""
        metrics = {key: (float(value) if isinstance(value, (int, float)) else None)
                   for key, value in (getattr(trainer, "metrics", None) or {}).items()}
        # ATENCAO: a Ultralytics chama on_fit_epoch_end de novo na validação final do best.pt
        # (final_eval, com trainer.epoch + 1). Sem época aberta, é essa chamada: vai para
        # final_eval.jsonl e não vira uma época a mais.
        if not self.epoch_open:
            _append(self.folder / "final_eval.jsonl", {
                "attempt": self.attempt["attempt"], "attempt_id": self.attempt["attempt_id"],
                "t": datetime.now(timezone.utc).isoformat(), "weights": str(getattr(trainer, "best", "")),
                "after_epoch": self.global_epoch, "metrics": metrics})
            return
        self.epoch_open = False
        epoch_seconds = time.perf_counter() - self.epoch_start
        self.global_epoch = int(trainer.epoch) + 1
        state_step = None
        try:
            for group in trainer.optimizer.param_groups:
                for param in group["params"]:
                    step = trainer.optimizer.state.get(param, {}).get("step")
                    if step is not None:
                        state_step = int(float(step))
                        break
                if state_step is not None:
                    break
        except Exception:  # noqa: BLE001
            pass
        row = {"epoch": self.global_epoch, "attempt": self.attempt["attempt"],
               "attempt_id": self.attempt["attempt_id"], "t": datetime.now(timezone.utc).isoformat(),
               "stages": {name: _stats(values) for name, values in self.stage.items()},
               **self.epoch_counts, "optimizer_state_step": state_step,
               "accumulate": int(getattr(trainer, "accumulate", 1)),
               "grad_norm_before_clip": _stats(self.grad_norms),
               "lr": {f"pg{index}": group.get("lr") for index, group in enumerate(trainer.optimizer.param_groups)},
               "validation_seconds": round(self.validation_seconds, 2),
               "checkpoint_seconds": round(self.checkpoint_seconds, 2), "epoch_seconds": round(epoch_seconds, 2),
               "metrics": metrics, "fitness": None if trainer.fitness is None else float(trainer.fitness),
               "best_fitness": None if trainer.best_fitness is None else float(trainer.best_fitness),
               "instrumentation_errors": self.errors[-3:]}
        _append(self.folder / "epochs.jsonl", row)
        for key in ("batches", "images", "optimizer_steps", "nonfinite_loss", "nonfinite_grad", "skipped_steps"):
            self.totals[key] += self.epoch_counts[key]
        self.totals["effective_seconds"] += epoch_seconds
        self.totals["epochs_completed"] += 1

    def close(self, status: str, reason: str | None) -> None:
        """Para a amostragem e grava o fim da tentativa na ficha (chamar num ``finally``)."""
        if self.sampler is not None:
            self.sampler.stop_event.set()
            self.sampler.join(timeout=5)
        end_attempt(self.attempt, status, reason, {
            "global_epoch_reached": self.global_epoch,
            **{key: (round(value, 1) if isinstance(value, float) else value) for key, value in self.totals.items()},
            "resource_peaks": self.sampler.peaks if self.sampler else {},
            "resource_samples": self.sampler.samples if self.sampler else 0,
            "instrumentation_errors": self.errors})


def monitor_settings(config) -> dict | None:
    """``pilot.instrumentation`` com os padrões; ``None`` quando desligada (``enabled: false``)."""
    settings = {**DEFAULTS, **((config.pilot.get("instrumentation") or {}) if hasattr(config, "pilot") else {})}
    return settings if settings.get("enabled", True) else None
