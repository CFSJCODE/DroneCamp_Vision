"""Worker de avaliação na GPU (gpu_eval.py): fila, laço do worker e resumo, sem GPU nem modelo.

Módulo coberto
--------------
``dronecamp_ia.gpu_eval`` (importado como ``gpu``): ``gpu_eval_settings`` (valida a seção
``gpu_eval`` do project.yaml e os overrides da CLI), ``work_directory``, ``enqueue_checkpoint``
e ``pending_checkpoints`` (fila de checkpoints por época), ``run_worker`` (laço que avalia
o checkpoint mais novo), ``read_curve``, ``finish_worker`` (encerra o processo e resume),
``folder``, ``DONE`` e os auxiliares privados ``_epoch_of`` e ``_append``.

Como rodar (na pasta ``sistema_ia``)
------------------------------------
- Só este arquivo::

      DRONECAMP_SKIP_E2E=1 python -m unittest tests.test_gpu_eval -v

- Um teste só::

      python -m unittest tests.test_gpu_eval.QueueTests.test_enqueue_is_atomic_and_sorted_by_epoch -v

O que cada classe prepara
-------------------------
- ``SettingsTests``: nada no disco; só dicts de configuração.
- ``QueueTests``: pasta temporária com ``runs/pilot_train_x`` (o run), a pasta de trabalho
  do worker (``gpu.work_directory``) e um checkpoint falso ``last.pt`` com o conteúdo
  ``b"pesos"``. O avaliador real (ONNX/DirectML) é trocado por uma função ``evaluator``
  local, então não há GPU, modelo nem ONNX.

EXTENSAO: para testar uma função nova do worker (ex.: uma métrica nova no resumo de
``finish_worker``), copie ``test_finish_worker_terminates_a_slow_process_and_summarizes``,
grave as entradas com ``gpu._append(self.run, {...})`` incluindo a chave nova e afirme o
campo no ``summary``. Para um parâmetro novo em ``gpu_eval_settings``, acrescente o caso
inválido na tupla de ``test_invalid_values_are_rejected`` e o valor padrão em
``test_disabled_by_default_and_overrides_enable``.
"""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time
import unittest

from dronecamp_ia import gpu_eval as gpu
from dronecamp_ia.config import load_config

# Config real do projeto (configs/project.yaml), só leitura; run_worker exige um ProjectConfig.
REAL = load_config()


# ---------------------------------------------------------------------------
# gpu_eval_settings: validação da configuração
# ---------------------------------------------------------------------------

class SettingsTests(unittest.TestCase):
    """Validação e valores padrão de ``gpu.gpu_eval_settings``."""

    def test_disabled_by_default_and_overrides_enable(self):
        """Sem seção ou com ``enabled: False`` retorna None; overrides ligam e ``None`` vira padrão.

        Padrões conferidos: ``every`` = 5 épocas e ``split`` = "val".
        """
        self.assertIsNone(gpu.gpu_eval_settings(None))
        self.assertIsNone(gpu.gpu_eval_settings({"enabled": False}))
        # 2º argumento = overrides (ex.: opções da CLI); valores None não sobrescrevem o padrão.
        settings = gpu.gpu_eval_settings({"enabled": False}, {"enabled": True, "every": None, "provider": "cpu", "split": None})
        self.assertEqual((settings["enabled"], settings["every"], settings["provider"], settings["split"]), (True, 5, "cpu", "val"))

    def test_invalid_values_are_rejected(self):
        """Valores fora do permitido lançam ValueError: every 0, provider cuda, split train, timeout negativo, chave desconhecida."""
        # EXTENSAO: nova regra de validação em gpu_eval_settings: acrescente o dict inválido nesta tupla.
        for bad in ({"every": 0}, {"provider": "cuda"}, {"split": "train"}, {"timeout": -1}, {"extra": 1}):
            with self.assertRaises(ValueError, msg=str(bad)):
                gpu.gpu_eval_settings({"enabled": True, **bad})


# ---------------------------------------------------------------------------
# Fila de checkpoints, laço do worker e encerramento
# ---------------------------------------------------------------------------

class QueueTests(unittest.TestCase):
    """Fila de checkpoints, ``run_worker`` com avaliador falso e ``finish_worker``."""

    def setUp(self):
        """Cria o run temporário, a pasta de trabalho e o checkpoint falso ``last.pt``."""
        self.temporary = TemporaryDirectory()
        # Pasta do treino, no mesmo formato de runs/ do projeto.
        self.run = Path(self.temporary.name) / "runs" / "pilot_train_x"
        self.run.mkdir(parents=True)
        # Pasta onde o worker guarda a fila (queue/) e cópias dos checkpoints.
        self.work = gpu.work_directory(self.run)
        # Checkpoint falso: o conteúdo só serve para conferir a cópia.
        self.checkpoint = self.run / "last.pt"
        self.checkpoint.write_bytes(b"pesos")

    def tearDown(self):
        """Apaga a pasta temporária."""
        self.temporary.cleanup()

    def test_enqueue_is_atomic_and_sorted_by_epoch(self):
        """A fila fica ordenada por época, sem ``.tmp`` sobrando, e com a cópia íntegra do checkpoint.

        Importa porque o worker lê a fila enquanto o treino grava: um arquivo pela metade
        não pode aparecer como pendente.
        """
        # Enfileira fora de ordem (10 antes de 5).
        gpu.enqueue_checkpoint(self.work, self.checkpoint, 10)
        gpu.enqueue_checkpoint(self.work, self.checkpoint, 5)
        pending = gpu.pending_checkpoints(self.work)
        # Nome com época em 4 dígitos (epoch_0005.pt) para a ordem alfabética ser a numérica.
        self.assertEqual([path.name for path in pending], ["epoch_0005.pt", "epoch_0010.pt"])
        self.assertFalse(list((self.work / "queue").glob("*.tmp")))
        self.assertEqual(pending[0].read_bytes(), b"pesos")

    def test_worker_evaluates_newest_marks_older_superseded_and_stops_on_done(self):
        """O worker avalia só o mais novo, marca os antigos como "superseded" e para ao ver ``DONE``.

        Sequência: fila 5 e 10 → avalia 10 (5 vira superseded) → durante a avaliação chega
        a 15 → avalia 15 → o avaliador grava DONE → worker termina com status "done".
        """
        # 1. Fila inicial com as épocas 5 e 10.
        gpu.enqueue_checkpoint(self.work, self.checkpoint, 5)
        gpu.enqueue_checkpoint(self.work, self.checkpoint, 10)
        seen = []

        def evaluator(config, run, work, data_path, checkpoint, provider, split, imgsz):
            """Avaliador falso com a mesma assinatura do real; registra o nome do checkpoint.

            Retorna:
                dict de entrada da curva (epoch, split, provider, map50, seconds, imgsz).
            """
            seen.append(checkpoint.name)
            # 1ª avaliação: agenda a chegada da época 15 daqui a 0,2 s (simula o treino seguindo).
            if len(seen) == 1:
                threading.Timer(0.2, lambda: gpu.enqueue_checkpoint(work, self.checkpoint, 15)).start()
            # 2ª avaliação: grava o marcador DONE (simula o fim do treino).
            if len(seen) == 2:
                (gpu.folder(run) / gpu.DONE).write_text("x", encoding="utf-8")
            return {"epoch": gpu._epoch_of(checkpoint), "split": split, "provider": provider, "map50": 0.1 * len(seen),
                    "seconds": 1.0, "imgsz": imgsz}

        # 2. Roda o laço do worker com sondagem rápida (0,05 s) e limite de ociosidade de 5 s.
        gpu.folder(self.run).mkdir(parents=True, exist_ok=True)
        result = gpu.run_worker(REAL, self.run, Path("dataset.yaml"), "cpu", "val", 640, self.work, poll_seconds=0.05,
                                evaluator=evaluator, max_idle_seconds=5)
        # 3. Confere ordem avaliada, curva (só sucessos), erros (5 superseded) e fila vazia.
        self.assertEqual(result["status"], "done")
        self.assertEqual(seen, ["epoch_0010.pt", "epoch_0015.pt"])
        curve = gpu.read_curve(self.run)
        self.assertEqual([entry["epoch"] for entry in curve], [10, 15])
        errors = [entry for entry in gpu.read_curve(self.run, include_errors=True) if "error" in entry]
        self.assertEqual([entry["epoch"] for entry in errors], [5])
        self.assertIn("superseded", errors[0]["error"])
        self.assertEqual(gpu.pending_checkpoints(self.work), [])

    def test_worker_records_failures_and_gives_up_after_consecutive_errors(self):
        """Erros do avaliador são gravados na curva (com o tipo da exceção) e o worker desiste.

        A curva "limpa" (sem erros) fica vazia; o status final é "failed" ou "idle_timeout".
        """
        for epoch in (1, 2, 3):
            gpu.enqueue_checkpoint(self.work, self.checkpoint, epoch)

        def evaluator(*args, **kwargs):
            """Avaliador falso que sempre falha (simula DirectML ausente).

            Erros:
                RuntimeError: sempre.
            """
            raise RuntimeError("sem DirectML")

        result = gpu.run_worker(REAL, self.run, Path("dataset.yaml"), "cpu", "val", 640, self.work, poll_seconds=0.05,
                                evaluator=evaluator, max_idle_seconds=1)
        # O mais novo é avaliado primeiro (os outros viram superseded); três erros seguidos exigem mais checkpoints.
        self.assertIn(result["status"], {"failed", "idle_timeout"})
        entries = gpu.read_curve(self.run, include_errors=True)
        self.assertTrue(any("RuntimeError" in entry.get("error", "") for entry in entries))
        self.assertEqual(gpu.read_curve(self.run), [])

    def test_finish_worker_terminates_a_slow_process_and_summarizes(self):
        """``finish_worker`` encerra um processo que não termina no prazo e grava o resumo.

        Resumo conferido: 2 entradas, melhor época por mAP50 = 10, tempo total de avaliação
        50 s (30 + 20), tempo de treino 250 s (última linha do results.csv) e fração
        ocupada 50/250 = 0,2.
        """
        # 1. results.csv mínimo: a coluna "time" é cumulativa; a última linha (250) é o tempo de treino.
        (self.run / "fit").mkdir()
        (self.run / "fit" / "results.csv").write_text("epoch,time\n1,100\n2,250\n", encoding="utf-8")
        # 2. Duas entradas na curva (épocas 5 e 10).
        gpu.folder(self.run).mkdir(parents=True, exist_ok=True)
        gpu._append(self.run, {"epoch": 5, "split": "val", "provider": "cpu", "map50": 0.2, "seconds": 30.0,
                               "export_seconds": 10.0, "inference_seconds": 20.0, "provider_active": "CPUExecutionProvider"})
        gpu._append(self.run, {"epoch": 10, "split": "val", "provider": "cpu", "map50": 0.3, "seconds": 20.0})
        # 3. Processo "worker" que dorme 60 s; com timeout 0,5 s precisa ser encerrado à força.
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], stdin=subprocess.PIPE)
        started = time.monotonic()
        summary = gpu.finish_worker(self.run, process, timeout=0.5)
        # 4. Não esperou os 60 s, gravou DONE e registrou o erro de encerramento no resumo.
        self.assertLess(time.monotonic() - started, 40)
        self.assertTrue((gpu.folder(self.run) / gpu.DONE).is_file())
        self.assertIn("error", summary)
        self.assertEqual(summary["entries"], 2)
        self.assertEqual(summary["best_epoch_by_map50"]["epoch"], 10)
        self.assertEqual(summary["parallel"]["eval_total_seconds"], 50.0)
        self.assertEqual(summary["parallel"]["training_seconds"], 250.0)
        self.assertEqual(summary["parallel"]["gpu_busy_fraction"], 0.2)
        self.assertIn("CPUExecutionProvider", summary["parallel"]["providers"])

    def test_finish_worker_without_process_only_marks_done(self):
        """Sem processo (worker não foi iniciado), só grava ``DONE`` e um resumo vazio sem erro."""
        summary = gpu.finish_worker(self.run, None, 0)
        self.assertEqual(summary["entries"], 0)
        self.assertNotIn("error", summary)
        self.assertTrue((gpu.folder(self.run) / gpu.DONE).is_file())


# ENTRADA: execução direta do arquivo.
if __name__ == "__main__":
    unittest.main()
