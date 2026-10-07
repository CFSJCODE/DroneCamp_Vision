"""Worker de avaliação na GPU (gpu_eval.py): fila, laço do worker e resumo, sem GPU nem modelo."""

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

REAL = load_config()


class SettingsTests(unittest.TestCase):
    def test_disabled_by_default_and_overrides_enable(self):
        self.assertIsNone(gpu.gpu_eval_settings(REAL))
        settings = gpu.gpu_eval_settings(REAL, {"enabled": True, "every": None, "provider": "cpu", "split": None})
        self.assertEqual((settings["enabled"], settings["every"], settings["provider"], settings["split"]), (True, 5, "cpu", "val"))

    def test_invalid_values_are_rejected(self):
        for bad in ({"every": 0}, {"provider": "cuda"}, {"split": "train"}, {"timeout": -1}, {"extra": 1}):
            with self.assertRaises(ValueError, msg=str(bad)):
                gpu.gpu_eval_settings({"enabled": True, **bad})


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.run = Path(self.temporary.name) / "runs" / "pilot_train_x"
        self.run.mkdir(parents=True)
        self.work = gpu.work_directory(self.run)
        self.checkpoint = self.run / "last.pt"
        self.checkpoint.write_bytes(b"pesos")

    def tearDown(self):
        self.temporary.cleanup()

    def test_enqueue_is_atomic_and_sorted_by_epoch(self):
        gpu.enqueue_checkpoint(self.work, self.checkpoint, 10)
        gpu.enqueue_checkpoint(self.work, self.checkpoint, 5)
        pending = gpu.pending_checkpoints(self.work)
        self.assertEqual([path.name for path in pending], ["epoch_0005.pt", "epoch_0010.pt"])
        self.assertFalse(list((self.work / "queue").glob("*.tmp")))
        self.assertEqual(pending[0].read_bytes(), b"pesos")

    def test_worker_evaluates_newest_marks_older_superseded_and_stops_on_done(self):
        gpu.enqueue_checkpoint(self.work, self.checkpoint, 5)
        gpu.enqueue_checkpoint(self.work, self.checkpoint, 10)
        seen = []

        def evaluator(config, run, work, data_path, checkpoint, provider, split, imgsz):
            seen.append(checkpoint.name)
            if len(seen) == 1:
                threading.Timer(0.2, lambda: gpu.enqueue_checkpoint(work, self.checkpoint, 15)).start()
            if len(seen) == 2:
                (gpu.folder(run) / gpu.DONE).write_text("x", encoding="utf-8")
            return {"epoch": gpu._epoch_of(checkpoint), "split": split, "provider": provider, "map50": 0.1 * len(seen),
                    "seconds": 1.0, "imgsz": imgsz}

        gpu.folder(self.run).mkdir(parents=True, exist_ok=True)
        result = gpu.run_worker(REAL, self.run, Path("dataset.yaml"), "cpu", "val", 640, self.work, poll_seconds=0.05,
                                evaluator=evaluator, max_idle_seconds=5)
        self.assertEqual(result["status"], "done")
        self.assertEqual(seen, ["epoch_0010.pt", "epoch_0015.pt"])
        curve = gpu.read_curve(self.run)
        self.assertEqual([entry["epoch"] for entry in curve], [10, 15])
        errors = [entry for entry in gpu.read_curve(self.run, include_errors=True) if "error" in entry]
        self.assertEqual([entry["epoch"] for entry in errors], [5])
        self.assertIn("superseded", errors[0]["error"])
        self.assertEqual(gpu.pending_checkpoints(self.work), [])

    def test_worker_records_failures_and_gives_up_after_consecutive_errors(self):
        for epoch in (1, 2, 3):
            gpu.enqueue_checkpoint(self.work, self.checkpoint, epoch)

        def evaluator(*args, **kwargs):
            raise RuntimeError("sem DirectML")

        result = gpu.run_worker(REAL, self.run, Path("dataset.yaml"), "cpu", "val", 640, self.work, poll_seconds=0.05,
                                evaluator=evaluator, max_idle_seconds=1)
        # O mais novo é avaliado primeiro (os outros viram superseded); três erros seguidos exigem mais checkpoints.
        self.assertIn(result["status"], {"failed", "idle_timeout"})
        entries = gpu.read_curve(self.run, include_errors=True)
        self.assertTrue(any("RuntimeError" in entry.get("error", "") for entry in entries))
        self.assertEqual(gpu.read_curve(self.run), [])

    def test_finish_worker_terminates_a_slow_process_and_summarizes(self):
        (self.run / "fit").mkdir()
        (self.run / "fit" / "results.csv").write_text("epoch,time\n1,100\n2,250\n", encoding="utf-8")
        gpu.folder(self.run).mkdir(parents=True, exist_ok=True)
        gpu._append(self.run, {"epoch": 5, "split": "val", "provider": "cpu", "map50": 0.2, "seconds": 30.0,
                               "export_seconds": 10.0, "inference_seconds": 20.0, "provider_active": "CPUExecutionProvider"})
        gpu._append(self.run, {"epoch": 10, "split": "val", "provider": "cpu", "map50": 0.3, "seconds": 20.0})
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], stdin=subprocess.PIPE)
        started = time.monotonic()
        summary = gpu.finish_worker(self.run, process, timeout=0.5)
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
        summary = gpu.finish_worker(self.run, None, 0)
        self.assertEqual(summary["entries"], 0)
        self.assertNotIn("error", summary)
        self.assertTrue((gpu.folder(self.run) / gpu.DONE).is_file())


if __name__ == "__main__":
    unittest.main()
