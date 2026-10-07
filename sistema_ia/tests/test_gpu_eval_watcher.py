"""Vigia de avaliação na GPU: acha os pesos da área de gravação e acompanha o treino pelos arquivos do run."""

from pathlib import Path
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import gpu_eval_watcher as watcher  # noqa: E402


class WatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="dronecamp vigia á ")
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name) / "runs" / "pilot_train_x"
        (self.run / "fit").mkdir(parents=True)

    def test_weights_follow_staging_when_declared(self) -> None:
        self.assertEqual(watcher.weights_directory(self.run), self.run / "fit" / "weights")
        (self.run / "execution.json").write_text(json.dumps({"fit_staging": r"C:\DroneCamp_runs\x\fit"}), encoding="utf-8")
        self.assertEqual(watcher.weights_directory(self.run), Path(r"C:\DroneCamp_runs\x\fit") / "weights")

    def test_epochs_and_state_come_from_run_files(self) -> None:
        self.assertEqual(watcher.epochs_done(self.run), 0)
        self.assertEqual(watcher.training_state(self.run), "running", "Sem summary, o treino ainda está começando.")
        (self.run / "fit" / "results.csv").write_text("epoch,time\n1,10\n2,20\n3,30\n", encoding="utf-8")
        self.assertEqual(watcher.epochs_done(self.run), 3)
        (self.run / "summary.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")
        self.assertEqual(watcher.training_state(self.run), "completed")


if __name__ == "__main__":
    unittest.main()
