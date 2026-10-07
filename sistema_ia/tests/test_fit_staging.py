"""Área de gravação do treino (DRONECAMP_FIT_STAGING): espelho do results.csv e cópia final para runs/."""

from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronecamp_ia.training import _copy_back_fit, _fit_staging, _mirror_results  # noqa: E402


class FitStagingTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="dronecamp staging á ")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.run = self.base / "E" / "runs" / "pilot_train_20261007T000000Z_abcdef12"
        self.run.mkdir(parents=True)

    def test_without_variable_training_writes_to_run(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DRONECAMP_FIT_STAGING", None)
            self.assertIsNone(_fit_staging(self.run))

    def test_results_are_mirrored_and_weights_copied_back(self) -> None:
        with mock.patch.dict(os.environ, {"DRONECAMP_FIT_STAGING": str(self.base / "C")}):
            staging = _fit_staging(self.run)
        self.assertEqual(staging, (self.base / "C" / self.run.name / "fit").resolve())
        (staging / "weights").mkdir(parents=True)
        (staging / "results.csv").write_text("epoch\n1\n", encoding="utf-8")
        _mirror_results(staging, self.run)
        self.assertEqual((self.run / "fit" / "results.csv").read_text(encoding="utf-8"), "epoch\n1\n")
        (staging / "weights" / "best.pt").write_bytes(b"pesos")
        _copy_back_fit(staging, self.run)
        self.assertEqual((self.run / "fit" / "weights" / "best.pt").read_bytes(), b"pesos")


if __name__ == "__main__":
    unittest.main()
