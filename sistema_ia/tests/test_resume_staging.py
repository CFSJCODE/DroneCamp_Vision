"""Retomada do treino piloto quando o last.pt ficou só na área de gravação.

Módulo coberto: ``dronecamp_ia.training.resume_pilot``. Cenário real: queda de energia
com ``DRONECAMP_FIT_STAGING`` definido; o ``fit`` não voltou para ``runs/`` e a retomada
precisa achar o checkpoint em ``<staging>/<run>/fit/weights/last.pt``. Sem Ultralytics:
``load_detector`` é trocado por um modelo falso que só registra o caminho e os callbacks.

Como rodar (na pasta ``sistema_ia``)::

    python -m unittest tests.test_resume_staging -v
"""

import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from dronecamp_ia import training
from dronecamp_ia.config import load_config

REAL = load_config()


class FakeModel:
    def __init__(self, path):
        self.path, self.callbacks, self.resumed = path, {}, False

    def add_callback(self, event, function):
        self.callbacks.setdefault(event, []).append(function)

    def train(self, resume=False):
        self.resumed = resume


class ResumeFromStagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)
        self.run = self.temp / "runs" / "pilot_train_x"
        (self.run / "fit").mkdir(parents=True)
        self.staging = self.temp / "staging"
        weights = self.staging / self.run.name / "fit" / "weights"
        weights.mkdir(parents=True)
        (weights / "last.pt").write_bytes(b"ckpt")

    def resume(self):
        models = []

        def load(config, path):
            models.append(FakeModel(path))
            return models[-1]

        with mock.patch.dict(os.environ, {"DRONECAMP_FIT_STAGING": str(self.staging)}), \
                mock.patch.object(training, "load_detector", load), \
                mock.patch.object(training, "review_training_result"), \
                mock.patch.object(training, "apply_cpu_threads"):
            training.resume_pilot(REAL, self.run)
        return models[0]

    def test_staging_checkpoint_is_used_and_copied_back(self):
        model = self.resume()
        self.assertEqual(Path(model.path), (self.staging / self.run.name / "fit" / "weights" / "last.pt").resolve())
        self.assertTrue(model.resumed)
        self.assertIn("on_fit_epoch_end", model.callbacks)
        self.assertIn("on_train_batch_end", model.callbacks)
        self.assertTrue((self.run / "fit" / "weights" / "last.pt").is_file())

    def test_missing_checkpoint_everywhere_is_refused(self):
        shutil.rmtree(self.staging)
        with self.assertRaises(ValueError):
            self.resume()


if __name__ == "__main__":
    unittest.main()
