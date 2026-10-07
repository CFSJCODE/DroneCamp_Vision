"""retrain-cycle (cycle.py): ordem das etapas e cycle.json, com cada etapa substituída por uma função falsa."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from dronecamp_ia import cycle as cycle_module
from dronecamp_ia.config import load_config
from dronecamp_ia.io import write_json

REAL = load_config()


class RetrainCycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = replace(REAL, root=self.root)
        self.calls = []

    def tearDown(self):
        self.temporary.cleanup()

    def _fakes(self, fail_at=None):
        root = self.root

        def build(config, registries, output, seed, sequence_block=0):
            self.calls.append(("dataset", sequence_block))
            Path(output).mkdir(parents=True)
            (Path(output) / "dataset.yaml").write_text("path: .\n", encoding="utf-8")
            return {"dataset": str(Path(output) / "dataset.yaml"), "counts": {}}

        def train(config, data_path, weights, overrides, gpu_eval):
            self.calls.append(("train", dict(overrides or {}), gpu_eval))
            run = root / "runs" / "pilot_train_x"
            (run / "fit" / "weights").mkdir(parents=True)
            (run / "fit" / "weights" / "best.pt").write_bytes(b"x")
            write_json(run / "execution.json", {"tiling": {"patch": 1280}, "gpu_eval": {"state": "finished"}})
            if fail_at == "train":
                raise RuntimeError("treino falhou")
            return run

        def export(config, weights, parity_images=None):
            self.calls.append(("export", weights))
            run = root / "runs" / "export_x"
            run.mkdir(parents=True)
            return run

        def gate(config, data_path, baseline, candidate, splits, onnx_provider="directml"):
            self.calls.append(("gate", baseline, onnx_provider))
            run = root / "runs" / f"gate_{len(self.calls)}"
            run.mkdir(parents=True)
            write_json(run / "gate.json", {"adopt": baseline == "b.pt", "reasons": [], "images_evaluated": 3})
            return run

        def measure(config, data_path, weights, splits, conf, onnx_provider):
            self.calls.append(("measure", list(weights)))
            run = root / "runs" / "measure_x"
            run.mkdir(parents=True)
            write_json(run / "measurement.json", {"models": []})
            return run

        def report(config, output, gates, compares, evals, runs, title, measurements=()):
            self.calls.append(("report", [str(path) for path in gates], [str(path) for path in measurements]))
            run = root / "runs" / "report_x"
            run.mkdir(parents=True)
            return run

        return {"build_pilot_dataset": build, "train_pilot": train, "export_onnx": export,
                "compare_models": gate, "measure_models": measure, "build_model_report": report}

    def _run(self, fail_at=None, **kwargs):
        fakes = self._fakes(fail_at)
        with mock.patch("dronecamp_ia.pilot.build_pilot_dataset", fakes["build_pilot_dataset"]), \
             mock.patch("dronecamp_ia.training.train_pilot", fakes["train_pilot"]), \
             mock.patch("dronecamp_ia.exporting.export_onnx", fakes["export_onnx"]), \
             mock.patch("dronecamp_ia.model_gate.compare_models", fakes["compare_models"]), \
             mock.patch("dronecamp_ia.model_gate.measure_models", fakes["measure_models"]), \
             mock.patch("dronecamp_ia.model_report.build_model_report", fakes["build_model_report"]):
            return cycle_module.run_cycle(self.config, [Path("r.json")], self.root / "data" / "pilot" / "v", **kwargs)

    def test_steps_run_in_order_and_cycle_json_records_each(self):
        cycle = self._run(baselines=["a.pt", "b.pt"], overrides={"tile": True, "epochs": 2}, sequence_block=10,
                          gpu_eval={"enabled": True}, onnx_provider="cpu")
        record = json.loads((cycle / "cycle.json").read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "completed")
        self.assertEqual([call[0] for call in self.calls], ["dataset", "train", "export", "gate", "gate", "measure", "report"])
        self.assertEqual(self.calls[0][1], 10)
        self.assertEqual(self.calls[1][1], {"tile": True, "epochs": 2})
        self.assertEqual(self.calls[3][2], "cpu")
        self.assertTrue(self.calls[5][1][-1].endswith("best.pt"))
        self.assertEqual(len(self.calls[6][1]), 2)  # dois gate.json no relatório
        self.assertEqual(record["steps"]["gates"][1]["adopt"], True)
        self.assertTrue(record["adopt_any"])
        self.assertEqual(record["steps"]["training"]["tiling"], {"patch": 1280})

    def test_failure_is_recorded_and_raised(self):
        with self.assertRaises(RuntimeError):
            self._run(fail_at="train")
        cycle = next((self.root / "runs").glob("cycle_*"))
        record = json.loads((cycle / "cycle.json").read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "failed")
        self.assertIn("treino falhou", record["error"])
        self.assertNotIn("export", record["steps"])


if __name__ == "__main__":
    unittest.main()
