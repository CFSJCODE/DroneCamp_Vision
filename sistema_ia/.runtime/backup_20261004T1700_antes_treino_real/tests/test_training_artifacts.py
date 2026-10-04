"""Treino só conclui com artefatos e métricas; fixtures não são checkpoints funcionais."""

import csv
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dronecamp_ia.config import detection_names, load_config, load_taxonomy
from dronecamp_ia import training


class TrainingArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name) / "run"
        self.run.mkdir()
        config = load_config()
        self.config = replace(config, root=Path(self.temporary.name), training={**config.training, "epochs": 100})
        self.names = detection_names(load_taxonomy(config.taxonomy_path))
        self.model = SimpleNamespace(names=dict(enumerate(self.names)))
        self.columns = [
            "epoch", "time", "train/box_loss", "train/cls_loss", "train/l1_loss",
            *training.DETECTION_METRIC_COLUMNS, "val/box_loss", "val/cls_loss", "val/l1_loss",
        ]
        self.rows = [self.row(1)]
        self.write_artifacts()

    def row(self, epoch):
        values = {column: 0.2 for column in self.columns}
        values.update(epoch=epoch, time=10.0)
        return values

    def write_csv(self):
        path = self.run / "fit/results.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.rows)

    def write_artifacts(self):
        weights = self.run / "fit/weights"
        weights.mkdir(parents=True, exist_ok=True)
        # O teste confere existência e tamanho, não desserializa nem aprova estes bytes.
        for name in ("best.pt", "last.pt"):
            (weights / name).write_bytes(b"opaque_test_fixture_not_a_model")
        self.write_csv()

    def review(self):
        return training.review_training_result(self.run, self.model, self.names)

    def summary(self):
        return json.loads((self.run / "summary.json").read_text(encoding="utf-8"))

    def assert_incomplete(self, reason):
        with self.assertRaisesRegex(ValueError, reason):
            self.review()
        result = self.summary()
        self.assertFalse(result["complete"])
        self.assertEqual(result["state"], "failed")
        self.assertTrue(result["reasons"])
        self.assertEqual(result["human_acceptance"], "pendente")

    def test_valid_early_stopping_does_not_require_requested_hundred_epochs(self):
        self.rows = [self.row(epoch) for epoch in range(1, 4)]
        self.write_csv()
        result = self.review()
        self.assertTrue(result["complete"])
        self.assertEqual(result["epochs_recorded"], 3)
        self.assertEqual(result["last_epoch"], 3)
        self.assertTrue(result["model_class_contract_verified"])
        self.assertEqual(result["human_acceptance"], "pendente")
        self.assertFalse(result["test_evaluated"])
        self.assertFalse(result["checkpoint_functional_validation"])
        self.assertFalse(result["export_parity_verified"])

    def test_zero_metrics_are_a_valid_execution_not_quality_acceptance(self):
        for column in training.DETECTION_METRIC_COLUMNS:
            self.rows[0][column] = 0.0
        self.write_csv()
        result = self.review()
        self.assertTrue(result["complete"])
        self.assertEqual(set(result["last_metrics"].values()), {0.0})
        self.assertEqual(result["human_acceptance"], "pendente")

    def test_detection_head_using_dfl_loss_is_also_supported(self):
        self.columns = [name.replace("l1_loss", "dfl_loss") for name in self.columns]
        self.rows = [{name.replace("l1_loss", "dfl_loss"): value for name, value in self.rows[0].items()}]
        self.write_csv()
        self.assertTrue(self.review()["complete"])

    def test_csv_header_whitespace_and_utf8_bom_are_accepted(self):
        path = self.run / "fit/results.csv"
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[0] = ",".join(f" {name} " for name in self.columns)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
        self.assertTrue(self.review()["complete"])

    def test_missing_best_last_or_csv_is_incomplete(self):
        for relative in ("weights/best.pt", "weights/last.pt", "results.csv"):
            with self.subTest(relative=relative):
                self.write_artifacts()
                (self.run / "fit" / relative).unlink()
                self.assert_incomplete("ausente")

    def test_empty_best_last_or_csv_is_incomplete(self):
        for relative in ("weights/best.pt", "weights/last.pt", "results.csv"):
            with self.subTest(relative=relative):
                self.write_artifacts()
                (self.run / "fit" / relative).write_bytes(b"")
                self.assert_incomplete("vazio")

    def test_checkpoint_directory_cannot_substitute_for_file(self):
        path = self.run / "fit/weights/best.pt"
        path.unlink()
        path.mkdir()
        self.assert_incomplete("não é um arquivo")

    def test_header_without_epochs_is_incomplete(self):
        self.rows = []
        self.write_csv()
        self.assert_incomplete("nenhuma época")

    def test_missing_metric_or_base_loss_column_is_incomplete(self):
        original_columns, original_row = list(self.columns), dict(self.rows[0])
        for column in ("epoch", *training.DETECTION_METRIC_COLUMNS, "train/box_loss", "val/cls_loss"):
            with self.subTest(column=column):
                self.columns = [name for name in original_columns if name != column]
                self.rows = [{name: value for name, value in original_row.items() if name != column}]
                self.write_csv()
                self.assert_incomplete("colunas obrigatórias")

    def test_unpaired_distance_loss_is_incomplete(self):
        self.columns.remove("val/l1_loss")
        self.rows[0].pop("val/l1_loss")
        self.write_csv()
        self.assert_incomplete("distância correspondentes")

    def test_all_epochs_metrics_must_be_finite_and_in_range(self):
        for column in training.DETECTION_METRIC_COLUMNS:
            for invalid in ("nan", "inf", "-inf", -0.01, 1.01):
                with self.subTest(column=column, invalid=invalid):
                    self.rows = [self.row(1), self.row(2)]
                    # Uma última época válida não pode esconder uma época inválida anterior.
                    self.rows[0][column] = invalid
                    self.write_csv()
                    self.assert_incomplete("não é finito|entre 0 e 1")

    def test_every_observed_training_and_validation_loss_must_be_finite(self):
        self.columns.extend(["train/extra_loss", "val/extra_loss"])
        for column in [name for name in self.columns if name.endswith("_loss")]:
            for invalid in ("nan", "inf", "", "not_numeric"):
                with self.subTest(column=column, invalid=invalid):
                    self.rows = [self.row(1)]
                    self.rows[0][column] = invalid
                    self.write_csv()
                    self.assert_incomplete("não é finito|não é um número")

    def test_epoch_must_be_positive_finite_integer_and_increasing(self):
        for epochs in ((0,), (1.5,), ("nan",), (1, 1), (2, 1)):
            with self.subTest(epochs=epochs):
                self.rows = [self.row(epoch) for epoch in epochs]
                self.write_csv()
                self.assert_incomplete("época deve ser|epoch não é finito")

    def test_duplicate_csv_columns_are_incomplete(self):
        path = self.run / "fit/results.csv"
        content = path.read_text(encoding="utf-8").splitlines()
        content[0] += ", metrics/precision(B) "
        content[1] += ",0.1"
        path.write_text("\n".join(content) + "\n", encoding="utf-8")
        self.assert_incomplete("colunas duplicadas")

    def test_missing_or_extra_csv_values_are_incomplete(self):
        self.write_csv()
        path = self.run / "fit/results.csv"
        original = path.read_text(encoding="utf-8").splitlines()
        for changed in (original[1].rsplit(",", 1)[0], original[1] + ",0.3"):
            with self.subTest(changed=changed):
                path.write_text(original[0] + "\n" + changed + "\n", encoding="utf-8")
                self.assert_incomplete("não é um número|além das colunas")

    def test_wrong_class_order_after_training_is_incomplete(self):
        self.model.names = dict(enumerate(reversed(self.names)))
        self.assert_incomplete("Contrato das classes")
        self.assertFalse(self.summary()["model_class_contract_verified"])
        self.assertTrue(all(item["present"] for item in self.summary()["artifacts"].values()))

    def mock_training(self, action):
        self.model.train = Mock(side_effect=action)
        return (
            patch("dronecamp_ia.training.make_run_directory", return_value=self.run),
            patch("dronecamp_ia.training.prepare_dataset", return_value=(self.run / "dataset_resolved.yaml", {})),
            patch("dronecamp_ia.training.load_detector", return_value=self.model),
            patch("dronecamp_ia.training.runtime_info", return_value={"fixture": True}),
        )

    def test_training_or_callback_exception_records_incomplete_summary(self):
        def failing_callback(**kwargs):
            raise RuntimeError("fixture callback failed")

        run_patch, data_patch, detector_patch, runtime_patch = self.mock_training(failing_callback)
        with run_patch, data_patch, detector_patch, runtime_patch:
            with self.assertRaisesRegex(ValueError, "fixture callback failed"):
                training.train(self.config, self.run / "dataset.yaml")
        summary = self.summary()
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["state"], "failed")
        self.assertIn("RuntimeError", summary["reasons"][0])

    def test_normal_train_return_without_artifacts_cannot_report_success(self):
        for path in (self.run / "fit").rglob("*"):
            if path.is_file():
                path.unlink()
        run_patch, data_patch, detector_patch, runtime_patch = self.mock_training(lambda **kwargs: None)
        with run_patch, data_patch, detector_patch, runtime_patch:
            with self.assertRaisesRegex(ValueError, "Treino incompleto"):
                training.train(self.config, self.run / "dataset.yaml")
        self.assertFalse(self.summary()["complete"])

    def test_valid_train_return_runs_artifact_guard_and_preserves_parameters(self):
        self.rows = [self.row(1), self.row(2)]
        run_patch, data_patch, detector_patch, runtime_patch = self.mock_training(lambda **kwargs: self.write_artifacts())
        with run_patch, data_patch, detector_patch, runtime_patch:
            result = training.train(self.config, self.run / "dataset.yaml")
        self.assertEqual(result, self.run)
        self.assertTrue(self.summary()["complete"])
        self.assertEqual(self.summary()["epochs_recorded"], 2)
        passed = self.model.train.call_args.kwargs
        self.assertEqual(passed["epochs"], 100)
        self.assertEqual(passed["device"], self.config.device)
        self.assertFalse(passed["exist_ok"])
        self.assertEqual(passed["project"], str(self.run))

    def test_class_contract_is_rechecked_after_train_returns(self):
        def return_with_wrong_names(**kwargs):
            self.write_artifacts()
            self.model.names = {0: "person"}

        run_patch, data_patch, detector_patch, runtime_patch = self.mock_training(return_with_wrong_names)
        with run_patch, data_patch, detector_patch, runtime_patch:
            with self.assertRaisesRegex(ValueError, "Contrato das classes"):
                training.train(self.config, self.run / "dataset.yaml")
        self.assertFalse(self.summary()["complete"])

    def test_preparation_failure_records_summary_before_loading_model(self):
        with patch("dronecamp_ia.training.make_run_directory", return_value=self.run), \
             patch("dronecamp_ia.training.prepare_dataset", side_effect=ValueError("fixture provenance rejected")), \
             patch("dronecamp_ia.training.load_detector") as detector:
            with self.assertRaisesRegex(ValueError, "fixture provenance rejected"):
                training.train(self.config, self.run / "dataset.yaml")
        detector.assert_not_called()
        self.assertFalse(self.summary()["complete"])


if __name__ == "__main__":
    unittest.main()
