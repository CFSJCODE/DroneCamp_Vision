"""Família/versão do YOLO: nomes oficiais, bloqueio do YOLO27 e colunas de perda por família."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import importlib.util
import unittest

from dronecamp_ia.backend import load_detector
from dronecamp_ia.config import load_config
from dronecamp_ia.model_family import (FAMILIES, GENERIC_DISTANCE_LOSS, describe_checkpoint, distance_loss_columns,
                                       ensure_downloadable, family_support, migration_checklist, model_info,
                                       parse_model_name)
from dronecamp_ia.training import _review_training_csv

ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None


class ParseModelNameTests(unittest.TestCase):
    def test_official_yolo26_detection_weight(self):
        spec = parse_model_name("yolo26l.pt")
        self.assertEqual((spec.family, spec.size, spec.task), (26, "l", "detect"))
        self.assertTrue(spec.official and spec.released and spec.end_to_end)
        self.assertEqual(spec.distance_loss, ("l1_loss",))
        self.assertEqual(spec.label, "YOLO26L")

    def test_other_tasks_and_case_are_parsed(self):
        self.assertEqual(parse_model_name("YOLO26M-seg.pt").task, "seg")
        self.assertEqual(parse_model_name("yolo26x-pose.pt").size, "x")

    def test_yolo27_is_known_but_not_released(self):
        spec = parse_model_name("yolo27l.pt")
        self.assertEqual(spec.family, 27)
        self.assertTrue(spec.official)
        self.assertFalse(spec.released)
        self.assertEqual(spec.distance_loss, tuple(FAMILIES[27]["distance_loss"]))

    def test_unknown_family_and_local_checkpoint(self):
        unknown = parse_model_name("yolo11n.pt")
        self.assertEqual(unknown.family, 11)
        self.assertTrue(unknown.official)
        self.assertFalse(unknown.end_to_end)
        local = parse_model_name(r"runs\pilot_train_x\fit\weights\best.pt")
        self.assertIsNone(local.family)
        self.assertFalse(local.official)
        self.assertEqual(local.distance_loss, GENERIC_DISTANCE_LOSS)


class DownloadPolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_yolo27_is_rejected_with_explanation(self):
        with self.assertRaisesRegex(ValueError, "YOLO27 ainda não foi publicado"):
            ensure_downloadable(parse_model_name("yolo27l.pt"), Path("yolo27l.pt"))
        with self.assertRaisesRegex(ValueError, "YOLO27 ainda não foi publicado"):
            load_detector(self.config, "yolo27l.pt")

    def test_non_detection_and_unknown_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "tarefa 'seg'"):
            ensure_downloadable(parse_model_name("yolo26l-seg.pt"), Path("yolo26l-seg.pt"))
        with self.assertRaisesRegex(ValueError, "Checkpoint local não encontrado"):
            ensure_downloadable(parse_model_name("yolo26q.pt"), Path("yolo26q.pt"))
        with self.assertRaisesRegex(ValueError, "Checkpoint local não encontrado"):
            load_detector(self.config, "runs/inexistente/best.pt")

    def test_released_family_passes_without_network(self):
        ensure_downloadable(parse_model_name("yolo26n.pt"), Path("yolo26n.pt"))

    @unittest.skipUnless(ULTRALYTICS, "Ultralytics ausente.")
    def test_installed_library_reports_support_without_downloading(self):
        current = family_support(parse_model_name("yolo26l.pt"))
        self.assertTrue(current["downloadable"] and current["architecture_available"])
        future = family_support(parse_model_name("yolo27l.pt"))
        # Quando a Ultralytics publicar o YOLO27, este teste passa a documentar a mudança.
        self.assertEqual(future["downloadable"], future["official_weights_listed"])
        if not future["downloadable"]:
            self.assertIn("ainda não foi publicado", future["reason"])


class ProvenanceAndLossTests(unittest.TestCase):
    def test_trained_checkpoint_inherits_family_from_base_weights(self):
        info = describe_checkpoint(Path("runs/pilot_train_x/fit/weights/best.pt"), {"model": "yolo26l.pt"})
        self.assertEqual((info["family"], info["label"], info["base_weights"]), (26, "YOLO26L", "yolo26l.pt"))
        self.assertIsNone(describe_checkpoint(Path("best.pt"))["family"])

    def test_distance_loss_follows_family(self):
        model = SimpleNamespace(ckpt={"train_args": {"model": "yolo26l.pt"}}, ckpt_path="runs/x/fit/weights/best.pt")
        self.assertEqual(distance_loss_columns(model), ("l1_loss",))
        self.assertEqual(distance_loss_columns(SimpleNamespace()), GENERIC_DISTANCE_LOSS)

    def test_results_csv_audit_uses_family_columns(self):
        header = ("epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),"
                  "metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss\n")
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.csv"
            path.write_text(header + "1,1,1,1,0.5,0.5,0.5,0.2,1,1,1\n", encoding="utf-8")
            generic, _ = _review_training_csv(path, GENERIC_DISTANCE_LOSS)
            self.assertEqual(generic, [])
            strict, _ = _review_training_csv(path, ("l1_loss",))
            self.assertTrue(any("l1_loss" in reason for reason in strict))


class ModelInfoTests(unittest.TestCase):
    def test_report_lists_configured_requested_and_migration(self):
        config = load_config()
        report = model_info(config)
        self.assertEqual(report["configured"]["family"], 26)
        self.assertEqual(report["requested"]["name"], config.requested_model)
        self.assertIn("migration", report)
        self.assertEqual(report["migration"][0]["step"], "pesos publicados")
        self.assertIn("compare-models", " ".join(step["how"] for step in report["migration"]))

    def test_same_model_has_no_migration_and_custom_model_is_accepted(self):
        config = replace(load_config(), requested_model="yolo26l.pt")
        self.assertNotIn("migration", model_info(config))
        self.assertEqual(model_info(config, "yolo26n.pt")["configured"]["size"], "n")

    def test_checklist_requires_gate_before_config_change(self):
        steps = migration_checklist(parse_model_name("yolo26l.pt"), parse_model_name("yolo27l.pt"))
        names = [step["step"] for step in steps]
        self.assertLess(names.index("comparação por classe com o modelo atual"), names.index("configuração"))
        self.assertTrue(next(step for step in steps if step["step"] == "modelo anterior preservado")["done"])


if __name__ == "__main__":
    unittest.main()
