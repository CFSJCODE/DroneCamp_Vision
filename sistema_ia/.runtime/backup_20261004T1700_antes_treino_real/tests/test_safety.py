"""Contratos essenciais: classes COCO, severidade e disponibilidade do modelo."""

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import json
import unittest

from dronecamp_ia.backend import check_domain_names, load_detector
from dronecamp_ia.config import load_config, load_taxonomy, detection_names
from dronecamp_ia.prediction import serialize_detections
from dronecamp_ia.training import review_tuning_result


class Values:
    def __init__(self, data):
        self.data = data

    def cpu(self):
        return self

    def tolist(self):
        return self.data


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.taxonomy = load_taxonomy(self.config.taxonomy_path)

    def test_coco_checkpoint_cannot_claim_roof_categories(self):
        model = SimpleNamespace(names={0: "person", 1: "car"})
        with self.assertRaisesRegex(ValueError, "Checkpoint não corresponde"):
            check_domain_names(model, detection_names(self.taxonomy))

    def test_same_classes_in_different_order_are_rejected(self):
        names = detection_names(self.taxonomy)
        model = SimpleNamespace(names=dict(enumerate(reversed(names))))
        with self.assertRaises(ValueError):
            check_domain_names(model, names)

    def test_future_model_is_rejected_before_download(self):
        with self.assertRaisesRegex(ValueError, "YOLO27 ainda não foi publicado"):
            load_detector(self.config, "yolo27l.pt")

    def test_candidate_keeps_severity_pending_despite_confidence(self):
        result = SimpleNamespace(
            names={0: "telha_quebrada"},
            boxes=SimpleNamespace(xyxy=Values([[10, 20, 100, 200]]), conf=Values([0.999]), cls=Values([0])),
        )
        findings = serialize_detections(result, self.taxonomy, demo=False)
        self.assertIsNone(findings[0]["severity"])
        self.assertEqual(findings[0]["review_status"], "pendente")
        self.assertEqual(findings[0]["report_reference"]["historical_severity_only"], "gravissima")

    def test_demo_never_assigns_project_reference(self):
        result = SimpleNamespace(
            names={0: "person"},
            boxes=SimpleNamespace(xyxy=Values([[1, 2, 3, 4]]), conf=Values([0.5]), cls=Values([0])),
        )
        findings = serialize_detections(result, self.taxonomy, demo=True)
        self.assertIsNone(findings[0]["category_label"])
        self.assertIsNone(findings[0]["report_reference"])

    def test_no_boxes_is_empty_evidence_not_conformity(self):
        result = SimpleNamespace(boxes=None)
        self.assertEqual(serialize_detections(result, self.taxonomy, demo=False), [])

    def test_tuner_all_failed_cannot_report_success(self):
        with TemporaryDirectory() as temporary:
            run = Path(temporary)
            search = run / "search"
            search.mkdir()
            record = {"fitness": 0, "datasets": {"dataset.yaml": {}}, "failed_datasets": ["dataset.yaml"]}
            (search / "tune_results.ndjson").write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Tuning incompleto"):
                review_tuning_result(run, 1)
            summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["successful"], 0)

    def test_zero_fitness_with_valid_metrics_is_successful_execution(self):
        with TemporaryDirectory() as temporary:
            run = Path(temporary)
            search = run / "search"
            (search / "weights").mkdir(parents=True)
            record = {"fitness": 0, "datasets": {"dataset.yaml": {"map": 0.0}}}
            (search / "tune_results.ndjson").write_text(json.dumps(record), encoding="utf-8")
            (search / "best_hyperparameters.yaml").write_text("lr0: 0.001", encoding="utf-8")
            (search / "weights/best.pt").write_bytes(b"test_fixture")
            summary = review_tuning_result(run, 1)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["human_acceptance"], "pendente")


if __name__ == "__main__":
    unittest.main()
