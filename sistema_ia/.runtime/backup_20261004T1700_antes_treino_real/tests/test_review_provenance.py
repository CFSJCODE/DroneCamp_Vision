"""Bloqueie treino quando dados ou aprovação divergem da revisão de origem."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

from dronecamp_ia import training
from dronecamp_ia.config import ProjectConfig
from dronecamp_ia.io import file_hash
from dronecamp_ia.review_dataset import MVP_NAMES, build_approved_dataset
from dronecamp_ia.review_provenance import validate_training_provenance


class TrainingProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dronecamp proveniência ç ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.taxonomy = self.root / "taxonomy.json"
        self.write(self.taxonomy, {"classes": [{"id": i, "slug": n, "phase": 1} for i, n in enumerate(MVP_NAMES)]})
        self.config = ProjectConfig(self.root, "unused", "unused", self.taxonomy, self.root / "unused.yaml", "cpu", {}, {})
        self.registry_path, self.assignments_path = self.root / "registry.json", self.root / "assignments.json"
        self.output = self.root / "dataset_v1"
        self.items = []
        for index, split in enumerate(("train", "val", "test")):
            source = self.root / f"source_{index}.png"
            Image.new("RGB", (32, 24), color=(30 + index * 60, 90, 100)).save(source)
            self.items.append({"image_sha256": file_hash(source), "source_path": str(source), "width": 32, "height": 24,
                "building_group": f"building_{index}", "status": "positive", "human_approved": True,
                "technical_status": "human_visual_reviewed", "human_reviewer": "human_test",
                "boxes": [{"class_id": c, "bbox_xyxy": [2, 3, 16, 18]} for c in (range(len(MVP_NAMES)) if split == "train" else [0])]})
        self.registry = {"schema_version": 1, "taxonomy_sha256": file_hash(self.taxonomy), "images": self.items}
        self.assignments = {"schema_version": 1, "groups": {f"building_{i}": s for i, s in enumerate(("train", "val", "test"))}}

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def build(self):
        self.write(self.registry_path, self.registry)
        self.write(self.assignments_path, self.assignments)
        build_approved_dataset(self.config, self.registry_path, self.assignments_path, self.output)
        self.data_path = self.output / "dataset.yaml"
        self.provenance_path = self.output / "provenance.json"
        self.provenance = json.loads(self.provenance_path.read_text(encoding="utf-8"))

    def check(self):
        return validate_training_provenance(self.config, self.data_path)

    def test_valid_approval_is_recomputed_without_readiness(self):
        self.build()
        (self.output / "readiness.json").unlink()
        result = self.check()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(result["counts"]["total"]["images"], 3)
        self.assertEqual(result["registry_sha256"], file_hash(self.registry_path))

    def test_structurally_valid_dataset_without_provenance_is_rejected(self):
        self.build()
        self.provenance_path.unlink()
        with self.assertRaisesRegex(ValueError, "Proveniência"):
            self.check()

    def test_changed_label_rejected_even_when_hash_resigned(self):
        self.build()
        sample = self.provenance["images"][0]
        label = self.output / sample["label"]
        label.write_text(label.read_text().replace("0.28125000", "0.30000000"), encoding="utf-8")
        self.assertNotEqual(file_hash(label), sample["label_sha256"])
        sample["label_sha256"] = file_hash(label)
        self.write(self.provenance_path, self.provenance)
        with self.assertRaisesRegex(ValueError, "Label normalizada"):
            self.check()

    def test_changed_image_bytes_rejected(self):
        self.build()
        image = self.output / self.provenance["images"][0]["image"]
        Image.new("RGB", (32, 24), color="purple").save(image)
        with self.assertRaisesRegex(ValueError, "Hash da imagem"):
            self.check()

    def test_false_approval_flag_rejected(self):
        self.build()
        self.provenance["images"][0]["human_approved"] = False
        self.write(self.provenance_path, self.provenance)
        with self.assertRaisesRegex(ValueError, "aprovação humana"):
            self.check()

    def test_missing_approval_author_rejected(self):
        self.build()
        self.provenance["images"][0]["approval_author"] = None
        self.write(self.provenance_path, self.provenance)
        with self.assertRaisesRegex(ValueError, "Autor"):
            self.check()

    def test_pending_technical_status_rejected(self):
        self.build()
        self.provenance["images"][0]["technical_status"] = "pending_human_review"
        self.write(self.provenance_path, self.provenance)
        with self.assertRaisesRegex(ValueError, "estado de revisão humana"):
            self.check()

    def test_original_image_changes_after_export_rejected(self):
        self.build()
        Image.new("RGB", (32, 24), color="black").save(self.items[0]["source_path"])
        with self.assertRaisesRegex(ValueError, "imagem original"):
            self.check()

    def test_exif_orientation_cannot_change_coordinate_frame(self):
        self.build()
        source = Path(self.items[0]["source_path"])
        sample = self.provenance["images"][0]
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (32, 24), color="orange").save(source, exif=exif)
        copied = self.output / sample["image"]
        copied.write_bytes(source.read_bytes())
        sample["image_sha256"] = self.items[0]["image_sha256"] = file_hash(source)
        self.write(self.registry_path, self.registry)
        self.provenance["registry_sha256"] = file_hash(self.registry_path)
        self.write(self.provenance_path, self.provenance)
        with self.assertRaisesRegex(ValueError, "Orientação EXIF"):
            self.check()

    def test_registry_hash_drift_rejected(self):
        self.build()
        self.registry["images"][0]["human_reviewer"] = "changed_author"
        self.write(self.registry_path, self.registry)
        with self.assertRaisesRegex(ValueError, "Hash de registry"):
            self.check()

    def test_assignment_hash_drift_rejected(self):
        self.build()
        self.assignments["groups"]["building_0"] = "val"
        self.write(self.assignments_path, self.assignments)
        with self.assertRaisesRegex(ValueError, "Hash de assignments"):
            self.check()

    def test_taxonomy_drift_rejected(self):
        self.build()
        self.taxonomy.write_text(self.taxonomy.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Taxonomia"):
            self.check()

    def test_added_image_and_label_missing_from_manifest_rejected(self):
        self.build()
        added = self.output / "images/train/extra.png"
        Image.new("RGB", (32, 24), color="green").save(added)
        (self.output / "labels/train/extra.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        with (self.output / "groups.csv").open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(["images/train/extra.png", "building_0", "train"])
        with self.assertRaisesRegex(ValueError, "Inventário"):
            self.check()

    def test_declared_group_change_rejected(self):
        self.build()
        self.provenance["images"][0]["building_group"] = "other_building"
        self.write(self.provenance_path, self.provenance)
        with self.assertRaisesRegex(ValueError, "Grupo"):
            self.check()

    def test_missing_train_class_cannot_be_unblocked_by_saved_readiness(self):
        complete_boxes = list(self.items[0]["boxes"])
        for missing_name in ("pedaco_telha", "rufo_deslocado", "pedaco_telha_sobreposto", "fixador_telha_frouxo", "reparo_rufo"):
            with self.subTest(missing_class=missing_name):
                missing_id = MVP_NAMES.index(missing_name)
                self.items[0]["boxes"] = [box for box in complete_boxes if box["class_id"] != missing_id]
                self.output = self.root / f"dataset_without_{missing_name}"
                self.build()
                self.write(self.output / "readiness.json", {"ready_for_training": True})
                with self.assertRaisesRegex(ValueError, "classes ativas.*" + missing_name):
                    self.check()

    def test_fragment_class_id8_remains_valid_in_human_provenance(self):
        self.build()
        result = self.check()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(result["train_boxes_by_class"]["8"], 1)
        self.assertEqual(MVP_NAMES[8], "pedaco_telha")

    def test_displaced_flashing_class_id9_validates_human_provenance(self):
        self.build()
        result = self.check()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(result["train_boxes_by_class"]["9"], 1)
        self.assertEqual(MVP_NAMES[9], "rufo_deslocado")

    def test_overlaid_fragment_and_loose_fastener_validate_as_ids10_and11(self):
        self.build()
        result = self.check()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(MVP_NAMES[10:12], ("pedaco_telha_sobreposto", "fixador_telha_frouxo"))
        self.assertEqual(len(result["train_boxes_by_class"]), 13)
        for class_id in (10, 11):
            with self.subTest(class_id=class_id):
                self.assertEqual(result["train_boxes_by_class"][str(class_id)], 1)

    def test_flashing_repair_class_id12_validates_human_provenance(self):
        self.build()
        result = self.check()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(MVP_NAMES[12], "reparo_rufo")
        self.assertEqual(len(result["train_boxes_by_class"]), 13)
        self.assertEqual(result["train_boxes_by_class"]["12"], 1)
        sample = next(item for item in self.provenance["images"] if item["split"] == "train")
        label = self.output / sample["label"]
        self.assertIn("12", [line.split()[0] for line in label.read_text(encoding="utf-8").splitlines()])
        self.assertEqual(sample["approval_author"], "human_test")

    def test_single_ceasa_draft_cannot_train(self):
        self.items = self.items[:1]
        self.registry["images"] = self.items
        self.build()
        with self.assertRaisesRegex(ValueError, "estruturalmente inválido"):
            self.check()

    def test_path_traversal_rejected(self):
        self.build()
        self.provenance["images"][0]["label"] = "../registry.json"
        self.write(self.provenance_path, self.provenance)
        with self.assertRaisesRegex(ValueError, "travessia"):
            self.check()

    def test_train_tune_evaluate_without_provenance_never_load_detector(self):
        self.build()
        self.provenance_path.unlink()
        stages = {
            "train": lambda: training.train(self.config, self.data_path),
            "tune": lambda: training.tune(self.config, self.data_path, iterations=1, epochs=1),
            "evaluate": lambda: training.evaluate(self.config, self.data_path, weights="never_loaded.pt"),
        }
        for stage, invoke in stages.items():
            with self.subTest(stage=stage), patch("dronecamp_ia.training.load_detector") as detector:
                with self.assertRaisesRegex(ValueError, "Proveniência"):
                    invoke()
                detector.assert_not_called()
        audits = list((self.root / "runs").glob("*/training_provenance.json"))
        self.assertEqual(len(audits), 3)
        for audit in audits:
            self.assertFalse(json.loads(audit.read_text(encoding="utf-8"))["valid"])

    def test_prepare_approved_dataset_records_valid_provenance_without_model(self):
        self.build()
        raw = yaml.safe_load(self.data_path.read_text(encoding="utf-8"))
        raw["download"] = "ignored_content_not_executed"
        self.data_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        run = self.root / "prepare_approved"
        run.mkdir()
        with patch("dronecamp_ia.training.load_detector") as detector:
            resolved_path, counts = training.prepare_dataset(self.config, self.data_path, run)
            detector.assert_not_called()
        self.assertEqual(resolved_path, run / "dataset_resolved.yaml")
        resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        self.assertTrue(Path(resolved["path"]).is_absolute())
        self.assertEqual(resolved["names"], list(MVP_NAMES))
        self.assertNotIn("download", resolved)
        for split in ("train", "val", "test"):
            self.assertEqual(Path(resolved[split]), self.output / "images" / split)
        audit = json.loads((run / "training_provenance.json").read_text(encoding="utf-8"))
        self.assertTrue(audit["valid"])
        self.assertTrue(audit["ready_for_training"])
        self.assertEqual(counts["train"], {name: 1 for name in MVP_NAMES})
        self.assertTrue((run / "dataset_snapshot.json").is_file())


if __name__ == "__main__":
    unittest.main()
