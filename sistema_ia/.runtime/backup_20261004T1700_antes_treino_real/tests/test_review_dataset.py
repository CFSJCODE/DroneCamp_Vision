"""Verifique promoção humana, originais, cobertura e independência dos splits."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import yaml

from dronecamp_ia.config import ProjectConfig
from dronecamp_ia.io import file_hash
from dronecamp_ia.review_dataset import MVP_NAMES, build_approved_dataset


class ApprovedDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="dronecamp revisão ç ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.taxonomy = self.root / "taxonomy.json"
        self.save_json(self.taxonomy, {"version": "test", "classes": [
            {"id": index, "slug": name, "phase": 1} for index, name in enumerate(MVP_NAMES)]})
        self.config = ProjectConfig(self.root, "unused", "unused", self.taxonomy, self.root / "unused.yaml", "cpu", {}, {})
        self.registry_path, self.assignments_path = self.root / "registry.json", self.root / "assignments.json"
        self.output = self.root / "dataset_v1"
        self.images = []
        self.groups = {}

    @staticmethod
    def save_json(path: Path, value: dict) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def add_image(self, group: str, split: str = "train", class_ids=(0,), approved: bool = True, status: str = "positive") -> dict:
        source = self.root / f"source_{len(self.images)}.png"
        Image.new("RGB", (32, 24), color=(20 + len(self.images) * 30, 80, 90)).save(source)
        item = {"image_sha256": file_hash(source), "source_path": str(source), "width": 32, "height": 24,
                "building_group": group, "status": status, "human_approved": approved,
                "technical_status": "human_visual_reviewed" if approved else "pending_human_review",
                "human_reviewer": "reviewer_test" if approved else None,
                "boxes": [{"class_id": index, "bbox_xyxy": [2, 3, 15, 18]} for index in class_ids] if status != "negative" else []}
        self.images.append(item)
        self.groups[group] = split
        return item

    def build(self) -> dict:
        self.save_json(self.registry_path, {"schema_version": 1, "taxonomy_sha256": file_hash(self.taxonomy), "images": self.images})
        self.save_json(self.assignments_path, {"schema_version": 1, "groups": self.groups})
        return build_approved_dataset(self.config, self.registry_path, self.assignments_path, self.output)

    def three_groups(self) -> None:
        self.add_image("building_a", "train", range(len(MVP_NAMES)))
        self.add_image("building_b", "val")
        self.add_image("building_c", "test")

    def test_zero_approved_creates_nothing(self) -> None:
        self.add_image("ceasa", approved=False)
        with self.assertRaisesRegex(ValueError, "Nenhuma imagem"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_single_ceasa_stays_draft_without_empty_splits(self) -> None:
        self.add_image("ceasa", class_ids=range(len(MVP_NAMES)))
        result = self.build()
        self.assertFalse(result["ready_for_training"])
        self.assertEqual(result["state"], "draft")
        self.assertTrue(any("três grupos" in reason for reason in result["reasons"]))
        self.assertFalse((self.output / "images/val").exists())
        self.assertFalse((self.output / "images/test").exists())
        self.assertEqual(result["counts"]["train"]["images"], 1)

    def test_three_independent_groups_and_all_train_classes_ready(self) -> None:
        self.three_groups()
        result = self.build()
        self.assertTrue(result["ready_for_training"], result)
        self.assertTrue(result["validation"]["valid"])
        self.assertEqual(result["reasons"], [])
        config = yaml.safe_load((self.output / "dataset.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["names"], list(MVP_NAMES))
        self.assertTrue(Path(config["path"]).is_absolute())
        provenance = json.loads((self.output / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["registry_sha256"], file_hash(self.registry_path))
        self.assertEqual(provenance["assignments_sha256"], file_hash(self.assignments_path))
        for item in provenance["images"]:
            self.assertEqual(file_hash(self.output / item["image"]), item["image_sha256"])
            self.assertEqual(item["approval_author"], "reviewer_test")

    def test_unapproved_ambiguous_and_excluded_do_not_enter(self) -> None:
        self.three_groups()
        self.add_image("ai_only", approved=False)
        self.add_image("ambiguous", status="ambiguous")
        self.add_image("excluded", status="excluded")
        result = self.build()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(result["counts"]["total"]["images"], 3)
        provenance = json.loads((self.output / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(len(provenance["excluded_images"]), 3)

    def test_changed_source_hash_rejected_before_writing(self) -> None:
        item = self.add_image("ceasa")
        Image.new("RGB", (32, 24), color="red").save(item["source_path"])
        with self.assertRaisesRegex(ValueError, "Hash do original"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_changed_dimensions_rejected_before_writing(self) -> None:
        item = self.add_image("ceasa")
        item["width"] = 31
        with self.assertRaisesRegex(ValueError, "Dimensões"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_taxonomy_hash_must_match_review(self) -> None:
        self.add_image("ceasa")
        self.build()
        self.output = self.root / "dataset_v2"
        self.taxonomy.write_text(self.taxonomy.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Taxonomia mudou"):
            build_approved_dataset(self.config, self.registry_path, self.assignments_path, self.output)
        self.assertFalse(self.output.exists())

    def test_output_is_immutable(self) -> None:
        self.three_groups()
        self.build()
        original = (self.output / "provenance.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "imutáveis"):
            self.build()
        self.assertEqual((self.output / "provenance.json").read_bytes(), original)

    def test_missing_train_class_keeps_dataset_not_ready(self) -> None:
        self.three_groups()
        complete_boxes = list(self.images[0]["boxes"])
        for missing_name in ("pedaco_telha", "rufo_deslocado", "pedaco_telha_sobreposto", "fixador_telha_frouxo", "reparo_rufo"):
            with self.subTest(missing_class=missing_name):
                missing_id = MVP_NAMES.index(missing_name)
                self.images[0]["boxes"] = [box for box in complete_boxes if box["class_id"] != missing_id]
                self.output = self.root / f"dataset_without_{missing_name}"
                result = self.build()
                self.assertTrue(result["validation"]["valid"])
                self.assertFalse(result["ready_for_training"])
                self.assertTrue(any(missing_name in reason for reason in result["reasons"]))

    def test_fragment_class_id8_is_exported_without_changing_previous_ids(self) -> None:
        self.three_groups()
        result = self.build()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(MVP_NAMES[8], "pedaco_telha")
        self.assertEqual(MVP_NAMES[7], "vegetacao_calha")
        label = self.output / "labels/train" / (self.images[0]["image_sha256"] + ".txt")
        self.assertIn("8", [line.split()[0] for line in label.read_text(encoding="utf-8").splitlines()])
        self.assertEqual(result["train_boxes_by_class"]["8"], 1)

    def test_displaced_flashing_class_id9_is_exported_with_stable_prior_ids(self) -> None:
        self.three_groups()
        result = self.build()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(MVP_NAMES[8:10], ("pedaco_telha", "rufo_deslocado"))
        label = self.output / "labels/train" / (self.images[0]["image_sha256"] + ".txt")
        self.assertIn("9", [line.split()[0] for line in label.read_text(encoding="utf-8").splitlines()])
        self.assertEqual(result["train_boxes_by_class"]["9"], 1)

    def test_new_classes_export_as_ids10_and11_with_previous_ids_preserved(self) -> None:
        self.three_groups()
        result = self.build()
        self.assertTrue(result["ready_for_training"])
        previous_names = (
            "telha_quebrada", "telha_ausente", "residuos_telha", "reparo_telha",
            "rufo_ausente", "rufo_quebrado", "residuos_calha", "vegetacao_calha",
            "pedaco_telha", "rufo_deslocado",
        )
        self.assertEqual(MVP_NAMES[:10], previous_names)
        self.assertEqual(MVP_NAMES[10:12], ("pedaco_telha_sobreposto", "fixador_telha_frouxo"))
        exported = yaml.safe_load((self.output / "dataset.yaml").read_text(encoding="utf-8"))
        self.assertEqual(exported["nc"], 13)
        self.assertEqual(len(exported["names"]), 13)
        label = self.output / "labels/train" / (self.images[0]["image_sha256"] + ".txt")
        rows = [line.split() for line in label.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([int(row[0]) for row in rows], list(range(13)))
        for class_id in (10, 11):
            with self.subTest(class_id=class_id):
                self.assertEqual(result["train_boxes_by_class"][str(class_id)], 1)

    def test_flashing_repair_class_id12_is_exported_after_existing_classes(self) -> None:
        self.three_groups()
        result = self.build()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(MVP_NAMES[12], "reparo_rufo")
        self.assertEqual(MVP_NAMES[11], "fixador_telha_frouxo")
        exported = yaml.safe_load((self.output / "dataset.yaml").read_text(encoding="utf-8"))
        self.assertEqual(exported["names"][12], "reparo_rufo")
        label = self.output / "labels/train" / (self.images[0]["image_sha256"] + ".txt")
        self.assertIn("12", [line.split()[0] for line in label.read_text(encoding="utf-8").splitlines()])
        self.assertEqual(result["train_boxes_by_class"]["12"], 1)

    def test_approved_negative_has_explicit_empty_label(self) -> None:
        self.three_groups()
        negative = self.add_image("building_a", status="negative")
        result = self.build()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual((self.output / "labels/train" / (negative["image_sha256"] + ".txt")).read_text(), "")


if __name__ == "__main__":
    unittest.main()
