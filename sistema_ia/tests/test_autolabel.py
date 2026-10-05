"""Marcação automática: frases do YOLOE, união com o piloto e filtros (sem rede neural).

O YOLOE real roda em test_e2e_ultralytics.py (test_10), se o checkpoint existir.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from dronecamp_ia.autolabel import (SOURCE_BOTH, SOURCE_OPEN, SOURCE_PILOT, load_prompts, merge_suggestions,
                                    zero_shot_boxes)
from dronecamp_ia.config import detection_names, load_config, load_taxonomy

REAL = load_config()
NAMES = detection_names(load_taxonomy(REAL.taxonomy_path))
GUTTER = NAMES.index("residuos_calha")


class FakeTensor(list):
    def cpu(self):
        return self

    def tolist(self):
        return list(self)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = type("Boxes", (), {"xyxy": FakeTensor(b for b, _, _ in boxes),
                                        "conf": FakeTensor(c for _, c, _ in boxes),
                                        "cls": FakeTensor(i for _, _, i in boxes)})()


class FakeModel:
    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, **kwargs):
        return iter([FakeResult(self.boxes)])


class PromptTests(unittest.TestCase):
    def test_project_prompts_cover_every_class_and_map_to_taxonomy(self):
        spec = load_prompts(REAL)
        mapped = {class_id for _, class_id in spec["vocabulary"] if class_id is not None}
        self.assertEqual(mapped, set(range(len(NAMES))))
        self.assertIn(("debris in roof gutter", GUTTER), spec["vocabulary"])

    def test_unknown_class_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copytree(REAL.root / "configs", root / "configs")
            path = root / "configs" / "zero_shot_prompts.json"
            spec = json.loads(path.read_text(encoding="utf-8"))
            spec["classes"]["classe_inventada"] = ["anything"]
            path.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_prompts(replace(REAL, root=root, taxonomy_path=root / "configs" / "taxonomy.json"))


class MergeTests(unittest.TestCase):
    def test_same_object_found_by_both_becomes_one_suggestion(self):
        pilot = [{"class_id": GUTTER, "bbox_xyxy": [0, 0, 10, 10], "confidence": 0.3, "id": "s001"}]
        found = [{"class_id": GUTTER, "bbox_xyxy": [0, 0, 10, 11], "confidence": 0.6, "source": SOURCE_OPEN,
                  "prompt": "clogged gutter"},
                 {"class_id": 7, "bbox_xyxy": [50, 50, 80, 80], "confidence": 0.2, "source": SOURCE_OPEN,
                  "prompt": "moss in gutter"}]
        merged = merge_suggestions(pilot, found)
        self.assertEqual([box["source"] for box in merged], [SOURCE_BOTH, SOURCE_OPEN])
        self.assertEqual(merged[0]["confidence"], 0.6)
        self.assertEqual([box["id"] for box in merged], ["s001", "s002"])

    def test_pilot_only_boxes_keep_their_source(self):
        merged = merge_suggestions([{"class_id": 0, "bbox_xyxy": [0, 0, 5, 5], "confidence": 0.5}], [])
        self.assertEqual(merged[0]["source"], SOURCE_PILOT)


class ZeroShotBoxesTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.image = Path(temporary.name) / "foto.jpg"
        Image.new("RGB", (100, 100)).save(self.image)
        self.spec = {"conf": 0.1, "max_area_fraction": 0.5,
                     "vocabulary": [("debris in roof gutter", GUTTER), ("clogged gutter", GUTTER), ("rust", None)]}

    def test_prompts_map_to_classes_duplicates_merge_and_whole_photo_boxes_drop(self):
        model = FakeModel([([10, 10, 30, 30], 0.4, 0), ([11, 10, 30, 31], 0.3, 1),
                           ([0, 0, 100, 90], 0.9, 0), ([60, 60, 70, 70], 0.5, 2)])
        catalog, uncatalogued = zero_shot_boxes(model, self.image, self.spec, "cpu")
        self.assertEqual(len(catalog), 1)  # duas frases, mesmo objeto; a caixa da foto inteira sai
        self.assertEqual(catalog[0]["class_id"], GUTTER)
        self.assertEqual(catalog[0]["prompt"], "debris in roof gutter")
        self.assertEqual([item["prompt"] for item in uncatalogued], ["rust"])
        self.assertNotIn("class_id", uncatalogued[0])


if __name__ == "__main__":
    unittest.main()
