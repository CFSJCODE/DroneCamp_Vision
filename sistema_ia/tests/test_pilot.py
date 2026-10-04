"""Regras do piloto e da paridade, sem carregar a Ultralytics."""

import unittest

from dronecamp_ia.exporting import compare_outputs
from dronecamp_ia.pilot import split_by_scene


def image(digest: str, scene: str, classes: list[int]) -> dict:
    return {"image_sha256": digest, "scene_group": scene,
            "boxes": [{"class_id": class_id, "bbox_xyxy": [0, 0, 1, 1]} for class_id in classes]}


class PilotSplitTests(unittest.TestCase):
    def setUp(self):
        self.images = [image(f"{index:064x}", f"cena_{index // 2}", [index % 3]) for index in range(20)]
        self.images.append(image("f" * 64, "cena_rara", [7]))

    def test_scene_never_crosses_splits_and_all_splits_exist(self):
        assignment = split_by_scene(self.images, 42, (0.15, 0.15))
        self.assertEqual(set(assignment.values()), {"train", "val", "test"})
        self.assertEqual(set(assignment), {item["scene_group"] for item in self.images})

    def test_class_with_single_photo_stays_in_training(self):
        for seed in range(10):
            self.assertEqual(split_by_scene(self.images, seed, (0.15, 0.15))["cena_rara"], "train")

    def test_split_is_deterministic_for_the_same_seed(self):
        self.assertEqual(split_by_scene(self.images, 7, (0.15, 0.15)), split_by_scene(self.images, 7, (0.15, 0.15)))

    def test_too_few_scenes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "insuficientes"):
            split_by_scene([image("a" * 64, "unica", [0])], 42, (0.15, 0.15))


class ParityComparisonTests(unittest.TestCase):
    def test_identical_outputs_are_equivalent(self):
        boxes = [(0, 0.9, [0, 0, 10, 10]), (1, 0.5, [20, 20, 40, 40])]
        self.assertTrue(compare_outputs(boxes, list(reversed(boxes)), 0.25)["equivalent"])

    def test_box_missing_well_above_threshold_breaks_parity(self):
        result = compare_outputs([(0, 0.9, [0, 0, 10, 10])], [], 0.25)
        self.assertFalse(result["equivalent"])
        self.assertEqual(result["missing_in_export"][0]["class_id"], 0)

    def test_flip_next_to_threshold_is_tolerated(self):
        self.assertTrue(compare_outputs([(0, 0.255, [0, 0, 10, 10])], [], 0.25)["equivalent"])

    def test_class_change_or_confidence_drift_breaks_parity(self):
        self.assertFalse(compare_outputs([(0, 0.9, [0, 0, 10, 10])], [(1, 0.9, [0, 0, 10, 10])], 0.25)["equivalent"])
        self.assertFalse(compare_outputs([(0, 0.9, [0, 0, 10, 10])], [(0, 0.8, [0, 0, 10, 10])], 0.25)["equivalent"])

    def test_sub_pixel_change_on_thin_box_is_equivalent(self):
        result = compare_outputs([(5, 0.5, [821.0, 0.0, 1074.4, 0.6])], [(5, 0.5, [821.1, 0.0, 1074.4, 0.6])], 0.25)
        self.assertTrue(result["equivalent"])


if __name__ == "__main__":
    unittest.main()
