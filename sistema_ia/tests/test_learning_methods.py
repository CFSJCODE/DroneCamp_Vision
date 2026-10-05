"""Bandit LinUCB (fila de revisão), calibração scikit-learn, RFS, regra de adoção e successive halving.

Testes rápidos, sem Ultralytics: a parte com modelo real está em test_e2e_ultralytics.py.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from dronecamp_ia.active_learning import (FEATURES, LinUCBPolicy, class_rarity, image_context, prioritize_review,
                                          review_reward, update_policy)
from dronecamp_ia.config import detection_names, load_config, load_taxonomy
from dronecamp_ia.hparam_bandit import arms_from_space, best_validation_reward, successive_halving
from dronecamp_ia.io import write_json
from dronecamp_ia.matching import image_errors, match_predictions, per_class_report
from dronecamp_ia.model_gate import average_precision, decide
from dronecamp_ia.sampling import repeat_factors, write_repeat_factor_list

REAL = load_config()
NAMES = detection_names(load_taxonomy(REAL.taxonomy_path))


def prediction(class_id, box, confidence):
    return {"class_id": class_id, "bbox_xyxy": box, "confidence": confidence}


def truth(class_id, box):
    return {"class_id": class_id, "bbox_xyxy": box}


class MatchingTests(unittest.TestCase):
    def test_each_human_box_is_matched_once_by_highest_confidence(self):
        predictions = [prediction(2, [0, 0, 10, 10], 0.4), prediction(2, [0, 0, 10, 11], 0.9)]
        hits, used = match_predictions(predictions, [truth(2, [0, 0, 10, 10])])
        self.assertEqual(hits, [False, True])
        self.assertEqual(used, {0})

    def test_wrong_class_is_false_positive_and_omission(self):
        errors = image_errors([prediction(1, [0, 0, 10, 10], 0.9)], [truth(2, [0, 0, 10, 10])])
        self.assertEqual((errors["tp"], errors["fp"], errors["fn"]), (0, 1, 1))

    def test_class_without_boxes_is_not_measured_rather_than_zero(self):
        report = per_class_report([image_errors([], [truth(0, [0, 0, 5, 5])])], NAMES)
        self.assertEqual(report[NAMES[0]]["recall"], 0.0)
        self.assertIsNone(report[NAMES[12]]["recall"])
        self.assertIsNone(report[NAMES[12]]["precision"])


class RepeatFactorSamplingTests(unittest.TestCase):
    def test_rare_class_images_are_repeated_and_common_ones_are_not(self):
        classes = [{0}] * 9 + [{11}]
        per_image, by_class = repeat_factors(classes, 0.3)
        self.assertEqual(by_class[0], 1.0)  # 90% das fotos: acima do limiar
        self.assertAlmostEqual(by_class[11], (0.3 / 0.1) ** 0.5)
        self.assertEqual(per_image[:9], [1.0] * 9)
        self.assertGreater(per_image[-1], 1.7)

    def test_list_keeps_every_image_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            for index in range(10):
                (root / "images" / f"{index}.jpg").write_bytes(b"x")
                label = "11 0.5 0.5 0.1 0.1\n" if index == 0 else ("0 0.5 0.5 0.1 0.1\n" if index < 9 else "")
                (root / "labels" / f"{index}.txt").write_text(label, encoding="utf-8")
            first = write_repeat_factor_list(root / "images", root / "labels", root / "a.txt", 0.3, seed=7)
            second = write_repeat_factor_list(root / "images", root / "labels", root / "b.txt", 0.3, seed=7)
            lines = (root / "a.txt").read_text(encoding="utf-8").split()
            self.assertEqual(lines, (root / "b.txt").read_text(encoding="utf-8").split())
            self.assertEqual(len(set(lines)), 10)
            self.assertGreaterEqual(lines.count((root / "images" / "0.jpg").resolve().as_posix()), 2)
            self.assertEqual(first["entries"], second["entries"])

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            repeat_factors([{0}], 0)

    def test_train_pilot_hook_rewrites_only_the_train_entry(self):
        from dronecamp_ia.training import _apply_repeat_factor_sampling

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in ("train", "val"):
                (root / "data" / "images" / split).mkdir(parents=True)
                (root / "data" / "labels" / split).mkdir(parents=True)
                (root / "data" / "images" / split / "a.jpg").write_bytes(b"x")
                (root / "data" / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            resolved = root / "dataset_resolved.yaml"
            resolved.write_text(yaml.safe_dump({"path": str(root / "data"), "train": str(root / "data/images/train"),
                                                "val": str(root / "data/images/val")}), encoding="utf-8")
            summary = _apply_repeat_factor_sampling(root, resolved, 0.3, 0)
            data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
            self.assertTrue(data["train"].endswith("train_repeat_factor.txt"))
            self.assertEqual(data["val"], str(root / "data/images/val"))
            self.assertEqual(summary["images"], 1)
            self.assertTrue((root / "sampling.json").is_file())


class LinUCBTests(unittest.TestCase):
    def setUp(self):
        self.rarity = class_rarity({0: 99, 11: 0}, len(NAMES))

    def test_prior_ranks_uncertain_rare_suggestions_above_confident_common_ones(self):
        policy = LinUCBPolicy(alpha=0.0)
        confident = image_context([prediction(0, [0, 0, 10, 10], 0.95)], self.rarity, False)
        uncertain_rare = image_context([prediction(11, [0, 0, 10, 10], 0.5)], self.rarity, False)
        self.assertGreater(policy.score(uncertain_rare)["score"], policy.score(confident)["score"])

    def test_rewards_move_the_policy_and_shrink_exploration(self):
        policy = LinUCBPolicy(alpha=1.0)
        context = image_context([prediction(0, [0, 0, 10, 10], 0.9)], self.rarity, True)
        before = policy.score(context)
        for _ in range(20):
            policy.update(context, 1.0)
        after = policy.score(context)
        self.assertLess(after["explore"], before["explore"])
        self.assertGreater(after["exploit"], before["exploit"])

    def test_reward_is_high_when_model_misses_a_rare_class(self):
        missed = review_reward([], [truth(11, [0, 0, 10, 10])], self.rarity)
        correct = review_reward([prediction(0, [0, 0, 10, 10], 0.9)], [truth(0, [0, 0, 10, 10])], self.rarity)
        self.assertGreater(missed["reward"], correct["reward"])
        self.assertEqual(missed["fn"], 1)

    def test_context_has_one_value_per_feature(self):
        self.assertEqual(image_context([], self.rarity, False).shape, (len(FEATURES),))


class ReviewQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.registry = self.root / "registry.json"
        images = [{"image_sha256": f"{index:064x}", "filename": f"{index}.jpg", "building_group": "galpao_b",
                   "human_approved": index == 0, "status": "positive" if index == 0 else "ambiguous",
                   "boxes": [truth(11, [0, 0, 10, 10])] if index == 0 else []} for index in range(3)]
        write_json(self.registry, {"images": images})
        write_json(self.root / "suggestions.json", {"weights_sha256": "w1", "images": {
            images[0]["image_sha256"]: [],
            images[1]["image_sha256"]: [prediction(0, [0, 0, 10, 10], 0.97)],
            images[2]["image_sha256"]: [prediction(11, [0, 0, 10, 10], 0.45), prediction(4, [0, 0, 10, 10], 0.4)],
        }})

    def test_queue_skips_decided_photos_and_puts_uncertain_rare_photo_first(self):
        output = prioritize_review(REAL, self.registry)
        queue = json.loads(output.read_text(encoding="utf-8"))["images"]
        self.assertEqual([row["filename"] for row in queue], ["2.jpg", "1.jpg"])
        self.assertTrue(output.with_suffix(".csv").is_file())

    def test_learning_counts_each_decision_once(self):
        policy = self.root / "policy.json"
        first = update_policy(REAL, self.registry, policy)
        second = update_policy(REAL, self.registry, policy)
        self.assertEqual(first["new_updates"], 1)
        self.assertEqual(second["new_updates"], 0)
        self.assertEqual(json.loads(policy.read_text(encoding="utf-8"))["updates"], 1)


class CalibrationTests(unittest.TestCase):
    def test_features_have_fixed_width_and_class_one_hot(self):
        from dronecamp_ia.calibration import feature_names, suggestion_features

        boxes = [prediction(3, [0, 0, 50, 20], 0.7), prediction(5, [10, 0, 60, 20], 0.2)]
        X = suggestion_features(boxes, 100, 100, len(NAMES))
        self.assertEqual(X.shape, (2, len(feature_names(len(NAMES)))))
        self.assertEqual(X[0, feature_names(len(NAMES)).index("class_3")], 1.0)

    def test_cross_validation_reports_gain_when_a_feature_explains_acceptance(self):
        from dronecamp_ia.calibration import cross_validate

        generator = np.random.default_rng(0)
        size = 240
        raw = generator.uniform(0.1, 0.9, size)  # confiança crua sem relação com o acerto
        informative = generator.normal(size=size)
        y = (informative + generator.normal(scale=0.3, size=size) > 0).astype(int)
        X = np.column_stack([raw, informative])
        groups = np.repeat(np.arange(12), size // 12)
        report = cross_validate(X, y, groups, raw)
        self.assertTrue(report["improves_over_raw_confidence"], report)
        self.assertGreater(report["calibrated"]["roc_auc"], 0.85)

    def test_tampered_model_file_is_refused(self):
        import joblib
        from sklearn.dummy import DummyClassifier

        from dronecamp_ia.calibration import Calibrator
        from dronecamp_ia.io import file_hash

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            joblib.dump(DummyClassifier().fit([[0], [1]], [0, 1]), directory / "calibrator.joblib")
            metadata = {"model_file": "calibrator.joblib", "model_sha256": file_hash(directory / "calibrator.joblib"),
                        "taxonomy_sha256": file_hash(REAL.taxonomy_path), "class_names": NAMES,
                        "cross_validation": {"improves_over_raw_confidence": False}}
            write_json(directory / "calibrator.json", metadata)
            with self.assertRaises(ValueError):  # sem ganho medido: recusado por padrão
                Calibrator(REAL, directory)
            Calibrator(REAL, directory, allow_unproven=True)
            (directory / "calibrator.joblib").write_bytes(b"outro conteudo")
            with self.assertRaises(ValueError):
                Calibrator(REAL, directory, allow_unproven=True)


class AdoptionRuleTests(unittest.TestCase):
    def test_average_precision_matches_hand_computed_value(self):
        # Acerto, erro, acerto com 2 positivos: AP = 0,5·1 + 0,5·(2/3).
        self.assertAlmostEqual(average_precision([(0.9, True), (0.8, False), (0.7, True)], 2), 0.8333, places=4)
        self.assertIsNone(average_precision([], 0))

    def _report(self, map50, recall, class_recall):
        report = {name: {"human_boxes": 5, "recall": None} for name in NAMES}
        report[NAMES[2]] = {"human_boxes": 5, "recall": class_recall}
        report["_total"] = {"map50": map50, "recall": recall}
        return report

    def test_candidate_with_gain_and_no_regression_is_adopted(self):
        decision = decide(self._report(0.2, 0.3, 0.4), self._report(0.3, 0.35, 0.4), NAMES)
        self.assertTrue(decision["adopt"], decision)

    def test_class_regression_blocks_adoption_even_with_higher_map(self):
        decision = decide(self._report(0.2, 0.3, 0.6), self._report(0.3, 0.35, 0.2), NAMES)
        self.assertFalse(decision["adopt"])
        self.assertEqual(decision["regressions"][0]["class"], NAMES[2])


class SuccessiveHalvingTests(unittest.TestCase):
    def test_budget_goes_to_the_best_arm_and_test_split_is_never_used(self):
        def fake_trainer(config, data, weights, overrides):
            run = config.root / "runs" / f"fake_{overrides['lr0']}_{overrides['epochs']}"
            (run / "fit").mkdir(parents=True)
            reward = {0.001: 0.3, 0.002: 0.2, 0.0005: 0.1}[overrides["lr0"]] + 0.01 * overrides["epochs"]
            (run / "fit" / "results.csv").write_text(f"epoch, metrics/mAP50(B)\n1, {reward / 2}\n2, {reward}\n",
                                                     encoding="utf-8")
            calls.append(overrides)
            return run

        calls = []
        with tempfile.TemporaryDirectory() as folder:
            config = replace(REAL, root=Path(folder))
            run = successive_halving(config, Path(folder) / "dataset.yaml", arms=None, min_epochs=1, eta=3,
                                     space={"lr0": [0.0005, 0.001, 0.002]}, trainer=fake_trainer)
            state = json.loads((run / "bandit.json").read_text(encoding="utf-8"))
        self.assertTrue(state["complete"])
        self.assertEqual(state["best_parameters"], {"lr0": 0.001})
        self.assertFalse(state["test_used"])
        self.assertEqual([call["epochs"] for call in calls], [1, 1, 1, 3])

    def test_arm_sampling_is_reproducible(self):
        space = {"lr0": [1, 2, 3], "mosaic": [0, 1]}
        self.assertEqual(arms_from_space(space, 3, 5), arms_from_space(space, 3, 5))
        self.assertEqual(len(arms_from_space(space, None, 5)), 6)

    def test_missing_results_has_no_reward(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(best_validation_reward(Path(folder)))


if __name__ == "__main__":
    unittest.main()
