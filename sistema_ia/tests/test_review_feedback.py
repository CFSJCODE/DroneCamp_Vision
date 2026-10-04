"""Travas de revisão: testes sintéticos não criam aprovações nos dados reais."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from dronecamp_ia.config import ProjectConfig, detection_names, load_taxonomy
from dronecamp_ia.io import file_hash
from dronecamp_ia.review_data import (
    build_review_registry, import_human_feedback, review_image_size, validate_boxes,
)
from dronecamp_ia.review_render import render_review_package


class ReviewBoxValidationTests(unittest.TestCase):
    def test_full_image_and_fractional_coordinates_are_valid(self) -> None:
        validate_boxes([
            {"class_id": 0, "bbox_xyxy": [0, 0, 20, 16]},
            {"class_id": 1, "bbox_xyxy": [1.5, 2.5, 6.75, 10.5]},
        ], width=20, height=16, class_count=2)

    def test_invalid_class_ids_are_rejected(self) -> None:
        for class_id in (-1, 2, 1.0, True, "0", None):
            with self.subTest(class_id=class_id):
                with self.assertRaises(ValueError):
                    validate_boxes([{"class_id": class_id, "bbox_xyxy": [1, 1, 5, 5]}], 20, 16, 2)

    def test_nonfinite_or_nonnumeric_coordinates_are_rejected(self) -> None:
        for coordinate in (math.nan, math.inf, -math.inf, True, "1", None):
            with self.subTest(coordinate=coordinate):
                with self.assertRaises(ValueError):
                    validate_boxes([{"class_id": 0, "bbox_xyxy": [coordinate, 1, 5, 5]}], 20, 16, 2)

    def test_negative_outside_or_empty_boxes_are_rejected(self) -> None:
        cases = (
            [-1, 1, 5, 5], [1, -1, 5, 5], [1, 1, 21, 5],
            [1, 1, 5, 17], [5, 1, 5, 5], [1, 5, 5, 5],
            [6, 1, 5, 5], [1, 6, 5, 5],
        )
        for coordinates in cases:
            with self.subTest(coordinates=coordinates):
                with self.assertRaises(ValueError):
                    validate_boxes([{"class_id": 0, "bbox_xyxy": coordinates}], 20, 16, 2)

    def test_invalid_container_and_coordinate_length_are_rejected(self) -> None:
        for boxes in (None, {}, [{"class_id": 0, "bbox_xyxy": [1, 1, 5]}],
                      [{"class_id": 0, "bbox_xyxy": (1, 1, 5, 5)}]):
            with self.subTest(boxes=boxes):
                with self.assertRaises(ValueError):
                    validate_boxes(boxes, 20, 16, 2)

    def test_box_limit_prevents_unbounded_feedback(self) -> None:
        with self.assertRaises(ValueError):
            validate_boxes([{"class_id": 0, "bbox_xyxy": [1, 1, 5, 5]}] * 2001, 20, 16, 2)

    def test_nonobject_box_is_reported_as_validation_error(self) -> None:
        for box in (None, 42, "box", [0, 1, 1, 5, 5]):
            with self.subTest(box=box):
                with self.assertRaises(ValueError):
                    validate_boxes([box], 20, 16, 2)


class ReviewFeedbackTests(unittest.TestCase):
    """Toda aprovação abaixo existe apenas no diretório temporário da fixture."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="dronecamp revisão teste á ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "configs").mkdir()
        self.directory = self.root / "revisao"
        self.directory.mkdir()
        self.taxonomy_path = self.root / "configs" / "taxonomy.json"
        self.taxonomy = {
            "version": "fixture-v1",
            "classes": [
                {"id": 0, "slug": "telha_quebrada", "label": "Telha quebrada", "phase": 1},
                {"id": 1, "slug": "residuos_calha", "label": "Resíduos em calha", "phase": 1},
            ],
        }
        self.write_json(self.taxonomy_path, self.taxonomy)
        self.config = ProjectConfig(
            root=self.root, model="fixture.pt", requested_model="fixture.pt",
            taxonomy_path=self.taxonomy_path, dataset_path=self.root / "dataset.yaml",
            device="cpu", prediction={}, training={},
        )
        self.box = {"class_id": 0, "bbox_xyxy": [2, 2, 9, 10], "note": "Fixture sintética."}
        records = []
        for index, color in enumerate(((180, 80, 40), (40, 80, 180))):
            path = self.root / f"fixture_{index}.png"
            Image.new("RGB", (20, 16), color=color).save(path)
            records.append({
                "image_sha256": file_hash(path), "filename": path.name,
                "source_path": str(path), "width": 20, "height": 16,
                "pages": [index + 1], "status": "positive" if index == 0 else "negative",
                "boxes": [deepcopy(self.box)] if index == 0 else [],
                "human_approved": False, "technical_status": "pending_human_review",
                "visual_review_status": "second_pass_ai", "building_group": "fixture_building",
                "notes": "Proposta sintética de teste, sem aprovação real.", "severity": None,
            })
        self.registry = {
            "schema_version": 1, "version": "fixture-v1",
            "taxonomy_sha256": file_hash(self.taxonomy_path), "images": records,
        }
        self.registry_path = self.directory / "registry.json"
        self.write_json(self.registry_path, self.registry)
        self.feedback_path = self.root / "feedback.json"
        self.output = self.directory / "registry_human.json"

    @staticmethod
    def write_json(path: Path, value: dict) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def feedback(self, *, index: int = 0, status: str = "positive", complete=True, boxes=None) -> dict:
        if boxes is None:
            boxes = [deepcopy(self.box)] if status == "positive" else []
        return {
            "schema_version": 1, "registry_sha256": file_hash(self.registry_path),
            "reviewer_id": "synthetic_fixture_reviewer", "exported_at": "2026-10-03T00:00:00Z",
            "images": [{
                "image_sha256": self.registry["images"][index]["image_sha256"],
                "status": status, "confirmed_complete": complete,
                "boxes": boxes, "notes": "Teste temporário; não corresponde a decisão humana real.",
            }],
        }

    def run_import(self, feedback: dict) -> dict:
        self.write_json(self.feedback_path, feedback)
        return import_human_feedback(self.config, self.registry_path, self.feedback_path, self.output)

    def use_exif_source(self, orientation: int) -> Path:
        """Atualize os hashes da fixture para isolar orientação, sem simular adulteração."""
        source = self.root / "fixture_orientation.jpg"
        exif = Image.Exif()
        exif[274] = orientation
        Image.new("RGB", (20, 16), color=(80, 100, 140)).save(source, exif=exif)
        self.registry["images"][0].update(
            image_sha256=file_hash(source), source_path=str(source), filename=source.name,
        )
        self.write_json(self.registry_path, self.registry)
        return source

    def test_positive_requires_exact_true_for_complete_confirmation(self) -> None:
        for complete in (False, None, 1, "true"):
            with self.subTest(complete=complete):
                with self.assertRaises(ValueError):
                    self.run_import(self.feedback(complete=complete))
                self.assertFalse(self.output.exists())

    def test_negative_requires_complete_confirmation(self) -> None:
        with self.assertRaises(ValueError):
            self.run_import(self.feedback(index=1, status="negative", complete=False))

    def test_confirmed_positive_approves_only_selected_image_and_keeps_severity_pending(self) -> None:
        feedback = self.feedback()
        feedback["images"][0]["severity"] = "gravissima"
        result = self.run_import(feedback)
        first, second = result["images"]
        self.assertTrue(first["human_approved"])
        self.assertEqual(first["technical_status"], "human_visual_reviewed")
        self.assertIsNone(first["severity"])
        self.assertEqual(first["human_reviewer"], "synthetic_fixture_reviewer")
        self.assertFalse(second["human_approved"])
        self.assertEqual(second["technical_status"], "pending_human_review")
        self.assertEqual(result["human_feedback"]["reviewed_images"], 1)

    def test_confirmed_negative_has_no_boxes_and_is_explicitly_approved(self) -> None:
        result = self.run_import(self.feedback(index=1, status="negative"))
        second = result["images"][1]
        self.assertEqual(second["status"], "negative")
        self.assertEqual(second["boxes"], [])
        self.assertTrue(second["human_approved"])

    def test_ambiguous_draft_retains_boxes_without_becoming_approval(self) -> None:
        result = self.run_import(self.feedback(status="ambiguous", complete=False, boxes=[deepcopy(self.box)]))
        first = result["images"][0]
        self.assertEqual(first["status"], "ambiguous")
        self.assertEqual(first["boxes"], [self.box])
        self.assertFalse(first["human_approved"])
        self.assertEqual(first["technical_status"], "pending_human_review")

    def test_excluded_feedback_cannot_be_approved_by_confirmation_checkbox(self) -> None:
        result = self.run_import(self.feedback(status="excluded", complete=True, boxes=[deepcopy(self.box)]))
        self.assertFalse(result["images"][0]["human_approved"])
        self.assertEqual(result["images"][0]["technical_status"], "pending_human_review")

    def test_pending_browser_draft_is_not_a_negative_or_importable_decision(self) -> None:
        with self.assertRaises(ValueError):
            self.run_import(self.feedback(status="pending", complete=False))
        self.assertFalse(self.output.exists())

    def test_feedback_requires_nonempty_image_decision_list(self) -> None:
        for index, decisions in enumerate(([], None, {}, "images")):
            with self.subTest(decisions=decisions):
                self.output = self.directory / f"invalid_feedback_{index}.json"
                feedback = self.feedback()
                feedback["images"] = decisions
                with self.assertRaises(ValueError):
                    self.run_import(feedback)
                self.assertFalse(self.output.exists())

    def test_positive_without_boxes_and_negative_with_boxes_are_rejected(self) -> None:
        cases = (("positive", []), ("negative", [deepcopy(self.box)]))
        for status, boxes in cases:
            with self.subTest(status=status):
                with self.assertRaises(ValueError):
                    self.run_import(self.feedback(status=status, boxes=boxes))
                self.assertFalse(self.output.exists())

    def test_feedback_from_another_registry_version_is_rejected(self) -> None:
        feedback = self.feedback()
        feedback["registry_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.run_import(feedback)

    def test_feedback_schema_cannot_change_silently(self) -> None:
        feedback = self.feedback()
        feedback["schema_version"] = 2
        with self.assertRaises(ValueError):
            self.run_import(feedback)

    def test_changed_taxonomy_is_rejected_before_approval(self) -> None:
        self.taxonomy["version"] = "fixture-v2"
        self.write_json(self.taxonomy_path, self.taxonomy)
        with self.assertRaises(ValueError):
            self.run_import(self.feedback())

    def test_duplicate_taxonomy_ids_are_rejected(self) -> None:
        self.taxonomy["classes"][1]["id"] = 0
        self.write_json(self.taxonomy_path, self.taxonomy)
        with self.assertRaises(ValueError):
            self.run_import(self.feedback())

    def test_changed_original_source_is_rejected_before_approval(self) -> None:
        source = Path(self.registry["images"][0]["source_path"])
        Image.new("RGB", (20, 16), color="white").save(source)
        with self.assertRaises(ValueError):
            self.run_import(self.feedback())

    def test_review_image_size_accepts_raw_pixels_without_implicit_rotation(self) -> None:
        source = Path(self.registry["images"][0]["source_path"])
        self.assertEqual(review_image_size(source), (20, 16))
        self.assertEqual(review_image_size(self.use_exif_source(1)), (20, 16))

    def test_review_image_size_rejects_exif_rotation_and_mirroring_without_rewriting(self) -> None:
        for orientation in range(2, 9):
            with self.subTest(orientation=orientation):
                source = self.use_exif_source(orientation)
                before = source.read_bytes()
                with self.assertRaises(ValueError):
                    review_image_size(source)
                self.assertEqual(source.read_bytes(), before)

    def test_import_cannot_approve_exif_rotated_source_even_with_matching_hashes(self) -> None:
        source = self.use_exif_source(6)
        before = source.read_bytes()
        with self.assertRaises(ValueError):
            self.run_import(self.feedback())
        self.assertFalse(self.output.exists())
        self.assertEqual(source.read_bytes(), before)

    def test_import_rejects_registry_dimensions_different_from_unchanged_photo(self) -> None:
        self.registry["images"][0]["width"] = 200
        self.write_json(self.registry_path, self.registry)
        feedback = self.feedback()
        # Caixa válida para o registro errado, mas fora do PNG real de largura20.
        feedback["images"][0]["boxes"][0]["bbox_xyxy"] = [30, 2, 40, 10]
        with self.assertRaises(ValueError):
            self.run_import(feedback)
        self.assertFalse(self.output.exists())

    def test_active_class_ids_must_be_contiguous_even_when_global_ids_are_valid(self) -> None:
        self.taxonomy["classes"][1]["phase"] = 2
        self.taxonomy["classes"].append({"id": 2, "slug": "another_active", "label": "Outra ativa", "phase": 1})
        self.write_json(self.taxonomy_path, self.taxonomy)
        loaded = load_taxonomy(self.taxonomy_path)
        self.assertEqual([item["id"] for item in loaded["classes"]], [0, 1, 2])
        with self.assertRaises(ValueError):
            detection_names(loaded)
        # Hash alinhado à taxonomia deixa a regressão focar o conjunto ativo.
        self.registry["taxonomy_sha256"] = file_hash(self.taxonomy_path)
        self.write_json(self.registry_path, self.registry)
        feedback = self.feedback()
        feedback["images"][0]["boxes"][0]["class_id"] = 1
        with self.assertRaises(ValueError):
            self.run_import(feedback)
        self.assertFalse(self.output.exists())

    def test_build_registry_rejects_exif_rotated_proposal_before_persisting_review(self) -> None:
        source = self.use_exif_source(6)
        original_directory = self.root / "data/reference/ceasa"
        original_directory.mkdir(parents=True)
        original = original_directory / source.name
        original.write_bytes(source.read_bytes())
        digest = file_hash(original)
        self.write_json(original_directory / "manifest.json", {
            "source_pdf_sha256": "0" * 64,
            "images": [{"sha256": digest, "filename": original.name, "page": 1}],
        })
        review_directory = self.root / "build_rotated_fixture"
        proposals = review_directory / "proposals"
        proposals.mkdir(parents=True)
        self.write_json(proposals / "fixture.json", {
            "schema_version": 1, "annotator": "fixture_ai",
            "images": [{"image_sha256": digest, "width": 20, "height": 16,
                        "status": "positive", "boxes": [deepcopy(self.box)]}],
        })
        with self.assertRaises(ValueError):
            build_review_registry(self.config, review_directory)
        self.assertFalse((review_directory / "registry.json").exists())

    def test_duplicate_feedback_image_ids_are_rejected(self) -> None:
        feedback = self.feedback()
        feedback["images"].append(deepcopy(feedback["images"][0]))
        with self.assertRaises(ValueError):
            self.run_import(feedback)
        self.assertFalse(self.output.exists())

    def test_unknown_feedback_image_id_is_rejected(self) -> None:
        feedback = self.feedback()
        feedback["images"][0]["image_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.run_import(feedback)

    def test_duplicate_registry_image_ids_cannot_be_silently_collapsed(self) -> None:
        self.registry["images"].append(deepcopy(self.registry["images"][0]))
        self.write_json(self.registry_path, self.registry)
        with self.assertRaises(ValueError):
            self.run_import(self.feedback())
        self.assertFalse(self.output.exists())

    def test_existing_output_cannot_be_overwritten(self) -> None:
        before = b"preserve previous reviewed version"
        self.output.write_bytes(before)
        with self.assertRaises(ValueError):
            self.run_import(self.feedback())
        self.assertEqual(self.output.read_bytes(), before)

    def test_reviewer_identifier_is_required(self) -> None:
        feedback = self.feedback()
        feedback["reviewer_id"] = "   "
        with self.assertRaises(ValueError):
            self.run_import(feedback)

    def test_import_preserves_registry_feedback_taxonomy_and_original_images(self) -> None:
        feedback = self.feedback()
        self.write_json(self.feedback_path, feedback)
        paths = [self.registry_path, self.feedback_path, self.taxonomy_path]
        paths.extend(Path(item["source_path"]) for item in self.registry["images"])
        before = {path: path.read_bytes() for path in paths}
        import_human_feedback(self.config, self.registry_path, self.feedback_path, self.output)
        self.assertTrue(self.output.is_file())
        self.assertEqual(before, {path: path.read_bytes() for path in paths})

    def test_render_rejects_changed_taxonomy_instead_of_relabeling_old_boxes(self) -> None:
        self.taxonomy["classes"][0]["label"] = "Categoria trocada após a anotação"
        self.write_json(self.taxonomy_path, self.taxonomy)
        with self.assertRaises(ValueError):
            render_review_package(self.config, self.registry_path)

    def test_render_rejects_registry_dimensions_different_from_photo(self) -> None:
        self.registry["images"][0]["height"] = 160
        self.write_json(self.registry_path, self.registry)
        with self.assertRaises(ValueError):
            render_review_package(self.config, self.registry_path)
        self.assertFalse((self.directory / "index.html").exists())

    def test_render_rejects_exif_rotated_source_instead_of_showing_displaced_boxes(self) -> None:
        self.use_exif_source(6)
        with self.assertRaises(ValueError):
            render_review_package(self.config, self.registry_path)
        self.assertFalse((self.directory / "index.html").exists())

    def test_render_versions_labels_so_ambiguous_image_has_no_current_label(self) -> None:
        first_version = file_hash(self.registry_path)
        render_review_package(self.config, self.registry_path)
        digest = self.registry["images"][0]["image_sha256"]
        label = self.directory / "labels_proposed" / first_version / f"{digest}.txt"
        self.assertTrue(label.is_file())
        before = label.read_bytes()
        self.registry["images"][0]["status"] = "ambiguous"
        self.write_json(self.registry_path, self.registry)
        second_version = file_hash(self.registry_path)
        self.assertNotEqual(first_version, second_version)
        render_review_package(self.config, self.registry_path)
        current_label = self.directory / "labels_proposed" / second_version / f"{digest}.txt"
        self.assertFalse(current_label.exists(), "Versão atual não pode reaproveitar rótulo antigo para imagem ambígua.")
        self.assertEqual(label.read_bytes(), before, "A anotação histórica permanece na pasta de sua versão.")

    def test_render_notice_uses_all_classes_still_pending_in_the_registry(self) -> None:
        # Caso sintético: a segunda classe já existia, mas ainda não foi conferida.
        for item in self.registry["images"]:
            item["taxonomy_recheck_pending"] = True
            item["classes_reviewed_before_migration"] = [0]
        self.write_json(self.registry_path, self.registry)
        render_review_package(self.config, self.registry_path)
        payload = json.loads((self.directory / "browser_data.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["classes_requiring_review"], ["Resíduos em calha"])
        html = (self.directory / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("Há categorias novas nesta revisão: Pedaço de telha", html)

    def test_render_preserves_human_history_and_distinguishes_current_approval(self) -> None:
        first = self.registry["images"][0]
        first.update(visual_review_status="prior_taxonomy_human_review", previous_human_approved=True,
                     human_approved=False, taxonomy_recheck_pending=True,
                     classes_reviewed_before_migration=[0], human_notes="Correção histórica sintética.")
        self.write_json(self.registry_path, self.registry)
        render_review_package(self.config, self.registry_path)
        payload = json.loads((self.directory / "browser_data.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["images"][0]["human_notes"], "Correção histórica sintética.")
        self.assertTrue(payload["images"][0]["previous_human_approved"])
        self.assertFalse(payload["images"][0]["human_approved"])
        summary = json.loads((self.directory / "review_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["historical_human_review_images"], 1)
        self.assertEqual(summary["human_approved_images"], 0)


if __name__ == "__main__":
    unittest.main()
