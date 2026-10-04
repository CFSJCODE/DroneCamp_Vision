"""Ciclo de aprendizado: duplicatas, caminhos de outra máquina e propostas de IA."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from PIL import Image

from dronecamp_ia.config import ProjectConfig
from dronecamp_ia.io import file_hash, resolve_local_path
from dronecamp_ia.pilot import build_pilot_dataset, consolidate_duplicates, validate_pilot_dataset
from dronecamp_ia.review_data import apply_ai_proposals, import_human_feedback

ACCEPTED = "Sugestão do modelo aceita (confiança {}%); conferida na revisão local."


def box(class_id: int, coordinates: list[int], note: str = "Caixa conferida/editada na revisão local.") -> dict:
    return {"class_id": class_id, "bbox_xyxy": coordinates, "note": note}


class ConsolidateDuplicatesTests(unittest.TestCase):
    def test_reviewer_box_wins_over_accepted_suggestion_of_same_object(self):
        drawn = box(2, [10, 10, 50, 50])
        result = consolidate_duplicates([box(2, [11, 10, 51, 50], ACCEPTED.format(97)), drawn], 0.7)
        self.assertEqual(result, [drawn])

    def test_highest_confidence_suggestion_wins_among_suggestions(self):
        low, high = box(3, [0, 0, 40, 40], ACCEPTED.format(20)), box(3, [1, 1, 40, 40], ACCEPTED.format(85))
        self.assertEqual(consolidate_duplicates([low, high], 0.7), [high])

    def test_other_class_or_low_overlap_is_kept_in_original_order(self):
        boxes = [box(2, [0, 0, 40, 40]), box(6, [0, 0, 40, 40]), box(2, [30, 30, 80, 80])]
        self.assertEqual(consolidate_duplicates(boxes, 0.7), boxes)

    def test_disabled_policy_keeps_every_box(self):
        boxes = [box(2, [0, 0, 40, 40]), box(2, [0, 0, 40, 40])]
        self.assertEqual(consolidate_duplicates(boxes, None), boxes)


class ResolveLocalPathTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "sistema_ia"
        self.target = self.root / "data" / "reference" / "foto.jpg"
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"x")

    def test_windows_and_posix_paths_from_other_machines_are_reanchored(self):
        for recorded in ("E:\\Acadêmico\\PUC\\sistema_ia\\data\\reference\\foto.jpg",
                         "/home/outra/clone/sistema_ia/data/reference/foto.jpg"):
            with self.subTest(recorded=recorded):
                self.assertEqual(resolve_local_path(recorded, self.root), self.target)

    def test_existing_absolute_path_and_unknown_path_are_left_alone(self):
        self.assertEqual(resolve_local_path(str(self.target), self.root), self.target)
        missing = "E:\\sistema_ia\\data\\nao_existe.jpg"
        self.assertEqual(resolve_local_path(missing, self.root), Path(missing))


class LearningLoopFixture(unittest.TestCase):
    """Aprovações abaixo existem só no diretório temporário."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="dronecamp ciclo ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "sistema_ia"
        (self.root / "configs").mkdir(parents=True)
        self.taxonomy_path = self.root / "configs" / "taxonomy.json"
        self.taxonomy_path.write_text(json.dumps({"version": "fixture", "classes": [
            {"id": 0, "slug": "telha_quebrada", "label": "Telha quebrada", "phase": 1},
            {"id": 1, "slug": "residuos_telha", "label": "Resíduos sobre telhas", "phase": 1},
        ]}), encoding="utf-8")
        self.config = ProjectConfig(
            root=self.root, model="fixture.pt", requested_model="fixture.pt", taxonomy_path=self.taxonomy_path,
            dataset_path=self.root / "dataset.yaml", device="cpu", prediction={}, training={},
            pilot={"duplicate_iou": 0.7})
        (self.root / "fotos").mkdir()
        self.records = []
        for index in range(8):
            path = self.root / "fotos" / f"foto_{index}.png"
            Image.new("RGB", (64, 48), color=(20 * index, 90, 160)).save(path)
            boxes = [box(index % 2, [4, 4, 30, 30])]
            if index == 0:
                boxes.append(box(0, [5, 4, 31, 30], ACCEPTED.format(90)))
            self.records.append({
                "image_sha256": file_hash(path), "filename": path.name, "width": 64, "height": 48,
                # Caminho gravado em outra máquina: precisa ser reancorado neste clone.
                "source_path": f"E:\\outra maquina\\sistema_ia\\fotos\\{path.name}",
                "status": "positive", "boxes": boxes, "human_approved": True,
                "technical_status": "human_visual_reviewed", "human_reviewer": "fixture",
                "building_group": "fixture", "scene_group": f"cena_{index}", "notes": "", "severity": None,
            })
        self.registry_path = self.root / "revisao" / "registry.json"
        self.registry_path.parent.mkdir()
        self.write(self.registry_path, {"schema_version": 1, "version": "revisao",
                                        "taxonomy_sha256": file_hash(self.taxonomy_path), "images": self.records})

    @staticmethod
    def write(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class PilotDatasetTests(LearningLoopFixture):
    def test_duplicates_are_consolidated_and_dataset_survives_being_moved(self):
        output = self.root / "data" / "pilot" / "v1"
        build_pilot_dataset(self.config, [self.registry_path], output)
        manifest = json.loads((output / "pilot.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["duplicate_policy"]["boxes_reviewed"], 9)
        self.assertEqual(manifest["duplicate_policy"]["boxes_for_training"], 8)
        first = next(sample for sample in manifest["images"] if sample["image_sha256"] == self.records[0]["image_sha256"])
        self.assertEqual((first["boxes"], first["duplicates_consolidated"]), (1, 1))
        moved = self.root / "data" / "pilot" / "v1_movido"
        shutil.copytree(output, moved)
        shutil.rmtree(output)
        self.assertTrue(validate_pilot_dataset(self.config, moved / "dataset.yaml")["valid"])

    def test_human_feedback_import_accepts_paths_recorded_on_other_machine(self):
        for record in self.records:
            record.update(human_approved=False, technical_status="pending_human_review")
        self.write(self.registry_path, {"schema_version": 1, "version": "revisao",
                                        "taxonomy_sha256": file_hash(self.taxonomy_path), "images": self.records})
        feedback = self.root / "feedback.json"
        self.write(feedback, {"schema_version": 1, "registry_sha256": file_hash(self.registry_path),
                              "reviewer_id": "fixture", "exported_at": "2026-10-04T00:00:00Z",
                              "images": [{"image_sha256": self.records[1]["image_sha256"], "status": "negative",
                                          "boxes": [], "confirmed_complete": True}]})
        result = import_human_feedback(self.config, self.registry_path, feedback, self.root / "v2" / "registry.json")
        self.assertTrue(result["images"][1]["human_approved"])


class AiProposalTests(LearningLoopFixture):
    def setUp(self):
        super().setUp()
        for record in self.records:
            record.update(human_approved=False, technical_status="pending_human_review", status="ambiguous", boxes=[])
        self.records[0].update(human_approved=True, technical_status="human_visual_reviewed",
                               status="positive", boxes=[box(0, [4, 4, 30, 30])])
        self.write(self.registry_path, {"schema_version": 1, "version": "revisao",
                                        "taxonomy_sha256": file_hash(self.taxonomy_path), "images": self.records})
        self.proposals = self.root / "propostas.json"
        self.output = self.root / "revisao_ia" / "registry.json"

    def proposal(self, index: int, **changes) -> dict:
        value = {"image_sha256": self.records[index]["image_sha256"], "status": "positive", "notes": "Proposta.",
                 "boxes": [{**box(0, [2, 2, 40, 20], "Trinca."), "proposed_new_class": "telha_trincada"}]}
        return {**value, **changes}

    def run_proposals(self, *images: dict, annotator: str = "claude_visual") -> dict:
        self.write(self.proposals, {"schema_version": 1, "annotator": annotator, "images": list(images)})
        return apply_ai_proposals(self.config, self.registry_path, self.proposals, self.output)

    def test_proposals_fill_review_without_approval_and_list_new_classes(self):
        result = self.run_proposals(self.proposal(1))
        item = result["images"][1]
        self.assertFalse(item["human_approved"])
        self.assertEqual((item["status"], item["annotator"], item["technical_status"]),
                         ("positive", "claude_visual", "pending_human_review"))
        self.assertEqual(result["ai_proposals"]["proposed_new_classes"], ["telha_trincada"])

    def test_human_decision_is_never_overwritten(self):
        with self.assertRaisesRegex(ValueError, "decisão humana"):
            self.run_proposals(self.proposal(0))
        self.assertFalse(self.output.exists())

    def test_invalid_class_slug_status_or_annotator_are_rejected(self):
        bad_slug = self.proposal(1, boxes=[{**box(0, [2, 2, 40, 20]), "proposed_new_class": "Telha Trincada"}])
        for images, annotator in (([bad_slug], "claude_visual"), ([self.proposal(1, boxes=[])], "claude_visual"),
                                  ([self.proposal(1)], " ")):
            with self.subTest(annotator=annotator):
                with self.assertRaises(ValueError):
                    self.run_proposals(*images, annotator=annotator)


if __name__ == "__main__":
    unittest.main()
