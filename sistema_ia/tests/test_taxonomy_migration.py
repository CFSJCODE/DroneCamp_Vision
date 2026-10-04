"""Expandir classes preserva evidências, mas exige nova revisão humana integral."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image

from dronecamp_ia.backend import check_domain_names
from dronecamp_ia.config import ProjectConfig, detection_names, load_config, load_taxonomy
from dronecamp_ia.io import file_hash
from dronecamp_ia.prediction import serialize_detections
from dronecamp_ia.review_data import import_human_feedback, migrate_review_taxonomy


class _Values:
    """Substituto mínimo de tensor; não baixa modelo nem executa inferência."""

    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class TaxonomyMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.taxonomy = load_taxonomy(self.config.taxonomy_path)
        self.names = detection_names(self.taxonomy)
        self.snapshot_path = self.config.taxonomy_path.with_name("taxonomy_ceasa_v1.json")

    def test_current_active_classes_preserve_ids_and_append_reparo_rufo(self) -> None:
        self.assertEqual(self.names, [
            "telha_quebrada", "telha_ausente", "residuos_telha", "reparo_telha",
            "rufo_ausente", "rufo_quebrado", "residuos_calha", "vegetacao_calha",
            "pedaco_telha", "rufo_deslocado", "pedaco_telha_sobreposto", "fixador_telha_frouxo",
            "reparo_rufo",
        ])
        category = self.taxonomy["classes"][8]
        self.assertEqual(category["id"], 8)
        self.assertEqual(category["slug"], "pedaco_telha")
        self.assertEqual(category["phase"], 1)
        self.assertEqual(self.taxonomy["classes"][9]["slug"], "rufo_deslocado")
        self.assertEqual(self.taxonomy["classes"][9]["phase"], 1)
        for class_id, slug in (
            (10, "pedaco_telha_sobreposto"), (11, "fixador_telha_frouxo"), (12, "reparo_rufo"),
        ):
            with self.subTest(class_id=class_id):
                self.assertEqual(self.taxonomy["classes"][class_id]["slug"], slug)
                self.assertEqual(self.taxonomy["classes"][class_id]["phase"], 1)
        self.assertEqual([category["id"] for category in self.taxonomy["classes"]], list(range(23)))
        corroded = self.taxonomy["classes"][21]
        self.assertEqual(corroded["slug"], "fixador_telha_corroido")
        self.assertEqual(corroded["phase"], 2)
        sealant = self.taxonomy["classes"][22]
        self.assertEqual(sealant["slug"], "reparo_selante_fixador_telha")
        self.assertEqual(sealant["phase"], 2)

    def test_snapshot_preserves_eight_class_contract_and_original_phase2_id8(self) -> None:
        snapshot = load_taxonomy(self.snapshot_path)
        self.assertEqual(detection_names(snapshot), self.names[:8])
        self.assertEqual(snapshot["classes"][8]["slug"], "telha_sobreposta_sem_travamento")
        self.assertEqual(snapshot["classes"][8]["phase"], 2)
        displaced = next(category for category in self.taxonomy["classes"]
                         if category["slug"] == "telha_sobreposta_sem_travamento")
        self.assertEqual(displaced["id"], 19)
        self.assertEqual(displaced["phase"], 2)
        old_fixador = next(category for category in self.taxonomy["classes"]
                           if category["slug"] == "fixador_telha_ausente")
        self.assertEqual(old_fixador["id"], 20)
        self.assertEqual(old_fixador["phase"], 2)

    def test_old_eight_class_checkpoint_cannot_claim_new_thirteen_class_contract(self) -> None:
        snapshot = load_taxonomy(self.snapshot_path)
        checkpoint = SimpleNamespace(names=dict(enumerate(detection_names(snapshot))))
        with self.assertRaises(ValueError):
            check_domain_names(checkpoint, self.names)

    def test_intermediate_snapshot_preserves_nine_class_contract(self) -> None:
        path = self.config.taxonomy_path.with_name("taxonomy_ceasa_v2_pedaco_telha.json")
        snapshot = load_taxonomy(path)
        names = detection_names(snapshot)
        self.assertEqual(names, self.names[:9])
        self.assertEqual(snapshot["classes"][9]["slug"], "fixador_telha_ausente")
        checkpoint = SimpleNamespace(names=dict(enumerate(names)))
        with self.assertRaises(ValueError):
            check_domain_names(checkpoint, self.names)

    def test_previous_snapshot_preserves_ten_classes_and_phase2_fixation_ids(self) -> None:
        path = self.config.taxonomy_path.with_name("taxonomy_ceasa_v3_pedaco_telha_rufo_deslocado.json")
        snapshot = load_taxonomy(path)
        names = detection_names(snapshot)
        self.assertEqual(names, self.names[:10])
        self.assertEqual(snapshot["classes"][10]["slug"], "fixador_telha_corroido")
        self.assertEqual(snapshot["classes"][10]["phase"], 2)
        self.assertEqual(snapshot["classes"][11]["slug"], "fixador_telha_frouxo")
        self.assertEqual(snapshot["classes"][11]["phase"], 2)
        checkpoint = SimpleNamespace(names=dict(enumerate(names)))
        with self.assertRaises(ValueError):
            check_domain_names(checkpoint, self.names)

    def test_previous_snapshot_preserves_twelve_classes_and_original_phase2_id12(self) -> None:
        path = self.config.taxonomy_path.with_name("taxonomy_ceasa_v4_fragmento_sobreposto_fixador_solto.json")
        snapshot = load_taxonomy(path)
        names = detection_names(snapshot)
        self.assertEqual(names, self.names[:12])
        self.assertEqual(snapshot["classes"][12]["slug"], "reparo_selante_fixador_telha")
        self.assertEqual(snapshot["classes"][12]["phase"], 2)
        self.assertEqual(snapshot["classes"][21]["slug"], "fixador_telha_corroido")
        checkpoint = SimpleNamespace(names=dict(enumerate(names)))
        with self.assertRaises(ValueError):
            check_domain_names(checkpoint, self.names)

    def test_matching_thirteen_class_checkpoint_is_accepted(self) -> None:
        checkpoint = SimpleNamespace(names=dict(enumerate(self.names)))
        check_domain_names(checkpoint, self.names)

    def test_new_visual_categories_do_not_inherit_historical_engineering_severity(self) -> None:
        for class_id, slug in (
            (8, "pedaco_telha"), (9, "rufo_deslocado"),
            (10, "pedaco_telha_sobreposto"), (11, "fixador_telha_frouxo"),
            (12, "reparo_rufo"),
        ):
            with self.subTest(class_id=class_id):
                result = SimpleNamespace(
                    names=dict(enumerate(self.names)),
                    boxes=SimpleNamespace(
                        xyxy=_Values([[10, 20, 50, 60]]),
                        conf=_Values([0.999]), cls=_Values([class_id]),
                    ),
                )
                findings = serialize_detections(result, self.taxonomy, demo=False)
                self.assertEqual(len(findings), 1)
                finding = findings[0]
                self.assertEqual(finding["class_id"], class_id)
                self.assertEqual(finding["class_name"], slug)
                self.assertEqual(finding["category_label"], self.taxonomy["classes"][class_id]["label"])
                self.assertIsNone(finding["severity"])
                self.assertEqual(finding["review_status"], "pendente")
                self.assertIsNone(finding["report_reference"]["historical_severity_only"])

    def test_reparo_rufo_label_is_explicit_and_requires_human_review(self) -> None:
        category = self.taxonomy["classes"][12]
        self.assertEqual(category["label"], "Reparo em rufo")
        self.assertIsNone(category["reference_severity"])
        result = SimpleNamespace(
            names=dict(enumerate(self.names)),
            boxes=SimpleNamespace(
                xyxy=_Values([[10, 20, 50, 60]]),
                conf=_Values([0.999]), cls=_Values([12]),
            ),
        )
        finding = serialize_detections(result, self.taxonomy, demo=False)[0]
        self.assertEqual(finding["category_label"], "Reparo em rufo")
        self.assertEqual(finding["review_status"], "pendente")
        self.assertIsNone(finding["severity"])


class ReviewTaxonomyMigrationFixtureTests(unittest.TestCase):
    """Duas classes antigas e duas novas bastam para verificar a migração geral."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="dronecamp migration fixture á ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.previous_taxonomy_path = self.root / "old_taxonomy.json"
        self.current_taxonomy_path = self.root / "new_taxonomy.json"
        self.old_taxonomy = {
            "version": "fixture-old",
            "classes": [self.category(0, "old_0"), self.category(1, "old_1")],
        }
        self.new_taxonomy = {
            "version": "fixture-new",
            "classes": deepcopy(self.old_taxonomy["classes"]) +
                       [self.category(2, "new_2"), self.category(3, "new_3")],
        }
        self.write_json(self.previous_taxonomy_path, self.old_taxonomy)
        self.write_json(self.current_taxonomy_path, self.new_taxonomy)
        self.config = ProjectConfig(
            root=self.root, model="fixture.pt", requested_model="fixture.pt",
            taxonomy_path=self.current_taxonomy_path, dataset_path=self.root / "dataset.yaml",
            device="cpu", prediction={}, training={},
        )
        records = []
        for index, color in enumerate(((160, 70, 30), (30, 70, 160))):
            source = self.root / f"fixture_{index}.png"
            Image.new("RGB", (32, 24), color=color).save(source)
            records.append({
                "image_sha256": file_hash(source), "source_path": str(source),
                "filename": source.name, "width": 32, "height": 24, "pages": [index + 1],
                "status": "positive" if index == 0 else "negative",
                "boxes": [
                    {"class_id": 0, "bbox_xyxy": [2, 3, 10, 13], "note": "old fixture box0"},
                    {"class_id": 1, "bbox_xyxy": [15, 4, 25, 17], "note": "old fixture box1"},
                ] if index == 0 else [],
                "human_approved": True, "technical_status": "human_visual_reviewed",
                "human_reviewer": "synthetic_old_fixture_review",
                "visual_review_status": "second_pass_ai", "building_group": "fixture_building",
                "severity": "fixture_old_value", "notes": "Synthetic test only.",
            })
        self.registry = {
            "schema_version": 1, "version": "fixture-old",
            "taxonomy_sha256": file_hash(self.previous_taxonomy_path), "images": records,
        }
        self.registry_path = self.root / "old_registry.json"
        self.write_json(self.registry_path, self.registry)
        self.output = self.root / "fixture-v2" / "registry.json"

    @staticmethod
    def category(class_id: int, slug: str) -> dict:
        return {"id": class_id, "slug": slug, "label": slug, "phase": 1}

    @staticmethod
    def write_json(path: Path, value: dict) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def migrate(self) -> dict:
        return migrate_review_taxonomy(self.config, self.registry_path, self.previous_taxonomy_path, self.output)

    def test_append_preserves_boxes_image_hashes_and_input_bytes_without_inventing_new_boxes(self) -> None:
        paths = [self.registry_path, self.previous_taxonomy_path, self.current_taxonomy_path]
        paths.extend(Path(image["source_path"]) for image in self.registry["images"])
        before = {path: path.read_bytes() for path in paths}
        result = self.migrate()
        self.assertEqual(before, {path: path.read_bytes() for path in paths})
        self.assertEqual([image["image_sha256"] for image in result["images"]],
                         [image["image_sha256"] for image in self.registry["images"]])
        self.assertEqual([image["boxes"] for image in result["images"]],
                         [image["boxes"] for image in self.registry["images"]])
        self.assertEqual(result["taxonomy_migration"]["new_class_ids"], [2, 3])
        self.assertEqual(result["taxonomy_migration"]["source_registry_sha256"], file_hash(self.registry_path))
        self.assertEqual(result["taxonomy_sha256"], file_hash(self.current_taxonomy_path))
        self.assertEqual(result["parent_registry_version"], "fixture-old")
        self.assertEqual(result["version"], "fixture-v2")
        self.assertFalse(any(box["class_id"] in (2, 3) for image in result["images"] for box in image["boxes"]))

    def test_previous_negative_is_ambiguous_until_new_classes_are_reviewed(self) -> None:
        second = self.migrate()["images"][1]
        self.assertEqual(second["status"], "ambiguous")
        self.assertEqual(second["previous_status"], "negative")
        self.assertEqual(second["boxes"], [])
        self.assertTrue(second["taxonomy_recheck_pending"])
        self.assertFalse(second["human_approved"])

    def prepare_ten_class_expansion(self) -> None:
        """Simula um catálogo antigo sem tocar nas fotos ou revisões do projeto."""
        old_active = [self.category(class_id, f"old_{class_id}") for class_id in range(10)]
        corroded = {**self.category(10, "fixador_telha_corroido"), "phase": 2}
        loose = {**self.category(11, "fixador_telha_frouxo"), "phase": 2}
        self.old_taxonomy["classes"] = old_active + [corroded, loose]
        self.new_taxonomy["classes"] = deepcopy(old_active) + [
            self.category(10, "pedaco_telha_sobreposto"),
            self.category(11, "fixador_telha_frouxo"),
            {**self.category(12, "fixador_telha_corroido"), "phase": 2},
        ]
        self.write_json(self.previous_taxonomy_path, self.old_taxonomy)
        self.write_json(self.current_taxonomy_path, self.new_taxonomy)
        self.registry["taxonomy_sha256"] = file_hash(self.previous_taxonomy_path)
        self.write_json(self.registry_path, self.registry)

    def test_ten_to_twelve_classes_adds_ids10_and11_without_relabeling_old_boxes(self) -> None:
        # A promoção de uma categoria do catálogo também expande o contrato ativo.
        # Os arquivos de origem são sintéticos; nenhuma aprovação real é criada.
        self.prepare_ten_class_expansion()
        before = self.registry_path.read_bytes()

        result = self.migrate()

        self.assertEqual(self.registry_path.read_bytes(), before)
        self.assertEqual(result["taxonomy_migration"]["new_class_ids"], [10, 11])
        self.assertEqual(result["taxonomy_migration"]["new_active_classes"][10:],
                         ["pedaco_telha_sobreposto", "fixador_telha_frouxo"])
        self.assertEqual([image["boxes"] for image in result["images"]],
                         [image["boxes"] for image in self.registry["images"]])
        self.assertEqual([image["image_sha256"] for image in result["images"]],
                         [image["image_sha256"] for image in self.registry["images"]])
        for image in result["images"]:
            self.assertEqual(image["classes_reviewed_before_migration"], list(range(10)))
            self.assertTrue(image["taxonomy_recheck_pending"])
            self.assertFalse(image["human_approved"])
            self.assertFalse(any(box["class_id"] in (10, 11) for box in image["boxes"]))

    def test_repeated_migration_preserves_actual_eight_class_review_scope(self) -> None:
        self.prepare_ten_class_expansion()
        for image in self.registry["images"]:
            image["taxonomy_recheck_pending"] = True
            image["classes_reviewed_before_migration"] = list(range(8))
            image["visual_review_status"] = "prior_taxonomy_ai_review"
        self.write_json(self.registry_path, self.registry)
        before = self.registry_path.read_bytes()

        result = self.migrate()

        self.assertEqual(self.registry_path.read_bytes(), before)
        for image in result["images"]:
            self.assertEqual(image["classes_reviewed_before_migration"], list(range(8)))
            self.assertTrue(image["taxonomy_recheck_pending"])
            self.assertFalse(image["human_approved"])

    def test_pending_previous_review_without_scope_does_not_invent_reviewed_classes(self) -> None:
        self.prepare_ten_class_expansion()
        self.registry["images"][0]["taxonomy_recheck_pending"] = True
        self.write_json(self.registry_path, self.registry)

        first = self.migrate()["images"][0]

        self.assertEqual(first["classes_reviewed_before_migration"], [])
        self.assertTrue(first["taxonomy_recheck_pending"])

    def test_invalid_previous_review_scope_is_rejected_without_migration_output(self) -> None:
        self.prepare_ten_class_expansion()
        for invalid in ([0, "1"], [0, 10], [0, True], "0,1", [0, [1]], [0, 0]):
            with self.subTest(invalid=invalid):
                self.registry["images"][0]["taxonomy_recheck_pending"] = True
                self.registry["images"][0]["classes_reviewed_before_migration"] = invalid
                self.write_json(self.registry_path, self.registry)
                with self.assertRaises(ValueError):
                    self.migrate()
                self.assertFalse(self.output.exists())

    def test_old_human_approvals_are_revoked_for_every_image_after_expansion(self) -> None:
        result = self.migrate()
        for image in result["images"]:
            self.assertFalse(image["human_approved"])
            self.assertEqual(image["technical_status"], "pending_human_review")
            self.assertTrue(image["taxonomy_recheck_pending"])
            self.assertEqual(image["classes_reviewed_before_migration"], [0, 1])
            self.assertEqual(image["previous_visual_review_status"], "second_pass_ai")
            self.assertEqual(image["visual_review_status"], "prior_taxonomy_human_review")
            self.assertTrue(image["previous_human_approved"])
            self.assertIsNone(image["severity"])
        summary = json.loads((self.output.parent / "review_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["human_approved_images"], 0)
        self.assertFalse(summary["ready_for_independent_evaluation"])

    def test_human_review_origin_notes_and_hashes_are_preserved_without_current_approval(self) -> None:
        for image in self.registry["images"]:
            image["human_notes"] = "Synthetic human correction preserved verbatim."
            image["human_review_at"] = "2026-10-03T22:15:00Z"
        self.write_json(self.registry_path, self.registry)
        before = self.registry_path.read_bytes()

        result = self.migrate()

        self.assertEqual(self.registry_path.read_bytes(), before)
        for image in result["images"]:
            self.assertEqual(image["visual_review_status"], "prior_taxonomy_human_review")
            self.assertTrue(image["previous_human_approved"])
            self.assertEqual(image["human_notes"], "Synthetic human correction preserved verbatim.")
            self.assertEqual(image["human_reviewer"], "synthetic_old_fixture_review")
            self.assertEqual(image["human_review_at"], "2026-10-03T22:15:00Z")
            historical = image["historical_human_review"]
            self.assertEqual(historical["taxonomy_sha256"], file_hash(self.previous_taxonomy_path))
            self.assertEqual(historical["source_registry_sha256"], file_hash(self.registry_path))
            self.assertEqual(historical["classes_reviewed"], [0, 1])
            self.assertEqual(historical["reviewer"], image["human_reviewer"])
            self.assertEqual(historical["review_at"], image["human_review_at"])
            self.assertFalse(image["human_approved"])
            self.assertTrue(image["taxonomy_recheck_pending"])
            self.assertEqual(image["technical_status"], "pending_human_review")

    def test_imported_completed_feedback_keeps_human_provenance_when_expanding_taxonomy(self) -> None:
        # Este fluxo usa somente fotos e revisor sintéticos no diretório temporário.
        # Ele reproduz a importação contra o catálogo da página original antes de
        # migrar suas correções para as classes acrescentadas posteriormente.
        for image in self.registry["images"]:
            image["human_approved"] = False
            image["technical_status"] = "pending_human_review"
            image["taxonomy_recheck_pending"] = True
            image["classes_reviewed_before_migration"] = [0]
            image["visual_review_status"] = "prior_taxonomy_ai_review"
        self.write_json(self.registry_path, self.registry)
        feedback_path = self.root / "synthetic_feedback.json"
        self.write_json(feedback_path, {
            "schema_version": 1, "registry_sha256": file_hash(self.registry_path),
            "reviewer_id": "synthetic_user_reviewer", "exported_at": "2026-10-03T22:15:00Z",
            "images": [
                {"image_sha256": self.registry["images"][0]["image_sha256"],
                 "status": "positive", "confirmed_complete": True,
                 "boxes": [{"class_id": 1, "bbox_xyxy": [3, 2, 14, 16]}],
                 "notes": "Synthetic corrected occurrence."},
                {"image_sha256": self.registry["images"][1]["image_sha256"],
                 "status": "ambiguous", "confirmed_complete": False,
                 "boxes": [], "notes": "Synthetic uncertain decision."},
            ],
        })
        old_config = ProjectConfig(
            root=self.root, model="fixture.pt", requested_model="fixture.pt",
            taxonomy_path=self.previous_taxonomy_path, dataset_path=self.root / "dataset.yaml",
            device="cpu", prediction={}, training={},
        )
        imported_path = self.root / "synthetic-human-v1" / "registry.json"
        imported = import_human_feedback(old_config, self.registry_path, feedback_path, imported_path)
        imported_bytes = imported_path.read_bytes()

        result = migrate_review_taxonomy(
            self.config, imported_path, self.previous_taxonomy_path, self.output,
        )

        self.assertEqual(imported_path.read_bytes(), imported_bytes)
        self.assertEqual(result["human_feedback"], imported["human_feedback"])
        self.assertEqual(result["human_feedback"]["sha256"], file_hash(feedback_path))
        completed, pending = result["images"]
        self.assertEqual(completed["boxes"], imported["images"][0]["boxes"])
        self.assertEqual(completed["visual_review_status"], "prior_taxonomy_human_review")
        self.assertEqual(completed["classes_reviewed_before_migration"], [0, 1])
        self.assertEqual(completed["human_notes"], "Synthetic corrected occurrence.")
        self.assertEqual(completed["historical_human_review"]["source_registry_sha256"], file_hash(imported_path))
        self.assertEqual(pending["visual_review_status"], "prior_taxonomy_ai_review")
        self.assertEqual(pending["classes_reviewed_before_migration"], [0])
        self.assertEqual(pending["human_notes"], "Synthetic uncertain decision.")
        self.assertFalse(any(image["human_approved"] for image in result["images"]))
        self.assertTrue(all(image["taxonomy_recheck_pending"] for image in result["images"]))

    def test_repeated_migration_keeps_human_origin_review_scope_and_hash_chain(self) -> None:
        first = self.migrate()
        first_bytes = self.output.read_bytes()
        expanded_taxonomy = self.root / "expanded_taxonomy.json"
        expanded = deepcopy(self.new_taxonomy)
        expanded["version"] = "fixture-expanded-again"
        expanded["classes"].append(self.category(4, "new_4"))
        self.write_json(expanded_taxonomy, expanded)
        second_config = ProjectConfig(
            root=self.root, model="fixture.pt", requested_model="fixture.pt",
            taxonomy_path=expanded_taxonomy, dataset_path=self.root / "dataset.yaml",
            device="cpu", prediction={}, training={},
        )
        second_output = self.root / "fixture-v3" / "registry.json"

        second = migrate_review_taxonomy(
            second_config, self.output, self.current_taxonomy_path, second_output,
        )

        self.assertEqual(self.output.read_bytes(), first_bytes)
        for first_image, second_image in zip(first["images"], second["images"]):
            self.assertEqual(second_image["boxes"], first_image["boxes"])
            self.assertEqual(second_image["classes_reviewed_before_migration"], [0, 1])
            self.assertEqual(second_image["historical_human_review"], first_image["historical_human_review"])
            self.assertEqual(second_image["visual_review_status"], "prior_taxonomy_human_review")
            self.assertEqual(second_image["previous_visual_review_status"], "prior_taxonomy_human_review")
            self.assertTrue(second_image["previous_human_approved"])
            self.assertFalse(second_image["human_approved"])
            self.assertTrue(second_image["taxonomy_recheck_pending"])
        history = second["taxonomy_migration_history"]
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0], first["taxonomy_migration"])
        self.assertEqual(history[0]["source_registry_sha256"], file_hash(self.registry_path))
        self.assertEqual(history[1]["source_registry_sha256"], file_hash(self.output))
        self.assertEqual(history[1]["source_registry_path"], str(self.output.resolve()))
        self.assertEqual(history[1], second["taxonomy_migration"])
        summary = json.loads((second_output.parent / "review_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["human_approved_images"], 0)
        self.assertEqual(summary["taxonomy_recheck_pending_images"], 2)

    def test_completed_ten_class_human_review_expands_scope_while_pending_review_stays_at_eight(self) -> None:
        self.prepare_ten_class_expansion()
        for image in self.registry["images"]:
            image["classes_reviewed_before_migration"] = list(range(8))
            image["visual_review_status"] = "prior_taxonomy_ai_review"
        completed, pending = self.registry["images"]
        completed["taxonomy_recheck_pending"] = False
        pending["human_approved"] = False
        pending["technical_status"] = "pending_human_review"
        pending["taxonomy_recheck_pending"] = True
        pending["status"] = "ambiguous"
        self.write_json(self.registry_path, self.registry)

        completed, pending = self.migrate()["images"]

        self.assertEqual(completed["classes_reviewed_before_migration"], list(range(10)))
        self.assertEqual(completed["visual_review_status"], "prior_taxonomy_human_review")
        self.assertTrue(completed["previous_human_approved"])
        self.assertEqual(pending["classes_reviewed_before_migration"], list(range(8)))
        self.assertEqual(pending["visual_review_status"], "prior_taxonomy_ai_review")
        self.assertFalse(pending["previous_human_approved"])
        self.assertNotIn("historical_human_review", pending)
        self.assertFalse(completed["human_approved"])
        self.assertFalse(pending["human_approved"])

    def test_inconsistent_human_flags_cannot_create_historical_human_approval(self) -> None:
        changes = (
            {"human_approved": "true"},
            {"technical_status": "pending_human_review"},
            {"human_reviewer": "   "},
            {"human_reviewer": None},
            {"status": "ambiguous"},
            {"taxonomy_recheck_pending": True, "classes_reviewed_before_migration": [0]},
        )
        for index, change in enumerate(changes):
            with self.subTest(change=change):
                registry = deepcopy(self.registry)
                registry["images"][0].update(change)
                self.write_json(self.registry_path, registry)
                output = self.root / f"invalid-human-{index}" / "registry.json"

                first = migrate_review_taxonomy(
                    self.config, self.registry_path, self.previous_taxonomy_path, output,
                )["images"][0]

                self.assertEqual(first["visual_review_status"], "prior_taxonomy_ai_review")
                self.assertFalse(first["previous_human_approved"])
                self.assertNotIn("historical_human_review", first)
                self.assertFalse(first["human_approved"])
                self.assertTrue(first["taxonomy_recheck_pending"])

    def test_malformed_migration_history_is_rejected_without_new_output(self) -> None:
        for history in ("history", [None], ["migration"]):
            with self.subTest(history=history):
                self.registry["taxonomy_migration_history"] = history
                self.write_json(self.registry_path, self.registry)
                with self.assertRaises(ValueError):
                    self.migrate()
                self.assertFalse(self.output.exists())

    def test_existing_migrated_output_cannot_be_overwritten(self) -> None:
        self.output.parent.mkdir()
        previous = b"preserve old migration output"
        self.output.write_bytes(previous)
        with self.assertRaises(ValueError):
            self.migrate()
        self.assertEqual(self.output.read_bytes(), previous)

    def test_reordering_existing_class_slugs_is_rejected(self) -> None:
        self.new_taxonomy["classes"][0]["slug"] = "old_1"
        self.new_taxonomy["classes"][1]["slug"] = "old_0"
        self.write_json(self.current_taxonomy_path, self.new_taxonomy)
        with self.assertRaises(ValueError):
            self.migrate()
        self.assertFalse(self.output.exists())

    def test_nonexpanding_taxonomy_is_rejected(self) -> None:
        self.new_taxonomy["classes"] = deepcopy(self.old_taxonomy["classes"])
        self.write_json(self.current_taxonomy_path, self.new_taxonomy)
        with self.assertRaises(ValueError):
            self.migrate()
        self.assertFalse(self.output.exists())

    def test_previous_taxonomy_must_match_original_registry_hash(self) -> None:
        self.old_taxonomy["version"] = "fixture-old-altered"
        self.write_json(self.previous_taxonomy_path, self.old_taxonomy)
        with self.assertRaises(ValueError):
            self.migrate()
        self.assertFalse(self.output.exists())

    def test_changed_original_image_blocks_migration_before_any_output(self) -> None:
        source = Path(self.registry["images"][0]["source_path"])
        Image.new("RGB", (32, 24), color="white").save(source)
        with self.assertRaises(ValueError):
            self.migrate()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
