"""Ponta a ponta com Ultralytics real: sem mocks de treino, predição ou exportação.

Usa o peso local models/yolo26l.pt e as fotos/revisões reais do CEASA. O treino é
curto (1 época, imagens pequenas) porque verifica o encadeamento, não a qualidade.
Defina DRONECAMP_SKIP_E2E=1 para pular em máquinas sem os arquivos ou sem tempo.
"""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import json
import os
import unittest

from dronecamp_ia.config import detection_names, load_config, load_taxonomy
from dronecamp_ia.io import file_hash, write_json
from dronecamp_ia.review_data import boxes_to_yolo

REAL = load_config()
WEIGHTS = REAL.root / "models" / "yolo26l.pt"
REGISTRY = REAL.root / "data" / "reviews" / "ceasa_v7_revisao002_ba8cc323c8a1" / "registry.json"


@unittest.skipIf(os.environ.get("DRONECAMP_SKIP_E2E") == "1", "E2E desativado por DRONECAMP_SKIP_E2E=1.")
@unittest.skipUnless(WEIGHTS.is_file() and REGISTRY.is_file(), "Peso yolo26l.pt ou revisão v7 ausente.")
class EndToEndUltralyticsTests(unittest.TestCase):
    """Os testes compartilham um treino real; a ordem numérica é intencional."""

    @classmethod
    def setUpClass(cls):
        from dronecamp_ia.pilot import build_pilot_dataset
        from dronecamp_ia.training import train_pilot

        cls.temporary = TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.config = replace(REAL, root=cls.root)
        cls.names = detection_names(load_taxonomy(REAL.taxonomy_path))
        cls.dataset = cls.root / "data" / "pilot" / "e2e"
        cls.build = build_pilot_dataset(cls.config, [REGISTRY], cls.dataset)
        cls.train_run = train_pilot(cls.config, cls.dataset / "dataset.yaml", str(WEIGHTS),
                              {"epochs": 1, "imgsz": 160, "batch": 8, "plots": False, "patience": 0})
        cls.best = cls.train_run / "fit" / "weights" / "best.pt"
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        cls.sample_images = [Path(item["source_path"]) for item in registry["images"][:2]]

    @classmethod
    def tearDownClass(cls):
        from dronecamp_ia.backend import _ultralytics

        _ultralytics(REAL)  # restaura as pastas reais nas configurações da biblioteca
        cls.temporary.cleanup()

    def test_1_pilot_dataset_keeps_scenes_in_one_split_and_labels_match_review(self):
        manifest = json.loads((self.dataset / "pilot.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["production_ready"])
        splits_by_scene = {}
        for sample in manifest["images"]:
            splits_by_scene.setdefault(sample["scene_group"], set()).add(sample["split"])
        self.assertTrue(all(len(splits) == 1 for splits in splits_by_scene.values()))
        self.assertEqual({sample["split"] for sample in manifest["images"]}, {"train", "val", "test"})
        reviewed = {item["image_sha256"]: item for item in json.loads(REGISTRY.read_text(encoding="utf-8"))["images"]}
        for sample in manifest["images"]:
            item = reviewed[sample["image_sha256"]]
            self.assertTrue(item["human_approved"])
            label = (self.dataset / sample["label"]).read_text(encoding="utf-8")
            self.assertEqual(label, boxes_to_yolo(item["boxes"], item["width"], item["height"]))
        approved = sum(item["human_approved"] is True for item in reviewed.values())
        self.assertEqual(len(manifest["images"]), approved)

    def test_2_real_training_produces_audited_checkpoint(self):
        summary = json.loads((self.train_run / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["complete"], summary["reasons"])
        self.assertEqual(summary["state"], "completed")
        self.assertTrue(summary["pilot"])
        self.assertFalse(summary["production_ready"])
        self.assertTrue(summary["model_class_contract_verified"])
        self.assertGreater(self.best.stat().st_size, 1_000_000)
        provenance = json.loads((self.train_run / "pilot_provenance.json").read_text(encoding="utf-8"))
        self.assertTrue(provenance["valid"])

    def test_3_real_prediction_writes_findings_marked_as_pilot(self):
        from dronecamp_ia.prediction import predict

        folder = self.root / "predict_input"
        folder.mkdir()
        for image in self.sample_images:
            (folder / image.name).write_bytes(image.read_bytes())
        run = predict(self.config, folder, str(self.best))
        execution = json.loads((run / "execution.json").read_text(encoding="utf-8"))
        self.assertTrue(execution["pilot"])
        rows = [json.loads(line) for line in (run / "findings.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["mode"], "candidatos_modelo_piloto_nao_validado")
            self.assertTrue(Path(row["evidence"]).is_file())
            for detection in row["detections"]:
                self.assertIn(detection["class_name"], self.names)
                self.assertIsNone(detection["severity"])

    def test_4_real_onnx_export_matches_pytorch_outputs(self):
        from dronecamp_ia.exporting import export_onnx, verify_onnx_parity

        run = export_onnx(self.config, str(self.best), parity_images=self.sample_images)
        export = json.loads((run / "export.json").read_text(encoding="utf-8"))
        onnx = Path(export["artifact"])
        self.assertTrue(onnx.is_file())
        self.assertEqual(file_hash(onnx), export["sha256"])
        self.assertTrue(export["runtime_parity_validated"], (run / "parity.json").read_text(encoding="utf-8"))
        # Limiar baixo: compara dezenas de caixas, não só a ausência delas.
        parity = verify_onnx_parity(self.config, run / "model.pt", onnx, self.sample_images, 0.001)
        self.assertGreater(sum(item["matched"] for item in parity["images"]), 0)
        self.assertTrue(parity["runtime_parity_validated"], parity)

    def test_5_suggestions_feed_review_and_accepted_boxes_reach_next_dataset(self):
        from dronecamp_ia.pilot import build_pilot_dataset
        from dronecamp_ia.review_data import import_human_feedback
        from dronecamp_ia.suggestions import create_review_for_new_images

        folder = self.root / "fotos_novas"
        folder.mkdir()
        for image in self.sample_images:
            (folder / image.name).write_bytes(image.read_bytes())
        review = self.root / "data" / "reviews" / "fotos_novas_e2e"
        page = create_review_for_new_images(self.config, folder, review, "edificacao_teste", str(self.best), 0.001)
        suggestions = json.loads((review / "suggestions.json").read_text(encoding="utf-8"))
        self.assertTrue(suggestions["pilot"])
        self.assertGreater(suggestions["total_suggestions"], 0)
        html = page.read_text(encoding="utf-8")
        self.assertIn("model_suggestions", html)
        self.assertIn("Modelo piloto (não validado)", html)

        # O revisor aceita uma sugestão: ela passa a ser caixa humana do registro.
        registry_path = review / "registry.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        digest, candidates = next((key, value) for key, value in suggestions["images"].items() if value)
        accepted = {key: candidates[0][key] for key in ("class_id", "bbox_xyxy")}
        feedback = self.root / "feedback.json"
        write_json(feedback, {"schema_version": 1, "registry_sha256": file_hash(registry_path),
                              "reviewer_id": "revisor_e2e", "exported_at": "2026-10-04T00:00:00Z",
                              "images": [{"image_sha256": digest, "status": "positive", "boxes": [accepted],
                                          "notes": "", "confirmed_complete": True}]})
        reviewed_path = review / "registry_revisado.json"
        import_human_feedback(self.config, registry_path, feedback, reviewed_path)
        self.assertEqual(len(registry["images"]), len(self.sample_images))

        # Fotos repetidas em duas revisões contam uma vez, com a decisão mais recente.
        merged = build_pilot_dataset(self.config, [REGISTRY, reviewed_path], self.root / "data" / "pilot" / "e2e_v2")
        manifest = json.loads((self.root / "data" / "pilot" / "e2e_v2" / "pilot.json").read_text(encoding="utf-8"))
        sample = next(item for item in manifest["images"] if item["image_sha256"] == digest)
        label = (self.root / "data" / "pilot" / "e2e_v2" / sample["label"]).read_text(encoding="utf-8")
        self.assertEqual(label.split()[0], str(accepted["class_id"]))
        self.assertEqual(len(label.splitlines()), 1)
        self.assertIn("edificacao_teste", merged["buildings"])
        with (self.root / "data" / "pilot" / "e2e_v2" / "groups.csv").open(encoding="utf-8") as stream:
            self.assertEqual(sum(1 for _ in csv.DictReader(stream)), len(manifest["images"]))


if __name__ == "__main__":
    unittest.main()
