"""Inferência ONNX na GPU (DirectML): proveniência do ONNX, falha explícita e paridade com o .pt."""

from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dronecamp_ia import backend  # noqa: E402
from dronecamp_ia.config import ProjectConfig  # noqa: E402
from dronecamp_ia.io import file_hash  # noqa: E402

EXPORT = ROOT / "runs" / "export_20261007T024620Z_516be2d6" / "model.onnx"
PILOT_TEST = ROOT / "data" / "pilot" / "ceasa_v9_piloto_s42" / "images" / "test"


class OnnxProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="dronecamp onnx á ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = ProjectConfig(root=self.root, model="x.pt", requested_model="x.pt",
                                    taxonomy_path=self.root / "t.json", dataset_path=self.root / "d.yaml",
                                    device="cpu", prediction={}, training={}, pilot={})
        source = self.root / "runs" / "pilot_train_x" / "fit" / "weights" / "best.pt"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"pesos piloto")
        self.onnx = self.root / "runs" / "export_x" / "model.onnx"
        self.onnx.parent.mkdir(parents=True)
        self.onnx.write_bytes(b"grafo onnx")
        self.export = {"source_weights": "runs\\pilot_train_x\\fit\\weights\\best.pt",
                       "runtime": {"checkpoint_sha256": file_hash(source)}, "sha256": file_hash(self.onnx)}
        self.source = source

    def write_export(self) -> None:
        (self.onnx.parent / "export.json").write_text(json.dumps(self.export), encoding="utf-8")

    def test_source_checkpoint_is_resolved_and_hash_checked(self) -> None:
        self.write_export()
        self.assertEqual(backend.onnx_source_checkpoint(self.config, self.onnx), self.source.resolve())

    def test_changed_onnx_or_checkpoint_is_refused(self) -> None:
        self.write_export()
        self.onnx.write_bytes(b"outro grafo")
        with self.assertRaises(ValueError):
            backend.onnx_source_checkpoint(self.config, self.onnx)
        self.export["sha256"] = file_hash(self.onnx)
        self.export["runtime"]["checkpoint_sha256"] = "0" * 64
        self.write_export()
        with self.assertRaises(ValueError):
            backend.onnx_source_checkpoint(self.config, self.onnx)

    def test_onnx_without_export_manifest_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            backend.onnx_source_checkpoint(self.config, self.onnx)

    def test_unknown_provider_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            backend.load_inference_model(self.config, str(self.onnx), "cuda")

    def test_missing_directml_fails_loudly(self) -> None:
        import onnxruntime

        with mock.patch.object(onnxruntime, "get_available_providers", return_value=["CPUExecutionProvider"]):
            with self.assertRaises(RuntimeError):
                backend._directml_session(str(self.onnx))


@unittest.skipIf(os.environ.get("DRONECAMP_SKIP_E2E") == "1", "E2E desativado por DRONECAMP_SKIP_E2E=1.")
@unittest.skipUnless(EXPORT.is_file() and PILOT_TEST.is_dir(), "ONNX exportado do piloto ou dataset v9 ausente.")
class DirectMLParityTests(unittest.TestCase):
    def test_directml_boxes_match_pytorch(self) -> None:
        import onnxruntime

        if "DmlExecutionProvider" not in onnxruntime.get_available_providers():
            self.skipTest("DirectML indisponível neste computador.")
        from dronecamp_ia.config import load_config
        from dronecamp_ia.exporting import compare_outputs
        from dronecamp_ia.suggestions import detect_boxes
        from dronecamp_ia.training import pilot_inference_config

        config = load_config()
        source = backend.onnx_source_checkpoint(config, EXPORT)
        reference = backend.load_inference_model(config, str(source))
        gpu = backend.load_inference_model(config, str(EXPORT), "directml")
        self.assertEqual(gpu.dronecamp_runtime, "onnx-directml")
        self.assertEqual(gpu.dronecamp_source, source)
        images = sorted(PILOT_TEST.iterdir())[:3]
        for image in images:
            boxes = [detect_boxes(model, image, pilot_inference_config(config, model), 0.05) for model in (reference, gpu)]
            pairs = [[(b["class_id"], b["confidence"], b["bbox_xyxy"]) for b in value] for value in boxes]
            self.assertTrue(compare_outputs(pairs[0], pairs[1], 0.05)["equivalent"], image.name)
        session = gpu.predictor.model.backend.session
        self.assertEqual(session.get_providers()[0], "DmlExecutionProvider", "A sessão usada é a da GPU.")


if __name__ == "__main__":
    unittest.main()
