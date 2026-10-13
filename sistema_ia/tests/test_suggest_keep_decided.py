"""Sugestões novas não reclassificam fotos já decididas (``suggest --keep-decided``).

Testa ``suggestions.suggest_for_registry`` com o modelo, a inferência e a página
substituídos por falsos: fotos aprovadas no registro ou listadas em ``keep`` mantêm
as sugestões do ``suggestions.json`` anterior (mesmos ids) e não passam pelo modelo;
as demais recebem as sugestões novas com o prefixo do hash dos pesos.
"""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest import mock

from PIL import Image

from dronecamp_ia import suggestions
from dronecamp_ia.config import load_config
from dronecamp_ia.io import file_hash, write_json

REAL = load_config(Path(__file__).resolve().parents[1] / "configs" / "project.yaml")


class KeepDecidedTests(unittest.TestCase):
    """Fotos decididas preservam as sugestões anteriores; as outras são refeitas."""

    def test_decided_photos_keep_previous_suggestions_and_skip_the_model(self):
        """Aprovada no registro e listada em keep: mesmas sugestões; pendente: sugestão nova."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            config = replace(REAL, root=root)
            items = []
            for shade, (name, approved) in enumerate((("aprovada", True), ("decidida", False), ("pendente", False))):
                path = root / f"{name}.jpg"
                Image.new("RGB", (64, 48), (40 * shade, 90, 90)).save(path)  # cores diferentes = hashes diferentes
                items.append({"image_sha256": file_hash(path), "source_path": str(path), "filename": path.name,
                              "width": 64, "height": 48, "human_approved": approved})
            registry = root / "review" / "registry.json"
            write_json(registry, {"taxonomy_sha256": file_hash(config.taxonomy_path), "images": items})
            old = {digest["image_sha256"]: [{"id": "velho-s001", "class_id": 0, "bbox_xyxy": [1, 1, 9, 9], "confidence": 0.9}]
                   for digest in items}
            write_json(registry.parent / "suggestions.json",
                       {"schema_version": 1, "registry_sha256": file_hash(registry), "weights_sha256": "velho", "images": old})
            weights = root / "best.pt"
            weights.write_bytes(b"pesos falsos")
            model = SimpleNamespace(dronecamp_source=str(weights), dronecamp_runtime="falso", names={})
            seen = []

            def fake_detect(_model, path, _config, _conf):
                seen.append(Path(path).name)
                return [{"id": "s001", "class_id": 1, "bbox_xyxy": [2, 2, 8, 8], "confidence": 0.6}]

            with mock.patch.object(suggestions, "load_inference_model", return_value=model), \
                    mock.patch.object(suggestions, "check_domain_names", return_value=[]), \
                    mock.patch.object(suggestions, "pilot_inference_config", side_effect=lambda cfg, _m: cfg), \
                    mock.patch.object(suggestions, "detect_boxes", side_effect=fake_detect), \
                    mock.patch("dronecamp_ia.review_render.render_review_package", return_value=registry):
                suggestions.suggest_for_registry(config, registry, str(weights), keep={items[1]["image_sha256"]})
            result = json.loads((registry.parent / "suggestions.json").read_text(encoding="utf-8"))
            self.assertEqual(seen, ["pendente.jpg"])
            self.assertEqual(result["images"][items[0]["image_sha256"]], old[items[0]["image_sha256"]])
            self.assertEqual(result["images"][items[1]["image_sha256"]], old[items[1]["image_sha256"]])
            new = result["images"][items[2]["image_sha256"]]
            self.assertEqual(new[0]["id"], f"{file_hash(weights)[:8]}-s001")
            self.assertEqual(result["kept_previous"]["images"], 2)


if __name__ == "__main__":
    unittest.main()
