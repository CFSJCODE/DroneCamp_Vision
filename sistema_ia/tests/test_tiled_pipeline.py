"""Pipeline fatiado: plano por tamanho de foto, inferência em janelas, grupos de voo e dataset de treino."""

from dataclasses import replace
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest import mock

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronecamp_ia.config import ProjectConfig  # noqa: E402
from dronecamp_ia.pilot import group_flight_sequences  # noqa: E402
from dronecamp_ia.suggestions import detect_boxes, tiling_plan  # noqa: E402

TILING = {"enabled": True, "min_side": 2000, "patch": 1280, "overlap": 0.2, "full_image": True, "merge_iou": 0.5}


class _Values:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class _Result:
    def __init__(self, boxes):
        self.boxes = type("B", (), {"xyxy": _Values([b[2] for b in boxes]), "conf": _Values([b[1] for b in boxes]),
                                    "cls": _Values([b[0] for b in boxes])})()


class FakeModel:
    """Detecta um 'defeito' (classe 1) quando o recorte contém o pixel (3000, 2000)."""

    names = {0: "a", 1: "b"}

    def __init__(self):
        self.calls = []

    def predict(self, source, **kwargs):
        self.calls.append(source if isinstance(source, str) else source.shape)
        if isinstance(source, str):  # foto inteira: nada visível em escala reduzida
            return iter([_Result([])])
        height, width = source.shape[:2]
        found = []
        # O teste conhece a janela pelo tamanho/ordem; a caixa local fica no centro do recorte.
        found.append((1, 0.9, [width / 2 - 20, height / 2 - 20, width / 2 + 20, height / 2 + 20]))
        return iter([_Result(found)])


def config_for(root: Path, tiling: dict | None = TILING) -> ProjectConfig:
    return ProjectConfig(root=root, model="x.pt", requested_model="x.pt", taxonomy_path=root / "t.json",
                         dataset_path=root / "d.yaml", device="cpu", prediction={"imgsz": 640, "nms": False},
                         training={}, pilot={"tiling": tiling} if tiling else {})


class TilingPlanTests(unittest.TestCase):
    def test_only_large_photos_are_tiled(self) -> None:
        config = config_for(Path("."))
        self.assertIsNone(tiling_plan(config, 1100, 800), "Recortes de laudo continuam inteiros.")
        self.assertEqual(tiling_plan(config, 5472, 3648)["patch"], 1280)
        self.assertIsNone(tiling_plan(config, 5472, 3648, {"enabled": False}))
        self.assertIsNone(tiling_plan(config_for(Path("."), None), 5472, 3648), "Sem pilot.tiling, foto inteira.")


class TiledInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="dronecamp fatias á ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.photo = self.root / "DJI_0194.jpg"
        Image.new("RGB", (5472, 3648), (120, 120, 120)).save(self.photo, quality=80)

    def test_windows_are_offset_to_photo_coordinates_and_merged(self) -> None:
        model = FakeModel()
        boxes = detect_boxes(model, self.photo, config_for(self.root), 0.15)
        crops = [call for call in model.calls if not isinstance(call, str)]
        self.assertEqual(len(crops), 24, "5472×3648 com janelas de 1280 e 20% de sobreposição.")
        self.assertEqual(sum(isinstance(call, str) for call in model.calls), 1, "Uma passada na foto inteira.")
        self.assertEqual(len(boxes), 24, "Centros distintos não se fundem.")
        for box in boxes:
            x1, y1, x2, y2 = box["bbox_xyxy"]
            self.assertTrue(0 <= x1 < x2 <= 5472 and 0 <= y1 < y2 <= 3648)
        self.assertTrue(any(box["bbox_xyxy"][0] > 4000 for box in boxes), "Caixas voltam às coordenadas da foto.")
        self.assertEqual(boxes[0]["id"], "s001")

    def test_disabled_tiling_runs_whole_photo_once(self) -> None:
        model = FakeModel()
        detect_boxes(model, self.photo, config_for(self.root), 0.15, {"enabled": False})
        self.assertEqual(model.calls, [str(self.photo)])


class FlightSequenceTests(unittest.TestCase):
    def item(self, name: str, scene: str | None = None) -> dict:
        digest = (name.encode().hex() * 8)[:64]
        return {"filename": name, "image_sha256": digest, "building_group": "ceasa_lado_a",
                "scene_group": scene or f"ceasa_lado_a_{digest[:12]}"}

    def test_neighbouring_photos_share_a_block_and_declared_scenes_stay(self) -> None:
        images = [self.item("DJI_0194.JPG"), self.item("DJI_0199.JPG"), self.item("DJI_0201.JPG"),
                  self.item("pagina_3.png", "ceasa_roof_overview")]
        self.assertEqual(group_flight_sequences(images, 10), 3)
        self.assertEqual(images[0]["scene_group"], images[1]["scene_group"])
        self.assertNotEqual(images[1]["scene_group"], images[2]["scene_group"])
        self.assertEqual(images[3]["scene_group"], "ceasa_roof_overview")
        self.assertEqual(group_flight_sequences([self.item("DJI_0001.JPG")], 0), 0)


class TiledTrainingDatasetTests(unittest.TestCase):
    def test_tiles_keep_splits_and_absolute_yaml(self) -> None:
        from dronecamp_ia.training import _tiled_training_dataset

        with tempfile.TemporaryDirectory(prefix="dronecamp treino fatiado á ") as name:
            root = Path(name)
            source = root / "pilot"
            for split, size in (("train", (2600, 1400)), ("val", (900, 600)), ("test", (900, 600))):
                (source / "images" / split).mkdir(parents=True)
                (source / "labels" / split).mkdir(parents=True)
                Image.new("RGB", size, (90, 90, 90)).save(source / "images" / split / f"{split}.jpg")
                (source / "labels" / split / f"{split}.txt").write_text("1 0.5 0.5 0.02 0.04\n", encoding="utf-8")
            (source / "dataset.yaml").write_text(yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val",
                                                                   "test": "images/test", "names": ["a", "b"]}),
                                                 encoding="utf-8")
            run = root / "runs" / "pilot_train_x"
            run.mkdir(parents=True)
            config = replace(config_for(root), pilot={"tiling": {**TILING, "negative_ratio": 0.2}})
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("DRONECAMP_FIT_STAGING", None)
                dataset, details = _tiled_training_dataset(config, run, source / "dataset.yaml", 0)
            resolved = yaml.safe_load(dataset.read_text(encoding="utf-8"))
            self.assertTrue(Path(resolved["train"]).is_absolute())
            self.assertEqual(resolved["names"], ["a", "b"])
            self.assertGreaterEqual(len(list(Path(resolved["train"]).iterdir())), 1)
            self.assertEqual(len(list(Path(resolved["val"]).iterdir())), 1, "Foto pequena continua inteira.")
            self.assertEqual(details["patch"], 1280)
            self.assertTrue((run / "tiling_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
