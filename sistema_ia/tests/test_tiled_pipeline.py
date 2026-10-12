"""Pipeline fatiado: plano por tamanho de foto, inferência em janelas, grupos de voo e dataset de treino.

Módulos cobertos
----------------
- ``dronecamp_ia.suggestions``: ``tiling_plan`` (decide se a foto é fatiada) e
  ``detect_boxes`` (inferência em janelas + passada na foto inteira, com as caixas
  devolvidas em coordenadas da foto).
- ``dronecamp_ia.pilot.group_flight_sequences``: agrupa fotos sequenciais de voo (DJI_0194...)
  em blocos para não vazar cenas quase iguais entre treino e teste.
- ``dronecamp_ia.training._tiled_training_dataset``: gera o dataset de janelas para o treino
  piloto e o YAML com caminhos absolutos.

Como rodar (na pasta ``sistema_ia``)
------------------------------------
- Só este arquivo::

      DRONECAMP_SKIP_E2E=1 python -m unittest tests.test_tiled_pipeline -v

- Um teste só::

      python -m unittest tests.test_tiled_pipeline.TilingPlanTests.test_only_large_photos_are_tiled -v

O que cada classe prepara
-------------------------
- ``TilingPlanTests``: nada no disco; só a config de ``config_for``.
- ``TiledInferenceTests``: pasta temporária com uma foto cinza 5472x3648 (tamanho de foto
  DJI de 20 MP) e um ``FakeModel`` no lugar do YOLO (não precisa de pesos nem de GPU).
- ``FlightSequenceTests``: só dicionários de imagem montados por ``item``.
- ``TiledTrainingDatasetTests``: pasta temporária com um dataset piloto mínimo
  (train 2600x1400 = fatiada; val/test 900x600 = inteiras) e um run vazio.

EXTENSAO: para testar uma função nova do fatiamento (ex.: um parâmetro novo em
``pilot.tiling``), acrescente a chave no dict ``TILING`` (ou num ``{**TILING, ...}`` local),
copie ``test_only_large_photos_are_tiled`` ou ``test_windows_are_offset_to_photo_coordinates_and_merged``
e ajuste as asserções. Se a inferência nova chamar outro método do modelo, implemente-o em
``FakeModel``.

ATENCAO: a contagem de 24 janelas depende de ``patch=1280`` e ``overlap=0.2`` em ``TILING``
e do tamanho 5472x3648; mudar qualquer um muda o número esperado.
"""

from dataclasses import replace
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest import mock

import yaml
from PIL import Image

# Garante que o pacote em src/ seja importável mesmo rodando o arquivo direto (sem instalar).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronecamp_ia.config import ProjectConfig  # noqa: E402
from dronecamp_ia.pilot import group_flight_sequences  # noqa: E402
from dronecamp_ia.suggestions import detect_boxes, tiling_plan  # noqa: E402

# ---------------------------------------------------------------------------
# Configuração de fatiamento e dublês do modelo YOLO
# ---------------------------------------------------------------------------

# Mesmo formato de ``pilot.tiling`` em configs/project.yaml:
#   enabled: liga o fatiamento; min_side: maior lado (px) a partir do qual a foto é fatiada;
#   patch: lado da janela (px); overlap: sobreposição entre janelas (fração, 0.2 = 20%);
#   full_image: também roda uma passada na foto inteira; merge_iou: IoU para fundir caixas
#   repetidas entre janelas vizinhas.
# EXTENSAO: parâmetro novo de pilot.tiling: acrescente aqui para que todos os testes o recebam.
TILING = {"enabled": True, "min_side": 2000, "patch": 1280, "overlap": 0.2, "full_image": True, "merge_iou": 0.5}


class _Values:
    """Imita um tensor da Ultralytics: ``.cpu().tolist()`` devolve a lista original."""

    def __init__(self, values):
        """Guarda a lista de valores.

        Parâmetros:
            values: lista Python (caixas xyxy, confianças ou classes).
        """
        self.values = values

    def cpu(self):
        """Retorna o próprio objeto (no tensor real, copia da GPU para a CPU)."""
        return self

    def tolist(self):
        """Retorna a lista guardada."""
        return self.values


class _Result:
    """Imita um ``Results`` da Ultralytics com ``boxes.xyxy``, ``boxes.conf`` e ``boxes.cls``."""

    def __init__(self, boxes):
        """Monta ``self.boxes`` a partir de tuplas.

        Parâmetros:
            boxes: lista de ``(classe, confiança, [x1, y1, x2, y2])`` em pixels do recorte.
        """
        # type("B", (), {...})() cria na hora um objeto simples com os três atributos.
        self.boxes = type("B", (), {"xyxy": _Values([b[2] for b in boxes]), "conf": _Values([b[1] for b in boxes]),
                                    "cls": _Values([b[0] for b in boxes])})()


class FakeModel:
    """Detecta um 'defeito' (classe 1) quando o recorte contém o pixel (3000, 2000).

    Na prática, devolve uma caixa de 40x40 px no centro de CADA janela recebida como
    array e nada na foto inteira (recebida como caminho ``str``). Guarda em
    ``self.calls`` o caminho ou o ``shape`` de cada chamada para os testes contarem.
    """

    # Nomes das classes, como em ``model.names`` do YOLO.
    names = {0: "a", 1: "b"}

    def __init__(self):
        """Começa com a lista de chamadas vazia."""
        self.calls = []

    def predict(self, source, **kwargs):
        """Simula ``YOLO.predict``.

        Parâmetros:
            source: caminho da foto (str) ou recorte (array numpy HxWx3).
            **kwargs: conf, imgsz etc.; ignorados.

        Retorna:
            Iterador com um ``_Result``.
        """
        # Registra a chamada: caminho (foto inteira) ou shape (janela).
        self.calls.append(source if isinstance(source, str) else source.shape)
        if isinstance(source, str):  # foto inteira: nada visível em escala reduzida
            return iter([_Result([])])
        height, width = source.shape[:2]
        found = []
        # O teste conhece a janela pelo tamanho/ordem; a caixa local fica no centro do recorte.
        found.append((1, 0.9, [width / 2 - 20, height / 2 - 20, width / 2 + 20, height / 2 + 20]))
        return iter([_Result(found)])


def config_for(root: Path, tiling: dict | None = TILING) -> ProjectConfig:
    """Cria um ``ProjectConfig`` mínimo em CPU, com ou sem ``pilot.tiling``.

    Parâmetros:
        root: raiz do projeto (pasta temporária ou ``Path(".")``).
        tiling: dict de fatiamento; None/vazio = sem ``pilot.tiling`` (foto inteira).

    Retorna:
        ProjectConfig com imgsz 640 e NMS desligado na predição.
    """
    return ProjectConfig(root=root, model="x.pt", requested_model="x.pt", taxonomy_path=root / "t.json",
                         dataset_path=root / "d.yaml", device="cpu", prediction={"imgsz": 640, "nms": False},
                         training={}, pilot={"tiling": tiling} if tiling else {})


# ---------------------------------------------------------------------------
# tiling_plan: quando fatiar
# ---------------------------------------------------------------------------

class TilingPlanTests(unittest.TestCase):
    """Decisão de fatiar por tamanho da foto (``suggestions.tiling_plan``)."""

    def test_only_large_photos_are_tiled(self) -> None:
        """Só fotos com maior lado >= ``min_side`` são fatiadas; ``enabled: False`` ou config sem tiling desligam.

        Importa porque recortes do laudo (pequenos) devem continuar inteiros, e fotos de
        drone de 20 MP precisam das janelas para os defeitos pequenos aparecerem.
        """
        config = config_for(Path("."))
        self.assertIsNone(tiling_plan(config, 1100, 800), "Recortes de laudo continuam inteiros.")
        self.assertEqual(tiling_plan(config, 5472, 3648)["patch"], 1280)
        # O 4º argumento substitui o pilot.tiling da config.
        self.assertIsNone(tiling_plan(config, 5472, 3648, {"enabled": False}))
        self.assertIsNone(tiling_plan(config_for(Path("."), None), 5472, 3648), "Sem pilot.tiling, foto inteira.")


# ---------------------------------------------------------------------------
# detect_boxes: inferência em janelas
# ---------------------------------------------------------------------------

class TiledInferenceTests(unittest.TestCase):
    """Inferência fatiada com ``FakeModel`` sobre uma foto 5472x3648."""

    def setUp(self) -> None:
        """Grava ``DJI_0194.jpg`` cinza 5472x3648 numa pasta temporária (com acento no nome)."""
        temporary = tempfile.TemporaryDirectory(prefix="dronecamp fatias á ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.photo = self.root / "DJI_0194.jpg"
        Image.new("RGB", (5472, 3648), (120, 120, 120)).save(self.photo, quality=80)

    def test_windows_are_offset_to_photo_coordinates_and_merged(self) -> None:
        """As caixas das janelas voltam às coordenadas da foto, dentro dos limites, com IDs s001...

        Confere também: 24 janelas + 1 passada na foto inteira, e que caixas de centros
        diferentes não são fundidas pelo ``merge_iou``.
        """
        model = FakeModel()
        boxes = detect_boxes(model, self.photo, config_for(self.root), 0.15)
        # Chamadas com array = janelas; com str = foto inteira.
        crops = [call for call in model.calls if not isinstance(call, str)]
        self.assertEqual(len(crops), 24, "5472×3648 com janelas de 1280 e 20% de sobreposição.")
        self.assertEqual(sum(isinstance(call, str) for call in model.calls), 1, "Uma passada na foto inteira.")
        self.assertEqual(len(boxes), 24, "Centros distintos não se fundem.")
        # Toda caixa fica dentro da foto, com x1 < x2 e y1 < y2.
        for box in boxes:
            x1, y1, x2, y2 = box["bbox_xyxy"]
            self.assertTrue(0 <= x1 < x2 <= 5472 and 0 <= y1 < y2 <= 3648)
        # Se o deslocamento da janela não fosse somado, nenhuma caixa passaria de x=1280.
        self.assertTrue(any(box["bbox_xyxy"][0] > 4000 for box in boxes), "Caixas voltam às coordenadas da foto.")
        self.assertEqual(boxes[0]["id"], "s001")

    def test_disabled_tiling_runs_whole_photo_once(self) -> None:
        """Com ``{"enabled": False}``, o modelo é chamado uma única vez com o caminho da foto."""
        model = FakeModel()
        detect_boxes(model, self.photo, config_for(self.root), 0.15, {"enabled": False})
        self.assertEqual(model.calls, [str(self.photo)])


# ---------------------------------------------------------------------------
# group_flight_sequences: blocos de fotos sequenciais
# ---------------------------------------------------------------------------

class FlightSequenceTests(unittest.TestCase):
    """Agrupamento de fotos vizinhas de voo (``pilot.group_flight_sequences``)."""

    def item(self, name: str, scene: str | None = None) -> dict:
        """Monta um registro de imagem mínimo.

        Parâmetros:
            name: nome do arquivo (ex.: "DJI_0194.JPG").
            scene: grupo de cena declarado; None gera o grupo automático
                ``ceasa_lado_a_<12 primeiros do hash>`` (o que a função pode reagrupar).

        Retorna:
            dict com filename, image_sha256 (64 hex derivados do nome), building_group e scene_group.
        """
        # Hash falso mas determinístico: hex do nome repetido e cortado em 64 caracteres.
        digest = (name.encode().hex() * 8)[:64]
        return {"filename": name, "image_sha256": digest, "building_group": "ceasa_lado_a",
                "scene_group": scene or f"ceasa_lado_a_{digest[:12]}"}

    def test_neighbouring_photos_share_a_block_and_declared_scenes_stay(self) -> None:
        """DJI_0194 e DJI_0199 caem no mesmo bloco de 10; DJI_0201 em outro; cena declarada fica.

        Também: ``block`` < 2 desliga o agrupamento (retorna 0 mudanças).
        """
        images = [self.item("DJI_0194.JPG"), self.item("DJI_0199.JPG"), self.item("DJI_0201.JPG"),
                  self.item("pagina_3.png", "ceasa_roof_overview")]
        # Retorno = quantas imagens tiveram o grupo trocado (as 3 DJI).
        self.assertEqual(group_flight_sequences(images, 10), 3)
        self.assertEqual(images[0]["scene_group"], images[1]["scene_group"])
        self.assertNotEqual(images[1]["scene_group"], images[2]["scene_group"])
        self.assertEqual(images[3]["scene_group"], "ceasa_roof_overview")
        self.assertEqual(group_flight_sequences([self.item("DJI_0001.JPG")], 0), 0)


# ---------------------------------------------------------------------------
# _tiled_training_dataset: dataset de janelas para o treino
# ---------------------------------------------------------------------------

class TiledTrainingDatasetTests(unittest.TestCase):
    """Geração do dataset fatiado de treino (``training._tiled_training_dataset``)."""

    def test_tiles_keep_splits_and_absolute_yaml(self) -> None:
        """As janelas mantêm a divisão train/val/test e o YAML gerado usa caminhos absolutos.

        Fotos pequenas (val/test 900x600) continuam inteiras; o run recebe
        ``tiling_manifest.json``.
        """
        from dronecamp_ia.training import _tiled_training_dataset

        with tempfile.TemporaryDirectory(prefix="dronecamp treino fatiado á ") as name:
            # 1. Dataset piloto mínimo: uma foto + um label YOLO por split.
            #    Label "1 0.5 0.5 0.02 0.04" = classe 1, centro (0.5, 0.5), largura 2% e altura 4% (normalizados).
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
            # 2. Pasta do run onde ficam o YAML e o manifesto.
            run = root / "runs" / "pilot_train_x"
            run.mkdir(parents=True)
            config = replace(config_for(root), pilot={"tiling": {**TILING, "negative_ratio": 0.2}})
            # 3. Sem DRONECAMP_FIT_STAGING, as janelas vão para dentro do run (não para outro disco).
            #    mock.patch.dict restaura o ambiente original ao sair do with.
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("DRONECAMP_FIT_STAGING", None)
                dataset, details = _tiled_training_dataset(config, run, source / "dataset.yaml", 0)
            # 4. Confere o YAML gerado e o relatório.
            resolved = yaml.safe_load(dataset.read_text(encoding="utf-8"))
            self.assertTrue(Path(resolved["train"]).is_absolute())
            self.assertEqual(resolved["names"], ["a", "b"])
            self.assertGreaterEqual(len(list(Path(resolved["train"]).iterdir())), 1)
            self.assertEqual(len(list(Path(resolved["val"]).iterdir())), 1, "Foto pequena continua inteira.")
            self.assertEqual(details["patch"], 1280)
            self.assertTrue((run / "tiling_manifest.json").is_file())


# ENTRADA: execução direta do arquivo.
if __name__ == "__main__":
    unittest.main()
