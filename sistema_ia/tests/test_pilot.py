"""Regras do piloto e da paridade, sem carregar a Ultralytics.

Módulos cobertos
----------------
- ``dronecamp_ia.pilot.split_by_scene``: divide as cenas entre treino/validação/teste
  (``{scene_group: "train" | "val" | "test"}``), sem deixar uma cena em dois conjuntos e
  mantendo no treino classes com uma única foto.
- ``dronecamp_ia.exporting.compare_outputs``: compara as caixas do .pt com as do .onnx
  exportado (paridade), tolerando só oscilações junto ao limiar de confiança.

Como rodar (na pasta ``sistema_ia``)
------------------------------------
- Só este arquivo::

      DRONECAMP_SKIP_E2E=1 python -m unittest tests.test_pilot -v

- Um teste só::

      python -m unittest tests.test_pilot.ParityComparisonTests.test_flip_next_to_threshold_is_tolerated -v

O que cada classe prepara
-------------------------
- ``PilotSplitTests``: 21 imagens em memória (nada no disco): 20 em 10 cenas de 2 fotos
  (``cena_0``..``cena_9``) com classes 0, 1 e 2 alternadas, mais ``cena_rara`` com a única
  foto da classe 7.
- ``ParityComparisonTests``: nada; cada teste monta tuplas ``(classe, confiança, [x1, y1, x2, y2])``.

EXTENSAO: para testar uma regra nova da divisão (ex.: agrupar por edificação), acrescente o
campo em ``image(...)`` e copie ``test_scene_never_crosses_splits_and_all_splits_exist``.
Para uma tolerância nova da paridade, copie ``test_sub_pixel_change_on_thin_box_is_equivalent``
com as caixas do caso real que motivou a regra.
"""

import unittest

from dronecamp_ia.exporting import compare_outputs
from dronecamp_ia.pilot import split_by_scene


# ---------------------------------------------------------------------------
# Helper de imagem
# ---------------------------------------------------------------------------

def image(digest: str, scene: str, classes: list[int]) -> dict:
    """Monta um registro de imagem mínimo para ``split_by_scene``.

    Parâmetros:
        digest: SHA-256 (64 caracteres hexadecimais; pode ser falso).
        scene: ``scene_group`` da foto (fotos da mesma cena ficam no mesmo conjunto).
        classes: IDs de classe; cada um vira uma caixa ``[0, 0, 1, 1]``.

    Retorna:
        dict com ``image_sha256``, ``scene_group`` e ``boxes``.
    """
    return {"image_sha256": digest, "scene_group": scene,
            "boxes": [{"class_id": class_id, "bbox_xyxy": [0, 0, 1, 1]} for class_id in classes]}


# ---------------------------------------------------------------------------
# split_by_scene: divisão treino/validação/teste por cena
# ---------------------------------------------------------------------------

class PilotSplitTests(unittest.TestCase):
    """Divisão por cena do dataset piloto.

    Importa porque fotos da mesma cena em treino e teste inflariam as métricas
    (o modelo "já viu" a cena).
    """

    def setUp(self):
        """Cria 20 fotos em 10 cenas (classes 0..2) e uma cena rara com a única foto da classe 7."""
        # f"{index:064x}" = índice em hexadecimal com 64 dígitos (formato de SHA-256).
        # index // 2 coloca duas fotos seguidas na mesma cena; index % 3 alterna as classes 0, 1, 2.
        self.images = [image(f"{index:064x}", f"cena_{index // 2}", [index % 3]) for index in range(20)]
        self.images.append(image("f" * 64, "cena_rara", [7]))

    def test_scene_never_crosses_splits_and_all_splits_exist(self):
        """Cada cena recebe exatamente um conjunto, e os três conjuntos (train/val/test) existem."""
        # fractions = (validação 15%, teste 15%); semente 42.
        assignment = split_by_scene(self.images, 42, (0.15, 0.15))
        self.assertEqual(set(assignment.values()), {"train", "val", "test"})
        self.assertEqual(set(assignment), {item["scene_group"] for item in self.images})

    def test_class_with_single_photo_stays_in_training(self):
        """A cena da classe que só tem uma foto fica no treino para qualquer semente (0..9)."""
        for seed in range(10):
            self.assertEqual(split_by_scene(self.images, seed, (0.15, 0.15))["cena_rara"], "train")

    def test_split_is_deterministic_for_the_same_seed(self):
        """Mesma semente, mesma divisão (reprodutível entre execuções)."""
        self.assertEqual(split_by_scene(self.images, 7, (0.15, 0.15)), split_by_scene(self.images, 7, (0.15, 0.15)))

    def test_too_few_scenes_are_rejected(self):
        """Com uma única cena não dá para formar os três conjuntos: ValueError com "insuficientes"."""
        with self.assertRaisesRegex(ValueError, "insuficientes"):
            split_by_scene([image("a" * 64, "unica", [0])], 42, (0.15, 0.15))


# ---------------------------------------------------------------------------
# compare_outputs: paridade .pt × .onnx
# ---------------------------------------------------------------------------

class ParityComparisonTests(unittest.TestCase):
    """Regras de equivalência entre as caixas do .pt (referência) e do .onnx (candidato).

    Em todos os testes o limiar de confiança das inferências é 0.25 (3º argumento).
    """

    def test_identical_outputs_are_equivalent(self):
        """As mesmas caixas em outra ordem são equivalentes (o pareamento não depende da ordem)."""
        boxes = [(0, 0.9, [0, 0, 10, 10]), (1, 0.5, [20, 20, 40, 40])]
        self.assertTrue(compare_outputs(boxes, list(reversed(boxes)), 0.25)["equivalent"])

    def test_box_missing_well_above_threshold_breaks_parity(self):
        """Caixa de confiança 0.9 que sumiu no ONNX quebra a paridade e aparece em ``missing_in_export``."""
        result = compare_outputs([(0, 0.9, [0, 0, 10, 10])], [], 0.25)
        self.assertFalse(result["equivalent"])
        self.assertEqual(result["missing_in_export"][0]["class_id"], 0)

    def test_flip_next_to_threshold_is_tolerated(self):
        """Caixa com 0.255 (logo acima do limiar 0.25) pode sumir num runtime sem quebrar a paridade."""
        self.assertTrue(compare_outputs([(0, 0.255, [0, 0, 10, 10])], [], 0.25)["equivalent"])

    def test_class_change_or_confidence_drift_breaks_parity(self):
        """Mesma caixa com outra classe, ou confiança 0.9 → 0.8, quebra a paridade."""
        self.assertFalse(compare_outputs([(0, 0.9, [0, 0, 10, 10])], [(1, 0.9, [0, 0, 10, 10])], 0.25)["equivalent"])
        self.assertFalse(compare_outputs([(0, 0.9, [0, 0, 10, 10])], [(0, 0.8, [0, 0, 10, 10])], 0.25)["equivalent"])

    def test_sub_pixel_change_on_thin_box_is_equivalent(self):
        """Diferença de 0,1 px numa caixa muito fina (0,6 px de altura) é equivalente.

        Numa caixa assim o IoU cai muito com qualquer variação; a tolerância em pixels evita
        reprovar a paridade por arredondamento.
        """
        result = compare_outputs([(5, 0.5, [821.0, 0.0, 1074.4, 0.6])], [(5, 0.5, [821.1, 0.0, 1074.4, 0.6])], 0.25)
        self.assertTrue(result["equivalent"])


# ENTRADA: execução direta do arquivo.
if __name__ == "__main__":
    unittest.main()
