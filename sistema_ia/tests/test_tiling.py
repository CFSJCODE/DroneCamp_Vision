"""Fatiamento de fotos grandes (tiling.py): janelas, projeção de caixas e fusão das predições.

Módulo coberto
--------------
``dronecamp_ia.tiling``:
- ``calculate_patch_windows``: janelas (x1, y1, x2, y2) com sobreposição que cobrem a foto toda;
- ``project_boxes_to_patch``: leva caixas da foto inteira para as coordenadas locais de uma
  janela, descartando as pouco visíveis;
- ``merge_tiled_predictions``: junta as predições de janelas vizinhas e funde duplicatas (NMS por classe).
``build_tiled_dataset`` é importado mas não é testado aqui (é exercitado em
test_tiled_pipeline.py via ``training._tiled_training_dataset``).

Como rodar (na pasta ``sistema_ia``)
------------------------------------
- Só este arquivo::

      DRONECAMP_SKIP_E2E=1 python -m unittest tests.test_tiling -v

- Um teste só::

      python -m unittest tests.test_tiling.TestTiling.test_merge_tiled_predictions_nms -v

O que a classe prepara
----------------------
``TestTiling`` não tem ``setUp`` nem usa disco: cada teste monta as coordenadas na hora.
Todas as coordenadas são em pixels, formato xyxy (canto superior esquerdo, canto inferior direito).

EXTENSAO: para testar uma função nova de tiling.py, copie o teste mais parecido (janelas,
projeção ou fusão), monte a entrada com números fáceis de conferir à mão (ex.: janela
começando em 1000) e escreva no comentário a conta esperada, como os testes abaixo fazem.
Se a função gravar arquivos, use ``tempfile.TemporaryDirectory()`` (já importado).
"""

from pathlib import Path
import tempfile
import unittest

from PIL import Image

from dronecamp_ia.tiling import (
    calculate_patch_windows,
    project_boxes_to_patch,
    merge_tiled_predictions,
    build_tiled_dataset,
)


# ---------------------------------------------------------------------------
# Janelas, projeção e fusão
# ---------------------------------------------------------------------------

class TestTiling(unittest.TestCase):
    """Funções puras de ``tiling.py`` (sem disco e sem modelo)."""

    def test_calculate_patch_windows_cover_entire_image(self):
        """As janelas cobrem a foto 5472x3648 de borda a borda, sem passar dos limites nem do tamanho 1280.

        Importa porque um defeito na borda da foto ficaria sem janela se a cobertura falhasse.
        """
        # Imagem 5472 x 3648
        w, h = 5472, 3648
        patch_size = 1280
        overlap = 0.2
        windows = calculate_patch_windows(w, h, patch_size=patch_size, overlap_ratio=overlap)

        self.assertGreater(len(windows), 1)
        # Confere se os cantos externos alcançam os limites exatos da imagem
        # win = (x1, y1, x2, y2): índices 0 e 1 = canto superior esquerdo; 2 e 3 = inferior direito.
        max_x = max(win[2] for win in windows)
        max_y = max(win[3] for win in windows)
        min_x = min(win[0] for win in windows)
        min_y = min(win[1] for win in windows)

        self.assertEqual(min_x, 0)
        self.assertEqual(min_y, 0)
        self.assertEqual(max_x, w)
        self.assertEqual(max_y, h)

        # Confere que nenhuma janela excede as bordas
        for x1, y1, x2, y2 in windows:
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, w)
            self.assertLessEqual(y2, h)
            self.assertLessEqual(x2 - x1, patch_size)
            self.assertLessEqual(y2 - y1, patch_size)

    def test_project_boxes_inside_and_crossing(self):
        """Caixas dentro da janela viram coordenadas locais; caixas cortadas são recortadas; < 30% visível sai.

        Também confere a fração visível (``visibility``) e o vínculo ``original_box_id``.
        """
        # Janela do patch: [1000, 1000, 2280, 2280] (tamanho 1280x1280)
        window = (1000, 1000, 2280, 2280)

        # EXTENSAO: caso novo de projeção (ex.: caixa cortada em duas bordas): acrescente um dict nesta lista.
        boxes = [
            # Caixa totalmente dentro: [1100, 1100, 1200, 1200]
            {"class_id": 0, "bbox_xyxy": [1100, 1100, 1200, 1200], "id": "b1"},
            # Caixa totalmente fora: [0, 0, 500, 500]
            {"class_id": 1, "bbox_xyxy": [0, 0, 500, 500], "id": "b2"},
            # Caixa cortada na margem esquerda: [950, 1100, 1050, 1200] (metade dentro: 50% visível)
            {"class_id": 2, "bbox_xyxy": [950, 1100, 1050, 1200], "id": "b3"},
            # Caixa com apenas 5% dentro: [900, 1100, 1005, 1200]
            {"class_id": 3, "bbox_xyxy": [900, 1100, 1005, 1200], "id": "b4"},
        ]

        # min_visibility=0.3: a caixa precisa ter pelo menos 30% da área dentro da janela.
        projected = project_boxes_to_patch(boxes, window, min_visibility=0.3)

        # Deve conter b1 (100% dentro) e b3 (50% visível), mas descartar b2 e b4 (<30% visível)
        self.assertEqual(len(projected), 2)

        # b1 nas coordenadas locais: 1100 - 1000 = 100
        p1 = next(b for b in projected if b["original_box_id"] == "b1")
        self.assertEqual(p1["bbox_xyxy"], [100.0, 100.0, 200.0, 200.0])
        self.assertEqual(p1["visibility"], 1.0)

        # b3 nas coordenadas locais: cortada em x1=0 (1000 - 1000) e x2=50 (1050 - 1000)
        p3 = next(b for b in projected if b["original_box_id"] == "b3")
        self.assertEqual(p3["bbox_xyxy"], [0.0, 100.0, 50.0, 200.0])
        self.assertAlmostEqual(p3["visibility"], 0.5, places=2)

    def test_merge_tiled_predictions_nms(self):
        """O mesmo objeto visto em duas janelas vizinhas vira uma predição só (a de maior score).

        Cada predição traz a caixa em coordenadas LOCAIS e a ``window`` de origem; a função
        soma o deslocamento da janela antes de comparar (IoU >= 0.5 funde).
        """
        # Duas predições da mesma classe sobrepostas em patches adjacentes
        preds = [
            # Patch 1 em [0, 0, 1280, 1280], detectou caixa em local [1010, 500, 1060, 550] (global: [1010, 500, 1060, 550])
            {"class_id": 4, "score": 0.88, "bbox_xyxy": [1010, 500, 1060, 550], "window": (0, 0, 1280, 1280)},
            # Patch 2 em [1000, 0, 2280, 1280], detectou a MESMA caixa em local [12, 502, 61, 551] (global: [1012, 502, 1061, 551])
            {"class_id": 4, "score": 0.94, "bbox_xyxy": [12, 502, 61, 551], "window": (1000, 0, 2280, 1280)},
            # Outra caixa independente da mesma classe
            {"class_id": 4, "score": 0.75, "bbox_xyxy": [200, 200, 300, 300], "window": (0, 0, 1280, 1280)},
        ]

        merged = merge_tiled_predictions(preds, iou_threshold=0.5)

        # As duas primeiras duplicadas devem se fundir na de maior confiança (0.94)
        # O resultado sai ordenado por score decrescente.
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["score"], 0.94)
        self.assertEqual(merged[1]["score"], 0.75)


# ENTRADA: execução direta do arquivo.
if __name__ == "__main__":
    unittest.main()
