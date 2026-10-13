"""Marcação automática: frases do YOLOE, união com o piloto e filtros (sem rede neural).

O YOLOE real roda em test_e2e_ultralytics.py (test_10), se o checkpoint existir.

Módulo coberto
--------------
``dronecamp_ia.autolabel``: ``load_prompts`` (lê configs/zero_shot_prompts.json e mapeia
cada frase para um ID da taxonomia), ``zero_shot_boxes`` (roda o modelo de vocabulário aberto,
separa caixas de classes catalogadas e "não catalogadas", funde duplicatas e descarta caixas
grandes demais), ``merge_suggestions`` (une as sugestões do piloto com as da busca aberta) e
as constantes de origem ``SOURCE_PILOT``, ``SOURCE_OPEN`` e ``SOURCE_BOTH``.

Como rodar (na pasta ``sistema_ia``)
------------------------------------
- Só este arquivo::

      DRONECAMP_SKIP_E2E=1 python -m unittest tests.test_autolabel -v

- Um teste só::

      python -m unittest tests.test_autolabel.MergeTests.test_pilot_only_boxes_keep_their_source -v

O que cada classe prepara
-------------------------
- ``PromptTests``: usa a config e as frases REAIS do projeto (só leitura); o teste de classe
  desconhecida copia ``configs/`` para uma pasta temporária e altera a cópia.
- ``MergeTests``: nada no disco; listas de caixas montadas no próprio teste.
- ``ZeroShotBoxesTests``: foto preta 100x100 numa pasta temporária, um vocabulário de 3 frases
  (duas de calha + "rust" sem classe) e um ``FakeModel`` no lugar do YOLOE.

EXTENSAO: para testar uma função nova da marcação automática (ex.: um filtro novo em
``zero_shot_boxes``, como área mínima), acrescente a chave em ``self.spec`` no ``setUp`` de
``ZeroShotBoxesTests``, copie ``test_prompts_map_to_classes_duplicates_merge_and_whole_photo_boxes_drop``
e monte no ``FakeModel`` uma caixa que o filtro deve remover. Para uma regra nova de união,
copie ``test_same_object_found_by_both_becomes_one_suggestion``.

ATENCAO: ``PromptTests`` depende do conteúdo real de configs/zero_shot_prompts.json e da
taxonomia: toda classe precisa ter ao menos uma frase, e a frase "debris in roof gutter"
precisa existir para ``residuos_calha``.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from dronecamp_ia.autolabel import (SOURCE_BOTH, SOURCE_OPEN, SOURCE_PILOT, load_prompts, merge_suggestions,
                                    zero_shot_boxes)
from dronecamp_ia.config import detection_names, load_config, load_taxonomy

# ---------------------------------------------------------------------------
# Dados reais do projeto e dublês do modelo
# ---------------------------------------------------------------------------

# Config real (configs/project.yaml), só leitura.
REAL = load_config()
# Nomes das classes do detector na ordem dos IDs (índice = class_id).
NAMES = detection_names(load_taxonomy(REAL.taxonomy_path))
# ID real da classe "calha_obstruida"; buscado pelo nome para não quebrar se os IDs mudarem.
GUTTER = NAMES.index("calha_obstruida")


class FakeTensor(list):
    """Lista que imita um tensor: ``.cpu().tolist()`` devolve uma lista comum."""

    def cpu(self):
        """Retorna o próprio objeto (no tensor real, copia para a CPU)."""
        return self

    def tolist(self):
        """Retorna uma cópia como lista Python."""
        return list(self)


class FakeResult:
    """Imita o resultado do YOLOE com ``boxes.xyxy``, ``boxes.conf`` e ``boxes.cls``."""

    def __init__(self, boxes):
        """Monta ``self.boxes``.

        Parâmetros:
            boxes: lista de ``([x1, y1, x2, y2], confiança, índice da frase no vocabulário)``.
        """
        # Cada geradora pega uma posição da tupla: caixa, confiança ou índice da frase.
        self.boxes = type("Boxes", (), {"xyxy": FakeTensor(b for b, _, _ in boxes),
                                        "conf": FakeTensor(c for _, c, _ in boxes),
                                        "cls": FakeTensor(i for _, _, i in boxes)})()


class FakeModel:
    """Modelo falso: ``predict`` sempre devolve as caixas recebidas no construtor."""

    def __init__(self, boxes):
        """Guarda as caixas a devolver.

        Parâmetros:
            boxes: mesmo formato de ``FakeResult``.
        """
        self.boxes = boxes

    def predict(self, **kwargs):
        """Ignora os argumentos (source, conf, device...) e devolve um iterador com um ``FakeResult``."""
        return iter([FakeResult(self.boxes)])


# ---------------------------------------------------------------------------
# load_prompts: frases da busca aberta
# ---------------------------------------------------------------------------

class PromptTests(unittest.TestCase):
    """Leitura e validação de configs/zero_shot_prompts.json."""

    def test_project_prompts_cover_every_class_and_map_to_taxonomy(self):
        """Toda classe da taxonomia tem pelo menos uma frase, e as frases apontam para IDs válidos.

        Importa porque uma classe sem frase nunca seria sugerida pela busca aberta.
        """
        spec = load_prompts(REAL)
        # "vocabulary" é uma lista de (frase, class_id ou None); None = frase sem classe no catálogo.
        mapped = {class_id for _, class_id in spec["vocabulary"] if class_id is not None}
        self.assertEqual(mapped, set(range(len(NAMES))))
        self.assertIn(("debris in roof gutter", GUTTER), spec["vocabulary"])

    def test_unknown_class_is_rejected(self):
        """Uma chave de classe que não existe na taxonomia faz ``load_prompts`` lançar ValueError."""
        with tempfile.TemporaryDirectory() as folder:
            # 1. Copia configs/ inteiro para não tocar no arquivo real.
            root = Path(folder)
            shutil.copytree(REAL.root / "configs", root / "configs")
            # 2. Acrescenta uma classe inventada na cópia.
            path = root / "configs" / "zero_shot_prompts.json"
            spec = json.loads(path.read_text(encoding="utf-8"))
            spec["classes"]["classe_inventada"] = ["anything"]
            path.write_text(json.dumps(spec), encoding="utf-8")
            # 3. Config apontando para a cópia.
            with self.assertRaises(ValueError):
                load_prompts(replace(REAL, root=root, taxonomy_path=root / "configs" / "taxonomy.json"))


# ---------------------------------------------------------------------------
# merge_suggestions: piloto + busca aberta
# ---------------------------------------------------------------------------

class MergeTests(unittest.TestCase):
    """União das sugestões do modelo piloto com as da busca aberta."""

    def test_same_object_found_by_both_becomes_one_suggestion(self):
        """Mesmo objeto achado pelos dois vira uma sugestão "ambos", com a maior confiança.

        Caixas só da busca aberta ficam com origem "aberta"; os IDs são renumerados (s001, s002).
        """
        # Piloto: uma caixa de calha com confiança 0.3.
        pilot = [{"class_id": GUTTER, "bbox_xyxy": [0, 0, 10, 10], "confidence": 0.3, "id": "s001"}]
        # Busca aberta: a mesma caixa (quase igual) com 0.6 e outra caixa de classe 7 em outro lugar.
        found = [{"class_id": GUTTER, "bbox_xyxy": [0, 0, 10, 11], "confidence": 0.6, "source": SOURCE_OPEN,
                  "prompt": "clogged gutter"},
                 {"class_id": 7, "bbox_xyxy": [50, 50, 80, 80], "confidence": 0.2, "source": SOURCE_OPEN,
                  "prompt": "moss in gutter"}]
        merged = merge_suggestions(pilot, found)
        self.assertEqual([box["source"] for box in merged], [SOURCE_BOTH, SOURCE_OPEN])
        self.assertEqual(merged[0]["confidence"], 0.6)
        self.assertEqual([box["id"] for box in merged], ["s001", "s002"])

    def test_pilot_only_boxes_keep_their_source(self):
        """Sem achados da busca aberta, a caixa do piloto recebe origem ``SOURCE_PILOT``."""
        merged = merge_suggestions([{"class_id": 0, "bbox_xyxy": [0, 0, 5, 5], "confidence": 0.5}], [])
        self.assertEqual(merged[0]["source"], SOURCE_PILOT)


# ---------------------------------------------------------------------------
# zero_shot_boxes: inferência de vocabulário aberto (com modelo falso)
# ---------------------------------------------------------------------------

class ZeroShotBoxesTests(unittest.TestCase):
    """Mapeamento frase → classe, fusão de duplicatas e filtro de área em ``zero_shot_boxes``."""

    def setUp(self):
        """Cria uma foto preta 100x100 e o vocabulário de teste."""
        from PIL import Image

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.image = Path(temporary.name) / "foto.jpg"
        Image.new("RGB", (100, 100)).save(self.image)
        # conf: confiança mínima; max_area_fraction: caixa maior que 50% da foto é descartada.
        # vocabulary: índice 0 e 1 = calha; índice 2 = "rust" sem classe (vai para "não catalogadas").
        # EXTENSAO: filtro novo em zero_shot_boxes: acrescente o parâmetro neste dict.
        self.spec = {"conf": 0.1, "max_area_fraction": 0.5,
                     "vocabulary": [("debris in roof gutter", GUTTER), ("clogged gutter", GUTTER), ("rust", None)]}

    def test_prompts_map_to_classes_duplicates_merge_and_whole_photo_boxes_drop(self):
        """Duas frases sobre o mesmo objeto viram uma caixa; a caixa de 90% da foto sai; "rust" fica fora do catálogo.

        A caixa que sobra fica com a frase de maior confiança ("debris in roof gutter", 0.4),
        e a não catalogada não tem ``class_id``.
        """
        # Caixas: (xyxy, confiança, índice da frase).
        #  - 1ª e 2ª: mesmo objeto, frases 0 e 1 (ambas calha) → fundem;
        #  - 3ª: 100x90 = 90% da foto → descartada por max_area_fraction;
        #  - 4ª: frase 2 ("rust", sem classe) → lista de não catalogadas.
        model = FakeModel([([10, 10, 30, 30], 0.4, 0), ([11, 10, 30, 31], 0.3, 1),
                           ([0, 0, 100, 90], 0.9, 0), ([60, 60, 70, 70], 0.5, 2)])
        catalog, uncatalogued = zero_shot_boxes(model, self.image, self.spec, "cpu")
        self.assertEqual(len(catalog), 1)  # duas frases, mesmo objeto; a caixa da foto inteira sai
        self.assertEqual(catalog[0]["class_id"], GUTTER)
        self.assertEqual(catalog[0]["prompt"], "debris in roof gutter")
        self.assertEqual([item["prompt"] for item in uncatalogued], ["rust"])
        self.assertNotIn("class_id", uncatalogued[0])


# ENTRADA: execução direta do arquivo.
if __name__ == "__main__":
    unittest.main()
