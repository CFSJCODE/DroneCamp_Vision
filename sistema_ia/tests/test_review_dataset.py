"""Verifique promoção humana, originais, cobertura e independência dos splits.

Módulo coberto: ``dronecamp_ia.review_dataset`` (``build_approved_dataset`` e a tupla
``MVP_NAMES``), que transforma as revisões humanas aprovadas (registry.json) num dataset
YOLO versionado e imutável (``dataset_vN``) com ``dataset.yaml``, labels e ``provenance.json``.

Como rodar (na pasta ``sistema_ia``, com o venv do projeto):
    python -m unittest tests.test_review_dataset
    python -m unittest tests.test_review_dataset.ApprovedDatasetTests.test_output_is_immutable

O que a classe de teste prepara (``ApprovedDatasetTests.setUp``):
    - pasta temporária com acento no nome, apagada no fim do teste;
    - um ``taxonomy.json`` mínimo com as classes de ``MVP_NAMES`` (IDs 0..N-1, fase 1);
    - um ``ProjectConfig`` falso que só aponta para essa taxonomia;
    - caminhos de ``registry.json``, ``assignments.json`` e da saída ``dataset_v1``.
    As imagens entram com ``add_image`` (cria um PNG 32x24 e o item do registro) e o
    dataset é gerado com ``build`` (grava registro + atribuições e chama o exportador).
    ``three_groups`` monta o cenário mínimo "pronto para treino": 3 prédios, um por split,
    e a imagem de train com uma caixa de cada classe.

ATENCAO: os IDs de classe são contrato (taxonomia, pesos, labels). Os testes
``test_*_class_idN_*`` travam a posição de cada classe; uma classe nova vai SEMPRE no fim.

EXTENSAO: para testar uma nova regra de exportação, copie
``test_unapproved_ambiguous_and_excluded_do_not_enter``: monte as imagens com
``self.three_groups()`` / ``self.add_image(...)``, chame ``self.build()`` e confira
``result`` e os arquivos em ``self.output``. Ao acrescentar uma classe nova à taxonomia,
copie ``test_gutter_piece_on_tile_class_id13_is_exported_after_existing_classes`` trocando
o ID/nome, atualize ``nc`` em ``test_new_classes_export_as_ids10_and11_with_previous_ids_preserved``
e, se ela for obrigatória no train, inclua o nome na lista de ``test_missing_train_class_keeps_dataset_not_ready``.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import yaml

from dronecamp_ia.config import ProjectConfig
from dronecamp_ia.io import file_hash
from dronecamp_ia.review_dataset import MVP_NAMES, build_approved_dataset


class ApprovedDatasetTests(unittest.TestCase):
    """Testes de ``build_approved_dataset`` com um registro de revisão montado em memória.

    Garantem que só imagens aprovadas por humano entram, que os originais não mudaram,
    que a saída nunca é sobrescrita e que os IDs de classe ficam estáveis.
    """
    def setUp(self) -> None:
        """Cria pasta temporária, taxonomia mínima, ``ProjectConfig`` falso e caminhos de saída.

        Atributos criados: ``root``, ``taxonomy``, ``config``, ``registry_path``,
        ``assignments_path``, ``output`` (dataset_v1), ``images`` (itens do registro) e
        ``groups`` (prédio -> split).
        Retorna: None.
        """
        # Pasta temporária (com acento) apagada sozinha no fim do teste.
        self.temp = tempfile.TemporaryDirectory(prefix="dronecamp revisão ç ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Taxonomia mínima: cada nome de MVP_NAMES com seu índice como ID, todos da fase 1.
        self.taxonomy = self.root / "taxonomy.json"
        self.save_json(self.taxonomy, {"version": "test", "classes": [
            {"id": index, "slug": name, "phase": 1} for index, name in enumerate(MVP_NAMES)]})
        # Configuração falsa: só root e taxonomy_path importam para o exportador.
        self.config = ProjectConfig(self.root, "unused", "unused", self.taxonomy, self.root / "unused.yaml", "cpu", {}, {})
        # Arquivos que build() grava e a pasta de saída (que NÃO pode existir antes).
        self.registry_path, self.assignments_path = self.root / "registry.json", self.root / "assignments.json"
        self.output = self.root / "dataset_v1"
        self.images = []
        self.groups = {}

    @staticmethod
    def save_json(path: Path, value: dict) -> None:
        """Grava ``value`` como JSON UTF-8 em ``path``.

        Parâmetros:
            path: arquivo de destino.
            value: dict a gravar.
        Retorna: None.
        """
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def add_image(self, group: str, split: str = "train", class_ids=(0,), approved: bool = True, status: str = "positive") -> dict:
        """Cria um PNG 32x24 e acrescenta o item correspondente ao registro de revisão.

        Parâmetros:
            group: ``building_group`` (prédio); define o split via ``self.groups``.
            split: "train", "val" ou "test" para esse grupo.
            class_ids: IDs das classes; cada um vira uma caixa ``[2, 3, 15, 18]`` (xyxy em px).
            approved: se True, marca ``human_approved`` e revisor ``reviewer_test``.
            status: "positive", "negative" (sem caixas), "ambiguous" ou "excluded".
        Retorna: o dict do item (o teste pode alterá-lo antes de ``build``).
        """
        # Imagem com cor diferente a cada chamada, para que cada uma tenha hash próprio.
        source = self.root / f"source_{len(self.images)}.png"
        Image.new("RGB", (32, 24), color=(20 + len(self.images) * 30, 80, 90)).save(source)
        # Item no formato do registry.json; negativas não têm caixas.
        # EXTENSAO: novo campo do registro a testar: acrescente aqui (e um parâmetro em add_image).
        item = {"image_sha256": file_hash(source), "source_path": str(source), "width": 32, "height": 24,
                "building_group": group, "status": status, "human_approved": approved,
                "technical_status": "human_visual_reviewed" if approved else "pending_human_review",
                "human_reviewer": "reviewer_test" if approved else None,
                "boxes": [{"class_id": index, "bbox_xyxy": [2, 3, 15, 18]} for index in class_ids] if status != "negative" else []}
        self.images.append(item)
        self.groups[group] = split
        return item

    def build(self) -> dict:
        """Grava ``registry.json`` e ``assignments.json`` e chama ``build_approved_dataset``.

        Retorna: o dict de resultado do exportador (``ready_for_training``, ``state``,
        ``reasons``, ``counts``, ``validation``, ``train_boxes_by_class``...).
        Erros: propaga o ValueError do exportador.
        """
        # Registro e atribuições prédio -> split, no formato lido por build_approved_dataset.
        self.save_json(self.registry_path, {"schema_version": 1, "taxonomy_sha256": file_hash(self.taxonomy), "images": self.images})
        self.save_json(self.assignments_path, {"schema_version": 1, "groups": self.groups})
        return build_approved_dataset(self.config, self.registry_path, self.assignments_path, self.output)

    def three_groups(self) -> None:
        """Cenário mínimo pronto para treino: 3 prédios (train/val/test), train com todas as classes.

        Retorna: None.
        """
        self.add_image("building_a", "train", range(len(MVP_NAMES)))
        self.add_image("building_b", "val")
        self.add_image("building_c", "test")

    def test_zero_approved_creates_nothing(self) -> None:
        """Sem nenhuma imagem aprovada, o exportador falha e não cria a pasta de saída."""
        self.add_image("ceasa", approved=False)
        with self.assertRaisesRegex(ValueError, "Nenhuma imagem"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_single_ceasa_stays_draft_without_empty_splits(self) -> None:
        """Um único prédio gera rascunho (``draft``), sem pastas val/test vazias.

        Importa porque splits vazios ou copiados simulariam uma diversidade que não existe.
        """
        self.add_image("ceasa", class_ids=range(len(MVP_NAMES)))
        result = self.build()
        self.assertFalse(result["ready_for_training"])
        self.assertEqual(result["state"], "draft")
        self.assertTrue(any("três grupos" in reason for reason in result["reasons"]))
        self.assertFalse((self.output / "images/val").exists())
        self.assertFalse((self.output / "images/test").exists())
        self.assertEqual(result["counts"]["train"]["images"], 1)

    def test_three_independent_groups_and_all_train_classes_ready(self) -> None:
        """Três grupos independentes com todas as classes no train = dataset pronto.

        Confere também o ``dataset.yaml`` (nomes e caminho absoluto) e o ``provenance.json``
        (hash do registro, das atribuições, de cada imagem e autor da aprovação).
        """
        self.three_groups()
        result = self.build()
        self.assertTrue(result["ready_for_training"], result)
        self.assertTrue(result["validation"]["valid"])
        self.assertEqual(result["reasons"], [])
        config = yaml.safe_load((self.output / "dataset.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["names"], list(MVP_NAMES))
        self.assertTrue(Path(config["path"]).is_absolute())
        provenance = json.loads((self.output / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["registry_sha256"], file_hash(self.registry_path))
        self.assertEqual(provenance["assignments_sha256"], file_hash(self.assignments_path))
        for item in provenance["images"]:
            self.assertEqual(file_hash(self.output / item["image"]), item["image_sha256"])
            self.assertEqual(item["approval_author"], "reviewer_test")

    def test_unapproved_ambiguous_and_excluded_do_not_enter(self) -> None:
        """Imagens não aprovadas, ambíguas ou excluídas ficam fora e são listadas na proveniência.

        ATENCAO: a IA nunca aprova; só a aprovação humana promove uma imagem ao dataset.
        """
        self.three_groups()
        self.add_image("ai_only", approved=False)
        self.add_image("ambiguous", status="ambiguous")
        self.add_image("excluded", status="excluded")
        result = self.build()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(result["counts"]["total"]["images"], 3)
        provenance = json.loads((self.output / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(len(provenance["excluded_images"]), 3)

    def test_changed_source_hash_rejected_before_writing(self) -> None:
        """Se o original mudou depois da revisão (hash diferente), nada é gravado."""
        item = self.add_image("ceasa")
        Image.new("RGB", (32, 24), color="red").save(item["source_path"])
        with self.assertRaisesRegex(ValueError, "Hash do original"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_changed_dimensions_rejected_before_writing(self) -> None:
        """Dimensões registradas diferentes das reais bloqueiam a exportação antes de gravar."""
        item = self.add_image("ceasa")
        item["width"] = 31
        with self.assertRaisesRegex(ValueError, "Dimensões"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_taxonomy_hash_must_match_review(self) -> None:
        """Se a taxonomia mudou depois da revisão (outro hash), a exportação é recusada."""
        self.add_image("ceasa")
        self.build()
        self.output = self.root / "dataset_v2"
        # Uma quebra de linha a mais já muda o hash da taxonomia.
        self.taxonomy.write_text(self.taxonomy.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Taxonomia mudou"):
            build_approved_dataset(self.config, self.registry_path, self.assignments_path, self.output)
        self.assertFalse(self.output.exists())

    def test_output_is_immutable(self) -> None:
        """Exportar de novo para a mesma pasta falha e não altera o ``provenance.json`` existente."""
        self.three_groups()
        self.build()
        original = (self.output / "provenance.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "imutáveis"):
            self.build()
        self.assertEqual((self.output / "provenance.json").read_bytes(), original)

    def test_missing_train_class_keeps_dataset_not_ready(self) -> None:
        """Faltando uma classe obrigatória no train, o dataset é válido mas não fica pronto.

        EXTENSAO: classe nova que deve ser obrigatória no train: acrescente o nome na tupla do laço.
        """
        self.three_groups()
        # Para cada classe obrigatória: remove só as caixas dela e exporta numa pasta nova.
        complete_boxes = list(self.images[0]["boxes"])
        for missing_name in ("pedaco_telha", "rufo_deslocado", "pedaco_telha_sobreposto", "fixador_telha_frouxo", "reparo_rufo", "pedaco_calha_sobre_telha"):
            with self.subTest(missing_class=missing_name):
                missing_id = MVP_NAMES.index(missing_name)
                self.images[0]["boxes"] = [box for box in complete_boxes if box["class_id"] != missing_id]
                self.output = self.root / f"dataset_without_{missing_name}"
                result = self.build()
                self.assertTrue(result["validation"]["valid"])
                self.assertFalse(result["ready_for_training"])
                self.assertTrue(any(missing_name in reason for reason in result["reasons"]))

    def test_fragment_class_id8_is_exported_without_changing_previous_ids(self) -> None:
        """``pedaco_telha`` é exportada com ID 8 sem mexer nos IDs 0..7."""
        self.three_groups()
        result = self.build()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(MVP_NAMES[8], "pedaco_telha")
        self.assertEqual(MVP_NAMES[7], "vegetacao_calha")
        label = self.output / "labels/train" / (self.images[0]["image_sha256"] + ".txt")
        self.assertIn("8", [line.split()[0] for line in label.read_text(encoding="utf-8").splitlines()])
        self.assertEqual(result["train_boxes_by_class"]["8"], 1)

    def test_displaced_flashing_class_id9_is_exported_with_stable_prior_ids(self) -> None:
        """``rufo_deslocado`` é exportada com ID 9, logo após ``pedaco_telha``."""
        self.three_groups()
        result = self.build()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(MVP_NAMES[8:10], ("pedaco_telha", "rufo_deslocado"))
        label = self.output / "labels/train" / (self.images[0]["image_sha256"] + ".txt")
        self.assertIn("9", [line.split()[0] for line in label.read_text(encoding="utf-8").splitlines()])
        self.assertEqual(result["train_boxes_by_class"]["9"], 1)

    def test_new_classes_export_as_ids10_and11_with_previous_ids_preserved(self) -> None:
        """IDs 10 e 11 vêm depois dos 10 anteriores; ``nc`` total é 14.

        ATENCAO: ao acrescentar classe na taxonomia, o 14 aqui (nc e range) precisa subir.
        """
        self.three_groups()
        result = self.build()
        self.assertTrue(result["ready_for_training"])
        previous_names = (
            "telha_quebrada", "telha_ausente", "residuos_telha", "reparo_telha",
            "rufo_ausente", "rufo_quebrado", "residuos_calha", "vegetacao_calha",
            "pedaco_telha", "rufo_deslocado",
        )
        self.assertEqual(MVP_NAMES[:10], previous_names)
        self.assertEqual(MVP_NAMES[10:12], ("pedaco_telha_sobreposto", "fixador_telha_frouxo"))
        exported = yaml.safe_load((self.output / "dataset.yaml").read_text(encoding="utf-8"))
        self.assertEqual(exported["nc"], 14)
        self.assertEqual(len(exported["names"]), 14)
        label = self.output / "labels/train" / (self.images[0]["image_sha256"] + ".txt")
        rows = [line.split() for line in label.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([int(row[0]) for row in rows], list(range(14)))
        for class_id in (10, 11):
            with self.subTest(class_id=class_id):
                self.assertEqual(result["train_boxes_by_class"][str(class_id)], 1)

    def test_flashing_repair_class_id12_is_exported_after_existing_classes(self) -> None:
        """``reparo_rufo`` é exportada com ID 12, depois das classes existentes."""
        self.three_groups()
        result = self.build()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(MVP_NAMES[12], "reparo_rufo")
        self.assertEqual(MVP_NAMES[11], "fixador_telha_frouxo")
        exported = yaml.safe_load((self.output / "dataset.yaml").read_text(encoding="utf-8"))
        self.assertEqual(exported["names"][12], "reparo_rufo")
        label = self.output / "labels/train" / (self.images[0]["image_sha256"] + ".txt")
        self.assertIn("12", [line.split()[0] for line in label.read_text(encoding="utf-8").splitlines()])
        self.assertEqual(result["train_boxes_by_class"]["12"], 1)

    def test_gutter_piece_on_tile_class_id13_is_exported_after_existing_classes(self) -> None:
        """``pedaco_calha_sobre_telha`` é exportada com ID 13, a última classe atual."""
        self.three_groups()
        result = self.build()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(MVP_NAMES[13], "pedaco_calha_sobre_telha")
        self.assertEqual(MVP_NAMES[12], "reparo_rufo")
        exported = yaml.safe_load((self.output / "dataset.yaml").read_text(encoding="utf-8"))
        self.assertEqual(exported["names"][13], "pedaco_calha_sobre_telha")
        label = self.output / "labels/train" / (self.images[0]["image_sha256"] + ".txt")
        self.assertIn("13", [line.split()[0] for line in label.read_text(encoding="utf-8").splitlines()])
        self.assertEqual(result["train_boxes_by_class"]["13"], 1)

    def test_approved_negative_has_explicit_empty_label(self) -> None:
        """Imagem negativa aprovada gera arquivo de label vazio (negativa explícita)."""
        self.three_groups()
        negative = self.add_image("building_a", status="negative")
        result = self.build()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual((self.output / "labels/train" / (negative["image_sha256"] + ".txt")).read_text(), "")


if __name__ == "__main__":
    unittest.main()
