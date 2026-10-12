"""Bloqueie treino quando dados ou aprovação divergem da revisão de origem.

Módulo coberto: ``dronecamp_ia.review_provenance`` (``validate_training_provenance``) e o
uso dele dentro de ``dronecamp_ia.training`` (``train``, ``tune``, ``evaluate``,
``prepare_dataset``). A verificação refaz, a partir do ``provenance.json`` do dataset
exportado, todas as conferências (hashes, aprovação humana, grupos, classes) antes de
qualquer modelo ser carregado.

Como rodar (na pasta ``sistema_ia``, com o venv do projeto):
    python -m unittest tests.test_review_provenance
    python -m unittest tests.test_review_provenance.TrainingProvenanceTests.test_path_traversal_rejected

O que a classe de teste prepara (``TrainingProvenanceTests.setUp``):
    - pasta temporária com acento, apagada no fim;
    - ``taxonomy.json`` mínimo com ``MVP_NAMES`` e um ``ProjectConfig`` falso;
    - 3 imagens PNG 32x24, uma por prédio/split; a de train tem uma caixa de cada classe,
      val e test só da classe 0; todas aprovadas pelo revisor ``human_test``;
    - os dicts ``self.registry`` e ``self.assignments`` (ainda não gravados).
    ``build()`` grava esses dicts, exporta o dataset com ``build_approved_dataset`` e carrega
    ``self.provenance``; ``check()`` roda o validador sobre o ``dataset.yaml`` exportado.
    O padrão dos testes de adulteração é: build -> alterar um arquivo/campo -> regravar ->
    ``assertRaisesRegex(ValueError, "trecho")`` em ``check()``.

ATENCAO: nenhum treino real acontece aqui; ``load_detector`` é substituído por mock nos
testes que passam por ``training`` e o teste exige que ele NUNCA seja chamado.

EXTENSAO: para testar uma nova conferência de proveniência, copie
``test_missing_approval_author_rejected``: chame ``self.build()``, altere
``self.provenance["images"][0][<campo>]`` (ou um arquivo em ``self.output``), regrave com
``self.write(self.provenance_path, self.provenance)`` e exija o trecho da mensagem nova com
``assertRaisesRegex``. Para uma classe nova, copie
``test_gutter_piece_on_tile_class_id13_validates_human_provenance`` e suba o total 14.
"""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

from dronecamp_ia import training
from dronecamp_ia.config import ProjectConfig
from dronecamp_ia.io import file_hash
from dronecamp_ia.review_dataset import MVP_NAMES, build_approved_dataset
from dronecamp_ia.review_provenance import validate_training_provenance


class TrainingProvenanceTests(unittest.TestCase):
    """Testes de adulteração e de bloqueio do treino sobre um dataset aprovado real (temporário).

    Importa porque o modelo só pode aprender com dados revisados por humano e idênticos
    ao que foi revisado; qualquer divergência deve parar o treino antes de carregar o YOLO.
    """
    def setUp(self):
        """Monta taxonomia, config falso, 3 imagens aprovadas (train/val/test) e os dicts de registro.

        Atributos criados: ``root``, ``taxonomy``, ``config``, ``registry_path``,
        ``assignments_path``, ``output``, ``items``, ``registry`` e ``assignments``.
        Retorna: None.
        """
        # 1. Pasta temporária, taxonomia mínima e configuração falsa (só taxonomy_path importa).
        self.temp = tempfile.TemporaryDirectory(prefix="dronecamp proveniência ç ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.taxonomy = self.root / "taxonomy.json"
        self.write(self.taxonomy, {"classes": [{"id": i, "slug": n, "phase": 1} for i, n in enumerate(MVP_NAMES)]})
        self.config = ProjectConfig(self.root, "unused", "unused", self.taxonomy, self.root / "unused.yaml", "cpu", {}, {})
        # 2. Caminhos dos arquivos que build() grava e da saída do dataset.
        self.registry_path, self.assignments_path = self.root / "registry.json", self.root / "assignments.json"
        self.output = self.root / "dataset_v1"
        self.items = []
        # 3. Uma imagem aprovada por split; a de train tem uma caixa de cada classe
        # (exigência de prontidão), val/test só a classe 0.
        for index, split in enumerate(("train", "val", "test")):
            source = self.root / f"source_{index}.png"
            Image.new("RGB", (32, 24), color=(30 + index * 60, 90, 100)).save(source)
            self.items.append({"image_sha256": file_hash(source), "source_path": str(source), "width": 32, "height": 24,
                "building_group": f"building_{index}", "status": "positive", "human_approved": True,
                "technical_status": "human_visual_reviewed", "human_reviewer": "human_test",
                "boxes": [{"class_id": c, "bbox_xyxy": [2, 3, 16, 18]} for c in (range(len(MVP_NAMES)) if split == "train" else [0])]})
        # 4. Registro de revisão e atribuições building_i -> split (gravados só em build()).
        self.registry = {"schema_version": 1, "taxonomy_sha256": file_hash(self.taxonomy), "images": self.items}
        self.assignments = {"schema_version": 1, "groups": {f"building_{i}": s for i, s in enumerate(("train", "val", "test"))}}

    @staticmethod
    def write(path, value):
        """Grava ``value`` como JSON UTF-8 em ``path``.

        Parâmetros:
            path (Path): destino.
            value (dict): conteúdo.
        Retorna: None.
        """
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def build(self):
        """Grava registro e atribuições, exporta o dataset e carrega a proveniência.

        Define ``self.data_path`` (dataset.yaml), ``self.provenance_path`` e
        ``self.provenance`` (dict que os testes alteram e regravam).
        Retorna: None.
        """
        self.write(self.registry_path, self.registry)
        self.write(self.assignments_path, self.assignments)
        build_approved_dataset(self.config, self.registry_path, self.assignments_path, self.output)
        self.data_path = self.output / "dataset.yaml"
        self.provenance_path = self.output / "provenance.json"
        self.provenance = json.loads(self.provenance_path.read_text(encoding="utf-8"))

    def check(self):
        """Roda ``validate_training_provenance`` sobre o dataset exportado.

        Retorna: o dict de resultado (``ready_for_training``, ``counts``,
        ``train_boxes_by_class``, ``registry_sha256``...).
        Erros: ValueError com a mensagem da primeira divergência encontrada.
        """
        return validate_training_provenance(self.config, self.data_path)

    def test_valid_approval_is_recomputed_without_readiness(self):
        """A prontidão é recalculada do zero, mesmo sem o ``readiness.json`` salvo."""
        self.build()
        (self.output / "readiness.json").unlink()
        result = self.check()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(result["counts"]["total"]["images"], 3)
        self.assertEqual(result["registry_sha256"], file_hash(self.registry_path))

    def test_structurally_valid_dataset_without_provenance_is_rejected(self):
        """Dataset estruturalmente válido mas sem ``provenance.json`` é recusado."""
        self.build()
        self.provenance_path.unlink()
        with self.assertRaisesRegex(ValueError, "Proveniência"):
            self.check()

    def test_changed_label_rejected_even_when_hash_resigned(self):
        """Label alterada é recusada mesmo se o hash for "reassinado" na proveniência.

        O validador regenera a label a partir das caixas revisadas e compara o conteúdo
        ("Label normalizada"), então editar label + hash não engana a conferência.
        """
        self.build()
        sample = self.provenance["images"][0]
        label = self.output / sample["label"]
        # 0.28125 = x central da caixa [2..16] em 32 px; muda o valor e "reassina" o hash.
        label.write_text(label.read_text().replace("0.28125000", "0.30000000"), encoding="utf-8")
        self.assertNotEqual(file_hash(label), sample["label_sha256"])
        sample["label_sha256"] = file_hash(label)
        self.write(self.provenance_path, self.provenance)
        with self.assertRaisesRegex(ValueError, "Label normalizada"):
            self.check()

    def test_changed_image_bytes_rejected(self):
        """Imagem copiada no dataset com bytes diferentes é recusada ("Hash da imagem")."""
        self.build()
        image = self.output / self.provenance["images"][0]["image"]
        Image.new("RGB", (32, 24), color="purple").save(image)
        with self.assertRaisesRegex(ValueError, "Hash da imagem"):
            self.check()

    def test_false_approval_flag_rejected(self):
        """``human_approved`` falso na proveniência bloqueia o treino.

        ATENCAO: a IA nunca aprova; só humano.
        """
        self.build()
        self.provenance["images"][0]["human_approved"] = False
        self.write(self.provenance_path, self.provenance)
        with self.assertRaisesRegex(ValueError, "aprovação humana"):
            self.check()

    def test_missing_approval_author_rejected(self):
        """Aprovação sem autor (``approval_author`` nulo) é recusada."""
        self.build()
        self.provenance["images"][0]["approval_author"] = None
        self.write(self.provenance_path, self.provenance)
        with self.assertRaisesRegex(ValueError, "Autor"):
            self.check()

    def test_pending_technical_status_rejected(self):
        """``technical_status`` pendente de revisão humana é recusado."""
        self.build()
        self.provenance["images"][0]["technical_status"] = "pending_human_review"
        self.write(self.provenance_path, self.provenance)
        with self.assertRaisesRegex(ValueError, "estado de revisão humana"):
            self.check()

    def test_original_image_changes_after_export_rejected(self):
        """Se a imagem ORIGINAL mudou depois da exportação, o treino é bloqueado."""
        self.build()
        Image.new("RGB", (32, 24), color="black").save(self.items[0]["source_path"])
        with self.assertRaisesRegex(ValueError, "imagem original"):
            self.check()

    def test_exif_orientation_cannot_change_coordinate_frame(self):
        """Imagem com orientação EXIF (rotação) é recusada mesmo com todos os hashes refeitos.

        EXIF 274 = Orientation; valor 6 = girar 90 graus. O YOLO giraria a imagem e as caixas
        (em coordenadas da imagem sem giro) ficariam no lugar errado.
        """
        self.build()
        source = Path(self.items[0]["source_path"])
        sample = self.provenance["images"][0]
        # Regrava o original com EXIF de rotação e refaz TODOS os hashes, para provar que a
        # recusa vem da orientação e não de um hash.
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (32, 24), color="orange").save(source, exif=exif)
        copied = self.output / sample["image"]
        copied.write_bytes(source.read_bytes())
        sample["image_sha256"] = self.items[0]["image_sha256"] = file_hash(source)
        self.write(self.registry_path, self.registry)
        self.provenance["registry_sha256"] = file_hash(self.registry_path)
        self.write(self.provenance_path, self.provenance)
        with self.assertRaisesRegex(ValueError, "Orientação EXIF"):
            self.check()

    def test_registry_hash_drift_rejected(self):
        """Registro de revisão alterado depois da exportação é recusado ("Hash de registry")."""
        self.build()
        self.registry["images"][0]["human_reviewer"] = "changed_author"
        self.write(self.registry_path, self.registry)
        with self.assertRaisesRegex(ValueError, "Hash de registry"):
            self.check()

    def test_assignment_hash_drift_rejected(self):
        """Atribuição prédio -> split alterada depois da exportação é recusada."""
        self.build()
        self.assignments["groups"]["building_0"] = "val"
        self.write(self.assignments_path, self.assignments)
        with self.assertRaisesRegex(ValueError, "Hash de assignments"):
            self.check()

    def test_taxonomy_drift_rejected(self):
        """Taxonomia alterada (mesmo só uma quebra de linha) é recusada."""
        self.build()
        self.taxonomy.write_text(self.taxonomy.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Taxonomia"):
            self.check()

    def test_added_image_and_label_missing_from_manifest_rejected(self):
        """Imagem + label + linha de grupo acrescentadas à mão fora da proveniência são recusadas.

        O inventário de arquivos precisa bater exatamente com o ``provenance.json``.
        """
        self.build()
        # Acrescenta imagem, label e linha no groups.csv sem passar pelo exportador.
        added = self.output / "images/train/extra.png"
        Image.new("RGB", (32, 24), color="green").save(added)
        (self.output / "labels/train/extra.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        with (self.output / "groups.csv").open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(["images/train/extra.png", "building_0", "train"])
        with self.assertRaisesRegex(ValueError, "Inventário"):
            self.check()

    def test_declared_group_change_rejected(self):
        """Trocar o ``building_group`` declarado na proveniência é recusado ("Grupo")."""
        self.build()
        self.provenance["images"][0]["building_group"] = "other_building"
        self.write(self.provenance_path, self.provenance)
        with self.assertRaisesRegex(ValueError, "Grupo"):
            self.check()

    def test_missing_train_class_cannot_be_unblocked_by_saved_readiness(self):
        """Um ``readiness.json`` forjado não libera dataset sem uma classe obrigatória no train.

        EXTENSAO: classe nova obrigatória no train: acrescente o nome na tupla do laço.
        """
        complete_boxes = list(self.items[0]["boxes"])
        for missing_name in ("pedaco_telha", "rufo_deslocado", "pedaco_telha_sobreposto", "fixador_telha_frouxo", "reparo_rufo", "pedaco_calha_sobre_telha"):
            with self.subTest(missing_class=missing_name):
                missing_id = MVP_NAMES.index(missing_name)
                self.items[0]["boxes"] = [box for box in complete_boxes if box["class_id"] != missing_id]
                self.output = self.root / f"dataset_without_{missing_name}"
                self.build()
                self.write(self.output / "readiness.json", {"ready_for_training": True})
                with self.assertRaisesRegex(ValueError, "classes ativas.*" + missing_name):
                    self.check()

    def test_fragment_class_id8_remains_valid_in_human_provenance(self):
        """A classe ID 8 (``pedaco_telha``) é contada e aceita pela validação."""
        self.build()
        result = self.check()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(result["train_boxes_by_class"]["8"], 1)
        self.assertEqual(MVP_NAMES[8], "pedaco_telha")

    def test_displaced_flashing_class_id9_validates_human_provenance(self):
        """A classe ID 9 (``rufo_deslocado``) é contada e aceita pela validação."""
        self.build()
        result = self.check()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(result["train_boxes_by_class"]["9"], 1)
        self.assertEqual(MVP_NAMES[9], "rufo_deslocado")

    def test_overlaid_fragment_and_loose_fastener_validate_as_ids10_and11(self):
        """As classes ID 10 e 11 são contadas; o total de classes é 14."""
        self.build()
        result = self.check()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(MVP_NAMES[10:12], ("pedaco_telha_sobreposto", "fixador_telha_frouxo"))
        self.assertEqual(len(result["train_boxes_by_class"]), 14)
        for class_id in (10, 11):
            with self.subTest(class_id=class_id):
                self.assertEqual(result["train_boxes_by_class"][str(class_id)], 1)

    def test_flashing_repair_class_id12_validates_human_provenance(self):
        """A classe ID 12 (``reparo_rufo``) aparece na label de train e mantém o autor humano."""
        self.build()
        result = self.check()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(MVP_NAMES[12], "reparo_rufo")
        self.assertEqual(len(result["train_boxes_by_class"]), 14)
        self.assertEqual(result["train_boxes_by_class"]["12"], 1)
        sample = next(item for item in self.provenance["images"] if item["split"] == "train")
        label = self.output / sample["label"]
        self.assertIn("12", [line.split()[0] for line in label.read_text(encoding="utf-8").splitlines()])
        self.assertEqual(sample["approval_author"], "human_test")

    def test_gutter_piece_on_tile_class_id13_validates_human_provenance(self):
        """A classe ID 13 (``pedaco_calha_sobre_telha``) é contada e aceita."""
        self.build()
        result = self.check()
        self.assertTrue(result["ready_for_training"])
        self.assertEqual(MVP_NAMES[13], "pedaco_calha_sobre_telha")
        self.assertEqual(len(result["train_boxes_by_class"]), 14)
        self.assertEqual(result["train_boxes_by_class"]["13"], 1)

    def test_single_ceasa_draft_cannot_train(self):
        """Um rascunho com um único prédio não pode treinar ("estruturalmente inválido")."""
        self.items = self.items[:1]
        self.registry["images"] = self.items
        self.build()
        with self.assertRaisesRegex(ValueError, "estruturalmente inválido"):
            self.check()

    def test_path_traversal_rejected(self):
        """Caminho com ``..`` na proveniência (travessia de diretório) é recusado."""
        self.build()
        self.provenance["images"][0]["label"] = "../registry.json"
        self.write(self.provenance_path, self.provenance)
        with self.assertRaisesRegex(ValueError, "travessia"):
            self.check()

    def test_train_tune_evaluate_without_provenance_never_load_detector(self):
        """train, tune e evaluate sem proveniência falham ANTES de carregar o detector.

        Cada etapa ainda grava um ``training_provenance.json`` de auditoria com ``valid: false``.
        EXTENSAO: nova etapa que usa dataset (ex.: outro comando de treino): acrescente no dict ``stages``.
        """
        self.build()
        self.provenance_path.unlink()
        # Cada etapa é chamada com load_detector substituído por mock.
        # EXTENSAO: nova etapa de treino que lê dataset: acrescente uma entrada aqui.
        stages = {
            "train": lambda: training.train(self.config, self.data_path),
            "tune": lambda: training.tune(self.config, self.data_path, iterations=1, epochs=1),
            "evaluate": lambda: training.evaluate(self.config, self.data_path, weights="never_loaded.pt"),
        }
        for stage, invoke in stages.items():
            with self.subTest(stage=stage), patch("dronecamp_ia.training.load_detector") as detector:
                with self.assertRaisesRegex(ValueError, "Proveniência"):
                    invoke()
                detector.assert_not_called()
        # Uma auditoria inválida por etapa, gravada em runs/<execução>/.
        audits = list((self.root / "runs").glob("*/training_provenance.json"))
        self.assertEqual(len(audits), 3)
        for audit in audits:
            self.assertFalse(json.loads(audit.read_text(encoding="utf-8"))["valid"])

    def test_prepare_approved_dataset_records_valid_provenance_without_model(self):
        """``prepare_dataset`` gera YAML resolvido seguro, auditoria válida e snapshot, sem modelo.

        Confere: caminhos absolutos, nomes na ordem, chave ``download`` removida, contagem de
        caixas por classe no train e ``dataset_snapshot.json`` criado.
        """
        self.build()
        raw = yaml.safe_load(self.data_path.read_text(encoding="utf-8"))
        # Injeta uma chave download no YAML; o YAML resolvido não pode carregá-la.
        raw["download"] = "ignored_content_not_executed"
        self.data_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        run = self.root / "prepare_approved"
        run.mkdir()
        with patch("dronecamp_ia.training.load_detector") as detector:
            resolved_path, counts = training.prepare_dataset(self.config, self.data_path, run)
            detector.assert_not_called()
        self.assertEqual(resolved_path, run / "dataset_resolved.yaml")
        resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        self.assertTrue(Path(resolved["path"]).is_absolute())
        self.assertEqual(resolved["names"], list(MVP_NAMES))
        self.assertNotIn("download", resolved)
        for split in ("train", "val", "test"):
            self.assertEqual(Path(resolved[split]), self.output / "images" / split)
        audit = json.loads((run / "training_provenance.json").read_text(encoding="utf-8"))
        self.assertTrue(audit["valid"])
        self.assertTrue(audit["ready_for_training"])
        self.assertEqual(counts["train"], {name: 1 for name in MVP_NAMES})
        self.assertTrue((run / "dataset_snapshot.json").is_file())


if __name__ == "__main__":
    unittest.main()
