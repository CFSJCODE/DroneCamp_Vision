"""Expandir classes preserva evidências, mas exige nova revisão humana integral.

Módulos cobertos:
    - ``configs/taxonomy.json`` REAL e as cópias congeladas de versões anteriores
      (``taxonomy_ceasa_v1.json`` ... ``taxonomy_ceasa_v5_reparo_rufo.json``), lidas por
      ``config.load_taxonomy``/``detection_names``: IDs, slugs, fases e rótulos;
    - ``backend.check_domain_names``: um checkpoint só serve se as classes baterem exatamente;
    - ``prediction.serialize_detections``: classes novas nunca herdam gravidade;
    - ``review_render.COLORS``: uma cor distinta por classe ativa;
    - ``review_data.migrate_review_taxonomy`` (comando ``migrate-review``): leva uma revisão
      para a taxonomia nova (acréscimo ou fusão, como a v6 → v7) preservando caixas e
      hashes, mas revogando as aprovações;
    - ``review_data.import_human_feedback`` (combinada com a migração).

Como rodar (na pasta ``sistema_ia``, com o venv do projeto):
    python -m unittest tests.test_taxonomy_migration
    python -m unittest tests.test_taxonomy_migration.TaxonomyMigrationTests
    python -m unittest tests.test_taxonomy_migration.ReviewTaxonomyMigrationFixtureTests.test_nonexpanding_taxonomy_is_rejected

O que cada classe de teste prepara:
    - ``TaxonomyMigrationTests.setUp``: carrega a configuração e a taxonomia REAIS do
      projeto (somente leitura) e o caminho da cópia congelada v1. ``_Values`` imita um tensor
      da Ultralytics para chamar ``serialize_detections`` sem modelo.
    - ``ReviewTaxonomyMigrationFixtureTests.setUp``: pasta temporária com acento contendo
      uma taxonomia antiga (2 classes) e uma nova (as 2 + 2 novas), ``ProjectConfig`` apontando
      para a nova, 2 fotos 32x24 (a 0 positiva com 2 caixas, a 1 negativa), ambas APROVADAS
      pelo revisor sintético, o ``old_registry.json`` e a saída ``fixture-v2/registry.json``.
      ``prepare_ten_class_expansion`` troca as taxonomias por 10 -> 12 classes ativas.

ATENCAO: ``TaxonomyMigrationTests`` trava o CONTRATO REAL de classes (v7, 13/10/2026: 9 ativas
e 6 de catálogo, fundidas a partir da v6). Ao mudar configs/taxonomy.json é OBRIGATÓRIO
atualizar aqui a lista de nomes e o total (15) de
``test_current_v7_has_nine_active_classes_without_overlap_and_six_in_catalog``, o mapa de
``test_v6_to_v7_map_only_merges_and_puts_tile_pieces_in_one_class`` e congelar a taxonomia
anterior num ``taxonomy_ceasa_vN_*.json`` byte a byte, com um teste de snapshot próprio.
Regras de migração travadas: acréscimo preserva IDs; fusão só pelo mapa declarado na taxonomia
nova, guardando a classe anterior em cada caixa; aprovações humanas antigas viram histórico
(``previous_human_approved``) e ``human_approved`` fica False; gravidade vira None; nenhuma
saída existente é sobrescrita.

EXTENSAO: para uma regra nova da migração, copie
``ReviewTaxonomyMigrationFixtureTests.test_nonexpanding_taxonomy_is_rejected`` (altere
``self.new_taxonomy``/``self.registry``, regrave com ``self.write_json`` e chame ``self.migrate()``);
para fusões, parta de ``prepare_merge``.
"""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image

from dronecamp_ia.backend import check_domain_names
from dronecamp_ia.config import ProjectConfig, active_class_remap, detection_names, load_config, load_taxonomy
from dronecamp_ia.io import file_hash
from dronecamp_ia.prediction import serialize_detections
from dronecamp_ia.review_data import import_human_feedback, migrate_review_taxonomy


# ---------------------------------------------------------------------------
# Substituto de tensor e testes do contrato REAL de classes
# ---------------------------------------------------------------------------
class _Values:
    """Substituto mínimo de tensor; não baixa modelo nem executa inferência.

    Imita ``boxes.xyxy``/``conf``/``cls`` da Ultralytics: só ``.cpu()`` e ``.tolist()``.
    """

    def __init__(self, values):
        """Guarda a lista que ``tolist`` vai devolver.

        Parâmetros:
            values (list): valores já no formato de lista (ex.: ``[[x1, y1, x2, y2]]``).
        """
        self.values = values

    def cpu(self):
        """Imita ``tensor.cpu()``. Retorna: o próprio objeto."""
        return self

    def tolist(self):
        """Imita ``tensor.tolist()``. Retorna: a lista guardada."""
        return self.values


class TaxonomyMigrationTests(unittest.TestCase):
    """Testes do contrato REAL de classes (configs/taxonomy.json e versões congeladas).

    Importa porque pesos, labels e revisões dependem dos IDs; uma troca silenciosa faria o
    modelo chamar um defeito pelo nome de outro.
    """
    def setUp(self) -> None:
        """Carrega a configuração e a taxonomia reais e aponta para a cópia congelada v1.

        Atributos criados: ``config``, ``taxonomy``, ``names`` (classes ativas) e ``snapshot_path``.
        Retorna: None.
        """
        self.config = load_config()
        self.taxonomy = load_taxonomy(self.config.taxonomy_path)
        self.names = detection_names(self.taxonomy)
        # Cópia congelada da primeira taxonomia (8 classes), ao lado de configs/taxonomy.json.
        self.snapshot_path = self.config.taxonomy_path.with_name("taxonomy_ceasa_v1.json")
        # Cópia congelada da v6 (14 classes): as versões anteriores são prefixos dela.
        self.v6_path = self.config.taxonomy_path.with_name("taxonomy_ceasa_v6_pedaco_calha_sobre_telha.json")
        self.v6_names = detection_names(load_taxonomy(self.v6_path))

    def test_current_v7_has_nine_active_classes_without_overlap_and_six_in_catalog(self) -> None:
        """A v7 tem 9 classes ativas (IDs 0–8) e 6 de catálogo (9–14), sem par sobreposto da v6.

        EXTENSAO: classe nova ativa: acrescente o slug no fim da lista e ajuste o total (15).
        """
        self.assertEqual(self.taxonomy["version"], "ceasa-v7-classes-sem-sobreposicao")
        self.assertEqual(self.names, [
            "telha_danificada", "telha_ausente", "residuos_cobertura", "objeto_solto_cobertura",
            "calha_obstruida", "rufo_ausente", "rufo_danificado", "reparo_material_aplicado",
            "fixador_irregular",
        ])
        # ATENCAO: 15 = total de classes (ativas + catálogo); suba ao acrescentar classe.
        self.assertEqual([category["id"] for category in self.taxonomy["classes"]], list(range(15)))
        catalog = [category for category in self.taxonomy["classes"] if category["phase"] != 1]
        self.assertEqual([category["id"] for category in catalog], list(range(9, 15)))
        self.assertTrue(all(category.get("requires_training") for category in catalog))
        for category in self.taxonomy["classes"]:
            with self.subTest(slug=category["slug"]):
                # Toda classe diz o que entra e o que não entra; gravidade nunca vem do rótulo.
                self.assertTrue(category["criterion"])
                self.assertTrue(category["excludes"])
                self.assertIsNone(category["reference_severity"])
        # A precedência resolve uma região com duas condições: cada classe aparece uma vez.
        self.assertCountEqual(self.taxonomy["precedence"], [category["slug"] for category in self.taxonomy["classes"]])

    def test_v6_snapshot_is_byte_identical_to_the_taxonomy_its_weights_and_reviews_use(self) -> None:
        """O snapshot v6 congelado tem o SHA-256 que pesos e revisões v6 guardam e 14 classes ativas."""
        snapshot = load_taxonomy(self.v6_path)
        self.assertEqual(file_hash(self.v6_path), "a4350ab8923b648642a653f4d30066b0c7745b0b9dabb1589a2b185c9ad9b08e")
        self.assertEqual(self.taxonomy["migration"]["from_taxonomy_sha256"], file_hash(self.v6_path))
        self.assertEqual(detection_names(snapshot), self.v6_names)
        self.assertEqual(len(self.v6_names), 14)
        self.assertEqual(self.taxonomy["migration"]["from_active_classes"], self.v6_names)

    def test_v6_to_v7_map_only_merges_and_puts_tile_pieces_in_one_class(self) -> None:
        """Cada classe ativa da v6 vai inteira para uma classe da v7 (fusão, nunca divisão).

        Pedaço de telha e pedaço de telha sobreposto (o par ambíguo da v6) viram a mesma
        classe; o apoio fica no atributo da caixa.
        """
        remap = active_class_remap(self.taxonomy, self.v6_names)
        targets = {self.v6_names[index]: entry["to_slug"] for index, entry in enumerate(remap)}
        self.assertEqual(targets, {
            "telha_quebrada": "telha_danificada", "telha_ausente": "telha_ausente",
            "residuos_telha": "residuos_cobertura", "reparo_telha": "reparo_material_aplicado",
            "rufo_ausente": "rufo_ausente", "rufo_quebrado": "rufo_danificado",
            "residuos_calha": "calha_obstruida", "vegetacao_calha": "calha_obstruida",
            "pedaco_telha": "objeto_solto_cobertura", "rufo_deslocado": "rufo_danificado",
            "pedaco_telha_sobreposto": "objeto_solto_cobertura", "fixador_telha_frouxo": "fixador_irregular",
            "reparo_rufo": "reparo_material_aplicado", "pedaco_calha_sobre_telha": "objeto_solto_cobertura",
        })
        self.assertEqual(remap[10]["attributes"], {"material": "telha", "apoio": "sobre_telha_instalada"})
        self.assertEqual(remap[8]["attributes"], {"material": "telha"})
        # Toda classe ativa da v7 recebe ao menos uma classe da v6: nenhuma caixa antiga se perde.
        self.assertEqual({entry["to_id"] for entry in remap}, set(range(len(self.names))))
        # Todo o catálogo v6 também tem destino (cobertura das não conformidades preservada).
        catalog_v6 = [category["slug"] for category in load_taxonomy(self.v6_path)["classes"] if category["phase"] != 1]
        self.assertCountEqual(self.taxonomy["migration"]["catalog_class_map"], catalog_v6)

    def test_v6_checkpoint_is_accepted_only_through_the_merge_map(self) -> None:
        """Pesos v6 entram na comparação/sugestões pelo mapa; sem a taxonomia são recusados."""
        checkpoint = SimpleNamespace(names=dict(enumerate(self.v6_names)))
        with self.assertRaises(ValueError):
            check_domain_names(checkpoint, self.names)
        self.assertEqual(check_domain_names(checkpoint, self.names, taxonomy=self.taxonomy), [])
        self.assertEqual(checkpoint.dronecamp_class_remap, [0, 1, 2, 7, 5, 6, 4, 4, 3, 6, 3, 8, 7, 3])
        # Pesos v5 (13 classes, prefixo da v6) também; nomes estranhos (COCO) continuam recusados.
        v5 = SimpleNamespace(names=dict(enumerate(self.v6_names[:13])))
        self.assertEqual(check_domain_names(v5, self.names, taxonomy=self.taxonomy), [])
        with self.assertRaises(ValueError):
            check_domain_names(SimpleNamespace(names={0: "person", 1: "car"}), self.names, taxonomy=self.taxonomy)

    def test_snapshot_preserves_eight_class_contract_and_original_phase2_id8(self) -> None:
        """A v1 congelada mantém as 8 classes originais; as classes de fase 2 movidas mantêm IDs novos."""
        snapshot = load_taxonomy(self.snapshot_path)
        self.assertEqual(detection_names(snapshot), self.v6_names[:8])
        self.assertEqual(snapshot["classes"][8]["slug"], "telha_sobreposta_sem_travamento")
        self.assertEqual(snapshot["classes"][8]["phase"], 2)
        v6 = load_taxonomy(self.v6_path)
        displaced = next(category for category in v6["classes"]
                         if category["slug"] == "telha_sobreposta_sem_travamento")
        self.assertEqual(displaced["id"], 19)
        self.assertEqual(displaced["phase"], 2)
        old_fixador = next(category for category in v6["classes"]
                           if category["slug"] == "fixador_telha_ausente")
        self.assertEqual(old_fixador["id"], 20)
        self.assertEqual(old_fixador["phase"], 2)

    def test_old_eight_class_checkpoint_cannot_claim_new_fourteen_class_contract(self) -> None:
        """Checkpoint de 8 classes não é aceito para o contrato atual."""
        snapshot = load_taxonomy(self.snapshot_path)
        checkpoint = SimpleNamespace(names=dict(enumerate(detection_names(snapshot))))
        with self.assertRaises(ValueError):
            check_domain_names(checkpoint, self.names)

    def test_intermediate_snapshot_preserves_nine_class_contract(self) -> None:
        """A v2 congelada (9 classes) é prefixo da v6 e seu checkpoint é recusado."""
        path = self.config.taxonomy_path.with_name("taxonomy_ceasa_v2_pedaco_telha.json")
        snapshot = load_taxonomy(path)
        names = detection_names(snapshot)
        self.assertEqual(names, self.v6_names[:9])
        self.assertEqual(snapshot["classes"][9]["slug"], "fixador_telha_ausente")
        checkpoint = SimpleNamespace(names=dict(enumerate(names)))
        with self.assertRaises(ValueError):
            check_domain_names(checkpoint, self.names)

    def test_prefix_checkpoint_only_suggests_with_allow_prefix(self) -> None:
        """Pesos de taxonomia anterior servem para sugestões (allow_prefix) e listam as classes novas."""
        checkpoint = SimpleNamespace(names=dict(enumerate(self.names[:-1])))
        self.assertEqual(check_domain_names(checkpoint, self.names, allow_prefix=True), self.names[-1:])
        self.assertEqual(check_domain_names(SimpleNamespace(names=dict(enumerate(self.names))), self.names,
                                            allow_prefix=True), [])
        with self.assertRaises(ValueError):
            check_domain_names(checkpoint, self.names)
        # Ordem trocada ou classe estranha continua recusada mesmo em modo sugestão.
        swapped = [self.names[1], self.names[0], *self.names[2:-1]]
        with self.assertRaises(ValueError):
            check_domain_names(SimpleNamespace(names=dict(enumerate(swapped))), self.names, allow_prefix=True)
        with self.assertRaises(ValueError):
            check_domain_names(SimpleNamespace(names=dict(enumerate([*self.names, "extra"]))), self.names,
                               allow_prefix=True)

    def test_previous_snapshot_preserves_ten_classes_and_phase2_fixation_ids(self) -> None:
        """A v3 congelada (10 classes) é prefixo da v6 e seu checkpoint é recusado."""
        path = self.config.taxonomy_path.with_name("taxonomy_ceasa_v3_pedaco_telha_rufo_deslocado.json")
        snapshot = load_taxonomy(path)
        names = detection_names(snapshot)
        self.assertEqual(names, self.v6_names[:10])
        self.assertEqual(snapshot["classes"][10]["slug"], "fixador_telha_corroido")
        self.assertEqual(snapshot["classes"][10]["phase"], 2)
        self.assertEqual(snapshot["classes"][11]["slug"], "fixador_telha_frouxo")
        self.assertEqual(snapshot["classes"][11]["phase"], 2)
        checkpoint = SimpleNamespace(names=dict(enumerate(names)))
        with self.assertRaises(ValueError):
            check_domain_names(checkpoint, self.names)

    def test_previous_snapshot_preserves_twelve_classes_and_original_phase2_id12(self) -> None:
        """A v4 congelada (12 classes) é prefixo da v6 e seu checkpoint é recusado."""
        path = self.config.taxonomy_path.with_name("taxonomy_ceasa_v4_fragmento_sobreposto_fixador_solto.json")
        snapshot = load_taxonomy(path)
        names = detection_names(snapshot)
        self.assertEqual(names, self.v6_names[:12])
        self.assertEqual(snapshot["classes"][12]["slug"], "reparo_selante_fixador_telha")
        self.assertEqual(snapshot["classes"][12]["phase"], 2)
        self.assertEqual(snapshot["classes"][21]["slug"], "fixador_telha_corroido")
        checkpoint = SimpleNamespace(names=dict(enumerate(names)))
        with self.assertRaises(ValueError):
            check_domain_names(checkpoint, self.names)

    def test_previous_snapshot_preserves_thirteen_classes_and_original_phase2_id13(self) -> None:
        """A v5 congelada (13 classes) é prefixo da v6 e, sem o mapa da v7, seu checkpoint é recusado."""
        path = self.config.taxonomy_path.with_name("taxonomy_ceasa_v5_reparo_rufo.json")
        snapshot = load_taxonomy(path)
        names = detection_names(snapshot)
        self.assertEqual(names, self.v6_names[:13])
        self.assertEqual(snapshot["version"], "ceasa-v5-proposta-reparo-rufo")
        self.assertEqual(snapshot["classes"][13]["slug"], "fixador_rufo_ausente")
        self.assertEqual(snapshot["classes"][13]["phase"], 2)
        checkpoint = SimpleNamespace(names=dict(enumerate(names)))
        with self.assertRaises(ValueError):
            check_domain_names(checkpoint, self.names)

    def test_review_palette_has_a_color_for_every_active_class(self) -> None:
        """A paleta da revisão tem uma cor distinta para cada classe ativa.

        EXTENSAO: ao acrescentar classe, acrescente uma cor em review_render.COLORS.
        """
        from dronecamp_ia.review_render import COLORS
        self.assertGreaterEqual(len(COLORS), len(self.names))
        self.assertEqual(len(set(COLORS[:len(self.names)])), len(self.names))

    def test_matching_fourteen_class_checkpoint_is_accepted(self) -> None:
        """Checkpoint com exatamente as classes atuais é aceito."""
        checkpoint = SimpleNamespace(names=dict(enumerate(self.names)))
        check_domain_names(checkpoint, self.names)

    def test_v7_categories_do_not_inherit_historical_engineering_severity(self) -> None:
        """Detecções de toda classe ativa da v7 saem pendentes, sem gravidade nem gravidade histórica.

        ATENCAO: gravidade nunca é inferida pela IA; as classes fundidas não escolhem uma das
        gravidades históricas das classes v6 de origem.
        """
        for class_id, slug in enumerate(self.names):
            with self.subTest(class_id=class_id):
                result = SimpleNamespace(
                    names=dict(enumerate(self.names)),
                    boxes=SimpleNamespace(
                        xyxy=_Values([[10, 20, 50, 60]]),
                        conf=_Values([0.999]), cls=_Values([class_id]),
                    ),
                )
                findings = serialize_detections(result, self.taxonomy, demo=False)
                self.assertEqual(len(findings), 1)
                finding = findings[0]
                self.assertEqual(finding["class_id"], class_id)
                self.assertEqual(finding["class_name"], slug)
                self.assertEqual(finding["category_label"], self.taxonomy["classes"][class_id]["label"])
                self.assertIsNone(finding["severity"])
                self.assertEqual(finding["review_status"], "pendente")
                self.assertIsNone(finding["report_reference"]["historical_severity_only"])


# ---------------------------------------------------------------------------
# Migração de revisões (dados sintéticos em pasta temporária)
# ---------------------------------------------------------------------------
class ReviewTaxonomyMigrationFixtureTests(unittest.TestCase):
    """Duas classes antigas e duas novas bastam para verificar a migração geral."""

    def setUp(self) -> None:
        """Monta taxonomias antiga/nova, config, 2 fotos aprovadas e o registro antigo numa pasta temporária.

        Atributos criados: ``root``, ``previous_taxonomy_path``, ``current_taxonomy_path``,
        ``old_taxonomy``, ``new_taxonomy``, ``config``, ``registry``, ``registry_path`` e
        ``output`` (fixture-v2/registry.json, que não existe ainda).
        Retorna: None.
        """
        # 1. Pasta temporária e as duas taxonomias: antiga (old_0, old_1) e nova (+ new_2, new_3).
        temporary = tempfile.TemporaryDirectory(prefix="dronecamp migration fixture á ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.previous_taxonomy_path = self.root / "old_taxonomy.json"
        self.current_taxonomy_path = self.root / "new_taxonomy.json"
        self.old_taxonomy = {
            "version": "fixture-old",
            "classes": [self.category(0, "old_0"), self.category(1, "old_1")],
        }
        self.new_taxonomy = {
            "version": "fixture-new",
            "classes": deepcopy(self.old_taxonomy["classes"]) +
                       [self.category(2, "new_2"), self.category(3, "new_3")],
        }
        self.write_json(self.previous_taxonomy_path, self.old_taxonomy)
        self.write_json(self.current_taxonomy_path, self.new_taxonomy)
        # 2. Configuração apontando para a taxonomia NOVA (destino da migração).
        self.config = ProjectConfig(
            root=self.root, model="fixture.pt", requested_model="fixture.pt",
            taxonomy_path=self.current_taxonomy_path, dataset_path=self.root / "dataset.yaml",
            device="cpu", prediction={}, training={},
        )
        # 3. Duas fotos aprovadas pelo revisor sintético: a 0 positiva com 2 caixas, a 1 negativa.
        # A gravidade antiga "fixture_old_value" deve virar None na migração.
        records = []
        for index, color in enumerate(((160, 70, 30), (30, 70, 160))):
            source = self.root / f"fixture_{index}.png"
            Image.new("RGB", (32, 24), color=color).save(source)
            records.append({
                "image_sha256": file_hash(source), "source_path": str(source),
                "filename": source.name, "width": 32, "height": 24, "pages": [index + 1],
                "status": "positive" if index == 0 else "negative",
                "boxes": [
                    {"class_id": 0, "bbox_xyxy": [2, 3, 10, 13], "note": "old fixture box0"},
                    {"class_id": 1, "bbox_xyxy": [15, 4, 25, 17], "note": "old fixture box1"},
                ] if index == 0 else [],
                "human_approved": True, "technical_status": "human_visual_reviewed",
                "human_reviewer": "synthetic_old_fixture_review",
                "visual_review_status": "second_pass_ai", "building_group": "fixture_building",
                "severity": "fixture_old_value", "notes": "Synthetic test only.",
            })
        # 4. Registro antigo, ligado ao hash da taxonomia antiga, e o caminho de saída.
        self.registry = {
            "schema_version": 1, "version": "fixture-old",
            "taxonomy_sha256": file_hash(self.previous_taxonomy_path), "images": records,
        }
        self.registry_path = self.root / "old_registry.json"
        self.write_json(self.registry_path, self.registry)
        self.output = self.root / "fixture-v2" / "registry.json"

    @staticmethod
    def category(class_id: int, slug: str) -> dict:
        """Monta uma classe ativa (fase 1) com rótulo igual ao slug.

        Parâmetros:
            class_id (int): ID da classe.
            slug (str): nome interno.
        Retorna: dict ``{"id", "slug", "label", "phase": 1}``.
        """
        return {"id": class_id, "slug": slug, "label": slug, "phase": 1}

    @staticmethod
    def write_json(path: Path, value: dict) -> None:
        """Grava ``value`` como JSON indentado com quebra de linha final.

        Parâmetros:
            path (Path): destino.
            value (dict): conteúdo.
        Retorna: None.
        """
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def migrate(self) -> dict:
        """Migra ``self.registry_path`` da taxonomia antiga para a atual, gravando em ``self.output``.

        Retorna: o registro migrado (dict).
        Erros: propaga o ValueError de ``migrate_review_taxonomy``.
        """
        return migrate_review_taxonomy(self.config, self.registry_path, self.previous_taxonomy_path, self.output)

    def test_append_preserves_boxes_image_hashes_and_input_bytes_without_inventing_new_boxes(self) -> None:
        """Migrar preserva caixas, hashes e arquivos de entrada, e não inventa caixas das classes novas."""
        paths = [self.registry_path, self.previous_taxonomy_path, self.current_taxonomy_path]
        paths.extend(Path(image["source_path"]) for image in self.registry["images"])
        before = {path: path.read_bytes() for path in paths}
        result = self.migrate()
        self.assertEqual(before, {path: path.read_bytes() for path in paths})
        self.assertEqual([image["image_sha256"] for image in result["images"]],
                         [image["image_sha256"] for image in self.registry["images"]])
        self.assertEqual([image["boxes"] for image in result["images"]],
                         [image["boxes"] for image in self.registry["images"]])
        self.assertEqual(result["taxonomy_migration"]["new_class_ids"], [2, 3])
        self.assertEqual(result["taxonomy_migration"]["source_registry_sha256"], file_hash(self.registry_path))
        self.assertEqual(result["taxonomy_sha256"], file_hash(self.current_taxonomy_path))
        self.assertEqual(result["parent_registry_version"], "fixture-old")
        self.assertEqual(result["version"], "fixture-v2")
        self.assertFalse(any(box["class_id"] in (2, 3) for image in result["images"] for box in image["boxes"]))

    def test_previous_negative_is_ambiguous_until_new_classes_are_reviewed(self) -> None:
        """Foto negativa vira "ambiguous" até as classes novas serem conferidas.

        "Sem defeito" antes não garante ausência das classes que ainda não existiam.
        """
        second = self.migrate()["images"][1]
        self.assertEqual(second["status"], "ambiguous")
        self.assertEqual(second["previous_status"], "negative")
        self.assertEqual(second["boxes"], [])
        self.assertTrue(second["taxonomy_recheck_pending"])
        self.assertFalse(second["human_approved"])

    def prepare_ten_class_expansion(self) -> None:
        """Simula um catálogo antigo sem tocar nas fotos ou revisões do projeto.

        Taxonomia antiga: 10 ativas + 2 de fase 2; nova: 12 ativas (10 e 11 novas) + 1 de fase 2.
        Regrava taxonomias e registro (com o hash da taxonomia antiga). Retorna: None.
        """
        # IDs 10 e 11 eram de fase 2 na antiga; na nova, 10 é classe nova, 11 foi promovida e a
        # de fase 2 "corroído" passou para o ID 12.
        old_active = [self.category(class_id, f"old_{class_id}") for class_id in range(10)]
        corroded = {**self.category(10, "fixador_telha_corroido"), "phase": 2}
        loose = {**self.category(11, "fixador_telha_frouxo"), "phase": 2}
        self.old_taxonomy["classes"] = old_active + [corroded, loose]
        self.new_taxonomy["classes"] = deepcopy(old_active) + [
            self.category(10, "pedaco_telha_sobreposto"),
            self.category(11, "fixador_telha_frouxo"),
            {**self.category(12, "fixador_telha_corroido"), "phase": 2},
        ]
        self.write_json(self.previous_taxonomy_path, self.old_taxonomy)
        self.write_json(self.current_taxonomy_path, self.new_taxonomy)
        self.registry["taxonomy_sha256"] = file_hash(self.previous_taxonomy_path)
        self.write_json(self.registry_path, self.registry)

    def prepare_merge(self, attributes: dict | None = None) -> None:
        """Taxonomia nova que FUNDE classes (como a v6 → v7), sem tocar nos dados do projeto.

        Antiga: old_0, old_1, old_2 (ativas). Nova: ``merged`` (old_0 + old_1, atributo
        ``tipo``) e ``kept`` (= old_2, escopo inalterado), com o mapa em ``migration``.
        A foto 0 ganha uma terceira caixa (old_2) e uma caixa old_1 sobre a caixa old_0.
        Parâmetros: attributes (dict | None): atributos do mapa para old_1 (padrão tipo b).
        Retorna: None.
        """
        self.old_taxonomy["classes"] = [self.category(index, f"old_{index}") for index in range(3)]
        self.write_json(self.previous_taxonomy_path, self.old_taxonomy)
        merged = {**self.category(0, "merged"), "attributes": {"tipo": ["a", "b"]}}
        self.new_taxonomy = {
            "version": "fixture-merge",
            "classes": [merged, self.category(1, "kept")],
            "migration": {
                "kind": "remap", "from_taxonomy_sha256": file_hash(self.previous_taxonomy_path),
                "from_active_classes": ["old_0", "old_1", "old_2"],
                "active_class_map": {
                    "old_0": {"to": "merged", "attributes": {"tipo": "a"}},
                    "old_1": {"to": "merged", "attributes": attributes if attributes is not None else {"tipo": "b"}},
                    "old_2": {"to": "kept", "attributes": {}},
                },
                "unchanged_scope_classes": ["kept"],
            },
        }
        self.write_json(self.current_taxonomy_path, self.new_taxonomy)
        self.registry["images"][0]["boxes"] += [
            {"class_id": 2, "bbox_xyxy": [20, 18, 30, 23], "note": "old fixture box2"},
            {"class_id": 1, "bbox_xyxy": [2, 3, 10, 12], "note": "old_1 sobre a caixa old_0"},
        ]
        self.registry["taxonomy_sha256"] = file_hash(self.previous_taxonomy_path)
        self.write_json(self.registry_path, self.registry)

    def test_merge_relabels_boxes_keeps_origin_and_inputs_and_requires_review_of_widened_class(self) -> None:
        """Fusão: caixas mudam de ID sem redesenho, guardam a classe antiga e só a classe inalterada conta como revista."""
        self.prepare_merge()
        paths = [self.registry_path, self.previous_taxonomy_path, self.current_taxonomy_path]
        before = {path: path.read_bytes() for path in paths}

        result = self.migrate()

        self.assertEqual(before, {path: path.read_bytes() for path in paths})
        first = result["images"][0]
        self.assertEqual([box["class_id"] for box in first["boxes"]], [0, 0, 1, 0])
        self.assertEqual([box["previous_class_slug"] for box in first["boxes"]], ["old_0", "old_1", "old_2", "old_1"])
        self.assertEqual([box["attributes"] for box in first["boxes"]], [{"tipo": "a"}, {"tipo": "b"}, {}, {"tipo": "b"}])
        self.assertEqual([box["bbox_xyxy"] for box in first["boxes"]],
                         [box["bbox_xyxy"] for box in self.registry["images"][0]["boxes"]])
        # A caixa old_1 sobre a old_0 virou par da mesma classe: sinalizado, nunca apagado.
        self.assertEqual(first["taxonomy_merge_overlaps"], 1)
        migration = result["taxonomy_migration"]
        self.assertEqual(migration["kind"], "remap")
        self.assertEqual(migration["new_class_ids"], [0])
        self.assertEqual(migration["merge_overlap_images"], 1)
        self.assertEqual([entry["to_slug"] for entry in migration["class_map"]], ["merged", "merged", "kept"])
        for image in result["images"]:
            self.assertEqual(image["classes_reviewed_before_migration"], [1])
            self.assertTrue(image["taxonomy_recheck_pending"])
            self.assertFalse(image["human_approved"])
            self.assertTrue(image["previous_human_approved"])
            self.assertEqual(image["historical_human_review"]["classes_reviewed"], [0, 1, 2])
        self.assertEqual(result["scope"], "propostas_preservadas_fusao_de_classes_pendente_de_revisao")

    def test_merge_map_written_for_another_taxonomy_is_rejected_without_output(self) -> None:
        """O mapa só vale para a taxonomia de origem declarada (hash ou prefixo das classes)."""
        self.prepare_merge()
        self.new_taxonomy["migration"]["from_taxonomy_sha256"] = "0" * 64
        self.new_taxonomy["migration"]["from_active_classes"] = ["outra", "old_1", "old_2"]
        self.write_json(self.current_taxonomy_path, self.new_taxonomy)
        with self.assertRaises(ValueError):
            self.migrate()
        self.assertFalse(self.output.exists())

    def test_merge_map_with_undeclared_attribute_value_is_rejected(self) -> None:
        """Atributo fora dos valores da classe de destino não vira dado "revisado"."""
        self.prepare_merge(attributes={"tipo": "inventado"})
        with self.assertRaises(ValueError):
            self.migrate()
        self.assertFalse(self.output.exists())

    def test_ten_to_twelve_classes_adds_ids10_and11_without_relabeling_old_boxes(self) -> None:
        """De 10 para 12 classes: IDs 10 e 11 entram, caixas antigas ficam iguais e o escopo revisado é 0..9."""
        # A promoção de uma categoria do catálogo também expande o contrato ativo.
        # Os arquivos de origem são sintéticos; nenhuma aprovação real é criada.
        self.prepare_ten_class_expansion()
        before = self.registry_path.read_bytes()

        result = self.migrate()

        self.assertEqual(self.registry_path.read_bytes(), before)
        self.assertEqual(result["taxonomy_migration"]["new_class_ids"], [10, 11])
        self.assertEqual(result["taxonomy_migration"]["new_active_classes"][10:],
                         ["pedaco_telha_sobreposto", "fixador_telha_frouxo"])
        self.assertEqual([image["boxes"] for image in result["images"]],
                         [image["boxes"] for image in self.registry["images"]])
        self.assertEqual([image["image_sha256"] for image in result["images"]],
                         [image["image_sha256"] for image in self.registry["images"]])
        for image in result["images"]:
            self.assertEqual(image["classes_reviewed_before_migration"], list(range(10)))
            self.assertTrue(image["taxonomy_recheck_pending"])
            self.assertFalse(image["human_approved"])
            self.assertFalse(any(box["class_id"] in (10, 11) for box in image["boxes"]))

    def test_repeated_migration_preserves_actual_eight_class_review_scope(self) -> None:
        """Uma revisão ainda pendente conserva o escopo realmente revisado (0..7) ao migrar de novo."""
        self.prepare_ten_class_expansion()
        for image in self.registry["images"]:
            image["taxonomy_recheck_pending"] = True
            image["classes_reviewed_before_migration"] = list(range(8))
            image["visual_review_status"] = "prior_taxonomy_ai_review"
        self.write_json(self.registry_path, self.registry)
        before = self.registry_path.read_bytes()

        result = self.migrate()

        self.assertEqual(self.registry_path.read_bytes(), before)
        for image in result["images"]:
            self.assertEqual(image["classes_reviewed_before_migration"], list(range(8)))
            self.assertTrue(image["taxonomy_recheck_pending"])
            self.assertFalse(image["human_approved"])

    def test_pending_previous_review_without_scope_does_not_invent_reviewed_classes(self) -> None:
        """Revisão pendente sem escopo registrado fica com escopo vazio (nada é presumido)."""
        self.prepare_ten_class_expansion()
        self.registry["images"][0]["taxonomy_recheck_pending"] = True
        self.write_json(self.registry_path, self.registry)

        first = self.migrate()["images"][0]

        self.assertEqual(first["classes_reviewed_before_migration"], [])
        self.assertTrue(first["taxonomy_recheck_pending"])

    def test_invalid_previous_review_scope_is_rejected_without_migration_output(self) -> None:
        """Escopo revisado malformado (texto, ID fora, bool, lista aninhada, repetido) é recusado sem saída."""
        self.prepare_ten_class_expansion()
        # Escopos inválidos: texto, ID fora da taxonomia antiga, bool, texto em vez de lista,
        # lista aninhada e ID repetido.
        for invalid in ([0, "1"], [0, 10], [0, True], "0,1", [0, [1]], [0, 0]):
            with self.subTest(invalid=invalid):
                self.registry["images"][0]["taxonomy_recheck_pending"] = True
                self.registry["images"][0]["classes_reviewed_before_migration"] = invalid
                self.write_json(self.registry_path, self.registry)
                with self.assertRaises(ValueError):
                    self.migrate()
                self.assertFalse(self.output.exists())

    def test_old_human_approvals_are_revoked_for_every_image_after_expansion(self) -> None:
        """Toda aprovação humana antiga é revogada (vira histórico) e a gravidade é apagada."""
        result = self.migrate()
        for image in result["images"]:
            self.assertFalse(image["human_approved"])
            self.assertEqual(image["technical_status"], "pending_human_review")
            self.assertTrue(image["taxonomy_recheck_pending"])
            self.assertEqual(image["classes_reviewed_before_migration"], [0, 1])
            self.assertEqual(image["previous_visual_review_status"], "second_pass_ai")
            self.assertEqual(image["visual_review_status"], "prior_taxonomy_human_review")
            self.assertTrue(image["previous_human_approved"])
            self.assertIsNone(image["severity"])
        summary = json.loads((self.output.parent / "review_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["human_approved_images"], 0)
        self.assertFalse(summary["ready_for_independent_evaluation"])

    def test_human_review_origin_notes_and_hashes_are_preserved_without_current_approval(self) -> None:
        """Notas, revisor, data e hashes da revisão humana antiga ficam em ``historical_human_review``."""
        for image in self.registry["images"]:
            image["human_notes"] = "Synthetic human correction preserved verbatim."
            image["human_review_at"] = "2026-10-03T22:15:00Z"
        self.write_json(self.registry_path, self.registry)
        before = self.registry_path.read_bytes()

        result = self.migrate()

        self.assertEqual(self.registry_path.read_bytes(), before)
        for image in result["images"]:
            self.assertEqual(image["visual_review_status"], "prior_taxonomy_human_review")
            self.assertTrue(image["previous_human_approved"])
            self.assertEqual(image["human_notes"], "Synthetic human correction preserved verbatim.")
            self.assertEqual(image["human_reviewer"], "synthetic_old_fixture_review")
            self.assertEqual(image["human_review_at"], "2026-10-03T22:15:00Z")
            historical = image["historical_human_review"]
            self.assertEqual(historical["taxonomy_sha256"], file_hash(self.previous_taxonomy_path))
            self.assertEqual(historical["source_registry_sha256"], file_hash(self.registry_path))
            self.assertEqual(historical["classes_reviewed"], [0, 1])
            self.assertEqual(historical["reviewer"], image["human_reviewer"])
            self.assertEqual(historical["review_at"], image["human_review_at"])
            self.assertFalse(image["human_approved"])
            self.assertTrue(image["taxonomy_recheck_pending"])
            self.assertEqual(image["technical_status"], "pending_human_review")

    def test_imported_completed_feedback_keeps_human_provenance_when_expanding_taxonomy(self) -> None:
        """Feedback importado na taxonomia antiga mantém a proveniência humana depois da migração.

        A foto concluída ganha escopo 0..1; a ambígua continua com escopo 0 e sem histórico humano.
        """
        # Este fluxo usa somente fotos e revisor sintéticos no diretório temporário.
        # Ele reproduz a importação contra o catálogo da página original antes de
        # migrar suas correções para as classes acrescentadas posteriormente.
        for image in self.registry["images"]:
            image["human_approved"] = False
            image["technical_status"] = "pending_human_review"
            image["taxonomy_recheck_pending"] = True
            image["classes_reviewed_before_migration"] = [0]
            image["visual_review_status"] = "prior_taxonomy_ai_review"
        self.write_json(self.registry_path, self.registry)
        # Feedback sintético: foto 0 concluída (classe 1 corrigida), foto 1 ambígua.
        feedback_path = self.root / "synthetic_feedback.json"
        self.write_json(feedback_path, {
            "schema_version": 1, "registry_sha256": file_hash(self.registry_path),
            "reviewer_id": "synthetic_user_reviewer", "exported_at": "2026-10-03T22:15:00Z",
            "images": [
                {"image_sha256": self.registry["images"][0]["image_sha256"],
                 "status": "positive", "confirmed_complete": True,
                 "boxes": [{"class_id": 1, "bbox_xyxy": [3, 2, 14, 16]}],
                 "notes": "Synthetic corrected occurrence."},
                {"image_sha256": self.registry["images"][1]["image_sha256"],
                 "status": "ambiguous", "confirmed_complete": False,
                 "boxes": [], "notes": "Synthetic uncertain decision."},
            ],
        })
        # A importação usa a taxonomia ANTIGA (a da página em que o revisor trabalhou).
        old_config = ProjectConfig(
            root=self.root, model="fixture.pt", requested_model="fixture.pt",
            taxonomy_path=self.previous_taxonomy_path, dataset_path=self.root / "dataset.yaml",
            device="cpu", prediction={}, training={},
        )
        imported_path = self.root / "synthetic-human-v1" / "registry.json"
        imported = import_human_feedback(old_config, self.registry_path, feedback_path, imported_path)
        imported_bytes = imported_path.read_bytes()

        result = migrate_review_taxonomy(
            self.config, imported_path, self.previous_taxonomy_path, self.output,
        )

        self.assertEqual(imported_path.read_bytes(), imported_bytes)
        self.assertEqual(result["human_feedback"], imported["human_feedback"])
        self.assertEqual(result["human_feedback"]["sha256"], file_hash(feedback_path))
        completed, pending = result["images"]
        self.assertEqual(completed["boxes"], imported["images"][0]["boxes"])
        self.assertEqual(completed["visual_review_status"], "prior_taxonomy_human_review")
        self.assertEqual(completed["classes_reviewed_before_migration"], [0, 1])
        self.assertEqual(completed["human_notes"], "Synthetic corrected occurrence.")
        self.assertEqual(completed["historical_human_review"]["source_registry_sha256"], file_hash(imported_path))
        self.assertEqual(pending["visual_review_status"], "prior_taxonomy_ai_review")
        self.assertEqual(pending["classes_reviewed_before_migration"], [0])
        self.assertEqual(pending["human_notes"], "Synthetic uncertain decision.")
        self.assertFalse(any(image["human_approved"] for image in result["images"]))
        self.assertTrue(all(image["taxonomy_recheck_pending"] for image in result["images"]))

    def test_repeated_migration_keeps_human_origin_review_scope_and_hash_chain(self) -> None:
        """Migrar duas vezes mantém o histórico humano e encadeia as migrações por hash."""
        first = self.migrate()
        first_bytes = self.output.read_bytes()
        # Segunda expansão: acrescenta new_4 e migra a saída da primeira migração.
        expanded_taxonomy = self.root / "expanded_taxonomy.json"
        expanded = deepcopy(self.new_taxonomy)
        expanded["version"] = "fixture-expanded-again"
        expanded["classes"].append(self.category(4, "new_4"))
        self.write_json(expanded_taxonomy, expanded)
        second_config = ProjectConfig(
            root=self.root, model="fixture.pt", requested_model="fixture.pt",
            taxonomy_path=expanded_taxonomy, dataset_path=self.root / "dataset.yaml",
            device="cpu", prediction={}, training={},
        )
        second_output = self.root / "fixture-v3" / "registry.json"

        second = migrate_review_taxonomy(
            second_config, self.output, self.current_taxonomy_path, second_output,
        )

        self.assertEqual(self.output.read_bytes(), first_bytes)
        for first_image, second_image in zip(first["images"], second["images"]):
            self.assertEqual(second_image["boxes"], first_image["boxes"])
            self.assertEqual(second_image["classes_reviewed_before_migration"], [0, 1])
            self.assertEqual(second_image["historical_human_review"], first_image["historical_human_review"])
            self.assertEqual(second_image["visual_review_status"], "prior_taxonomy_human_review")
            self.assertEqual(second_image["previous_visual_review_status"], "prior_taxonomy_human_review")
            self.assertTrue(second_image["previous_human_approved"])
            self.assertFalse(second_image["human_approved"])
            self.assertTrue(second_image["taxonomy_recheck_pending"])
        history = second["taxonomy_migration_history"]
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0], first["taxonomy_migration"])
        self.assertEqual(history[0]["source_registry_sha256"], file_hash(self.registry_path))
        self.assertEqual(history[1]["source_registry_sha256"], file_hash(self.output))
        self.assertEqual(history[1]["source_registry_path"], str(self.output.resolve()))
        self.assertEqual(history[1], second["taxonomy_migration"])
        summary = json.loads((second_output.parent / "review_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["human_approved_images"], 0)
        self.assertEqual(summary["taxonomy_recheck_pending_images"], 2)

    def test_completed_ten_class_human_review_expands_scope_while_pending_review_stays_at_eight(self) -> None:
        """Revisão humana concluída na taxonomia de 10 conta as 10; a pendente fica com as 8 de antes."""
        self.prepare_ten_class_expansion()
        for image in self.registry["images"]:
            image["classes_reviewed_before_migration"] = list(range(8))
            image["visual_review_status"] = "prior_taxonomy_ai_review"
        completed, pending = self.registry["images"]
        completed["taxonomy_recheck_pending"] = False
        pending["human_approved"] = False
        pending["technical_status"] = "pending_human_review"
        pending["taxonomy_recheck_pending"] = True
        pending["status"] = "ambiguous"
        self.write_json(self.registry_path, self.registry)

        completed, pending = self.migrate()["images"]

        self.assertEqual(completed["classes_reviewed_before_migration"], list(range(10)))
        self.assertEqual(completed["visual_review_status"], "prior_taxonomy_human_review")
        self.assertTrue(completed["previous_human_approved"])
        self.assertEqual(pending["classes_reviewed_before_migration"], list(range(8)))
        self.assertEqual(pending["visual_review_status"], "prior_taxonomy_ai_review")
        self.assertFalse(pending["previous_human_approved"])
        self.assertNotIn("historical_human_review", pending)
        self.assertFalse(completed["human_approved"])
        self.assertFalse(pending["human_approved"])

    def test_inconsistent_human_flags_cannot_create_historical_human_approval(self) -> None:
        """Sinais humanos inconsistentes nunca viram aprovação histórica.

        EXTENSAO: nova combinação inválida de campos: acrescente um dict na tupla ``changes``.
        """
        # Cada mudança torna incoerente a aprovação humana da foto 0 (texto no lugar de bool,
        # estado pendente, revisor vazio/ausente, foto ambígua, revisão ainda pendente).
        changes = (
            {"human_approved": "true"},
            {"technical_status": "pending_human_review"},
            {"human_reviewer": "   "},
            {"human_reviewer": None},
            {"status": "ambiguous"},
            {"taxonomy_recheck_pending": True, "classes_reviewed_before_migration": [0]},
        )
        for index, change in enumerate(changes):
            with self.subTest(change=change):
                registry = deepcopy(self.registry)
                registry["images"][0].update(change)
                self.write_json(self.registry_path, registry)
                output = self.root / f"invalid-human-{index}" / "registry.json"

                first = migrate_review_taxonomy(
                    self.config, self.registry_path, self.previous_taxonomy_path, output,
                )["images"][0]

                self.assertEqual(first["visual_review_status"], "prior_taxonomy_ai_review")
                self.assertFalse(first["previous_human_approved"])
                self.assertNotIn("historical_human_review", first)
                self.assertFalse(first["human_approved"])
                self.assertTrue(first["taxonomy_recheck_pending"])

    def test_malformed_migration_history_is_rejected_without_new_output(self) -> None:
        """``taxonomy_migration_history`` malformado é recusado sem gravar saída."""
        for history in ("history", [None], ["migration"]):
            with self.subTest(history=history):
                self.registry["taxonomy_migration_history"] = history
                self.write_json(self.registry_path, self.registry)
                with self.assertRaises(ValueError):
                    self.migrate()
                self.assertFalse(self.output.exists())

    def test_existing_migrated_output_cannot_be_overwritten(self) -> None:
        """Saída de migração já existente nunca é sobrescrita."""
        self.output.parent.mkdir()
        previous = b"preserve old migration output"
        self.output.write_bytes(previous)
        with self.assertRaises(ValueError):
            self.migrate()
        self.assertEqual(self.output.read_bytes(), previous)

    def test_reordering_existing_class_slugs_is_rejected(self) -> None:
        """Trocar a ordem de classes existentes na taxonomia nova é recusado."""
        self.new_taxonomy["classes"][0]["slug"] = "old_1"
        self.new_taxonomy["classes"][1]["slug"] = "old_0"
        self.write_json(self.current_taxonomy_path, self.new_taxonomy)
        with self.assertRaises(ValueError):
            self.migrate()
        self.assertFalse(self.output.exists())

    def test_nonexpanding_taxonomy_is_rejected(self) -> None:
        """Taxonomia "nova" sem classe nova não é migração e é recusada."""
        self.new_taxonomy["classes"] = deepcopy(self.old_taxonomy["classes"])
        self.write_json(self.current_taxonomy_path, self.new_taxonomy)
        with self.assertRaises(ValueError):
            self.migrate()
        self.assertFalse(self.output.exists())

    def test_previous_taxonomy_must_match_original_registry_hash(self) -> None:
        """A taxonomia antiga informada precisa ter o hash gravado no registro."""
        self.old_taxonomy["version"] = "fixture-old-altered"
        self.write_json(self.previous_taxonomy_path, self.old_taxonomy)
        with self.assertRaises(ValueError):
            self.migrate()
        self.assertFalse(self.output.exists())

    def test_changed_original_image_blocks_migration_before_any_output(self) -> None:
        """Foto original alterada bloqueia a migração antes de gravar qualquer coisa."""
        source = Path(self.registry["images"][0]["source_path"])
        Image.new("RGB", (32, 24), color="white").save(source)
        with self.assertRaises(ValueError):
            self.migrate()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
