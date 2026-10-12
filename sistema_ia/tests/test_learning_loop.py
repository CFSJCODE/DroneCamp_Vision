"""Ciclo de aprendizado: duplicatas, caminhos de outra máquina e propostas de IA.

Módulos cobertos
----------------
- ``dronecamp_ia.pilot``: ``consolidate_duplicates`` (uma ocorrência = uma caixa),
  ``build_pilot_dataset`` (monta o dataset piloto a partir do registry.json) e
  ``validate_pilot_dataset`` (reconfere hashes e labels antes do treino).
- ``dronecamp_ia.io``: ``resolve_local_path`` (reancora caminhos gravados em outra
  máquina, ex.: ``E:\\...\\sistema_ia\\...``) e ``file_hash`` (SHA-256 usado nos registros).
- ``dronecamp_ia.review_data``: ``import_human_feedback`` (importa a revisão humana
  exportada pela plataforma) e ``apply_ai_proposals`` (preenche a revisão com
  propostas de uma IA, sem nunca aprovar nada).

Como rodar (na pasta ``sistema_ia``)
------------------------------------
- Só este arquivo::

      DRONECAMP_SKIP_E2E=1 python -m unittest tests.test_learning_loop -v

- Um teste só (classe.método)::

      python -m unittest tests.test_learning_loop.AiProposalTests.test_human_decision_is_never_overwritten -v

O que cada classe prepara
-------------------------
- ``ConsolidateDuplicatesTests``: nada no disco; usa só dicionários de caixa feitos por ``box``.
- ``ResolveLocalPathTests``: pasta temporária com ``sistema_ia/data/reference/foto.jpg``.
- ``LearningLoopFixture`` (base, sem testes próprios): pasta temporária com um projeto
  mínimo ``sistema_ia/`` contendo taxonomia de 2 classes, ``ProjectConfig`` em CPU,
  8 fotos PNG 64x48 em ``fotos/`` e um ``revisao/registry.json`` com as 8 fotos
  aprovadas por humano (a foto 0 tem uma sugestão aceita duplicada).
- ``PilotDatasetTests``: herda a fixture e monta/valida o dataset piloto.
- ``AiProposalTests``: herda a fixture, deixa as fotos 1..7 pendentes (só a 0 aprovada)
  e grava propostas de IA em ``propostas.json``.

EXTENSAO: para testar uma função nova do ciclo de aprendizado (ex.: uma nova regra em
``consolidate_duplicates`` ou um novo campo em ``apply_ai_proposals``), copie o teste
mais parecido (``test_highest_confidence_suggestion_wins_among_suggestions`` para
regras de caixa, ``test_proposals_fill_review_without_approval_and_list_new_classes``
para propostas), troque o nome do método (precisa começar com ``test_``), monte a
entrada com ``box(...)``/``self.proposal(...)`` e afirme o resultado esperado. Se a
função precisar de projeto no disco, crie a classe herdando de ``LearningLoopFixture``.

ATENCAO: tudo é gravado em pastas temporárias apagadas no fim (``addCleanup``); nenhum
teste toca ``data/`` nem ``configs/`` reais do projeto.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from PIL import Image

from dronecamp_ia.config import ProjectConfig
from dronecamp_ia.io import file_hash, resolve_local_path
from dronecamp_ia.pilot import build_pilot_dataset, consolidate_duplicates, validate_pilot_dataset
from dronecamp_ia.review_data import apply_ai_proposals, import_human_feedback

# ---------------------------------------------------------------------------
# Constantes e helpers compartilhados pelos testes
# ---------------------------------------------------------------------------

# Nota que a plataforma grava quando o revisor aceita uma sugestão do modelo.
# O ``{}`` recebe a confiança em % (ex.: ACCEPTED.format(97)). consolidate_duplicates
# lê essa confiança da nota para decidir qual sugestão aceita fica.
# ATENCAO: se o texto da nota mudar na plataforma/pilot.py, atualize aqui também,
# senão os testes de duplicata deixam de reconhecer a caixa como "sugestão aceita".
ACCEPTED = "Sugestão do modelo aceita (confiança {}%); conferida na revisão local."


def box(class_id: int, coordinates: list[int], note: str = "Caixa conferida/editada na revisão local.") -> dict:
    """Monta uma caixa no formato do registry.json.

    Parâmetros:
        class_id: ID da classe na taxonomia (inteiro, ex.: 0 = telha_quebrada na fixture).
        coordinates: [x1, y1, x2, y2] em pixels da foto (formato xyxy).
        note: nota da caixa. O padrão indica caixa desenhada/editada pelo revisor;
            use ``ACCEPTED.format(conf)`` para simular sugestão aceita do modelo.

    Retorna:
        dict com ``class_id``, ``bbox_xyxy`` e ``note``.
    """
    return {"class_id": class_id, "bbox_xyxy": coordinates, "note": note}


# ---------------------------------------------------------------------------
# consolidate_duplicates: uma ocorrência, uma caixa
# ---------------------------------------------------------------------------

class ConsolidateDuplicatesTests(unittest.TestCase):
    """Regras de ``pilot.consolidate_duplicates`` sem tocar no disco.

    Importa porque caixas duplicadas sobre o mesmo alvo ensinam ao modelo previsões
    conflitantes. Todos os testes usam IoU limite 0.7 (o mesmo de ``pilot.duplicate_iou``
    em configs/project.yaml) ou ``None`` (política desligada).
    """

    def test_reviewer_box_wins_over_accepted_suggestion_of_same_object(self):
        """Caixa desenhada pelo revisor vence a sugestão aceita sobre o mesmo objeto.

        A decisão humana tem prioridade sobre a do modelo, mesmo com confiança 97%.
        """
        # Duas caixas quase iguais (IoU alto) da classe 2: a primeira é sugestão aceita.
        drawn = box(2, [10, 10, 50, 50])
        result = consolidate_duplicates([box(2, [11, 10, 51, 50], ACCEPTED.format(97)), drawn], 0.7)
        # Só a do revisor deve sobrar.
        self.assertEqual(result, [drawn])

    def test_highest_confidence_suggestion_wins_among_suggestions(self):
        """Entre duas sugestões aceitas do mesmo objeto, fica a de maior confiança (85% > 20%)."""
        low, high = box(3, [0, 0, 40, 40], ACCEPTED.format(20)), box(3, [1, 1, 40, 40], ACCEPTED.format(85))
        self.assertEqual(consolidate_duplicates([low, high], 0.7), [high])

    def test_other_class_or_low_overlap_is_kept_in_original_order(self):
        """Caixas de classe diferente ou com pouca sobreposição não são duplicatas.

        Também garante que a ordem original é preservada (o label YOLO sai nessa ordem).
        """
        # 1ª e 2ª: mesmo lugar, classes diferentes; 1ª e 3ª: mesma classe, IoU baixo.
        boxes = [box(2, [0, 0, 40, 40]), box(6, [0, 0, 40, 40]), box(2, [30, 30, 80, 80])]
        self.assertEqual(consolidate_duplicates(boxes, 0.7), boxes)

    def test_disabled_policy_keeps_every_box(self):
        """Com limite ``None`` (política desligada), nada é removido, nem caixas idênticas."""
        boxes = [box(2, [0, 0, 40, 40]), box(2, [0, 0, 40, 40])]
        self.assertEqual(consolidate_duplicates(boxes, None), boxes)


# ---------------------------------------------------------------------------
# resolve_local_path: caminhos gravados em outra máquina
# ---------------------------------------------------------------------------

class ResolveLocalPathTests(unittest.TestCase):
    """Reancoragem de caminhos por ``io.resolve_local_path``.

    Importa porque o registry.json guarda caminhos absolutos da máquina onde a
    revisão foi feita (Windows do Cláudio, outro clone etc.).
    """

    def setUp(self):
        """Cria ``<tmp>/sistema_ia/data/reference/foto.jpg`` (1 byte) como alvo real."""
        # Pasta temporária apagada automaticamente ao fim de cada teste.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        # A raiz precisa se chamar "sistema_ia": é essa pasta que a função procura no caminho gravado.
        self.root = Path(temporary.name) / "sistema_ia"
        self.target = self.root / "data" / "reference" / "foto.jpg"
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"x")

    def test_windows_and_posix_paths_from_other_machines_are_reanchored(self):
        """Caminhos Windows (``E:\\...``) e POSIX de outro clone apontam para o arquivo local.

        O trecho depois de ``sistema_ia`` é reaproveitado sob a raiz deste projeto.
        """
        # EXTENSAO: novo formato de caminho a aceitar (ex.: UNC \\servidor\...): acrescente na tupla.
        for recorded in ("E:\\Acadêmico\\PUC\\sistema_ia\\data\\reference\\foto.jpg",
                         "/home/outra/clone/sistema_ia/data/reference/foto.jpg"):
            with self.subTest(recorded=recorded):
                self.assertEqual(resolve_local_path(recorded, self.root), self.target)

    def test_existing_absolute_path_and_unknown_path_are_left_alone(self):
        """Caminho que já existe volta igual; caminho sem correspondente local volta inalterado.

        Quem chama decide o que fazer com um caminho inexistente (a função não inventa).
        """
        self.assertEqual(resolve_local_path(str(self.target), self.root), self.target)
        missing = "E:\\sistema_ia\\data\\nao_existe.jpg"
        self.assertEqual(resolve_local_path(missing, self.root), Path(missing))


# ---------------------------------------------------------------------------
# Fixture do ciclo: projeto mínimo com 8 fotos revisadas
# ---------------------------------------------------------------------------

class LearningLoopFixture(unittest.TestCase):
    """Aprovações abaixo existem só no diretório temporário.

    Classe-base (não tem testes próprios). Monta um projeto mínimo para as classes
    ``PilotDatasetTests`` e ``AiProposalTests``. Atributos criados:
        self.root: Path ``<tmp>/sistema_ia``.
        self.taxonomy_path: taxonomia de 2 classes (0 telha_quebrada, 1 residuos_telha).
        self.config: ``ProjectConfig`` em CPU com ``pilot.duplicate_iou = 0.7``.
        self.records: lista de 8 dicts no formato do registry.json.
        self.registry_path: ``<root>/revisao/registry.json``.
    """

    def setUp(self):
        """Grava taxonomia, config, 8 fotos e o registry.json no diretório temporário."""
        # 1. Pasta temporária. O espaço no prefixo testa caminhos com espaço (comum no Windows).
        temporary = tempfile.TemporaryDirectory(prefix="dronecamp ciclo ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "sistema_ia"
        (self.root / "configs").mkdir(parents=True)
        # 2. Taxonomia mínima com 2 classes de fase 1.
        # EXTENSAO: teste que precise de mais classes: acrescente itens com "id" sequencial aqui.
        self.taxonomy_path = self.root / "configs" / "taxonomy.json"
        self.taxonomy_path.write_text(json.dumps({"version": "fixture", "classes": [
            {"id": 0, "slug": "telha_quebrada", "label": "Telha quebrada", "phase": 1},
            {"id": 1, "slug": "residuos_telha", "label": "Resíduos sobre telhas", "phase": 1},
        ]}), encoding="utf-8")
        # 3. Configuração do projeto montada à mão (sem ler project.yaml).
        # EXTENSAO: parâmetro novo do piloto ou do treino: acrescente no dict pilot=/training= abaixo.
        self.config = ProjectConfig(
            root=self.root, model="fixture.pt", requested_model="fixture.pt", taxonomy_path=self.taxonomy_path,
            dataset_path=self.root / "dataset.yaml", device="cpu", prediction={}, training={},
            pilot={"duplicate_iou": 0.7})
        # 4. Oito fotos PNG 64x48 de cores diferentes (hashes diferentes) e um registro para cada.
        (self.root / "fotos").mkdir()
        self.records = []
        for index in range(8):
            path = self.root / "fotos" / f"foto_{index}.png"
            Image.new("RGB", (64, 48), color=(20 * index, 90, 160)).save(path)
            # Uma caixa por foto; a classe alterna entre 0 e 1 (index % 2).
            boxes = [box(index % 2, [4, 4, 30, 30])]
            # Foto 0 ganha uma sugestão aceita quase igual: é a duplicata que o piloto deve remover.
            if index == 0:
                boxes.append(box(0, [5, 4, 31, 30], ACCEPTED.format(90)))
            # EXTENSAO: campo novo do registro de imagem: acrescente a chave neste dict.
            self.records.append({
                "image_sha256": file_hash(path), "filename": path.name, "width": 64, "height": 48,
                # Caminho gravado em outra máquina: precisa ser reancorado neste clone.
                "source_path": f"E:\\outra maquina\\sistema_ia\\fotos\\{path.name}",
                "status": "positive", "boxes": boxes, "human_approved": True,
                "technical_status": "human_visual_reviewed", "human_reviewer": "fixture",
                "building_group": "fixture", "scene_group": f"cena_{index}", "notes": "", "severity": None,
            })
        # 5. Grava o registry.json com o hash da taxonomia (o código recusa revisão de taxonomia diferente).
        self.registry_path = self.root / "revisao" / "registry.json"
        self.registry_path.parent.mkdir()
        self.write(self.registry_path, {"schema_version": 1, "version": "revisao",
                                        "taxonomy_sha256": file_hash(self.taxonomy_path), "images": self.records})

    @staticmethod
    def write(path: Path, value: dict) -> None:
        """Grava ``value`` como JSON UTF-8 em ``path``, criando as pastas.

        Parâmetros:
            path: arquivo de destino.
            value: dicionário serializável.

        Retorna:
            None.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Dataset piloto e importação da revisão humana
# ---------------------------------------------------------------------------

class PilotDatasetTests(LearningLoopFixture):
    """Montagem do dataset piloto e importação de feedback humano sobre a fixture de 8 fotos."""

    def test_duplicates_are_consolidated_and_dataset_survives_being_moved(self):
        """O piloto remove a duplicata da foto 0 e continua válido depois de movido de pasta.

        Garante que o manifesto conta caixas revisadas (9) × caixas de treino (8) e que o
        dataset usa caminhos relativos (não quebra ao copiar para outra pasta).
        """
        # 1. Monta o dataset piloto a partir do registry da fixture.
        output = self.root / "data" / "pilot" / "v1"
        build_pilot_dataset(self.config, [self.registry_path], output)
        # 2. Confere o manifesto pilot.json: 8 fotos x 1 caixa + 1 duplicata = 9 revisadas, 8 para treino.
        manifest = json.loads((output / "pilot.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["duplicate_policy"]["boxes_reviewed"], 9)
        self.assertEqual(manifest["duplicate_policy"]["boxes_for_training"], 8)
        # 3. A foto 0 ficou com 1 caixa e registra 1 duplicata consolidada.
        first = next(sample for sample in manifest["images"] if sample["image_sha256"] == self.records[0]["image_sha256"])
        self.assertEqual((first["boxes"], first["duplicates_consolidated"]), (1, 1))
        # 4. Move a pasta inteira e valida de novo no novo lugar.
        moved = self.root / "data" / "pilot" / "v1_movido"
        shutil.copytree(output, moved)
        shutil.rmtree(output)
        self.assertTrue(validate_pilot_dataset(self.config, moved / "dataset.yaml")["valid"])

    def test_human_feedback_import_accepts_paths_recorded_on_other_machine(self):
        """A importação do feedback humano funciona mesmo com ``source_path`` de outra máquina.

        A foto 1 é marcada como negativa e completa no feedback; depois da importação
        ela precisa sair aprovada por humano.
        """
        # 1. Volta todas as fotos para "pendente" e regrava o registry.
        for record in self.records:
            record.update(human_approved=False, technical_status="pending_human_review")
        self.write(self.registry_path, {"schema_version": 1, "version": "revisao",
                                        "taxonomy_sha256": file_hash(self.taxonomy_path), "images": self.records})
        # 2. Feedback no formato exportado pela plataforma, amarrado ao hash do registry.
        feedback = self.root / "feedback.json"
        self.write(feedback, {"schema_version": 1, "registry_sha256": file_hash(self.registry_path),
                              "reviewer_id": "fixture", "exported_at": "2026-10-04T00:00:00Z",
                              "images": [{"image_sha256": self.records[1]["image_sha256"], "status": "negative",
                                          "boxes": [], "confirmed_complete": True}]})
        # 3. Importa para uma revisão nova (v2) e confere a aprovação da foto 1.
        result = import_human_feedback(self.config, self.registry_path, feedback, self.root / "v2" / "registry.json")
        self.assertTrue(result["images"][1]["human_approved"])


# ---------------------------------------------------------------------------
# Propostas de IA (apply_ai_proposals)
# ---------------------------------------------------------------------------

class AiProposalTests(LearningLoopFixture):
    """Regras de ``review_data.apply_ai_proposals``.

    ATENCAO: a regra central é "IA nunca aprova": propostas só preenchem a revisão
    como pendente e nunca sobrescrevem uma decisão humana.
    """

    def setUp(self):
        """Deixa a foto 0 aprovada por humano e as fotos 1..7 pendentes, sem caixas."""
        super().setUp()
        # Todas pendentes/ambíguas e sem caixas...
        for record in self.records:
            record.update(human_approved=False, technical_status="pending_human_review", status="ambiguous", boxes=[])
        # ...exceto a foto 0, que fica com decisão humana (não pode ser sobrescrita).
        self.records[0].update(human_approved=True, technical_status="human_visual_reviewed",
                               status="positive", boxes=[box(0, [4, 4, 30, 30])])
        self.write(self.registry_path, {"schema_version": 1, "version": "revisao",
                                        "taxonomy_sha256": file_hash(self.taxonomy_path), "images": self.records})
        # Caminhos do arquivo de propostas (entrada) e da revisão gerada (saída).
        self.proposals = self.root / "propostas.json"
        self.output = self.root / "revisao_ia" / "registry.json"

    def proposal(self, index: int, **changes) -> dict:
        """Monta a proposta padrão para a foto ``index``, com uma caixa que sugere classe nova.

        Parâmetros:
            index: posição da foto em ``self.records`` (0..7).
            **changes: chaves que substituem as do padrão (ex.: ``boxes=[]``, ``status="negative"``).

        Retorna:
            dict de imagem no formato do arquivo de propostas.
        """
        # Proposta padrão: positiva, uma caixa da classe 0 com "proposed_new_class".
        value = {"image_sha256": self.records[index]["image_sha256"], "status": "positive", "notes": "Proposta.",
                 "boxes": [{**box(0, [2, 2, 40, 20], "Trinca."), "proposed_new_class": "telha_trincada"}]}
        # As mudanças pedidas sobrescrevem as chaves do padrão.
        return {**value, **changes}

    def run_proposals(self, *images: dict, annotator: str = "claude_visual") -> dict:
        """Grava ``propostas.json`` com as imagens dadas e roda ``apply_ai_proposals``.

        Parâmetros:
            *images: dicts de imagem (use ``self.proposal``).
            annotator: nome do anotador (IA); vazio/espaços deve ser recusado.

        Retorna:
            O registry resultante (dict) devolvido por ``apply_ai_proposals``.

        Erros:
            ValueError: repassado de ``apply_ai_proposals`` quando a proposta é inválida.
        """
        self.write(self.proposals, {"schema_version": 1, "annotator": annotator, "images": list(images)})
        return apply_ai_proposals(self.config, self.registry_path, self.proposals, self.output)

    def test_proposals_fill_review_without_approval_and_list_new_classes(self):
        """A proposta preenche a foto 1, deixa-a pendente e lista a classe nova sugerida.

        Importa porque só o humano aprova; a IA apenas adianta o trabalho.
        """
        result = self.run_proposals(self.proposal(1))
        item = result["images"][1]
        # Nunca aprovada pela IA.
        self.assertFalse(item["human_approved"])
        self.assertEqual((item["status"], item["annotator"], item["technical_status"]),
                         ("positive", "claude_visual", "pending_human_review"))
        # A classe sugerida fica registrada para o humano decidir se cria na taxonomia.
        self.assertEqual(result["ai_proposals"]["proposed_new_classes"], ["telha_trincada"])

    def test_human_decision_is_never_overwritten(self):
        """Proposta sobre a foto 0 (já aprovada por humano) é recusada e nada é gravado."""
        with self.assertRaisesRegex(ValueError, "decisão humana"):
            self.run_proposals(self.proposal(0))
        # A recusa acontece antes de gravar a saída.
        self.assertFalse(self.output.exists())

    def test_invalid_class_slug_status_or_annotator_are_rejected(self):
        """Slug de classe fora do padrão, "positiva sem caixas" e anotador vazio são recusados."""
        # Slug com maiúscula e espaço: inválido (slugs são minúsculos com "_").
        bad_slug = self.proposal(1, boxes=[{**box(0, [2, 2, 40, 20]), "proposed_new_class": "Telha Trincada"}])
        # Casos: (imagens, anotador). EXTENSAO: nova validação de proposta: acrescente um par nesta tupla.
        for images, annotator in (([bad_slug], "claude_visual"), ([self.proposal(1, boxes=[])], "claude_visual"),
                                  ([self.proposal(1)], " ")):
            with self.subTest(annotator=annotator):
                with self.assertRaises(ValueError):
                    self.run_proposals(*images, annotator=annotator)


# ENTRADA: execução direta do arquivo (python tests/test_learning_loop.py).
if __name__ == "__main__":
    unittest.main()
