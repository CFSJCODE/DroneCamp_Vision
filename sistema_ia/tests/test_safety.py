"""Contratos essenciais: classes COCO, severidade e disponibilidade do modelo.

Módulos cobertos
----------------
- ``dronecamp_ia.backend``: ``check_domain_names`` (recusa pesos cujas classes não são as da
  taxonomia, na mesma ordem) e ``load_detector`` (recusa YOLO27 antes de tentar baixar).
- ``dronecamp_ia.prediction.serialize_detections``: converte o resultado do YOLO em achados;
  a gravidade fica sempre pendente (nunca inferida pela confiança).
- ``dronecamp_ia.training.review_tuning_result``: confere o resultado do ajuste de
  hiperparâmetros (tuner) e grava ``summary.json``.

Como rodar (na pasta ``sistema_ia``)
------------------------------------
- Só este arquivo::

      DRONECAMP_SKIP_E2E=1 python -m unittest tests.test_safety -v

- Um teste só::

      python -m unittest tests.test_safety.SafetyTests.test_candidate_keeps_severity_pending_despite_confidence -v

O que a classe prepara
----------------------
``SafetyTests``: carrega a config e a taxonomia REAIS (só leitura). Os resultados do YOLO são
imitados com ``SimpleNamespace`` + ``Values``; os testes do tuner gravam ``search/`` numa
pasta temporária.

ATENCAO: estes testes protegem regras de segurança do laudo: gravidade nunca inferida pela IA,
modo demonstração (COCO) nunca usa categorias do projeto, e "sem caixas" nunca vira
"conforme". Não afrouxe as asserções sem decisão explícita do responsável técnico.

EXTENSAO: para testar um campo novo em ``serialize_detections``, copie
``test_candidate_keeps_severity_pending_despite_confidence`` e afirme o campo em
``findings[0]``. ``test_candidate_keeps_severity_pending_despite_confidence`` depende de
``telha_quebrada`` ser o ID 0 e de sua gravidade histórica na taxonomia real ser "gravissima".
"""

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import json
import unittest

from dronecamp_ia.backend import check_domain_names, load_detector
from dronecamp_ia.config import load_config, load_taxonomy, detection_names
from dronecamp_ia.prediction import serialize_detections
from dronecamp_ia.training import review_tuning_result


# ---------------------------------------------------------------------------
# Dublê de tensor
# ---------------------------------------------------------------------------

class Values:
    """Imita um tensor da Ultralytics: ``.cpu().tolist()`` devolve a lista guardada."""

    def __init__(self, data):
        """Guarda a lista.

        Parâmetros:
            data: lista Python (caixas xyxy, confianças ou classes).
        """
        self.data = data

    def cpu(self):
        """Retorna o próprio objeto."""
        return self

    def tolist(self):
        """Retorna a lista guardada."""
        return self.data


# ---------------------------------------------------------------------------
# Contratos de segurança
# ---------------------------------------------------------------------------

class SafetyTests(unittest.TestCase):
    """Regras que nunca podem ser quebradas por mudanças de modelo, inferência ou tuner."""

    def setUp(self):
        """Carrega a config real e a taxonomia real (configs/taxonomy.json)."""
        self.config = load_config()
        self.taxonomy = load_taxonomy(self.config.taxonomy_path)

    def test_coco_checkpoint_cannot_claim_roof_categories(self):
        """Pesos COCO (person, car) são recusados: não podem rotular defeitos de telhado."""
        # SimpleNamespace imita o modelo carregado; só ``names`` é lido.
        model = SimpleNamespace(names={0: "person", 1: "car"})
        with self.assertRaisesRegex(ValueError, "Checkpoint não corresponde"):
            check_domain_names(model, detection_names(self.taxonomy))

    def test_same_classes_in_different_order_are_rejected(self):
        """Mesmas classes em outra ordem também são recusadas.

        ATENCAO: a ordem é o ID; aceitar ordem diferente trocaria a classe de todas as caixas.
        """
        names = detection_names(self.taxonomy)
        model = SimpleNamespace(names=dict(enumerate(reversed(names))))
        with self.assertRaises(ValueError):
            check_domain_names(model, names)

    def test_future_model_is_rejected_before_download(self):
        """``load_detector`` recusa YOLO27 com a explicação, sem tentar baixar."""
        with self.assertRaisesRegex(ValueError, "YOLO27 ainda não foi publicado"):
            load_detector(self.config, "yolo27l.pt")

    def test_candidate_keeps_severity_pending_despite_confidence(self):
        """Mesmo com confiança 99,9%, o achado sai com gravidade None e revisão "pendente".

        A gravidade histórica do laudo de referência aparece só como informação
        (``historical_severity_only``).
        """
        # Uma caixa da classe 0 (telha_quebrada) com confiança 0.999.
        result = SimpleNamespace(
            names={0: "telha_quebrada"},
            boxes=SimpleNamespace(xyxy=Values([[10, 20, 100, 200]]), conf=Values([0.999]), cls=Values([0])),
        )
        findings = serialize_detections(result, self.taxonomy, demo=False)
        self.assertIsNone(findings[0]["severity"])
        self.assertEqual(findings[0]["review_status"], "pendente")
        self.assertEqual(findings[0]["report_reference"]["historical_severity_only"], "gravissima")

    def test_demo_never_assigns_project_reference(self):
        """No modo demonstração (COCO), o achado não recebe categoria nem referência do laudo."""
        result = SimpleNamespace(
            names={0: "person"},
            boxes=SimpleNamespace(xyxy=Values([[1, 2, 3, 4]]), conf=Values([0.5]), cls=Values([0])),
        )
        findings = serialize_detections(result, self.taxonomy, demo=True)
        self.assertIsNone(findings[0]["category_label"])
        self.assertIsNone(findings[0]["report_reference"])

    def test_no_boxes_is_empty_evidence_not_conformity(self):
        """Resultado sem caixas vira lista vazia (sem evidência), nunca um achado de "conforme"."""
        result = SimpleNamespace(boxes=None)
        self.assertEqual(serialize_detections(result, self.taxonomy, demo=False), [])

    def test_tuner_all_failed_cannot_report_success(self):
        """Se todas as iterações do tuner falharam, ``review_tuning_result`` lança erro e grava summary incompleto.

        Importa porque o tuner da Ultralytics pode terminar "normalmente" mesmo com tudo falho.
        """
        with TemporaryDirectory() as temporary:
            # 1. Histórico do tuner com 1 iteração cujo dataset falhou.
            run = Path(temporary)
            search = run / "search"
            search.mkdir()
            record = {"fitness": 0, "datasets": {"dataset.yaml": {}}, "failed_datasets": ["dataset.yaml"]}
            (search / "tune_results.ndjson").write_text(json.dumps(record), encoding="utf-8")
            # 2. 2º argumento = iterações esperadas (1).
            with self.assertRaisesRegex(ValueError, "Tuning incompleto"):
                review_tuning_result(run, 1)
            # 3. O summary.json é gravado mesmo assim, marcando a falha.
            summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["successful"], 0)

    def test_zero_fitness_with_valid_metrics_is_successful_execution(self):
        """Fitness 0 com métricas válidas, hiperparâmetros e best.pt é execução completa, com aceite humano pendente."""
        with TemporaryDirectory() as temporary:
            # Monta tudo o que um tuner bem-sucedido deixa em search/.
            run = Path(temporary)
            search = run / "search"
            (search / "weights").mkdir(parents=True)
            record = {"fitness": 0, "datasets": {"dataset.yaml": {"map": 0.0}}}
            (search / "tune_results.ndjson").write_text(json.dumps(record), encoding="utf-8")
            (search / "best_hyperparameters.yaml").write_text("lr0: 0.001", encoding="utf-8")
            (search / "weights/best.pt").write_bytes(b"test_fixture")
            summary = review_tuning_result(run, 1)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["human_acceptance"], "pendente")


# ENTRADA: execução direta do arquivo.
if __name__ == "__main__":
    unittest.main()
