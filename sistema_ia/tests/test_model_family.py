"""Família/versão do YOLO: nomes oficiais, bloqueio do YOLO27 e colunas de perda por família.

Módulos cobertos
----------------
- ``dronecamp_ia.model_family``: ``parse_model_name`` (lê família/tamanho/tarefa do nome do
  peso), ``FAMILIES`` (tabela das famílias conhecidas), ``GENERIC_DISTANCE_LOSS``,
  ``ensure_downloadable`` (bloqueia download de família não publicada), ``family_support``,
  ``describe_checkpoint``, ``distance_loss_columns``, ``model_info`` e ``migration_checklist``.
- ``dronecamp_ia.backend.load_detector``: recusa YOLO27 e checkpoints locais inexistentes.
- ``dronecamp_ia.training._review_training_csv``: auditoria do results.csv usando as
  colunas de perda da família.

Como rodar (na pasta ``sistema_ia``)
------------------------------------
- Só este arquivo::

      DRONECAMP_SKIP_E2E=1 python -m unittest tests.test_model_family -v

- Um teste só::

      python -m unittest tests.test_model_family.ParseModelNameTests.test_yolo27_is_known_but_not_released -v

O que cada classe prepara
-------------------------
- ``ParseModelNameTests`` e ``ProvenanceAndLossTests``: nada no disco (só nomes de arquivo);
  ``test_results_csv_audit_uses_family_columns`` grava um results.csv numa pasta temporária.
- ``DownloadPolicyTests``: carrega a config REAL (``load_config()`` lê configs/project.yaml,
  só leitura). Nenhum teste baixa pesos: os casos válidos param antes da rede.
- ``ModelInfoTests``: também usa a config real. ATENCAO: os testes supõem que o projeto
  está configurado com ``yolo26l`` e que o modelo pedido (``requested_model``) é o
  ``yolo27l``; se o project.yaml mudar, estes testes precisam acompanhar.

EXTENSAO: quando uma família nova for publicada (ex.: YOLO27 liberado ou YOLO28 cadastrada
em ``FAMILIES``), copie ``test_yolo27_is_known_but_not_released`` trocando o nome do peso e
o valor esperado de ``released``; e copie ``test_yolo27_is_rejected_with_explanation`` para
a família que continuar bloqueada.
"""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import importlib.util
import unittest

from dronecamp_ia.backend import load_detector
from dronecamp_ia.config import load_config
from dronecamp_ia.model_family import (FAMILIES, GENERIC_DISTANCE_LOSS, describe_checkpoint, distance_loss_columns,
                                       ensure_downloadable, family_support, migration_checklist, model_info,
                                       parse_model_name)
from dronecamp_ia.training import _review_training_csv

# True se a biblioteca Ultralytics está instalada; sem ela, o teste que consulta a lib é pulado.
ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None


# ---------------------------------------------------------------------------
# parse_model_name: o que o nome do peso diz
# ---------------------------------------------------------------------------

class ParseModelNameTests(unittest.TestCase):
    """Leitura de família, tamanho, tarefa e propriedades a partir do nome do arquivo de pesos."""

    def test_official_yolo26_detection_weight(self):
        """``yolo26l.pt`` é detecção oficial, publicada, end-to-end, com perda de distância ``l1_loss``."""
        spec = parse_model_name("yolo26l.pt")
        self.assertEqual((spec.family, spec.size, spec.task), (26, "l", "detect"))
        self.assertTrue(spec.official and spec.released and spec.end_to_end)
        # Coluna de perda de distância que a auditoria do results.csv procura para a família 26.
        self.assertEqual(spec.distance_loss, ("l1_loss",))
        self.assertEqual(spec.label, "YOLO26L")

    def test_other_tasks_and_case_are_parsed(self):
        """Sufixos de tarefa (-seg, -pose) e maiúsculas no nome são reconhecidos."""
        self.assertEqual(parse_model_name("YOLO26M-seg.pt").task, "seg")
        self.assertEqual(parse_model_name("yolo26x-pose.pt").size, "x")

    def test_yolo27_is_known_but_not_released(self):
        """YOLO27 é oficial mas ainda não publicado; as perdas vêm da tabela ``FAMILIES``."""
        spec = parse_model_name("yolo27l.pt")
        self.assertEqual(spec.family, 27)
        self.assertTrue(spec.official)
        self.assertFalse(spec.released)
        self.assertEqual(spec.distance_loss, tuple(FAMILIES[27]["distance_loss"]))

    def test_unknown_family_and_local_checkpoint(self):
        """Família fora da tabela (yolo11) e checkpoint local (best.pt) recebem valores seguros.

        O checkpoint local não tem família e usa ``GENERIC_DISTANCE_LOSS``.
        """
        unknown = parse_model_name("yolo11n.pt")
        self.assertEqual(unknown.family, 11)
        self.assertTrue(unknown.official)
        self.assertFalse(unknown.end_to_end)
        # Caminho Windows de um run local: não é nome oficial.
        local = parse_model_name(r"runs\pilot_train_x\fit\weights\best.pt")
        self.assertIsNone(local.family)
        self.assertFalse(local.official)
        self.assertEqual(local.distance_loss, GENERIC_DISTANCE_LOSS)


# ---------------------------------------------------------------------------
# Política de download (ensure_downloadable / load_detector)
# ---------------------------------------------------------------------------

class DownloadPolicyTests(unittest.TestCase):
    """O sistema nunca baixa pesos errados em silêncio.

    Importa porque a Ultralytics baixa automaticamente qualquer nome oficial; um caminho
    digitado errado viraria pesos genéricos sem aviso.
    """

    def setUp(self):
        """Carrega a config real do projeto (configs/project.yaml), só leitura."""
        self.config = load_config()

    def test_yolo27_is_rejected_with_explanation(self):
        """YOLO27 é recusado com mensagem explicando que ainda não foi publicado (função e backend)."""
        with self.assertRaisesRegex(ValueError, "YOLO27 ainda não foi publicado"):
            ensure_downloadable(parse_model_name("yolo27l.pt"), Path("yolo27l.pt"))
        with self.assertRaisesRegex(ValueError, "YOLO27 ainda não foi publicado"):
            load_detector(self.config, "yolo27l.pt")

    def test_non_detection_and_unknown_names_are_rejected(self):
        """Peso de segmentação, tamanho inexistente (q) e caminho local ausente são recusados."""
        with self.assertRaisesRegex(ValueError, "tarefa 'seg'"):
            ensure_downloadable(parse_model_name("yolo26l-seg.pt"), Path("yolo26l-seg.pt"))
        with self.assertRaisesRegex(ValueError, "Checkpoint local não encontrado"):
            ensure_downloadable(parse_model_name("yolo26q.pt"), Path("yolo26q.pt"))
        with self.assertRaisesRegex(ValueError, "Checkpoint local não encontrado"):
            load_detector(self.config, "runs/inexistente/best.pt")

    def test_missing_checkpoint_path_with_official_basename_never_downloads(self):
        """Caminho com pasta e nome oficial (ex.: runs/anterior/yolo26l.pt) inexistente é erro, nunca download."""
        # runs/anterior/yolo26l.pt inexistente é erro, não um download silencioso de pesos genéricos.
        # EXTENSAO: outro formato de caminho a proteger: acrescente na tupla.
        for path in ("runs/anterior/yolo26l.pt", r"runs\anterior\yolo26l.pt", "./yolo26l.pt"):
            with self.assertRaisesRegex(ValueError, "Checkpoint local não encontrado"):
                load_detector(self.config, path)

    def test_released_family_passes_without_network(self):
        """Família publicada (yolo26n) passa na checagem sem acessar a rede (não lança)."""
        ensure_downloadable(parse_model_name("yolo26n.pt"), Path("yolo26n.pt"))

    @unittest.skipUnless(ULTRALYTICS, "Ultralytics ausente.")
    def test_installed_library_reports_support_without_downloading(self):
        """``family_support`` consulta a Ultralytics instalada: YOLO26 disponível; YOLO27 coerente com a lista oficial."""
        current = family_support(parse_model_name("yolo26l.pt"))
        self.assertTrue(current["downloadable"] and current["architecture_available"])
        future = family_support(parse_model_name("yolo27l.pt"))
        # Quando a Ultralytics publicar o YOLO27, este teste passa a documentar a mudança.
        self.assertEqual(future["downloadable"], future["official_weights_listed"])
        if not future["downloadable"]:
            self.assertIn("ainda não foi publicado", future["reason"])


# ---------------------------------------------------------------------------
# Proveniência de checkpoint e colunas de perda
# ---------------------------------------------------------------------------

class ProvenanceAndLossTests(unittest.TestCase):
    """Família herdada por pesos treinados e colunas de perda usadas na auditoria do treino."""

    def test_trained_checkpoint_inherits_family_from_base_weights(self):
        """Um best.pt treinado herda a família dos pesos base informados (train_args.model)."""
        info = describe_checkpoint(Path("runs/pilot_train_x/fit/weights/best.pt"), {"model": "yolo26l.pt"})
        self.assertEqual((info["family"], info["label"], info["base_weights"]), (26, "YOLO26L", "yolo26l.pt"))
        # Sem argumentos de treino, a família fica desconhecida.
        self.assertIsNone(describe_checkpoint(Path("best.pt"))["family"])

    def test_distance_loss_follows_family(self):
        """``distance_loss_columns`` usa ``ckpt.train_args.model``; sem ckpt, cai no genérico."""
        # SimpleNamespace imita o objeto do modelo carregado com os atributos lidos pela função.
        model = SimpleNamespace(ckpt={"train_args": {"model": "yolo26l.pt"}}, ckpt_path="runs/x/fit/weights/best.pt")
        self.assertEqual(distance_loss_columns(model), ("l1_loss",))
        self.assertEqual(distance_loss_columns(SimpleNamespace()), GENERIC_DISTANCE_LOSS)

    def test_results_csv_audit_uses_family_columns(self):
        """Um results.csv com ``dfl_loss`` passa no genérico, mas acusa falta de ``l1_loss`` para YOLO26."""
        # Cabeçalho no formato da Ultralytics (colunas de perda dfl, sem l1).
        header = ("epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),"
                  "metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss\n")
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.csv"
            path.write_text(header + "1,1,1,1,0.5,0.5,0.5,0.2,1,1,1\n", encoding="utf-8")
            # Retorno: (lista de motivos de problema, outro valor ignorado aqui).
            generic, _ = _review_training_csv(path, GENERIC_DISTANCE_LOSS)
            self.assertEqual(generic, [])
            strict, _ = _review_training_csv(path, ("l1_loss",))
            self.assertTrue(any("l1_loss" in reason for reason in strict))


# ---------------------------------------------------------------------------
# model_info e migration_checklist (comando de informação do modelo)
# ---------------------------------------------------------------------------

class ModelInfoTests(unittest.TestCase):
    """Relatório de modelo e roteiro de migração de família (``model_info``/``migration_checklist``)."""

    def test_report_lists_configured_requested_and_migration(self):
        """O relatório traz o modelo configurado (26), o pedido e os passos de migração (com compare-models)."""
        config = load_config()
        report = model_info(config)
        self.assertEqual(report["configured"]["family"], 26)
        self.assertEqual(report["requested"]["name"], config.requested_model)
        self.assertIn("migration", report)
        self.assertEqual(report["migration"][0]["step"], "pesos publicados")
        self.assertIn("compare-models", " ".join(step["how"] for step in report["migration"]))

    def test_same_model_has_no_migration_and_custom_model_is_accepted(self):
        """Pedido igual ao configurado não gera migração; inspecionar outro peso gera ``migration_to_inspected``."""
        config = replace(load_config(), requested_model="yolo26l.pt")
        self.assertNotIn("migration", model_info(config))
        report = model_info(config, "yolo26n.pt")
        self.assertEqual((report["configured"]["size"], report["inspected"]["size"]), ("l", "n"))
        self.assertIn("migration_to_inspected", report)

    def test_inspecting_the_target_keeps_the_configured_model_and_the_migration(self):
        """Inspecionar o próprio alvo (yolo27l) mantém ``migration`` e não duplica ``migration_to_inspected``."""
        config = load_config()  # configurado yolo26l, alvo yolo27l
        report = model_info(config, "yolo27l.pt")
        self.assertEqual((report["configured"]["family"], report["inspected"]["family"]), (26, 27))
        self.assertIn("migration", report)
        self.assertNotIn("migration_to_inspected", report)

    def test_checklist_requires_gate_before_config_change(self):
        """A comparação por classe vem antes de mudar a configuração; preservar o modelo anterior já conta como feito."""
        steps = migration_checklist(parse_model_name("yolo26l.pt"), parse_model_name("yolo27l.pt"))
        names = [step["step"] for step in steps]
        self.assertLess(names.index("comparação por classe com o modelo atual"), names.index("configuração"))
        self.assertTrue(next(step for step in steps if step["step"] == "modelo anterior preservado")["done"])


# ENTRADA: execução direta do arquivo.
if __name__ == "__main__":
    unittest.main()
