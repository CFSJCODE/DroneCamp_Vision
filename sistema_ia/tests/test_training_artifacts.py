"""Treino só conclui com artefatos e métricas; fixtures não são checkpoints funcionais.

Módulo coberto: ``dronecamp_ia.training``, em especial ``review_training_result`` (a
"guarda de artefatos" que lê ``fit/results.csv`` e ``fit/weights/{best,last}.pt`` e grava
``summary.json``) e ``train`` (que chama essa guarda depois do ``model.train`` da Ultralytics).
Também usa ``training.DETECTION_METRIC_COLUMNS`` (colunas de métricas obrigatórias).

Como rodar (na pasta ``sistema_ia``, com o venv do projeto):
    python -m unittest tests.test_training_artifacts
    python -m unittest tests.test_training_artifacts.TrainingArtifactTests.test_header_without_epochs_is_incomplete

O que a classe de teste prepara (``TrainingArtifactTests.setUp``):
    - pasta temporária com uma subpasta ``run`` (a "execução" de treino);
    - a configuração REAL do projeto (``load_config``) com ``root`` apontado para a pasta
      temporária e ``epochs`` = 100;
    - os nomes reais das classes da taxonomia e um "modelo" falso (``SimpleNamespace``) cujo
      ``names`` bate com eles;
    - um ``results.csv`` com 1 época (todas as métricas 0.2) e ``best.pt``/``last.pt`` com
      bytes opacos (NÃO são modelos de verdade; só existência e tamanho são conferidos).
    ``mock_training`` devolve os patches que trocam diretório de execução, preparo do
    dataset, carregamento do detector e informações de ambiente, para rodar ``train`` sem YOLO.

ATENCAO: ``human_acceptance`` precisa continuar "pendente" em todo caso: métricas boas
NUNCA significam aceitação de qualidade; isso é decisão humana.

EXTENSAO: para testar uma nova regra da guarda de artefatos, copie
``test_header_without_epochs_is_incomplete``: ajuste ``self.columns``/``self.rows`` (ou
apague/esvazie um arquivo em ``self.run / "fit"``), chame ``self.write_csv()`` e então
``self.assert_incomplete("trecho da mensagem")``. Para testar o fluxo de ``train``, copie
``test_class_contract_is_rechecked_after_train_returns`` e escreva a ação que o
``model.train`` falso executa. Se a Ultralytics passar a gravar uma coluna nova obrigatória,
acrescente-a em ``self.columns`` no ``setUp``.
"""

import csv
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dronecamp_ia.config import detection_names, load_config, load_taxonomy
from dronecamp_ia import training


class TrainingArtifactTests(unittest.TestCase):
    """Testes da guarda que decide se um treino terminou com artefatos e métricas válidos.

    Importa porque a Ultralytics pode "terminar" sem pesos ou com métricas NaN; sem essa
    guarda o sistema registraria um treino quebrado como concluído.
    """
    def setUp(self):
        """Cria a execução temporária, a configuração, o modelo falso e artefatos válidos de 1 época.

        Atributos criados: ``run``, ``config``, ``names``, ``model``, ``columns``, ``rows``.
        Retorna: None.
        """
        # 1. Pasta temporária com a subpasta "run" que faz papel de execução de treino.
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name) / "run"
        self.run.mkdir()
        # 2. Configuração real do projeto, mas com root temporário e epochs = 100.
        config = load_config()
        self.config = replace(config, root=Path(self.temporary.name), training={**config.training, "epochs": 100})
        # 3. Classes reais da taxonomia e um modelo falso que só tem o atributo names.
        self.names = detection_names(load_taxonomy(config.taxonomy_path))
        self.model = SimpleNamespace(names=dict(enumerate(self.names)))
        # 4. Colunas do results.csv no formato da Ultralytics (perdas + métricas de detecção).
        # EXTENSAO: nova coluna que a Ultralytics passe a gravar: acrescente aqui.
        self.columns = [
            "epoch", "time", "train/box_loss", "train/cls_loss", "train/l1_loss",
            *training.DETECTION_METRIC_COLUMNS, "val/box_loss", "val/cls_loss", "val/l1_loss",
        ]
        # 5. Estado inicial: 1 época válida e checkpoints presentes.
        self.rows = [self.row(1)]
        self.write_artifacts()

    def row(self, epoch):
        """Monta uma linha válida do results.csv para a época ``epoch``.

        Parâmetros:
            epoch: número da época (int; os testes também passam valores inválidos).
        Retorna: dict coluna -> valor, com todas as métricas/perdas 0.2 e ``time`` 10.0.
        """
        # Todas as colunas valem 0.2 (dentro de 0..1); epoch e time são sobrescritos.
        values = {column: 0.2 for column in self.columns}
        values.update(epoch=epoch, time=10.0)
        return values

    def write_csv(self):
        """Grava ``self.rows`` em ``run/fit/results.csv`` com as colunas de ``self.columns``.

        Retorna: None.
        """
        path = self.run / "fit/results.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.rows)

    def write_artifacts(self):
        """Cria ``best.pt`` e ``last.pt`` falsos (bytes opacos) e regrava o CSV.

        Retorna: None.
        """
        weights = self.run / "fit/weights"
        weights.mkdir(parents=True, exist_ok=True)
        # O teste confere existência e tamanho, não desserializa nem aprova estes bytes.
        for name in ("best.pt", "last.pt"):
            (weights / name).write_bytes(b"opaque_test_fixture_not_a_model")
        self.write_csv()

    def review(self):
        """Roda ``training.review_training_result`` sobre a execução temporária.

        Retorna: o dict do resumo (``complete``, ``epochs_recorded``, ``last_metrics``...).
        Erros: ValueError quando o treino está incompleto.
        """
        return training.review_training_result(self.run, self.model, self.names)

    def summary(self):
        """Lê o ``summary.json`` gravado pela guarda.

        Retorna: dict do resumo salvo em disco.
        """
        return json.loads((self.run / "summary.json").read_text(encoding="utf-8"))

    def assert_incomplete(self, reason):
        """Exige ValueError com ``reason`` e um ``summary.json`` marcado como falho.

        Parâmetros:
            reason: regex da mensagem esperada (pode ter alternativas com ``|``).
        Retorna: None. Também confere ``state == "failed"``, ``reasons`` não vazio e
        ``human_acceptance == "pendente"``.
        """
        with self.assertRaisesRegex(ValueError, reason):
            self.review()
        result = self.summary()
        self.assertFalse(result["complete"])
        self.assertEqual(result["state"], "failed")
        self.assertTrue(result["reasons"])
        self.assertEqual(result["human_acceptance"], "pendente")

    def test_valid_early_stopping_does_not_require_requested_hundred_epochs(self):
        """Parar cedo (3 de 100 épocas) é treino completo; o resumo não promete mais do que verificou.

        ``test_evaluated``, ``checkpoint_functional_validation`` e ``export_parity_verified``
        ficam falsos porque a guarda não faz essas verificações.
        """
        self.rows = [self.row(epoch) for epoch in range(1, 4)]
        self.write_csv()
        result = self.review()
        self.assertTrue(result["complete"])
        self.assertEqual(result["epochs_recorded"], 3)
        self.assertEqual(result["last_epoch"], 3)
        self.assertTrue(result["model_class_contract_verified"])
        self.assertEqual(result["human_acceptance"], "pendente")
        self.assertFalse(result["test_evaluated"])
        self.assertFalse(result["checkpoint_functional_validation"])
        self.assertFalse(result["export_parity_verified"])

    def test_zero_metrics_are_a_valid_execution_not_quality_acceptance(self):
        """Métricas zero são execução válida, mas não aceitação de qualidade."""
        for column in training.DETECTION_METRIC_COLUMNS:
            self.rows[0][column] = 0.0
        self.write_csv()
        result = self.review()
        self.assertTrue(result["complete"])
        self.assertEqual(set(result["last_metrics"].values()), {0.0})
        self.assertEqual(result["human_acceptance"], "pendente")

    def test_detection_head_using_dfl_loss_is_also_supported(self):
        """Cabeças que gravam ``dfl_loss`` no lugar de ``l1_loss`` também são aceitas."""
        self.columns = [name.replace("l1_loss", "dfl_loss") for name in self.columns]
        self.rows = [{name.replace("l1_loss", "dfl_loss"): value for name, value in self.rows[0].items()}]
        self.write_csv()
        self.assertTrue(self.review()["complete"])

    def test_csv_header_whitespace_and_utf8_bom_are_accepted(self):
        """Cabeçalho com espaços em volta dos nomes e BOM UTF-8 é aceito (formato real da Ultralytics)."""
        path = self.run / "fit/results.csv"
        lines = path.read_text(encoding="utf-8").splitlines()
        # Põe espaços em volta de cada nome de coluna e grava com BOM (utf-8-sig).
        lines[0] = ",".join(f" {name} " for name in self.columns)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
        self.assertTrue(self.review()["complete"])

    def test_missing_best_last_or_csv_is_incomplete(self):
        """Faltando best.pt, last.pt ou results.csv, o treino é incompleto ("ausente")."""
        for relative in ("weights/best.pt", "weights/last.pt", "results.csv"):
            with self.subTest(relative=relative):
                self.write_artifacts()
                (self.run / "fit" / relative).unlink()
                self.assert_incomplete("ausente")

    def test_empty_best_last_or_csv_is_incomplete(self):
        """Arquivo de artefato vazio (0 bytes) torna o treino incompleto ("vazio")."""
        for relative in ("weights/best.pt", "weights/last.pt", "results.csv"):
            with self.subTest(relative=relative):
                self.write_artifacts()
                (self.run / "fit" / relative).write_bytes(b"")
                self.assert_incomplete("vazio")

    def test_checkpoint_directory_cannot_substitute_for_file(self):
        """Uma PASTA chamada best.pt não vale como checkpoint ("não é um arquivo")."""
        path = self.run / "fit/weights/best.pt"
        path.unlink()
        path.mkdir()
        self.assert_incomplete("não é um arquivo")

    def test_header_without_epochs_is_incomplete(self):
        """CSV só com cabeçalho (nenhuma época) é incompleto."""
        self.rows = []
        self.write_csv()
        self.assert_incomplete("nenhuma época")

    def test_missing_metric_or_base_loss_column_is_incomplete(self):
        """Sem qualquer coluna obrigatória (epoch, métricas, perdas base), o treino é incompleto.

        EXTENSAO: nova coluna obrigatória: acrescente-a na tupla do laço.
        """
        # Remove uma coluna por vez, do cabeçalho e da linha, partindo sempre do original.
        original_columns, original_row = list(self.columns), dict(self.rows[0])
        for column in ("epoch", *training.DETECTION_METRIC_COLUMNS, "train/box_loss", "val/cls_loss"):
            with self.subTest(column=column):
                self.columns = [name for name in original_columns if name != column]
                self.rows = [{name: value for name, value in original_row.items() if name != column}]
                self.write_csv()
                self.assert_incomplete("colunas obrigatórias")

    def test_unpaired_distance_loss_is_incomplete(self):
        """Perda de distância de treino sem a de validação correspondente é incompleta."""
        self.columns.remove("val/l1_loss")
        self.rows[0].pop("val/l1_loss")
        self.write_csv()
        self.assert_incomplete("distância correspondentes")

    def test_all_epochs_metrics_must_be_finite_and_in_range(self):
        """Métrica NaN/inf ou fora de 0..1 em QUALQUER época (não só na última) reprova o treino."""
        for column in training.DETECTION_METRIC_COLUMNS:
            for invalid in ("nan", "inf", "-inf", -0.01, 1.01):
                with self.subTest(column=column, invalid=invalid):
                    self.rows = [self.row(1), self.row(2)]
                    # Uma última época válida não pode esconder uma época inválida anterior.
                    self.rows[0][column] = invalid
                    self.write_csv()
                    self.assert_incomplete("não é finito|entre 0 e 1")

    def test_every_observed_training_and_validation_loss_must_be_finite(self):
        """Toda coluna ``*_loss`` presente, inclusive extras, precisa ser número finito."""
        # Colunas extras de perda também precisam ser finitas.
        self.columns.extend(["train/extra_loss", "val/extra_loss"])
        for column in [name for name in self.columns if name.endswith("_loss")]:
            for invalid in ("nan", "inf", "", "not_numeric"):
                with self.subTest(column=column, invalid=invalid):
                    self.rows = [self.row(1)]
                    self.rows[0][column] = invalid
                    self.write_csv()
                    self.assert_incomplete("não é finito|não é um número")

    def test_epoch_must_be_positive_finite_integer_and_increasing(self):
        """``epoch`` precisa ser inteiro positivo, finito e estritamente crescente."""
        for epochs in ((0,), (1.5,), ("nan",), (1, 1), (2, 1)):
            with self.subTest(epochs=epochs):
                self.rows = [self.row(epoch) for epoch in epochs]
                self.write_csv()
                self.assert_incomplete("época deve ser|epoch não é finito")

    def test_duplicate_csv_columns_are_incomplete(self):
        """Coluna repetida no cabeçalho (mesmo com espaços) é rejeitada."""
        path = self.run / "fit/results.csv"
        content = path.read_text(encoding="utf-8").splitlines()
        # Duplica uma coluna de métrica (com espaços) e acrescenta o valor correspondente.
        content[0] += ", metrics/precision(B) "
        content[1] += ",0.1"
        path.write_text("\n".join(content) + "\n", encoding="utf-8")
        self.assert_incomplete("colunas duplicadas")

    def test_missing_or_extra_csv_values_are_incomplete(self):
        """Linha com valor a menos ou a mais que o cabeçalho é rejeitada."""
        self.write_csv()
        path = self.run / "fit/results.csv"
        original = path.read_text(encoding="utf-8").splitlines()
        # Primeiro caso: tira o último valor; segundo: acrescenta um valor sobrando.
        for changed in (original[1].rsplit(",", 1)[0], original[1] + ",0.3"):
            with self.subTest(changed=changed):
                path.write_text(original[0] + "\n" + changed + "\n", encoding="utf-8")
                self.assert_incomplete("não é um número|além das colunas")

    def test_wrong_class_order_after_training_is_incomplete(self):
        """Modelo treinado com ordem de classes diferente da taxonomia é rejeitado.

        ATENCAO: IDs de classe são contrato entre taxonomia, labels e pesos.
        """
        self.model.names = dict(enumerate(reversed(self.names)))
        self.assert_incomplete("Contrato das classes")
        self.assertFalse(self.summary()["model_class_contract_verified"])
        self.assertTrue(all(item["present"] for item in self.summary()["artifacts"].values()))

    def mock_training(self, action):
        """Prepara os 4 patches que permitem rodar ``training.train`` sem YOLO nem dataset real.

        Parâmetros:
            action: função chamada no lugar de ``model.train(**kwargs)`` (pode gravar
                artefatos, mudar ``names`` ou lançar exceção).
        Retorna: tupla (patch do diretório de execução, patch de prepare_dataset, patch de
        load_detector, patch de runtime_info), para usar em ``with``.
        """
        # model.train vira um Mock que executa "action"; os patches abaixo isolam o treino.
        # EXTENSAO: se train() passar a chamar outra função pesada, acrescente o patch dela aqui
        # e desempacote o novo item nos testes que usam mock_training.
        self.model.train = Mock(side_effect=action)
        return (
            patch("dronecamp_ia.training.make_run_directory", return_value=self.run),
            patch("dronecamp_ia.training.prepare_dataset", return_value=(self.run / "dataset_resolved.yaml", {})),
            patch("dronecamp_ia.training.load_detector", return_value=self.model),
            patch("dronecamp_ia.training.runtime_info", return_value={"fixture": True}),
        )

    def test_training_or_callback_exception_records_incomplete_summary(self):
        """Exceção durante o treino (ou num callback) vira ValueError e resumo "failed"."""
        def failing_callback(**kwargs):
            """Simula falha dentro do treino.

            Parâmetros: kwargs (ignorados). Erros: sempre RuntimeError.
            """
            raise RuntimeError("fixture callback failed")

        run_patch, data_patch, detector_patch, runtime_patch = self.mock_training(failing_callback)
        with run_patch, data_patch, detector_patch, runtime_patch:
            with self.assertRaisesRegex(ValueError, "fixture callback failed"):
                training.train(self.config, self.run / "dataset.yaml")
        summary = self.summary()
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["state"], "failed")
        self.assertIn("RuntimeError", summary["reasons"][0])

    def test_normal_train_return_without_artifacts_cannot_report_success(self):
        """``model.train`` que retorna normalmente sem gerar artefatos não pode ser sucesso."""
        for path in (self.run / "fit").rglob("*"):
            if path.is_file():
                path.unlink()
        run_patch, data_patch, detector_patch, runtime_patch = self.mock_training(lambda **kwargs: None)
        with run_patch, data_patch, detector_patch, runtime_patch:
            with self.assertRaisesRegex(ValueError, "Treino incompleto"):
                training.train(self.config, self.run / "dataset.yaml")
        self.assertFalse(self.summary()["complete"])

    def test_valid_train_return_runs_artifact_guard_and_preserves_parameters(self):
        """Treino válido passa pela guarda e repassa os parâmetros certos à Ultralytics.

        Confere epochs do config, device, ``exist_ok=False`` (nunca sobrescrever) e ``project``.
        """
        self.rows = [self.row(1), self.row(2)]
        run_patch, data_patch, detector_patch, runtime_patch = self.mock_training(lambda **kwargs: self.write_artifacts())
        with run_patch, data_patch, detector_patch, runtime_patch:
            result = training.train(self.config, self.run / "dataset.yaml")
        self.assertEqual(result, self.run)
        self.assertTrue(self.summary()["complete"])
        self.assertEqual(self.summary()["epochs_recorded"], 2)
        # Argumentos que train() repassou à Ultralytics.
        passed = self.model.train.call_args.kwargs
        self.assertEqual(passed["epochs"], 100)
        self.assertEqual(passed["device"], self.config.device)
        self.assertFalse(passed["exist_ok"])
        self.assertEqual(passed["project"], str(self.run))

    def test_class_contract_is_rechecked_after_train_returns(self):
        """O contrato de classes é conferido de novo DEPOIS do treino retornar."""
        def return_with_wrong_names(**kwargs):
            """Grava artefatos válidos, mas troca as classes do modelo por ``{0: "person"}``.

            Parâmetros: kwargs (ignorados). Retorna: None.
            """
            self.write_artifacts()
            self.model.names = {0: "person"}

        run_patch, data_patch, detector_patch, runtime_patch = self.mock_training(return_with_wrong_names)
        with run_patch, data_patch, detector_patch, runtime_patch:
            with self.assertRaisesRegex(ValueError, "Contrato das classes"):
                training.train(self.config, self.run / "dataset.yaml")
        self.assertFalse(self.summary()["complete"])

    def test_preparation_failure_records_summary_before_loading_model(self):
        """Falha no preparo do dataset grava resumo falho e nunca carrega o modelo."""
        with patch("dronecamp_ia.training.make_run_directory", return_value=self.run), \
             patch("dronecamp_ia.training.prepare_dataset", side_effect=ValueError("fixture provenance rejected")), \
             patch("dronecamp_ia.training.load_detector") as detector:
            with self.assertRaisesRegex(ValueError, "fixture provenance rejected"):
                training.train(self.config, self.run / "dataset.yaml")
        detector.assert_not_called()
        self.assertFalse(self.summary()["complete"])


if __name__ == "__main__":
    unittest.main()
