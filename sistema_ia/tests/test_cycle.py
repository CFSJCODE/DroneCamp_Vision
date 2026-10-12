"""retrain-cycle (cycle.py): ordem das etapas e cycle.json, com cada etapa substituída por uma função falsa.

Módulo coberto
--------------
``dronecamp_ia.cycle.run_cycle`` (comando ``retrain-cycle`` da CLI): encadeia
dataset piloto → treino → exportação ONNX → portão de comparação contra cada modelo-base →
medição → relatório, e grava ``runs/cycle_*/cycle.json`` com o estado de cada etapa.

Como rodar (na pasta ``sistema_ia``)
------------------------------------
- Só este arquivo::

      DRONECAMP_SKIP_E2E=1 python -m unittest tests.test_cycle -v

- Um teste só::

      python -m unittest tests.test_cycle.RetrainCycleTests.test_failure_is_recorded_and_raised -v

O que a classe prepara
----------------------
``RetrainCycleTests``: a config real (``REAL``) com a raiz trocada por uma pasta temporária
(``self.config``), e uma lista ``self.calls`` onde cada etapa falsa anota (nome, argumentos
principais). ``_fakes`` cria as seis etapas falsas; ``_run`` aplica os ``mock.patch`` e chama
``run_cycle``. Nada é treinado ou exportado de verdade.

ATENCAO: os ``mock.patch`` trocam as funções nos módulos de origem
(``dronecamp_ia.pilot.build_pilot_dataset`` etc.). Isso só funciona porque ``cycle.py``
importa as etapas DENTRO de ``run_cycle``; se elas passarem a ser importadas no topo de
cycle.py, estes testes deixam de interceptar as etapas e tentam treinar de verdade.

EXTENSAO: para testar uma etapa nova do ciclo (ex.: "publicar pesos"):
1. em ``_fakes``, crie a função falsa com a MESMA assinatura da real, anotando em
   ``self.calls`` e gravando o mínimo que ``run_cycle`` lê;
2. acrescente-a no dict retornado por ``_fakes``;
3. em ``_run``, acrescente o ``mock.patch("dronecamp_ia.<modulo>.<funcao>", ...)``;
4. acrescente o nome na lista de ordem esperada em ``test_steps_run_in_order_and_cycle_json_records_each``.
"""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from dronecamp_ia import cycle as cycle_module
from dronecamp_ia.config import load_config
from dronecamp_ia.io import write_json

# Config real do projeto (configs/project.yaml), só leitura; cada teste troca a raiz por uma pasta temporária.
REAL = load_config()


# ---------------------------------------------------------------------------
# run_cycle com etapas falsas
# ---------------------------------------------------------------------------

class RetrainCycleTests(unittest.TestCase):
    """Ordem das etapas, argumentos repassados e registro em ``cycle.json`` (sucesso e falha)."""

    def setUp(self):
        """Cria a pasta temporária, a config apontando para ela e a lista de chamadas."""
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        # Mesma config real, mas com root temporário: runs/ e data/ ficam dentro dele.
        self.config = replace(REAL, root=self.root)
        # Cada etapa falsa acrescenta uma tupla (nome, ...) aqui, na ordem em que foi chamada.
        self.calls = []

    def tearDown(self):
        """Apaga a pasta temporária."""
        self.temporary.cleanup()

    def _fakes(self, fail_at=None):
        """Cria as seis etapas falsas do ciclo.

        Parâmetros:
            fail_at: nome da etapa que deve lançar erro; hoje só ``"train"`` é usado.

        Retorna:
            dict {nome da função real: função falsa}.
        """
        root = self.root

        # --- Etapa 1: dataset piloto. Grava um dataset.yaml mínimo e devolve o caminho.
        def build(config, registries, output, seed, sequence_block=0):
            """Falso de ``pilot.build_pilot_dataset``; anota ("dataset", sequence_block)."""
            self.calls.append(("dataset", sequence_block))
            Path(output).mkdir(parents=True)
            (Path(output) / "dataset.yaml").write_text("path: .\n", encoding="utf-8")
            return {"dataset": str(Path(output) / "dataset.yaml"), "counts": {}}

        # --- Etapa 2: treino. Cria best.pt e execution.json (o ciclo lê tiling e gpu_eval dele).
        def train(config, data_path, weights, overrides, gpu_eval):
            """Falso de ``training.train_pilot``; anota ("train", overrides, gpu_eval).

            Erros:
                RuntimeError("treino falhou") quando ``fail_at == "train"`` (depois de gravar o run).
            """
            self.calls.append(("train", dict(overrides or {}), gpu_eval))
            run = root / "runs" / "pilot_train_x"
            (run / "fit" / "weights").mkdir(parents=True)
            (run / "fit" / "weights" / "best.pt").write_bytes(b"x")
            write_json(run / "execution.json", {"tiling": {"patch": 1280}, "gpu_eval": {"state": "finished"}})
            if fail_at == "train":
                raise RuntimeError("treino falhou")
            return run

        # --- Etapa 3: exportação ONNX. Só cria a pasta do run.
        def export(config, weights, parity_images=None):
            """Falso de ``exporting.export_onnx``; anota ("export", weights)."""
            self.calls.append(("export", weights))
            run = root / "runs" / "export_x"
            run.mkdir(parents=True)
            return run

        # --- Etapa 4: portão de comparação, uma vez por modelo-base. Só "b.pt" resulta em adotar.
        def gate(config, data_path, baseline, candidate, splits, onnx_provider="directml"):
            """Falso de ``model_gate.compare_models``; anota ("gate", baseline, onnx_provider).

            Grava ``gate.json`` com ``adopt`` = True apenas quando o baseline é "b.pt".
            """
            self.calls.append(("gate", baseline, onnx_provider))
            # len(self.calls) torna o nome da pasta único para cada portão.
            run = root / "runs" / f"gate_{len(self.calls)}"
            run.mkdir(parents=True)
            write_json(run / "gate.json", {"adopt": baseline == "b.pt", "reasons": [], "images_evaluated": 3})
            return run

        # --- Etapa 5: medição dos modelos (inclui o candidato best.pt no fim da lista).
        def measure(config, data_path, weights, splits, conf, onnx_provider):
            """Falso de ``model_gate.measure_models``; anota ("measure", lista de pesos)."""
            self.calls.append(("measure", list(weights)))
            run = root / "runs" / "measure_x"
            run.mkdir(parents=True)
            write_json(run / "measurement.json", {"models": []})
            return run

        # --- Etapa 6: relatório final, recebe os gate.json e as medições.
        def report(config, output, gates, compares, evals, runs, title, measurements=()):
            """Falso de ``model_report.build_model_report``; anota ("report", gates, measurements)."""
            self.calls.append(("report", [str(path) for path in gates], [str(path) for path in measurements]))
            run = root / "runs" / "report_x"
            run.mkdir(parents=True)
            return run

        # EXTENSAO: etapa nova do ciclo: acrescente o par "nome_real": falso aqui (e o patch em _run).
        return {"build_pilot_dataset": build, "train_pilot": train, "export_onnx": export,
                "compare_models": gate, "measure_models": measure, "build_model_report": report}

    def _run(self, fail_at=None, **kwargs):
        """Roda ``run_cycle`` com as seis etapas trocadas pelas falsas.

        Parâmetros:
            fail_at: repassado a ``_fakes`` (ex.: "train").
            **kwargs: argumentos opcionais de ``run_cycle`` (baselines, overrides, sequence_block,
                gpu_eval, onnx_provider...).

        Retorna:
            Path da pasta ``runs/cycle_*`` criada por ``run_cycle``.
        """
        fakes = self._fakes(fail_at)
        # Cada patch troca a função no módulo de origem enquanto o with estiver ativo.
        with mock.patch("dronecamp_ia.pilot.build_pilot_dataset", fakes["build_pilot_dataset"]), \
             mock.patch("dronecamp_ia.training.train_pilot", fakes["train_pilot"]), \
             mock.patch("dronecamp_ia.exporting.export_onnx", fakes["export_onnx"]), \
             mock.patch("dronecamp_ia.model_gate.compare_models", fakes["compare_models"]), \
             mock.patch("dronecamp_ia.model_gate.measure_models", fakes["measure_models"]), \
             mock.patch("dronecamp_ia.model_report.build_model_report", fakes["build_model_report"]):
            return cycle_module.run_cycle(self.config, [Path("r.json")], self.root / "data" / "pilot" / "v", **kwargs)

    def test_steps_run_in_order_and_cycle_json_records_each(self):
        """As etapas rodam na ordem certa, recebem os argumentos repassados e ``cycle.json`` registra tudo.

        Com dois modelos-base, o portão roda duas vezes; como "b.pt" resulta em adotar,
        ``adopt_any`` fica True.
        """
        cycle = self._run(baselines=["a.pt", "b.pt"], overrides={"tile": True, "epochs": 2}, sequence_block=10,
                          gpu_eval={"enabled": True}, onnx_provider="cpu")
        record = json.loads((cycle / "cycle.json").read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "completed")
        # Ordem esperada. EXTENSAO: etapa nova: acrescente o nome na posição em que ela roda.
        self.assertEqual([call[0] for call in self.calls], ["dataset", "train", "export", "gate", "gate", "measure", "report"])
        # Índices de self.calls: 0 dataset, 1 train, 3 primeiro gate, 5 measure, 6 report.
        self.assertEqual(self.calls[0][1], 10)
        self.assertEqual(self.calls[1][1], {"tile": True, "epochs": 2})
        self.assertEqual(self.calls[3][2], "cpu")
        # O candidato (best.pt do treino) é o último da lista medida.
        self.assertTrue(self.calls[5][1][-1].endswith("best.pt"))
        self.assertEqual(len(self.calls[6][1]), 2)  # dois gate.json no relatório
        self.assertEqual(record["steps"]["gates"][1]["adopt"], True)
        self.assertTrue(record["adopt_any"])
        # O ciclo copia o resumo do fatiamento do execution.json do treino.
        self.assertEqual(record["steps"]["training"]["tiling"], {"patch": 1280})

    def test_failure_is_recorded_and_raised(self):
        """Uma etapa que falha marca ``state = failed`` com a mensagem em ``cycle.json`` e relança o erro.

        As etapas seguintes (export em diante) não rodam nem aparecem no registro.
        """
        with self.assertRaises(RuntimeError):
            self._run(fail_at="train")
        # A pasta do ciclo é localizada pelo prefixo (o sufixo é data/hora).
        cycle = next((self.root / "runs").glob("cycle_*"))
        record = json.loads((cycle / "cycle.json").read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "failed")
        self.assertIn("treino falhou", record["error"])
        self.assertNotIn("export", record["steps"])


# ENTRADA: execução direta do arquivo.
if __name__ == "__main__":
    unittest.main()
