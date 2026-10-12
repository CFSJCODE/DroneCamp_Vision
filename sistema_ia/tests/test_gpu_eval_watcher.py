"""Vigia de avaliação na GPU: acha os pesos da área de gravação e acompanha o treino pelos arquivos do run.

Módulo coberto
--------------
``scripts/gpu_eval_watcher.py`` (script avulso, não faz parte do pacote; importado como
``watcher``): ``weights_directory`` (onde estão os pesos do run, seguindo ``fit_staging`` do
execution.json), ``epochs_done`` (épocas concluídas, contadas no results.csv) e
``training_state`` (estado do treino pelo summary.json).

Como rodar (na pasta ``sistema_ia``)
------------------------------------
- Só este arquivo::

      DRONECAMP_SKIP_E2E=1 python -m unittest tests.test_gpu_eval_watcher -v

- Um teste só::

      python -m unittest tests.test_gpu_eval_watcher.WatcherTests.test_epochs_and_state_come_from_run_files -v

O que a classe prepara
----------------------
``WatcherTests``: pasta temporária com o run ``runs/pilot_train_x/fit`` vazio. Cada teste
grava execution.json, results.csv ou summary.json conforme o caso.

EXTENSAO: para testar uma função nova do vigia (ex.: ler a época do last.pt), copie
``test_epochs_and_state_come_from_run_files``, grave no run os arquivos que a função lê e
afirme o valor antes e depois de gravá-los.
"""

from pathlib import Path
import json
import sys
import tempfile
import unittest

# Pasta sistema_ia; scripts/ (onde está o vigia) e src/ (pacote) entram no caminho de import.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import gpu_eval_watcher as watcher  # noqa: E402


# ---------------------------------------------------------------------------
# Leitura do run pelo vigia
# ---------------------------------------------------------------------------

class WatcherTests(unittest.TestCase):
    """O vigia só lê arquivos do run; estes testes conferem o que ele deduz deles."""

    def setUp(self) -> None:
        """Cria ``runs/pilot_train_x/fit`` vazio numa pasta temporária."""
        temporary = tempfile.TemporaryDirectory(prefix="dronecamp vigia á ")
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name) / "runs" / "pilot_train_x"
        (self.run / "fit").mkdir(parents=True)

    def test_weights_follow_staging_when_declared(self) -> None:
        """Sem ``fit_staging`` os pesos ficam em ``<run>/fit/weights``; com ele, em ``<fit_staging>/weights``.

        Importa porque, com a área de gravação, o last.pt vive no disco interno e não no run.
        """
        # 1. Padrão: pesos dentro do run.
        self.assertEqual(watcher.weights_directory(self.run), self.run / "fit" / "weights")
        # 2. execution.json declara a área de gravação (caminho Windows, como no PC do projeto).
        (self.run / "execution.json").write_text(json.dumps({"fit_staging": r"C:\DroneCamp_runs\x\fit"}), encoding="utf-8")
        self.assertEqual(watcher.weights_directory(self.run), Path(r"C:\DroneCamp_runs\x\fit") / "weights")

    def test_epochs_and_state_come_from_run_files(self) -> None:
        """Épocas = linhas de dados do results.csv; estado = ``state`` do summary.json, ou "running" sem ele."""
        # 1. Run recém-criado: 0 épocas e "running".
        self.assertEqual(watcher.epochs_done(self.run), 0)
        self.assertEqual(watcher.training_state(self.run), "running", "Sem summary, o treino ainda está começando.")
        # 2. results.csv com cabeçalho + 3 linhas = 3 épocas.
        (self.run / "fit" / "results.csv").write_text("epoch,time\n1,10\n2,20\n3,30\n", encoding="utf-8")
        self.assertEqual(watcher.epochs_done(self.run), 3)
        # 3. summary.json grava o estado final.
        (self.run / "summary.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")
        self.assertEqual(watcher.training_state(self.run), "completed")


# ENTRADA: execução direta do arquivo.
if __name__ == "__main__":
    unittest.main()
