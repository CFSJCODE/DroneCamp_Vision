"""Área de gravação do treino (DRONECAMP_FIT_STAGING): espelho do results.csv e cópia final para runs/.

Módulo coberto
--------------
``dronecamp_ia.training`` (funções privadas da área de gravação):
- ``_fit_staging(run)``: devolve a pasta ``<DRONECAMP_FIT_STAGING>/<nome do run>/fit`` quando a
  variável de ambiente existe, ou None (o treino grava direto em ``runs/<run>/fit``);
- ``_mirror_results(staging, run)``: copia o results.csv da área de gravação para o run
  (para a plataforma acompanhar o treino);
- ``_copy_back_fit(staging, run)``: copia os pesos e arquivos finais de volta para o run.

Como rodar (na pasta ``sistema_ia``)
------------------------------------
- Só este arquivo::

      DRONECAMP_SKIP_E2E=1 python -m unittest tests.test_fit_staging -v

- Um teste só::

      python -m unittest tests.test_fit_staging.FitStagingTests.test_without_variable_training_writes_to_run -v

O que a classe prepara
----------------------
``FitStagingTests``: pasta temporária (``self.base``) com o run vazio
``E/runs/pilot_train_20261007T000000Z_abcdef12``. A subpasta ``E`` imita o disco externo
do projeto e ``C`` (criada no teste) o disco interno usado como área de gravação.
A variável ``DRONECAMP_FIT_STAGING`` é trocada só dentro de ``mock.patch.dict``.

EXTENSAO: para testar um arquivo novo que deva voltar para o run (ex.: ``args.yaml``), copie
``test_results_are_mirrored_and_weights_copied_back``, grave o arquivo em ``staging`` antes de
``_copy_back_fit`` e afirme que ele existe em ``self.run / "fit"``.
"""

from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest import mock

# Garante que o pacote em src/ seja importável mesmo rodando o arquivo direto.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronecamp_ia.training import _copy_back_fit, _fit_staging, _mirror_results  # noqa: E402


# ---------------------------------------------------------------------------
# Área de gravação do treino
# ---------------------------------------------------------------------------

class FitStagingTests(unittest.TestCase):
    """Comportamento com e sem ``DRONECAMP_FIT_STAGING``.

    Importa porque treinar gravando num HD externo lento atrasa cada época; a área de
    gravação usa o disco interno, mas o run em ``runs/`` precisa terminar completo.
    """

    def setUp(self) -> None:
        """Cria a pasta temporária e o run vazio dentro de ``E/runs/``."""
        temporary = tempfile.TemporaryDirectory(prefix="dronecamp staging á ")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.run = self.base / "E" / "runs" / "pilot_train_20261007T000000Z_abcdef12"
        self.run.mkdir(parents=True)

    def test_without_variable_training_writes_to_run(self) -> None:
        """Sem a variável de ambiente, ``_fit_staging`` retorna None (treino grava no próprio run)."""
        # mock.patch.dict restaura o ambiente ao sair; dentro dele a variável é removida, se existir.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DRONECAMP_FIT_STAGING", None)
            self.assertIsNone(_fit_staging(self.run))

    def test_results_are_mirrored_and_weights_copied_back(self) -> None:
        """Com a variável, a área é ``C/<run>/fit``; o results.csv é espelhado e o best.pt volta para o run."""
        # 1. Área de gravação calculada a partir da variável e do nome do run.
        with mock.patch.dict(os.environ, {"DRONECAMP_FIT_STAGING": str(self.base / "C")}):
            staging = _fit_staging(self.run)
        self.assertEqual(staging, (self.base / "C" / self.run.name / "fit").resolve())
        # 2. Simula o treino gravando results.csv na área; o espelho copia para runs/<run>/fit.
        (staging / "weights").mkdir(parents=True)
        (staging / "results.csv").write_text("epoch\n1\n", encoding="utf-8")
        _mirror_results(staging, self.run)
        self.assertEqual((self.run / "fit" / "results.csv").read_text(encoding="utf-8"), "epoch\n1\n")
        # 3. No fim, os pesos voltam para o run.
        (staging / "weights" / "best.pt").write_bytes(b"pesos")
        _copy_back_fit(staging, self.run)
        self.assertEqual((self.run / "fit" / "weights" / "best.pt").read_bytes(), b"pesos")


# ENTRADA: execução direta do arquivo.
if __name__ == "__main__":
    unittest.main()
