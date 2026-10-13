"""Estado das execuções no Monitoramento (``operations._run_state``).

O train-pilot grava ``summary.json`` com ``state: "running"`` e ``complete: false`` logo no
início; esse treino tem de aparecer como "em andamento", não como falha. Um treino parado
de propósito e anotado com ``state: "interrupted"`` aparece como interrompido.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import os
import time
import unittest

from dronecamp_ia.operations import _run_state


class RunStateTests(unittest.TestCase):
    """running / interrupted / failed / completed a partir do summary e do heartbeat."""

    def test_states(self):
        """Início do treino com heartbeat recente = running; anotado = interrupted; falha e fim continuam."""
        with TemporaryDirectory() as folder:
            fit = Path(folder) / "fit"
            fit.mkdir()
            results = fit / "results.csv"
            (fit / "heartbeat").write_text("1", encoding="utf-8")
            now = time.time()
            starting = {"state": "running", "complete": False}
            self.assertEqual(_run_state(starting, results, now), "running")
            self.assertEqual(_run_state({"state": "interrupted", "complete": False}, results, now), "interrupted")
            self.assertEqual(_run_state({"state": "failed", "complete": False}, results, now), "failed")
            self.assertEqual(_run_state({"state": "completed", "complete": True}, results, now), "completed")
            # Heartbeat velho e nenhuma época: o processo morreu antes da 1ª época.
            old = now - 3600
            os.utime(fit / "heartbeat", (old, old))
            self.assertEqual(_run_state(starting, results, now), "prepared")


if __name__ == "__main__":
    unittest.main()
