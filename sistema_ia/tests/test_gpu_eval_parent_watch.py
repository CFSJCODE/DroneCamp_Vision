"""O gpu-eval-worker sai quando o treino termina, sem ler o stdin no Windows.

No Windows, uma thread presa em ``sys.stdin.buffer.read()`` travava os imports do
worker (o treino de 13/10 ficou com a GPU parada). ``_watch_parent`` espera o processo
do treino pelo pid; ``start_worker`` passa ``--parent-pid``.
"""

from pathlib import Path
import os
import subprocess
import sys
import time
import unittest
from unittest import mock

from dronecamp_ia import gpu_eval
from dronecamp_ia.config import load_config

ROOT = Path(__file__).resolve().parents[1]


class ParentWatchTests(unittest.TestCase):
    """--parent-pid na linha de comando e saída do worker quando o pai morre."""

    def test_start_worker_passes_parent_pid(self):
        """start_worker inclui --parent-pid com o pid do processo do treino."""
        config = load_config(ROOT / "configs" / "project.yaml")
        with mock.patch.object(gpu_eval, "directml_available", return_value=True), \
                mock.patch.object(gpu_eval.subprocess, "Popen") as popen, \
                mock.patch.object(gpu_eval, "write_json"), \
                mock.patch.object(gpu_eval, "folder", return_value=Path(os.environ.get("TEMP", "."))), \
                mock.patch.object(gpu_eval, "work_directory", return_value=Path(os.environ.get("TEMP", "."))), \
                mock.patch("pathlib.Path.open", mock.mock_open()), \
                mock.patch("pathlib.Path.write_text"), mock.patch("pathlib.Path.mkdir"):
            gpu_eval.start_worker(config, ROOT / "runs" / "x", ROOT / "x.yaml", {"provider": "directml", "split": "val"}, 640)
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--parent-pid") + 1], str(os.getpid()))

    @unittest.skipUnless(os.name == "nt", "WaitForSingleObject só no Windows")
    def test_worker_exits_when_parent_dies(self):
        """Um filho com _watch_parent(pid) termina (código 3) logo depois que o pai é encerrado."""
        parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        code = (f"import sys, time; sys.path.insert(0, {str(ROOT / 'src')!r}); "
                f"from dronecamp_ia.gpu_eval import _watch_parent; _watch_parent({parent.pid}); "
                "import numpy; time.sleep(60)")  # o import de numpy não pode travar
        child = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            time.sleep(3)
            self.assertIsNone(child.poll(), "o worker saiu antes do pai terminar")
            parent.kill()
            self.assertEqual(child.wait(timeout=20), 3)
        finally:
            for process in (parent, child):
                if process.poll() is None:
                    process.kill()


class AutoinstallTests(unittest.TestCase):
    """Nenhum comando do pacote deixa a Ultralytics instalar pacotes sozinha."""

    def test_package_import_disables_ultralytics_autoinstall(self):
        """Importar só ``dronecamp_ia`` já define YOLO_AUTOINSTALL=false (antes da Ultralytics)."""
        code = ("import os; os.environ.pop('YOLO_AUTOINSTALL', None); import sys; "
                f"sys.path.insert(0, {str(ROOT / 'src')!r}); import dronecamp_ia; "
                "print(os.environ.get('YOLO_AUTOINSTALL'))")
        environment = {key: value for key, value in os.environ.items() if key != "YOLO_AUTOINSTALL"}
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=environment, timeout=60)
        self.assertEqual(result.stdout.strip(), "false")


if __name__ == "__main__":
    unittest.main()
