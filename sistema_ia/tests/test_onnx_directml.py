"""Inferência ONNX na GPU (DirectML): proveniência do ONNX, falha explícita e paridade com o .pt.

Módulo coberto
--------------
``dronecamp_ia.backend``: ``onnx_source_checkpoint`` (acha o .pt de origem de um model.onnx
pelo export.json e confere os SHA-256 dos dois), ``load_inference_model`` (carrega .pt ou
.onnx com o provedor pedido) e ``_directml_session`` (sessão ONNX Runtime no DirectML, que
falha alto em vez de cair na CPU em silêncio). O teste de paridade também usa
``exporting.compare_outputs``, ``suggestions.detect_boxes`` e ``training.pilot_inference_config``.

Como rodar (na pasta ``sistema_ia``)
------------------------------------
- Só este arquivo (sem o teste de GPU)::

      DRONECAMP_SKIP_E2E=1 python -m unittest tests.test_onnx_directml -v

- Um teste só::

      python -m unittest tests.test_onnx_directml.OnnxProvenanceTests.test_unknown_provider_is_refused -v

- Com o teste de paridade na GPU (Windows com onnxruntime-directml, ONNX e dataset v9 presentes):
  rode sem ``DRONECAMP_SKIP_E2E=1``.

O que cada classe prepara
-------------------------
- ``OnnxProvenanceTests``: pasta temporária com um ``best.pt`` falso em
  ``runs/pilot_train_x/fit/weights/``, um ``model.onnx`` falso em ``runs/export_x/`` e o dict
  ``self.export`` (conteúdo do export.json, gravado só quando o teste chama ``write_export``).
  Os arquivos são bytes quaisquer: só os hashes importam.
- ``DirectMLParityTests``: usa o projeto REAL (``EXPORT`` e ``PILOT_TEST``); é pulada quando
  ``DRONECAMP_SKIP_E2E=1``, quando os arquivos não existem ou sem DirectML.

EXTENSAO: para testar uma regra nova de proveniência (ex.: conferir a versão do opset no
export.json), copie ``test_changed_onnx_or_checkpoint_is_refused``, altere a chave em
``self.export``, chame ``self.write_export()`` e espere o ValueError. Para um provedor novo
(ex.: "openvino"), copie ``test_missing_directml_fails_loudly`` e
``test_unknown_provider_is_refused``.
"""

from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

# Pasta sistema_ia (dois níveis acima deste arquivo) e src/ no caminho de import.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dronecamp_ia import backend  # noqa: E402
from dronecamp_ia.config import ProjectConfig  # noqa: E402
from dronecamp_ia.io import file_hash  # noqa: E402

# ONNX real exportado do piloto e fotos de teste do dataset v9, usados só no teste de paridade.
# EXTENSAO: ao exportar um modelo novo, atualize o nome da pasta do export e do dataset aqui.
EXPORT = ROOT / "runs" / "export_20261007T024620Z_516be2d6" / "model.onnx"
PILOT_TEST = ROOT / "data" / "pilot" / "ceasa_v9_piloto_s42" / "images" / "test"


# ---------------------------------------------------------------------------
# Proveniência do ONNX e escolha do provedor (sem GPU)
# ---------------------------------------------------------------------------

class OnnxProvenanceTests(unittest.TestCase):
    """O ONNX só é usado se o export.json provar de qual .pt ele veio, com hashes conferidos.

    Importa porque um ONNX trocado ou desatualizado produziria sugestões de outro modelo
    sem ninguém perceber.
    """

    def setUp(self) -> None:
        """Cria best.pt e model.onnx falsos e o conteúdo do export.json (ainda não gravado)."""
        # 1. Pasta temporária com acento e espaço no nome (caso comum no Windows).
        temporary = tempfile.TemporaryDirectory(prefix="dronecamp onnx á ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = ProjectConfig(root=self.root, model="x.pt", requested_model="x.pt",
                                    taxonomy_path=self.root / "t.json", dataset_path=self.root / "d.yaml",
                                    device="cpu", prediction={}, training={}, pilot={})
        # 2. Checkpoint de origem (bytes falsos).
        source = self.root / "runs" / "pilot_train_x" / "fit" / "weights" / "best.pt"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"pesos piloto")
        # 3. ONNX exportado (bytes falsos).
        self.onnx = self.root / "runs" / "export_x" / "model.onnx"
        self.onnx.parent.mkdir(parents=True)
        self.onnx.write_bytes(b"grafo onnx")
        # 4. export.json: caminho de origem no formato Windows (relativo à raiz) + hashes dos dois arquivos.
        # EXTENSAO: campo novo no export.json que o backend passe a conferir: acrescente aqui.
        self.export = {"source_weights": "runs\\pilot_train_x\\fit\\weights\\best.pt",
                       "runtime": {"checkpoint_sha256": file_hash(source)}, "sha256": file_hash(self.onnx)}
        self.source = source

    def write_export(self) -> None:
        """Grava ``self.export`` como ``export.json`` ao lado do model.onnx."""
        (self.onnx.parent / "export.json").write_text(json.dumps(self.export), encoding="utf-8")

    def test_source_checkpoint_is_resolved_and_hash_checked(self) -> None:
        """Com hashes corretos, devolve o caminho absoluto do best.pt de origem (mesmo gravado com ``\\``)."""
        self.write_export()
        self.assertEqual(backend.onnx_source_checkpoint(self.config, self.onnx), self.source.resolve())

    def test_changed_onnx_or_checkpoint_is_refused(self) -> None:
        """ONNX alterado depois da exportação, ou checkpoint com hash diferente, é recusado (ValueError)."""
        # 1. ONNX mudou: o hash não bate com export.json["sha256"].
        self.write_export()
        self.onnx.write_bytes(b"outro grafo")
        with self.assertRaises(ValueError):
            backend.onnx_source_checkpoint(self.config, self.onnx)
        # 2. Corrige o hash do ONNX, mas declara um hash de checkpoint que não confere.
        self.export["sha256"] = file_hash(self.onnx)
        self.export["runtime"]["checkpoint_sha256"] = "0" * 64
        self.write_export()
        with self.assertRaises(ValueError):
            backend.onnx_source_checkpoint(self.config, self.onnx)

    def test_onnx_without_export_manifest_is_refused(self) -> None:
        """ONNX sem export.json ao lado é recusado: não há como provar a origem."""
        with self.assertRaises(ValueError):
            backend.onnx_source_checkpoint(self.config, self.onnx)

    def test_unknown_provider_is_refused(self) -> None:
        """Provedor fora da lista aceita (ex.: "cuda") lança ValueError."""
        with self.assertRaises(ValueError):
            backend.load_inference_model(self.config, str(self.onnx), "cuda")

    def test_missing_directml_fails_loudly(self) -> None:
        """Sem ``DmlExecutionProvider`` disponível, ``_directml_session`` lança RuntimeError em vez de usar a CPU.

        ATENCAO: cair na CPU em silêncio faria a avaliação "na GPU" ficar lenta e com medidas erradas.
        """
        import onnxruntime

        # Simula uma instalação do onnxruntime só com CPU.
        with mock.patch.object(onnxruntime, "get_available_providers", return_value=["CPUExecutionProvider"]):
            with self.assertRaises(RuntimeError):
                backend._directml_session(str(self.onnx))


# ---------------------------------------------------------------------------
# Paridade real DirectML × PyTorch (só no PC com GPU e os arquivos do piloto)
# ---------------------------------------------------------------------------

@unittest.skipIf(os.environ.get("DRONECAMP_SKIP_E2E") == "1", "E2E desativado por DRONECAMP_SKIP_E2E=1.")
@unittest.skipUnless(EXPORT.is_file() and PILOT_TEST.is_dir(), "ONNX exportado do piloto ou dataset v9 ausente.")
class DirectMLParityTests(unittest.TestCase):
    """Compara as caixas do ONNX no DirectML com as do .pt original em 3 fotos de teste."""

    def test_directml_boxes_match_pytorch(self) -> None:
        """As caixas da GPU (ONNX/DirectML) são equivalentes às do PyTorch, e a sessão usa mesmo a GPU.

        Usa tolerância 0.05 em ``compare_outputs`` e confiança mínima 0.05 em ``detect_boxes``.
        """
        import onnxruntime

        # 1. Sem DirectML neste computador, pula (não é falha).
        if "DmlExecutionProvider" not in onnxruntime.get_available_providers():
            self.skipTest("DirectML indisponível neste computador.")
        from dronecamp_ia.config import load_config
        from dronecamp_ia.exporting import compare_outputs
        from dronecamp_ia.suggestions import detect_boxes
        from dronecamp_ia.training import pilot_inference_config

        # 2. Carrega o .pt de origem (referência) e o ONNX no DirectML.
        config = load_config()
        source = backend.onnx_source_checkpoint(config, EXPORT)
        reference = backend.load_inference_model(config, str(source))
        gpu = backend.load_inference_model(config, str(EXPORT), "directml")
        self.assertEqual(gpu.dronecamp_runtime, "onnx-directml")
        self.assertEqual(gpu.dronecamp_source, source)
        # 3. Para as 3 primeiras fotos (ordem alfabética), roda os dois modelos e compara.
        images = sorted(PILOT_TEST.iterdir())[:3]
        for image in images:
            boxes = [detect_boxes(model, image, pilot_inference_config(config, model), 0.05) for model in (reference, gpu)]
            # Converte cada lista de caixas em tuplas (classe, confiança, xyxy) para compare_outputs.
            pairs = [[(b["class_id"], b["confidence"], b["bbox_xyxy"]) for b in value] for value in boxes]
            self.assertTrue(compare_outputs(pairs[0], pairs[1], 0.05)["equivalent"], image.name)
        # 4. Confere que a sessão ONNX Runtime realmente usou o DirectML como primeiro provedor.
        session = gpu.predictor.model.backend.session
        self.assertEqual(session.get_providers()[0], "DmlExecutionProvider", "A sessão usada é a da GPU.")


# ENTRADA: execução direta do arquivo.
if __name__ == "__main__":
    unittest.main()
