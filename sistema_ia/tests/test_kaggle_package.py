"""Pacote para a GPU do Kaggle: montagem no PC, conferência e treino a partir dele.

Módulos cobertos
----------------
- ``dronecamp_ia.kaggle_package``: ``build_package`` (treino novo e continuação),
  ``verify_package``, ``find_package``, ``rebase_checkpoint`` e ``train_from_package``.

Como rodar (na pasta ``sistema_ia``)
------------------------------------
- Só as regras do pacote, sem treino::

      DRONECAMP_SKIP_E2E=1 python -m unittest tests.test_kaggle_package -v

- Com o treino real curto na CPU (baixa o yolo26n.pt oficial, ~5 MB)::

      python -m unittest tests.test_kaggle_package -v

O que cada classe prepara
-------------------------
- ``PackageRuleTests``: um run piloto falso (janelas sintéticas, lista de classes raras
  e um ``last.pt`` mínimo) numa pasta temporária.
- ``SendScriptTests``: ``kaggle/enviar_kaggle.py --so-preparar`` sobre o pacote falso
  (nada é enviado; confere o notebook com as opções e os metadados do Kaggle).
- ``PackageTrainingTests``: dataset piloto real da revisão v7 (taxonomia v5), pacote
  em janelas e dois treinos curtos: um parado pelo limite de horas e a continuação dele.
"""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import json
import os
import shutil
import unittest

import yaml

from dronecamp_ia.config import detection_names, load_config, load_taxonomy
from dronecamp_ia.kaggle_package import build_package, find_package, rebase_checkpoint, verify_package

REAL = load_config()
REGISTRY = REAL.root / "data" / "reviews" / "ceasa_v7_revisao002_ba8cc323c8a1" / "registry.json"
TAXONOMY_V5 = REAL.root / "configs" / "taxonomy_ceasa_v5_reparo_rufo.json"


def fake_run(base: Path, names: list[str]) -> Path:
    """Monta ``runs/pilot_train_x`` com janelas sintéticas, lista de raras e ``last.pt`` mínimo.

    Retorna:
        Path: a pasta do run falso.
    """
    import torch

    run = base / "runs" / "pilot_train_falso"
    tiles = base / "staging" / "tiled_dataset"
    for split in ("train", "val"):
        (tiles / "images" / split).mkdir(parents=True)
        (tiles / "labels" / split).mkdir(parents=True)
        for index in range(2):
            (tiles / "images" / split / f"{split}_{index}.jpg").write_bytes(b"jpg" + bytes([index]))
            (tiles / "labels" / split / f"{split}_{index}.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (run / "fit" / "weights").mkdir(parents=True)
    # Classes raras: a foto train_1 aparece duas vezes na lista (caminhos absolutos, como no PC).
    listing = run / "train_repeat_factor.txt"
    listing.write_text("\n".join(str(tiles / "images" / "train" / name)
                                 for name in ("train_0.jpg", "train_1.jpg", "train_1.jpg")) + "\n", encoding="utf-8")
    (run / "dataset_tiled_resolved.yaml").write_text(yaml.safe_dump(
        {"path": str(tiles), "names": names, "train": str(listing), "val": str(tiles / "images" / "val")}), encoding="utf-8")
    (run / "execution.json").write_text(json.dumps({"parameters": {"epochs": 120, "imgsz": 640}}), encoding="utf-8")
    torch.save({"epoch": 60, "train_args": {"data": "E:/x/data.yaml", "device": "cpu", "amp": False}},
               run / "fit" / "weights" / "last.pt")
    # A época 62 simula o treino do PC fechando outra época depois da cópia do last.pt (época 61).
    (run / "fit" / "results.csv").write_text("epoch,x\n60,0\n61,0\n62,0\n", encoding="utf-8")
    return run


class PackageRuleTests(unittest.TestCase):
    """Regras do pacote sem treinar: o que vai, como se confere e o que é recusado."""

    def setUp(self):
        """Raiz temporária com configs reais e um run piloto falso."""
        self.temporary = TemporaryDirectory()
        self.base = Path(self.temporary.name)
        shutil.copytree(REAL.root / "configs", self.base / "configs")
        self.config = replace(REAL, root=self.base, taxonomy_path=self.base / "configs" / REAL.taxonomy_path.name)
        self.names = detection_names(load_taxonomy(self.config.taxonomy_path))
        self.run = fake_run(self.base, self.names)
        self.package = build_package(self.config, self.base / "out", resume_run=self.run)

    def tearDown(self):
        self.temporary.cleanup()

    def test_resume_package_carries_windows_checkpoint_and_relative_paths(self):
        """Continuação leva janelas, last.pt, results.csv e só caminhos relativos ao pacote."""
        manifest = verify_package(self.package)
        self.assertEqual(manifest["mode"], "resume")
        self.assertEqual(manifest["checkpoint"]["epoch_completed"], 61)
        history = (self.package / "checkpoint" / "results.csv").read_text(encoding="utf-8").split()
        self.assertEqual(history, ["epoch,x", "60,0", "61,0"])
        self.assertTrue(self.package.with_suffix(".zip").is_file())
        data = yaml.safe_load((self.package / "dataset" / "dataset.yaml").read_text(encoding="utf-8"))
        self.assertEqual(data["path"], ".")
        self.assertEqual(data["train"], "train_list.txt")
        # A repetição das fotos raras é mantida, agora relativa à pasta do .txt.
        lines = (self.package / "dataset" / "train_list.txt").read_text(encoding="utf-8").split()
        self.assertEqual(lines, ["./images/train/train_0.jpg", "./images/train/train_1.jpg", "./images/train/train_1.jpg"])
        self.assertEqual(manifest["counts"]["train"]["images"], 2)
        # Nenhum caminho absoluto do PC dentro do package.json.
        self.assertNotIn(str(self.base), (self.package / "package.json").read_text(encoding="utf-8"))

    def test_tampered_or_extra_file_is_refused(self):
        """Foto alterada ou arquivo fora do inventário reprovam o pacote."""
        extra = self.package / "dataset" / "images" / "train" / "intrusa.jpg"
        extra.write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "fora do inventário"):
            verify_package(self.package)
        extra.unlink()
        (self.package / "dataset" / "labels" / "val" / "val_0.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "alterado"):
            verify_package(self.package)

    def test_find_package_accepts_kaggle_extra_folder(self):
        """O Kaggle pode descompactar dentro de mais uma pasta; find_package acha o package.json."""
        nested = self.base / "input" / "dataset"
        shutil.copytree(self.package, nested / self.package.name)
        self.assertEqual(find_package(self.base / "input"), (nested / self.package.name).resolve())

    def test_rebase_changes_copy_and_keeps_original(self):
        """A cópia recebe dados/device/amp novos; o last.pt do pacote não muda."""
        import torch

        source = self.package / "checkpoint" / "last.pt"
        before = source.read_bytes()
        copy = rebase_checkpoint(source, self.base / "rebased.pt", {"data": "/kaggle/x.yaml", "device": "0", "amp": True})
        args = torch.load(copy, weights_only=False)["train_args"]
        self.assertEqual((args["data"], args["device"], args["amp"]), ("/kaggle/x.yaml", "0", True))
        self.assertEqual(source.read_bytes(), before)

    def test_checkpoint_from_another_package_is_refused(self):
        """--checkpoint de uma sessão que usou outro pacote é recusado antes de treinar."""
        from dronecamp_ia.kaggle_package import train_from_package

        previous = self.base / "anterior" / "fit" / "weights"
        previous.mkdir(parents=True)
        shutil.copy2(self.package / "checkpoint" / "last.pt", previous / "resume.pt")
        (self.base / "anterior" / "summary.json").write_text(json.dumps({"package": {"package_sha256": "0" * 64}}),
                                                             encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "outro pacote"):
            train_from_package(self.config, self.package, device="cpu", checkpoint=previous / "resume.pt")

    def test_requires_exactly_one_source(self):
        """Sem origem, ou com as duas, o comando explica e não cria pasta."""
        with self.assertRaisesRegex(ValueError, "não os dois"):
            build_package(self.config, self.base / "vazio")
        self.assertFalse((self.base / "vazio").exists())


def load_send_script():
    """Importa ``kaggle/enviar_kaggle.py`` (pasta de scripts, fora do pacote Python)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("enviar_kaggle", REAL.root / "kaggle" / "enviar_kaggle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SendScriptTests(unittest.TestCase):
    """Preparação do envio: notebook com as opções certas e Dataset/notebook privados na T4."""

    def setUp(self):
        self.temporary = TemporaryDirectory()
        base = Path(self.temporary.name)
        shutil.copytree(REAL.root / "configs", base / "configs")
        config = replace(REAL, root=base, taxonomy_path=base / "configs" / REAL.taxonomy_path.name)
        self.package = build_package(config, base / "out", resume_run=fake_run(base, detection_names(
            load_taxonomy(config.taxonomy_path))))
        self.saida = base / "saida"
        self.send = load_send_script()

    def tearDown(self):
        self.temporary.cleanup()

    def test_prepare_only_writes_patched_notebook_and_private_metadata(self):
        """--so-preparar grava o zip, o notebook com branch/horas/extras e metadados privados."""
        code = self.send.main(["--usuario", "Claudio", "--pacote", str(self.package), "--saida", str(self.saida),
                               "--branch", "teste-ramo", "--max-horas", "0.25", "--extra", "--batch 8", "--so-preparar"])
        self.assertEqual(code, 0)
        dataset = json.loads((self.saida / f"envio_{self.package.name}" / "dataset-metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(dataset["id"], "claudio/dronecamp-pacote")
        self.assertTrue((self.saida / f"envio_{self.package.name}" / f"{self.package.name}.zip").is_file())
        kernel = json.loads((self.saida / "notebook" / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.assertTrue(kernel["is_private"] and kernel["enable_gpu"] and kernel["enable_internet"])
        self.assertEqual((kernel["machine_shape"], kernel["dataset_sources"]), ("NvidiaTeslaT4", ["claudio/dronecamp-pacote"]))
        notebook = json.loads((self.saida / "notebook" / "treino_t4.ipynb").read_text(encoding="utf-8"))
        source = "".join("".join(cell["source"]) for cell in notebook["cells"])
        for line in ("BRANCH = 'teste-ramo'", "MAX_HOURS = 0.25", "EXTRA_ARGS = ['--batch', '8']", "CONTINUE_FROM = None"):
            self.assertIn(line, source)
        # O comentário explicativo da linha continua lá.
        self.assertIn("MAX_HOURS = 0.25                 # para com segurança", source)

    def test_unknown_setting_is_refused(self):
        """Opção que não existe na célula de configuração é erro, não silêncio."""
        with self.assertRaisesRegex(ValueError, "ausentes"):
            self.send.patch_notebook(self.send.NOTEBOOK.read_text(encoding="utf-8"), {"NAO_EXISTE": 1})


@unittest.skipIf(os.environ.get("DRONECAMP_SKIP_E2E") == "1", "E2E desativado por DRONECAMP_SKIP_E2E=1.")
@unittest.skipUnless(REGISTRY.is_file() and TAXONOMY_V5.is_file(), "Revisão v7 ou taxonomia v5 ausente.")
class PackageTrainingTests(unittest.TestCase):
    """Treino real curto a partir do pacote, com parada pelo limite de horas e continuação."""

    @classmethod
    def setUpClass(cls):
        """Dataset piloto real → pacote em janelas → treino parado após 1 época → continuação."""
        from dronecamp_ia.kaggle_package import train_from_package
        from dronecamp_ia.pilot import build_pilot_dataset

        cls.temporary = TemporaryDirectory()
        root = Path(cls.temporary.name)
        shutil.copytree(REAL.root / "configs", root / "configs")
        taxonomy = root / "configs" / "taxonomy.json"
        shutil.copyfile(TAXONOMY_V5, taxonomy)
        cls.config = replace(REAL, root=root, taxonomy_path=taxonomy)
        dataset = root / "data" / "pilot" / "kaggle"
        build_pilot_dataset(cls.config, [REGISTRY], dataset)
        cls.package = build_package(cls.config, root / "pacotes", data_path=dataset / "dataset.yaml", make_zip=False)
        tiny = {"epochs": 2, "imgsz": 160, "batch": 8, "plots": False, "workers": 0}
        # max_hours minúsculo: para depois da 1ª época salva, como no fim de uma sessão do Kaggle.
        cls.first = train_from_package(cls.config, cls.package, device="cpu", overrides=tiny,
                                       weights="yolo26n.pt", max_hours=1e-6)
        cls.second = train_from_package(cls.config, cls.package, device="cpu",
                                        checkpoint=cls.first / "fit" / "weights" / "resume.pt")

    @classmethod
    def tearDownClass(cls):
        from dronecamp_ia.backend import _ultralytics

        _ultralytics(REAL)
        cls.temporary.cleanup()

    def test_fresh_package_keeps_pilot_audit_and_tiling(self):
        """O pacote novo leva a auditoria do piloto e o fatiamento do project.yaml."""
        manifest = verify_package(self.package)
        self.assertEqual(manifest["mode"], "fresh")
        self.assertTrue(manifest["audit"]["valid"])
        self.assertEqual(manifest["tiling"]["patch"], REAL.pilot["tiling"]["patch"])
        self.assertGreater(manifest["counts"]["train"]["images"], 0)

    def test_time_limit_stops_with_resumable_checkpoint(self):
        """A 1ª sessão para como "interrupted" e deixa resume.pt com otimizador."""
        summary = json.loads((self.first / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["state"], "interrupted")
        self.assertEqual(summary["epochs_recorded"], 1)
        self.assertTrue((self.first / "fit" / "weights" / "resume.pt").is_file())

    def test_continuation_finishes_and_keeps_epoch_history(self):
        """A 2ª sessão termina a época 2, auditada, com o histórico da 1ª no results.csv."""
        summary = json.loads((self.second / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["state"], "completed", summary.get("reasons"))
        self.assertTrue(summary["pilot"])
        self.assertFalse(summary["production_ready"])
        with (self.second / "fit" / "results.csv").open(encoding="utf-8") as handle:
            epochs = [int(float(row["epoch"])) for row in csv.DictReader(handle)]
        self.assertEqual(epochs, [1, 2])
        self.assertFalse((self.second / "checkpoint_rebased.pt").exists())


if __name__ == "__main__":
    unittest.main()
