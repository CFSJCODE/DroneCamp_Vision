"""Piloto remapeado da taxonomia anterior (v6) para a atual (v7) pelo mapa da migração.

Módulos cobertos
----------------
- ``dronecamp_ia.pilot.migration_class_remap``: mapa de IDs derivado de
  ``migration.active_class_map`` (toda classe ativa antiga precisa de destino).
- ``dronecamp_ia.pilot.remap_pilot_dataset`` e ``validate_pilot_dataset``: a cópia
  remapeada passa na validação completa (hashes, aprovação humana, labels recalculadas)
  e um mapa adulterado no manifesto é recusado.

Usa um mini piloto (uma foto por split) recortado do piloto real
``data/pilot/ceasa_lado_a_v4_rotulos_v6_s42``; sem ele, os testes de dataset são pulados.

Como rodar (na pasta ``sistema_ia``)::

    python -m unittest tests.test_pilot_remap -v
"""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import yaml

from dronecamp_ia.config import detection_names, load_config, load_taxonomy
from dronecamp_ia.dataset import SPLITS
from dronecamp_ia.io import write_json
from dronecamp_ia.pilot import migration_class_remap, remap_pilot_dataset, validate_pilot_dataset

REAL = load_config()
SOURCE = REAL.root / "data" / "pilot" / "ceasa_lado_a_v4_rotulos_v6_s42"


class MigrationMapTests(unittest.TestCase):
    def test_every_previous_class_maps_into_current_names(self):
        source_sha, mapping = migration_class_remap(REAL)
        migration = load_taxonomy(REAL.taxonomy_path)["migration"]
        previous = detection_names(load_taxonomy(REAL.root / migration["from_taxonomy_file"]))
        self.assertEqual(source_sha, migration["from_taxonomy_sha256"])
        self.assertEqual(sorted(mapping), list(range(len(previous))))
        self.assertTrue(all(0 <= target < len(detection_names(load_taxonomy(REAL.taxonomy_path)))
                            for target in mapping.values()))


@unittest.skipUnless((SOURCE / "pilot.json").exists(), "piloto v6 real ausente")
class RemapPilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)
        # Mini piloto: a primeira foto de cada split, com o manifesto original reduzido.
        manifest = json.loads((SOURCE / "pilot.json").read_text(encoding="utf-8"))
        picked = []
        for split in SPLITS:
            picked.append(next(s for s in manifest["images"] if s["image"].split("/")[1] == split))
        self.mini = self.temp / "mini_v6"
        for sample in picked:
            for key in ("image", "label"):
                (self.mini / sample[key]).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(SOURCE / sample[key], self.mini / sample[key])
        shutil.copy2(SOURCE / "dataset.yaml", self.mini / "dataset.yaml")
        keep = {sample["image"] for sample in picked}
        rows = (SOURCE / "groups.csv").read_text(encoding="utf-8").splitlines()
        (self.mini / "groups.csv").write_text(
            "\n".join([rows[0], *(row for row in rows[1:] if row.split(",")[0] in keep)]) + "\n", encoding="utf-8")
        write_json(self.mini / "pilot.json", {**manifest, "images": picked})

    def test_remapped_copy_validates_and_uses_current_names(self):
        report = remap_pilot_dataset(REAL, self.mini, self.temp / "mini_v7")
        data = yaml.safe_load(Path(report["dataset"]).read_text(encoding="utf-8"))
        self.assertEqual(data["names"], detection_names(load_taxonomy(REAL.taxonomy_path)))
        self.assertTrue(validate_pilot_dataset(REAL, Path(report["dataset"]))["valid"])

    def test_tampered_map_is_rejected(self):
        output = self.temp / "mini_v7"
        remap_pilot_dataset(REAL, self.mini, output)
        manifest = json.loads((output / "pilot.json").read_text(encoding="utf-8"))
        first = next(iter(manifest["class_remap"]["map"]))
        manifest["class_remap"]["map"][first] = (manifest["class_remap"]["map"][first] + 1) % 9
        write_json(output / "pilot.json", manifest)
        with self.assertRaises(ValueError):
            validate_pilot_dataset(REAL, output / "dataset.yaml")

    def test_already_remapped_source_is_refused(self):
        output = self.temp / "mini_v7"
        remap_pilot_dataset(REAL, self.mini, output)
        with self.assertRaises(ValueError):
            remap_pilot_dataset(REAL, output, self.temp / "de_novo")


if __name__ == "__main__":
    unittest.main()
