"""Casos que bloqueiam treino quando os dados violam o contrato do projeto."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from dronecamp_ia.dataset import validate_dataset


class DatasetValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="dronecamp dataset á ")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "dados telhado ç"
        self.root.mkdir()
        self.names = ["telha_quebrada", "corrosao_aparente"]
        self.config = {
            "path": self.root.name,
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": self.names,
        }
        self.yaml_path = self.base / "dataset.yaml"
        for index, split in enumerate(("train", "val", "test")):
            image_directory = self.root / "images" / split
            label_directory = self.root / "labels" / split
            image_directory.mkdir(parents=True)
            label_directory.mkdir(parents=True)
            Image.new("RGB", (16, 16), color=(30 + index * 50, 60, 90)).save(image_directory / "foto.png")
            (label_directory / "foto.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        self.write_yaml()
        self.write_groups()

    def write_yaml(self) -> None:
        self.yaml_path.write_text(yaml.safe_dump(self.config, allow_unicode=True), encoding="utf-8")

    def write_groups(self, same_group: bool = False) -> None:
        with (self.root / "groups.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image", "group_id", "split"])
            writer.writeheader()
            for split in ("train", "val", "test"):
                writer.writerow({"image": f"images/{split}/foto.png", "group_id": "telhado_a" if same_group else f"telhado_{split}", "split": split})

    def validate(self) -> dict:
        result = validate_dataset(self.yaml_path, self.names)
        json.dumps(result, allow_nan=False)  # Todo resultado deve poder ser persistido como JSON.
        return result

    def assert_error(self, fragment: str) -> None:
        result = self.validate()
        self.assertFalse(result["valid"], result)
        self.assertTrue(any(fragment in error for error in result["errors"]), result)

    def test_valid_dataset_resolves_unicode_and_spaces(self) -> None:
        result = self.validate()
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["counts"]["total"], {"images": 3, "labels": 3, "objects": 3, "negative_images": 0})

    def test_names_dictionary_is_accepted_with_contiguous_ids(self) -> None:
        self.config["names"] = {0: self.names[0], 1: self.names[1]}
        self.write_yaml()
        self.assertTrue(self.validate()["valid"])

    def test_class_order_cannot_change(self) -> None:
        self.config["names"] = list(reversed(self.names))
        self.write_yaml()
        self.assert_error("ordem de IDs")

    def test_noncontiguous_class_ids_are_rejected(self) -> None:
        self.config["names"] = {0: self.names[0], 2: self.names[1]}
        self.write_yaml()
        self.assert_error("consecutivos")

    def test_invalid_boxes_and_ids_are_rejected(self) -> None:
        cases = {
            "0 nan 0.5 0.1 0.1": "finitas",
            "0 0.5 inf 0.1 0.1": "finitas",
            "0 0.5 0.5 0 0.2": "maiores que zero",
            "0 0.05 0.5 0.2 0.2": "ultrapassa",
            "0 1.5 0.5 0.2 0.2": "intervalo",
            "0.0 0.5 0.5 0.2 0.2": "deve ser inteiro",
            "2 0.5 0.5 0.2 0.2": "fora dos IDs",
            "0 0.5 0.5 0.2": "5 valores",
        }
        for label, fragment in cases.items():
            with self.subTest(label=label):
                (self.root / "labels/train/foto.txt").write_text(label, encoding="utf-8")
                self.assert_error(fragment)

    def test_empty_label_is_explicit_negative(self) -> None:
        (self.root / "labels/train/foto.txt").write_text("", encoding="utf-8")
        result = self.validate()
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["counts"]["train"]["negative_images"], 1)
        self.assertEqual(result["counts"]["train"]["objects"], 0)

    def test_missing_label_is_not_silently_negative(self) -> None:
        (self.root / "labels/train/foto.txt").unlink()
        self.assert_error("label ausente")

    def test_duplicate_image_between_train_and_val_is_leakage(self) -> None:
        shutil.copyfile(self.root / "images/train/foto.png", self.root / "images/val/foto.png")
        self.assert_error("vazamento entre splits")

    def test_group_cannot_span_splits(self) -> None:
        self.write_groups(same_group=True)
        self.assert_error("vazamento por grupo")

    def test_manifest_is_required(self) -> None:
        (self.root / "groups.csv").unlink()
        self.assert_error("groups.csv obrigatório")

    def test_manifest_requires_complete_membership(self) -> None:
        with (self.root / "groups.csv").open("w", encoding="utf-8", newline="") as stream:
            stream.write("image,group_id,split\nimages/train/foto.png,telhado_a,train\n")
        self.assert_error("imagem sem grupo")

    def test_empty_splits_do_not_pass(self) -> None:
        for split in ("train", "val", "test"):
            (self.root / f"images/{split}/foto.png").unlink()
            (self.root / f"labels/{split}/foto.txt").unlink()
        self.assert_error("nenhum arquivo de imagem")

    def test_corrupt_image_is_rejected(self) -> None:
        (self.root / "images/train/foto.png").write_bytes(b"not an image")
        self.assert_error("imagem ilegível")

    def test_png_with_invalid_checksum_is_reported_without_crashing(self) -> None:
        image_path = self.root / "images/train/foto.png"
        contents = bytearray(image_path.read_bytes())
        chunk_start = contents.index(b"IDAT")
        chunk_length = int.from_bytes(contents[chunk_start - 4 : chunk_start], "big")
        crc_start = chunk_start + 4 + chunk_length
        contents[crc_start] ^= 1
        image_path.write_bytes(contents)
        self.assert_error("imagem ilegível")

    def test_images_smaller_than_loader_minimum_are_rejected(self) -> None:
        for size in ((9, 16), (16, 9)):
            with self.subTest(size=size):
                Image.new("RGB", size, color="red").save(self.root / "images/train/foto.png")
                self.assert_error("pelo menos 10 pixels")

    def test_labels_with_utf8_bom_are_rejected_without_rewriting(self) -> None:
        label_path = self.root / "labels/train/foto.txt"
        for contents in ("0 0.5 0.5 0.2 0.2\n", ""):
            with self.subTest(contents=contents):
                original = b"\xef\xbb\xbf" + contents.encode("utf-8")
                label_path.write_bytes(original)
                self.assert_error("BOM UTF-8")
                self.assertEqual(label_path.read_bytes(), original)

    def test_jpeg_without_final_eoi_is_rejected_without_repair(self) -> None:
        image_path = self.root / "images/train/foto.png"
        Image.new("RGB", (16, 16), color="red").save(image_path, format="JPEG")
        complete = image_path.read_bytes()
        for corrupted in (complete[:-2], complete + b"trailing data"):
            with self.subTest(size=len(corrupted)):
                image_path.write_bytes(corrupted)
                self.assert_error("marcador EOI final")
                self.assertEqual(image_path.read_bytes(), corrupted)

    def test_nested_images_directory_cannot_redirect_labels(self) -> None:
        image_path = self.root / "images/train/foto.png"
        label_path = self.root / "labels/train/foto.txt"
        (image_path.parent / "images").mkdir()
        (label_path.parent / "images").mkdir()
        image_path.rename(image_path.parent / "images/foto.png")
        label_path.rename(label_path.parent / "images/foto.txt")
        self.assert_error("caminho de label incompatível")

    def test_capitalized_images_root_cannot_hide_label_mapping_failure(self) -> None:
        images_root = self.root / "images"
        renamed_root = self.root / "Images"
        # Um passo intermediário garante renomeação só de casing no Windows.
        temporary_root = self.root / "images_case_change"
        images_root.rename(temporary_root)
        temporary_root.rename(renamed_root)
        for split in ("train", "val", "test"):
            self.config[split] = f"Images/{split}"
        self.write_yaml()
        # Em sistemas case-sensitive, a ausência de images já bloqueia antes.
        self.assert_error("caminho de label incompatível" if images_root.is_dir() else "path/images")

    def test_orphan_label_is_rejected(self) -> None:
        (self.root / "labels/train/sem_foto.txt").write_text("", encoding="utf-8")
        self.assert_error("label órfã")

    def test_same_stem_different_extensions_is_rejected(self) -> None:
        Image.new("RGB", (16, 16), color="white").save(self.root / "images/train/foto.jpg")
        self.assert_error("colisão de stem")

    def test_overlapping_split_directories_are_rejected(self) -> None:
        self.config["val"] = "images/train"
        self.write_yaml()
        self.assert_error("diretórios iguais ou sobrepostos")

    def test_download_value_is_not_executed(self) -> None:
        self.config["download"] = "raise RuntimeError('não executar')"
        self.write_yaml()
        result = self.validate()
        self.assertTrue(result["valid"], result)
        self.assertTrue(any("download" in warning for warning in result["warnings"]))

    def test_validation_does_not_change_dataset_files(self) -> None:
        before = {path.relative_to(self.base): path.read_bytes() for path in self.base.rglob("*") if path.is_file()}
        self.validate()
        after = {path.relative_to(self.base): path.read_bytes() for path in self.base.rglob("*") if path.is_file()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
