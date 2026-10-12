"""Casos que bloqueiam treino quando os dados violam o contrato do projeto.

Módulo coberto: ``dronecamp_ia.dataset`` (função ``validate_dataset``), a validação
que roda antes de qualquer treino e decide se o dataset YOLO pode ser usado.

Como rodar (na pasta ``sistema_ia``, com o venv do projeto):
    python -m unittest tests.test_dataset                       # só este arquivo
    python -m unittest tests.test_dataset.DatasetValidationTests.test_orphan_label_is_rejected
                                                                # um único teste
    python -m unittest tests.test_dataset -v                    # mostra a 1ª linha de cada docstring

O que a classe de teste prepara (``DatasetValidationTests.setUp``):
    - uma pasta temporária com acento e espaço no nome (``dronecamp dataset á ...``) e,
      dentro dela, a raiz do dataset ``dados telhado ç`` (para provar que Unicode e espaços funcionam);
    - um ``dataset.yaml`` ao lado da raiz, com ``path``/``train``/``val``/``test``/``names``;
    - uma imagem PNG 16x16 e uma label YOLO válida em cada split (train, val, test);
    - um ``groups.csv`` que coloca cada imagem num grupo (telhado) diferente.
    Assim o estado inicial é um dataset VÁLIDO; cada teste estraga uma coisa e confere o erro.
    A pasta temporária é apagada sozinha no fim (``addCleanup``).

Helpers: ``write_yaml`` (regrava o YAML a partir de ``self.config``), ``write_groups``
(regrava o manifesto de grupos), ``validate`` (chama ``validate_dataset`` e exige JSON válido)
e ``assert_error`` (exige ``valid == False`` e uma mensagem contendo o trecho dado).

ATENCAO: os testes procuram TRECHOS das mensagens de erro em português (ex.: "label órfã").
Se você mudar o texto de uma mensagem em dataset.py, atualize o trecho correspondente aqui.

EXTENSAO: para testar uma regra nova de ``validate_dataset``, copie
``test_orphan_label_is_rejected``: (1) altere um arquivo em ``self.root`` ou uma chave de
``self.config`` (chame ``self.write_yaml()`` depois de mexer no config); (2) chame
``self.assert_error("trecho da mensagem nova")``. Para uma regra que só gera aviso, siga
``test_download_value_is_not_executed`` e confira ``result["warnings"]``. Para um caso de
label inválida, basta acrescentar uma linha no dicionário ``cases`` de
``test_invalid_boxes_and_ids_are_rejected``.
"""

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
    """Testes de ``validate_dataset`` sobre um dataset YOLO mínimo criado em pasta temporária.

    Cada teste parte de um dataset válido (montado em ``setUp``) e quebra uma única regra
    do contrato: ordem de classes, labels, imagens, splits, manifesto de grupos etc.
    Isso importa porque um dataset errado gera um modelo errado sem nenhum aviso do YOLO.
    """
    def setUp(self) -> None:
        """Monta um dataset YOLO válido com 1 imagem + 1 label por split e um ``groups.csv``.

        Cria: ``self.base`` (pasta temporária), ``self.root`` (raiz do dataset), ``self.names``
        (as 2 classes usadas), ``self.config`` (conteúdo do YAML) e ``self.yaml_path``.
        Retorna: None.
        """
        # 1. Pasta temporária com acento e espaço no nome; apagada sozinha no fim do teste.
        self.temporary = tempfile.TemporaryDirectory(prefix="dronecamp dataset á ")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        # 2. Raiz do dataset (também com caractere especial) e as classes usadas nos testes.
        self.root = self.base / "dados telhado ç"
        self.root.mkdir()
        self.names = ["telha_quebrada", "corrosao_aparente"]
        # 3. Conteúdo do dataset.yaml; os testes alteram este dict e chamam write_yaml().
        # EXTENSAO: nova chave do YAML a testar: acrescente aqui (vale para todos os testes).
        self.config = {
            "path": self.root.name,
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": self.names,
        }
        self.yaml_path = self.base / "dataset.yaml"
        # 4. Uma imagem 16x16 (cor diferente por split, para não haver duplicata) e uma label
        # com uma caixa central da classe 0 em cada split.
        for index, split in enumerate(("train", "val", "test")):
            image_directory = self.root / "images" / split
            label_directory = self.root / "labels" / split
            image_directory.mkdir(parents=True)
            label_directory.mkdir(parents=True)
            Image.new("RGB", (16, 16), color=(30 + index * 50, 60, 90)).save(image_directory / "foto.png")
            (label_directory / "foto.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        # 5. Grava o YAML e o manifesto de grupos: estado inicial válido.
        self.write_yaml()
        self.write_groups()

    def write_yaml(self) -> None:
        """Grava ``self.config`` em ``self.yaml_path`` (chame depois de alterar o config).

        Retorna: None.
        """
        self.yaml_path.write_text(yaml.safe_dump(self.config, allow_unicode=True), encoding="utf-8")

    def write_groups(self, same_group: bool = False) -> None:
        """Regrava ``groups.csv`` com uma linha por split.

        Parâmetros:
            same_group: se True, as 3 imagens ficam no MESMO grupo ``telhado_a`` (caso de
                vazamento por grupo); se False, cada split tem seu próprio grupo.
        Retorna: None.
        """
        with (self.root / "groups.csv").open("w", encoding="utf-8", newline="") as stream:
            # Colunas do manifesto: caminho da imagem relativo à raiz, grupo (telhado) e split.
            writer = csv.DictWriter(stream, fieldnames=["image", "group_id", "split"])
            writer.writeheader()
            for split in ("train", "val", "test"):
                writer.writerow({"image": f"images/{split}/foto.png", "group_id": "telhado_a" if same_group else f"telhado_{split}", "split": split})

    def validate(self) -> dict:
        """Roda ``validate_dataset`` e garante que o resultado é serializável em JSON estrito.

        Retorna: o dict do validador (chaves ``valid``, ``errors``, ``warnings``, ``counts``).
        Erros: ValueError se o resultado tiver NaN/Infinity (``allow_nan=False``).
        """
        result = validate_dataset(self.yaml_path, self.names)
        # Garante que o resultado pode ser gravado como JSON estrito (sem NaN).
        json.dumps(result, allow_nan=False)  # Todo resultado deve poder ser persistido como JSON.
        return result

    def assert_error(self, fragment: str) -> None:
        """Exige que o dataset seja inválido e que algum erro contenha ``fragment``.

        Parâmetros:
            fragment: trecho (em português) da mensagem esperada, ex.: "label órfã".
        Retorna: None (falha o teste se a condição não valer).
        """
        result = self.validate()
        self.assertFalse(result["valid"], result)
        self.assertTrue(any(fragment in error for error in result["errors"]), result)

    def test_valid_dataset_resolves_unicode_and_spaces(self) -> None:
        """Dataset com acentos e espaços no caminho é aceito e contado corretamente.

        Importa porque as pastas reais do Cláudio no Windows têm acentos e espaços.
        """
        result = self.validate()
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["counts"]["total"], {"images": 3, "labels": 3, "objects": 3, "negative_images": 0})

    def test_names_dictionary_is_accepted_with_contiguous_ids(self) -> None:
        """``names`` no formato dicionário {0: ..., 1: ...} é aceito quando os IDs são 0..N-1.

        O YOLO aceita as duas formas (lista e dict); o validador não pode rejeitar a válida.
        """
        self.config["names"] = {0: self.names[0], 1: self.names[1]}
        self.write_yaml()
        self.assertTrue(self.validate()["valid"])

    def test_class_order_cannot_change(self) -> None:
        """Inverter a ordem das classes no YAML é erro ("ordem de IDs").

        ATENCAO: o ID de cada classe é contrato com taxonomy.json, pesos e labels; trocar a
        ordem faria o modelo aprender a classe errada.
        """
        self.config["names"] = list(reversed(self.names))
        self.write_yaml()
        self.assert_error("ordem de IDs")

    def test_noncontiguous_class_ids_are_rejected(self) -> None:
        """IDs de classe com buraco (0 e 2) são rejeitados ("consecutivos")."""
        self.config["names"] = {0: self.names[0], 2: self.names[1]}
        self.write_yaml()
        self.assert_error("consecutivos")

    def test_invalid_boxes_and_ids_are_rejected(self) -> None:
        """Cada linha de label malformada gera o erro específico esperado.

        Cobre NaN/inf, largura zero, caixa que sai da imagem, centro fora de 0..1, ID não
        inteiro, ID inexistente e quantidade errada de valores.
        EXTENSAO: novo caso de label inválida = nova entrada "linha da label": "trecho do erro"
        no dicionário ``cases``.
        """
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
        """Arquivo de label vazio conta como imagem negativa (sem defeito), não como erro."""
        (self.root / "labels/train/foto.txt").write_text("", encoding="utf-8")
        result = self.validate()
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["counts"]["train"]["negative_images"], 1)
        self.assertEqual(result["counts"]["train"]["objects"], 0)

    def test_missing_label_is_not_silently_negative(self) -> None:
        """Imagem sem arquivo de label é erro ("label ausente"), não negativa silenciosa.

        Importa porque uma label esquecida ensinaria ao modelo que não há defeito ali.
        """
        (self.root / "labels/train/foto.txt").unlink()
        self.assert_error("label ausente")

    def test_duplicate_image_between_train_and_val_is_leakage(self) -> None:
        """A mesma imagem (bytes idênticos) em train e val é vazamento entre splits.

        Vazamento infla as métricas de validação e esconde que o modelo não generaliza.
        """
        shutil.copyfile(self.root / "images/train/foto.png", self.root / "images/val/foto.png")
        self.assert_error("vazamento entre splits")

    def test_group_cannot_span_splits(self) -> None:
        """Um mesmo grupo (telhado) em splits diferentes é "vazamento por grupo"."""
        self.write_groups(same_group=True)
        self.assert_error("vazamento por grupo")

    def test_manifest_is_required(self) -> None:
        """Sem ``groups.csv`` o dataset é bloqueado ("groups.csv obrigatório")."""
        (self.root / "groups.csv").unlink()
        self.assert_error("groups.csv obrigatório")

    def test_manifest_requires_complete_membership(self) -> None:
        """Toda imagem precisa aparecer no manifesto; faltando alguma, erro "imagem sem grupo"."""
        with (self.root / "groups.csv").open("w", encoding="utf-8", newline="") as stream:
            stream.write("image,group_id,split\nimages/train/foto.png,telhado_a,train\n")
        self.assert_error("imagem sem grupo")

    def test_empty_splits_do_not_pass(self) -> None:
        """Splits sem nenhuma imagem não passam ("nenhum arquivo de imagem")."""
        for split in ("train", "val", "test"):
            (self.root / f"images/{split}/foto.png").unlink()
            (self.root / f"labels/{split}/foto.txt").unlink()
        self.assert_error("nenhum arquivo de imagem")

    def test_corrupt_image_is_rejected(self) -> None:
        """Arquivo .png que não é imagem é rejeitado ("imagem ilegível")."""
        (self.root / "images/train/foto.png").write_bytes(b"not an image")
        self.assert_error("imagem ilegível")

    def test_png_with_invalid_checksum_is_reported_without_crashing(self) -> None:
        """PNG com CRC do bloco IDAT corrompido vira erro de validação, não exceção.

        O teste localiza o bloco IDAT, lê seu tamanho (4 bytes big-endian antes do nome) e
        inverte 1 bit do CRC que vem depois dos dados.
        """
        image_path = self.root / "images/train/foto.png"
        # Corrompe 1 bit do CRC do bloco IDAT: [tamanho 4B][nome "IDAT"][dados][CRC 4B].
        contents = bytearray(image_path.read_bytes())
        chunk_start = contents.index(b"IDAT")
        chunk_length = int.from_bytes(contents[chunk_start - 4 : chunk_start], "big")
        crc_start = chunk_start + 4 + chunk_length
        contents[crc_start] ^= 1
        image_path.write_bytes(contents)
        self.assert_error("imagem ilegível")

    def test_images_smaller_than_loader_minimum_are_rejected(self) -> None:
        """Imagens com largura ou altura menor que 10 px são rejeitadas (mínimo do carregador YOLO)."""
        for size in ((9, 16), (16, 9)):
            with self.subTest(size=size):
                Image.new("RGB", size, color="red").save(self.root / "images/train/foto.png")
                self.assert_error("pelo menos 10 pixels")

    def test_labels_with_utf8_bom_are_rejected_without_rewriting(self) -> None:
        """Label com BOM UTF-8 é rejeitada e o arquivo NÃO é reescrito pelo validador.

        ATENCAO: o validador só lê; nunca "conserta" arquivos do usuário.
        """
        label_path = self.root / "labels/train/foto.txt"
        for contents in ("0 0.5 0.5 0.2 0.2\n", ""):
            with self.subTest(contents=contents):
                original = b"\xef\xbb\xbf" + contents.encode("utf-8")
                label_path.write_bytes(original)
                self.assert_error("BOM UTF-8")
                self.assertEqual(label_path.read_bytes(), original)

    def test_jpeg_without_final_eoi_is_rejected_without_repair(self) -> None:
        """JPEG truncado ou com lixo após o marcador EOI é rejeitado e não é reparado."""
        image_path = self.root / "images/train/foto.png"
        Image.new("RGB", (16, 16), color="red").save(image_path, format="JPEG")
        complete = image_path.read_bytes()
        for corrupted in (complete[:-2], complete + b"trailing data"):
            with self.subTest(size=len(corrupted)):
                image_path.write_bytes(corrupted)
                self.assert_error("marcador EOI final")
                self.assertEqual(image_path.read_bytes(), corrupted)

    def test_nested_images_directory_cannot_redirect_labels(self) -> None:
        """Subpasta ``images`` dentro de ``images/train`` não pode confundir o mapeamento imagem->label.

        O YOLO troca o ÚLTIMO "/images/" do caminho por "/labels/"; uma pasta aninhada mudaria
        qual label é lida, por isso o validador acusa "caminho de label incompatível".
        """
        image_path = self.root / "images/train/foto.png"
        label_path = self.root / "labels/train/foto.txt"
        (image_path.parent / "images").mkdir()
        (label_path.parent / "images").mkdir()
        image_path.rename(image_path.parent / "images/foto.png")
        label_path.rename(label_path.parent / "images/foto.txt")
        self.assert_error("caminho de label incompatível")

    def test_capitalized_images_root_cannot_hide_label_mapping_failure(self) -> None:
        """Pasta ``Images`` (maiúscula) não pode esconder falha de mapeamento das labels.

        No Windows (sem diferença de maiúsculas) o erro é "caminho de label incompatível"; em
        sistemas case-sensitive a falta de ``images`` já bloqueia antes ("path/images").
        """
        images_root = self.root / "images"
        renamed_root = self.root / "Images"
        # Renomeia images -> Images e aponta os splits do YAML para "Images/<split>".
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
        """Label sem imagem correspondente é rejeitada ("label órfã")."""
        (self.root / "labels/train/sem_foto.txt").write_text("", encoding="utf-8")
        self.assert_error("label órfã")

    def test_same_stem_different_extensions_is_rejected(self) -> None:
        """``foto.png`` e ``foto.jpg`` no mesmo split colidem (ambas usariam ``foto.txt``)."""
        Image.new("RGB", (16, 16), color="white").save(self.root / "images/train/foto.jpg")
        self.assert_error("colisão de stem")

    def test_overlapping_split_directories_are_rejected(self) -> None:
        """train e val apontando para a mesma pasta são rejeitados ("diretórios iguais ou sobrepostos")."""
        self.config["val"] = "images/train"
        self.write_yaml()
        self.assert_error("diretórios iguais ou sobrepostos")

    def test_download_value_is_not_executed(self) -> None:
        """A chave ``download`` do YAML só gera aviso; o código nela nunca é executado.

        ATENCAO: o Ultralytics pode executar o campo ``download``; o validador não pode.
        """
        self.config["download"] = "raise RuntimeError('não executar')"
        self.write_yaml()
        result = self.validate()
        self.assertTrue(result["valid"], result)
        self.assertTrue(any("download" in warning for warning in result["warnings"]))

    def test_validation_does_not_change_dataset_files(self) -> None:
        """Validar não altera nenhum arquivo do dataset (compara bytes antes e depois)."""
        # Fotografa todos os arquivos (caminho -> bytes) antes e depois da validação.
        before = {path.relative_to(self.base): path.read_bytes() for path in self.base.rglob("*") if path.is_file()}
        self.validate()
        after = {path.relative_to(self.base): path.read_bytes() for path in self.base.rglob("*") if path.is_file()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
