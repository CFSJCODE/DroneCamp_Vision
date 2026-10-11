"""Importação de mídia da plataforma (``ingest.py``), sem modelo e sem Ultralytics.

Dados sintéticos: fotos PNG/JPG, uma foto com rotação EXIF, um vídeo curto
(OpenCV ou ffmpeg; o teste é pulado sem nenhum dos dois). Confere que a revisão
gerada tem os campos de ``create_review_for_new_images``, que quadros do mesmo
vídeo ficam na mesma cena e que a página é gerada ao lado do registro.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from dronecamp_ia import ingest
from dronecamp_ia.config import ProjectConfig
from dronecamp_ia.ingest import (
    IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, canonicalize_photo, extract_video_frames, ingest_media, media_kind,
    slugify_review_name,
)
from dronecamp_ia.io import file_hash
from dronecamp_ia.review_data import review_image_size


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class IngestFixture(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="dronecamp importação á ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.taxonomy_path = self.root / "configs" / "taxonomy.json"
        write_json(self.taxonomy_path, {"version": "fixture", "classes": [
            {"id": 0, "slug": "telha_quebrada", "label": "Telha quebrada", "phase": 1},
            {"id": 1, "slug": "residuos_calha", "label": "Resíduos em calha", "phase": 1}]})
        self.config = ProjectConfig(root=self.root, model="fixture.pt", requested_model="fixture.pt",
                                    taxonomy_path=self.taxonomy_path, dataset_path=self.root / "dataset.yaml",
                                    device="cpu", prediction={}, training={"epochs": 5, "imgsz": 640, "batch": 2},
                                    pilot={"suggestion_conf": 0.15})
        self.uploads = self.root / "uploads"
        self.uploads.mkdir()
        self.reviews = self.root / "data" / "reviews"

    def photo(self, name: str, size: tuple[int, int] = (24, 18), color=(90, 90, 90), **options) -> Path:
        """Foto cinza com um bloco branco no canto (0..3, 0..3) para conferir rotações."""
        path = self.uploads / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", size, color)
        image.paste((255, 255, 255), (0, 0, 4, 4))
        image.save(path, **options)
        return path

    def video(self, name: str = "voo.mp4", seconds: int = 3, fps: int = 10, size: tuple[int, int] = (64, 48)) -> Path:
        """Vídeo sintético com uma barra que anda a cada quadro (quadros distintos entre si)."""
        path = self.uploads / name
        try:
            import cv2
            import numpy as np
        except ImportError:
            cv2 = None
        if cv2 is not None:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            opened = writer.isOpened()
            if opened:
                for index in range(seconds * fps):
                    frame = np.zeros((size[1], size[0], 3), np.uint8)
                    column = (index * 2) % size[0]
                    frame[:, column:column + 4] = (0, 255, 0)
                    writer.write(frame)
            writer.release()
            if opened and path.is_file() and path.stat().st_size:
                return path
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                       f"testsrc=duration={seconds}:size={size[0]}x{size[1]}:rate={fps}", "-pix_fmt", "yuv420p", str(path)]
            if subprocess.run(command, capture_output=True).returncode == 0 and path.is_file():
                return path
        self.skipTest("Vídeo sintético requer OpenCV com gravador mp4v ou ffmpeg no PATH.")

    def registry(self, result: dict) -> dict:
        return json.loads(Path(result["registry"]).read_text(encoding="utf-8"))


class SlugAndExtensionTests(unittest.TestCase):
    def test_slug_removes_accents_symbols_and_case(self) -> None:
        self.assertEqual(slugify_review_name("Galpão B – Outubro/2026"), "galpao_b_outubro_2026")
        self.assertEqual(slugify_review_name("  Ção  "), "cao")
        self.assertEqual(slugify_review_name("x__--__y"), "x_y")

    def test_slug_starts_with_letter_and_is_bounded_to_60(self) -> None:
        self.assertEqual(slugify_review_name("2026 voo"), "r_2026_voo")
        self.assertEqual(len(slugify_review_name("a" * 80)), 60)
        long_name = slugify_review_name("b" * 59 + "_c")
        self.assertEqual(long_name, "b" * 59, "Corte não deixa '_' no fim.")

    def test_empty_slug_is_rejected(self) -> None:
        for name in ("", "   ", "!!!", "ß"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    slugify_review_name(name)

    def test_extensions_and_media_kind(self) -> None:
        self.assertEqual(VIDEO_EXTENSIONS, {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"})
        self.assertIn(".jpg", IMAGE_EXTENSIONS)
        self.assertEqual((media_kind(Path("a.JPG")), media_kind(Path("b.MOV")), media_kind(Path("c.txt"))),
                         ("photo", "video", None))


class PhotoIngestTests(IngestFixture):
    def test_photos_are_frozen_by_hash_with_review_fields_and_page(self) -> None:
        png = self.photo("a.png")
        jpg = self.photo("b.JPG", quality=90)
        result = ingest_media(self.config, [self.uploads], self.reviews / "lote_v1", "galpao_b", uploaded_by="cliente@x")
        self.assertEqual((result["review"], result["images"], result["photos"], result["video_frames"], result["videos"]),
                         ("lote_v1", 2, 2, 0, 0))
        self.assertEqual(result["skipped"], [])
        registry = self.registry(result)
        self.assertEqual(registry["schema_version"], 1)
        self.assertEqual(registry["version"], "lote_v1")
        self.assertEqual(registry["taxonomy_sha256"], file_hash(self.taxonomy_path))
        self.assertEqual(registry["taxonomy_version"], "fixture")
        self.assertEqual(registry["scope"], "midia_importada_pela_plataforma_pendente_de_revisao")
        self.assertIn("created_at", registry)
        self.assertEqual(registry["ingest"], {"tool": "platform-upload", "every_seconds": 2.0, "max_frames": 200,
                                              "uploaded_by": "cliente@x", "notes": ""})
        images = {item["filename"]: item for item in registry["images"]}
        item = images["a.png"]
        digest = file_hash(png)
        self.assertEqual(item["image_sha256"], digest, "Foto sem rotação mantém o hash original.")
        frozen = Path(item["source_path"])
        self.assertEqual(frozen, self.reviews / "lote_v1" / "originals" / f"{digest}.png")
        self.assertTrue(frozen.is_file())
        self.assertEqual(item["original_path"], str(png))
        expected = {"pages": [], "width": 24, "height": 18, "status": "ambiguous", "boxes": [],
                    "annotator": "sem_anotacao_humana", "visual_review_status": "pending_model_suggestions",
                    "technical_status": "pending_human_review", "human_approved": False, "building_group": "galpao_b",
                    "scene_group": f"galpao_b_{digest[:12]}", "severity": None}
        self.assertEqual({key: item[key] for key in expected}, expected)
        self.assertIn("nenhuma caixa revisada", item["notes"])
        self.assertEqual(item["ingest"], {"kind": "photo", "source_file": "a.png", "source_sha256": digest,
                                          "frame_time_s": None, "frame_index": None,
                                          "canonicalized_from_sha256": None, "exif_orientation": None})
        self.assertTrue(images["b.JPG"]["source_path"].endswith(".jpg"), "Extensão congelada em minúsculas.")
        # Fontes: um registro por arquivo enviado.
        self.assertEqual([(source["file"], source["kind"], source["frames_extracted"]) for source in registry["sources"]],
                         [("a.png", "photo", None), ("b.JPG", "photo", None)])
        self.assertEqual(registry["sources"][0]["bytes"], png.stat().st_size)
        self.assertEqual(registry["sources"][1]["sha256"], file_hash(jpg))
        # Página gerada ao lado do registro, presa a esta versão dele.
        directory = Path(result["registry"]).parent
        self.assertEqual(result["page"], str(directory / "index.html"))
        self.assertTrue((directory / "index.html").is_file())
        payload = json.loads((directory / "browser_data.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["registry_sha256"], file_hash(Path(result["registry"])))
        self.assertEqual(len(payload["images"]), 2)
        self.assertTrue(all(image["image_url"].startswith("images/") for image in payload["images"]))

    def test_registry_is_accepted_by_pilot_selection(self) -> None:
        """Mesma leitura que build-pilot-data faz: schema, taxonomia e campos por foto."""
        from dronecamp_ia.pilot import _approved

        self.photo("a.png")
        result = ingest_media(self.config, [self.uploads], self.reviews / "lote_v1", "galpao_b")
        self.assertEqual(_approved(Path(result["registry"]), file_hash(self.taxonomy_path), 2, self.root), [],
                         "Nada aprovado ainda; o registro passa pelas conferências sem erro.")

    def test_exif_rotated_photo_is_canonicalized_with_new_hash(self) -> None:
        exif = Image.Exif()
        exif[274] = 6
        rotated = self.photo("girada.jpg", exif=exif)
        with self.assertRaises(ValueError):
            review_image_size(rotated)  # a foto enviada realmente exige canonicalização
        result = ingest_media(self.config, [rotated], self.reviews / "lote_v1", "galpao_b")
        item = self.registry(result)["images"][0]
        self.assertEqual((item["width"], item["height"]), (18, 24), "Orientação 6 gira 90°.")
        self.assertEqual(item["ingest"]["canonicalized_from_sha256"], file_hash(rotated))
        self.assertEqual(item["ingest"]["exif_orientation"], 6)
        self.assertNotEqual(item["image_sha256"], file_hash(rotated))
        frozen = Path(item["source_path"])
        self.assertEqual(frozen.name, f"{item['image_sha256']}.jpg")
        self.assertEqual(file_hash(frozen), item["image_sha256"])
        self.assertEqual(review_image_size(frozen), (18, 24))
        with Image.open(frozen) as image:
            self.assertGreater(image.getpixel((16, 1))[0], 200, "Bloco branco do canto foi para a direita.")
            self.assertLess(image.getpixel((1, 1))[0], 150)

    def test_canonicalize_photo_contract(self) -> None:
        plain = self.photo("plana.PNG")
        destination = self.root / "congeladas"
        path, info = canonicalize_photo(plain, destination)
        self.assertEqual((path, info), (destination / f"{file_hash(plain)}.png", {}))
        exif = Image.Exif()
        exif[274] = 3
        rotated = self.photo("girada.jpg", exif=exif)
        path, info = canonicalize_photo(rotated, destination)
        self.assertEqual(path, destination / f"{file_hash(path)}.jpg")
        self.assertEqual(info, {"canonicalized_from_sha256": file_hash(rotated), "exif_orientation": 3})
        self.assertEqual(review_image_size(path), (24, 18), "Orientação 3 (180°) mantém as dimensões.")
        # Repetir é idempotente: mesmo arquivo, nada sobrescrito.
        self.assertEqual(canonicalize_photo(plain, destination)[0], destination / f"{file_hash(plain)}.png")

    def test_duplicate_photo_enters_once_and_is_reported(self) -> None:
        first = self.photo("a.png")
        shutil.copyfile(first, self.uploads / "b.png")
        result = ingest_media(self.config, [self.uploads], self.reviews / "lote_v1", "galpao_b")
        self.assertEqual(result["images"], 1)
        self.assertEqual(result["skipped"], [{"file": "b.png", "reason": "foto_repetida", "sha256": file_hash(first),
                                              "duplicate_of": "a.png"}])
        registry = self.registry(result)
        self.assertEqual(len(registry["sources"]), 2, "As duas entradas enviadas ficam registradas.")
        self.assertEqual(len(list((Path(result["registry"]).parent / "originals").iterdir())), 1)

    def test_existing_output_and_invalid_arguments_are_rejected(self) -> None:
        photo = self.photo("a.png")
        output = self.reviews / "lote_v1"
        output.mkdir(parents=True)
        with self.assertRaises(ValueError):
            ingest_media(self.config, [photo], output, "galpao_b")
        for group in ("", "   ", "x" * 101):
            with self.subTest(group=group):
                with self.assertRaises(ValueError):
                    ingest_media(self.config, [photo], self.reviews / "lote_v2", group)
        for options in ({"every_seconds": 0.1}, {"every_seconds": 601}, {"max_frames": 0}, {"max_frames": 2001}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    ingest_media(self.config, [photo], self.reviews / "lote_v2", "galpao_b", **options)
        with self.assertRaises(ValueError):
            ingest_media(self.config, [self.uploads / "nao_existe.png"], self.reviews / "lote_v2", "galpao_b")
        self.assertFalse((self.reviews / "lote_v2").exists(), "Argumentos inválidos não criam pasta.")

    def test_unsupported_files_are_ignored(self) -> None:
        self.photo("fotos/a.png")
        (self.uploads / "fotos" / "notas.txt").write_text("campanha", encoding="utf-8")
        (self.uploads / "fotos" / "planilha.xlsx").write_bytes(b"x")
        result = ingest_media(self.config, [self.uploads / "fotos"], self.reviews / "lote_v1", "galpao_b")
        self.assertEqual((result["images"], result["skipped"]), (1, []), "Em pastas, outros arquivos passam em silêncio.")
        # Arquivo informado explicitamente com extensão não suportada é relatado; só ele, recusa.
        animation = self.uploads / "anim.gif"
        animation.write_bytes(b"GIF89a")
        result = ingest_media(self.config, [self.uploads / "fotos" / "a.png", animation], self.reviews / "lote_v2", "galpao_b")
        self.assertEqual(result["skipped"], [{"file": "anim.gif", "reason": "extensao_nao_suportada"}])
        with self.assertRaises(ValueError):
            ingest_media(self.config, [animation], self.reviews / "lote_v3", "galpao_b")
        self.assertFalse((self.reviews / "lote_v3").exists())

    def test_unreadable_photo_is_skipped_and_all_unreadable_marks_incomplete(self) -> None:
        self.photo("a.png")
        (self.uploads / "quebrada.jpg").write_bytes(b"isto nao e um jpeg")
        result = ingest_media(self.config, [self.uploads], self.reviews / "lote_v1", "galpao_b")
        self.assertEqual(result["images"], 1)
        self.assertEqual([(item["file"], item["reason"]) for item in result["skipped"]], [("quebrada.jpg", "arquivo_ilegivel")])
        with self.assertRaises(ValueError):
            ingest_media(self.config, [self.uploads / "quebrada.jpg"], self.reviews / "lote_v2", "galpao_b")
        self.assertTrue((self.reviews / "lote_v2" / "INCOMPLETO.txt").is_file())


class VideoIngestTests(IngestFixture):
    def test_video_frames_share_the_scene_and_are_sampled_every_second(self) -> None:
        video = self.video()
        result = ingest_media(self.config, [video], self.reviews / "voo_v1", "galpao_b", every_seconds=1.0,
                              campaign_notes="voo de teste")
        self.assertEqual((result["images"], result["photos"], result["video_frames"], result["videos"]), (3, 0, 3, 1))
        self.assertEqual(result["skipped"], [])
        registry = self.registry(result)
        frames = sorted(registry["images"], key=lambda item: item["ingest"]["frame_time_s"])
        self.assertEqual([item["ingest"]["frame_time_s"] for item in frames], [0.0, 1.0, 2.0])
        self.assertEqual([item["ingest"]["frame_index"] for item in frames], [0, 1, 2])
        self.assertEqual([item["filename"] for item in frames], ["voo_t0.0.jpg", "voo_t1.0.jpg", "voo_t2.0.jpg"])
        digest = file_hash(video)
        self.assertEqual({item["scene_group"] for item in frames}, {f"galpao_b_{digest[:12]}"}, "Mesma cena: mesmo vídeo.")
        self.assertEqual({item["ingest"]["kind"] for item in frames}, {"video_frame"})
        self.assertEqual({item["ingest"]["source_file"] for item in frames}, {"voo.mp4"})
        self.assertEqual({item["ingest"]["source_sha256"] for item in frames}, {digest})
        self.assertEqual({item["original_path"] for item in frames}, {str(video)})
        for item in frames:
            frozen = Path(item["source_path"])
            self.assertEqual(frozen.name, f"{item['image_sha256']}.jpg")
            self.assertEqual(review_image_size(frozen), (item["width"], item["height"]))
            self.assertEqual((item["status"], item["boxes"], item["human_approved"], item["building_group"]),
                             ("ambiguous", [], False, "galpao_b"))
        source = registry["sources"][0]
        self.assertEqual((source["file"], source["kind"], source["sha256"], source["frames_extracted"], source["every_seconds_used"]),
                         ("voo.mp4", "video", digest, 3, 1.0))
        self.assertEqual(source["bytes"], video.stat().st_size)
        self.assertEqual(registry["ingest"]["notes"], "voo de teste")
        directory = Path(result["registry"]).parent
        self.assertEqual([path.name for path in directory.iterdir() if path.name.startswith("quadros_")], [],
                         "Pasta temporária dos quadros é apagada.")
        self.assertEqual(len(list((directory / "originals").iterdir())), 3)
        self.assertTrue((directory / "index.html").is_file())
        self.assertEqual(len(json.loads((directory / "browser_data.json").read_text(encoding="utf-8"))["images"]), 3)

    def test_max_frames_increases_the_interval_uniformly(self) -> None:
        video = self.video()
        result = ingest_media(self.config, [video], self.reviews / "voo_v1", "galpao_b", every_seconds=1.0, max_frames=2)
        self.assertEqual(result["video_frames"], 2)
        registry = self.registry(result)
        self.assertEqual(sorted(item["ingest"]["frame_time_s"] for item in registry["images"]), [0.0, 1.5])
        self.assertEqual(registry["sources"][0]["every_seconds_used"], 1.5)
        self.assertEqual(registry["ingest"]["max_frames"], 2)

    def test_extract_video_frames_contract_and_limits(self) -> None:
        video = self.video()
        output = self.root / "quadros"
        frames = extract_video_frames(video, output, every_seconds=2.0)
        self.assertEqual([(frame["path"].name, frame["time_s"], frame["frame_index"], frame["every_seconds_used"]) for frame in frames],
                         [("voo_t0.0.jpg", 0.0, 0, 2.0), ("voo_t2.0.jpg", 2.0, 1, 2.0)])
        self.assertTrue(all(frame["path"].is_file() and frame["path"].parent == output for frame in frames))
        with Image.open(frames[0]["path"]) as image:
            self.assertEqual((image.format, image.size), ("JPEG", (64, 48)))
        with self.assertRaises(ValueError):
            extract_video_frames(video, self.root / "outra", every_seconds=0.1)
        with self.assertRaises(ValueError):
            extract_video_frames(video, self.root / "outra", max_frames=0)
        with self.assertRaises(ValueError):
            extract_video_frames(video, output, every_seconds=2.0)  # nunca sobrescreve um quadro

    def test_ffmpeg_fallback_follows_the_same_contract(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("ffmpeg ausente do PATH.")
        video = self.video()
        frames = ingest._extract_with_ffmpeg(video, self.root / "ff", 1.0, 200, ffmpeg)
        self.assertEqual([(frame["path"].name, frame["time_s"], frame["frame_index"]) for frame in frames],
                         [("voo_t0.0.jpg", 0.0, 0), ("voo_t1.0.jpg", 1.0, 1), ("voo_t2.0.jpg", 2.0, 2)])
        resampled = ingest._extract_with_ffmpeg(video, self.root / "ff2", 1.0, 2, ffmpeg)
        self.assertEqual([(frame["time_s"], frame["every_seconds_used"]) for frame in resampled], [(0.0, 1.5), (1.5, 1.5)])
        try:
            import cv2  # noqa: F401
        except ImportError:
            return
        # Com OpenCV disponível, os dois caminhos escolhem os mesmos quadros.
        reference = ingest._extract_with_opencv(video, self.root / "cv", 1.0, 200)
        for mine, theirs in zip(frames, reference):
            with Image.open(mine["path"]) as left, Image.open(theirs["path"]) as right:
                self.assertEqual(left.size, right.size)
                self.assertLess(abs(left.getpixel((32, 10))[1] - right.getpixel((32, 10))[1]), 60, mine["path"].name)

    def test_duplicate_video_is_not_decoded_twice(self) -> None:
        video = self.video()
        shutil.copyfile(video, self.uploads / "voo_copia.mp4")
        result = ingest_media(self.config, [self.uploads], self.reviews / "voo_v1", "galpao_b", every_seconds=1.0)
        self.assertEqual((result["video_frames"], result["videos"]), (3, 2))
        self.assertEqual([(item["file"], item["reason"]) for item in result["skipped"]], [("voo_copia.mp4", "video_repetido")])


class CliTests(IngestFixture):
    def test_import_media_command_prints_report_and_output_folder(self) -> None:
        from dronecamp_ia.cli import main

        config_path = self.root / "configs" / "project.yaml"
        config_path.write_text(
            "schema_version: 1\ntask: detect\nmodel: fixture.pt\nrequested_model: fixture.pt\n"
            "taxonomy: configs/taxonomy.json\ndataset: dataset.yaml\ndevice: cpu\n"
            "prediction: {imgsz: 640, conf: 0.25, iou: 0.7, nms: false, max_det: 300}\n"
            "training: {epochs: 1, imgsz: 640, batch: 2}\npilot: {suggestion_conf: 0.15}\n", encoding="utf-8")
        self.addCleanup(os.environ.pop, "DRONECAMP_CONFIG", None)
        self.photo("a.png")
        output = self.reviews / "cli_v1"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(["--config", str(config_path), "import-media", "--source", str(self.uploads), "--output", str(output),
                         "--group", "galpao_b", "--every-seconds", "1", "--max-frames", "5", "--notes", "via cli"])
        self.assertEqual(code, 0, stdout.getvalue())
        text, _, footer = stdout.getvalue().rpartition("Execução salva em: ")
        self.assertEqual(footer.strip(), str(output.resolve()))
        report = json.loads(text)
        self.assertEqual((report["review"], report["images"], report["skipped"]), ("cli_v1", 1, []))
        self.assertEqual(self.registry(report)["ingest"], {"tool": "platform-upload", "every_seconds": 1.0, "max_frames": 5,
                                                           "uploaded_by": None, "notes": "via cli"})
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--config", str(config_path), "import-media", "--source", str(self.uploads),
                                   "--output", str(output), "--group", "galpao_b"]), 2, "Saída existente: código 2.")


if __name__ == "__main__":
    unittest.main()
