"""Apoio do VLM gratuito sem rede: transporte falso, chave fora dos arquivos e campos técnicos em branco."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from PIL import Image

from dronecamp_ia.config import detection_names, load_config, load_taxonomy
from dronecamp_ia.io import file_hash, write_json
from dronecamp_ia.vlm_assist import (ENV_KEY, NOTES_FILE, OpenRouterClient, VLMUnavailable, build_messages, crop_box,
                                     describe_occurrences, draft_report_support, list_free_vision_models,
                                     load_vlm_settings, parse_answer, prompt_digest)

REAL = load_config()
NAMES = detection_names(load_taxonomy(REAL.taxonomy_path))
KEY = "sk-or-v1-TESTE-nao-e-uma-chave-real-0123456789"


def answer(text: str) -> bytes:
    return json.dumps({"id": "x", "model": "fake/vision-model:free", "choices": [{"message": {"content": text},
                       "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode("utf-8")


class FakeTransport:
    """Guarda cada chamada e responde pela ordem da lista (reaproveita a última)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": dict(headers), "body": json.loads(body) if body else None})
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return reply if isinstance(reply, tuple) else (200, reply)


def make_review(root: Path, approved: bool = True) -> Path:
    """Registro mínimo com duas fotos sintéticas, caixas humanas e uma sugestão."""
    review = root / "data" / "reviews" / "teste_vlm"
    review.mkdir(parents=True)
    images = []
    for index, (name, boxes) in enumerate((("a.jpg", [(0, [10, 10, 60, 50]), (6, [100, 120, 220, 180])]), ("b.jpg", [(3, [5, 5, 15, 15])]))):
        path = review / name
        Image.new("RGB", (320, 240), (40 + 50 * index, 90, 140)).save(path, quality=90)
        images.append({"image_sha256": file_hash(path), "filename": name, "pages": [1], "width": 320, "height": 240,
                       "status": "positive", "human_approved": approved or index == 0, "human_reviewer": "001",
                       "human_notes": "nota humana" if index == 0 else "", "building_group": "g", "scene_group": f"s{index}",
                       "source_path": str(path), "boxes": [{"class_id": c, "bbox_xyxy": b, "id": f"h{index}{j}"} for j, (c, b) in enumerate(boxes)]})
    registry = {"schema_version": 1, "version": "teste_vlm", "taxonomy_sha256": file_hash(REAL.taxonomy_path), "images": images}
    write_json(review / "registry.json", registry)
    write_json(review / "suggestions.json", {"schema_version": 1, "registry_sha256": file_hash(review / "registry.json"),
                                             "images": {images[1]["image_sha256"]: [{"class_id": 2, "bbox_xyxy": [200, 100, 260, 160],
                                                                                    "confidence": 0.3, "id": "s001"}]}})
    return review / "registry.json"


GOOD = '{"descricao": "Telha com ruptura visível no corpo.", "classe_sugerida": "telha_quebrada", "evidencias": ["borda irregular", "fundo escuro"], "legibilidade": "boa"}'
NONE = '```json\n{"descricao": "Superfície uniforme, sem alteração visível.", "classe_sugerida": "nenhuma", "evidencias": [], "legibilidade": "limitada"}\n```'


class SettingsAndClientTests(unittest.TestCase):
    def test_missing_key_fails_before_any_network_call(self):
        settings = load_vlm_settings(REAL, environ={})
        self.assertIsNone(settings.api_key)
        with self.assertRaisesRegex(VLMUnavailable, ENV_KEY):
            OpenRouterClient(settings, FakeTransport([answer("x")]))

    def test_key_comes_from_environment_or_dotenv_and_is_not_described(self):
        self.assertEqual(load_vlm_settings(REAL, environ={ENV_KEY: KEY}).api_key, KEY)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".env").write_text(f"# chave local\nexport {ENV_KEY}='{KEY}'\nOPENROUTER_MODEL=fake/model:free\n", encoding="utf-8")
            settings = load_vlm_settings(replace(REAL, root=root), environ={})
            self.assertEqual((settings.api_key, settings.model), (KEY, "fake/model:free"))
            self.assertNotIn(KEY, json.dumps(settings.describe()))
            self.assertTrue(settings.describe()["api_key_present"])
        with self.assertRaisesRegex(ValueError, "formato"):
            load_vlm_settings(REAL, environ={ENV_KEY: "chave com espaço"})

    def test_http_errors_become_unavailable(self):
        settings = load_vlm_settings(REAL, environ={ENV_KEY: KEY})
        for status, pattern in ((401, "recusou a chave"), (429, "Limite de uso"), (500, "HTTP 500")):
            client = OpenRouterClient(settings, FakeTransport([(status, b'{"error": {"message": "x"}}')]))
            with self.assertRaisesRegex(VLMUnavailable, pattern):
                client.chat([{"role": "user", "content": "oi"}])
        client = OpenRouterClient(settings, FakeTransport([b'{"error": {"message": "sem modelo"}}']))
        with self.assertRaisesRegex(VLMUnavailable, "sem modelo"):
            client.chat([{"role": "user", "content": "oi"}])

    def test_free_vision_models_are_filtered(self):
        catalog = {"data": [
            {"id": "a/vision:free", "name": "A", "pricing": {"prompt": "0", "completion": "0"}, "architecture": {"input_modalities": ["text", "image"]}},
            {"id": "b/text:free", "pricing": {"prompt": "0", "completion": "0"}, "architecture": {"input_modalities": ["text"]}},
            {"id": "c/paid-vision", "pricing": {"prompt": "0.001", "completion": "0.002"}, "architecture": {"input_modalities": ["text", "image"]}},
        ]}
        transport = FakeTransport([json.dumps(catalog).encode("utf-8")])
        rows = list_free_vision_models(load_vlm_settings(REAL, environ={ENV_KEY: KEY}), transport)
        self.assertEqual([row["id"] for row in rows], ["a/vision:free"])
        self.assertTrue(transport.calls[0]["url"].endswith("/models"))


class PromptAndParsingTests(unittest.TestCase):
    def test_messages_carry_taxonomy_and_image_without_severity_request(self):
        messages = build_messages(load_taxonomy(REAL.taxonomy_path), b"jpegbytes")
        system = messages[0]["content"]
        self.assertIn("Não estime gravidade", system)
        parts = messages[1]["content"]
        self.assertIn("telha_quebrada", parts[0]["text"])
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(len(prompt_digest(load_taxonomy(REAL.taxonomy_path))), 64)

    def test_parse_accepts_fences_prose_and_rejects_unknown_classes(self):
        self.assertEqual(parse_answer(GOOD, NAMES)["class_guess"], "telha_quebrada")
        fenced = parse_answer(NONE, NAMES)
        self.assertEqual((fenced["class_guess"], fenced["legibility"]), ("nenhuma", "limitada"))
        prose = parse_answer("Claro! Aqui está: " + GOOD + " Espero ter ajudado.", NAMES)
        self.assertTrue(prose["parsed"])
        unknown = parse_answer('{"descricao": "x", "classe_sugerida": "fissura_inventada", "legibilidade": "ótima"}', NAMES)
        self.assertEqual((unknown["class_guess"], unknown["legibility"]), ("incerto", None))
        broken = parse_answer("não consigo ver nada", NAMES)
        self.assertEqual((broken["class_guess"], broken["parsed"]), ("incerto", False))

    def test_crop_has_margin_minimum_size_and_maximum_side(self):
        image = Image.new("RGB", (2000, 1500))
        jpeg, crop = crop_box(image, [500, 500, 520, 520], margin=0.3, max_side=768, min_side=96)
        self.assertGreaterEqual(crop[2] - crop[0], 96)
        with Image.open(__import__("io").BytesIO(jpeg)) as decoded:
            self.assertLessEqual(max(decoded.size), 768)
        _, big = crop_box(image, [0, 0, 1800, 1400], margin=0.3, max_side=768, min_side=96)
        self.assertEqual(big, [0, 0, 2000, 1500])
        with self.assertRaises(ValueError):
            crop_box(image, [2100, 2100, 2200, 2200], 0.3, 768, 96)


class DescribeOccurrencesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = replace(REAL, root=self.root)
        self.registry = make_review(self.root)
        self.environ = {ENV_KEY: KEY}

    def tearDown(self):
        self.temporary.cleanup()

    def test_notes_are_pending_keyed_to_registry_and_never_contain_the_key(self):
        transport = FakeTransport([answer(GOOD), answer(NONE), answer(GOOD), answer(GOOD)])
        notes_path = describe_occurrences(self.config, self.registry, include_suggestions=True, transport=transport,
                                          environ=self.environ)
        self.assertEqual(notes_path.name, NOTES_FILE)
        raw = notes_path.read_text(encoding="utf-8")
        self.assertNotIn(KEY, raw)
        notes = json.loads(raw)
        self.assertEqual((notes["total_boxes"], notes["described"], notes["calls_this_run"]), (4, 4, 4))
        self.assertEqual(notes["registry_sha256"], file_hash(self.registry))
        self.assertEqual(notes["human_acceptance"], "pendente")
        entries = [entry for entries in notes["images"].values() for entry in entries]
        self.assertEqual({entry["status"] for entry in entries}, {"pendente"})
        self.assertTrue(all(entry["severity"] is None and entry["recommendation"] is None for entry in entries))
        self.assertEqual([entry["source"] for entry in entries], ["humana", "humana", "humana", "sugestao"])
        self.assertEqual(entries[0]["model_used"], "fake/vision-model:free")
        # A chave vai só no cabeçalho; o prompt não revela a classe humana da caixa.
        self.assertEqual(transport.calls[0]["headers"]["Authorization"], f"Bearer {KEY}")
        self.assertEqual(transport.calls[0]["body"]["model"], "openrouter/free")
        self.assertNotIn("classe atribuída", transport.calls[0]["body"]["messages"][1]["content"][0]["text"])
        agreement = notes["agreement"]
        self.assertEqual(agreement["human_boxes_described"], 3)
        self.assertEqual(agreement["per_class"]["telha_quebrada"]["agree"], 1)
        self.assertEqual(agreement["per_class"]["residuos_calha"]["none_or_uncertain"], 1)
        self.assertEqual(agreement["per_class"]["reparo_telha"]["confusions"], {"telha_quebrada": 1})

    def test_resume_reuses_notes_and_limit_caps_calls(self):
        first = FakeTransport([answer(GOOD)])
        describe_occurrences(self.config, self.registry, limit=1, transport=first, environ=self.environ)
        notes = json.loads((self.registry.parent / NOTES_FILE).read_text(encoding="utf-8"))
        self.assertEqual((notes["calls_this_run"], notes["described"]), (1, 1))
        skipped = [entry for entries in notes["images"].values() for entry in entries if entry.get("skipped")]
        self.assertEqual(len(skipped), 2)
        second = FakeTransport([answer(NONE)])
        describe_occurrences(self.config, self.registry, transport=second, environ=self.environ)
        notes = json.loads((self.registry.parent / NOTES_FILE).read_text(encoding="utf-8"))
        self.assertEqual((notes["calls_this_run"], notes["described"]), (2, 3))
        self.assertEqual(sum(1 for entries in notes["images"].values() for entry in entries if entry.get("reused")), 1)
        self.assertEqual(len(second.calls), 2)

    def test_failures_stop_without_key_and_after_first_call_are_recorded(self):
        with self.assertRaises(VLMUnavailable):
            describe_occurrences(self.config, self.registry, transport=FakeTransport([answer(GOOD)]), environ={})
        self.assertFalse((self.registry.parent / NOTES_FILE).exists())
        with self.assertRaises(VLMUnavailable):
            describe_occurrences(self.config, self.registry, transport=FakeTransport([(429, b"{}")]), environ=self.environ)
        self.assertFalse((self.registry.parent / NOTES_FILE).exists())
        transport = FakeTransport([answer(GOOD), (503, b"indisponivel"), (503, b"indisponivel")])
        describe_occurrences(self.config, self.registry, transport=transport, environ=self.environ)
        notes = json.loads((self.registry.parent / NOTES_FILE).read_text(encoding="utf-8"))
        self.assertEqual((notes["described"], notes["errors_this_run"]), (1, 2))

    def test_taxonomy_or_photo_change_is_refused(self):
        registry = json.loads(self.registry.read_text(encoding="utf-8"))
        registry["taxonomy_sha256"] = "0" * 64
        write_json(self.registry, registry)
        with self.assertRaisesRegex(ValueError, "Taxonomia mudou"):
            describe_occurrences(self.config, self.registry, transport=FakeTransport([answer(GOOD)]), environ=self.environ)


class ReportSupportTests(unittest.TestCase):
    def test_draft_links_occurrences_to_review_and_leaves_technical_fields_blank(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = replace(REAL, root=root)
            registry = make_review(root, approved=False)
            describe_occurrences(config, registry, transport=FakeTransport([answer(GOOD)]), environ={ENV_KEY: KEY})
            output = draft_report_support(config, registry)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual((report["approved_images"], report["images_without_human_approval"]), (1, 1))
            self.assertEqual(len(report["occurrences"]), 2)
            first = report["occurrences"][0]
            self.assertEqual(first["class_slug"], "telha_quebrada")
            self.assertEqual(first["human_notes"], "nota humana")
            self.assertEqual(first["ai_description"]["status"], "pendente")
            self.assertIn("ruptura", first["ai_description"]["text"])
            self.assertTrue(all(occ["severity"] is None and occ["diagnosis"] is None and occ["recommendation"] is None
                                for occ in report["occurrences"]))
            self.assertEqual(report["human_acceptance"], "pendente")
            markdown = (output.parent / "laudo_apoio.md").read_text(encoding="utf-8")
            self.assertIn("Gravidade: a definir pelo responsável técnico", markdown)
            self.assertIn("Descrição da IA (pendente)", markdown)
            self.assertNotIn(KEY, markdown)


if __name__ == "__main__":
    unittest.main()
