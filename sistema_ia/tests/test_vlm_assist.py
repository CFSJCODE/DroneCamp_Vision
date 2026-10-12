"""Apoio do VLM gratuito sem rede: transporte falso, chave fora dos arquivos e campos técnicos em branco.

Módulo coberto: ``dronecamp_ia.vlm_assist`` (descrição das ocorrências por um modelo de
visão gratuito do OpenRouter e rascunho de apoio ao laudo). Funções testadas:
``load_vlm_settings``, ``OpenRouterClient``, ``list_free_vision_models``, ``build_messages``,
``prompt_digest``, ``parse_answer``, ``crop_box``, ``describe_occurrences`` e
``draft_report_support``.

Como rodar (na pasta ``sistema_ia``, com o venv do projeto):
    python -m unittest tests.test_vlm_assist
    python -m unittest tests.test_vlm_assist.DescribeOccurrencesTests.test_resume_reuses_notes_and_limit_caps_calls

NENHUM teste acessa a rede: ``FakeTransport`` substitui o HTTP, guarda cada chamada
(url, cabeçalhos, corpo JSON) e devolve respostas prontas; ``answer`` monta uma resposta no
formato do OpenRouter. ``KEY`` é uma chave falsa usada para provar que ela nunca vai
parar em arquivo.

O que cada classe de teste prepara:
    - ``SettingsAndClientTests``: nada fixo; cada teste cria configurações/transportes.
    - ``PromptAndParsingTests``: nada fixo; usa a taxonomia real e as respostas ``GOOD``/``NONE``.
    - ``DescribeOccurrencesTests.setUp``: pasta temporária com uma revisão sintética
      (``make_review``: 2 fotos, 3 caixas humanas, 1 sugestão da IA) e o ambiente com a chave.
    - ``ReportSupportTests``: cada teste cria sua pasta temporária com ``make_review``.

ATENCAO: regras que estes testes travam: a chave só vai no cabeçalho Authorization; as
descrições ficam "pendente" até aceite humano; gravidade, diagnóstico e recomendação
ficam SEMPRE em branco (``None``) - isso é do responsável técnico, nunca da IA.

EXTENSAO: para testar uma função nova de vlm_assist, copie
``DescribeOccurrencesTests.test_taxonomy_or_photo_change_is_refused`` (usa a revisão do
``setUp``) e passe um ``FakeTransport([...])`` com as respostas desejadas; para um formato
de resposta novo do modelo, acrescente uma constante ao lado de ``GOOD``/``NONE`` e um caso
em ``test_parse_accepts_fences_prose_and_rejects_unknown_classes``.
"""

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

# ---------------------------------------------------------------------------
# Dados comuns: configuração real, chave falsa e transporte falso
# ---------------------------------------------------------------------------
# Configuração e classes reais do projeto; KEY é falsa e serve para provar que a chave
# nunca é gravada em arquivo.
REAL = load_config()
NAMES = detection_names(load_taxonomy(REAL.taxonomy_path))
KEY = "sk-or-v1-TESTE-nao-e-uma-chave-real-0123456789"


def answer(text: str) -> bytes:
    """Monta o corpo de uma resposta de chat do OpenRouter com o texto dado.

    Parâmetros:
        text (str): conteúdo da mensagem do modelo (ex.: ``GOOD``).
    Retorna: bytes JSON com ``choices[0].message.content`` = text e modelo
    ``fake/vision-model:free``.
    """
    return json.dumps({"id": "x", "model": "fake/vision-model:free", "choices": [{"message": {"content": text},
                       "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode("utf-8")


class FakeTransport:
    """Guarda cada chamada e responde pela ordem da lista (reaproveita a última).

    Substitui o transporte HTTP do ``OpenRouterClient`` nos testes.
    """

    def __init__(self, replies):
        """Guarda as respostas prontas e começa a lista de chamadas vazia.

        Parâmetros:
            replies (list): respostas na ordem; cada item é ``bytes`` (vira status 200) ou
                uma tupla ``(status_http, bytes)``.
        """
        self.replies = list(replies)
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        """Registra a chamada em ``self.calls`` e devolve a próxima resposta.

        Parâmetros:
            url (str): endereço chamado.
            headers (dict): cabeçalhos HTTP.
            body (bytes | None): corpo JSON da requisição.
            timeout (float): ignorado.
        Retorna: tupla ``(status, bytes)``. A última resposta é repetida quando a lista acaba.
        """
        self.calls.append({"url": url, "headers": dict(headers), "body": json.loads(body) if body else None})
        # Consome a lista na ordem, mas mantém sempre a última resposta para repetir.
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return reply if isinstance(reply, tuple) else (200, reply)


def make_review(root: Path, approved: bool = True) -> Path:
    """Registro mínimo com duas fotos sintéticas, caixas humanas e uma sugestão.

    Grava em ``root/data/reviews/teste_vlm``: ``a.jpg`` (classes 0 e 6), ``b.jpg``
    (classe 3), ``registry.json`` (com o hash da taxonomia real) e ``suggestions.json``
    (uma sugestão classe 2 na foto b).

    Parâmetros:
        root (Path): pasta temporária que faz papel de ``sistema_ia``.
        approved (bool): se False, só a foto a fica aprovada pelo humano.
    Retorna: Path do ``registry.json``.
    """
    review = root / "data" / "reviews" / "teste_vlm"
    review.mkdir(parents=True)
    images = []
    # Duas fotos 320x240; cada caixa é (class_id, [x1, y1, x2, y2] em px).
    # Foto a: telha_quebrada (0) e residuos_calha (6); foto b: reparo_telha (3).
    for index, (name, boxes) in enumerate((("a.jpg", [(0, [10, 10, 60, 50]), (6, [100, 120, 220, 180])]), ("b.jpg", [(3, [5, 5, 15, 15])]))):
        path = review / name
        Image.new("RGB", (320, 240), (40 + 50 * index, 90, 140)).save(path, quality=90)
        images.append({"image_sha256": file_hash(path), "filename": name, "pages": [1], "width": 320, "height": 240,
                       "status": "positive", "human_approved": approved or index == 0, "human_reviewer": "001",
                       "human_notes": "nota humana" if index == 0 else "", "building_group": "g", "scene_group": f"s{index}",
                       "source_path": str(path), "boxes": [{"class_id": c, "bbox_xyxy": b, "id": f"h{index}{j}"} for j, (c, b) in enumerate(boxes)]})
    # Registro com o hash da taxonomia real e sugestões da IA ligadas ao hash do registro.
    registry = {"schema_version": 1, "version": "teste_vlm", "taxonomy_sha256": file_hash(REAL.taxonomy_path), "images": images}
    write_json(review / "registry.json", registry)
    write_json(review / "suggestions.json", {"schema_version": 1, "registry_sha256": file_hash(review / "registry.json"),
                                             "images": {images[1]["image_sha256"]: [{"class_id": 2, "bbox_xyxy": [200, 100, 260, 160],
                                                                                    "confidence": 0.3, "id": "s001"}]}})
    return review / "registry.json"


# Respostas prontas do modelo: GOOD é JSON puro; NONE vem dentro de cerca ```json.
# EXTENSAO: novo formato de resposta a testar: acrescente outra constante aqui.
GOOD = '{"descricao": "Telha com ruptura visível no corpo.", "classe_sugerida": "telha_quebrada", "evidencias": ["borda irregular", "fundo escuro"], "legibilidade": "boa"}'
NONE = '```json\n{"descricao": "Superfície uniforme, sem alteração visível.", "classe_sugerida": "nenhuma", "evidencias": [], "legibilidade": "limitada"}\n```'


# ---------------------------------------------------------------------------
# Configuração e cliente HTTP
# ---------------------------------------------------------------------------
class SettingsAndClientTests(unittest.TestCase):
    """Testes de configuração (chave/modelo) e do cliente HTTP do OpenRouter."""
    def test_missing_key_fails_before_any_network_call(self):
        """Sem chave, o cliente falha (VLMUnavailable) antes de qualquer chamada de rede."""
        settings = load_vlm_settings(REAL, environ={})
        self.assertIsNone(settings.api_key)
        with self.assertRaisesRegex(VLMUnavailable, ENV_KEY):
            OpenRouterClient(settings, FakeTransport([answer("x")]))

    def test_key_comes_from_environment_or_dotenv_and_is_not_described(self):
        """A chave vem do ambiente ou do ``.env`` e nunca aparece em ``settings.describe()``.

        Também confere que chave com espaço é recusada ("formato").
        """
        self.assertEqual(load_vlm_settings(REAL, environ={ENV_KEY: KEY}).api_key, KEY)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            # .env com comentário, "export" e aspas, como o usuário costuma escrever.
            (root / ".env").write_text(f"# chave local\nexport {ENV_KEY}='{KEY}'\nOPENROUTER_MODEL=fake/model:free\n", encoding="utf-8")
            settings = load_vlm_settings(replace(REAL, root=root), environ={})
            self.assertEqual((settings.api_key, settings.model), (KEY, "fake/model:free"))
            self.assertNotIn(KEY, json.dumps(settings.describe()))
            self.assertTrue(settings.describe()["api_key_present"])
        with self.assertRaisesRegex(ValueError, "formato"):
            load_vlm_settings(REAL, environ={ENV_KEY: "chave com espaço"})

    def test_http_errors_become_unavailable(self):
        """Erros HTTP (401, 429, 500) e corpo com "error" viram VLMUnavailable com mensagem clara.

        EXTENSAO: novo status HTTP com mensagem própria: acrescente (status, trecho) na tupla do laço.
        """
        settings = load_vlm_settings(REAL, environ={ENV_KEY: KEY})
        for status, pattern in ((401, "recusou a chave"), (429, "Limite de uso"), (500, "HTTP 500")):
            client = OpenRouterClient(settings, FakeTransport([(status, b'{"error": {"message": "x"}}')]))
            with self.assertRaisesRegex(VLMUnavailable, pattern):
                client.chat([{"role": "user", "content": "oi"}])
        client = OpenRouterClient(settings, FakeTransport([b'{"error": {"message": "sem modelo"}}']))
        with self.assertRaisesRegex(VLMUnavailable, "sem modelo"):
            client.chat([{"role": "user", "content": "oi"}])

    def test_free_vision_models_are_filtered(self):
        """Do catálogo, só ficam modelos gratuitos (preço 0) que aceitam imagem."""
        catalog = {"data": [
            {"id": "a/vision:free", "name": "A", "pricing": {"prompt": "0", "completion": "0"}, "architecture": {"input_modalities": ["text", "image"]}},
            {"id": "b/text:free", "pricing": {"prompt": "0", "completion": "0"}, "architecture": {"input_modalities": ["text"]}},
            {"id": "c/paid-vision", "pricing": {"prompt": "0.001", "completion": "0.002"}, "architecture": {"input_modalities": ["text", "image"]}},
        ]}
        transport = FakeTransport([json.dumps(catalog).encode("utf-8")])
        rows = list_free_vision_models(load_vlm_settings(REAL, environ={ENV_KEY: KEY}), transport)
        self.assertEqual([row["id"] for row in rows], ["a/vision:free"])
        self.assertTrue(transport.calls[0]["url"].endswith("/models"))


# ---------------------------------------------------------------------------
# Prompt, leitura de respostas e recorte
# ---------------------------------------------------------------------------
class PromptAndParsingTests(unittest.TestCase):
    """Testes da montagem do prompt, da leitura das respostas e do recorte da imagem."""
    def test_messages_carry_taxonomy_and_image_without_severity_request(self):
        """O prompt traz as classes da taxonomia e a imagem em base64, e proíbe estimar gravidade."""
        messages = build_messages(load_taxonomy(REAL.taxonomy_path), b"jpegbytes")
        system = messages[0]["content"]
        self.assertIn("Não estime gravidade", system)
        parts = messages[1]["content"]
        self.assertIn("telha_quebrada", parts[0]["text"])
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(len(prompt_digest(load_taxonomy(REAL.taxonomy_path))), 64)

    def test_parse_accepts_fences_prose_and_rejects_unknown_classes(self):
        """A leitura aceita JSON cercado por ``` ou por texto, e troca classe desconhecida por "incerto"."""
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
        """O recorte tem margem, tamanho mínimo, lado máximo, fica dentro da foto e recusa caixa fora."""
        image = Image.new("RGB", (2000, 1500))
        jpeg, crop = crop_box(image, [500, 500, 520, 520], margin=0.3, max_side=768, min_side=96)
        self.assertGreaterEqual(crop[2] - crop[0], 96)
        with Image.open(__import__("io").BytesIO(jpeg)) as decoded:
            self.assertLessEqual(max(decoded.size), 768)
        _, big = crop_box(image, [0, 0, 1800, 1400], margin=0.3, max_side=768, min_side=96)
        self.assertEqual(big, [0, 0, 2000, 1500])
        with self.assertRaises(ValueError):
            crop_box(image, [2100, 2100, 2200, 2200], 0.3, 768, 96)


# ---------------------------------------------------------------------------
# describe_occurrences (notas do VLM)
# ---------------------------------------------------------------------------
class DescribeOccurrencesTests(unittest.TestCase):
    """Testes de ``describe_occurrences``, que grava as notas do VLM ao lado do registro."""
    def setUp(self):
        """Cria a pasta temporária, o config com ``root`` nela, a revisão sintética e o ambiente com a chave.

        Retorna: None.
        """
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = replace(REAL, root=self.root)
        self.registry = make_review(self.root)
        self.environ = {ENV_KEY: KEY}

    def tearDown(self):
        """Apaga a pasta temporária. Retorna: None."""
        self.temporary.cleanup()

    def test_notes_are_pending_keyed_to_registry_and_never_contain_the_key(self):
        """Notas ficam pendentes, ligadas ao hash do registro, sem a chave e sem campos técnicos.

        Também confere que o prompt não revela a classe humana e a estatística de concordância.
        """
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
        # Concordância IA x humano por classe: respostas GOOD, NONE, GOOD para as 3 caixas humanas.
        agreement = notes["agreement"]
        self.assertEqual(agreement["human_boxes_described"], 3)
        self.assertEqual(agreement["per_class"]["telha_quebrada"]["agree"], 1)
        self.assertEqual(agreement["per_class"]["residuos_calha"]["none_or_uncertain"], 1)
        self.assertEqual(agreement["per_class"]["reparo_telha"]["confusions"], {"telha_quebrada": 1})

    def test_resume_reuses_notes_and_limit_caps_calls(self):
        """``limit`` limita as chamadas; a rodada seguinte reaproveita o que já foi descrito."""
        # 1ª rodada com limit=1: descreve uma caixa e marca as outras como puladas.
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
        """Sem chave ou com 429 na 1ª chamada nada é gravado; falhas após a 1ª ficam registradas."""
        with self.assertRaises(VLMUnavailable):
            describe_occurrences(self.config, self.registry, transport=FakeTransport([answer(GOOD)]), environ={})
        self.assertFalse((self.registry.parent / NOTES_FILE).exists())
        with self.assertRaises(VLMUnavailable):
            describe_occurrences(self.config, self.registry, transport=FakeTransport([(429, b"{}")]), environ=self.environ)
        self.assertFalse((self.registry.parent / NOTES_FILE).exists())
        # A 1ª chamada dá certo; as seguintes falham com 503 e são contadas como erros.
        transport = FakeTransport([answer(GOOD), (503, b"indisponivel"), (503, b"indisponivel")])
        describe_occurrences(self.config, self.registry, transport=transport, environ=self.environ)
        notes = json.loads((self.registry.parent / NOTES_FILE).read_text(encoding="utf-8"))
        self.assertEqual((notes["described"], notes["errors_this_run"]), (1, 2))

    def test_boxes_of_unapproved_photos_are_proposals_outside_the_agreement(self):
        """Caixas de foto não aprovada viram "proposta" e ficam fora da estatística de concordância."""
        registry = json.loads(self.registry.read_text(encoding="utf-8"))
        registry["images"][1]["human_approved"] = False
        registry["images"][1]["annotator"] = "first_pass_ai"
        write_json(self.registry, registry)
        describe_occurrences(self.config, self.registry, transport=FakeTransport([answer(GOOD)]), environ=self.environ)
        notes = json.loads((self.registry.parent / NOTES_FILE).read_text(encoding="utf-8"))
        entries = [entry for entries in notes["images"].values() for entry in entries]
        self.assertEqual(sorted(entry["source"] for entry in entries), ["humana", "humana", "proposta"])
        self.assertEqual(notes["agreement"]["human_boxes_described"], 2)
        self.assertNotIn("reparo_telha", notes["agreement"]["per_class"])

    def test_changing_the_model_invalidates_resumed_notes(self):
        """Mesmo modelo reaproveita as notas; trocar o modelo descreve tudo de novo."""
        describe_occurrences(self.config, self.registry, transport=FakeTransport([answer(GOOD)]), environ=self.environ)
        same = FakeTransport([answer(NONE)])
        describe_occurrences(self.config, self.registry, transport=same, environ=self.environ)
        self.assertEqual(len(same.calls), 0)
        other = FakeTransport([answer(NONE)])
        describe_occurrences(self.config, self.registry, model="outro/modelo:free", transport=other, environ=self.environ)
        self.assertEqual(len(other.calls), 3)
        notes = json.loads((self.registry.parent / NOTES_FILE).read_text(encoding="utf-8"))
        self.assertEqual(notes["settings"]["model_requested"], "outro/modelo:free")
        self.assertFalse(any(entry.get("reused") for entries in notes["images"].values() for entry in entries))

    def test_taxonomy_or_photo_change_is_refused(self):
        """Registro feito com outra taxonomia é recusado ("Taxonomia mudou")."""
        registry = json.loads(self.registry.read_text(encoding="utf-8"))
        registry["taxonomy_sha256"] = "0" * 64
        write_json(self.registry, registry)
        with self.assertRaisesRegex(ValueError, "Taxonomia mudou"):
            describe_occurrences(self.config, self.registry, transport=FakeTransport([answer(GOOD)]), environ=self.environ)


# ---------------------------------------------------------------------------
# draft_report_support (apoio ao laudo)
# ---------------------------------------------------------------------------
class ReportSupportTests(unittest.TestCase):
    """Testes de ``draft_report_support`` (laudo_apoio.json e laudo_apoio.md)."""
    def test_draft_links_occurrences_to_review_and_leaves_technical_fields_blank(self):
        """O rascunho liga ocorrências à revisão e deixa gravidade, diagnóstico e recomendação em branco.

        ATENCAO: gravidade nunca é inferida; o texto "a definir pelo responsável técnico" é exigido.
        """
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

    def test_registry_from_another_taxonomy_is_refused(self):
        """Registro de outra taxonomia é recusado e nenhum laudo_apoio.json é gravado."""
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = make_review(root)
            data = json.loads(registry.read_text(encoding="utf-8"))
            data["taxonomy_sha256"] = "0" * 64
            write_json(registry, data)
            with self.assertRaisesRegex(ValueError, "Taxonomia mudou"):
                draft_report_support(replace(REAL, root=root), registry)
            self.assertFalse((registry.parent / "laudo_apoio.json").exists())


if __name__ == "__main__":
    unittest.main()
