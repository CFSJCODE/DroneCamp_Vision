"""Painel Assistente IA sem rede: chave no .env, consumo só com metadados e chat ancorado nos dados.

Módulos cobertos: ``dronecamp_ia.ai_usage`` (registro e agregação do consumo, limites do
OpenRouter), ``dronecamp_ia.ai_panel`` (estado/gravação/remoção da chave, instantâneo do
projeto e chat) e as rotas ``/api/ai/...`` de ``platform_server``.

Como rodar (na pasta ``sistema_ia``, com o venv do projeto):
    python -m unittest tests.test_ai_panel

NENHUM teste acessa a rede: ``FakeTransport`` (de test_vlm_assist) responde no lugar do
OpenRouter. ``KEY`` é falsa e serve para provar que a chave não aparece em respostas,
no registro de consumo nem no instantâneo enviado ao modelo.

ATENCAO: regras travadas aqui: a chave só é gravada depois do teste passar; o registro de
consumo nunca guarda pergunta, resposta ou chave; consumo ausente fica None (não zero); a
consulta de saldo só manda a chave para o endereço oficial do OpenRouter; o chat recusa
conversa malformada e responde com o instantâneo do projeto na mensagem de sistema.
"""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import threading
import unittest
import urllib.error
import urllib.request

from dronecamp_ia import ai_panel, ai_usage
from dronecamp_ia.platform_server import JobRunner, make_handler
from dronecamp_ia.vlm_assist import ENV_KEY, ENV_MODEL, describe_occurrences, load_vlm_settings

from tests.test_vlm_assist import GOOD, KEY, REAL, FakeTransport, answer, make_review

NEW_KEY = "sk-or-v1-OUTRA-chave-falsa-para-teste-9876543210"


def usage_lines(root: Path) -> list[dict]:
    """Linhas gravadas em ``.runtime/ai_usage.jsonl`` (lista vazia se não houver)."""
    path = root / ai_usage.USAGE_FILE
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.is_file() else []


class Base(unittest.TestCase):
    """Pasta temporária no papel de ``sistema_ia`` com a revisão sintética de test_vlm_assist."""
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = replace(REAL, root=self.root)
        self.registry = make_review(self.root)

    def tearDown(self):
        self.temporary.cleanup()


class UsageTests(Base):
    """``ai_usage``: o que é registrado e como é somado."""
    def test_unknown_usage_stays_none_and_zero_stays_zero(self):
        self.assertEqual(ai_usage.completion_usage(None), {"input_tokens": None, "output_tokens": None, "total_tokens": None, "cost_usd": None})
        self.assertEqual(ai_usage.completion_usage({"prompt_tokens": 0, "completion_tokens": -1, "total_tokens": 2.5, "cost": 0, "x": True}),
                         {"input_tokens": 0, "output_tokens": None, "total_tokens": None, "cost_usd": 0.0})

    def test_each_described_box_is_one_metadata_line_without_text_or_key(self):
        transport = FakeTransport([answer(GOOD)])
        describe_occurrences(self.config, self.registry, transport=transport, environ={ENV_KEY: KEY})
        lines = usage_lines(self.root)
        self.assertEqual(len(lines), len(transport.calls))
        self.assertTrue(all(line["feature"] == "describe" and line["status"] == "success" for line in lines))
        self.assertEqual((lines[0]["input_tokens"], lines[0]["output_tokens"], lines[0]["model_used"]), (10, 5, "fake/vision-model:free"))
        raw = (self.root / ai_usage.USAGE_FILE).read_text(encoding="utf-8")
        self.assertNotIn(KEY, raw)
        self.assertNotIn("ruptura", raw)  # texto da resposta do modelo nunca entra

    def test_failures_record_http_status_but_not_the_message(self):
        transport = FakeTransport([(429, b'{"error": "limite privado"}')])
        with self.assertRaises(Exception):
            describe_occurrences(self.config, self.registry, transport=transport, environ={ENV_KEY: KEY})
        line = usage_lines(self.root)[0]
        self.assertEqual((line["status"], line["http_status"]), ("error", 429))
        self.assertNotIn("privado", json.dumps(line))

    def test_local_usage_filters_period_and_keeps_missing_totals_partial(self):
        now = datetime(2026, 10, 12, 12, tzinfo=timezone.utc)
        ai_usage.record_usage(self.root, {"feature": "chat", "provider": "openrouter", "model": "m", "status": "success",
                                          "duration_ms": 1000, "total_tokens": 30, "input_tokens": 20, "output_tokens": 10}, now)
        ai_usage.record_usage(self.root, {"feature": "chat", "provider": "openrouter", "model": "m", "status": "error", "duration_ms": 3000}, now)
        ai_usage.record_usage(self.root, {"feature": "describe", "provider": "openrouter", "model": "m", "status": "success",
                                          "duration_ms": 500, "cost_usd": 0}, now - timedelta(days=10))
        with (self.root / ai_usage.USAGE_FILE).open("a", encoding="utf-8") as stream:
            stream.write("{linha cortada\n")
        week = ai_usage.local_usage(self.root, "7d", now)
        self.assertEqual((week["requests"], week["successful"], week["errors"]), (2, 1, 1))
        self.assertEqual((week["total_tokens"], week["token_reports"], week["cost_usd"]), (30, 1, None))
        self.assertEqual(week["average_duration_ms"], 2000)
        self.assertEqual(week["features"][0]["feature_label"], "Chat do assistente")
        everything = ai_usage.local_usage(self.root, "all", now)
        self.assertEqual((everything["requests"], everything["cost_usd"], everything["cost_reports"]), (3, 0.0, 1))
        self.assertTrue(everything["tracking_since"].startswith("2026-10-02"))
        with self.assertRaises(ValueError):
            ai_usage.local_usage(self.root, "1ano", now)


class ProviderUsageTests(Base):
    """``ai_usage.provider_usage``: só o endpoint oficial, sem inferência."""
    def settings(self, **changes):
        return replace(load_vlm_settings(self.config, environ={ENV_KEY: KEY}), **changes)

    def test_key_limits_survive_credits_needing_management_key(self):
        transport = FakeTransport([json.dumps({"data": {"limit": 10, "limit_remaining": 4, "usage": 6, "usage_daily": 0.5,
                                                        "limit_reset": "monthly", "is_free_tier": True}}).encode(), (403, b"{}")])
        result = ai_usage.provider_usage(self.settings(), transport)
        self.assertEqual((result["state"], result["key_remaining"], result["account_balance"], result["is_free_tier"]), ("available", 4.0, None, True))
        self.assertIn("gerenciamento", result["message"])
        self.assertEqual([call["url"] for call in transport.calls], [f"{ai_usage.OPENROUTER_BASE}/key", f"{ai_usage.OPENROUTER_BASE}/credits"])
        self.assertNotIn(KEY, json.dumps(result))

    def test_balance_is_credits_minus_usage(self):
        transport = FakeTransport([json.dumps({"data": {}}).encode(), json.dumps({"data": {"total_credits": 10, "total_usage": 2.5}}).encode()])
        self.assertEqual(ai_usage.provider_usage(self.settings(), transport)["account_balance"], 7.5)

    def test_other_endpoints_and_missing_key_send_nothing(self):
        transport = FakeTransport([b"{}"])
        self.assertEqual(ai_usage.provider_usage(self.settings(base_url="https://outro.exemplo/v1"), transport)["state"], "unsupported")
        self.assertEqual(ai_usage.provider_usage(self.settings(api_key=None), transport)["state"], "unconfigured")
        self.assertEqual(transport.calls, [])


class SettingsTests(Base):
    """``ai_panel``: a chave vai para o .env só depois do teste e nunca volta."""
    def test_failed_test_saves_nothing(self):
        with self.assertRaisesRegex(ValueError, "nada foi salvo"):
            ai_panel.save_settings(self.config, {"api_key": KEY, "model": "fake/model:free"}, FakeTransport([(401, b"{}")]), environ={})
        self.assertFalse((self.root / ".env").exists())
        self.assertEqual(usage_lines(self.root)[0]["feature"], "connection-test")

    def test_saved_key_keeps_other_lines_and_status_only_shows_hint(self):
        (self.root / ".env").write_text("# minhas variáveis\nOUTRA=1\n", encoding="utf-8")
        transport = FakeTransport([answer("")])
        status = ai_panel.save_settings(self.config, {"api_key": KEY, "model": "fake/model:free"}, transport, environ={})
        self.assertEqual(transport.calls[0]["body"]["max_tokens"], 1)
        env = (self.root / ".env").read_text(encoding="utf-8")
        self.assertIn("OUTRA=1", env)
        self.assertIn(f"{ENV_KEY}={KEY}", env)
        self.assertIn(f"{ENV_MODEL}=fake/model:free", env)
        self.assertEqual((status["configured"], status["key_hint"], status["key_source"], status["model"]), (True, KEY[-4:], ".env", "fake/model:free"))
        self.assertNotIn(KEY, json.dumps(status))
        # Trocar só o modelo mantém a chave; remover tira só a chave.
        ai_panel.save_settings(self.config, {"model": "outro/model:free"}, FakeTransport([answer("ok")]), environ={})
        self.assertIn(KEY, (self.root / ".env").read_text(encoding="utf-8"))
        status = ai_panel.delete_key(self.config, environ={})
        self.assertFalse(status["configured"])
        env = (self.root / ".env").read_text(encoding="utf-8")
        self.assertNotIn(KEY, env)
        self.assertIn("outro/model:free", env)

    def test_system_variable_locks_the_key_and_bad_input_is_refused(self):
        locked = {ENV_KEY: KEY}
        self.assertTrue(ai_panel.ai_status(self.config, environ=locked)["key_locked"])
        with self.assertRaisesRegex(ValueError, "variável de ambiente"):
            ai_panel.save_settings(self.config, {"api_key": NEW_KEY, "model": "m:free"}, FakeTransport([answer("")]), environ=locked)
        with self.assertRaisesRegex(ValueError, "variável de ambiente"):
            ai_panel.delete_key(self.config, environ=locked)
        with self.assertRaisesRegex(ValueError, "formato"):
            ai_panel.save_settings(self.config, {"api_key": "curta demais", "model": "m:free"}, environ={})
        with self.assertRaisesRegex(ValueError, "modelo"):
            ai_panel.save_settings(self.config, {"api_key": KEY, "model": "tem espaço"}, environ={})
        with self.assertRaisesRegex(ValueError, "Cole a chave"):
            ai_panel.save_settings(self.config, {"model": "m:free"}, environ={})


class ChatTests(Base):
    """``ai_panel.project_context`` e ``ai_panel.ask``."""
    def context(self):
        workspace = {"reviews": [{"name": "teste_vlm", "display_name": "Teste VLM", "images": 2, "decided": 1, "suggestions_pending": 1, "current": True}]}
        operations = {"runs": [{"id": "pilot_train_x", "display_name": "Treino Piloto ABCD1234", "state": "completed"}], "comparisons": []}
        classes = {"classes": [{"id": 0, "display_name": "Telha quebrada"}, {"id": 2, "display_name": "Rufo quebrado"}]}
        return ai_panel.project_context(self.config, workspace, operations, classes, self.registry)

    def test_context_counts_boxes_and_pending_suggestions_by_readable_class(self):
        context = self.context()
        opened = context["inspecao_aberta"]
        self.assertEqual(opened["fotos"], 2)
        self.assertEqual(opened["caixas_registradas_por_classe"]["Telha quebrada"], 1)
        self.assertEqual(opened["sugestoes_do_modelo"]["por_classe"], {"Rufo quebrado": 1})
        self.assertEqual(opened["sugestoes_do_modelo"]["situacao"], "pendentes até decisão humana")
        self.assertEqual(context["treinos"][0]["nome"], "Treino Piloto ABCD1234")

    def test_answer_uses_snapshot_and_rules_without_the_key(self):
        transport = FakeTransport([answer("Há **1** sugestão pendente.")])
        result = ai_panel.ask(self.config, [{"role": "user", "content": "Quantas sugestões pendentes?"}], self.context(), transport, environ={ENV_KEY: KEY})
        self.assertEqual(result["answer"], "Há **1** sugestão pendente.")
        system = transport.calls[0]["body"]["messages"][0]
        self.assertEqual(system["role"], "system")
        self.assertIn("Rufo quebrado", system["content"])
        self.assertIn("validação técnica", system["content"])
        self.assertNotIn(KEY, json.dumps(transport.calls[0]["body"]))
        self.assertEqual(usage_lines(self.root)[-1]["feature"], "chat")

    def test_malformed_conversations_and_missing_key_are_refused(self):
        for messages in ([], [{"role": "system", "content": "x"}], [{"role": "user", "content": ""}],
                         [{"role": "assistant", "content": "x"}], [{"role": "user", "content": "x" * 4001}]):
            with self.assertRaises(ValueError):
                ai_panel.ask(self.config, messages, {}, FakeTransport([answer("x")]), environ={ENV_KEY: KEY})
        with self.assertRaisesRegex(ValueError, "IA desativada"):
            ai_panel.ask(self.config, [{"role": "user", "content": "oi"}], {}, FakeTransport([answer("x")]), environ={})


class RouteTests(Base):
    """Rotas ``/api/ai/...``: leitura sem chave e alterações só com o token."""
    def setUp(self):
        super().setUp()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.config, self.registry, "token-de-teste", JobRunner(self.root)))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def request(self, path, body=None, token=None):
        request = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                         method="GET" if body is None else "POST")
        if token:
            request.add_header("X-DroneCamp-Token", token)
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_routes(self):
        (self.root / ".env").write_text(f"{ENV_KEY}={KEY}\n", encoding="utf-8")
        status, body = self.request("/api/ai/status")
        self.assertEqual(status, 200)
        if body["key_source"] == ".env":  # sem a variável do sistema definida na máquina dos testes
            self.assertEqual(body["key_hint"], KEY[-4:])
        self.assertNotIn(KEY, json.dumps(body))
        self.assertEqual(self.request("/api/ai/usage?period=7d")[1]["requests"], 0)
        self.assertEqual(self.request("/api/ai/usage?period=x")[0], 400)
        for path in ("/api/ai/settings", "/api/ai/key/delete", "/api/ai/chat", "/api/ai/provider-usage", "/api/ai/models"):
            self.assertEqual(self.request(path, {})[0], 403, path)
        status, body = self.request("/api/ai/chat", {"messages": [{"role": "tool", "content": "x"}]}, "token-de-teste")
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
