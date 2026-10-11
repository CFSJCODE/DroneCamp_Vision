"""Cadastro de classes pela plataforma: edições em taxonomy_overrides.json, nunca em taxonomy.json."""

import json
import threading
import unittest
import urllib.error
import urllib.request

from dronecamp_ia import class_catalog
from dronecamp_ia.io import file_hash
from dronecamp_ia.platform_server import JobRunner, make_handler
from dronecamp_ia.review_render import render_review_package

from tests.test_platform import PlatformFixture


class ClassCatalogTests(PlatformFixture):
    def test_list_merges_taxonomy_with_defaults(self) -> None:
        listing = class_catalog.list_classes(self.config)
        self.assertEqual([item["name"] for item in listing["classes"]], ["telha_quebrada", "residuos_calha"])
        first = listing["classes"][0]
        self.assertEqual((first["display_name"], first["active"], first["in_detector"], first["requires_training"]), ("Telha quebrada", True, True, False))
        self.assertEqual(first["color"], class_catalog.default_color(0))
        self.assertEqual(listing["notice"], class_catalog.TRAINING_NOTICE)
        self.assertEqual(listing["taxonomy_sha256"], file_hash(self.taxonomy_path))

    def test_create_appends_proposed_class_without_touching_taxonomy(self) -> None:
        digest = file_hash(self.taxonomy_path)
        created = class_catalog.create_class(self.config, {"name": "calha_danificada", "display_name": "Calha danificada", "criterion": "Perfil deformado."}, "revisor")
        self.assertEqual((created["id"], created["requires_training"], created["in_detector"], created["source"]), (2, True, False, "proposta"))
        self.assertEqual(file_hash(self.taxonomy_path), digest, "taxonomy.json não muda: revisões e pesos guardam o seu hash.")
        overrides = json.loads(class_catalog.overrides_path(self.config).read_text(encoding="utf-8"))
        self.assertEqual(overrides["proposed"][0]["slug"], "calha_danificada")
        self.assertEqual(overrides["history"][-1]["action"], "create")
        with self.assertRaisesRegex(ValueError, "Já existe"):
            class_catalog.create_class(self.config, {"name": "calha_danificada", "display_name": "Outra"})
        for bad in ({"name": "Calha Danificada", "display_name": "x"}, {"name": "calha-danificada", "display_name": "x"},
                    {"name": "ok_nome"}, {"name": "ok_nome", "display_name": "x", "color": "azul"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                class_catalog.create_class(self.config, bad)

    def test_update_changes_display_name_color_and_active_but_not_slug(self) -> None:
        updated = class_catalog.update_class(self.config, 0, {"display_name": "Telha rompida", "color": "#112233"})
        self.assertEqual((updated["display_name"], updated["color"], updated["name"]), ("Telha rompida", "#112233", "telha_quebrada"))
        with self.assertRaisesRegex(ValueError, "nova versão da taxonomia"):
            class_catalog.update_class(self.config, 0, {"name": "telha_rompida"})
        with self.assertRaises(ValueError):
            class_catalog.update_class(self.config, 0, {"id": 7})
        with self.assertRaises(class_catalog.ClassNotFound):
            class_catalog.update_class(self.config, 99, {"display_name": "x"})
        # Voltar ao nome original e cor vazia limpa a edição em vez de guardá-la.
        restored = class_catalog.update_class(self.config, 0, {"display_name": "Telha quebrada", "color": None})
        self.assertEqual(restored["color"], class_catalog.default_color(0))
        self.assertNotIn("display_name", class_catalog.load_overrides(self.config)["classes"]["0"])

    def test_deactivate_is_soft_delete_and_reversible(self) -> None:
        gone = class_catalog.deactivate_class(self.config, 1)
        self.assertFalse(gone["active"])
        listing = class_catalog.list_classes(self.config)
        self.assertEqual(len(listing["classes"]), 2, "Exclusão lógica: a classe continua listada.")
        self.assertEqual(listing["summary"]["inactive"], 1)
        self.assertTrue(class_catalog.update_class(self.config, 1, {"active": True})["active"])

    def test_served_page_uses_display_name_and_color(self) -> None:
        render_review_package(self.config, self.registry_path)
        class_catalog.update_class(self.config, 1, {"display_name": "Detritos na calha", "color": "#ABCDEF"})
        class_catalog.deactivate_class(self.config, 0)
        payload = json.loads((self.directory / "browser_data.json").read_text(encoding="utf-8"))
        classes = class_catalog.apply_to_page_classes(self.config, payload["classes"])
        self.assertEqual((classes[1]["label"], classes[1]["color"], classes[0]["active"]), ("Detritos na calha", "#ABCDEF", False))
        self.assertEqual(payload["classes"][1]["label"], "Resíduos em calha", "browser_data.json em disco não muda.")


class ClassRoutesTests(PlatformFixture):
    def setUp(self) -> None:
        super().setUp()
        from http.server import ThreadingHTTPServer
        render_review_package(self.config, self.registry_path)
        self.token = "token-de-teste"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.config, None, self.token, JobRunner(self.root)))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def call(self, method: str, path: str, body: dict | None = None, token: str | None = None) -> tuple[int, dict]:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method=method)
        request.add_header("Content-Type", "application/json")
        if token:
            request.add_header("X-DroneCamp-Token", token)
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read() or b"{}")

    def test_crud_routes(self) -> None:
        status, listing = self.call("GET", "/api/classes")
        self.assertEqual((status, len(listing["classes"])), (200, 2))
        self.assertEqual(self.call("POST", "/api/classes", {"name": "ralo_obstruido", "display_name": "Ralo obstruído"})[0], 403)
        status, created = self.call("POST", "/api/classes", {"name": "ralo_obstruido", "display_name": "Ralo obstruído", "author": "revisor"}, self.token)
        self.assertEqual((status, created["id"], created["requires_training"]), (201, 2, True))
        status, error = self.call("POST", "/api/classes", {"name": "Ralo", "display_name": "x"}, self.token)
        self.assertEqual(status, 400)
        self.assertIn("snake_case", error["error"])
        status, updated = self.call("PUT", "/api/classes/2", {"display_name": "Ralo entupido", "color": "#00ff00"}, self.token)
        self.assertEqual((status, updated["display_name"], updated["color"]), (200, "Ralo entupido", "#00FF00"))
        self.assertEqual(self.call("PUT", "/api/classes/2", {"display_name": "x"})[0], 403)
        self.assertEqual(self.call("PUT", "/api/classes/42", {"display_name": "x"}, self.token)[0], 404)
        self.assertEqual(self.call("DELETE", "/api/classes/2", token=self.token)[0], 200)
        self.assertEqual(self.call("DELETE", "/api/classes/2")[0], 403)
        self.assertEqual(self.call("DELETE", "/api/classes/42", token=self.token)[0], 404)
        listing = self.call("GET", "/api/classes")[1]
        self.assertEqual([(item["id"], item["active"]) for item in listing["classes"]], [(0, True), (1, True), (2, False)])
        # A página servida mostra o nome de exibição editado.
        self.call("PUT", "/api/classes/1", {"display_name": "Detritos na calha"}, self.token)
        with urllib.request.urlopen(self.base + "/") as response:
            page = response.read()
        self.assertIn("Detritos na calha".encode(), page)
        self.assertIn(class_catalog.TRAINING_NOTICE.encode(), page)


if __name__ == "__main__":
    unittest.main()
