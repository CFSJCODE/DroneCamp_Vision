"""Cadastro de classes pela plataforma: edições em taxonomy_overrides.json, nunca em taxonomy.json.

Módulos cobertos
----------------
- ``dronecamp_ia.class_catalog``: ``list_classes``, ``create_class``, ``update_class``,
  ``deactivate_class``, ``load_overrides``, ``overrides_path``, ``apply_to_page_classes``,
  ``default_color``, ``TRAINING_NOTICE`` e a exceção ``ClassNotFound``.
- ``dronecamp_ia.platform_server``: as rotas ``GET/POST /api/classes`` e
  ``PUT/DELETE /api/classes/<id>`` (via ``make_handler`` + ``JobRunner``) e a página ``/``.
- ``dronecamp_ia.review_render.render_review_package``: gera ``browser_data.json`` usado na página.

Como rodar (na pasta ``sistema_ia``; o import ``tests.test_platform`` exige essa pasta como atual)
-------------------------------------------------------------------------------------------------
- Só este arquivo::

      DRONECAMP_SKIP_E2E=1 python -m unittest tests.test_class_catalog -v

- Um teste só::

      python -m unittest tests.test_class_catalog.ClassCatalogTests.test_deactivate_is_soft_delete_and_reversible -v

O que cada classe prepara
-------------------------
- ``ClassCatalogTests`` herda ``tests.test_platform.PlatformFixture``: pasta temporária com
  ``configs/taxonomy.json`` de 2 classes (0 telha_quebrada, 1 residuos_calha "Resíduos em calha"),
  ``ProjectConfig`` em CPU, uma foto 20x16 em ``data/reference/foto.png`` e a revisão
  ``data/reviews/fixture_v1/registry.json`` (``self.directory`` é essa pasta). Chama as funções
  de ``class_catalog`` diretamente.
- ``ClassRoutesTests``: a mesma fixture, mais a página gerada e um servidor HTTP real em
  ``127.0.0.1`` numa porta livre, com token ``token-de-teste``.

EXTENSAO: para testar uma função nova de ``class_catalog`` (ex.: um campo editável novo,
como "descrição curta"), copie ``test_update_changes_display_name_color_and_active_but_not_slug``,
troque o campo do dict passado a ``update_class`` e as asserções. Para uma rota nova da seção
Classes, acrescente chamadas ``self.call(...)`` em ``ClassRoutesTests`` (ou copie
``test_crud_routes``), sempre testando com e sem ``self.token`` (sem token deve dar 403).

ATENCAO: a regra que estes testes protegem é que ``taxonomy.json`` nunca é alterado pela
plataforma: revisões e pesos guardam o hash dele. Toda edição vai para
``taxonomy_overrides.json`` e acompanha a classe pelo slug, não pelo ID.
"""

import json
import threading
import unittest
import urllib.error
import urllib.request

from dronecamp_ia import class_catalog
from dronecamp_ia.io import file_hash
from dronecamp_ia.platform_server import JobRunner, make_handler
from dronecamp_ia.review_render import render_review_package

# Fixture compartilhada com os testes da plataforma (projeto temporário + revisão de 1 foto).
from tests.test_platform import PlatformFixture


# ---------------------------------------------------------------------------
# Funções de class_catalog chamadas diretamente
# ---------------------------------------------------------------------------

class ClassCatalogTests(PlatformFixture):
    """Regras do cadastro de classes chamando ``class_catalog`` sem servidor.

    Usa a taxonomia de 2 classes da ``PlatformFixture``; vários testes reescrevem
    ``taxonomy.json`` no diretório temporário para simular uma nova versão da taxonomia.
    """

    def test_list_merges_taxonomy_with_defaults(self) -> None:
        """A listagem junta taxonomia + valores padrão (nome de exibição, cor, ativo, no detector).

        Também confere o aviso de treino e o hash da taxonomia que a página usa.
        """
        listing = class_catalog.list_classes(self.config)
        # Ordem e slugs vêm da taxonomia.
        self.assertEqual([item["name"] for item in listing["classes"]], ["telha_quebrada", "residuos_calha"])
        # Sem overrides: exibição = label da taxonomia, ativa, já no detector, sem exigir treino.
        first = listing["classes"][0]
        self.assertEqual((first["display_name"], first["active"], first["in_detector"], first["requires_training"]), ("Telha quebrada", True, True, False))
        self.assertEqual(first["color"], class_catalog.default_color(0))
        self.assertEqual(listing["notice"], class_catalog.TRAINING_NOTICE)
        self.assertEqual(listing["taxonomy_sha256"], file_hash(self.taxonomy_path))

    def test_create_appends_proposed_class_without_touching_taxonomy(self) -> None:
        """Criar classe gera uma "proposta" com o próximo ID, sem alterar taxonomy.json.

        Também cobre: histórico no overrides, recusa de slug repetido e validações de
        entrada (slug em snake_case, nome de exibição obrigatório, cor hexadecimal).
        """
        # 1. Hash antes, para provar que taxonomy.json não muda.
        digest = file_hash(self.taxonomy_path)
        created = class_catalog.create_class(self.config, {"name": "calha_danificada", "display_name": "Calha danificada", "criterion": "Perfil deformado."}, "revisor")
        # 2. Próximo ID (2), exige treino e ainda não está nos pesos.
        self.assertEqual((created["id"], created["requires_training"], created["in_detector"], created["source"]), (2, True, False, "proposta"))
        self.assertEqual(file_hash(self.taxonomy_path), digest, "taxonomy.json não muda: revisões e pesos guardam o seu hash.")
        # 3. A proposta e a ação "create" ficam em taxonomy_overrides.json.
        overrides = json.loads(class_catalog.overrides_path(self.config).read_text(encoding="utf-8"))
        self.assertEqual(overrides["proposed"][0]["slug"], "calha_danificada")
        self.assertEqual(overrides["history"][-1]["action"], "create")
        # 4. Slug repetido é recusado.
        with self.assertRaisesRegex(ValueError, "Já existe"):
            class_catalog.create_class(self.config, {"name": "calha_danificada", "display_name": "Outra"})
        # 5. Entradas inválidas: maiúsculas/espaço, hífen, sem display_name, cor não hexadecimal.
        # EXTENSAO: nova validação em create_class: acrescente o dict inválido nesta tupla.
        for bad in ({"name": "Calha Danificada", "display_name": "x"}, {"name": "calha-danificada", "display_name": "x"},
                    {"name": "ok_nome"}, {"name": "ok_nome", "display_name": "x", "color": "azul"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                class_catalog.create_class(self.config, bad)

    def test_update_changes_display_name_color_and_active_but_not_slug(self) -> None:
        """Editar muda nome de exibição/cor/ativo, mas nunca o slug nem o ID.

        Trocar o slug exige nova versão da taxonomia (os labels e pesos usam o ID/slug).
        """
        # 1. Edição permitida.
        updated = class_catalog.update_class(self.config, 0, {"display_name": "Telha rompida", "color": "#112233"})
        self.assertEqual((updated["display_name"], updated["color"], updated["name"]), ("Telha rompida", "#112233", "telha_quebrada"))
        # 2. Slug e ID são recusados.
        with self.assertRaisesRegex(ValueError, "nova versão da taxonomia"):
            class_catalog.update_class(self.config, 0, {"name": "telha_rompida"})
        with self.assertRaises(ValueError):
            class_catalog.update_class(self.config, 0, {"id": 7})
        # 3. ID inexistente lança ClassNotFound (a rota converte em 404).
        with self.assertRaises(class_catalog.ClassNotFound):
            class_catalog.update_class(self.config, 99, {"display_name": "x"})
        # Voltar ao nome original e cor vazia limpa a edição em vez de guardá-la.
        restored = class_catalog.update_class(self.config, 0, {"display_name": "Telha quebrada", "color": None})
        self.assertEqual(restored["color"], class_catalog.default_color(0))
        self.assertNotIn("display_name", class_catalog.load_overrides(self.config)["classes"]["telha_quebrada"])

    def test_active_class_that_requires_training_is_not_in_detector(self) -> None:
        """Classe marcada ``requires_training`` na taxonomia aparece fora do detector.

        Vale para a listagem, para o resumo e para as classes que a página recebe.
        """
        # Ex.: ID 13 da v6: ativo na taxonomia, mas os pesos piloto têm 13 classes.
        render_review_package(self.config, self.registry_path)  # página gerada antes da mudança (como no PC)
        # Marca a classe 1 como "precisa de treino" direto no taxonomy.json temporário.
        taxonomy = json.loads(self.taxonomy_path.read_text(encoding="utf-8"))
        taxonomy["classes"][1]["requires_training"] = True
        self.taxonomy_path.write_text(json.dumps(taxonomy), encoding="utf-8")
        listing = class_catalog.list_classes(self.config)
        self.assertEqual([(item["in_detector"], item["requires_training"]) for item in listing["classes"]], [(True, False), (False, True)])
        self.assertEqual((listing["summary"]["in_detector"], listing["summary"]["requires_training"]), (1, 1))
        # As classes embutidas na página (geradas antes) são corrigidas ao servir.
        payload = json.loads((self.directory / "browser_data.json").read_text(encoding="utf-8"))
        classes = class_catalog.apply_to_page_classes(self.config, payload["classes"])
        self.assertEqual([item["requires_training"] for item in classes], [False, True])

    def test_overrides_follow_the_slug_when_a_migration_moves_ids(self) -> None:
        """Edições acompanham o slug quando uma nova taxonomia muda os IDs.

        ATENCAO: se o overrides fosse indexado por ID, a edição passaria para a classe errada.
        """
        # 1. Edita residuos_calha (ID 1) e desativa.
        class_catalog.update_class(self.config, 1, {"display_name": "Detritos na calha", "active": False})
        taxonomy = json.loads(self.taxonomy_path.read_text(encoding="utf-8"))
        # Nova versão: uma classe nova ocupa o ID 1 e residuos_calha vai para o 2 (como o antigo 13 → 23 na v6).
        taxonomy["classes"] = [taxonomy["classes"][0], {"id": 1, "slug": "nova_ativa", "label": "Nova ativa", "phase": 1},
                               {**taxonomy["classes"][1], "id": 2}]
        self.taxonomy_path.write_text(json.dumps(taxonomy), encoding="utf-8")
        # 2. Indexa a listagem por slug e confere: a nova fica intocada, a edição seguiu residuos_calha.
        listing = {item["name"]: item for item in class_catalog.list_classes(self.config)["classes"]}
        self.assertEqual((listing["nova_ativa"]["display_name"], listing["nova_ativa"]["active"]), ("Nova ativa", True))
        self.assertEqual((listing["residuos_calha"]["id"], listing["residuos_calha"]["display_name"], listing["residuos_calha"]["active"]), (2, "Detritos na calha", False))

    def test_proposals_are_reconciled_when_the_taxonomy_grows(self) -> None:
        """Quando a taxonomia promove uma proposta, ela é absorvida; as demais são renumeradas.

        Garante IDs contíguos e sem colisão entre taxonomia e propostas, e que o
        histórico registra "absorbed" e "renumbered".
        """
        # 1. Duas propostas: calha_danificada (ID 2) e ralo_obstruido (ID 3).
        class_catalog.create_class(self.config, {"name": "calha_danificada", "display_name": "Calha danificada"})
        class_catalog.create_class(self.config, {"name": "ralo_obstruido", "display_name": "Ralo obstruído"})
        taxonomy = json.loads(self.taxonomy_path.read_text(encoding="utf-8"))
        # A taxonomia cresce até o ID 2 com outra classe e promove calha_danificada como ID 3.
        taxonomy["classes"] += [{"id": 2, "slug": "outra", "label": "Outra", "phase": 2},
                                {"id": 3, "slug": "calha_danificada", "label": "Calha danificada", "phase": 1}]
        self.taxonomy_path.write_text(json.dumps(taxonomy), encoding="utf-8")
        # 2. ralo_obstruido passa para o ID 4, depois da última classe da taxonomia.
        listing = class_catalog.list_classes(self.config)
        self.assertEqual([(item["id"], item["name"], item["source"]) for item in listing["classes"]],
                         [(0, "telha_quebrada", "taxonomia"), (1, "residuos_calha", "taxonomia"), (2, "outra", "taxonomia"),
                          (3, "calha_danificada", "taxonomia"), (4, "ralo_obstruido", "proposta")])
        # 3. Só a proposta não promovida continua no overrides.
        overrides = class_catalog.load_overrides(self.config)
        self.assertEqual([item["slug"] for item in overrides["proposed"]], ["ralo_obstruido"])
        self.assertEqual([entry["action"] for entry in overrides["history"][-2:]], ["absorbed", "renumbered"])
        # 4. A proposta renumerada continua editável pelo novo ID.
        updated = class_catalog.update_class(self.config, 4, {"display_name": "Ralo entupido"})
        self.assertEqual(updated["source"], "proposta")

    def test_deactivate_is_soft_delete_and_reversible(self) -> None:
        """Excluir é lógico: a classe fica inativa, continua listada e pode ser reativada."""
        gone = class_catalog.deactivate_class(self.config, 1)
        self.assertFalse(gone["active"])
        listing = class_catalog.list_classes(self.config)
        self.assertEqual(len(listing["classes"]), 2, "Exclusão lógica: a classe continua listada.")
        self.assertEqual(listing["summary"]["inactive"], 1)
        # Reativar pelo update_class.
        self.assertTrue(class_catalog.update_class(self.config, 1, {"active": True})["active"])

    def test_served_page_uses_display_name_and_color(self) -> None:
        """As edições valem na página servida, sem regravar ``browser_data.json`` em disco."""
        # 1. Gera a página e depois edita/desativa classes.
        render_review_package(self.config, self.registry_path)
        class_catalog.update_class(self.config, 1, {"display_name": "Detritos na calha", "color": "#ABCDEF"})
        class_catalog.deactivate_class(self.config, 0)
        # 2. apply_to_page_classes aplica as edições sobre as classes do arquivo.
        payload = json.loads((self.directory / "browser_data.json").read_text(encoding="utf-8"))
        classes = class_catalog.apply_to_page_classes(self.config, payload["classes"])
        self.assertEqual((classes[1]["label"], classes[1]["color"], classes[0]["active"]), ("Detritos na calha", "#ABCDEF", False))
        # 3. O arquivo em disco continua com o label original da taxonomia.
        self.assertEqual(payload["classes"][1]["label"], "Resíduos em calha", "browser_data.json em disco não muda.")


# ---------------------------------------------------------------------------
# Rotas HTTP da seção Classes (servidor real em 127.0.0.1)
# ---------------------------------------------------------------------------

class ClassRoutesTests(PlatformFixture):
    """Rotas ``/api/classes`` do ``platform_server`` com servidor HTTP de verdade.

    Importa porque é o caminho que a seção Classes da plataforma usa; confere
    códigos HTTP (200/201/400/403/404) e a exigência do token nas escritas.
    """

    def setUp(self) -> None:
        """Gera a página da revisão e sobe o servidor numa porta livre, em thread de fundo."""
        super().setUp()
        from http.server import ThreadingHTTPServer
        # 1. Página da revisão (a rota "/" serve a página gerada).
        render_review_package(self.config, self.registry_path)
        # 2. Token exigido no cabeçalho X-DroneCamp-Token para POST/PUT/DELETE.
        self.token = "token-de-teste"
        # 3. Porta 0 = o sistema escolhe uma porta livre. JobRunner real, sem tarefas.
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.config, None, self.token, JobRunner(self.root)))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        # 4. Limpeza em ordem inversa: primeiro shutdown (para o laço), depois server_close.
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def call(self, method: str, path: str, body: dict | None = None, token: str | None = None) -> tuple[int, dict]:
        """Faz uma requisição JSON ao servidor de teste.

        Parâmetros:
            method: "GET", "POST", "PUT" ou "DELETE".
            path: caminho da rota (ex.: "/api/classes/2").
            body: corpo JSON (dict) ou None para requisição sem corpo.
            token: valor do X-DroneCamp-Token; None simula chamada sem token.

        Retorna:
            (código HTTP, corpo JSON como dict). Erros HTTP não lançam: voltam com o código.
        """
        # Serializa o corpo, se houver.
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method=method)
        request.add_header("Content-Type", "application/json")
        if token:
            request.add_header("X-DroneCamp-Token", token)
        # urllib lança HTTPError em 4xx/5xx; aqui o erro vira (código, corpo) para a asserção.
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read() or b"{}")

    def test_crud_routes(self) -> None:
        """Ciclo completo listar → criar → editar → excluir, com e sem token, e a página servida."""
        # 1. Listar (GET não exige token).
        status, listing = self.call("GET", "/api/classes")
        self.assertEqual((status, len(listing["classes"])), (200, 2))
        # 2. Criar: sem token 403; com token 201; slug inválido 400 com mensagem citando snake_case.
        self.assertEqual(self.call("POST", "/api/classes", {"name": "ralo_obstruido", "display_name": "Ralo obstruído"})[0], 403)
        status, created = self.call("POST", "/api/classes", {"name": "ralo_obstruido", "display_name": "Ralo obstruído", "author": "revisor"}, self.token)
        self.assertEqual((status, created["id"], created["requires_training"]), (201, 2, True))
        status, error = self.call("POST", "/api/classes", {"name": "Ralo", "display_name": "x"}, self.token)
        self.assertEqual(status, 400)
        self.assertIn("snake_case", error["error"])
        # 3. Editar: a cor volta normalizada em maiúsculas; sem token 403; ID inexistente 404.
        status, updated = self.call("PUT", "/api/classes/2", {"display_name": "Ralo entupido", "color": "#00ff00"}, self.token)
        self.assertEqual((status, updated["display_name"], updated["color"]), (200, "Ralo entupido", "#00FF00"))
        self.assertEqual(self.call("PUT", "/api/classes/2", {"display_name": "x"})[0], 403)
        self.assertEqual(self.call("PUT", "/api/classes/42", {"display_name": "x"}, self.token)[0], 404)
        # 4. Excluir (desativar): 200 com token, 403 sem, 404 para ID inexistente.
        self.assertEqual(self.call("DELETE", "/api/classes/2", token=self.token)[0], 200)
        self.assertEqual(self.call("DELETE", "/api/classes/2")[0], 403)
        self.assertEqual(self.call("DELETE", "/api/classes/42", token=self.token)[0], 404)
        # 5. A classe excluída continua listada, inativa.
        listing = self.call("GET", "/api/classes")[1]
        self.assertEqual([(item["id"], item["active"]) for item in listing["classes"]], [(0, True), (1, True), (2, False)])
        # A página servida mostra o nome de exibição editado.
        self.call("PUT", "/api/classes/1", {"display_name": "Detritos na calha"}, self.token)
        with urllib.request.urlopen(self.base + "/") as response:
            page = response.read()
        self.assertIn("Detritos na calha".encode(), page)
        self.assertIn(class_catalog.TRAINING_NOTICE.encode(), page)


# ENTRADA: execução direta do arquivo (na pasta sistema_ia).
if __name__ == "__main__":
    unittest.main()
