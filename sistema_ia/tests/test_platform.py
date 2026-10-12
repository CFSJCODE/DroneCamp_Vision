"""Plataforma de operações: painel de treinos, página gerada e servidor local.

Dados sintéticos (sem Ultralytics): confere que o painel lê ``runs/`` sem
alterar nada, que a página embute o instantâneo, e que o servidor só serve
arquivos da revisão/``runs`` e só inicia comandos da lista fechada com token.

Módulos cobertos:
    - ``dronecamp_ia.operations``: ``read_curves``, ``collect_runs``, ``collect_comparisons``,
      ``run_id_from_path`` (painel de treinos);
    - ``dronecamp_ia.review_render``: ``render_review_package`` e ``refresh_review_page``
      (página da revisão com o instantâneo das operações embutido);
    - ``dronecamp_ia.platform_server``: ``make_handler`` (rotas HTTP), ``JobRunner`` (tarefas
      em segundo plano), ``_job_arguments`` (lista fechada de comandos e validação dos
      argumentos), ``_decisions_map`` e ``serve_platform``;
    - ``review_templates/platform.js``: só a função ``evolutionPoints`` (via Node, se instalado).

Como rodar (na pasta ``sistema_ia``, com o venv do projeto):
    python -m unittest tests.test_platform
    python -m unittest tests.test_platform.ServerTests
    python -m unittest tests.test_platform.WorkspaceServerTests.test_upload_flow_creates_a_new_inspection_from_photos

O que cada classe de teste prepara:
    - ``PlatformFixture`` (base, sem testes próprios) -> ``setUp`` cria pasta temporária com
      acento, taxonomia de 2 classes, ``ProjectConfig`` à mão (com bloco ``pilot``), uma foto
      20x16 em ``data/reference`` e ``data/reviews/fixture_v1/registry.json`` com 1 foto
      positiva não aprovada. ``make_run`` cria uma execução falsa em ``runs/<nome>``.
    - ``OperationsTests``: só a base (sem servidor).
    - ``ServerTests``: base + 1 execução + página gerada + servidor HTTP real em
      127.0.0.1 numa porta livre (thread), aberto com ``--registry`` (modo "fixed") e o token
      ``token-de-teste``. Helpers: ``request`` (GET/POST JSON), ``put`` (envio de arquivo) e
      ``wait_ingest`` (espera a importação terminar, até ~20 s).
    - ``MultiReviewServerTests``: servidor com DUAS revisões (fixture_v1 e internet_v2).
    - ``WorkspaceServerTests``: servidor sem ``--registry`` (modo "workspace": descobre as
      revisões em ``data/reviews``).
    Os servidores são fechados sozinhos no fim (``addCleanup``).

ATENCAO: o servidor só aceita alterações (POST/PUT) com o cabeçalho ``X-DroneCamp-Token``
correto; só inicia comandos da lista fechada; nunca serve arquivos fora da revisão e de
``runs/`` (nem pesos .pt); o ``index.html`` gravado em disco nunca leva o token.

EXTENSAO: para testar uma rota nova do servidor, copie
``WorkspaceServerTests.test_workspace_counts_only_suggestions_still_pending_after_feedback``:
use ``self.request("/api/<rota>", corpo, self.token)`` e confira (status, corpo JSON).
Teste também SEM token (deve dar 403) se a rota alterar algo. Para um comando novo na lista
fechada, acrescente um caso em ``test_job_arguments_are_validated`` (argumentos válidos e
inválidos). Para um campo novo do painel de treinos, copie
``OperationsTests.test_run_states_best_epoch_and_estimate`` usando ``self.make_run``.
"""

from pathlib import Path
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
import zipfile

# Permite rodar este arquivo direto (sem instalar o pacote): põe sistema_ia/src no caminho.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from dronecamp_ia.config import ProjectConfig
from dronecamp_ia.io import file_hash
from dronecamp_ia.operations import collect_comparisons, collect_runs, read_curves, run_id_from_path
from dronecamp_ia.platform_server import JobRunner, _decisions_map, _job_arguments, make_handler, serve_platform
from dronecamp_ia.review_render import refresh_review_page, render_review_package

# Cabeçalho do results.csv da Ultralytics usado nas execuções falsas.
RESULTS_HEADER = ("epoch,time,train/box_loss,train/cls_loss,train/l1_loss,metrics/precision(B),metrics/recall(B),"
                  "metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/l1_loss,lr/pg0\n")


def write_json(path: Path, value: dict) -> None:
    """Grava ``value`` como JSON UTF-8, criando as pastas que faltarem.

    Parâmetros:
        path (Path): destino.
        value (dict): conteúdo.
    Retorna: None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class PlatformFixture(unittest.TestCase):
    """Base comum: pasta temporária, taxonomia, config, uma foto e uma revisão (fixture_v1).

    Não tem testes próprios; as classes de teste herdam dela.
    """
    def setUp(self) -> None:
        """Cria a raiz temporária, a taxonomia de 2 classes, o config e a revisão com 1 foto.

        Atributos criados: ``root``, ``taxonomy_path``, ``config``, ``directory`` (pasta da
        revisão fixture_v1) e ``registry_path``.
        Retorna: None.
        """
        # 1. Pasta temporária com acento, apagada no fim do teste.
        temporary = tempfile.TemporaryDirectory(prefix="dronecamp plataforma á ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # 2. Taxonomia de 2 classes e configuração montada à mão (o bloco pilot é lido pelo servidor).
        self.taxonomy_path = self.root / "configs" / "taxonomy.json"
        write_json(self.taxonomy_path, {"version": "fixture", "classes": [
            {"id": 0, "slug": "telha_quebrada", "label": "Telha quebrada", "phase": 1},
            {"id": 1, "slug": "residuos_calha", "label": "Resíduos em calha", "phase": 1}]})
        self.config = ProjectConfig(root=self.root, model="fixture.pt", requested_model="fixture.pt",
                                    taxonomy_path=self.taxonomy_path, dataset_path=self.root / "dataset.yaml",
                                    device="cpu", prediction={}, training={"epochs": 5, "imgsz": 640, "batch": 2},
                                    pilot={"training": {"epochs": 3}, "suggestion_conf": 0.15})
        # 3. Uma foto 20x16 e a revisão fixture_v1 com essa foto positiva, sem aprovação humana.
        self.directory = self.root / "data" / "reviews" / "fixture_v1"
        photo = self.root / "data" / "reference" / "foto.png"
        photo.parent.mkdir(parents=True)
        Image.new("RGB", (20, 16), color=(90, 90, 90)).save(photo)
        self.registry_path = self.directory / "registry.json"
        write_json(self.registry_path, {"schema_version": 1, "version": "fixture_v1", "taxonomy_sha256": file_hash(self.taxonomy_path),
                                        "images": [{"image_sha256": file_hash(photo), "filename": photo.name, "source_path": str(photo),
                                                    "width": 20, "height": 16, "pages": [], "status": "positive",
                                                    "boxes": [{"class_id": 0, "bbox_xyxy": [2, 2, 9, 10], "note": "Sintética."}],
                                                    "human_approved": False, "technical_status": "pending_human_review",
                                                    "visual_review_status": "second_pass_ai", "building_group": "b1", "notes": "Sintética.", "severity": None}]})

    def make_run(self, name: str, rows: int, summary: dict | None) -> Path:
        """Cria uma execução de treino falsa em ``runs/<name>``.

        Grava ``fit/results.csv`` (cabeçalho com espaços; mAP50 = 0.1 x época, 30 s por época),
        ``execution.json`` (checkpoint com caminho Windows), ``pilot_provenance.json``,
        ``summary.json`` (se dado) e ``fit/weights/best.pt`` falso.

        Parâmetros:
            name (str): nome da pasta, no padrão ``<tipo>_<AAAAMMDDTHHMMSSZ>_<id8>``; o tipo e a
                data definem o nome exibido e a ordem.
            rows (int): quantidade de épocas no CSV.
            summary (dict | None): conteúdo do summary.json; None simula treino em andamento.
        Retorna: Path da execução.
        """
        run = self.root / "runs" / name
        fit = run / "fit"
        fit.mkdir(parents=True)
        # Cabeçalho com espaços, como em algumas versões da Ultralytics.
        lines = [RESULTS_HEADER.replace(",", ",  ")]
        # Colunas: epoch, time (30 s/época), perdas de treino, P, R, mAP50 (0.1 x época), mAP50-95,
        # perdas de validação e lr.
        for epoch in range(1, rows + 1):
            lines.append(f"{epoch},{epoch * 30.0},1.5,2.5,0.1,0.2,0.3,{0.1 * epoch},{0.05 * epoch},1.6,2.6,0.1,0.001\n")
        (fit / "results.csv").write_text("".join(lines), encoding="utf-8")
        write_json(run / "execution.json", {"runtime": {"checkpoint": "E:\\proj\\sistema_ia\\models\\yolo26l.pt", "device_requested": "cpu"},
                                            "parameters": {"epochs": 3, "imgsz": 640, "batch": 2, "workers": 0},
                                            "class_counts": {"train": {"telha_quebrada": 4, "residuos_calha": 0}}})
        write_json(run / "pilot_provenance.json", {"pilot_sha256": "abc", "buildings": ["b1"], "independent_evaluation": False})
        if summary is not None:
            write_json(run / "summary.json", summary)
        (fit / "weights").mkdir()
        (fit / "weights" / "best.pt").write_bytes(b"pesos sinteticos")
        return run


class OperationsTests(PlatformFixture):
    """Testes do painel de treinos (operations.py) e da página gerada, sem servidor."""
    def test_curves_tolerate_spaced_headers_and_keep_epoch_order(self) -> None:
        """As curvas aceitam cabeçalho com espaços e ficam na ordem das épocas."""
        run = self.make_run("pilot_train_20261004T170212Z_dd284411", 3, {"state": "completed", "complete": True})
        curves = read_curves(run / "fit" / "results.csv")
        self.assertEqual([point["epoch"] for point in curves], [1, 2, 3])
        self.assertAlmostEqual(curves[-1]["map50"], 0.3)
        self.assertAlmostEqual(curves[0]["val_cls"], 2.6)

    def test_run_states_best_epoch_and_estimate(self) -> None:
        """Estado (concluído/rodando/interrompido), melhor época, tempo por época e ordem das execuções.

        Sem summary.json o treino está "running"; uma hora depois sem atividade vira "interrupted".
        """
        self.make_run("pilot_train_20261004T170212Z_dd284411", 3, {"state": "completed", "complete": True})
        self.make_run("pilot_train_20261004T200000Z_aaaaaaaa", 2, None)
        runs = {run["id"]: run for run in collect_runs(self.root)}
        done = runs["pilot_train_20261004T170212Z_dd284411"]
        self.assertEqual(done["state"], "completed")
        self.assertEqual(done["best_epoch"]["epoch"], 3)
        self.assertEqual(done["seconds_per_epoch"], 30.0)
        self.assertEqual(done["checkpoint"], "yolo26l.pt", "Caminho do Windows vira só o nome do arquivo.")
        self.assertEqual(done["weights"]["best"], "runs/pilot_train_20261004T170212Z_dd284411/fit/weights/best.pt")
        self.assertEqual(runs["pilot_train_20261004T200000Z_aaaaaaaa"]["state"], "running")
        # Simula "agora" uma hora depois: o treino sem summary.json passa a "interrupted".
        later = {run["id"]: run for run in collect_runs(self.root, now=time.time() + 3600)}
        self.assertEqual(later["pilot_train_20261004T200000Z_aaaaaaaa"]["state"], "interrupted")
        self.assertEqual(list(runs)[0], "pilot_train_20261004T200000Z_aaaaaaaa", "Mais recente primeiro.")

    def test_runs_get_readable_names_from_type_and_short_hash_and_customizable(self) -> None:
        """Cada execução ganha nome legível (tipo + código curto) e pode ser renomeada em runs/nomes.json."""
        first, second = "pilot_train_20261004T170212Z_dd284411", "pilot_train_20261004T200000Z_aaaaaaaa"
        self.make_run(first, 3, {"state": "completed", "complete": True})
        self.make_run(second, 2, None)
        self.make_run("train_20261005T100000Z_bbbbbbbb", 1, None)
        runs = {run["id"]: run for run in collect_runs(self.root)}
        # Tipo da execução + código curto do id; o resumo traz modelo, épocas e mAP50.
        self.assertEqual(runs[first]["display_name"], "Treino Piloto DD284411")
        self.assertEqual(runs[second]["display_name"], "Treino Piloto AAAAAAAA")
        self.assertEqual(runs["train_20261005T100000Z_bbbbbbbb"]["display_name"], "Treino de Produção BBBBBBBB")
        self.assertEqual(runs[first]["display_detail"], "yolo26l · 3/3 épocas · mAP50 val 30,0% (época 3)")
        write_json(self.root / "runs" / "nomes.json", {"dd284411": "v7 base", second: "v7.1"})
        renamed = {run["id"]: run["display_name"] for run in collect_runs(self.root)}
        self.assertEqual((renamed[first], renamed[second]), ("v7 base", "v7.1"))

    @unittest.skipUnless(shutil.which("node"), "Node não instalado: gráfico de evolução não verificado.")
    def test_evolution_chart_takes_every_run_with_validation_in_start_order(self) -> None:
        """O gráfico da seção Modelos lê os treinos do instantâneo: um treino novo entra sozinho.

        Extrai ``METRIC_SERIES`` e ``evolutionPoints`` de platform.js e roda no Node.
        ATENCAO: o teste corta o JS pelos textos "const METRIC_SERIES", "function evolutionPoints"
        e "function renderEvolution"; renomear essas funções em platform.js quebra o teste.
        """
        self.make_run("pilot_train_20261007T010057Z_f5e5ba8b", 3, {"state": "completed", "complete": True})
        self.make_run("pilot_train_20261004T170212Z_dd284411", 2, {"state": "completed", "complete": True})
        self.make_run("tune_20261008T120000Z_0000abcd", 1, {"state": "failed", "complete": False})
        source = (Path(__file__).resolve().parents[1] / "src" / "dronecamp_ia" / "review_templates" / "platform.js").read_text(encoding="utf-8")
        # Recorta do platform.js só a constante METRIC_SERIES e a função evolutionPoints.
        series = source[source.index("const METRIC_SERIES"):source.index("];", source.index("const METRIC_SERIES")) + 2]
        points = source[source.index("function evolutionPoints"):source.index("function renderEvolution")]

        def evolution(runs: list[dict], filter_: str) -> list[list]:
            """Roda ``evolutionPoints(filter_)`` no Node com as execuções dadas.

            Parâmetros:
                runs (list[dict]): execuções (saída de ``collect_runs``).
                filter_ (str): filtro do gráfico ("all", "completed"...).
            Retorna: lista de [x, últimos 8 caracteres do id, map50] por ponto.
            """
            script = f"{series}\nconst OPS = {{ runs: {json.dumps(runs)} }};\n{points}\n" + \
                f"process.stdout.write(JSON.stringify(evolutionPoints({json.dumps(filter_)}).map(p => [p.x, p.run.id.slice(-8), p.map50])));"
            return json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, encoding="utf-8").stdout)

        runs = collect_runs(self.root)
        everything = evolution(runs, "all")
        self.assertEqual([point[1] for point in everything], ["dd284411", "f5e5ba8b", "0000abcd"], "Ordem em que os treinos começaram.")
        self.assertEqual([point[0] for point in everything], [1, 2, 3])
        self.assertAlmostEqual(everything[1][2], 0.3, msg="Métricas da melhor época de validação.")
        self.assertEqual([point[1] for point in evolution(runs, "completed")], ["dd284411", "f5e5ba8b"])
        self.make_run("pilot_train_20261012T090000Z_1234beef", 1, None)
        self.assertEqual(evolution(collect_runs(self.root), "all")[-1][1], "1234beef", "Treino novo aparece sem mudar o código.")

    def test_dataset_identity_and_run_of_weights_cover_production_runs(self) -> None:
        """Treino de produção também tem identidade de dataset (divisória do gráfico de evolução).

        Também confere que ``run_id_from_path`` acha a execução num caminho Windows de pesos.
        """
        run = self.make_run("train_20261009T120000Z_0badc0de", 2, {"state": "completed", "complete": True})
        (run / "pilot_provenance.json").unlink()
        write_json(run / "training_provenance.json", {"valid": True, "dataset_yaml_sha256": "yaml123", "provenance_sha256": "prov456"})
        self.assertEqual(collect_runs(self.root)[0]["dataset_sha256"], "yaml123")
        self.assertEqual(run_id_from_path("E:\\proj\\runs\\train_20261009T120000Z_0badc0de\\fit\\weights\\last.pt"), "train_20261009T120000Z_0badc0de")
        self.assertIsNone(run_id_from_path("models/yolo26l.pt"))

    def test_comparisons_link_weights_to_runs(self) -> None:
        """As comparações ligam cada peso à execução que o gerou."""
        write_json(self.root / "runs" / "compare_fixture.json", {"dataset": "d.yaml", "conf": 0.15, "iou_match": 0.5, "models": [
            {"weights": "runs\\pilot_train_20261004T170212Z_dd284411\\fit\\weights\\best.pt", "weights_sha256": "f" * 64,
             "splits": {"val": {"images": 1, "truth": 2, "found": 1, "suggestions": 2, "correct": 1}}}]})
        comparison = collect_comparisons(self.root)[0]
        self.assertEqual(comparison["models"][0]["run_id"], "pilot_train_20261004T170212Z_dd284411")
        self.assertEqual(comparison["models"][0]["file"], "best.pt")

    def test_page_embeds_snapshot_without_changing_runs_or_browser_data_contract(self) -> None:
        """A página embute o instantâneo das execuções, sem token e sem alterar runs/ nem browser_data.json."""
        run = self.make_run("pilot_train_20261004T170212Z_dd284411", 3, {"state": "completed", "complete": True})
        before = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
        render_review_package(self.config, self.registry_path)
        html = (self.directory / "index.html").read_text(encoding="utf-8")
        self.assertIn("pilot_train_20261004T170212Z_dd284411", html)
        self.assertIn('<meta name="dronecamp-server" content="">', html, "Arquivo em disco nunca leva token.")
        self.assertNotIn("__REVIEW_DATA__", html)
        payload = json.loads((self.directory / "browser_data.json").read_text(encoding="utf-8"))
        self.assertNotIn("operations", payload)
        self.assertEqual(before, {path: path.read_bytes() for path in run.rglob("*") if path.is_file()})

    def test_refresh_rewrites_only_page_and_rejects_other_registry_version(self) -> None:
        """``refresh_review_page`` regrava só o index.html e recusa registro de outra versão."""
        render_review_package(self.config, self.registry_path)
        evidence = next((self.directory / "evidence").iterdir())
        stamp = evidence.stat().st_mtime_ns
        (self.directory / "index.html").write_text("antiga", encoding="utf-8")
        refresh_review_page(self.config, self.registry_path)
        self.assertIn("DroneCamp Vision", (self.directory / "index.html").read_text(encoding="utf-8"))
        self.assertEqual(evidence.stat().st_mtime_ns, stamp)
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        registry["version"] = "outra"
        write_json(self.registry_path, registry)
        with self.assertRaises(ValueError):
            refresh_review_page(self.config, self.registry_path)

    def test_platform_refuses_review_from_previous_taxonomy(self) -> None:
        """O servidor recusa revisão feita com taxonomia anterior e indica ``migrate-review``."""
        # Ex.: página v8 (13 classes) aberta depois da troca para a taxonomia v6.
        render_review_package(self.config, self.registry_path)
        taxonomy = json.loads(self.taxonomy_path.read_text(encoding="utf-8"))
        taxonomy["classes"].append({"id": 2, "slug": "nova", "label": "Nova", "phase": 1})
        write_json(self.taxonomy_path, taxonomy)
        with self.assertRaisesRegex(ValueError, "migrate-review"):
            serve_platform(self.config, self.registry_path, port=0, open_browser=False)


class ServerTests(PlatformFixture):
    """Servidor real (thread) com UMA revisão aberta via ``--registry`` (modo "fixed")."""
    def setUp(self) -> None:
        """Cria uma execução, gera a página e sobe o servidor em 127.0.0.1 numa porta livre.

        Atributos criados: ``token``, ``server`` e ``base`` (URL ``http://127.0.0.1:<porta>``).
        Retorna: None.
        """
        super().setUp()
        # Servidor real em thread, porta 0 (o sistema escolhe uma livre).
        from http.server import ThreadingHTTPServer
        self.make_run("pilot_train_20261004T170212Z_dd284411", 3, {"state": "completed", "complete": True})
        render_review_package(self.config, self.registry_path)
        self.token = "token-de-teste"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.config, self.registry_path, self.token, JobRunner(self.root)))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def request(self, path: str, body: dict | None = None, token: str | None = None) -> tuple[int, bytes]:
        """Faz GET (sem corpo) ou POST JSON (com corpo) no servidor de teste.

        Também é reaproveitada por ``MultiReviewServerTests`` e ``WorkspaceServerTests``.

        Parâmetros:
            path (str): caminho com query, ex.: "/api/workspace?review=fixture_v1".
            body (dict | None): corpo JSON; None faz GET.
            token (str | None): valor do cabeçalho ``X-DroneCamp-Token``.
        Retorna: tupla (status HTTP, bytes do corpo), inclusive para erros 4xx/5xx.
        """
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method="GET" if body is None else "POST")
        if token:
            request.add_header("X-DroneCamp-Token", token)
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def put(self, path: str, data: bytes, filename: str, token: str | None = None) -> tuple[int, bytes]:
        """Envia um arquivo por PUT (como o navegador faz no envio de fotos).

        Parâmetros:
            path (str): ex.: "/api/uploads/<lote>/files".
            data (bytes): conteúdo do arquivo.
            filename (str): nome enviado no cabeçalho ``X-File-Name`` (codificado em URL).
            token (str | None): token de alteração.
        Retorna: tupla (status HTTP, bytes do corpo).
        """
        request = urllib.request.Request(self.base + path, data=data, method="PUT")
        request.add_header("X-File-Name", urllib.parse.quote(filename))
        request.add_header("Content-Type", "application/octet-stream")
        if token:
            request.add_header("X-DroneCamp-Token", token)
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def wait_ingest(self, batch: str) -> dict:
        """Consulta o progresso da importação de um lote até sair de "running" (até 200 x 0.1 s).

        Parâmetros:
            batch (str): id do lote devolvido por POST /api/uploads.
        Retorna: dict do progresso (``state``, ``result``...); o último lido se o tempo acabar.
        """
        for _ in range(200):
            progress = json.loads(self.request(f"/api/uploads/{batch}")[1])
            if progress["state"] != "running":
                return progress
            time.sleep(0.1)
        return progress

    def test_served_page_has_token_and_live_operations(self) -> None:
        """A página servida leva o token e /api/operations lista as execuções."""
        status, page = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn(f'content="{self.token}"'.encode(), page)
        status, body = self.request("/api/operations")
        self.assertEqual(json.loads(body)["runs"][0]["id"], "pilot_train_20261004T170212Z_dd284411")

    def test_only_review_and_runs_files_are_served(self) -> None:
        """Só arquivos da revisão e de runs/ são servidos; ``..`` (normal ou codificado) e pesos .pt dão 404."""
        self.assertEqual(self.request("/runs/pilot_train_20261004T170212Z_dd284411/fit/results.csv")[0], 200)
        self.assertEqual(self.request("/../../configs/taxonomy.json")[0], 404)
        self.assertEqual(self.request("/%2e%2e/%2e%2e/%2e%2e/configs/taxonomy.json")[0], 404)
        self.assertEqual(self.request("/runs/pilot_train_20261004T170212Z_dd284411/fit/weights/best.pt")[0], 404)

    def test_changes_require_token_and_allowed_commands(self) -> None:
        """Iniciar tarefa exige token correto e comando da lista fechada; nada fica rodando."""
        self.assertEqual(self.request("/api/jobs", {"kind": "train-pilot"})[0], 403)
        self.assertEqual(self.request("/api/jobs", {"kind": "train-pilot"}, "errado")[0], 403)
        status, body = self.request("/api/jobs", {"kind": "export", "options": {}}, self.token)
        self.assertEqual(status, 400)
        self.assertIn("não permitida", json.loads(body)["error"])
        self.assertEqual(json.loads(self.request("/api/jobs")[1])["state"], "idle")

    def test_job_arguments_are_validated(self) -> None:
        """``_job_arguments`` normaliza caminhos e recusa valores fora do padrão, injeção e ``..``.

        EXTENSAO: comando novo na lista fechada: acrescente aqui um caso válido e os inválidos.
        """
        dataset = self.root / "data" / "pilot" / "v1" / "dataset.yaml"
        dataset.parent.mkdir(parents=True)
        dataset.write_text("path: .\n", encoding="utf-8")
        # Caso válido: caminho com barra invertida vira "/" e os números passam.
        arguments = _job_arguments(self.root, None, "train-pilot",  # treinar não depende de uma revisão aberta
                                   {"data": "data\\pilot\\v1\\dataset.yaml", "epochs": 10, "imgsz": 640, "batch": 2,
                                    "weights": "runs/pilot_train_20261004T170212Z_dd284411/fit/weights/best.pt"})
        self.assertEqual(arguments[:2], ["--data", "data/pilot/v1/dataset.yaml"])
        self.assertIn("--epochs", arguments)
        # Casos inválidos: imgsz fora do padrão, texto com ";" (injeção), arquivo que não é dataset
        # e pesos fora do projeto.
        for options in ({"data": "data/pilot/v1/dataset.yaml", "imgsz": 650},
                        {"data": "data/pilot/v1/dataset.yaml", "epochs": "10; rm"},
                        {"data": str(self.taxonomy_path)},
                        {"data": "data/pilot/v1/dataset.yaml", "weights": "../fora.pt"}):
            with self.assertRaises(ValueError):
                _job_arguments(self.root, self.registry_path, "train-pilot", options)
        suggest = _job_arguments(self.root, self.registry_path, "suggest",
                                 {"weights": "runs/pilot_train_20261004T170212Z_dd284411/fit/weights/best.pt", "conf": 0.2})
        self.assertEqual(suggest[suggest.index("--registry") + 1], "data/reviews/fixture_v1/registry.json")

    def test_feedback_is_saved_next_to_registry_without_overwriting(self) -> None:
        """O feedback vai para ``feedback_inbox`` da revisão, com nome seguro, e exige o hash certo."""
        # reviewer_id com "/../" prova que o nome do arquivo salvo é saneado (o conteúdo fica igual).
        feedback = {"schema_version": 1, "registry_sha256": file_hash(self.registry_path), "reviewer_id": "rev/../01",
                    "images": [{"image_sha256": "x", "status": "ambiguous"}]}
        status, body = self.request("/api/feedback", {**feedback, "registry_sha256": "0" * 64}, self.token)
        self.assertEqual(status, 400)
        status, body = self.request("/api/feedback", feedback, self.token)
        self.assertEqual(status, 200, body)
        saved = self.root / json.loads(body)["saved"]
        self.assertEqual(saved.parent, self.directory / "feedback_inbox")
        self.assertNotIn("..", saved.name)
        self.assertEqual(json.loads(saved.read_text(encoding="utf-8"))["reviewer_id"], "rev/../01")

    def test_imported_inspection_joins_the_fixed_list(self) -> None:
        """Inspeção importada pela página entra na lista mesmo no modo "fixed" (com --registry)."""
        batch = json.loads(self.request("/api/uploads", {"name": "lote_fixo"}, self.token)[1])["batch"]
        self.assertEqual(self.put(f"/api/uploads/{batch}/files", png_bytes((10, 120, 200)), "foto.png", self.token)[0], 200)
        self.assertEqual(self.request(f"/api/uploads/{batch}/ingest", {"group": "galpao_f"}, self.token)[0], 200)
        progress = self.wait_ingest(batch)
        self.assertEqual(progress["state"], "finished", progress)
        self.assertEqual(self.request("/r/lote_fixo/")[0], 200, "Com --registry, a inspeção importada também fica aberta.")
        workspace = json.loads(self.request("/api/workspace")[1])
        self.assertEqual((workspace["mode"], workspace["default"]), ("fixed", "fixture_v1"))
        self.assertIn("lote_fixo", {item["name"] for item in workspace["reviews"]})
        self.assertEqual(self.request("/r/outra_pasta/")[0], 404)



class MultiReviewServerTests(PlatformFixture):
    """Duas revisões no mesmo servidor: cada uma na sua página, fotos e arquivos separados."""

    def setUp(self) -> None:
        """Gera a página de fixture_v1, cria internet_v2 (cópia do registro com outra versão) e sobe o servidor com as duas.

        Atributos criados: ``other_directory``, ``other_registry``, ``token``, ``server`` e ``base``.
        Retorna: None.
        """
        super().setUp()
        from http.server import ThreadingHTTPServer
        render_review_package(self.config, self.registry_path)
        # Segunda revisão: mesmo conteúdo com outra "version" (logo, outro hash).
        self.other_directory = self.root / "data" / "reviews" / "internet_v2"
        self.other_directory.mkdir(parents=True)
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        registry["version"] = "internet_v2"
        self.other_registry = self.other_directory / "registry.json"
        write_json(self.other_registry, registry)
        render_review_package(self.config, self.other_registry)
        self.token = "token-de-teste"
        handler = make_handler(self.config, [self.registry_path, self.other_registry], self.token, JobRunner(self.root))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    # Reaproveita o helper de requisição de ServerTests.
    request = ServerTests.request

    def test_each_review_has_its_page_picker_and_files(self) -> None:
        """Cada revisão tem página em /r/<nome>/, seletor com as outras e arquivos próprios; / abre a primeira."""
        status, page = self.request("/r/internet_v2/")
        self.assertEqual(status, 200)
        self.assertIn(file_hash(self.other_registry).encode(), page)
        self.assertIn(b"/r/fixture_v1/", page, "A página lista as outras revisões no seletor.")
        status, page = self.request("/")
        self.assertIn(file_hash(self.registry_path).encode(), page, "A primeira revisão abre em /.")
        evidence = next((self.other_directory / "evidence").iterdir()).name
        self.assertEqual(self.request(f"/r/internet_v2/evidence/{evidence}")[0], 200)
        self.assertEqual(self.request("/r/inexistente/")[0], 404)
        self.assertEqual(self.request("/r/internet_v2/../../configs/taxonomy.json")[0], 404)

    def test_feedback_goes_to_the_registry_it_came_from(self) -> None:
        """O feedback é salvo na revisão cujo hash ele cita."""
        feedback = {"schema_version": 1, "registry_sha256": file_hash(self.other_registry), "reviewer_id": "r1",
                    "images": [{"image_sha256": "x", "status": "ambiguous"}]}
        status, body = self.request("/api/feedback", feedback, self.token)
        self.assertEqual(status, 200, body)
        self.assertEqual((self.root / json.loads(body)["saved"]).parent, self.other_directory / "feedback_inbox")

    def test_unknown_review_is_rejected_for_jobs(self) -> None:
        """Tarefa que cita revisão não aberta é recusada."""
        status, body = self.request("/api/jobs", {"kind": "suggest", "options": {"review": "outra", "weights": "x.pt"}}, self.token)
        self.assertEqual(status, 400)
        self.assertIn("não aberta", json.loads(body)["error"])

# ---------------------------------------------------------------------------
# Helper de envio e servidor em modo espaço de trabalho
# ---------------------------------------------------------------------------
def png_bytes(color: tuple[int, int, int]) -> bytes:
    """Gera um PNG 32x24 de uma cor, em memória (para envios pelo servidor).

    Parâmetros:
        color (tuple): cor RGB; cores diferentes geram hashes diferentes.
    Retorna: bytes do PNG.
    """
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


class WorkspaceServerTests(PlatformFixture):
    """Modo espaço de trabalho: inspeções descobertas em data/reviews, importação pela página, laudos e pacotes."""

    def setUp(self) -> None:
        """Gera a página de fixture_v1 e sobe o servidor SEM --registry (modo "workspace").

        Atributos criados: ``token``, ``server`` e ``base``.
        Retorna: None.
        """
        super().setUp()
        from http.server import ThreadingHTTPServer
        render_review_package(self.config, self.registry_path)
        self.token = "token-de-teste"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.config, None, self.token, JobRunner(self.root)))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    # Reaproveita os helpers HTTP de ServerTests.
    request, put, wait_ingest = ServerTests.request, ServerTests.put, ServerTests.wait_ingest

    def test_workspace_lists_reviews_and_opens_the_most_recent_page(self) -> None:
        """/api/workspace lista as revisões; registro sem página aparece, mas não vira padrão (409 ao abrir)."""
        status, body = self.request("/api/workspace?review=fixture_v1")
        self.assertEqual(status, 200)
        workspace = json.loads(body)
        self.assertEqual((workspace["mode"], workspace["default"]), ("workspace", "fixture_v1"))
        review = workspace["reviews"][0]
        self.assertEqual((review["name"], review["images"], review["current"], review["page_ready"]), ("fixture_v1", 1, True, True))
        self.assertEqual(review["buildings"], ["b1"])
        self.assertIn("pdf", workspace["capabilities"])
        status, page = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn(file_hash(self.registry_path).encode(), page, "Sem --registry, a revisão mais recente abre em /.")
        # Registro sem página gerada: aparece na lista, não vira padrão e a página explica o que falta.
        other = self.root / "data" / "reviews" / "sem_pagina" / "registry.json"
        write_json(other, {**json.loads(self.registry_path.read_text(encoding="utf-8")), "version": "sem_pagina"})
        workspace = json.loads(self.request("/api/workspace")[1])
        names = {item["name"]: item for item in workspace["reviews"]}
        self.assertFalse(names["sem_pagina"]["page_ready"])
        self.assertEqual(workspace["default"], "fixture_v1")
        self.assertEqual(self.request("/r/sem_pagina/")[0], 409)
        self.assertIn(b"/r/sem_pagina/", self.request("/")[1], "O seletor lista as revisões descobertas.")

    def test_empty_workspace_serves_the_page_without_an_inspection(self) -> None:
        """Sem nenhuma revisão, a página abre vazia, o painel funciona e só "suggest" exige revisão."""
        shutil.rmtree(self.root / "data" / "reviews")
        status, page = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn(b'"images": []', page)
        self.assertIn("Resíduos em calha".encode(), page, "As classes da taxonomia continuam na página.")
        workspace = json.loads(self.request("/api/workspace")[1])
        self.assertEqual((workspace["reviews"], workspace["default"]), ([], None))
        self.assertEqual(self.request("/api/operations")[0], 200)
        self.assertEqual(self.request("/r/fixture_v1/")[0], 404)
        # Treinar não exige uma inspeção aberta: só a validação do dataset falha aqui; sugerir exige.
        status, body = self.request("/api/jobs", {"kind": "train-pilot", "options": {"data": "data/pilot/inexistente.yaml"}}, self.token)
        self.assertEqual(status, 400)
        self.assertNotIn("Revisão", json.loads(body)["error"])
        status, body = self.request("/api/jobs", {"kind": "suggest", "options": {"weights": "models/x.pt"}}, self.token)
        self.assertIn("Revisão não aberta", json.loads(body)["error"])

    def test_upload_flow_creates_a_new_inspection_from_photos(self) -> None:
        """Fluxo completo de envio: criar lote, enviar fotos, importar e abrir a inspeção nova.

        Confere token, nome seguro do arquivo, extensão recusada, foto repetida, nome exibido com
        acento, mídia guardada em ``midia_enviada`` e limpeza da área temporária.
        """
        # 1. Criar o lote (sem token: 403) e enviar arquivos (nome com "../", texto, repetida).
        self.assertEqual(self.request("/api/uploads", {"name": "Galpão Norte 2026"})[0], 403)
        status, body = self.request("/api/uploads", {"name": "Galpão Norte 2026"}, self.token)
        self.assertEqual(status, 200, body)
        batch = json.loads(body)["batch"]
        self.assertEqual(json.loads(body)["name"], "galpao_norte_2026")
        first, second = png_bytes((10, 120, 200)), png_bytes((200, 90, 10))
        self.assertEqual(self.put(f"/api/uploads/{batch}/files", first, "foto1.png")[0], 403)
        status, body = self.put(f"/api/uploads/{batch}/files", first, "../../foto1.png", self.token)
        self.assertEqual(status, 200, body)
        self.assertEqual(json.loads(body)["stored"], "foto1.png", "Só o nome do arquivo conta; pastas são descartadas.")
        self.assertEqual(self.put(f"/api/uploads/{batch}/files", b"texto", "notas.txt", self.token)[0], 400)
        self.assertEqual(self.put(f"/api/uploads/{batch}/files", second, "foto2.png", self.token)[0], 200)
        self.assertEqual(self.put(f"/api/uploads/{batch}/files", first, "foto1 copia.png", self.token)[0], 200)
        self.assertEqual(self.put(f"/api/uploads/lote_inexistente/files", first, "x.png", self.token)[0], 400)
        # 2. Importar: grupo vazio e pesos fora do projeto são recusados; depois importa de verdade.
        self.assertEqual(self.request(f"/api/uploads/{batch}/ingest", {"group": ""}, self.token)[0], 400)
        self.assertEqual(self.request(f"/api/uploads/{batch}/ingest", {"group": "galpao_norte", "weights": "../fora.pt"}, self.token)[0], 400)
        status, body = self.request(f"/api/uploads/{batch}/ingest", {"group": "galpao_norte", "every_seconds": 2, "max_frames": 50,
                                                                   "notes": "voo de teste", "uploaded_by": "rev01"}, self.token)
        self.assertEqual(status, 200, body)
        progress = self.wait_ingest(batch)
        self.assertEqual(progress["state"], "finished", progress)
        result = progress["result"]
        self.assertEqual((result["images"], result["photos"], result["videos"]), (2, 2, 0))
        self.assertEqual([item["reason"] for item in result["skipped"]], ["foto_repetida"])
        self.assertEqual(result["href"], "/r/galpao_norte_2026/")
        self.assertIsNone(result["suggest_job"], "Sem pesos escolhidos, nenhuma tarefa de sugestões é iniciada.")
        # 3. Confere o registro gerado, a mídia guardada e a inspeção nova na lista e no servidor.
        directory = self.root / "data" / "reviews" / "galpao_norte_2026"
        registry = json.loads((directory / "registry.json").read_text(encoding="utf-8"))
        self.assertEqual(registry["ingest"]["uploaded_by"], "rev01")
        # O nome digitado na criação do lote (com acentos) vale mesmo quando o ingest não o repete.
        self.assertEqual((registry["display_name"], result["display_name"]), ("Galpão Norte 2026", "Galpão Norte 2026"))
        self.assertEqual({item["building_group"] for item in registry["images"]}, {"galpao_norte"})
        self.assertTrue((directory / "index.html").is_file())
        kept = directory / "midia_enviada"
        self.assertEqual(sorted(path.name for path in kept.iterdir()), ["foto1 copia.png", "foto1.png", "foto2.png"], "Os arquivos enviados ficam na inspeção.")
        self.assertEqual({Path(source["path"]).parent for source in registry["sources"]}, {kept.resolve()})
        self.assertFalse((self.root / ".runtime" / "uploads" / batch).exists(), "Os arquivos do lote saem da área temporária.")
        workspace = json.loads(self.request("/api/workspace")[1])
        created = next(item for item in workspace["reviews"] if item["name"] == "galpao_norte_2026")
        self.assertEqual(created["display_name"], "Galpão Norte 2026")
        self.assertEqual((created["images"], created["page_ready"], created["scope_label"]), (2, True, "importada pela plataforma"))
        status, page = self.request("/r/galpao_norte_2026/")
        self.assertEqual(status, 200)
        image_url = json.loads((directory / "browser_data.json").read_text(encoding="utf-8"))["images"][0]["image_url"]
        self.assertEqual(self.request(f"/r/galpao_norte_2026/{image_url}")[0], 200)
        self.assertEqual(self.request("/api/uploads", {"name": "galpao_norte_2026"}, self.token)[0], 400, "Nome já usado é recusado antes do envio.")
        self.assertEqual(self.request("/api/uploads/lote_inexistente")[0], 404)

    def test_workspace_counts_only_suggestions_still_pending_after_feedback(self) -> None:
        """Sugestões aceitas (mesma caixa) ou descartadas no feedback deixam de contar como pendentes."""
        digest = json.loads(self.registry_path.read_text(encoding="utf-8"))["images"][0]["image_sha256"]
        write_json(self.directory / "suggestions.json", {"schema_version": 1, "registry_sha256": file_hash(self.registry_path), "images": {digest: [
            {"id": "s001", "class_id": 0, "bbox_xyxy": [3, 3, 12, 11], "confidence": 0.8},
            {"id": "s002", "class_id": 1, "bbox_xyxy": [10, 3, 18, 12], "confidence": 0.4}]}})
        review = json.loads(self.request("/api/workspace")[1])["reviews"][0]
        self.assertEqual((review["suggestions_pending"], review["decided"]), (2, 0))
        feedback = {"schema_version": 1, "registry_sha256": file_hash(self.registry_path), "reviewer_id": "rev01",
                    "images": [{"image_sha256": digest, "status": "positive", "boxes": [{"class_id": 0, "bbox_xyxy": [3, 3, 12, 11], "note": ""}],
                                "notes": "", "confirmed_complete": True, "dismissed_suggestions": ["s002"]}]}
        self.assertEqual(self.request("/api/feedback", feedback, self.token)[0], 200)
        review = json.loads(self.request("/api/workspace")[1])["reviews"][0]
        self.assertEqual((review["suggestions_pending"], review["decided"]), (0, 1), "Aceita (mesma caixa) e descartada deixam de ser pendentes.")

    def test_evidence_package_and_report_are_written_inside_the_review(self) -> None:
        """ZIP de evidências e laudo são gravados dentro da revisão e servidos só se forem dos tipos gerados.

        ATENCAO: o laudo nunca infere gravidade/diagnóstico ("pendente de validação técnica").
        """
        digest = json.loads(self.registry_path.read_text(encoding="utf-8"))["images"][0]["image_sha256"]
        decisions = [{"image_sha256": digest, "status": "positive", "boxes": [{"class_id": 1, "bbox_xyxy": [1, 1, 8, 9], "note": "revisada"}],
                      "notes": "", "confirmed_complete": True, "dismissed_suggestions": []}]
        self.assertEqual(_decisions_map(decisions)[digest]["boxes"][0]["class_id"], 1)
        # 1. Pacote de evidências: "only" inválido dá 400; depois gera e baixa o ZIP.
        self.assertEqual(self.request("/api/export/evidence", {"review": "fixture_v1", "only": "tudo"}, self.token)[0], 400)
        status, body = self.request("/api/export/evidence", {"review": "fixture_v1", "decisions": decisions, "max_side": 1024,
                                                            "only": "all", "include_suggestions": False}, self.token)
        self.assertEqual(status, 200, body)
        result = json.loads(body)
        self.assertEqual(result["images"], 1)
        self.assertTrue(result["href"].startswith("exportacoes/"))
        archive = self.directory / result["file"]
        with zipfile.ZipFile(archive) as package:
            self.assertIn("Resíduos em calha", package.read("ocorrencias.csv").decode("utf-8-sig"), "A decisão da página vale sobre o registro.")
        status, data = self.request("/" + result["href"])
        self.assertEqual((status, data[:2]), (200, b"PK"))
        # 2. Laudo HTML gravado em laudos/ e servido pela revisão.
        status, body = self.request("/api/report", {"review": "fixture_v1", "decisions": decisions, "reviewer": "rev01", "client": "CEASA Minas",
                                                   "site": "Pavilhão A", "pdf": False}, self.token)
        self.assertEqual(status, 200, body)
        report = json.loads(body)
        self.assertEqual((report["occurrences"], report["pdf"]), (1, None))
        self.assertTrue(report["html"].startswith("laudos/"))
        status, page = self.request("/r/fixture_v1/" + report["html"])
        self.assertEqual(status, 200)
        self.assertIn("CEASA Minas".encode(), page)
        # Dados técnicos chegam ao laudo; campo com nome errado é recusado, não descartado em silêncio.
        status, body = self.request("/api/report", {"review": "fixture_v1", "decisions": decisions, "pdf": False,
                                                   "details": {"art_number": "MG-123", "inspection_date": "06/08/2025"}}, self.token)
        self.assertEqual(status, 200, body)
        status, page = self.request("/r/fixture_v1/" + json.loads(body)["html"])
        self.assertIn(b"MG-123", page)
        status, body = self.request("/api/report", {"review": "fixture_v1", "pdf": False, "details": {"art_numero": "MG-123"}}, self.token)
        self.assertEqual(status, 400, body)
        self.assertIn("art_numero", body.decode() if isinstance(body, bytes) else body)
        self.assertIn("pendente de validação técnica".encode(), page, "Gravidade e diagnóstico nunca são inferidos.")
        # 3. Lista de arquivos gerados da revisão; outros tipos (ex.: .pt) não são servidos.
        files = json.loads(self.request("/api/reviews/fixture_v1/files")[1])["files"]
        self.assertEqual({item["kind"] for item in files}, {"zip", "html", "json"})
        self.assertEqual(json.loads(self.request("/api/operations?review=fixture_v1")[1])["review_files"][0]["kind"], files[0]["kind"])
        (self.directory / "laudos" / "pesos.pt").write_bytes(b"x")
        self.assertEqual(self.request("/laudos/pesos.pt")[0], 404, "Só os tipos de arquivo gerados são servidos.")
        self.assertEqual(self.request("/api/reviews/outra/files")[0], 404)


if __name__ == "__main__":
    unittest.main()
