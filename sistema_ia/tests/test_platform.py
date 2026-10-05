"""Plataforma de operações: painel de treinos, página gerada e servidor local.

Dados sintéticos (sem Ultralytics): confere que o painel lê ``runs/`` sem
alterar nada, que a página embute o instantâneo, e que o servidor só serve
arquivos da revisão/``runs`` e só inicia comandos da lista fechada com token.
"""

from pathlib import Path
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from dronecamp_ia.config import ProjectConfig
from dronecamp_ia.io import file_hash
from dronecamp_ia.operations import collect_comparisons, collect_runs, read_curves
from dronecamp_ia.platform_server import JobRunner, _job_arguments, make_handler
from dronecamp_ia.review_render import refresh_review_page, render_review_package

RESULTS_HEADER = ("epoch,time,train/box_loss,train/cls_loss,train/l1_loss,metrics/precision(B),metrics/recall(B),"
                  "metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/l1_loss,lr/pg0\n")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class PlatformFixture(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="dronecamp plataforma á ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.taxonomy_path = self.root / "configs" / "taxonomy.json"
        write_json(self.taxonomy_path, {"version": "fixture", "classes": [
            {"id": 0, "slug": "telha_quebrada", "label": "Telha quebrada", "phase": 1},
            {"id": 1, "slug": "residuos_calha", "label": "Resíduos em calha", "phase": 1}]})
        self.config = ProjectConfig(root=self.root, model="fixture.pt", requested_model="fixture.pt",
                                    taxonomy_path=self.taxonomy_path, dataset_path=self.root / "dataset.yaml",
                                    device="cpu", prediction={}, training={"epochs": 5, "imgsz": 640, "batch": 2},
                                    pilot={"training": {"epochs": 3}, "suggestion_conf": 0.15})
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
        run = self.root / "runs" / name
        fit = run / "fit"
        fit.mkdir(parents=True)
        # Cabeçalho com espaços, como em algumas versões da Ultralytics.
        lines = [RESULTS_HEADER.replace(",", ",  ")]
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
    def test_curves_tolerate_spaced_headers_and_keep_epoch_order(self) -> None:
        run = self.make_run("pilot_train_20261004T170212Z_dd284411", 3, {"state": "completed", "complete": True})
        curves = read_curves(run / "fit" / "results.csv")
        self.assertEqual([point["epoch"] for point in curves], [1, 2, 3])
        self.assertAlmostEqual(curves[-1]["map50"], 0.3)
        self.assertAlmostEqual(curves[0]["val_cls"], 2.6)

    def test_run_states_best_epoch_and_estimate(self) -> None:
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
        later = {run["id"]: run for run in collect_runs(self.root, now=time.time() + 3600)}
        self.assertEqual(later["pilot_train_20261004T200000Z_aaaaaaaa"]["state"], "interrupted")
        self.assertEqual(list(runs)[0], "pilot_train_20261004T200000Z_aaaaaaaa", "Mais recente primeiro.")

    def test_comparisons_link_weights_to_runs(self) -> None:
        write_json(self.root / "runs" / "compare_fixture.json", {"dataset": "d.yaml", "conf": 0.15, "iou_match": 0.5, "models": [
            {"weights": "runs\\pilot_train_20261004T170212Z_dd284411\\fit\\weights\\best.pt", "weights_sha256": "f" * 64,
             "splits": {"val": {"images": 1, "truth": 2, "found": 1, "suggestions": 2, "correct": 1}}}]})
        comparison = collect_comparisons(self.root)[0]
        self.assertEqual(comparison["models"][0]["run_id"], "pilot_train_20261004T170212Z_dd284411")
        self.assertEqual(comparison["models"][0]["file"], "best.pt")

    def test_page_embeds_snapshot_without_changing_runs_or_browser_data_contract(self) -> None:
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


class ServerTests(PlatformFixture):
    def setUp(self) -> None:
        super().setUp()
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
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method="GET" if body is None else "POST")
        if token:
            request.add_header("X-DroneCamp-Token", token)
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def test_served_page_has_token_and_live_operations(self) -> None:
        status, page = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn(f'content="{self.token}"'.encode(), page)
        status, body = self.request("/api/operations")
        self.assertEqual(json.loads(body)["runs"][0]["id"], "pilot_train_20261004T170212Z_dd284411")

    def test_only_review_and_runs_files_are_served(self) -> None:
        self.assertEqual(self.request("/runs/pilot_train_20261004T170212Z_dd284411/fit/results.csv")[0], 200)
        self.assertEqual(self.request("/../../configs/taxonomy.json")[0], 404)
        self.assertEqual(self.request("/%2e%2e/%2e%2e/%2e%2e/configs/taxonomy.json")[0], 404)
        self.assertEqual(self.request("/runs/pilot_train_20261004T170212Z_dd284411/fit/weights/best.pt")[0], 404)

    def test_changes_require_token_and_allowed_commands(self) -> None:
        self.assertEqual(self.request("/api/jobs", {"kind": "train-pilot"})[0], 403)
        self.assertEqual(self.request("/api/jobs", {"kind": "train-pilot"}, "errado")[0], 403)
        status, body = self.request("/api/jobs", {"kind": "export", "options": {}}, self.token)
        self.assertEqual(status, 400)
        self.assertIn("não permitida", json.loads(body)["error"])
        self.assertEqual(json.loads(self.request("/api/jobs")[1])["state"], "idle")

    def test_job_arguments_are_validated(self) -> None:
        dataset = self.root / "data" / "pilot" / "v1" / "dataset.yaml"
        dataset.parent.mkdir(parents=True)
        dataset.write_text("path: .\n", encoding="utf-8")
        arguments = _job_arguments(self.root, self.registry_path, "train-pilot",
                                   {"data": "data\\pilot\\v1\\dataset.yaml", "epochs": 10, "imgsz": 640, "batch": 2,
                                    "weights": "runs/pilot_train_20261004T170212Z_dd284411/fit/weights/best.pt"})
        self.assertEqual(arguments[:2], ["--data", "data/pilot/v1/dataset.yaml"])
        self.assertIn("--epochs", arguments)
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


if __name__ == "__main__":
    unittest.main()


class MultiReviewServerTests(PlatformFixture):
    """Duas revisões no mesmo servidor: cada uma na sua página, fotos e arquivos separados."""

    def setUp(self) -> None:
        super().setUp()
        from http.server import ThreadingHTTPServer
        render_review_package(self.config, self.registry_path)
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

    request = ServerTests.request

    def test_each_review_has_its_page_picker_and_files(self) -> None:
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
        feedback = {"schema_version": 1, "registry_sha256": file_hash(self.other_registry), "reviewer_id": "r1",
                    "images": [{"image_sha256": "x", "status": "ambiguous"}]}
        status, body = self.request("/api/feedback", feedback, self.token)
        self.assertEqual(status, 200, body)
        self.assertEqual((self.root / json.loads(body)["saved"]).parent, self.other_directory / "feedback_inbox")

    def test_unknown_review_is_rejected_for_jobs(self) -> None:
        status, body = self.request("/api/jobs", {"kind": "suggest", "options": {"review": "outra", "weights": "x.pt"}}, self.token)
        self.assertEqual(status, 400)
        self.assertIn("não aberta", json.loads(body)["error"])