"""Relatório comparativo: lê os quatro formatos, um modelo = um rótulo e uma cor, HTML autônomo."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from dronecamp_ia.config import detection_names, load_config, load_taxonomy
from dronecamp_ia.model_report import ModelCatalog, build_model_report

RUN_A = "pilot_train_20261004T170212Z_aaaaaaaa"
RUN_B = "pilot_train_20261004T170212Z_bbbbbbbb"
SHA_A, SHA_B = "a" * 64, "b" * 64
# Mesmo arquivo gravado por máquinas diferentes: absoluto com "\" no Windows e relativo com "/".
WEIGHTS_A_WINDOWS = f"E:\\Acadêmico\\sistema_ia\\runs\\{RUN_A}\\fit\\weights\\best.pt"
WEIGHTS_A_RELATIVE = f"runs\\{RUN_A}\\fit\\weights\\best.pt"
WEIGHTS_B = f"runs/{RUN_B}/fit/weights/best.pt"
CSV_HEADER = ("epoch,time,train/box_loss,train/cls_loss,train/l1_loss,metrics/precision(B),metrics/recall(B),"
              "metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/l1_loss")
EXPECTED_CHARTS = ("overall_metrics.png", "per_class_recall.png", "per_class_errors.png", "training_curves.png",
                   "gpu_eval_curves.png", "compare_found.png")


def class_metrics(tp, fp, fn, boxes, ap50):
    recall = round(tp / boxes, 4) if boxes else None
    precision = round(tp / (tp + fp), 4) if tp + fp else None
    return {"tp": tp, "fp": fp, "fn": fn, "human_boxes": boxes, "precision": precision, "recall": recall, "ap50": ap50}


def gate_metrics(names, measured):
    """Métricas no formato do gate: classes não medidas ficam com zeros e None."""
    metrics = {name: class_metrics(0, 0, 0, 0, None) for name in names}
    for name, values in measured.items():
        metrics[name] = class_metrics(*values)
    tp = sum(item["tp"] for item in metrics.values())
    fp = sum(item["fp"] for item in metrics.values())
    fn = sum(item["fn"] for item in metrics.values())
    boxes = sum(item["human_boxes"] for item in metrics.values())
    aps = [item["ap50"] for item in metrics.values() if item["ap50"] is not None]
    metrics["_total"] = {"tp": tp, "fp": fp, "fn": fn, "human_boxes": boxes, "precision": round(tp / (tp + fp), 4),
                         "recall": round(tp / boxes, 4), "map50": round(sum(aps) / len(aps), 4),
                         "classes_measured": len(aps)}
    return metrics


def write_fake_run(root, name, epochs, with_gpu_curve):
    run = root / "runs" / name
    (run / "fit").mkdir(parents=True)
    rows = [CSV_HEADER]
    for epoch in range(1, epochs + 1):
        map50 = 0.1 * epoch
        rows.append(f"{epoch},{40.0 * epoch},1.9,6.3,0.01,{0.3},{0.2},{map50},{map50 / 2},3.1,15.9,0.04")
    (run / "fit/results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (run / "execution.json").write_text(json.dumps({
        "parameters": {"epochs": 80, "imgsz": 640, "batch": 4}, "runtime": {"checkpoint": "models\\yolo26l.pt"}}),
        encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")
    if with_gpu_curve:
        (run / "gpu_eval").mkdir()
        lines = [json.dumps({"epoch": epoch, "split": "test", "map50": 0.05 * epoch, "precision": 0.4, "recall": 0.1,
                             "per_class": {}, "seconds": 12.5, "onnx_sha256": "c" * 64}) for epoch in (1, 2, 3)]
        lines.append(json.dumps({"epoch": 4, "error": "ONNX Runtime falhou"}))
        lines.append('{"epoch": 5, "split": "test", "map50": 0.2')  # linha truncada
        (run / "gpu_eval/eval_curve.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run


class ModelReportTests(unittest.TestCase):
    """Um relatório completo é montado uma vez; cada teste confere uma parte dele."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory()
        root = Path(cls.temporary.name)
        base = load_config()
        cls.config = replace(base, root=root)
        cls.names = detection_names(load_taxonomy(base.taxonomy_path))
        names = cls.names
        measured = {names[1]: (0, 0, 1, 1, 0.0), names[3]: (3, 1, 2, 5, 0.61), names[5]: (4, 2, 8, 12, 0.35),
                    names[2]: (0, 3, 0, 0, None)}
        candidate = {names[1]: (1, 0, 0, 1, 1.0), names[3]: (2, 2, 3, 5, 0.42), names[5]: (6, 5, 6, 12, 0.44)}
        cls.gate = root / "gate.json"
        cls.gate.write_text(json.dumps({
            "conf": 0.15, "splits": ["test"], "images_evaluated": 9, "images_excluded_seen_in_training": 21,
            "tiling": None, "adopt": False,
            "reasons": ["mAP50 0.620 não supera o atual 0.480 em 0.010.", "Regressão de recall em: reparo_telha."],
            "regressions": [{"class": names[3], "recall_before": 0.6, "recall_after": 0.4, "human_boxes": 5}],
            "baseline": {"weights": WEIGHTS_A_WINDOWS, "sha256": SHA_A, "metrics": gate_metrics(names, measured)},
            "candidate": {"weights": WEIGHTS_B, "sha256": SHA_B, "metrics": gate_metrics(names, candidate)},
        }, ensure_ascii=False), encoding="utf-8")
        cls.compare = root / "compare_test.json"
        cls.compare.write_text(json.dumps({
            "dataset": "data\\pilot\\ceasa_v9\\dataset.yaml", "pilot_sha256": "f" * 64, "conf": 0.15, "iou_match": 0.5,
            "models": [
                {"weights": WEIGHTS_A_RELATIVE, "weights_sha256": SHA_A, "imgsz": 640, "splits": {
                    "test": {"images": 9, "truth": 33, "found": 2, "suggestions": 6, "correct": 2},
                    "val": {"images": 5, "truth": 7, "found": 2, "suggestions": 3, "correct": 2},
                    "train (visto no treino)": {"images": 20, "truth": 114, "found": 31, "suggestions": 90, "correct": 31}}},
                {"weights": WEIGHTS_B, "weights_sha256": SHA_B, "imgsz": 640, "splits": {
                    "test": {"images": 9, "truth": 33, "found": 3, "suggestions": 29, "correct": 3},
                    "val": {"images": 7, "truth": 12, "found": 1, "suggestions": 28, "correct": 1},
                    "train (visto no treino)": {"images": 21, "truth": 114, "found": 83, "suggestions": 205, "correct": 83}}},
            ]}, ensure_ascii=False), encoding="utf-8")
        cls.eval = root / "eval_por_classe_test.json"
        cls.eval.write_text(json.dumps({"conf": 0.15, "iou": 0.5, "models": {
            WEIGHTS_A_RELATIVE: {"test": {
                "overall": {"P": 0.425, "R": 0.134, "mAP50": 0.089, "mAP50-95": 0.04},
                "per_class_ultralytics": {names[5]: {"P": 0.175, "R": 0.25, "mAP50": 0.087}},
                "per_class_at_conf": {names[5]: {"truth": 12, "tp": 4, "fp": 2, "fn": 8},
                                      names[0]: {"truth": 0, "tp": 0, "fp": 3, "fn": 0}}}},
            WEIGHTS_B: {"test": {
                "overall": {"P": 0.3, "R": 0.2, "mAP50": 0.12, "mAP50-95": 0.05},
                "per_class_ultralytics": {names[5]: {"P": 0.2, "R": 0.5, "mAP50": 0.3}},
                "per_class_at_conf": {names[5]: {"truth": 12, "tp": 6, "fp": 5, "fn": 6}}}},
        }}, ensure_ascii=False), encoding="utf-8")
        cls.run_a = write_fake_run(root, RUN_A, 5, with_gpu_curve=True)
        cls.run_b = write_fake_run(root, RUN_B, 5, with_gpu_curve=False)
        cls.output = build_model_report(cls.config, root / "report", gates=[str(cls.gate)], compares=[cls.compare],
                                        evals=[cls.eval], runs=[cls.run_a, str(cls.run_b)], title="Teste")
        cls.models = json.loads((cls.output / "models.json").read_text(encoding="utf-8"))
        cls.html = (cls.output / "report.html").read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_writes_json_html_and_every_chart(self):
        self.assertTrue((self.output / "models.json").is_file())
        self.assertTrue((self.output / "report.html").is_file())
        for name in EXPECTED_CHARTS:
            with self.subTest(chart=name):
                self.assertTrue((self.output / name).is_file(), name)
                with Image.open(self.output / name) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertGreaterEqual(image.width, 300)
        self.assertEqual(self.models["charts"], list(EXPECTED_CHARTS))

    def test_same_model_gets_one_label_and_one_color_across_inputs(self):
        models = self.models["models"]
        self.assertEqual(len(models), 2)
        self.assertEqual([model["label"] for model in models], ["aaaaaaaa · best.pt", "bbbbbbbb · best.pt"])
        self.assertEqual(len({model["color"] for model in models}), 2)
        self.assertEqual([model["id"] for model in models], [SHA_A, SHA_B])
        self.assertEqual(models[0]["run_id"], "aaaaaaaa")
        self.assertNotIn("\\", models[0]["weights"])
        # Comparação, avaliação e treinos apontam para os mesmos dois modelos.
        self.assertEqual(self.models["compares"][0]["models"], [SHA_A, SHA_B])
        self.assertEqual(set(self.models["evals"][0]["models"]), {SHA_A, SHA_B})
        self.assertEqual([run["model"] for run in self.models["runs"]], [SHA_A, SHA_B])

    def test_normalized_gate_compare_eval_and_runs(self):
        gate = self.models["gates"][0]
        self.assertEqual((gate["baseline"], gate["candidate"]), (SHA_A, SHA_B))
        self.assertFalse(gate["adopt"])
        self.assertEqual(gate["images_excluded_seen_in_training"], 21)
        self.assertEqual(gate["per_class"][self.names[5]][SHA_A]["human_boxes"], 12)
        self.assertAlmostEqual(gate["overall"][SHA_B]["map50"], 0.62)
        compare = self.models["compares"][0]
        self.assertEqual(compare["dataset"], "data/pilot/ceasa_v9/dataset.yaml")
        self.assertTrue(compare["splits"]["train (visto no treino)"][SHA_A]["seen_in_training"])
        self.assertFalse(compare["splits"]["test"][SHA_A]["seen_in_training"])
        self.assertAlmostEqual(compare["splits"]["test"][SHA_B]["found_fraction"], 3 / 33, places=4)
        eval_a = self.models["evals"][0]["models"][SHA_A]["test"]
        self.assertAlmostEqual(eval_a["per_class_at_conf"][self.names[5]]["recall"], 4 / 12, places=4)
        self.assertEqual(eval_a["overall"]["map50"], 0.089)
        run_a, run_b = self.models["runs"]
        self.assertEqual((run_a["id"], run_b["id"]), ("aaaaaaaa", "bbbbbbbb"))
        self.assertEqual(run_a["state"], "completed")
        self.assertEqual(run_a["epochs"], 5)
        self.assertAlmostEqual(run_a["best_val_map50"], 0.5)
        self.assertEqual(run_a["best_epoch"], 5)
        self.assertEqual(run_a["parameters"], {"epochs": 80, "imgsz": 640, "batch": 4})
        # Três linhas válidas; a linha com "error" e a truncada ficam de fora.
        self.assertEqual([point["epoch"] for point in run_a["gpu_eval"]], [1, 2, 3])
        self.assertEqual(run_a["gpu_eval"][0]["split"], "test")
        self.assertEqual(run_b["gpu_eval"], [])

    def test_html_is_standalone_and_in_portuguese(self):
        self.assertIn("aaaaaaaa · best.pt", self.html)
        self.assertIn("bbbbbbbb · best.pt", self.html)
        self.assertIn("excluídas", self.html)
        self.assertIn("produção", self.html)
        self.assertIn("visto no treino", self.html)
        self.assertIn("pouco medidas", self.html)
        self.assertIn("não medida", self.html)
        self.assertEqual(self.html.count("data:image/png;base64,"), len(EXPECTED_CHARTS))
        self.assertNotIn("<link", self.html)
        self.assertNotIn("<script", self.html)
        self.assertIn('<meta charset="utf-8">', self.html)
        # O caminho do Windows passa pelo escape do HTML, com "/" normalizado.
        self.assertIn("E:/Acadêmico/sistema_ia/runs/", self.html)

    def test_notes_cover_gate_decision_and_poorly_measured_classes(self):
        notes = self.models["notes"]
        self.assertIn("Nenhum resultado aqui aprova um modelo para produção: a avaliação usa uma única edificação.", notes)
        self.assertTrue(any(note.startswith("Fotos vistas no treino") and "(21 excluídas)" in note for note in notes))
        decision = next(note for note in notes if note.startswith("Gate "))
        self.assertIn("não", decision)
        self.assertIn("Regressão de recall em: reparo_telha.", decision)
        few = next(note for note in notes if "pouco medidas" in note)
        self.assertIn(f"{self.names[1]} (1)", few)
        self.assertNotIn(f"{self.names[5]} (12)", few)

    def test_runs_none_discovers_pilot_runs_under_root(self):
        output = build_model_report(self.config, gates=[self.gate])
        self.addCleanup(lambda: None)
        self.assertEqual(output.parent, Path(self.config.root) / "runs")
        self.assertTrue(output.name.startswith("report_"))
        models = json.loads((output / "models.json").read_text(encoding="utf-8"))
        self.assertEqual([run["id"] for run in models["runs"]], ["aaaaaaaa", "bbbbbbbb"])
        self.assertIn("training_curves.png", models["charts"])

    def test_gpu_chart_absent_when_no_run_has_the_curve(self):
        output = build_model_report(self.config, Path(self.config.root) / "report_sem_gpu", gates=[self.gate],
                                    runs=[self.run_b])
        self.assertFalse((output / "gpu_eval_curves.png").exists())
        self.assertTrue((output / "training_curves.png").is_file())
        models = json.loads((output / "models.json").read_text(encoding="utf-8"))
        self.assertNotIn("gpu_eval_curves.png", models["charts"])
        self.assertNotIn("compare_found.png", models["charts"])

    def test_refuses_without_any_data(self):
        with self.assertRaises(ValueError):
            build_model_report(self.config, Path(self.config.root) / "report_vazio", runs=[])
        with TemporaryDirectory() as empty:
            with self.assertRaises(ValueError):
                build_model_report(replace(self.config, root=Path(empty)))
            self.assertFalse((Path(empty) / "runs").exists())

    def test_measurement_is_primary_and_older_inputs_become_archive(self):
        names = self.names
        root = Path(self.config.root)
        series = []
        for weights, sha, regime, measured in (
                (WEIGHTS_A_WINDOWS, SHA_A, "janelas", {names[3]: (3, 1, 2, 5, 0.61), names[5]: (4, 2, 8, 12, 0.35),
                                                        names[1]: (1, 0, 1, 2, 0.5), names[0]: (0, 4, 0, 0, None)}),
                (WEIGHTS_B, SHA_B, "janelas", {names[3]: (2, 2, 3, 5, 0.42), names[5]: (6, 5, 6, 12, 0.44),
                                               names[1]: (0, 0, 2, 2, 0.0)}),
                (WEIGHTS_B, SHA_B, "inteira", {names[3]: (1, 0, 4, 5, 0.2), names[5]: (2, 1, 10, 12, 0.1),
                                               names[1]: (0, 0, 2, 2, 0.0)})):
            metrics = gate_metrics(names, measured)
            series.append({"weights": weights, "sha256": sha, "source_checkpoint": weights, "runtime": "pytorch",
                           "regime": regime, "metrics": metrics,
                           "strata": {"fotos_fatiadas": metrics, "fotos_inteiras": gate_metrics(names, {names[5]: (1, 1, 2, 3, 0.3)})}})
        measurement = root / "measurement.json"
        measurement.write_text(json.dumps({
            "schema_version": 1, "dataset": "data\\pilot\\ceasa_v9\\dataset.yaml", "pilot_sha256": "f" * 64,
            "splits": ["test"], "conf": 0.15, "ap_conf": 0.001, "tiling": {"size": 1024}, "images_evaluated": 11,
            "images_excluded_seen_in_training": 19, "strata": {"fotos_fatiadas": 8, "fotos_inteiras": 3},
            "models": series}, ensure_ascii=False), encoding="utf-8")
        output = build_model_report(self.config, root / "report_medicao", measurements=[measurement],
                                    gates=[self.gate], compares=[self.compare], runs=[self.run_a])
        models = json.loads((output / "models.json").read_text(encoding="utf-8"))
        labels = [model["label"] for model in models["models"]]
        self.assertEqual(labels, ["aaaaaaaa · best.pt · janelas", "bbbbbbbb · best.pt · janelas",
                                  "bbbbbbbb · best.pt · inteira", "aaaaaaaa · best.pt", "bbbbbbbb · best.pt"])
        colors = [model["color"] for model in models["models"]]
        self.assertEqual(len(set(colors)), 5)
        self.assertEqual(colors[1:3], ["#eb6834", "#1baf7a"])  # dois regimes do mesmo peso: cores vizinhas
        self.assertEqual(models["charts"], [
            "overall_metrics.png", "per_class_recall.png", "per_class_errors.png", "strata_recall.png",
            "overall_metrics_arquivo.png", "per_class_recall_arquivo.png", "per_class_errors_arquivo.png",
            "training_curves.png", "gpu_eval_curves.png", "compare_found_arquivo.png"])
        for name in models["charts"]:
            with Image.open(output / name) as image:
                self.assertGreaterEqual(image.width, 300, name)
        self.assertFalse((output / "compare_found.png").exists())
        measured = models["measurements"][0]
        self.assertEqual(len(measured["models"]), 3)
        self.assertEqual(measured["strata"]["fotos_inteiras"][measured["models"][0]]["recall"], round(1 / 3, 4))
        page = (output / "report.html").read_text(encoding="utf-8")
        self.assertIn("Medições anteriores (conjuntos de fotos diferentes; não comparáveis entre si)", page)
        self.assertIn("bbbbbbbb · best.pt · inteira", page)
        self.assertIn("n&lt;3: pouco medida", page)
        self.assertIn("(19 excluídas)", page)
        self.assertIn("Recall [IC95] · aaaaaaaa · best.pt · janelas", page)
        # IC de Wilson para 4/12 e AP50 só com 3 ou mais caixas.
        self.assertIn("0.333 [0.14–0.61]", page)
        notes = models["notes"]
        self.assertTrue(any("mesmas fotos" in note for note in notes))

    def test_catalog_merges_windows_and_relative_paths_without_sha(self):
        catalog = ModelCatalog()
        first = catalog.register(WEIGHTS_A_WINDOWS)
        self.assertEqual(catalog.register(WEIGHTS_A_RELATIVE), first)
        self.assertEqual(catalog.register(WEIGHTS_A_RELATIVE, SHA_A), first)
        self.assertNotEqual(catalog.register(WEIGHTS_B, SHA_B), first)
        self.assertEqual(catalog.by_id(first)["sha256"], SHA_A)
        self.assertEqual(catalog.by_id(first)["label"], "aaaaaaaa · best.pt")
        self.assertEqual(catalog.register("models/yolo26l.pt"), "models/yolo26l.pt")
        self.assertEqual(catalog.by_id("models/yolo26l.pt")["label"], "yolo26l.pt")


if __name__ == "__main__":
    unittest.main()
