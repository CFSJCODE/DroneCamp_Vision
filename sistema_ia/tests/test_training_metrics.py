"""Testes das métricas de treino e avaliação (PR "métricas depois do 614d070a").

O que cobre:
- ``detection_metrics``: previsões perfeitas, classe sem exemplo = não avaliada,
  tipos de falso positivo, faixas de tamanho, recall com limite de FP por foto e
  parcela de etiqueta desconhecida nas condições.
- ``training_progress``: results.csv com o relógio zerado por retomada, melhor época
  e identidade do checkpoint.
- ``run_record``: ficha imutável, conferência que acusa edição e tentativa sem fim.
- ``data_coverage``: fotos, objetos, janelas e repetições por classe.
- ``quality_report``: relatório completo com previsões injetadas e recortes dos erros.
- ``training_monitor``: treino REAL de 2 épocas da Ultralytics (yolo26n.yaml do zero,
  imagens sintéticas de 64 px) com o monitor ligado; confere etapas, passos do
  otimizador e a ficha "concluida". Pule com ``DRONECAMP_SKIP_E2E=1``.

Como rodar: ``python -m unittest tests.test_training_metrics`` dentro de ``sistema_ia``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dronecamp_ia.detection_metrics import evaluate_detections, size_bucket


def _box(class_id, x, y, size=20.0, conf=None):
    """Caixa quadrada ``[x, y, x+size, y+size]``; com ``conf`` vira previsão."""
    box = {"class_id": class_id, "bbox_xyxy": [x, y, x + size, y + size]}
    if conf is not None:
        box["confidence"] = conf
    return box


class DetectionMetricsTests(unittest.TestCase):
    """Régua por foto original (``evaluate_detections``)."""

    names = ["a", "b", "c"]

    def test_perfect_predictions_and_class_without_examples_is_not_evaluated(self):
        truth = [_box(0, 10, 10), _box(1, 100, 100, 120)]
        samples = [{"image": "f1", "truth": truth, "predictions": [dict(box, confidence=0.9) for box in truth]}]
        report = evaluate_detections(samples, self.names)
        a, b, c = (report["per_class"][name] for name in self.names)
        self.assertEqual((a["ap50"], a["ap75"], a["ap50_95"], a["f1"]), (1.0, 1.0, 1.0, 1.0))
        self.assertEqual((a["support_images"], a["support_boxes"], a["tp"], a["fp"], a["fn"]), (1, 1, 1, 0, 0))
        self.assertEqual(c["status"], "nao_avaliada")
        self.assertEqual((c["support_boxes"], c["ap50"], c["f1"]), (0, None, None))
        self.assertEqual(report["total"]["classes_not_evaluated"], ["c"])
        self.assertEqual(report["total"]["map50"], 1.0)
        # Tamanho: 20 px de lado = pequeno (< 32²), 120 px = grande (> 96²).
        self.assertEqual(a["by_size"]["pequeno"]["tp"], 1)
        self.assertEqual(b["by_size"]["grande"]["tp"], 1)
        self.assertEqual(report["protocol"]["unit"], "foto_original")

    def test_false_positive_types_and_omission_reasons(self):
        truth = [_box(0, 0, 0, 40), _box(1, 200, 200, 40)]
        predictions = [
            _box(0, 0, 0, 40, conf=0.9),        # acerto
            _box(0, 1, 1, 40, conf=0.8),        # duplicata do acerto
            _box(0, 200, 200, 40, conf=0.7),    # região certa, classe errada (b vira a)
            _box(0, 15, 15, 40, conf=0.6),      # caixa imprecisa (IoU 0,24 com a)
            _box(2, 400, 400, 40, conf=0.5),    # fundo
        ]
        report = evaluate_detections([{"image": "f", "truth": truth, "predictions": predictions}], self.names)
        self.assertEqual(report["total"]["fp_types"],
                         {"fundo": 1, "classe_trocada": 1, "caixa_imprecisa": 1, "duplicata": 1})
        omissions = [item for item in report["errors"] if item["type"] == "omissao"]
        self.assertEqual([(item["class"], item["fn_reason"]) for item in omissions], [("b", "classe_trocada")])
        self.assertEqual(report["total"]["fp_per_image"], 4.0)

    def test_recall_at_false_alarm_budget_and_unknown_conditions(self):
        samples = []
        for index in range(4):
            samples.append({"image": f"f{index}", "truth": [_box(0, 10, 10)],
                            "predictions": [_box(0, 10, 10, conf=0.9 - index * 0.2), _box(0, 300, 300, conf=0.95)],
                            "conditions": {"material": "metal" if index < 2 else None}})
        report = evaluate_detections(samples, self.names, conf=0.1, fp_budgets=(0.5, 1.0),
                                     condition_fields=("material",))
        budgets = report["operating"]["_total"]["recall_at_fp_per_image"]
        # Os 4 alarmes de 0,95 vêm antes de qualquer acerto: com 0,5 FP/foto nenhum acerto cabe.
        self.assertEqual(budgets["0.5"]["recall"], 0.0)
        self.assertEqual(budgets["1.0"]["recall"], 1.0)
        material = report["conditions"]["material"]
        self.assertEqual(material["unknown_share"], 0.5)
        self.assertEqual(material["values"]["metal"]["images"], 2)
        self.assertEqual(material["values"]["desconhecido"]["images"], 2)

    def test_size_bucket_edges(self):
        self.assertEqual(size_bucket(31 * 31, (32, 96)), "pequeno")
        self.assertEqual(size_bucket(50 * 50, (32, 96)), "medio")
        self.assertEqual(size_bucket(100 * 100, (32, 96)), "grande")


RESULTS_HEADER = "epoch,time,train/box_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,lr/pg0"


def _write_results(run: Path, rows: list[tuple]) -> Path:
    """Grava ``run/fit/results.csv`` com (época, tempo, mAP50-95)."""
    (run / "fit").mkdir(parents=True, exist_ok=True)
    lines = [RESULTS_HEADER] + [f"{e},{t},1.0,0.5,0.5,{m * 2},{m},1.0,0.01" for e, t, m in rows]
    path = run / "fit" / "results.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TrainingProgressTests(unittest.TestCase):
    """Leitura só do results.csv (vale para o 614d070a em andamento)."""

    def test_time_reset_on_resume_and_best_epoch(self):
        from dronecamp_ia.training_progress import epoch_timeline, evolution, read_results

        with tempfile.TemporaryDirectory() as folder:
            # Épocas 1-3 na tentativa 1; queda; retomada refaz a 3 com o relógio zerado.
            path = _write_results(Path(folder), [(1, 100, 0.1), (2, 200, 0.3), (3, 300, 0.2),
                                                 (3, 110, 0.25), (4, 220, 0.28)])
            rows = read_results(path)
            timeline = epoch_timeline(rows)
            self.assertEqual(timeline["attempts"], 2)
            self.assertEqual(timeline["redone_epochs"], [3])
            self.assertEqual(timeline["effective_seconds"], 100 + 100 + 110 + 110)
            report = evolution(rows, patience=10, planned_epochs=120)
            self.assertEqual(report["best_epoch"], 2)
            self.assertEqual(report["epochs_since_best"], 2)

    def test_checkpoint_identity_reads_epoch_of_best(self):
        from dronecamp_ia.training_progress import checkpoint_identity, write_progress

        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "pilot_train_x"
            _write_results(run, [(1, 10, 0.1), (2, 20, 0.4), (3, 30, 0.2)])
            (run / "fit" / "weights").mkdir()
            best = run / "fit" / "weights" / "best.pt"
            best.write_bytes(b"peso")
            info = checkpoint_identity(best)
            self.assertEqual((info["role"], info["epoch"], info["run"]), ("best", 2, str(run)))
            self.assertIn("época 2", info["reason"])
            output = write_progress(Path(folder), run, Path(folder) / "saida")
            report = json.loads((output / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(report["instrumentation"]["state"], "nao_registrado")
            self.assertTrue((output / "progress.md").is_file())
            self.assertFalse((run / "progress.json").exists())  # a pasta do treino só é lida


class RunRecordTests(unittest.TestCase):
    """Ficha imutável (``run_record``)."""

    def test_card_is_write_once_and_verify_detects_edits(self):
        from dronecamp_ia.run_record import end_attempt, run_card_summary, start_attempt, unfinished_attempts, \
            verify_run_card, write_once

        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            attempt = start_attempt(run, "novo", {"epochs": 2, "seed": 0}, weights="yolo26n.pt")
            self.assertEqual(attempt["attempt"], 1)
            self.assertEqual([item["attempt"] for item in unfinished_attempts(run)], [1])
            second = start_attempt(run, "retomada", {"epochs": 2})
            start = json.loads((run / "ficha" / "attempt_002_start.json").read_text(encoding="utf-8"))
            self.assertEqual([item["attempt"] for item in start["previous_unfinished"]], [1])
            end_attempt(second, "concluida", None, {"effective_seconds": 12.5})
            end_attempt(second, "falhou", "segunda chamada é ignorada")
            with self.assertRaises(FileExistsError):
                write_once(run, "attempt_001_start.json", {})
            summary = run_card_summary(run)
            self.assertEqual([item["status"] for item in summary["attempt_list"]], ["sem_registro_de_fim", "concluida"])
            self.assertEqual(verify_run_card(run)["integrity"], "ok")
            path = run / "ficha" / "attempt_002_end.json"
            path.write_text(path.read_text(encoding="utf-8").replace("concluida", "falhou"), encoding="utf-8")
            check = verify_run_card(run)
            self.assertEqual((check["integrity"], check["changed"]), ("alterada", ["attempt_002_end.json"]))


def _pilot_dataset(root: Path) -> Path:
    """Dataset piloto mínimo: 3 fotos 200×200 (2 treino, 1 teste) com pilot.json e labels."""
    from PIL import Image

    entries = []
    labels = {"t0": "0 0.25 0.25 0.1 0.1\n1 0.75 0.75 0.2 0.2", "t1": "0 0.5 0.5 0.1 0.1", "x0": "1 0.5 0.5 0.3 0.3"}
    for stem, split in (("t0", "train"), ("t1", "train"), ("x0", "test")):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (200, 200), (90, 90, 90)).save(root / "images" / split / f"{stem}.jpg")
        (root / "labels" / split / f"{stem}.txt").write_text(labels[stem], encoding="utf-8")
        entries.append({"image": f"images/{split}/{stem}.jpg", "label": f"labels/{split}/{stem}.txt", "split": split,
                        "image_sha256": f"sha_{stem}", "scene_group": f"cena_{stem}", "building_group": "galpao_1"})
    (root / "pilot.json").write_text(json.dumps({"images": entries}), encoding="utf-8")
    data = root / "dataset.yaml"
    data.write_text("path: .\ntrain: images/train\nval: images/test\ntest: images/test\nnames: [a, b]\n", encoding="utf-8")
    return data


class DataCoverageTests(unittest.TestCase):
    """Cobertura por classe, ligando janelas à foto de origem pelo nome."""

    def test_photos_tiles_and_repeats_per_class(self):
        from dronecamp_ia.data_coverage import compare_coverage, coverage_report

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = _pilot_dataset(root / "piloto")
            run, tiles = root / "run", root / "janelas"
            (tiles / "labels" / "train").mkdir(parents=True)
            (tiles / "labels" / "train" / "t0_p00_0_0.txt").write_text("0 0.5 0.5 0.1 0.1", encoding="utf-8")
            (tiles / "labels" / "train" / "t0_p01_100_100.txt").write_text("1 0.5 0.5 0.1 0.1", encoding="utf-8")
            (tiles / "labels" / "train" / "t1_p00_0_0.txt").write_text("", encoding="utf-8")
            run.mkdir()
            (run / "execution.json").write_text(json.dumps({"tiling": {"tiles_directory": str(tiles)}}), encoding="utf-8")
            (run / "train_repeat_factor.txt").write_text("a/t0_p00_0_0.jpg\na/t0_p00_0_0.jpg\na/t0_p01_100_100.jpg\n"
                                                         "a/t1_p00_0_0.jpg\n", encoding="utf-8")
            report = coverage_report(data, ["a", "b", "c"], run)
            train = report["splits"]["train"]
            self.assertEqual((train["totals"]["photos"], train["totals"]["objects"], train["totals"]["tiles"],
                              train["totals"]["negative_tiles"]), (2, 3, 3, 1))
            self.assertEqual((train["classes"]["a"]["photos"], train["classes"]["a"]["objects"],
                              train["classes"]["a"]["tiles"], train["classes"]["a"]["sampled_appearances"],
                              train["classes"]["a"]["max_repeat"]), (2, 2, 1, 2, 2))
            self.assertEqual(train["classes"]["c"]["status"], "sem_exemplos")
            self.assertEqual(report["splits"]["test"]["classes"]["b"]["objects"], 1)
            # Mesma cobertura com mais repetição da classe a = "so_repeticao", não exemplo novo.
            more = json.loads(json.dumps(report))
            more["splits"]["train"]["classes"]["a"]["sampled_appearances"] = 5
            self.assertEqual(compare_coverage(report, more)["a"]["verdict"], "so_repeticao")


class QualityReportTests(unittest.TestCase):
    """``evaluate-detailed`` de ponta a ponta com previsões injetadas."""

    def test_report_crops_errors_and_links_checkpoint(self):
        from dronecamp_ia import quality_report

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = _pilot_dataset(root / "piloto")
            run = root / "runs" / "pilot_train_x"
            _write_results(run, [(1, 10, 0.1), (2, 20, 0.3)])
            (run / "fit" / "weights").mkdir()
            best = run / "fit" / "weights" / "best.pt"
            best.write_bytes(b"peso")
            (run / "ficha").mkdir()
            taxonomy = root / "t.json"
            taxonomy.write_text("{}", encoding="utf-8")
            config = SimpleNamespace(root=root, taxonomy_path=taxonomy, pilot={"suggestion_conf": 0.25},
                                     prediction={"conf": 0.25, "max_det": 300})
            model = SimpleNamespace(dronecamp_runtime="pytorch-cpu")
            loaded = [{"model": model, "config": SimpleNamespace(prediction={"imgsz": 640}), "source": best,
                       "used": best}]

            def predict(_model, _config, image):
                # Foto de teste x0: caixa b humana em [70,70,130,130]; o modelo a chama de "a" e erra o fundo.
                return [{"class_id": 0, "bbox_xyxy": [70.0, 70.0, 130.0, 130.0], "confidence": 0.8},
                        {"class_id": 1, "bbox_xyxy": [0.0, 0.0, 20.0, 20.0], "confidence": 0.6}]

            with patch("dronecamp_ia.model_gate._load_models", return_value=(loaded, {"sha_t0", "sha_t1"})), \
                    patch("dronecamp_ia.config.load_taxonomy", return_value={}), \
                    patch("dronecamp_ia.config.detection_names", return_value=["a", "b"]):
                output = quality_report.evaluate_weights(config, data, [str(best)], ["modelo"], ("test", "train"),
                                                         output=root / "saida", predict=predict)
            report = json.loads((output / "quality.json").read_text(encoding="utf-8"))
            result = report["models"]["modelo"]
            self.assertEqual(len(report["images"]), 1)  # fotos de treino saíram
            self.assertEqual(result["per_class"]["a"]["status"], "nao_avaliada")
            self.assertEqual(result["per_class"]["b"]["fn"], 1)
            self.assertEqual(result["total"]["fp_types"]["classe_trocada"], 1)
            self.assertEqual(result["protocol"]["checkpoint"]["epoch"], 2)
            self.assertEqual(result["conditions"]["edificacao"]["values"]["galpao_1"]["images"], 1)
            self.assertTrue(all(item["checkpoint_epoch"] == 2 for item in result["errors"]))
            self.assertTrue(list((output / "errors").rglob("*.jpg")))
            self.assertTrue((output / "report.md").is_file())
            self.assertEqual(len(list((run / "ficha").glob("evaluation_*.json"))), 1)


def _tiny_dataset(root: Path) -> Path:
    """4 fotos de treino e 2 de validação 64×64 com um quadrado por foto."""
    from PIL import Image, ImageDraw

    for split, count in (("train", 4), ("val", 2)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for index in range(count):
            image = Image.new("RGB", (64, 64), (90, 90, 90))
            ImageDraw.Draw(image).rectangle([10 + index, 10, 40 + index, 40], fill=(220, 40, 40))
            image.save(root / "images" / split / f"{split}_{index}.jpg")
            (root / "labels" / split / f"{split}_{index}.txt").write_text(
                f"0 {(25 + index) / 64:.4f} {25 / 64:.4f} {30 / 64:.4f} {30 / 64:.4f}", encoding="utf-8")
    data = root / "dataset.yaml"
    data.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames: [vermelho]\n", encoding="utf-8")
    return data


@unittest.skipIf(os.environ.get("DRONECAMP_SKIP_E2E") == "1", "E2E desativado por DRONECAMP_SKIP_E2E=1.")
class TrainingMonitorTests(unittest.TestCase):
    """Monitor ligado num treino real curto da Ultralytics (CPU)."""

    def test_real_training_records_stages_steps_and_closed_card(self):
        from ultralytics import YOLO

        from dronecamp_ia.run_record import run_card_summary, start_attempt
        from dronecamp_ia.training_monitor import TrainingMonitor
        from dronecamp_ia.training_progress import progress_report

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = _tiny_dataset(root / "dados")
            run = root / "pilot_train_teste"
            run.mkdir()
            parameters = {"epochs": 2, "imgsz": 64, "batch": 2, "seed": 0}
            attempt = start_attempt(run, "novo", parameters, weights="yolo26n.yaml")
            monitor = TrainingMonitor(run, attempt, {"resource_seconds": 0.2})
            model = YOLO("yolo26n.yaml")
            monitor.attach(model)
            try:
                model.train(data=str(data), epochs=2, imgsz=64, batch=2, device="cpu", workers=0, project=str(run),
                            name="fit", exist_ok=True, plots=False, verbose=False, amp=False, seed=0)
            finally:
                monitor.close("concluida", None)
            epochs = [json.loads(line) for line in (run / "instrumentation" / "epochs.jsonl").read_text(
                encoding="utf-8").splitlines()]
            self.assertEqual([row["epoch"] for row in epochs], [1, 2])
            first = epochs[0]
            self.assertEqual((first["batches"], first["images"]), (2, 4))
            self.assertGreaterEqual(first["optimizer_steps"], 1)
            for stage in ("data_wait", "preprocess", "forward_loss", "backward", "optimizer_step", "batch"):
                self.assertEqual(first["stages"][stage]["n"], 2 if stage != "optimizer_step" else first["optimizer_steps"])
            self.assertGreater(first["validation_seconds"], 0)
            self.assertGreater(first["checkpoint_seconds"], 0)
            self.assertEqual(first["instrumentation_errors"], [])
            # Validação final do best.pt fica à parte, sem virar época 3.
            self.assertTrue((run / "instrumentation" / "final_eval.jsonl").is_file())
            self.assertTrue((run / "instrumentation" / "resources.jsonl").is_file())
            summary = run_card_summary(run)
            self.assertEqual(summary["integrity"], "ok")
            self.assertEqual(summary["attempt_list"][0]["status"], "concluida")
            self.assertEqual(summary["attempt_list"][0]["global_epoch_reached"], 2)
            self.assertTrue((run / "ficha" / "attempt_001_trainer.json").is_file())
            report = progress_report(run)
            self.assertEqual(report["instrumentation"]["state"], "registrado")
            self.assertEqual(report["instrumentation"]["totals"]["batches"], 4)


if __name__ == "__main__":
    unittest.main()
