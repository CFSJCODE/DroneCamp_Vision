"""Aprendizado mútuo (DML) entre YOLO26, RF-DETR e D-FINE, controlador RL e rodadas na rede local.

Módulos cobertos (um grupo de testes por assunto):
    - ``dronecamp_ia.dml_controller``   -> ``DiscountedUCBTests`` (bandit do peso de consistência);
    - ``dronecamp_ia.mutual_learning``  -> ``FusionTests`` (fusão das previsões dos pares),
      ``ConsistencyLossTests`` (perda e gradiente), ``EvaluationTests`` (régua por classe),
      ``PeerCacheTests`` (cache .npz e espelhos);
    - ``dronecamp_ia.dml_lan``          -> ``CoordinatorTests`` (rodadas, impressão digital, token);
    - integração com os três modelos reais (pesos aleatórios, sem internet) ->
      ``ThreeModelIntegrationTests``: treino online, destilação, cache e rede local
      com um coordenador e dois trabalhadores no mesmo processo. Pulado se faltar
      ``rfdetr`` ou ``transformers`` (``pip install -e ".[dml]"``).

Como rodar (na pasta ``sistema_ia``):
    python -m unittest tests.test_mutual_learning
    python -m unittest tests.test_mutual_learning.FusionTests

Preparação: ``_synthetic_dataset`` grava num diretório temporário 6 fotos 320×320 com
quadrados coloridos e labels YOLO (2 classes), em train/val, e o ``dataset.yaml``.

EXTENSAO: família nova em detector_adapters.FAMILIES: acrescente o spec dela em
``TINY_MODELS`` (com pesos aleatórios) e os testes de integração passam a cobri-la.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest

import numpy as np

from dronecamp_ia.dml_controller import DiscountedUCB, reward_from_metrics

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_ALL = HAS_TORCH and all(importlib.util.find_spec(name) for name in ("ultralytics", "rfdetr", "transformers", "scipy"))

# Três modelos minúsculos com pesos aleatórios (não baixam nada).
TINY_MODELS = [
    {"name": "yolo26", "family": "yolo26", "weights": "yolo26n.yaml"},
    {"name": "rfdetr", "family": "rfdetr", "variant": "nano", "weights": "none"},
    {"name": "dfine", "family": "dfine", "variant": "random-small"},
]


def _synthetic_dataset(root: Path) -> Path:
    """Grava 4 fotos de treino e 2 de validação com quadrados e as labels YOLO; devolve o dataset.yaml."""
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(0)
    for split, count in (("train", 4), ("val", 2)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for index in range(count):
            image = Image.new("RGB", (320, 320), (90, 90, 90))
            draw = ImageDraw.Draw(image)
            lines = []
            for class_id, color in ((0, (220, 40, 40)), (1, (40, 40, 220))):
                x, y = rng.integers(20, 220, size=2)
                size = int(rng.integers(40, 80))
                draw.rectangle([x, y, x + size, y + size], fill=color)
                lines.append(f"{class_id} {(x + size / 2) / 320:.5f} {(y + size / 2) / 320:.5f} {size / 320:.5f} {size / 320:.5f}")
            image.save(root / "images" / split / f"{split}_{index}.jpg")
            (root / "labels" / split / f"{split}_{index}.txt").write_text("\n".join(lines), encoding="utf-8")
    data = root / "dataset.yaml"
    data.write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/val\nnames: [vermelho, azul]\n",
                    encoding="utf-8")
    return data


def _tiny_settings() -> dict:
    """Configuração de teste: 2 épocas curtas, τ baixo (modelos aleatórios têm confiança baixa)."""
    from dronecamp_ia.mutual_learning import DEFAULT_SETTINGS, FAMILY_DEFAULTS, _merge

    settings = _merge(DEFAULT_SETTINGS, {"imgsz": 320, "train": {"epochs": 2, "batch": 2},
                                         "consistency": {"ramp_epochs": 0, "tau": 0.0001, "anchor": "all"}})
    settings["models"] = [{**FAMILY_DEFAULTS[model["family"]], **model} for model in TINY_MODELS]
    return settings


# ---------------------------------------------------------------------------
# Controlador RL
# ---------------------------------------------------------------------------

class DiscountedUCBTests(unittest.TestCase):
    def test_untried_arms_follow_list_order(self):
        policy = DiscountedUCB([0.5, 0.0, 1.0])
        chosen = []
        for step in range(3):
            arm = policy.select()
            chosen.append(arm)
            policy.update(arm, 0.0, step)
        self.assertEqual(chosen, [0, 1, 2])

    def test_prefers_arm_with_higher_reward(self):
        policy = DiscountedUCB([0.0, 0.5], gamma=1.0, exploration=0.0)
        policy.update(0, -0.02)
        policy.update(1, 0.03)
        self.assertEqual(policy.select(), 1)

    def test_discount_forgets_old_rewards(self):
        # Braço 1 foi ótimo no começo e ruim depois; com γ baixo o controlador muda de ideia.
        policy = DiscountedUCB([0.0, 1.0], gamma=0.5, exploration=0.0)
        policy.update(1, 0.10)
        policy.update(0, 0.01)
        for _ in range(4):
            policy.update(1, -0.05)
        self.assertEqual(policy.select(), 0)

    def test_round_trip_and_summary(self):
        policy = DiscountedUCB()
        policy.update(policy.select(), 0.02, 0)
        clone = DiscountedUCB.from_dict(json.loads(json.dumps(policy.to_dict())))
        self.assertEqual(clone.summary(), policy.summary())
        self.assertEqual(clone.summary()["0.5"]["uses"], 1)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            DiscountedUCB([0.5, 0.5])
        with self.assertRaises(ValueError):
            DiscountedUCB([0.5], gamma=0)

    def test_reward_is_clipped_delta(self):
        self.assertAlmostEqual(reward_from_metrics(0.20, 0.25), 0.05)
        self.assertEqual(reward_from_metrics(None, 0.3), 0.0)
        self.assertEqual(reward_from_metrics(0.0, 5.0), 1.0)


# ---------------------------------------------------------------------------
# Fusão, perda e avaliação (só torch)
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_ALL, "precisa de torch, torchvision e scipy")
class FusionTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch

    def test_agreeing_peers_become_one_target(self):
        from dronecamp_ia.mutual_learning import fuse_peer_predictions

        t = self.torch
        a = (t.tensor([[0.5, 0.5, 0.2, 0.2]]), t.tensor([[0.9, 0.1]]))
        b = (t.tensor([[0.51, 0.5, 0.2, 0.2]]), t.tensor([[0.7, 0.3]]))
        boxes, probs, weights = fuse_peer_predictions([a, b], tau=0.3)
        self.assertEqual(boxes.shape[0], 1)
        self.assertTrue(t.allclose(probs[0], t.tensor([0.8, 0.2])))
        self.assertAlmostEqual(float(weights[0]), 0.8, places=5)  # confiança média × 2/2 pares

    def test_disagreeing_peers_get_half_weight_and_tau_filters(self):
        from dronecamp_ia.mutual_learning import fuse_peer_predictions

        t = self.torch
        a = (t.tensor([[0.2, 0.2, 0.1, 0.1], [0.6, 0.6, 0.1, 0.1]]), t.tensor([[0.8, 0.0], [0.1, 0.1]]))
        b = (t.tensor([[0.8, 0.8, 0.1, 0.1]]), t.tensor([[0.0, 0.6]]))
        boxes, probs, weights = fuse_peer_predictions([a, b], tau=0.3)
        self.assertEqual(boxes.shape[0], 2)  # a 2.ª previsão de a ficou abaixo de τ
        self.assertTrue(t.allclose(weights.sort().values, t.tensor([0.3, 0.4])))
        self.assertTrue(t.allclose(probs.sum(0), t.tensor([0.4, 0.3])))

    def test_anchor_keeps_only_targets_on_human_boxes(self):
        from dronecamp_ia.mutual_learning import anchor_to_truth

        t = self.torch
        boxes = t.tensor([[0.1, 0.1, 0.3, 0.3], [0.6, 0.6, 0.9, 0.9]])
        kept = anchor_to_truth(boxes, t.ones(2, 2), t.ones(2), t.tensor([[0.1, 0.1, 0.31, 0.3]]))
        self.assertEqual(kept[0].shape[0], 1)
        self.assertTrue(t.equal(kept[0][0], boxes[0]))


@unittest.skipUnless(HAS_ALL, "precisa de torch, torchvision e scipy")
class ConsistencyLossTests(unittest.TestCase):
    def test_kl_is_zero_when_student_agrees(self):
        import torch

        from dronecamp_ia.mutual_learning import bernoulli_kl

        target = torch.tensor([[0.8, 0.1]])
        self.assertAlmostEqual(float(bernoulli_kl(target, torch.logit(target))[0]), 0.0, places=5)

    def test_loss_pulls_student_towards_peers(self):
        import torch

        from dronecamp_ia.mutual_learning import consistency_loss, cxcywh_to_xyxy

        torch.manual_seed(0)
        boxes = torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]], requires_grad=True)
        logits = torch.zeros(2, 2, requires_grad=True)
        target_boxes = cxcywh_to_xyxy(torch.tensor([[0.32, 0.31, 0.22, 0.2]]))
        target_probs, weights = torch.tensor([[0.9, 0.05]]), torch.tensor([1.0])
        optimizer = torch.optim.Adam([boxes, logits], lr=0.01)
        first = None
        for _ in range(400):
            loss, stats = consistency_loss(boxes, logits, target_boxes, target_probs, weights)
            first = first if first is not None else float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertEqual(stats["matched"], 1)
        self.assertLess(float(loss), first * 0.2)
        self.assertGreater(float(logits[0, 0].sigmoid()), 0.85)  # a query casada é a 0 (perto do alvo)

    def test_no_target_gives_zero_with_graph(self):
        import torch

        from dronecamp_ia.mutual_learning import consistency_loss

        logits = torch.zeros(3, 2, requires_grad=True)
        loss, stats = consistency_loss(torch.rand(3, 4), logits, torch.zeros(0, 4), torch.zeros(0, 2), torch.zeros(0))
        loss.backward()
        self.assertEqual((float(loss), stats["matched"]), (0.0, 0))

    def test_ramp(self):
        from dronecamp_ia.mutual_learning import ramp_weight

        self.assertAlmostEqual(ramp_weight(4, 5, 0.5), 0.5)
        self.assertLess(ramp_weight(0, 5, 0.5), 0.05)
        self.assertEqual(ramp_weight(0, 0, 0.7), 0.7)


@unittest.skipUnless(HAS_ALL, "precisa de torch, torchvision e scipy")
class EvaluationTests(unittest.TestCase):
    def test_perfect_predictions_score_one(self):
        from dronecamp_ia.mutual_learning import summarize_detections

        truth = [{"class_id": 0, "bbox_xyxy": [10, 10, 50, 50]}, {"class_id": 1, "bbox_xyxy": [60, 60, 90, 90]}]
        predictions = [dict(box, confidence=0.9) for box in truth]
        result = summarize_detections([(predictions, truth)], ["a", "b", "c"], 0.25)
        self.assertEqual(result["map50"], 1.0)
        self.assertIsNone(result["per_class"]["c"]["ap50"])  # classe sem caixa humana fica fora da média

    def test_complementarity_union(self):
        from dronecamp_ia.mutual_learning import complementarity

        truth = [{"class_id": 0, "bbox_xyxy": [0, 0, 10, 10]}, {"class_id": 0, "bbox_xyxy": [20, 20, 30, 30]}]
        first = {"_detections": [([dict(truth[0], confidence=0.9)], truth)]}
        second = {"_detections": [([dict(truth[1], confidence=0.9)], truth)]}
        report = complementarity({"a": first, "b": second}, 0.25)
        self.assertEqual(report["recall"], {"a": 0.5, "b": 0.5})
        self.assertEqual(report["union_recall"], 1.0)
        self.assertEqual(report["jaccard"]["a|b"], 0.0)


class _FakeAdapter:
    """Adaptador falso: devolve sempre a mesma previsão (para o cache)."""

    name, family, num_classes, device = "falso", "yolo26", 2, "cpu"

    def predict_raw(self, images):
        import torch

        from dronecamp_ia.detector_adapters import RawPredictions

        count = images.shape[0]
        boxes = torch.tensor([[[0.25, 0.4, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]]]).repeat(count, 1, 1)
        logits = torch.tensor([[[3.0, -3.0], [-9.0, -9.0]]]).repeat(count, 1, 1)
        return RawPredictions(boxes, logits)


@unittest.skipUnless(HAS_ALL, "precisa de torch")
class PeerCacheTests(unittest.TestCase):
    def test_cache_round_trip_with_flips(self):
        from dronecamp_ia.mutual_learning import PeerCache, TileDataset, build_peer_cache

        with tempfile.TemporaryDirectory() as folder:
            data = _synthetic_dataset(Path(folder))
            dataset = TileDataset(data, "train", 320)
            path = build_peer_cache(_FakeAdapter(), dataset, Path(folder) / "cache", batch=3, min_conf=0.05)
            cache = PeerCache(path)
            key = TileDataset(data, "train", 320)[0][2]["key"]
            boxes, probs = cache.lookup(key)
            self.assertEqual(boxes.shape[0], 1)  # a 2.ª previsão (confiança ~0) não é guardada
            flipped, _ = cache.lookup(key, hflip=True, vflip=True)
            self.assertAlmostEqual(float(flipped[0, 0]), 0.75, places=3)
            self.assertAlmostEqual(float(flipped[0, 1]), 0.6, places=3)
            self.assertIsNone(cache.lookup("nao_existe.jpg"))

    def test_dataset_flip_moves_boxes(self):
        from dronecamp_ia.mutual_learning import TileDataset

        with tempfile.TemporaryDirectory() as folder:
            data = _synthetic_dataset(Path(folder))
            plain = TileDataset(data, "train", 320)[0]
            flipped = TileDataset(data, "train", 320, augment=True, hflip=1.0)[0]
            self.assertTrue(flipped[2]["hflip"])
            self.assertAlmostEqual(float(flipped[1]["boxes"][0, 0]), 1 - float(plain[1]["boxes"][0, 0]), places=5)
            self.assertTrue((flipped[0].flip(-1) == plain[0]).all())


# ---------------------------------------------------------------------------
# Coordenador da rede local (sem modelos)
# ---------------------------------------------------------------------------

def _npz_bytes() -> bytes:
    import io

    buffer = io.BytesIO()
    np.savez(buffer, keys=np.array(["a.jpg"]), offsets=np.array([0, 0]), boxes=np.zeros((0, 4), np.float16),
             probs=np.zeros((0, 2), np.float16))
    return buffer.getvalue()


@unittest.skipUnless(HAS_TORCH, "precisa de torch")
class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        from dronecamp_ia.dml_lan import CoordinatorState

        self.folder = tempfile.TemporaryDirectory()
        settings = _tiny_settings()
        self.state = CoordinatorState(Path(self.folder.name), settings, rounds=2, epochs_per_round=1)

    def tearDown(self):
        self.folder.cleanup()

    def test_rounds_open_only_when_every_model_uploaded(self):
        state = self.state
        state.register({"models": ["yolo26", "rfdetr", "dfine"], "fingerprint": "abc"})
        self.assertEqual(state.task("yolo26")["action"], "train")
        self.assertEqual(state.task("yolo26")["lambda"], 0.0)  # rodada 0 = cada um sozinho
        state.receive(0, "yolo26", _npz_bytes(), {"val_map50": 0.1})
        self.assertEqual(state.task("yolo26")["action"], "wait")
        state.receive(0, "rfdetr", _npz_bytes(), {"val_map50": 0.1})
        state.receive(0, "dfine", _npz_bytes(), {"val_map50": 0.1})
        task = state.task("rfdetr")
        self.assertEqual((task["action"], task["round"]), ("train", 1))
        self.assertEqual(sorted(task["peers"]), ["dfine", "yolo26"])
        self.assertGreater(task["lambda"], 0.0)  # 1.º braço do RL é 0,5
        for name in ("yolo26", "rfdetr", "dfine"):
            state.receive(1, name, _npz_bytes(), {"val_map50": 0.2})
        self.assertEqual(state.task("dfine")["action"], "done")
        reloaded = type(state)(Path(self.folder.name), _tiny_settings(), 2, 1)
        self.assertTrue(reloaded.data["complete"])
        self.assertEqual(reloaded.controllers["yolo26"].history[0]["reward"], 0.1)

    def test_rejects_other_dataset_unknown_model_and_bad_files(self):
        state = self.state
        state.register({"models": ["yolo26"], "fingerprint": "abc"})
        with self.assertRaises(ValueError):
            state.register({"models": ["rfdetr"], "fingerprint": "outro"})
        with self.assertRaises(ValueError):
            state.register({"models": ["yolo99"], "fingerprint": "abc"})
        with self.assertRaises(ValueError):
            state.receive(0, "yolo26", b"isto nao e npz", {})
        with self.assertRaises(ValueError):
            state.receive(3, "yolo26", _npz_bytes(), {})
        with self.assertRaises(ValueError):
            state.peer_file(0, "../../segredo")

    def test_http_requires_token(self):
        from urllib import error, request

        from dronecamp_ia.dml_lan import CoordinatorClient, serve_coordinator

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        ready, stop = threading.Event(), threading.Event()
        with tempfile.TemporaryDirectory() as folder:
            thread = threading.Thread(target=serve_coordinator, kwargs=dict(
                config_root=Path(folder), settings=_tiny_settings(), host="127.0.0.1", port=port, rounds=1,
                epochs_per_round=1, token="certo", ready=ready, stop=stop))
            thread.start()
            ready.wait(10)
            try:
                with self.assertRaises(error.HTTPError) as caught:
                    request.urlopen(request.Request(f"http://127.0.0.1:{port}/status", headers={"X-DroneCamp-Token": "errado"}))
                self.assertEqual(caught.exception.code, 401)
                status = CoordinatorClient(f"http://127.0.0.1:{port}", "certo")._call("GET", "/status")
                self.assertEqual(status["round"], 0)
            finally:
                stop.set()
                thread.join(10)


# ---------------------------------------------------------------------------
# Integração com os três modelos reais (pesos aleatórios)
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_ALL, "precisa de ultralytics, rfdetr e transformers (pip install -e .[dml])")
class ThreeModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        cls.data = _synthetic_dataset(cls.root / "data")

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_online_distill_and_cached_modes(self):
        from dronecamp_ia.detector_adapters import build_adapter
        from dronecamp_ia.mutual_learning import TileDataset, build_peer_cache, evaluate_models, train_mutual

        settings = _tiny_settings()
        run = train_mutual(self.root, self.data, settings, "online", max_batches=1, val_max_batches=1)
        lines = [json.loads(line) for line in (run / "history.jsonl").read_text(encoding="utf-8").splitlines()]
        first = [line for line in lines if line["epoch"] == 0]
        self.assertEqual({line["model"] for line in first}, {"yolo26", "rfdetr", "dfine"})
        # 1.ª época: braço 0,5 do RL, com alvos dos pares; 2.ª: braço 0,0 (controle, sem consistência).
        self.assertTrue(all(line["multiplier"] == 0.5 and line["targets"] > 0 for line in first))
        self.assertTrue(all(line["multiplier"] == 0.0 and line["targets"] == 0 for line in lines if line["epoch"] == 1))
        self.assertTrue((run / "weights" / "yolo26" / "last.pt").exists())
        self.assertTrue((run / "weights" / "rfdetr" / "last.pth").exists())
        self.assertTrue((run / "weights" / "dfine" / "last" / "config.json").exists())
        # Professores = os pesos que acabaram de ser gravados.
        trained_settings = json.loads(json.dumps(settings))
        for model in trained_settings["models"]:
            suffix = {"yolo26": "last.pt", "rfdetr": "last.pth", "dfine": "last"}[model["family"]]
            model["weights"] = str(run / "weights" / model["name"] / suffix)
        distill = train_mutual(self.root, self.data, trained_settings, "distill", "yolo26", max_batches=1, val_max_batches=1)
        self.assertIn("yolo26", json.loads((distill / "summary.json").read_text(encoding="utf-8"))["models"])
        dataset = TileDataset(self.data, "train", 320)
        cache_dir = self.root / "cache"
        for spec in trained_settings["models"][1:]:
            build_peer_cache(build_adapter(spec, dataset.names, 320), dataset, cache_dir, batch=2, min_conf=0.0)
        cached = train_mutual(self.root, self.data, trained_settings, "cached", "yolo26", cache_dir,
                              max_batches=1, val_max_batches=1)
        history = (cached / "history.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(json.loads(history[-1])["mode"], "cached")
        evaluation = json.loads((evaluate_models(self.root, self.data, trained_settings, "val") / "evaluation.json")
                                .read_text(encoding="utf-8"))
        self.assertEqual(set(evaluation["models"]), {"yolo26", "rfdetr", "dfine"})
        self.assertIn("union_recall", evaluation["complementarity"])
        # O .pt do YOLO26 abre na Ultralytics (então compare-models/suggest funcionam com ele).
        from ultralytics import YOLO

        self.assertEqual(YOLO(str(run / "weights" / "yolo26" / "last.pt")).names, {0: "vermelho", 1: "azul"})

    def test_lan_rounds_with_two_workers(self):
        from dronecamp_ia.dml_lan import run_worker, serve_coordinator

        settings = _tiny_settings()
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        ready, stop, result, errors = threading.Event(), threading.Event(), {}, []
        coordinator = threading.Thread(target=lambda: result.setdefault("dir", serve_coordinator(
            self.root / "lan", settings, "127.0.0.1", port, rounds=2, epochs_per_round=1, token="t", ready=ready,
            stop=stop)))
        coordinator.start()
        ready.wait(10)

        def worker(models, label):
            try:
                run_worker(self.root / "lan", self.data, settings, models, f"http://127.0.0.1:{port}", "t",
                           host_label=label, max_batches=1, val_max_batches=1, poll_seconds=0.5)
            except Exception as error:  # noqa: BLE001 - o teste mostra o erro do trabalhador
                errors.append(error)

        workers = [threading.Thread(target=worker, args=(["yolo26", "dfine"], "HP-Z230")),
                   threading.Thread(target=worker, args=(["rfdetr"], "6006U"))]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join(600)
        coordinator.join(30)
        stop.set()
        coordinator.join(10)
        self.assertEqual(errors, [])
        state = json.loads((result["dir"] / "state.json").read_text(encoding="utf-8"))
        self.assertTrue(state["complete"])
        self.assertEqual(set(state["uploads"]["1"]), {"yolo26", "rfdetr", "dfine"})
        self.assertEqual(state["workers"]["rfdetr"]["host"], "6006U")


if __name__ == "__main__":
    unittest.main()
