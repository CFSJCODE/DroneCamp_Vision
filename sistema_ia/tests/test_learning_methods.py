"""Bandit LinUCB (fila de revisão), calibração scikit-learn, RFS, regra de adoção e successive halving.

Testes rápidos, sem Ultralytics: a parte com modelo real está em test_e2e_ultralytics.py.

Módulos cobertos (um grupo de testes por módulo):
    - ``dronecamp_ia.matching``        -> ``MatchingTests`` (casamento predição x caixa humana);
    - ``dronecamp_ia.sampling``        -> ``RepeatFactorSamplingTests`` (RFS, repetir fotos de
      classes raras) e o gancho ``training._apply_repeat_factor_sampling``;
    - ``dronecamp_ia.active_learning`` -> ``LinUCBTests`` e ``ReviewQueueTests`` (bandit que
      ordena a fila de revisão e aprende com as decisões humanas);
    - ``dronecamp_ia.calibration``     -> ``CalibrationTests`` (calibrador scikit-learn);
    - ``dronecamp_ia.model_gate``      -> ``AdoptionRuleTests`` (quando um modelo novo é adotado);
    - ``dronecamp_ia.hparam_bandit``   -> ``SuccessiveHalvingTests`` (busca de hiperparâmetros).

Como rodar (na pasta ``sistema_ia``, com o venv do projeto):
    python -m unittest tests.test_learning_methods
    python -m unittest tests.test_learning_methods.LinUCBTests
    python -m unittest tests.test_learning_methods.AdoptionRuleTests.test_class_regression_blocks_adoption_even_with_higher_map

Preparação:
    - ``REAL`` e ``NAMES`` (no topo) carregam a configuração e as classes REAIS do projeto
      (configs/project.yaml e configs/taxonomy.json); nada é gravado nelas.
    - ``prediction`` e ``truth`` montam caixas no formato interno (classe, bbox xyxy em px).
    - ``LinUCBTests.setUp`` calcula a raridade com a classe 0 comum e a 11 inexistente.
    - ``ReviewQueueTests.setUp`` grava, numa pasta temporária, um ``registry.json`` com 3 fotos
      (só a 0 decidida pelo humano) e um ``suggestions.json`` com as sugestões da IA.
    - Os demais testes criam pastas temporárias próprias com ``tempfile``.

EXTENSAO: para testar uma função nova de um destes módulos, copie o teste mais parecido da
classe do módulo (ex.: ``MatchingTests.test_wrong_class_is_false_positive_and_omission``),
monte as caixas com ``prediction(...)``/``truth(...)`` e confira o resultado. Para um módulo
de aprendizado novo, crie uma classe ``NovoModuloTests(unittest.TestCase)`` no mesmo estilo,
antes do bloco ``if __name__ == "__main__"``.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from dronecamp_ia.active_learning import (FEATURES, LinUCBPolicy, class_rarity, image_context, prioritize_review,
                                          review_reward, update_policy)
from dronecamp_ia.config import detection_names, load_config, load_taxonomy
from dronecamp_ia.hparam_bandit import arms_from_space, best_validation_reward, successive_halving
from dronecamp_ia.io import write_json
from dronecamp_ia.matching import image_errors, match_predictions, per_class_report
from dronecamp_ia.model_gate import average_precision, decide
from dronecamp_ia.sampling import repeat_factors, write_repeat_factor_list

# ---------------------------------------------------------------------------
# Configuração real e helpers de caixas
# ---------------------------------------------------------------------------
# Configuração e nomes de classes REAIS do projeto (somente leitura nos testes).
REAL = load_config()
NAMES = detection_names(load_taxonomy(REAL.taxonomy_path))


def prediction(class_id, box, confidence):
    """Monta uma predição da IA no formato interno.

    Parâmetros:
        class_id (int): ID da classe na taxonomia.
        box (list): ``[x1, y1, x2, y2]`` em pixels.
        confidence (float): confiança 0..1.
    Retorna: dict ``{"class_id", "bbox_xyxy", "confidence"}``.
    """
    return {"class_id": class_id, "bbox_xyxy": box, "confidence": confidence}


def truth(class_id, box):
    """Monta uma caixa humana (verdade) no formato interno.

    Parâmetros:
        class_id (int): ID da classe.
        box (list): ``[x1, y1, x2, y2]`` em pixels.
    Retorna: dict ``{"class_id", "bbox_xyxy"}``.
    """
    return {"class_id": class_id, "bbox_xyxy": box}


# ---------------------------------------------------------------------------
# matching.py
# ---------------------------------------------------------------------------
class MatchingTests(unittest.TestCase):
    """Testes de ``matching``: casamento predição x verdade e relatório por classe.

    Essas contagens (TP/FP/FN) alimentam métricas, recompensas do bandit e a regra de adoção.
    """
    def test_each_human_box_is_matched_once_by_highest_confidence(self):
        """Cada caixa humana casa com no máximo uma predição, a de maior confiança."""
        predictions = [prediction(2, [0, 0, 10, 10], 0.4), prediction(2, [0, 0, 10, 11], 0.9)]
        hits, used = match_predictions(predictions, [truth(2, [0, 0, 10, 10])])
        self.assertEqual(hits, [False, True])
        self.assertEqual(used, {0})

    def test_wrong_class_is_false_positive_and_omission(self):
        """Caixa no lugar certo mas classe errada conta como 1 FP e 1 FN (nenhum TP)."""
        errors = image_errors([prediction(1, [0, 0, 10, 10], 0.9)], [truth(2, [0, 0, 10, 10])])
        self.assertEqual((errors["tp"], errors["fp"], errors["fn"]), (0, 1, 1))

    def test_class_without_boxes_is_not_measured_rather_than_zero(self):
        """Classe sem caixas humanas tem recall/precisão ``None`` ("não medido"), não 0.

        Importa porque 0 sugeriria que o modelo é ruim numa classe que nem foi avaliada.
        """
        report = per_class_report([image_errors([], [truth(0, [0, 0, 5, 5])])], NAMES)
        self.assertEqual(report[NAMES[0]]["recall"], 0.0)
        self.assertIsNone(report[NAMES[8]]["recall"])
        self.assertIsNone(report[NAMES[8]]["precision"])


# ---------------------------------------------------------------------------
# sampling.py (Repeat Factor Sampling)
# ---------------------------------------------------------------------------
class RepeatFactorSamplingTests(unittest.TestCase):
    """Testes do Repeat Factor Sampling (``sampling``): repetir no treino fotos de classes raras."""
    def test_rare_class_images_are_repeated_and_common_ones_are_not(self):
        """Classe em 90% das fotos tem fator 1; classe em 10% tem fator sqrt(limiar / frequência).

        Com limiar 0.3 e frequência 0.1, o fator é sqrt(3) ~ 1.73.
        """
        classes = [{0}] * 9 + [{11}]
        per_image, by_class = repeat_factors(classes, 0.3)
        self.assertEqual(by_class[0], 1.0)  # 90% das fotos: acima do limiar
        self.assertAlmostEqual(by_class[11], (0.3 / 0.1) ** 0.5)
        self.assertEqual(per_image[:9], [1.0] * 9)
        self.assertGreater(per_image[-1], 1.7)

    def test_list_keeps_every_image_and_is_reproducible(self):
        """A lista de treino mantém todas as fotos, repete a rara e é igual com a mesma semente."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            # 10 fotos: a 0 tem a classe rara 11, as 1..8 a classe 0 e a 9 é negativa.
            for index in range(10):
                (root / "images" / f"{index}.jpg").write_bytes(b"x")
                label = "11 0.5 0.5 0.1 0.1\n" if index == 0 else ("0 0.5 0.5 0.1 0.1\n" if index < 9 else "")
                (root / "labels" / f"{index}.txt").write_text(label, encoding="utf-8")
            first = write_repeat_factor_list(root / "images", root / "labels", root / "a.txt", 0.3, seed=7)
            second = write_repeat_factor_list(root / "images", root / "labels", root / "b.txt", 0.3, seed=7)
            lines = (root / "a.txt").read_text(encoding="utf-8").split()
            self.assertEqual(lines, (root / "b.txt").read_text(encoding="utf-8").split())
            self.assertEqual(len(set(lines)), 10)
            self.assertGreaterEqual(lines.count((root / "images" / "0.jpg").resolve().as_posix()), 2)
            self.assertEqual(first["entries"], second["entries"])

    def test_invalid_threshold_is_rejected(self):
        """Limiar 0 é inválido (ValueError)."""
        with self.assertRaises(ValueError):
            repeat_factors([{0}], 0)

    def test_train_pilot_hook_rewrites_only_the_train_entry(self):
        """O gancho do treino troca só a entrada ``train`` do YAML resolvido pela lista RFS.

        ``val`` fica intacto (validação nunca é reamostrada) e ``sampling.json`` é gravado.
        """
        from dronecamp_ia.training import _apply_repeat_factor_sampling

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            # Dataset mínimo com uma foto em train e uma em val, e o YAML resolvido apontando para elas.
            for split in ("train", "val"):
                (root / "data" / "images" / split).mkdir(parents=True)
                (root / "data" / "labels" / split).mkdir(parents=True)
                (root / "data" / "images" / split / "a.jpg").write_bytes(b"x")
                (root / "data" / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            resolved = root / "dataset_resolved.yaml"
            resolved.write_text(yaml.safe_dump({"path": str(root / "data"), "train": str(root / "data/images/train"),
                                                "val": str(root / "data/images/val")}), encoding="utf-8")
            summary = _apply_repeat_factor_sampling(root, resolved, 0.3, 0)
            data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
            self.assertTrue(data["train"].endswith("train_repeat_factor.txt"))
            self.assertEqual(data["val"], str(root / "data/images/val"))
            self.assertEqual(summary["images"], 1)
            self.assertTrue((root / "sampling.json").is_file())


# ---------------------------------------------------------------------------
# active_learning.py (bandit LinUCB e fila de revisão)
# ---------------------------------------------------------------------------
class LinUCBTests(unittest.TestCase):
    """Testes da política LinUCB que pontua fotos para a fila de revisão."""
    def setUp(self):
        """Raridade de classes: classe 0 com 99 caixas (comum) e classe 11 com 0 (rara).

        Retorna: None.
        """
        self.rarity = class_rarity({0: 99, 8: 0}, len(NAMES))

    def test_prior_ranks_uncertain_rare_suggestions_above_confident_common_ones(self):
        """Sem aprendizado (alpha 0), sugestão incerta de classe rara vem antes de comum e confiante."""
        policy = LinUCBPolicy(alpha=0.0)
        confident = image_context([prediction(0, [0, 0, 10, 10], 0.95)], self.rarity, False)
        uncertain_rare = image_context([prediction(8, [0, 0, 10, 10], 0.5)], self.rarity, False)
        self.assertGreater(policy.score(uncertain_rare)["score"], policy.score(confident)["score"])

    def test_rewards_move_the_policy_and_shrink_exploration(self):
        """Recompensas repetidas aumentam a parte "exploit" e reduzem a "explore" do escore."""
        policy = LinUCBPolicy(alpha=1.0)
        context = image_context([prediction(0, [0, 0, 10, 10], 0.9)], self.rarity, True)
        before = policy.score(context)
        for _ in range(20):
            policy.update(context, 1.0)
        after = policy.score(context)
        self.assertLess(after["explore"], before["explore"])
        self.assertGreater(after["exploit"], before["exploit"])

    def test_reward_is_high_when_model_misses_a_rare_class(self):
        """Revisão que revela classe rara perdida pelo modelo rende mais que uma correta."""
        missed = review_reward([], [truth(8, [0, 0, 10, 10])], self.rarity)
        correct = review_reward([prediction(0, [0, 0, 10, 10], 0.9)], [truth(0, [0, 0, 10, 10])], self.rarity)
        self.assertGreater(missed["reward"], correct["reward"])
        self.assertEqual(missed["fn"], 1)

    def test_context_has_one_value_per_feature(self):
        """O vetor de contexto tem exatamente um valor por item de ``FEATURES``.

        ATENCAO: mudar ``FEATURES`` muda a dimensão da política salva (policy.json).
        """
        self.assertEqual(image_context([], self.rarity, False).shape, (len(FEATURES),))


class ReviewQueueTests(unittest.TestCase):
    """Testes da fila priorizada (``prioritize_review``) e do aprendizado (``update_policy``)."""
    def setUp(self):
        """Grava registry.json (3 fotos, só a 0 decidida) e suggestions.json numa pasta temporária.

        Foto 1: sugestão comum e confiante; foto 2: sugestão rara e incerta.
        Retorna: None.
        """
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.registry = self.root / "registry.json"
        # 3 fotos com hash falso de 64 dígitos; só a 0 tem aprovação humana (classe 11).
        images = [{"image_sha256": f"{index:064x}", "filename": f"{index}.jpg", "building_group": "galpao_b",
                   "human_approved": index == 0, "status": "positive" if index == 0 else "ambiguous",
                   "boxes": [truth(8, [0, 0, 10, 10])] if index == 0 else []} for index in range(3)]
        write_json(self.registry, {"images": images})
        # Sugestões da IA por hash: foto 1 comum e confiante, foto 2 rara e incerta.
        write_json(self.root / "suggestions.json", {"weights_sha256": "w1", "images": {
            images[0]["image_sha256"]: [],
            images[1]["image_sha256"]: [prediction(0, [0, 0, 10, 10], 0.97)],
            images[2]["image_sha256"]: [prediction(8, [0, 0, 10, 10], 0.45), prediction(4, [0, 0, 10, 10], 0.4)],
        }})

    def test_queue_skips_decided_photos_and_puts_uncertain_rare_photo_first(self):
        """A fila ignora fotos já decididas e põe primeiro a sugestão rara e incerta; gera CSV também."""
        output = prioritize_review(REAL, self.registry)
        queue = json.loads(output.read_text(encoding="utf-8"))["images"]
        self.assertEqual([row["filename"] for row in queue], ["2.jpg", "1.jpg"])
        self.assertTrue(output.with_suffix(".csv").is_file())

    def test_learning_counts_each_decision_once(self):
        """Rodar o aprendizado duas vezes não conta a mesma decisão humana duas vezes."""
        policy = self.root / "policy.json"
        first = update_policy(REAL, self.registry, policy)
        second = update_policy(REAL, self.registry, policy)
        self.assertEqual(first["new_updates"], 1)
        self.assertEqual(second["new_updates"], 0)
        self.assertEqual(json.loads(policy.read_text(encoding="utf-8"))["updates"], 1)


# ---------------------------------------------------------------------------
# calibration.py
# ---------------------------------------------------------------------------
class CalibrationTests(unittest.TestCase):
    """Testes do calibrador (``calibration``) que reajusta a confiança das sugestões."""
    def test_features_have_fixed_width_and_class_one_hot(self):
        """As features têm largura fixa e a classe vem em one-hot (``class_N`` = 1)."""
        from dronecamp_ia.calibration import feature_names, suggestion_features

        boxes = [prediction(3, [0, 0, 50, 20], 0.7), prediction(5, [10, 0, 60, 20], 0.2)]
        X = suggestion_features(boxes, 100, 100, len(NAMES))
        self.assertEqual(X.shape, (2, len(feature_names(len(NAMES)))))
        self.assertEqual(X[0, feature_names(len(NAMES)).index("class_3")], 1.0)

    def test_lightgbm_calibrator_learns_an_interaction_the_logistic_misses(self):
        """LightGBM (fit-calibrator --algorithm lightgbm) ganha da confiança crua numa interação.

        Rótulo = sinal de a×b: invisível para a regressão logística, fácil para árvores.
        """
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            self.skipTest("lightgbm não instalado (extra learning)")
        from dronecamp_ia.calibration import cross_validate

        generator = np.random.default_rng(1)
        size = 240
        raw = generator.uniform(0.1, 0.9, size)
        a, b = generator.normal(size=size), generator.normal(size=size)
        y = (a * b > 0).astype(int)
        X = np.column_stack([raw, a, b])
        groups = np.repeat(np.arange(12), size // 12)
        self.assertTrue(cross_validate(X, y, groups, raw, algorithm="lightgbm")["improves_over_raw_confidence"])
        with self.assertRaises(ValueError):
            cross_validate(X, y, groups, raw, algorithm="xgboost")

    def test_cross_validation_reports_gain_when_a_feature_explains_acceptance(self):
        """A validação cruzada por grupo detecta ganho quando uma feature explica o aceite.

        Dados sintéticos: a confiança crua é ruído e a feature ``informative`` define o rótulo.
        """
        from dronecamp_ia.calibration import cross_validate

        # Dados sintéticos reprodutíveis: 240 sugestões em 12 grupos de 20.
        generator = np.random.default_rng(0)
        size = 240
        raw = generator.uniform(0.1, 0.9, size)  # confiança crua sem relação com o acerto
        informative = generator.normal(size=size)
        y = (informative + generator.normal(scale=0.3, size=size) > 0).astype(int)
        X = np.column_stack([raw, informative])
        groups = np.repeat(np.arange(12), size // 12)
        report = cross_validate(X, y, groups, raw)
        self.assertTrue(report["improves_over_raw_confidence"], report)
        self.assertGreater(report["calibrated"]["roc_auc"], 0.85)

    def test_tampered_model_file_is_refused(self):
        """Calibrador sem ganho medido é recusado por padrão, e arquivo adulterado é sempre recusado."""
        import joblib
        from sklearn.dummy import DummyClassifier

        from dronecamp_ia.calibration import Calibrator
        from dronecamp_ia.io import file_hash

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            # Salva um modelo trivial com metadados sem ganho medido e confere as três situações.
            joblib.dump(DummyClassifier().fit([[0], [1]], [0, 1]), directory / "calibrator.joblib")
            metadata = {"model_file": "calibrator.joblib", "model_sha256": file_hash(directory / "calibrator.joblib"),
                        "taxonomy_sha256": file_hash(REAL.taxonomy_path), "class_names": NAMES,
                        "cross_validation": {"improves_over_raw_confidence": False}}
            write_json(directory / "calibrator.json", metadata)
            with self.assertRaises(ValueError):  # sem ganho medido: recusado por padrão
                Calibrator(REAL, directory)
            Calibrator(REAL, directory, allow_unproven=True)
            (directory / "calibrator.joblib").write_bytes(b"outro conteudo")
            with self.assertRaises(ValueError):
                Calibrator(REAL, directory, allow_unproven=True)


# ---------------------------------------------------------------------------
# model_gate.py (regra de adoção)
# ---------------------------------------------------------------------------
class AdoptionRuleTests(unittest.TestCase):
    """Testes da regra que decide se o modelo candidato substitui o atual (``model_gate``)."""
    def test_average_precision_matches_hand_computed_value(self):
        """AP calculado bate com a conta feita à mão; sem positivos, AP é ``None``."""
        # Acerto, erro, acerto com 2 positivos: AP = 0,5·1 + 0,5·(2/3).
        self.assertAlmostEqual(average_precision([(0.9, True), (0.8, False), (0.7, True)], 2), 0.8333, places=4)
        self.assertIsNone(average_precision([], 0))

    def _report(self, map50, recall, class_recall):
        """Monta um relatório por classe falso para ``decide``.

        Parâmetros:
            map50 (float): mAP50 total.
            recall (float): recall total.
            class_recall (float): recall da classe ``NAMES[2]`` (as outras ficam ``None``).
        Retorna: dict nome da classe -> métricas, mais a chave ``_total``.
        """
        # Todas as classes com 5 caixas e recall não medido, exceto NAMES[2].
        report = {name: {"human_boxes": 5, "recall": None} for name in NAMES}
        report[NAMES[2]] = {"human_boxes": 5, "recall": class_recall}
        report["_total"] = {"map50": map50, "recall": recall}
        return report

    def test_candidate_with_gain_and_no_regression_is_adopted(self):
        """Candidato com ganho total e sem piora em classe nenhuma é adotado."""
        decision = decide(self._report(0.2, 0.3, 0.4), self._report(0.3, 0.35, 0.4), NAMES)
        self.assertTrue(decision["adopt"], decision)

    def test_class_regression_blocks_adoption_even_with_higher_map(self):
        """Piora no recall de uma classe bloqueia a adoção mesmo com mAP maior."""
        decision = decide(self._report(0.2, 0.3, 0.6), self._report(0.3, 0.35, 0.2), NAMES)
        self.assertFalse(decision["adopt"])
        self.assertEqual(decision["regressions"][0]["class"], NAMES[2])


# ---------------------------------------------------------------------------
# hparam_bandit.py (successive halving)
# ---------------------------------------------------------------------------
class SuccessiveHalvingTests(unittest.TestCase):
    """Testes da busca de hiperparâmetros por successive halving (``hparam_bandit``)."""
    def test_budget_goes_to_the_best_arm_and_test_split_is_never_used(self):
        """As 3 opções rodam 1 época; só a melhor ganha 3 épocas; o split de teste nunca é usado."""
        def fake_trainer(config, data, weights, overrides):
            """Treinador falso: grava um results.csv com recompensa conhecida para cada ``lr0``.

            Parâmetros:
                config: ProjectConfig (usa ``root``).
                data: caminho do dataset (ignorado).
                weights: pesos iniciais (ignorados).
                overrides (dict): hiperparâmetros da rodada (``lr0``, ``epochs``).
            Retorna: Path da execução falsa criada.
            """
            run = config.root / "runs" / f"fake_{overrides['lr0']}_{overrides['epochs']}"
            (run / "fit").mkdir(parents=True)
            # Recompensa fixa por lr0 (0.001 é a melhor) + bônus pequeno por época.
            reward = {0.001: 0.3, 0.002: 0.2, 0.0005: 0.1}[overrides["lr0"]] + 0.01 * overrides["epochs"]
            (run / "fit" / "results.csv").write_text(f"epoch, metrics/mAP50(B)\n1, {reward / 2}\n2, {reward}\n",
                                                     encoding="utf-8")
            calls.append(overrides)
            return run

        calls = []
        with tempfile.TemporaryDirectory() as folder:
            config = replace(REAL, root=Path(folder))
            run = successive_halving(config, Path(folder) / "dataset.yaml", arms=None, min_epochs=1, eta=3,
                                     space={"lr0": [0.0005, 0.001, 0.002]}, trainer=fake_trainer)
            state = json.loads((run / "bandit.json").read_text(encoding="utf-8"))
        self.assertTrue(state["complete"])
        self.assertEqual(state["best_parameters"], {"lr0": 0.001})
        self.assertFalse(state["test_used"])
        self.assertEqual([call["epochs"] for call in calls], [1, 1, 1, 3])

    def test_arm_sampling_is_reproducible(self):
        """Sortear opções do espaço é reprodutível; ``None`` devolve todas as combinações (3x2=6)."""
        space = {"lr0": [1, 2, 3], "mosaic": [0, 1]}
        self.assertEqual(arms_from_space(space, 3, 5), arms_from_space(space, 3, 5))
        self.assertEqual(len(arms_from_space(space, None, 5)), 6)

    def test_missing_results_has_no_reward(self):
        """Execução sem results.csv tem recompensa ``None``, não zero."""
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(best_validation_reward(Path(folder)))


if __name__ == "__main__":
    unittest.main()
