"""Ponta a ponta com Ultralytics real: sem mocks de treino, predição ou exportação.

Usa o peso local models/yolo26l.pt e as fotos/revisões reais do CEASA. O treino é
curto (1 época, imagens pequenas) porque verifica o encadeamento, não a qualidade.
Defina DRONECAMP_SKIP_E2E=1 para pular em máquinas sem os arquivos ou sem tempo.

Módulos cobertos (em conjunto, com a biblioteca real): ``pilot`` (dataset piloto),
``training`` (train_pilot, tiling, Repeat Factor Sampling), ``gpu_eval`` (avaliação em
paralelo durante o treino), ``operations`` (collect_runs), ``prediction`` (predict),
``exporting`` (ONNX e paridade), ``suggestions`` + ``review_data`` (sugestões -> revisão ->
próximo dataset), ``model_gate`` (compare_models, measure_models), ``model_report``,
``hparam_bandit`` (successive halving) e ``calibration`` (fit_calibrator).

Como rodar (na pasta ``sistema_ia``, com o venv do projeto; leva minutos):
    python -m unittest tests.test_e2e_ultralytics
    python -m unittest tests.test_e2e_ultralytics.EndToEndUltralyticsTests.test_4_real_onnx_export_matches_pytorch_outputs
    (rodar um teste isolado ainda executa o treino do setUpClass)
Para pular: defina a variável de ambiente DRONECAMP_SKIP_E2E=1.

O que a classe de teste prepara (``EndToEndUltralyticsTests.setUpClass``, UMA vez):
    - pasta temporária que faz papel de ``sistema_ia`` com cópia de ``configs/``;
    - a taxonomia v5 (13 classes) no lugar de ``configs/taxonomy.json``, porque os dados
      e pesos reais ainda seguem esse contrato;
    - ``zero_shot_prompts.json`` filtrado para as classes v5;
    - o dataset piloto ``data/pilot/e2e`` a partir da revisão real v7;
    - UM treino real curto (``TINY``) com avaliação GPU (na CPU) a cada época;
    - ``cls.best`` (best.pt desse treino) e ``cls.sample_images`` (2 fotos reais).
    ``tearDownClass`` devolve as pastas reais à configuração global da Ultralytics.

ATENCAO: os testes compartilham estado e dependem de ordem: ``test_6`` grava
``type(self).rfs_best`` e ``test_7`` o usa (falha com AttributeError se o 6 não rodar
antes). O unittest ordena por NOME, então ``test_10_...`` roda ANTES de ``test_1_...``
("0" < "_"); hoje isso é inofensivo porque o test_10 só usa o que o setUpClass prepara.
Ao numerar um teste novo, lembre dessa ordenação alfabética.

EXTENSAO: para testar uma função nova contra a Ultralytics real, copie
``test_3_real_prediction_writes_findings_marked_as_pilot``: importe a função dentro do
teste (importação pesada), use ``self.config``, ``self.best`` e ``self.sample_images`` e
grave tudo sob ``self.root``. Escolha um número/sufixo que coloque o teste depois dos
testes de que ele depende. Se depender de arquivos grandes opcionais (Git LFS), proteja
com ``@unittest.skipUnless(...)`` como em ``test_9``.
"""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import json
import os
import shutil
import unittest

from dronecamp_ia.config import detection_names, load_config, load_taxonomy
from dronecamp_ia.io import file_hash, resolve_local_path, write_json
from dronecamp_ia.pilot import consolidate_duplicates
from dronecamp_ia.review_data import boxes_to_yolo

# ---------------------------------------------------------------------------
# Arquivos reais usados pelo teste (pulado se faltarem)
# ---------------------------------------------------------------------------
# Configuração real, peso base e a revisão real v7 do CEASA.
REAL = load_config()
WEIGHTS = REAL.root / "models" / "yolo26l.pt"
REGISTRY = REAL.root / "data" / "reviews" / "ceasa_v7_revisao002_ba8cc323c8a1" / "registry.json"
# Pesos piloto v7 reais (Git LFS) e dataset v8: usados no teste do calibrador.
PILOT_V7 = REAL.root / "runs" / "pilot_train_20261004T170212Z_dd284411" / "fit" / "weights" / "best.pt"
DATASET_V8 = REAL.root / "data" / "pilot" / "ceasa_v8_piloto_s42" / "dataset.yaml"
# Fotos aprovadas, pesos piloto e dataset acima seguem a taxonomia v5 (13 classes). O teste
# verifica o encadeamento, então roda nesse contrato; a v6 só terá dados após nova revisão.
TAXONOMY_V5 = REAL.root / "configs" / "taxonomy_ceasa_v5_reparo_rufo.json"
# YOLOE com as frases de configs/zero_shot_prompts.json já embutidas (Git LFS).
ZERO_SHOT_DIR = REAL.root / "models" / "zero_shot"
# Parâmetros de treino mínimos: 1 época, imagens de 160 px, sem gráficos nem paciência.
# ATENCAO: são para testar o encadeamento; nunca use estes valores num treino real.
TINY = {"epochs": 1, "imgsz": 160, "batch": 8, "plots": False, "patience": 0}


@unittest.skipIf(os.environ.get("DRONECAMP_SKIP_E2E") == "1", "E2E desativado por DRONECAMP_SKIP_E2E=1.")
@unittest.skipUnless(WEIGHTS.is_file() and REGISTRY.is_file(), "Peso yolo26l.pt ou revisão v7 ausente.")
class EndToEndUltralyticsTests(unittest.TestCase):
    """Os testes compartilham um treino real; a ordem numérica é intencional.

    Pulada inteira se DRONECAMP_SKIP_E2E=1 ou se faltar o peso yolo26l.pt / a revisão v7.
    """

    @classmethod
    def setUpClass(cls):
        """Monta a raiz temporária, o dataset piloto e roda UM treino real curto para todos os testes.

        Atributos de classe: ``temporary``, ``root``, ``config``, ``names``, ``dataset``,
        ``build`` (resultado de build_pilot_dataset), ``train_run``, ``best`` e ``sample_images``.
        Retorna: None.
        """
        from dronecamp_ia.pilot import build_pilot_dataset
        from dronecamp_ia.training import train_pilot

        cls.temporary = TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        # O worker de avaliação (processo à parte) lê configs/project.yaml da raiz do teste.
        # 1. Copia configs/ e troca a taxonomia pela v5 (contrato dos dados e pesos reais).
        shutil.copytree(REAL.root / "configs", cls.root / "configs")
        taxonomy = cls.root / "configs" / "taxonomy.json"
        shutil.copyfile(TAXONOMY_V5, taxonomy)
        cls.config = replace(REAL, root=cls.root, taxonomy_path=taxonomy)
        cls.names = detection_names(load_taxonomy(taxonomy))
        # Frases da busca aberta só das classes v5: mesmo vocabulário do checkpoint YOLOE salvo.
        prompts_path = cls.root / "configs" / "zero_shot_prompts.json"
        prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
        # A taxonomia atual (v7) fundiu classes; as frases v5 ficam congeladas aqui (as de
        # configs/zero_shot_prompts.json até a v6), para o checkpoint YOLOE salvo continuar valendo.
        prompts["classes"] = {
            "telha_quebrada": ["broken roof tile", "cracked roof sheet", "broken fiber cement roof sheet"],
            "telha_ausente": ["missing roof tile", "hole in the roof where a tile is missing"],
            "residuos_telha": ["debris on roof", "pile of rubble on roof tiles", "trash on the roof"],
            "reparo_telha": ["patched roof tile", "repair patch on roof sheet", "tape on roof"],
            "rufo_ausente": ["missing roof flashing"],
            "rufo_quebrado": ["broken roof flashing", "damaged metal flashing"],
            "residuos_calha": ["debris in roof gutter", "dirt and leaves inside gutter", "clogged gutter"],
            "vegetacao_calha": ["plants growing in gutter", "grass growing inside gutter", "moss in gutter"],
            "pedaco_telha": ["loose piece of roof tile", "fragment of broken tile"],
            "rufo_deslocado": ["displaced roof flashing", "lifted metal flashing"],
            "pedaco_telha_sobreposto": ["piece of broken tile lying on roof sheet"],
            "fixador_telha_frouxo": ["loose roof screw", "loose roof hook bolt", "missing washer on roof fastener"],
            "reparo_rufo": ["sealant repair on flashing", "patched metal flashing"],
        }
        prompts["uncatalogued"] = ["rust on roof sheet", "standing water on roof", "hole in roof", "bird nest on roof", "damaged skylight"]
        prompts_path.write_text(json.dumps(prompts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # 2. Dataset piloto a partir da revisão real.
        cls.dataset = cls.root / "data" / "pilot" / "e2e"
        cls.build = build_pilot_dataset(cls.config, [REGISTRY], cls.dataset)
        # 3. Treino real curto com avaliação em paralelo a cada época.
        # CPU + GPU em conjunto, com o ONNX Runtime na CPU no lugar do DirectML (ausente aqui).
        cls.train_run = train_pilot(cls.config, cls.dataset / "dataset.yaml", str(WEIGHTS), dict(TINY),
                                    {"enabled": True, "every": 1, "provider": "cpu", "split": "val"})
        cls.best = cls.train_run / "fit" / "weights" / "best.pt"
        # 4. Duas fotos reais da revisão, usadas por predição, exportação e sugestões.
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        cls.sample_images = [resolve_local_path(item["source_path"], REAL.root) for item in registry["images"][:2]]

    @classmethod
    def tearDownClass(cls):
        """Restaura as pastas reais nas configurações da Ultralytics e apaga a raiz temporária.

        Retorna: None.
        """
        from dronecamp_ia.backend import _ultralytics

        _ultralytics(REAL)  # restaura as pastas reais nas configurações da biblioteca
        cls.temporary.cleanup()

    def test_1_pilot_dataset_keeps_scenes_in_one_split_and_labels_match_review(self):
        """O dataset piloto mantém cada cena num só split e as labels batem com a revisão humana.

        Só fotos aprovadas entram, e as caixas duplicadas são consolidadas como no piloto.
        """
        manifest = json.loads((self.dataset / "pilot.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["production_ready"])
        # Agrupa os splits por cena: cada cena deve aparecer em exatamente um split.
        splits_by_scene = {}
        for sample in manifest["images"]:
            splits_by_scene.setdefault(sample["scene_group"], set()).add(sample["split"])
        self.assertTrue(all(len(splits) == 1 for splits in splits_by_scene.values()))
        self.assertEqual({sample["split"] for sample in manifest["images"]}, {"train", "val", "test"})
        # Compara cada label do dataset com as caixas da revisão (após consolidar duplicatas).
        reviewed = {item["image_sha256"]: item for item in json.loads(REGISTRY.read_text(encoding="utf-8"))["images"]}
        duplicate_iou = (manifest.get("duplicate_policy") or {}).get("iou")
        for sample in manifest["images"]:
            item = reviewed[sample["image_sha256"]]
            self.assertTrue(item["human_approved"])
            label = (self.dataset / sample["label"]).read_text(encoding="utf-8")
            expected = consolidate_duplicates(item["boxes"], duplicate_iou)
            self.assertEqual(label, boxes_to_yolo(expected, item["width"], item["height"]))
        approved = sum(item["human_approved"] is True for item in reviewed.values())
        self.assertEqual(len(manifest["images"]), approved)

    def test_2_real_training_produces_audited_checkpoint(self):
        """O treino real termina completo, com contrato de classes conferido e proveniência válida.

        ``production_ready`` fica falso: modelo piloto nunca vai direto para produção.
        """
        summary = json.loads((self.train_run / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["complete"], summary["reasons"])
        self.assertEqual(summary["state"], "completed")
        self.assertTrue(summary["pilot"])
        self.assertFalse(summary["production_ready"])
        self.assertTrue(summary["model_class_contract_verified"])
        self.assertGreater(self.best.stat().st_size, 1_000_000)
        provenance = json.loads((self.train_run / "pilot_provenance.json").read_text(encoding="utf-8"))
        self.assertTrue(provenance["valid"])

    def test_2b_gpu_eval_worker_evaluated_the_checkpoint_while_training(self):
        """O worker de avaliação em paralelo avaliou a época 1 e a curva aparece em collect_runs.

        Em caso de falha, a mensagem mostra o fim do ``worker.log``.
        """
        from dronecamp_ia.gpu_eval import read_curve
        from dronecamp_ia.operations import collect_runs

        execution = json.loads((self.train_run / "execution.json").read_text(encoding="utf-8"))
        state = execution["gpu_eval"]
        self.assertEqual(state["state"], "finished", state)
        self.assertEqual(state["enqueued"], [1])
        self.assertNotIn("error", state, state)
        curve = read_curve(self.train_run)
        log = self.train_run / "gpu_eval" / "worker.log"
        self.assertEqual([entry["epoch"] for entry in curve], [1],
                         log.read_text(encoding="utf-8", errors="replace")[-2000:] if log.is_file() else "sem worker.log")
        self.assertEqual(curve[0]["split"], "val")
        self.assertEqual(curve[0]["imgsz"], TINY["imgsz"])
        self.assertIn("CPUExecutionProvider", curve[0]["provider_active"])
        self.assertEqual(curve[0]["regime"], "janelas")
        self.assertIsNotNone(curve[0]["map50"])
        self.assertEqual(state["parallel"]["evaluations"], 1)
        self.assertFalse(list((self.train_run / "gpu_eval").glob("export_*")))  # exportação temporária apagada
        run = next(item for item in collect_runs(self.root) if item["id"] == self.train_run.name)
        self.assertEqual([point["epoch"] for point in run["gpu_eval"]], [1])
        self.assertEqual(run["label_unit"], "fotos")
        self.assertTrue((self.train_run / "fit" / "heartbeat").is_file())

    def test_3_real_prediction_writes_findings_marked_as_pilot(self):
        """A predição real grava achados marcados como "modelo piloto não validado", sem gravidade.

        ATENCAO: gravidade nunca é inferida pela IA (``severity`` sempre None).
        """
        from dronecamp_ia.prediction import predict

        # Copia as 2 fotos para uma pasta de entrada e roda a predição com o best.pt do treino.
        folder = self.root / "predict_input"
        folder.mkdir()
        for image in self.sample_images:
            (folder / image.name).write_bytes(image.read_bytes())
        run = predict(self.config, folder, str(self.best))
        execution = json.loads((run / "execution.json").read_text(encoding="utf-8"))
        self.assertTrue(execution["pilot"])
        rows = [json.loads(line) for line in (run / "findings.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["mode"], "candidatos_modelo_piloto_nao_validado")
            self.assertTrue(Path(row["evidence"]).is_file())
            for detection in row["detections"]:
                self.assertIn(detection["class_name"], self.names)
                self.assertIsNone(detection["severity"])

    def test_4_real_onnx_export_matches_pytorch_outputs(self):
        """O ONNX exportado tem hash registrado e dá as mesmas caixas que o PyTorch."""
        from dronecamp_ia.exporting import export_onnx, verify_onnx_parity

        run = export_onnx(self.config, str(self.best), parity_images=self.sample_images)
        export = json.loads((run / "export.json").read_text(encoding="utf-8"))
        onnx = Path(export["artifact"])
        self.assertTrue(onnx.is_file())
        self.assertEqual(file_hash(onnx), export["sha256"])
        self.assertTrue(export["runtime_parity_validated"], (run / "parity.json").read_text(encoding="utf-8"))
        # Paridade de novo, agora chamada diretamente com limiar de confiança 0.001.
        # Limiar baixo: compara dezenas de caixas, não só a ausência delas.
        parity = verify_onnx_parity(self.config, run / "model.pt", onnx, self.sample_images, 0.001)
        self.assertGreater(sum(item["matched"] for item in parity["images"]), 0)
        self.assertTrue(parity["runtime_parity_validated"], parity)

    def test_5_suggestions_feed_review_and_accepted_boxes_reach_next_dataset(self):
        """Sugestões viram página de revisão; a caixa aceita pelo revisor chega ao próximo dataset.

        Também confere que fotos repetidas em duas revisões contam uma vez (decisão mais recente).
        """
        from dronecamp_ia.pilot import build_pilot_dataset
        from dronecamp_ia.review_data import import_human_feedback
        from dronecamp_ia.suggestions import create_review_for_new_images

        # 1. Fotos novas -> revisão com sugestões do modelo piloto (limiar 0.001).
        folder = self.root / "fotos_novas"
        folder.mkdir()
        for image in self.sample_images:
            (folder / image.name).write_bytes(image.read_bytes())
        review = self.root / "data" / "reviews" / "fotos_novas_e2e"
        page = create_review_for_new_images(self.config, folder, review, "edificacao_teste", str(self.best), 0.001)
        suggestions = json.loads((review / "suggestions.json").read_text(encoding="utf-8"))
        self.assertTrue(suggestions["pilot"])
        self.assertGreater(suggestions["total_suggestions"], 0)
        html = page.read_text(encoding="utf-8")
        self.assertIn("model_suggestions", html)
        self.assertIn("Modelo piloto (não validado)", html)

        # 2. Simula o retorno do revisor (feedback.json) e importa no registro.
        # O revisor aceita uma sugestão: ela passa a ser caixa humana do registro.
        registry_path = review / "registry.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        digest, candidates = next((key, value) for key, value in suggestions["images"].items() if value)
        accepted = {key: candidates[0][key] for key in ("class_id", "bbox_xyxy")}
        feedback = self.root / "feedback.json"
        write_json(feedback, {"schema_version": 1, "registry_sha256": file_hash(registry_path),
                              "reviewer_id": "revisor_e2e", "exported_at": "2026-10-04T00:00:00Z",
                              "images": [{"image_sha256": digest, "status": "positive", "boxes": [accepted],
                                          "notes": "", "confirmed_complete": True}]})
        reviewed_path = review / "registry_revisado.json"
        import_human_feedback(self.config, registry_path, feedback, reviewed_path)
        self.assertEqual(len(registry["images"]), len(self.sample_images))

        # 3. Novo dataset juntando a revisão v7 e a revisão nova.
        # Fotos repetidas em duas revisões contam uma vez, com a decisão mais recente.
        merged = build_pilot_dataset(self.config, [REGISTRY, reviewed_path], self.root / "data" / "pilot" / "e2e_v2")
        manifest = json.loads((self.root / "data" / "pilot" / "e2e_v2" / "pilot.json").read_text(encoding="utf-8"))
        sample = next(item for item in manifest["images"] if item["image_sha256"] == digest)
        label = (self.root / "data" / "pilot" / "e2e_v2" / sample["label"]).read_text(encoding="utf-8")
        self.assertEqual(label.split()[0], str(accepted["class_id"]))
        self.assertEqual(len(label.splitlines()), 1)
        self.assertIn("edificacao_teste", merged["buildings"])
        with (self.root / "data" / "pilot" / "e2e_v2" / "groups.csv").open(encoding="utf-8") as stream:
            self.assertEqual(sum(1 for _ in csv.DictReader(stream)), len(manifest["images"]))


    def test_5b_tiled_training_measures_and_reports_models_window_by_window(self):
        """Treino por janelas (tiling), predição janela a janela, measure-models e report-models."""
        from dronecamp_ia.model_gate import measure_models
        from dronecamp_ia.model_report import build_model_report
        from dronecamp_ia.prediction import predict
        from dronecamp_ia.training import train_pilot

        # Janelas de 320 px a partir de 600 px: as fotos de laudo (1199 px) passam a ser fatiadas.
        tiling = {"enabled": True, "min_side": 600, "patch": 320, "overlap": 0.2, "full_image": True, "merge_iou": 0.5}
        config = replace(self.config, pilot={**self.config.pilot, "tiling": tiling})
        run = train_pilot(config, self.dataset / "dataset.yaml", str(WEIGHTS), {**TINY, "tile": True})
        execution = json.loads((run / "execution.json").read_text(encoding="utf-8"))
        self.assertEqual(execution["label_unit"], "janelas")
        self.assertIn("tiling", execution)
        best = run / "fit" / "weights" / "best.pt"
        # predict com pesos piloto: a foto de laudo entra janela a janela e a evidência é desenhada.
        folder = self.root / "predict_janelas"
        folder.mkdir()
        (folder / self.sample_images[0].name).write_bytes(self.sample_images[0].read_bytes())
        predict_run = predict(config, folder, str(best))
        rows = [json.loads(line) for line in (predict_run / "findings.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[0]["inference"]["regime"], "janelas")
        self.assertGreater(rows[0]["inference"]["windows"], 1)
        self.assertTrue(Path(rows[0]["evidence"]).is_file())
        # measure-models: dois modelos × dois regimes nas mesmas fotos de teste; report-models desenha.
        measurement = measure_models(config, self.dataset / "dataset.yaml", [str(self.best), str(best)], ("test",),
                                     onnx_provider="cpu")
        data = json.loads((measurement / "measurement.json").read_text(encoding="utf-8"))
        self.assertEqual([series["regime"] for series in data["models"]], ["janelas", "inteira"] * 2)
        self.assertEqual(set(data["strata"]), {"fotos_fatiadas"})
        self.assertGreater(data["images_evaluated"], 0)
        report = build_model_report(config, None, measurements=[measurement / "measurement.json"], runs=[run])
        self.assertTrue((report / "report.html").is_file())
        self.assertTrue((report / "strata_recall.png").is_file())
        # Guarda o treino por janelas na classe; hoje nenhum outro teste lê tiled_run.
        type(self).tiled_run = run

    def test_6_repeat_factor_sampling_trains_with_rare_photos_repeated(self):
        """Treino com Repeat Factor Sampling repete fotos raras sem perder nenhuma foto.

        Grava ``type(self).rfs_best`` para o test_7.
        """
        from dronecamp_ia.training import train_pilot

        # Treino com RFS (limiar 0.3); o resultado vira o "candidato" do test_7.
        run = train_pilot(self.config, self.dataset / "dataset.yaml", str(WEIGHTS),
                          {**TINY, "repeat_factor_threshold": 0.3})
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["complete"], summary["reasons"])
        sampling = json.loads((run / "sampling.json").read_text(encoding="utf-8"))
        self.assertGreater(sampling["entries"], sampling["images"])  # houve repetição
        lines = Path(sampling["list"]).read_text(encoding="utf-8").split()
        self.assertEqual(len(set(lines)), sampling["images"])
        self.assertIn("train_repeat_factor.txt", (run / "dataset_resolved.yaml").read_text(encoding="utf-8"))
        # ATENCAO: o test_7 depende deste atributo de classe.
        type(self).rfs_best = run / "fit" / "weights" / "best.pt"

    def test_7_gate_compares_two_real_models_per_class_without_trained_photos(self):
        """O gate compara dois modelos reais por classe, sem usar fotos vistas no treino.

        Depende de ``rfs_best`` gravado pelo test_6.
        """
        from dronecamp_ia.model_gate import compare_models

        run = compare_models(self.config, self.dataset / "dataset.yaml", str(self.best), str(self.rfs_best),
                             ("val", "test"))
        gate = json.loads((run / "gate.json").read_text(encoding="utf-8"))
        self.assertGreater(gate["images_evaluated"], 0)
        self.assertFalse(gate["production_ready"])
        self.assertIn("_total", gate["candidate"]["metrics"])
        self.assertEqual(set(self.names) | {"_total"}, set(gate["baseline"]["metrics"]))
        self.assertIsInstance(gate["adopt"], bool)

    def test_8_successive_halving_runs_real_trainings_and_never_uses_test(self):
        """Successive halving roda treinos reais (1, 1 e 2 épocas) e nunca usa o split de teste."""
        from dronecamp_ia.hparam_bandit import successive_halving

        run = successive_halving(self.config, self.dataset / "dataset.yaml", str(WEIGHTS), arms=2, min_epochs=1,
                                 eta=2, rounds=2, space={"lr0": [0.0005, 0.002]},
                                 fixed={key: TINY[key] for key in ("imgsz", "batch", "plots", "patience")})
        state = json.loads((run / "bandit.json").read_text(encoding="utf-8"))
        self.assertTrue(state["complete"])
        self.assertEqual([trial["epochs"] for trial in state["trials"]], [1, 1, 2])
        self.assertFalse(state["test_used"])
        for trial in state["trials"]:
            summary = json.loads((Path(trial["run"]) / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["complete"], summary["reasons"])

    @unittest.skipUnless(PILOT_V7.is_file() and DATASET_V8.is_file(), "Pesos piloto v7 (Git LFS) ou dataset v8 ausentes.")
    def test_9_calibrator_learns_from_human_boxes_and_annotates_suggestions(self):
        """O calibrador aprende com caixas humanas (excluindo fotos de treino) e anota as sugestões.

        Só roda se os pesos piloto v7 e o dataset v8 (Git LFS) estiverem presentes.
        """
        from dronecamp_ia.calibration import fit_calibrator
        from dronecamp_ia.suggestions import suggest_for_registry

        run = fit_calibrator(self.config, DATASET_V8, str(PILOT_V7))
        metadata = json.loads((run / "calibrator.json").read_text(encoding="utf-8"))
        self.assertTrue(metadata["excluded_training_images"])
        self.assertGreater(metadata["skipped_trained_images"], 0)
        self.assertGreaterEqual(metadata["positives"], 5)
        self.assertIn("improves_over_raw_confidence", metadata["cross_validation"])

        # Revisão com 2 fotos reais para receber sugestões anotadas pelo calibrador.
        review = self.root / "data" / "reviews" / "calibrada"
        review.mkdir(parents=True)
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        registry["images"] = registry["images"][:2]
        for item in registry["images"]:
            item["source_path"] = str(resolve_local_path(item["source_path"], REAL.root))
        write_json(review / "registry.json", registry)
        suggest_for_registry(self.config, review / "registry.json", str(PILOT_V7), 0.05, run,
                             allow_unproven_calibrator=True)
        suggestions = json.loads((review / "suggestions.json").read_text(encoding="utf-8"))
        self.assertEqual(suggestions["calibrator"]["model_sha256"], metadata["model_sha256"])
        boxes = [box for values in suggestions["images"].values() for box in values]
        self.assertTrue(boxes)
        for values in suggestions["images"].values():
            probabilities = [box["acceptance_probability"] for box in values]
            self.assertEqual(probabilities, sorted(probabilities, reverse=True))
            self.assertTrue(all(0 <= value <= 1 for value in probabilities))


    @unittest.skipUnless(any(ZERO_SHOT_DIR.glob("*.pt")), "Checkpoint da busca aberta (models/zero_shot) ausente.")
    def test_10_open_search_adds_pending_suggestions_with_prompt_and_source(self):
        """A busca aberta (YOLOE) acrescenta sugestões pendentes com frase e origem registradas.

        Só roda se houver checkpoint em models/zero_shot. Nenhuma foto sai aprovada.
        """
        from dronecamp_ia.suggestions import create_review_for_new_images

        (self.root / "models").mkdir(exist_ok=True)
        # Liga models/zero_shot real dentro da raiz temporária (link simbólico).
        link = self.root / "models" / "zero_shot"
        if not link.exists():
            link.symlink_to(ZERO_SHOT_DIR, target_is_directory=True)
        folder = self.root / "fotos_busca"
        folder.mkdir()
        for image in self.sample_images:
            (folder / image.name).write_bytes(image.read_bytes())
        review = self.root / "data" / "reviews" / "busca_aberta_e2e"
        create_review_for_new_images(self.config, folder, review, "edificacao_teste", str(self.best), 0.05,
                                     zero_shot=True)
        suggestions = json.loads((review / "suggestions.json").read_text(encoding="utf-8"))
        registry = json.loads((review / "registry.json").read_text(encoding="utf-8"))
        self.assertTrue(suggestions["zero_shot"]["checkpoint_sha256"])
        self.assertTrue(all(item["human_approved"] is False for item in registry["images"]))
        boxes = [box for values in suggestions["images"].values() for box in values]
        self.assertTrue(all(box["source"] in {"piloto", "busca_aberta", "piloto+busca_aberta"} for box in boxes))
        for box in boxes:
            if box["source"] != "piloto":
                self.assertIn(box["class_id"], range(len(self.names)))
                self.assertTrue(box["prompt"])


if __name__ == "__main__":
    unittest.main()
