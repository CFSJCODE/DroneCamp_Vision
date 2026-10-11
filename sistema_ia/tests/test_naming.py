"""Nomes legíveis (naming.py) e a cópia do algoritmo na página (platform.js).

Sem mocks: os ids são os de verdade das pastas do projeto. O teste de paridade
executa o bloco "Nomes legíveis" do platform.js no Node (pulado se não houver
Node) e exige o mesmo resultado do Python para cada id.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import unittest

from dronecamp_ia import naming

TEMPLATE = Path(__file__).resolve().parents[1] / "src" / "dronecamp_ia" / "review_templates" / "platform.js"
REVIEWS = ["ceasa_v8_revisao002_ff34e226416c", "ceasa_lado_a_v1", "ceasa_v10_taxonomia_v6", "ceasa_lado_a_v2_prerevisao1",
           "ceasa_v3_humana_a36ea7f67d08", "ceasa_v6_corrigida_a36ea7f67d08", "ceasa_v9_rev001", "ceasa_v4",
           "internet_v1_sugestoes_modelo", "internet_v2_propostas_claude", "internet_v3_busca_aberta", "galpao_norte_2026",
           "inspecao_11_10_2026", "r_2026_lote"]
RUNS = ["pilot_train_20261007T010057Z_f5e5ba8b", "train_20261003T232842Z_7a2f3d12", "tune_20261008T120000Z_0a1b2c3d",
        "autolabel_eval_20261005T151719Z_a99aebbb", "review_validation_v5_20261004T020100Z_61a9423b", "detect"]
WEIGHTS = ["runs/pilot_train_20261007T010057Z_56038eb9/fit/weights/best.pt", "runs/pilot_train_20261007T010057Z_56038eb9/fit/weights/last.pt",
           "models/yolo26l.pt", "models/yolo27s.pt", "models/outro_modelo.pt"]
COMPARISONS = ["compare_v9_f5e5ba8b_vs_56038eb9_20261007.json", "compare_v8_20261004.json", "compare_v8b_20261004.json",
               "eval_por_classe_630dc0e0_vs_56038eb9_20261006.json", "dataset_review.json"]
DATASETS = ["ceasa_v8_piloto_s42", "ceasa_v10_ladoA_parcial_s42", "lote_teste"]
GENERATED = ["20261011T140301Z_laudo_ceasa_v8_revisao002_ff34e226416c", "20261011T140301Z_evidencias_ceasa_v4.zip", "outro.zip"]


class NamingTests(unittest.TestCase):
    def test_runs_are_type_plus_short_hash(self) -> None:
        self.assertEqual(naming.run_name("pilot_train_20261007T010057Z_f5e5ba8b"), "Treino Piloto F5E5BA8B")
        self.assertEqual(naming.run_name("train_20261003T232842Z_7a2f3d12"), "Treino de Produção 7A2F3D12")
        self.assertEqual(naming.run_name("pilot_train_20261007T010057Z_f5e5ba8b", "v7.4 referência"), "v7.4 referência")
        self.assertEqual(naming.run_started("pilot_train_20261007T010057Z_f5e5ba8b").isoformat(), "2026-10-07T01:00:57+00:00")

    def test_reviews_read_as_place_version_and_stage(self) -> None:
        self.assertEqual(naming.review_name("ceasa_v8_revisao002_ff34e226416c"), "CEASA Pavilhão A v8 · Revisão 002 FF34E226")
        self.assertEqual(naming.review_name("ceasa_lado_a_v1"), "CEASA Lado A v1")
        self.assertEqual(naming.review_name("ceasa_v10_taxonomia_v6"), "CEASA Pavilhão A v10 · Taxonomia v6")
        self.assertEqual(naming.review_name("ceasa_v3_humana_a36ea7f67d08"), "CEASA Pavilhão A v3 · Revisão Humana A36EA7F6")
        self.assertEqual(naming.review_name("internet_v1_sugestoes_modelo"), "Fotos da Internet v1 · Sugestões do Modelo")
        self.assertEqual(naming.review_name("galpao_norte_2026"), "Galpão Norte 2026")
        # Nome digitado no envio pela plataforma vence o nome derivado.
        self.assertEqual(naming.review_name("galpao_b_outubro", "Galpão B · outubro"), "Galpão B · outubro")

    def test_weights_comparisons_datasets_and_generated_files(self) -> None:
        self.assertEqual(naming.weights_name(WEIGHTS[0]), "Treino Piloto 56038EB9 · melhor época")
        self.assertEqual(naming.weights_name("models/yolo26l.pt"), "Modelo Base YOLO26l")
        self.assertEqual(naming.comparison_name(COMPARISONS[0]), "Comparação F5E5BA8B × 56038EB9 · 07/10/2026")
        self.assertEqual(naming.comparison_name("compare_v8_20261004.json"), "Comparação v8 · 04/10/2026")
        self.assertEqual(naming.dataset_name("ceasa_v8_piloto_s42"), "Dataset Piloto CEASA Pavilhão A v8 · Semente 42")
        self.assertEqual(naming.generated_file_name(GENERATED[0]), ("Laudo Técnico", "2026-10-11T14:03:01+00:00"))
        self.assertEqual(naming.format_datetime("2026-10-11T14:03:01+00:00"), "11/10/2026 11:03")

    @unittest.skipUnless(shutil.which("node"), "Node não instalado: paridade com a página não verificada.")
    def test_page_uses_the_same_names(self) -> None:
        source = TEMPLATE.read_text(encoding="utf-8")
        start = source.index("/* ======================= Nomes legíveis")
        end = source.index("/* ======================= fim dos nomes legíveis")
        cases = {"review": REVIEWS, "run": RUNS, "weights": WEIGHTS, "comparison": COMPARISONS, "dataset": DATASETS, "generated": GENERATED}
        script = source[start:end] + "\nconst cases = " + json.dumps(cases) + ";\n" + \
            "const out = {}; for (const [kind, ids] of Object.entries(cases)) out[kind] = ids.map(id => NAMING[kind](id));\n" + \
            "process.stdout.write(JSON.stringify(out));"
        result = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, encoding="utf-8").stdout)
        python = {"review": naming.review_name, "run": naming.run_name, "weights": naming.weights_name,
                  "comparison": naming.comparison_name, "dataset": naming.dataset_name,
                  "generated": lambda value: list(naming.generated_file_name(value))}
        for kind, ids in cases.items():
            for identifier, page_name in zip(ids, result[kind]):
                self.assertEqual(page_name, python[kind](identifier), f"{kind}: {identifier}")


if __name__ == "__main__":
    unittest.main()
