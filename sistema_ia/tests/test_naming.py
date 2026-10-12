"""Nomes legíveis (naming.py) e a cópia do algoritmo na página (platform.js).

Sem mocks: os ids são os de verdade das pastas do projeto. O teste de paridade
executa o bloco "Nomes legíveis" do platform.js no Node (pulado se não houver
Node) e exige o mesmo resultado do Python para cada id.

Módulo coberto
--------------
``dronecamp_ia.naming``: ``run_name``, ``run_started``, ``review_name``, ``weights_name``,
``comparison_name``, ``dataset_name``, ``generated_file_name`` e ``format_datetime``; e o
bloco equivalente ``NAMING`` em ``review_templates/platform.js`` (entre os separadores
"Nomes legíveis" e "fim dos nomes legíveis").

Como rodar (na pasta ``sistema_ia``)
------------------------------------
- Só este arquivo::

      DRONECAMP_SKIP_E2E=1 python -m unittest tests.test_naming -v

- Um teste só::

      python -m unittest tests.test_naming.NamingTests.test_page_uses_the_same_names -v

O que a classe prepara
----------------------
``NamingTests`` não tem ``setUp``: usa as listas de ids do módulo (``REVIEWS``, ``RUNS``,
``WEIGHTS``, ``COMPARISONS``, ``DATASETS``, ``GENERATED``). O teste de paridade lê
platform.js do disco e roda um script no Node via ``subprocess``.

ATENCAO: paridade Python × JS. Toda mudança de regra em naming.py precisa ser repetida no
bloco "Nomes legíveis" do platform.js (e vice-versa), senão ``test_page_uses_the_same_names``
falha. Não renomeie os separadores ``/* ======================= Nomes legíveis`` e
``/* ======================= fim dos nomes legíveis`` do platform.js: o teste os procura pelo texto.

EXTENSAO: para um tipo de id novo (ex.: exportações ONNX):
1. crie a função em naming.py e a entrada correspondente em ``NAMING`` no platform.js;
2. crie aqui uma lista de ids reais (como ``RUNS``);
3. acrescente o par em ``cases`` e em ``python`` dentro de ``test_page_uses_the_same_names``;
4. copie ``test_runs_are_type_plus_short_hash`` para afirmar alguns nomes esperados.
Para um formato novo de um tipo existente, basta acrescentar o id na lista e uma asserção.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import unittest

from dronecamp_ia import naming

# ---------------------------------------------------------------------------
# Ids reais usados nos testes (um por formato conhecido)
# ---------------------------------------------------------------------------

# Arquivo da página com a cópia JS do algoritmo de nomes.
TEMPLATE = Path(__file__).resolve().parents[1] / "src" / "dronecamp_ia" / "review_templates" / "platform.js"
# Pastas de data/reviews/ (revisões/inspeções).
# EXTENSAO: novo padrão de nome de revisão: acrescente um id de exemplo nesta lista.
REVIEWS = ["ceasa_v8_revisao002_ff34e226416c", "ceasa_lado_a_v1", "ceasa_v10_taxonomia_v6", "ceasa_lado_a_v2_prerevisao1",
           "ceasa_v3_humana_a36ea7f67d08", "ceasa_v6_corrigida_a36ea7f67d08", "ceasa_v9_rev001", "ceasa_v4",
           "internet_v1_sugestoes_modelo", "internet_v2_propostas_claude", "internet_v3_busca_aberta", "galpao_norte_2026",
           "inspecao_11_10_2026", "r_2026_lote", "galpao_v2_deadbeef"]
# Pastas de runs/ (<tipo>_<AAAAMMDDTHHMMSSZ>_<hash curto>); "detect" é um nome fora do padrão.
RUNS = ["pilot_train_20261007T010057Z_f5e5ba8b", "train_20261003T232842Z_7a2f3d12", "tune_20261008T120000Z_0a1b2c3d",
        "autolabel_eval_20261005T151719Z_a99aebbb", "review_validation_v5_20261004T020100Z_61a9423b", "detect"]
# Caminhos de pesos: best/last de um treino e pesos base em models/.
WEIGHTS = ["runs/pilot_train_20261007T010057Z_56038eb9/fit/weights/best.pt", "runs/pilot_train_20261007T010057Z_56038eb9/fit/weights/last.pt",
           "models/yolo26l.pt", "models/yolo27s.pt", "models/outro_modelo.pt"]
# Arquivos de comparação de modelos.
COMPARISONS = ["compare_v9_f5e5ba8b_vs_56038eb9_20261007.json", "compare_v8_20261004.json", "compare_v8b_20261004.json",
               "eval_por_classe_630dc0e0_vs_56038eb9_20261006.json", "dataset_review.json"]
# Pastas de datasets (data/pilot/, data/versions/).
DATASETS = ["ceasa_v8_piloto_s42", "ceasa_v10_ladoA_parcial_s42", "lote_teste"]
# Arquivos gerados pela plataforma (<data>_<tipo>_<revisão>), e um nome fora do padrão.
GENERATED = ["20261011T140301Z_laudo_ceasa_v8_revisao002_ff34e226416c", "20261011T140301Z_evidencias_ceasa_v4.zip", "outro.zip"]


# ---------------------------------------------------------------------------
# Nomes esperados e paridade com a página
# ---------------------------------------------------------------------------

class NamingTests(unittest.TestCase):
    """Nomes legíveis gerados no Python e a mesma saída no JavaScript da página."""

    def test_runs_are_type_plus_short_hash(self) -> None:
        """Run vira "<tipo legível> <hash em maiúsculas>"; nome dado pelo usuário vence; data vem do id em UTC."""
        self.assertEqual(naming.run_name("pilot_train_20261007T010057Z_f5e5ba8b"), "Treino Piloto F5E5BA8B")
        self.assertEqual(naming.run_name("train_20261003T232842Z_7a2f3d12"), "Treino de Produção 7A2F3D12")
        # 2º argumento = nome escolhido pelo usuário (tem prioridade).
        self.assertEqual(naming.run_name("pilot_train_20261007T010057Z_f5e5ba8b", "v7.4 referência"), "v7.4 referência")
        self.assertEqual(naming.run_started("pilot_train_20261007T010057Z_f5e5ba8b").isoformat(), "2026-10-07T01:00:57+00:00")

    def test_reviews_read_as_place_version_and_stage(self) -> None:
        """Revisão vira "<local> v<n> · <etapa> <hash>"; hash de 12 caracteres é mostrado com 8 em maiúsculas."""
        self.assertEqual(naming.review_name("ceasa_v8_revisao002_ff34e226416c"), "CEASA Pavilhão A v8 · Revisão 002 FF34E226")
        self.assertEqual(naming.review_name("ceasa_lado_a_v1"), "CEASA Lado A v1")
        self.assertEqual(naming.review_name("ceasa_v10_taxonomia_v6"), "CEASA Pavilhão A v10 · Taxonomia v6")
        self.assertEqual(naming.review_name("ceasa_v3_humana_a36ea7f67d08"), "CEASA Pavilhão A v3 · Revisão Humana A36EA7F6")
        self.assertEqual(naming.review_name("internet_v1_sugestoes_modelo"), "Fotos da Internet v1 · Sugestões do Modelo")
        self.assertEqual(naming.review_name("galpao_norte_2026"), "Galpão Norte 2026")
        self.assertEqual(naming.review_name("galpao_v2_deadbeef"), "Galpão v2 · DEADBEEF", "Hash só com letras a–f também é hash.")
        # Nome digitado no envio pela plataforma vence o nome derivado.
        self.assertEqual(naming.review_name("galpao_b_outubro", "Galpão B · outubro"), "Galpão B · outubro")

    def test_weights_comparisons_datasets_and_generated_files(self) -> None:
        """Pesos, comparações, datasets e arquivos gerados recebem nomes legíveis e datas em dd/mm/aaaa.

        ``format_datetime`` converte de UTC para o horário de Brasília (UTC−3): 14:03 → 11:03.
        """
        self.assertEqual(naming.weights_name(WEIGHTS[0]), "Treino Piloto 56038EB9 · melhor época")
        self.assertEqual(naming.weights_name("models/yolo26l.pt"), "Modelo Base YOLO26l")
        self.assertEqual(naming.comparison_name(COMPARISONS[0]), "Comparação F5E5BA8B × 56038EB9 · 07/10/2026")
        self.assertEqual(naming.comparison_name("compare_v8_20261004.json"), "Comparação v8 · 04/10/2026")
        self.assertEqual(naming.dataset_name("ceasa_v8_piloto_s42"), "Dataset Piloto CEASA Pavilhão A v8 · Semente 42")
        # generated_file_name devolve (tipo legível, data ISO em UTC).
        self.assertEqual(naming.generated_file_name(GENERATED[0]), ("Laudo Técnico", "2026-10-11T14:03:01+00:00"))
        self.assertEqual(naming.format_datetime("2026-10-11T14:03:01+00:00"), "11/10/2026 11:03")

    @unittest.skipUnless(shutil.which("node"), "Node não instalado: paridade com a página não verificada.")
    def test_page_uses_the_same_names(self) -> None:
        """O bloco "Nomes legíveis" do platform.js produz exatamente os mesmos nomes do Python para todos os ids.

        Importa porque a página e os textos gerados no servidor (laudo, listas) precisam
        mostrar o mesmo nome para a mesma pasta.
        """
        # 1. Recorta do platform.js só o trecho entre os dois separadores.
        source = TEMPLATE.read_text(encoding="utf-8")
        start = source.index("/* ======================= Nomes legíveis")
        end = source.index("/* ======================= fim dos nomes legíveis")
        # 2. Casos por tipo. EXTENSAO: tipo de id novo: acrescente "tipo": LISTA aqui e a função em ``python`` abaixo.
        cases = {"review": REVIEWS, "run": RUNS, "weights": WEIGHTS, "comparison": COMPARISONS, "dataset": DATASETS, "generated": GENERATED}
        # 3. Script Node: trecho do JS + os casos + laço que aplica NAMING[tipo](id) e imprime JSON.
        script = source[start:end] + "\nconst cases = " + json.dumps(cases) + ";\n" + \
            "const out = {}; for (const [kind, ids] of Object.entries(cases)) out[kind] = ids.map(id => NAMING[kind](id));\n" + \
            "process.stdout.write(JSON.stringify(out));"
        result = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, encoding="utf-8").stdout)
        # 4. Função Python equivalente a cada NAMING[tipo]; "generated" vira lista porque o JS devolve array.
        python = {"review": naming.review_name, "run": naming.run_name, "weights": naming.weights_name,
                  "comparison": naming.comparison_name, "dataset": naming.dataset_name,
                  "generated": lambda value: list(naming.generated_file_name(value))}
        # 5. Compara id a id; a mensagem de falha diz o tipo e o id que divergiu.
        for kind, ids in cases.items():
            for identifier, page_name in zip(ids, result[kind]):
                self.assertEqual(page_name, python[kind](identifier), f"{kind}: {identifier}")


# ENTRADA: execução direta do arquivo.
if __name__ == "__main__":
    unittest.main()
