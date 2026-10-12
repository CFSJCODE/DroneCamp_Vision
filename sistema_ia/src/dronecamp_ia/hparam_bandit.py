"""Busca de hiperparâmetros do treino piloto como bandit (successive halving).

Função no projeto: cada treino YOLO26l na CPU do Ryzen 5 custa de minutos a
horas. Em vez de treinar todas as combinações até o fim (grade) ou mutar ao
acaso (``model.tune`` da Ultralytics, algoritmo genético), tratamos cada
combinação como um braço de um bandit e gastamos o orçamento de épocas nos
braços promissores.

Algoritmo (successive halving, núcleo do Hyperband; Jamieson & Talwalkar 2016,
Li et al. 2018):
    rodada 0: n braços × e épocas
    rodada k: os melhores ⌈n/η^k⌉ braços × e·η^k épocas
    recompensa = melhor mAP50 de validação no results.csv do treino real
Cada braço de cada rodada é um ``train_pilot`` real e auditado (summary.json,
pilot_provenance.json); nada é simulado.

Limites honestos:
- A validação do piloto tem ~6 fotos de uma edificação; a recompensa é ruidosa.
  O vencedor é uma recomendação, não uma adoção: confirme com
  ``compare-models`` no teste antes de copiar os valores para
  ``configs/project.yaml``. Este módulo não altera o project.yaml.
- O split de teste nunca é usado aqui.

Quem usa: comando ``tune-pilot-bandit``.

Conteúdo do arquivo:
- ``DEFAULT_SPACE``: espaço de busca padrão (usado quando o project.yaml não tem
  ``pilot.search_space`` e não foi passado ``--space``).
- ``REWARD_COLUMN``: coluna do results.csv usada como recompensa.
- ``arms_from_space(space, limit, seed)``: transforma o espaço em lista de braços.
- ``best_validation_reward(run)``: lê a recompensa de um treino.
- ``successive_halving(...)``: o bandit em si; grava ``runs/bandit_*/bandit.json``.
- ``load_space(path)``: lê o JSON do ``--space``.

Quem chama:
- ``cli.py``, comando ``python -m dronecamp_ia tune-pilot-bandit`` (ramo ``elif
  args.command == "tune-pilot-bandit"``), que monta ``fixed`` com ``--imgsz``/``--batch``.
- Testes: ``tests/test_learning_methods.py`` (treino falso) e
  ``tests/test_e2e_ultralytics.py`` (treinos reais).

O que lê e grava no disco:
- Lê: ``<treino>/fit/results.csv`` de cada treino (gerado pela Ultralytics) e o JSON
  opcional do ``--space``.
- Grava: ``runs/bandit_<data>_<id>/bandit.json`` (reescrito depois de cada treino).
  Cada treino cria a própria pasta ``runs/pilot_train_*`` (via ``train_pilot``).

Formato de ``bandit.json`` (chaves principais):
    arms           lista de braços (dicts parâmetro → valor); o índice na lista é o "arm"
    trials         um item por treino: round, arm, epochs, overrides, run, reward, error
    rounds         um item por rodada: epochs, evaluated, kept (braços que seguem), rewards
    complete       True quando terminou com vencedor
    best_arm, best_parameters, best_reward, best_run, next_step   (só se complete)
    reason         motivo quando nenhum treino teve métrica

Onde mexer:
- Para buscar outro hiperparâmetro: acrescente a chave em ``DEFAULT_SPACE`` (ou em
  ``pilot.search_space`` do configs/project.yaml, ou num JSON passado em ``--space``).
  A chave precisa ser um argumento aceito por ``train_pilot`` (parâmetro de treino da
  Ultralytics ou um dos tratados no training.py, como ``tile`` e ``repeat_factor_threshold``).
- Para usar outra métrica como recompensa (ex.: mAP50-95): troque ``REWARD_COLUMN``.
- Para fixar um parâmetro igual em todos os braços: passe em ``fixed`` (no cli.py,
  ramo ``tune-pilot-bandit``, junto de imgsz/batch).
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import random

from .config import ProjectConfig
from .io import make_run_directory, write_json

# ---------------------------------------------------------------------------
# Espaço de busca padrão e métrica de recompensa
# ---------------------------------------------------------------------------

# Espaço padrão: taxa de aprendizado, mosaic e reamostragem de classes raras.
# Formato: {parâmetro: [valores possíveis]}; cada combinação vira um braço
# (aqui 3 × 2 × 2 = 12 braços). Mais valores = mais treinos na rodada 0 (custo alto na CPU).
# - lr0: taxa de aprendizado inicial do otimizador (Ultralytics).
# - mosaic: probabilidade (0 a 1) do aumento de dados "mosaico" (4 fotos numa só).
# - repeat_factor_threshold: limiar da reamostragem de classes raras (sampling.py).
# ATENCAO: None em repeat_factor_threshold NÃO garante "sem reamostragem": o training.py
# troca None pelo pilot.sampling.repeat_factor_threshold do project.yaml, se existir.
# EXTENSAO: novo hiperparâmetro na busca padrão: acrescente "nome": [valores] aqui
# (o nome tem de ser aceito por train_pilot) e, se quiser, também no comentário
# pilot.search_space do configs/project.yaml.
DEFAULT_SPACE = {
    "lr0": [0.0005, 0.001, 0.002],
    "mosaic": [0.0, 0.5],
    "repeat_factor_threshold": [None, 0.3],
}
# Coluna do results.csv da Ultralytics usada como recompensa: mAP50 das caixas (B = box)
# na validação, uma linha por época. Trocar muda o critério de quem sobrevive às rodadas.
# EXTENSAO: outra métrica de recompensa (ex.: "metrics/mAP50-95(B)"): troque aqui; o
# texto "reward" do bandit.json acompanha automaticamente.
REWARD_COLUMN = "metrics/mAP50(B)"


# ---------------------------------------------------------------------------
# Braços e recompensa
# ---------------------------------------------------------------------------

def arms_from_space(space: dict, limit: int | None, seed: int) -> list[dict]:
    """Produto cartesiano do espaço; com ``limit``, sorteia esse número de braços (seed fixa).

    Parâmetros:
        space: dict ``{parâmetro: [valores]}``, ex.: ``{"lr0": [0.001, 0.002], "mosaic": [0.0]}``.
        limit: quantos braços manter (``--arms``); ``None`` ou >= total mantém todos.
        seed: semente do sorteio (mesma semente = mesmos braços).

    Retorna:
        Lista de dicts, um por combinação, ex.: ``[{"lr0": 0.001, "mosaic": 0.0}, ...]``.
        Com ``limit``, a ordem é a do sorteio.

    Chamada por: ``successive_halving``.
    """
    # Chaves em ordem alfabética: a lista de braços (e os índices "arm" no bandit.json)
    # não depende da ordem em que o espaço foi escrito.
    keys = sorted(space)
    # itertools.product gera todas as combinações de valores (uma por braço);
    # zip(keys, values) monta o dict parâmetro → valor de cada combinação.
    arms = [dict(zip(keys, values)) for values in itertools.product(*(space[key] for key in keys))]
    # Limite de braços: amostra sem repetição com gerador próprio (não mexe no random global).
    if limit is not None and limit < len(arms):
        arms = random.Random(seed).sample(arms, limit)
    return arms


def best_validation_reward(run: Path) -> float | None:
    """Maior mAP50 de validação entre as épocas do results.csv (None se não houver).

    Parâmetros:
        run: pasta do treino (``runs/pilot_train_*``); o CSV fica em ``<run>/fit/results.csv``.

    Retorna:
        O maior valor finito de ``REWARD_COLUMN`` entre as épocas, ou ``None`` se o arquivo
        não existir ou não houver valor válido.

    Erros:
        ``ValueError`` se uma célula da coluna não for número (``float`` falha);
        ``successive_halving`` captura e registra como erro do treino.

    Chamada por: ``successive_halving``.
    """
    # Treino que não gerou results.csv (falhou cedo ou é falso nos testes): sem recompensa.
    path = Path(run) / "fit" / "results.csv"
    if not path.is_file():
        return None
    # Leitura do CSV: "utf-8-sig" descarta o BOM, se houver.
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        # ATENCAO: a Ultralytics alinha o cabeçalho com espaços ("   metrics/mAP50(B)");
        # sem o strip a coluna não seria encontrada e a recompensa sairia sempre None.
        reader.fieldnames = [name.strip() for name in reader.fieldnames or []]
        # Um valor por época; linhas com a célula vazia ou ausente são ignoradas.
        values = [float(row[REWARD_COLUMN]) for row in reader if row.get(REWARD_COLUMN) not in (None, "")]
    # Remove NaN/infinito (épocas sem validação válida); write_json recusaria esses valores.
    values = [value for value in values if math.isfinite(value)]
    return max(values) if values else None


# ---------------------------------------------------------------------------
# Successive halving (comando tune-pilot-bandit)
# ---------------------------------------------------------------------------

def successive_halving(config: ProjectConfig, data_path: Path, weights: str | None = None, arms: int | None = None,
                       min_epochs: int = 5, eta: int = 3, rounds: int | None = None, seed: int = 42,
                       space: dict | None = None, fixed: dict | None = None, trainer=None) -> Path:
    """Executa o bandit e grava ``runs/bandit_*/bandit.json`` com todos os treinos e o vencedor.

    ``fixed`` são parâmetros iguais para todos os braços (ex.: imgsz, batch).
    ``trainer`` permite injetar outra função de treino nos testes; o padrão é ``train_pilot``.

    ENTRADA: comando ``tune-pilot-bandit`` do cli.py.

    Parâmetros:
        config: configuração do projeto; ``config.pilot["search_space"]`` é o espaço
            padrão quando ``space`` não é passado.
        data_path: ``dataset.yaml`` do dataset piloto (ex.: ``data/pilot/v6/dataset.yaml``).
        weights: peso inicial de cada treino (``None`` = ``pilot.model``).
        arms: quantos braços sortear do espaço (``None`` = todos).
        min_epochs: épocas de cada braço na rodada 0 (``e``); precisa ser >= 1.
        eta: fator de corte (``η``): a cada rodada fica ~1/eta dos braços e as épocas
            multiplicam por eta; precisa ser >= 2.
        rounds: número de rodadas; ``None`` calcula ``ceil(log_eta(n)) + 1``, o suficiente
            para chegar a um único braço.
        seed: semente do sorteio dos braços e de cada treino (vai em ``overrides["seed"]``).
        space: espaço ``{parâmetro: [valores]}`` (``--space``); tem prioridade sobre o config.
        fixed: parâmetros iguais em todos os braços (ex.: ``{"imgsz": 1280, "batch": 2}``).
        trainer: função ``(config, data_path, weights, overrides) -> Path``; padrão ``train_pilot``.

    Retorna:
        Path da pasta ``runs/bandit_<data>_<id>`` com ``bandit.json``.

    Erros:
        ``ValueError("min_epochs >= 1 e eta >= 2.")`` com parâmetros inválidos;
        ``ValueError("Bandit sem treinos válidos. ...")`` se nenhum treino teve recompensa
        (o bandit.json fica gravado com ``reason``).

    Chamada por: ``cli.py`` (``tune-pilot-bandit``) e testes.
    """
    # 1. Validação dos parâmetros do algoritmo.
    if min_epochs < 1 or eta < 2:
        raise ValueError("min_epochs >= 1 e eta >= 2.")
    # 2. Função de treino: a real (importada só aqui, porque puxa a Ultralytics) ou a injetada.
    if trainer is None:
        from .training import train_pilot as trainer
    # 3. Espaço de busca, em ordem de prioridade: --space > pilot.search_space > DEFAULT_SPACE.
    space = space or (config.pilot.get("search_space") or DEFAULT_SPACE)
    candidates = arms_from_space(space, arms, seed)
    # 4. Rodadas: log na base eta do número de braços, arredondado para cima, + 1
    #    (ex.: 12 braços, eta=3 → ceil(2,26)+1 = 4 rodadas: 12 → 4 → 2 → 1 braço).
    # ATENCAO: espaço vazio (0 braços) faz math.log(0) lançar ValueError ("math domain error").
    rounds = rounds if rounds is not None else max(1, math.ceil(math.log(len(candidates), eta)) + 1)
    # 5. Pasta nova do bandit e estado inicial gravado no bandit.json.
    run = make_run_directory(config.root, "bandit")
    # ATENCAO: "test_used": False documenta a regra do módulo: o split de teste nunca entra
    # aqui (a recompensa vem só da validação). Não use o teste para escolher braços.
    # EXTENSAO: novo campo de auditoria no bandit.json: acrescente a chave aqui.
    state = {"schema_version": 1, "algorithm": "successive_halving", "created_at": datetime.now(timezone.utc).isoformat(),
             "dataset": str(Path(data_path).resolve()), "reward": f"max {REWARD_COLUMN} (validação)",
             "eta": eta, "min_epochs": min_epochs, "seed": seed, "space": space, "fixed": fixed or {},
             "arms": candidates, "trials": [], "rounds": [], "test_used": False, "complete": False}
    # Sobreviventes: índices dos braços em "candidates"; começam todos.
    survivors = list(range(len(candidates)))
    # 6. Laço das rodadas.
    for round_index in range(rounds):
        # Orçamento da rodada: e · eta^k épocas (ex.: 5, 15, 45, 135 com e=5 e eta=3).
        epochs = min_epochs * eta ** round_index
        results = []
        # 6a. Um treino real por braço sobrevivente.
        for arm in survivors:
            # Parâmetros do treino: fixed, depois os do braço (sobrepõem fixed), e por fim
            # epochs e seed, que o bandit controla e sempre vencem.
            overrides = {**(fixed or {}), **candidates[arm], "epochs": epochs, "seed": seed}
            trial = {"round": round_index, "arm": arm, "epochs": epochs, "overrides": overrides}
            # Um treino que falha não derruba o bandit: fica registrado com error e reward None.
            # ATENCAO: só ValueError/RuntimeError/OSError são capturados; outra exceção
            # (ex.: KeyError) interrompe o bandit inteiro.
            try:
                trial_run = trainer(config, Path(data_path), weights, overrides)
                trial.update(run=str(trial_run), reward=best_validation_reward(trial_run))
            except (ValueError, RuntimeError, OSError) as error:
                trial.update(run=None, reward=None, error=str(error))
            state["trials"].append(trial)
            results.append(trial)
            write_json(run / "bandit.json", state)  # salva a cada treino: dá para acompanhar e retomar a análise
        # 6b. Corte: ordena por recompensa decrescente (sem recompensa fica fora) e mantém
        #     ceil(sobreviventes / eta), no mínimo 1.
        scored = sorted((trial for trial in results if trial["reward"] is not None), key=lambda trial: -trial["reward"])
        keep = max(1, math.ceil(len(survivors) / eta))
        survivors = [trial["arm"] for trial in scored[:keep]]
        # Resumo da rodada. ATENCAO: em "rewards" as chaves são índices inteiros dos braços,
        # mas no JSON gravado elas viram texto ("0", "1", ...).
        state["rounds"].append({"round": round_index, "epochs": epochs, "evaluated": len(results),
                                "kept": survivors, "rewards": {trial["arm"]: trial["reward"] for trial in results}})
        # Um sobrevivente ainda treina nas rodadas restantes: o vencedor sai com o orçamento maior.
        if not survivors:
            break
    # 7. Nenhum treino com métrica: grava o motivo e falha.
    if not survivors:
        state["reason"] = "Nenhum treino terminou com métrica de validação."
        write_json(run / "bandit.json", state)
        raise ValueError(f"Bandit sem treinos válidos. Revise {run / 'bandit.json'}.")
    # 8. Vencedor: o melhor da última rodada; o resultado usado é o último treino dele
    #    ([-1]), que é o de maior orçamento de épocas.
    final = [trial for trial in state["trials"] if trial["arm"] == survivors[0]][-1]
    # ATENCAO: o bandit só recomenda; não altera configs/project.yaml (ver next_step).
    state.update(complete=True, best_arm=survivors[0], best_parameters=candidates[survivors[0]],
                 best_reward=final["reward"], best_run=final["run"],
                 next_step="Confirme com compare-models (split test) contra o piloto atual antes de "
                           "copiar best_parameters para pilot.training no configs/project.yaml.")
    write_json(run / "bandit.json", state)
    return run


# ---------------------------------------------------------------------------
# Leitura do espaço de busca (--space)
# ---------------------------------------------------------------------------

def load_space(path: Path | None) -> dict | None:
    """Lê o JSON do espaço de busca passado em ``--space``.

    Parâmetros:
        path: caminho de um JSON ``{"parâmetro": [valores]}`` ou ``None``.

    Retorna:
        O dict lido, ou ``None`` sem caminho (``successive_halving`` usa então o
        ``pilot.search_space`` do project.yaml ou ``DEFAULT_SPACE``).

    Erros:
        ``OSError`` se o arquivo não existir; ``json.JSONDecodeError`` se o JSON for inválido.

    Chamada por: ``cli.py`` (``tune-pilot-bandit``).
    """
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None
