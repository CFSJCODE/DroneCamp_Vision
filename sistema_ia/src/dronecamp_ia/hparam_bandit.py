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

# Espaço padrão: taxa de aprendizado, mosaic e reamostragem de classes raras.
DEFAULT_SPACE = {
    "lr0": [0.0005, 0.001, 0.002],
    "mosaic": [0.0, 0.5],
    "repeat_factor_threshold": [None, 0.3],
}
REWARD_COLUMN = "metrics/mAP50(B)"


def arms_from_space(space: dict, limit: int | None, seed: int) -> list[dict]:
    """Produto cartesiano do espaço; com ``limit``, sorteia esse número de braços (seed fixa)."""
    keys = sorted(space)
    arms = [dict(zip(keys, values)) for values in itertools.product(*(space[key] for key in keys))]
    if limit is not None and limit < len(arms):
        arms = random.Random(seed).sample(arms, limit)
    return arms


def best_validation_reward(run: Path) -> float | None:
    """Maior mAP50 de validação entre as épocas do results.csv (None se não houver)."""
    path = Path(run) / "fit" / "results.csv"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        reader.fieldnames = [name.strip() for name in reader.fieldnames or []]
        values = [float(row[REWARD_COLUMN]) for row in reader if row.get(REWARD_COLUMN) not in (None, "")]
    values = [value for value in values if math.isfinite(value)]
    return max(values) if values else None


def successive_halving(config: ProjectConfig, data_path: Path, weights: str | None = None, arms: int | None = None,
                       min_epochs: int = 5, eta: int = 3, rounds: int | None = None, seed: int = 42,
                       space: dict | None = None, fixed: dict | None = None, trainer=None) -> Path:
    """Executa o bandit e grava ``runs/bandit_*/bandit.json`` com todos os treinos e o vencedor.

    ``fixed`` são parâmetros iguais para todos os braços (ex.: imgsz, batch).
    ``trainer`` permite injetar outra função de treino nos testes; o padrão é ``train_pilot``.
    """
    if min_epochs < 1 or eta < 2:
        raise ValueError("min_epochs >= 1 e eta >= 2.")
    if trainer is None:
        from .training import train_pilot as trainer
    space = space or (config.pilot.get("search_space") or DEFAULT_SPACE)
    candidates = arms_from_space(space, arms, seed)
    rounds = rounds if rounds is not None else max(1, math.ceil(math.log(len(candidates), eta)) + 1)
    run = make_run_directory(config.root, "bandit")
    state = {"schema_version": 1, "algorithm": "successive_halving", "created_at": datetime.now(timezone.utc).isoformat(),
             "dataset": str(Path(data_path).resolve()), "reward": f"max {REWARD_COLUMN} (validação)",
             "eta": eta, "min_epochs": min_epochs, "seed": seed, "space": space, "fixed": fixed or {},
             "arms": candidates, "trials": [], "rounds": [], "test_used": False, "complete": False}
    survivors = list(range(len(candidates)))
    for round_index in range(rounds):
        epochs = min_epochs * eta ** round_index
        results = []
        for arm in survivors:
            overrides = {**(fixed or {}), **candidates[arm], "epochs": epochs, "seed": seed}
            trial = {"round": round_index, "arm": arm, "epochs": epochs, "overrides": overrides}
            try:
                trial_run = trainer(config, Path(data_path), weights, overrides)
                trial.update(run=str(trial_run), reward=best_validation_reward(trial_run))
            except (ValueError, RuntimeError, OSError) as error:
                trial.update(run=None, reward=None, error=str(error))
            state["trials"].append(trial)
            results.append(trial)
            write_json(run / "bandit.json", state)  # salva a cada treino: dá para acompanhar e retomar a análise
        scored = sorted((trial for trial in results if trial["reward"] is not None), key=lambda trial: -trial["reward"])
        keep = max(1, math.ceil(len(survivors) / eta))
        survivors = [trial["arm"] for trial in scored[:keep]]
        state["rounds"].append({"round": round_index, "epochs": epochs, "evaluated": len(results),
                                "kept": survivors, "rewards": {trial["arm"]: trial["reward"] for trial in results}})
        # Um sobrevivente ainda treina nas rodadas restantes: o vencedor sai com o orçamento maior.
        if not survivors:
            break
    if not survivors:
        state["reason"] = "Nenhum treino terminou com métrica de validação."
        write_json(run / "bandit.json", state)
        raise ValueError(f"Bandit sem treinos válidos. Revise {run / 'bandit.json'}.")
    final = [trial for trial in state["trials"] if trial["arm"] == survivors[0]][-1]
    state.update(complete=True, best_arm=survivors[0], best_parameters=candidates[survivors[0]],
                 best_reward=final["reward"], best_run=final["run"],
                 next_step="Confirme com compare-models (split test) contra o piloto atual antes de "
                           "copiar best_parameters para pilot.training no configs/project.yaml.")
    write_json(run / "bandit.json", state)
    return run


def load_space(path: Path | None) -> dict | None:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None
