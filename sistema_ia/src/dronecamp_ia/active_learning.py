"""Aprendizado ativo com aprendizado por reforço: qual foto o revisor deve ver primeiro.

Função no projeto: o gargalo do DroneCamp é tempo de revisão humana, não CPU.
Este módulo decide a ORDEM da fila de revisão para que cada hora de revisão
traga o máximo de informação nova ao próximo treino (erros do modelo e classes
raras). Ele nunca aprova, rejeita ou edita caixas: só prioriza.

Formulação (bandit contextual, o caso de um passo do aprendizado por reforço):
- Estado/contexto x_i: características da foto i calculadas a partir das
  sugestões do modelo (``image_context``): incerteza, classes raras sugeridas,
  conflito entre classes, edificação nova, ausência de sugestões etc.
- Ação: escolher a próxima foto a revisar.
- Recompensa r_i (observada depois que o humano revisa, ``review_reward``):
  taxa de erro do modelo na foto, (FP + FN) / (TP + FP + FN + 1), mais um bônus
  por caixas humanas de classes raras. Foto em que o modelo já acerta tudo
  ensina pouco; foto em que erra ou que tem classe rara ensina muito.
- Política: LinUCB (Li et al., WWW 2010) com um único vetor de pesos θ:
      A = λI + Σ x xᵀ,   b = A·θ₀ + Σ r x,   θ = A⁻¹ b
      pontuação(x) = θᵀx + α √(xᵀ A⁻¹ x)
  O primeiro termo explora o que já se aprendeu (exploitation); o segundo
  favorece contextos pouco vistos (exploration). ``θ₀`` é uma priorização
  inicial explícita (``PRIOR_WEIGHTS``) usada enquanto não há recompensas.
- Atualização: depois de importar o feedback humano, ``update_policy`` soma
  x xᵀ em A e r x em b para cada foto revisada. O estado fica em um JSON
  versionável (``policy.json``), com o histórico das fotos já contadas para não
  contar duas vezes a mesma decisão.

Por que bandit e não RL profundo: a cada foto revisada há uma única recompensa
e nenhuma transição de estado relevante; um bandit linear aprende com dezenas de
revisões, enquanto uma rede de política precisaria de milhares.

Quem usa: comandos ``prioritize-review`` e ``learn-review``.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path

import numpy as np

from .config import ProjectConfig, detection_names, load_taxonomy
from .io import file_hash, write_json
from .matching import image_errors
from .pilot import box_iou

SCHEMA = 1
FEATURES = ["bias", "log_suggestions", "mean_confidence", "max_confidence", "uncertainty", "rare_class_score",
            "class_conflict", "new_building", "no_suggestions", "calibrated_uncertainty"]
# Priorização inicial (θ₀), antes de qualquer recompensa: incerteza, classe rara,
# conflito entre classes e edificação nova sobem a foto na fila.
PRIOR_WEIGHTS = {"uncertainty": 0.5, "rare_class_score": 0.5, "class_conflict": 0.3, "new_building": 0.3,
                 "no_suggestions": 0.2, "calibrated_uncertainty": 0.3}
RARE_BONUS = 0.5


def class_rarity(train_counts: dict[int, int], class_count: int) -> list[float]:
    """1/√(1 + caixas de treino): classe sem exemplos vale 1; com 99 exemplos vale 0,1."""
    return [1.0 / math.sqrt(1 + train_counts.get(index, 0)) for index in range(class_count)]


def image_context(boxes: list[dict], rarity: list[float], new_building: bool) -> np.ndarray:
    """Vetor de contexto de uma foto a partir das sugestões do modelo (na ordem de ``FEATURES``)."""
    confidences = [float(box["confidence"]) for box in boxes]
    # Incerteza binária: 1 quando a confiança é 0,5; 0 quando é 0 ou 1.
    uncertainty = [1 - abs(2 * value - 1) for value in confidences]
    calibrated = [1 - abs(2 * float(box["acceptance_probability"]) - 1) for box in boxes
                  if box.get("acceptance_probability") is not None]
    conflicts = sum(1 for index, box in enumerate(boxes)
                    if any(other["class_id"] != box["class_id"] and box_iou(box["bbox_xyxy"], other["bbox_xyxy"]) > 0.5
                           for position, other in enumerate(boxes) if position != index))
    return np.asarray([
        1.0, math.log1p(len(boxes)),
        float(np.mean(confidences)) if boxes else 0.0, max(confidences, default=0.0),
        float(np.mean(uncertainty)) if boxes else 0.0,
        max((rarity[box["class_id"]] for box in boxes), default=0.0),
        conflicts / len(boxes) if boxes else 0.0,
        1.0 if new_building else 0.0, 0.0 if boxes else 1.0,
        float(np.mean(calibrated)) if calibrated else 0.0,
    ], dtype=float)


def review_reward(suggestions: list[dict], human_boxes: list[dict], rarity: list[float]) -> dict:
    """Recompensa de uma foto revisada: erro do modelo + bônus por classes raras anotadas."""
    errors = image_errors(suggestions, human_boxes)
    error_rate = (errors["fp"] + errors["fn"]) / (errors["tp"] + errors["fp"] + errors["fn"] + 1)
    rare = max((rarity[box["class_id"]] for box in human_boxes), default=0.0)
    return {"reward": round(error_rate + RARE_BONUS * rare, 6), "error_rate": round(error_rate, 6),
            "rare_bonus": round(RARE_BONUS * rare, 6), "tp": errors["tp"], "fp": errors["fp"], "fn": errors["fn"]}


# ---------------------------------------------------------------------------
# Política LinUCB persistida em JSON.
# ---------------------------------------------------------------------------

class LinUCBPolicy:
    """LinUCB com vetor de pesos compartilhado; ``A`` e ``b`` são o estado aprendido."""

    def __init__(self, alpha: float = 1.0, ridge: float = 1.0, A=None, b=None, history=None):
        size = len(FEATURES)
        prior = np.asarray([PRIOR_WEIGHTS.get(name, 0.0) for name in FEATURES])
        self.alpha, self.ridge = float(alpha), float(ridge)
        self.A = np.eye(size) * ridge if A is None else np.asarray(A, dtype=float)
        self.b = self.A @ prior if b is None else np.asarray(b, dtype=float)
        self.history = list(history or [])

    @property
    def theta(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    def score(self, context: np.ndarray) -> dict:
        inverse_context = np.linalg.solve(self.A, context)
        exploit = float(self.theta @ context)
        explore = self.alpha * math.sqrt(max(0.0, float(context @ inverse_context)))
        return {"score": exploit + explore, "exploit": exploit, "explore": explore}

    def update(self, context: np.ndarray, reward: float) -> None:
        self.A += np.outer(context, context)
        self.b += reward * context

    def to_json(self) -> dict:
        return {"schema_version": SCHEMA, "algorithm": "linucb_shared", "features": FEATURES,
                "prior_weights": PRIOR_WEIGHTS, "alpha": self.alpha, "ridge": self.ridge,
                "A": self.A.round(10).tolist(), "b": self.b.round(10).tolist(),
                "theta": dict(zip(FEATURES, self.theta.round(6).tolist())),
                "updates": len(self.history), "history": self.history}

    @classmethod
    def load(cls, path: Path | None, alpha: float | None = None) -> "LinUCBPolicy":
        if path is None or not Path(path).is_file():
            return cls(alpha if alpha is not None else 1.0)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA or data.get("features") != FEATURES:
            raise ValueError("policy.json de outra versão das características; crie uma política nova.")
        return cls(alpha if alpha is not None else data["alpha"], data["ridge"], data["A"], data["b"], data["history"])


# ---------------------------------------------------------------------------
# Leitura de registro, sugestões e contagem de classes de treino.
# ---------------------------------------------------------------------------

def _load_review(registry_path: Path, suggestions_path: Path | None) -> tuple[dict, dict]:
    registry_path = Path(registry_path).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    suggestions_path = Path(suggestions_path) if suggestions_path else registry_path.parent / "suggestions.json"
    if not suggestions_path.is_file():
        raise ValueError(f"Sem {suggestions_path.name}: gere sugestões com o comando suggest antes de priorizar.")
    return registry, json.loads(suggestions_path.read_text(encoding="utf-8"))


def training_context(pilot_manifest: Path | None, class_count: int) -> tuple[dict[int, int], set[str]]:
    """Caixas de treino por classe e edificações já vistas, a partir de um pilot.json."""
    if not pilot_manifest:
        return {}, set()
    manifest = json.loads(Path(pilot_manifest).read_text(encoding="utf-8"))
    names = list(manifest["boxes_by_class_and_split"])
    counts = {index: manifest["boxes_by_class_and_split"][name]["train"] for index, name in enumerate(names[:class_count])}
    return counts, set(manifest.get("buildings", []))


def prioritize_review(config: ProjectConfig, registry_path: Path, policy_path: Path | None = None,
                      pilot_manifest: Path | None = None, suggestions_path: Path | None = None,
                      alpha: float | None = None) -> Path:
    """Ordena as fotos ainda sem decisão humana e grava review_priority.json/.csv ao lado do registro."""
    names = detection_names(load_taxonomy(config.taxonomy_path))
    registry, suggestions = _load_review(registry_path, suggestions_path)
    policy = LinUCBPolicy.load(policy_path, alpha)
    counts, buildings = training_context(pilot_manifest, len(names))
    rarity = class_rarity(counts, len(names))
    rows = []
    for item in registry["images"]:
        if item.get("human_approved") is True:
            continue  # já decidida; a fila é só do que falta revisar
        boxes = suggestions["images"].get(item["image_sha256"], [])
        context = image_context(boxes, rarity, bool(buildings) and item.get("building_group") not in buildings)
        score = policy.score(context)
        suggested = Counter(names[box["class_id"]] for box in boxes)
        rows.append({"image_sha256": item["image_sha256"], "filename": item.get("filename"),
                     "building_group": item.get("building_group"), "suggestions": len(boxes),
                     "suggested_classes": dict(suggested), **{key: round(value, 4) for key, value in score.items()},
                     "context": dict(zip(FEATURES, context.round(4).tolist()))})
    rows.sort(key=lambda row: -row["score"])
    for position, row in enumerate(rows, start=1):
        row["priority"] = position
    output = Path(registry_path).resolve().parent / "review_priority.json"
    write_json(output, {
        "schema_version": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "registry_sha256": file_hash(Path(registry_path)), "suggestions_weights_sha256": suggestions.get("weights_sha256"),
        "policy": str(policy_path) if policy_path else None, "policy_updates": len(policy.history),
        "alpha": policy.alpha, "theta": dict(zip(FEATURES, policy.theta.round(4).tolist())),
        "note": "Ordem sugerida de revisão. Não altera status, caixas nem aprovações.", "images": rows,
    })
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["prioridade", "arquivo", "edificacao", "sugestoes", "pontuacao", "aproveitamento", "exploracao",
                         "classes_sugeridas"])
        for row in rows:
            writer.writerow([row["priority"], row["filename"], row["building_group"], row["suggestions"],
                             row["score"], row["exploit"], row["explore"],
                             "; ".join(f"{name}={count}" for name, count in row["suggested_classes"].items())])
    return output


def update_policy(config: ProjectConfig, reviewed_registry: Path, policy_path: Path,
                  pilot_manifest: Path | None = None, suggestions_path: Path | None = None,
                  alpha: float | None = None) -> dict:
    """Aprende com as decisões humanas: uma atualização LinUCB por foto aprovada que tinha sugestões.

    ``reviewed_registry`` é o registro já com o feedback importado (``import-review``);
    ``suggestions_path`` é o suggestions.json que o revisor viu (padrão: ao lado do registro).
    """
    names = detection_names(load_taxonomy(config.taxonomy_path))
    registry, suggestions = _load_review(reviewed_registry, suggestions_path)
    policy = LinUCBPolicy.load(policy_path, alpha)
    counts, buildings = training_context(pilot_manifest, len(names))
    rarity = class_rarity(counts, len(names))
    counted = {(entry["image_sha256"], entry["suggestions_weights_sha256"]) for entry in policy.history}
    rewards = []
    for item in registry["images"]:
        key = (item["image_sha256"], suggestions.get("weights_sha256"))
        if item.get("human_approved") is not True or key in counted:
            continue
        boxes = suggestions["images"].get(item["image_sha256"], [])
        context = image_context(boxes, rarity, bool(buildings) and item.get("building_group") not in buildings)
        reward = review_reward(boxes, item["boxes"], rarity)
        policy.update(context, reward["reward"])
        entry = {"image_sha256": item["image_sha256"], "suggestions_weights_sha256": key[1],
                 "registry_sha256": file_hash(Path(reviewed_registry)), **reward}
        policy.history.append(entry)
        rewards.append(entry)
    write_json(Path(policy_path), policy.to_json())
    return {"policy": str(Path(policy_path).resolve()), "new_updates": len(rewards), "total_updates": len(policy.history),
            "mean_reward": round(float(np.mean([entry["reward"] for entry in rewards])), 4) if rewards else None,
            "theta": dict(zip(FEATURES, policy.theta.round(4).tolist()))}
