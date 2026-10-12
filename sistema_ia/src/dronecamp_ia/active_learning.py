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

Glossário das fórmulas (para quem for mexer):
- λ (``ridge``): regularização; A começa como λ·I (matriz identidade 10x10).
  λ maior = a política demora mais para se afastar de θ₀.
- α (``alpha``): peso da exploração; α = 0 ordena só pelo que já aprendeu,
  α grande favorece fotos com contexto "diferente" do que já foi revisado.
- √(xᵀ A⁻¹ x): incerteza da política sobre o contexto x; diminui à medida que
  contextos parecidos são revisados (A cresce nessa direção).
- Raridade de uma classe c: 1 / √(1 + n_c), n_c = caixas de treino da classe.
- Recompensa: r = (FP + FN) / (TP + FP + FN + 1) + RARE_BONUS · max(raridade das
  classes humanas da foto). O "+1" no denominador evita divisão por zero e reduz
  o peso de fotos com uma única caixa.
- Incerteza binária de uma confiança p: 1 - |2p - 1| (1 em p=0,5; 0 em p=0 ou 1).

O que o arquivo contém:
- Constantes: ``SCHEMA``, ``FEATURES``, ``PRIOR_WEIGHTS``, ``RARE_BONUS``.
- ``class_rarity``, ``image_context``, ``review_reward``: contexto e recompensa.
- ``LinUCBPolicy``: estado (A, b, histórico), pontuação, atualização e JSON.
- ``_load_review``, ``training_context``: leitura de registro/sugestões/pilot.json.
- ``prioritize_review`` (ENTRADA de ``prioritize-review``) e ``update_policy``
  (ENTRADA de ``learn-review``).

Disco:
- Lê: ``registry.json`` (fotos e decisões), ``suggestions.json`` (caixas do modelo),
  ``pilot.json`` opcional (contagem de caixas de treino e edificações) e
  ``policy.json`` opcional (política aprendida).
- Grava: ``review_priority.json`` e ``review_priority.csv`` ao lado do registro
  (``prioritize-review``) e o ``policy.json`` indicado (``learn-review``).

Onde mexer:
- Para acrescentar uma característica ao contexto: veja o EXTENSAO em ``FEATURES``
  (lista + ``image_context`` + opcionalmente ``PRIOR_WEIGHTS``). ATENCAO: policy.json
  antigos deixam de carregar (a lista de FEATURES é conferida).
- Para mudar quanto uma característica pesa ANTES de haver revisões: ``PRIOR_WEIGHTS``.
- Para mudar a recompensa (ex.: pesar mais omissões que falsos positivos):
  ``review_reward``.
- Para uma coluna nova no CSV da fila: ``prioritize_review``, bloco do CSV.
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

# ---------------------------------------------------------------------------
# Constantes: versão do formato, características do contexto e priorização inicial
# ---------------------------------------------------------------------------

# Versão do formato de policy.json e review_priority.json. ATENCAO: mude (2, 3...)
# quando o significado do vetor de contexto mudar; LinUCBPolicy.load recusa
# política de outra versão para não misturar A/b calculados com outro contexto.
SCHEMA = 1
# Nomes das características do contexto x, NA ORDEM das posições do vetor que
# image_context devolve (10 posições hoje):
#   bias                   sempre 1 (termo constante da regressão linear)
#   log_suggestions        ln(1 + nº de sugestões na foto)
#   mean_confidence        média das confianças das sugestões (0-1)
#   max_confidence         maior confiança (0-1)
#   uncertainty            média de 1 - |2p - 1| das confianças (1 = mais incerto)
#   rare_class_score       maior raridade entre as classes sugeridas (0-1)
#   class_conflict         fração de sugestões sobrepostas (IoU > 0,5) a outra de classe diferente
#   new_building           1 se a edificação da foto não estava no treino
#   no_suggestions         1 se o modelo não sugeriu nada na foto
#   calibrated_uncertainty média de 1 - |2q - 1| com q = acceptance_probability do calibrador
# EXTENSAO: característica nova = (1) nome no FIM desta lista; (2) valor na MESMA
# posição no np.asarray de image_context; (3) peso inicial opcional em PRIOR_WEIGHTS.
# ATENCAO: qualquer mudança aqui invalida policy.json existentes (load confere a
# lista inteira); crie uma política nova depois e considere subir SCHEMA.
FEATURES = ["bias", "log_suggestions", "mean_confidence", "max_confidence", "uncertainty", "rare_class_score",
            "class_conflict", "new_building", "no_suggestions", "calibrated_uncertainty"]
# Priorização inicial (θ₀), antes de qualquer recompensa: incerteza, classe rara,
# conflito entre classes e edificação nova sobem a foto na fila.
# Valores adimensionais somados como θ₀ᵀx; característica ausente do dict tem peso 0.
# Subir um valor faz fotos com aquela característica subirem na fila enquanto há
# poucas revisões; com muitas revisões o efeito de θ₀ dilui (A e b crescem).
# EXTENSAO: peso inicial de uma característica nova (o nome precisa estar em FEATURES).
PRIOR_WEIGHTS = {"uncertainty": 0.5, "rare_class_score": 0.5, "class_conflict": 0.3, "new_building": 0.3,
                 "no_suggestions": 0.2, "calibrated_uncertainty": 0.3}
# Peso do bônus de classe rara na recompensa (adimensional). Com 0,5, uma foto com
# caixa humana de classe sem exemplos de treino (raridade 1) ganha +0,5, o mesmo
# que uma taxa de erro de 50%. Subir = a política aprende a caçar classes raras.
RARE_BONUS = 0.5


# ---------------------------------------------------------------------------
# Contexto da foto e recompensa da revisão
# ---------------------------------------------------------------------------

def class_rarity(train_counts: dict[int, int], class_count: int) -> list[float]:
    """1/√(1 + caixas de treino): classe sem exemplos vale 1; com 99 exemplos vale 0,1.

    Parâmetros:
        train_counts: ``{class_id: nº de caixas no split de treino}`` (de
            ``training_context``); classe ausente conta como 0.
        class_count: nº de classes da taxonomia (tamanho da lista devolvida).

    Retorna:
        Lista de float (0, 1], indexada por class_id.
    """
    # Raridade_c = 1 / sqrt(1 + n_c). Sem pilot.json, todas as classes valem 1.
    return [1.0 / math.sqrt(1 + train_counts.get(index, 0)) for index in range(class_count)]


def image_context(boxes: list[dict], rarity: list[float], new_building: bool) -> np.ndarray:
    """Vetor de contexto de uma foto a partir das sugestões do modelo (na ordem de ``FEATURES``).

    Parâmetros:
        boxes: sugestões do modelo na foto (de ``suggestions.json``), cada uma com
            ``class_id``, ``bbox_xyxy`` (pixels), ``confidence`` (0-1) e,
            se houve calibrador, ``acceptance_probability`` (0-1).
        rarity: saída de ``class_rarity`` (indexada por class_id).
        new_building: True se a edificação da foto não aparece no treino.

    Retorna:
        ``np.ndarray`` float de tamanho ``len(FEATURES)`` (10), na ordem de FEATURES.

    Erros:
        IndexError se uma caixa tiver class_id >= len(rarity) (sugestão de outra taxonomia).
    """
    # Confianças cruas do YOLO de cada sugestão.
    confidences = [float(box["confidence"]) for box in boxes]
    # Incerteza binária: 1 quando a confiança é 0,5; 0 quando é 0 ou 1.
    uncertainty = [1 - abs(2 * value - 1) for value in confidences]
    # Mesma incerteza, mas sobre a probabilidade do calibrador (só caixas que a têm).
    calibrated = [1 - abs(2 * float(box["acceptance_probability"]) - 1) for box in boxes
                  if box.get("acceptance_probability") is not None]
    # Conflito: quantas sugestões se sobrepõem (IoU > 0,5) a outra sugestão de
    # CLASSE DIFERENTE (o modelo está em dúvida entre classes naquele lugar).
    conflicts = sum(1 for index, box in enumerate(boxes)
                    if any(other["class_id"] != box["class_id"] and box_iou(box["bbox_xyxy"], other["bbox_xyxy"]) > 0.5
                           for position, other in enumerate(boxes) if position != index))
    # ATENCAO: a ordem abaixo TEM de ser a de FEATURES (posição = significado).
    # Foto sem sugestões: médias e máximos ficam 0 e no_suggestions = 1.
    return np.asarray([
        # bias, log_suggestions = ln(1 + n)
        1.0, math.log1p(len(boxes)),
        # mean_confidence, max_confidence
        float(np.mean(confidences)) if boxes else 0.0, max(confidences, default=0.0),
        # uncertainty (média)
        float(np.mean(uncertainty)) if boxes else 0.0,
        # rare_class_score: a classe sugerida mais rara
        max((rarity[box["class_id"]] for box in boxes), default=0.0),
        # class_conflict: fração (0-1) das sugestões em conflito
        conflicts / len(boxes) if boxes else 0.0,
        # new_building, no_suggestions
        1.0 if new_building else 0.0, 0.0 if boxes else 1.0,
        # calibrated_uncertainty (0 sem calibrador)
        float(np.mean(calibrated)) if calibrated else 0.0,
        # EXTENSAO: valor da característica nova aqui, na mesma posição do nome em FEATURES.
    ], dtype=float)


def review_reward(suggestions: list[dict], human_boxes: list[dict], rarity: list[float]) -> dict:
    """Recompensa de uma foto revisada: erro do modelo + bônus por classes raras anotadas.

    Parâmetros:
        suggestions: o que o modelo sugeriu na foto (o que o revisor viu).
        human_boxes: caixas finais aprovadas pelo humano (``item["boxes"]`` do registro).
        rarity: saída de ``class_rarity``.

    Retorna:
        ``{"reward", "error_rate", "rare_bonus", "tp", "fp", "fn"}``, com
        ``reward = error_rate + rare_bonus`` (6 casas decimais).
    """
    # TP/FP/FN da foto com a régua comum (matching.py, IoU >= 0,5 e mesma classe).
    errors = image_errors(suggestions, human_boxes)
    # Taxa de erro = (FP + FN) / (TP + FP + FN + 1); 0 quando o modelo acertou tudo.
    # EXTENSAO: para pesar mais omissões (FN) que alarmes (FP), use pesos diferentes aqui
    # (ex.: (0.5 * fp + 1.0 * fn)); policy.json antigos continuam válidos, mas misturam réguas.
    error_rate = (errors["fp"] + errors["fn"]) / (errors["tp"] + errors["fp"] + errors["fn"] + 1)
    # Bônus pela classe humana mais rara presente na foto (0 se a foto não tem caixa).
    rare = max((rarity[box["class_id"]] for box in human_boxes), default=0.0)
    return {"reward": round(error_rate + RARE_BONUS * rare, 6), "error_rate": round(error_rate, 6),
            "rare_bonus": round(RARE_BONUS * rare, 6), "tp": errors["tp"], "fp": errors["fp"], "fn": errors["fn"]}


# ---------------------------------------------------------------------------
# Política LinUCB persistida em JSON.
# ---------------------------------------------------------------------------

class LinUCBPolicy:
    """LinUCB com vetor de pesos compartilhado; ``A`` e ``b`` são o estado aprendido.

    "Compartilhado" quer dizer um único θ para todas as fotos (as fotos são os
    braços, mas o que distingue uma da outra é só o contexto x).

    Atributos:
        alpha (float): peso da exploração α.
        ridge (float): regularização λ usada para criar A.
        A (np.ndarray 10x10): λI + Σ x xᵀ.
        b (np.ndarray 10): A₀·θ₀ + Σ r x.
        history (list[dict]): uma entrada por foto já contada (hash da foto, hash
            dos pesos das sugestões, hash do registro e a recompensa).
    """

    def __init__(self, alpha: float = 1.0, ridge: float = 1.0, A=None, b=None, history=None):
        """Cria uma política nova (A=λI, b=λθ₀) ou reconstrói uma salva.

        Parâmetros:
            alpha: peso da exploração (padrão 1.0).
            ridge: λ da regularização (padrão 1.0).
            A, b: listas/arrays salvos no policy.json; None = política nova.
            history: lista de entradas já contadas; None = vazia.
        """
        size = len(FEATURES)
        # θ₀ como vetor na ordem de FEATURES (peso 0 para quem não está em PRIOR_WEIGHTS).
        prior = np.asarray([PRIOR_WEIGHTS.get(name, 0.0) for name in FEATURES])
        self.alpha, self.ridge = float(alpha), float(ridge)
        # A₀ = λI: sem dados, a política é uma regressão ridge "vazia".
        self.A = np.eye(size) * ridge if A is None else np.asarray(A, dtype=float)
        # b₀ = A₀·θ₀, para que θ = A⁻¹b comece exatamente igual a θ₀.
        self.b = self.A @ prior if b is None else np.asarray(b, dtype=float)
        self.history = list(history or [])

    @property
    def theta(self) -> np.ndarray:
        """Pesos atuais θ = A⁻¹ b (resolve o sistema linear em vez de inverter A).

        Retorna:
            ``np.ndarray`` de tamanho ``len(FEATURES)``.
        """
        return np.linalg.solve(self.A, self.b)

    def score(self, context: np.ndarray) -> dict:
        """Pontuação LinUCB de um contexto: θᵀx + α √(xᵀ A⁻¹ x).

        Parâmetros:
            context: vetor x de ``image_context``.

        Retorna:
            ``{"score": total, "exploit": θᵀx, "explore": α√(xᵀA⁻¹x)}`` (floats).
        """
        # A⁻¹x sem inverter A explicitamente (mais estável numericamente).
        inverse_context = np.linalg.solve(self.A, context)
        # Parte "aproveitamento": quanto a política já aprendeu que esse contexto rende.
        exploit = float(self.theta @ context)
        # Parte "exploração": incerteza sobre o contexto; max(0, ...) protege contra
        # um valor negativo minúsculo por arredondamento antes da raiz.
        explore = self.alpha * math.sqrt(max(0.0, float(context @ inverse_context)))
        return {"score": exploit + explore, "exploit": exploit, "explore": explore}

    def update(self, context: np.ndarray, reward: float) -> None:
        """Aprende com uma foto revisada: A += x xᵀ e b += r x.

        Parâmetros:
            context: vetor x da foto (o mesmo cálculo usado ao priorizar).
            reward: recompensa r de ``review_reward``.
        """
        self.A += np.outer(context, context)
        self.b += reward * context

    def to_json(self) -> dict:
        """Estado completo para gravar em policy.json (também legível por humanos).

        Retorna:
            Dict com ``schema_version``, ``algorithm``, ``features``, ``prior_weights``,
            ``alpha``, ``ridge``, ``A``, ``b`` (listas), ``theta`` (por nome),
            ``updates`` (nº de fotos contadas) e ``history``.
        """
        # EXTENSAO: campo novo no policy.json: acrescente aqui; se ele precisar ser
        # lido de volta, leia em load() e passe ao __init__.
        return {"schema_version": SCHEMA, "algorithm": "linucb_shared", "features": FEATURES,
                "prior_weights": PRIOR_WEIGHTS, "alpha": self.alpha, "ridge": self.ridge,
                "A": self.A.round(10).tolist(), "b": self.b.round(10).tolist(),
                "theta": dict(zip(FEATURES, self.theta.round(6).tolist())),
                "updates": len(self.history), "history": self.history}

    @classmethod
    def load(cls, path: Path | None, alpha: float | None = None) -> "LinUCBPolicy":
        """Abre policy.json, ou cria uma política nova se o arquivo não existir.

        Parâmetros:
            path: caminho do policy.json (None ou inexistente = política nova).
            alpha: α informado na CLI; None mantém o salvo (ou 1.0 se for nova).

        Retorna:
            ``LinUCBPolicy`` pronta para pontuar/atualizar.

        Erros:
            ValueError "policy.json de outra versão das características; crie uma
            política nova." quando ``schema_version`` ou ``features`` diferem.
        """
        # Arquivo ausente: política nova com θ = θ₀ (λ = 1.0 padrão).
        if path is None or not Path(path).is_file():
            return cls(alpha if alpha is not None else 1.0)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # ATENCAO: A e b só fazem sentido com a MESMA lista de características.
        if data.get("schema_version") != SCHEMA or data.get("features") != FEATURES:
            raise ValueError("policy.json de outra versão das características; crie uma política nova.")
        return cls(alpha if alpha is not None else data["alpha"], data["ridge"], data["A"], data["b"], data["history"])


# ---------------------------------------------------------------------------
# Leitura de registro, sugestões e contagem de classes de treino.
# ---------------------------------------------------------------------------

def _load_review(registry_path: Path, suggestions_path: Path | None) -> tuple[dict, dict]:
    """Lê o registro da revisão e o suggestions.json que acompanha.

    Parâmetros:
        registry_path: ``data/reviews/<inspeção>/registry.json``.
        suggestions_path: suggestions.json explícito; None usa o que está ao lado do registro.

    Retorna:
        ``(registry, suggestions)`` como dicts. ``suggestions["images"]`` é
        ``{image_sha256: [caixas]}``; ``suggestions["weights_sha256"]`` identifica o modelo.

    Erros:
        ValueError "Sem suggestions.json: gere sugestões com o comando suggest antes
        de priorizar." quando o arquivo não existe; OSError/JSONDecodeError se ilegível.
    """
    registry_path = Path(registry_path).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    # Padrão: suggestions.json na mesma pasta do registro (onde o comando suggest grava).
    suggestions_path = Path(suggestions_path) if suggestions_path else registry_path.parent / "suggestions.json"
    if not suggestions_path.is_file():
        raise ValueError(f"Sem {suggestions_path.name}: gere sugestões com o comando suggest antes de priorizar.")
    return registry, json.loads(suggestions_path.read_text(encoding="utf-8"))


def training_context(pilot_manifest: Path | None, class_count: int) -> tuple[dict[int, int], set[str]]:
    """Caixas de treino por classe e edificações já vistas, a partir de um pilot.json.

    Parâmetros:
        pilot_manifest: ``data/pilot/<versão>/pilot.json`` do dataset que treinou o
            modelo atual; None = sem informação (todas as classes contam como raras e
            nenhuma edificação é "nova").
        class_count: nº de classes da taxonomia (corta classes excedentes do manifesto).

    Retorna:
        ``(counts, buildings)``: ``counts = {class_id: caixas no split train}`` e
        ``buildings`` = conjunto de nomes de edificação presentes no dataset.

    Erros:
        KeyError se o pilot.json não tiver ``boxes_by_class_and_split`` (manifesto antigo).
    """
    # Sem manifesto: contagens vazias e nenhuma edificação conhecida.
    if not pilot_manifest:
        return {}, set()
    manifest = json.loads(Path(pilot_manifest).read_text(encoding="utf-8"))
    # ATENCAO: o class_id é deduzido da POSIÇÃO da classe no dict do manifesto;
    # isso só é correto porque pilot.py grava as classes na ordem da taxonomia.
    names = list(manifest["boxes_by_class_and_split"])
    counts = {index: manifest["boxes_by_class_and_split"][name]["train"] for index, name in enumerate(names[:class_count])}
    return counts, set(manifest.get("buildings", []))


# ---------------------------------------------------------------------------
# Comandos: prioritize-review (ordenar a fila) e learn-review (aprender)
# ---------------------------------------------------------------------------

# ENTRADA: comando ``prioritize-review`` (cli.py chama esta função).
def prioritize_review(config: ProjectConfig, registry_path: Path, policy_path: Path | None = None,
                      pilot_manifest: Path | None = None, suggestions_path: Path | None = None,
                      alpha: float | None = None) -> Path:
    """Ordena as fotos ainda sem decisão humana e grava review_priority.json/.csv ao lado do registro.

    Parâmetros:
        config: configuração do projeto (para achar a taxonomia).
        registry_path: registry.json da revisão a priorizar.
        policy_path: policy.json aprendido (``learn-review``); None = só θ₀.
        pilot_manifest: pilot.json do treino atual (raridade e edificações).
        suggestions_path: suggestions.json; None = ao lado do registro.
        alpha: peso da exploração; None = o salvo na política (ou 1.0).

    Retorna:
        Path de ``review_priority.json`` (o .csv fica ao lado, mesmo nome).

    Erros:
        ValueError de ``_load_review`` (sem sugestões) e de ``LinUCBPolicy.load``.
        ATENCAO: sobrescreve review_priority.json/.csv anteriores da mesma pasta
        (é só uma ordem sugerida, não dado de revisão).
    """
    # 1. Classes, registro, sugestões, política e contexto de treino.
    names = detection_names(load_taxonomy(config.taxonomy_path))
    registry, suggestions = _load_review(registry_path, suggestions_path)
    policy = LinUCBPolicy.load(policy_path, alpha)
    counts, buildings = training_context(pilot_manifest, len(names))
    rarity = class_rarity(counts, len(names))
    rows = []
    # 2. Uma linha por foto pendente, com contexto e pontuação.
    for item in registry["images"]:
        if item.get("human_approved") is True:
            continue  # já decidida; a fila é só do que falta revisar
        # Sugestões da foto, achadas pelo hash SHA-256 (foto sem sugestão = lista vazia).
        boxes = suggestions["images"].get(item["image_sha256"], [])
        # "Edificação nova" só é avaliada quando há pilot.json com edificações.
        context = image_context(boxes, rarity, bool(buildings) and item.get("building_group") not in buildings)
        score = policy.score(context)
        # Quantas sugestões de cada classe (por nome), para o revisor ver na fila.
        suggested = Counter(names[box["class_id"]] for box in boxes)
        # EXTENSAO: campo novo por foto no review_priority.json: acrescente aqui.
        rows.append({"image_sha256": item["image_sha256"], "filename": item.get("filename"),
                     "building_group": item.get("building_group"), "suggestions": len(boxes),
                     "suggested_classes": dict(suggested), **{key: round(value, 4) for key, value in score.items()},
                     "context": dict(zip(FEATURES, context.round(4).tolist()))})
    # 3. Maior pontuação primeiro; "priority" = posição 1, 2, 3...
    rows.sort(key=lambda row: -row["score"])
    for position, row in enumerate(rows, start=1):
        row["priority"] = position
    # 4. Grava o JSON completo ao lado do registro (com hashes para auditoria).
    output = Path(registry_path).resolve().parent / "review_priority.json"
    write_json(output, {
        "schema_version": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "registry_sha256": file_hash(Path(registry_path)), "suggestions_weights_sha256": suggestions.get("weights_sha256"),
        "policy": str(policy_path) if policy_path else None, "policy_updates": len(policy.history),
        "alpha": policy.alpha, "theta": dict(zip(FEATURES, policy.theta.round(4).tolist())),
        "note": "Ordem sugerida de revisão. Não altera status, caixas nem aprovações.", "images": rows,
    })
    # 5. Grava o CSV (mesmo nome, extensão .csv) para abrir em planilha.
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        # EXTENSAO: coluna nova no CSV: acrescente o título aqui E o valor na mesma
        # posição do writer.writerow do laço logo abaixo.
        writer.writerow(["prioridade", "arquivo", "edificacao", "sugestoes", "pontuacao", "aproveitamento", "exploracao",
                         "classes_sugeridas"])
        for row in rows:
            # Última coluna: "classe=quantidade; classe=quantidade".
            writer.writerow([row["priority"], row["filename"], row["building_group"], row["suggestions"],
                             row["score"], row["exploit"], row["explore"],
                             "; ".join(f"{name}={count}" for name, count in row["suggested_classes"].items())])
    return output


# ENTRADA: comando ``learn-review`` (cli.py chama esta função).
def update_policy(config: ProjectConfig, reviewed_registry: Path, policy_path: Path,
                  pilot_manifest: Path | None = None, suggestions_path: Path | None = None,
                  alpha: float | None = None) -> dict:
    """Aprende com as decisões humanas: uma atualização LinUCB por foto aprovada que tinha sugestões.

    ``reviewed_registry`` é o registro já com o feedback importado (``import-review``);
    ``suggestions_path`` é o suggestions.json que o revisor viu (padrão: ao lado do registro).

    Parâmetros:
        config: configuração do projeto.
        reviewed_registry: registry.json com ``human_approved`` e ``boxes`` finais.
        policy_path: policy.json a ler (se existir) e a regravar.
        pilot_manifest: pilot.json do treino (raridade/edificações); deve ser o
            mesmo usado no prioritize-review para o contexto ser o mesmo.
        suggestions_path: suggestions.json visto pelo revisor.
        alpha: α a gravar na política; None mantém o salvo.

    Retorna:
        ``{"policy": caminho, "new_updates": int, "total_updates": int,
        "mean_reward": float | None, "theta": {característica: peso}}``.

    Erros:
        ValueError de ``_load_review`` e ``LinUCBPolicy.load``.
        ATENCAO: regrava policy_path no lugar (é o estado acumulado da política).
    """
    # 1. Mesmo preparo de prioritize_review (o contexto precisa ser calculado igual).
    names = detection_names(load_taxonomy(config.taxonomy_path))
    registry, suggestions = _load_review(reviewed_registry, suggestions_path)
    policy = LinUCBPolicy.load(policy_path, alpha)
    counts, buildings = training_context(pilot_manifest, len(names))
    rarity = class_rarity(counts, len(names))
    # 2. Pares (foto, modelo das sugestões) já contados: não somar a mesma decisão duas vezes.
    counted = {(entry["image_sha256"], entry["suggestions_weights_sha256"]) for entry in policy.history}
    rewards = []
    # 3. Uma atualização por foto aprovada ainda não contada.
    for item in registry["images"]:
        key = (item["image_sha256"], suggestions.get("weights_sha256"))
        # Só decisões humanas aprovadas ensinam algo (pendentes não têm verdade).
        # ATENCAO: o docstring fala em "foto que tinha sugestões", mas não há filtro
        # para foto sem sugestão: ela também atualiza (contexto no_suggestions = 1).
        if item.get("human_approved") is not True or key in counted:
            continue
        boxes = suggestions["images"].get(item["image_sha256"], [])
        context = image_context(boxes, rarity, bool(buildings) and item.get("building_group") not in buildings)
        # Recompensa: quanto o modelo errou nessa foto + bônus de classe rara.
        reward = review_reward(boxes, item["boxes"], rarity)
        policy.update(context, reward["reward"])
        # Entrada do histórico (auditoria e controle de duplicidade).
        entry = {"image_sha256": item["image_sha256"], "suggestions_weights_sha256": key[1],
                 "registry_sha256": file_hash(Path(reviewed_registry)), **reward}
        policy.history.append(entry)
        rewards.append(entry)
    # 4. Regrava a política e devolve o resumo que a CLI imprime.
    write_json(Path(policy_path), policy.to_json())
    return {"policy": str(Path(policy_path).resolve()), "new_updates": len(rewards), "total_updates": len(policy.history),
            "mean_reward": round(float(np.mean([entry["reward"] for entry in rewards])), 4) if rewards else None,
            "theta": dict(zip(FEATURES, policy.theta.round(4).tolist()))}
