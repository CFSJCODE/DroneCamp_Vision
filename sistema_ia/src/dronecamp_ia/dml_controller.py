"""Controlador por reforço do peso de consistência no aprendizado mútuo (DML).

Função no projeto: no aprendizado mútuo (``mutual_learning.py``) cada detector
soma à própria perda um termo que o aproxima das previsões dos outros dois:

    L_i = L_detecção,i + λ(t) · a_i · L_consistência,i

O risco conhecido do DML é um par fraco puxar um modelo bom para baixo. Em vez
de fixar ``a_i`` à mão, este módulo trata a escolha de ``a_i`` a cada época como
um problema de reforço de um passo (bandit) por modelo:

- Ação: um multiplicador ``a`` de uma lista discreta (``DEFAULT_ARMS``). O braço
  ``0.0`` é o controle: aquela época o modelo treina só com as anotações. Assim
  o próprio treino mede se a consistência ajuda ou atrapalha.
- Recompensa: variação do mAP50 de validação do modelo depois da época,
  ``r = mAP50(t) − mAP50(t−1)``, limitada a [−1, 1].
- Política: UCB descontado (Garivier & Moulines, ALT 2011). O ambiente não é
  estacionário (o que ajuda na época 5 pode atrapalhar na 50), então as
  contagens e somas antigas decaem por ``γ`` a cada passo:

      N_a ← γ·N_a + 1[a escolhido]        S_a ← γ·S_a + r·1[a escolhido]
      escolha = argmax_a  S_a/N_a + c·√(ξ·ln(Σ N) / N_a)

  Braço nunca tentado (``N_a`` ≈ 0) é escolhido primeiro, na ordem da lista.

Por que bandit e não RL profundo (PPO, DQN): há uma decisão e uma recompensa
por época e por modelo, umas dezenas no treino inteiro. Uma rede de política
precisaria de milhares de transições; o UCB descontado funciona com dezenas e
é auditável (o JSON guarda cada escolha e cada recompensa).

Limites honestos:
- A validação vem de uma edificação (CEASA); a recompensa é ruidosa. O
  controlador escolhe pesos de treino, não adota modelo. Adoção continua sendo
  ``compare-models``/``dml-evaluate`` no teste.
- Com poucas épocas o controlador quase só explora; a conclusão útil é a
  recompensa média por braço (``summary``), lida junto com o relatório final.

Quem usa: ``mutual_learning.train_mutual`` (um controlador por modelo).

Conteúdo do arquivo:
- ``DEFAULT_ARMS``, ``DEFAULT_GAMMA``, ``DEFAULT_EXPLORATION``: padrões.
- ``DiscountedUCB``: estado, ``select``, ``update``, ``summary``, ``to_dict``/``from_dict``.
- ``reward_from_metrics``: recompensa a partir do mAP50 antes/depois.

O que lê e grava no disco: nada diretamente; ``train_mutual`` grava o estado de
todos os controladores em ``runs/dml_*/controller.json``.

Onde mexer:
- Outros multiplicadores: ``DEFAULT_ARMS`` (ou ``--rl-arms`` no ``dml-train``).
- Esquecer mais rápido o passado: diminuir ``DEFAULT_GAMMA`` (0,9 ≈ memória de 10 épocas).
- Outra recompensa (ex.: recall, menos falsos positivos): ``reward_from_metrics``.
"""

from __future__ import annotations

import math

# Multiplicadores do λ que o controlador pode escolher. 0.0 = época sem
# consistência (controle). A ORDEM importa: braços nunca tentados são escolhidos
# nesta ordem, então a 1.ª época com consistência usa 0,5 e a 2.ª mede o controle.
# EXTENSAO: novos valores entram aqui ou por --rl-arms.
DEFAULT_ARMS = (0.5, 0.0, 1.0, 0.25)
# Fator de esquecimento por passo (adimensional, 0 < γ ≤ 1). 1.0 = UCB clássico
# (nunca esquece); 0.9 dá peso ~1/e às recompensas de 10 épocas atrás.
DEFAULT_GAMMA = 0.9
# Peso c do termo de exploração. Recompensas são Δ mAP50 (tipicamente ±0,05),
# então c pequeno; c = 0 vira guloso puro.
DEFAULT_EXPLORATION = 0.05
# ξ do UCB descontado (Garivier & Moulines usam ξ ∈ (1/2, 1]; 0,6 é o valor do artigo).
XI = 0.6
# Contagem abaixo da qual o braço é considerado "nunca tentado".
_UNTRIED = 1e-9


class DiscountedUCB:
    """Bandit UCB descontado sobre multiplicadores do peso de consistência.

    Parâmetros:
        arms: valores possíveis de ``a`` (lista de floats, sem repetição).
        gamma: fator de esquecimento por passo, 0 < γ ≤ 1.
        exploration: peso ``c`` do bônus de exploração (>= 0).

    Erros:
        ValueError: lista de braços vazia ou repetida, ``gamma`` fora de (0, 1]
        ou ``exploration`` negativo.

    Chamada por: ``mutual_learning.train_mutual``.
    """

    def __init__(self, arms=DEFAULT_ARMS, gamma: float = DEFAULT_GAMMA, exploration: float = DEFAULT_EXPLORATION):
        arms = [float(value) for value in arms]
        if not arms or len(set(arms)) != len(arms):
            raise ValueError("Informe ao menos um multiplicador, sem repetição.")
        if not 0 < gamma <= 1:
            raise ValueError("gamma precisa estar em (0, 1].")
        if exploration < 0:
            raise ValueError("exploration não pode ser negativo.")
        self.arms = arms
        self.gamma = float(gamma)
        self.exploration = float(exploration)
        # N e S descontados por braço (mesma ordem de self.arms).
        self.counts = [0.0] * len(arms)
        self.sums = [0.0] * len(arms)
        # Histórico auditável: uma entrada por update.
        self.history: list[dict] = []

    def select(self) -> int:
        """Índice do braço a usar na próxima época.

        Retorna:
            Posição em ``self.arms``. Primeiro cada braço nunca tentado, na ordem
            da lista; depois o maior índice UCB (empate: o de menor posição).

        Chamada por: ``mutual_learning.train_mutual`` no começo de cada época.
        """
        # 1. Exploração inicial: braço sem contagem vai primeiro.
        for index, count in enumerate(self.counts):
            if count < _UNTRIED:
                return index
        # 2. UCB descontado: média descontada + bônus que cresce para braços pouco usados.
        total = sum(self.counts)
        best, best_score = 0, -math.inf
        for index, (count, value) in enumerate(zip(self.counts, self.sums)):
            score = value / count + self.exploration * math.sqrt(XI * math.log(max(total, 1.0 + 1e-9)) / count)
            if score > best_score + 1e-12:
                best, best_score = index, score
        return best

    def update(self, arm: int, reward: float, step: int | None = None) -> None:
        """Registra a recompensa observada depois de usar ``arm``.

        Parâmetros:
            arm: posição do braço usado (o que ``select`` devolveu).
            reward: recompensa (já limitada por ``reward_from_metrics``).
            step: época/rodada, só para o histórico.

        Erros:
            IndexError: ``arm`` fora da lista.

        Chamada por: ``mutual_learning.train_mutual`` depois da validação da época.
        """
        if not 0 <= arm < len(self.arms):
            raise IndexError(f"Braço {arm} fora de 0..{len(self.arms) - 1}.")
        # ATENCAO: o desconto vale para TODOS os braços antes de somar o atual;
        # é isso que faz o controlador esquecer o que deixou de ser verdade.
        self.counts = [count * self.gamma for count in self.counts]
        self.sums = [value * self.gamma for value in self.sums]
        self.counts[arm] += 1.0
        self.sums[arm] += float(reward)
        self.history.append({"step": step, "arm": arm, "multiplier": self.arms[arm], "reward": round(float(reward), 6)})

    def summary(self) -> dict:
        """Resumo por braço: vezes usado, recompensa média (sem desconto) e média descontada.

        Retorna:
            ``{"<multiplicador>": {"uses", "mean_reward", "discounted_mean"}}``;
            médias ``None`` quando o braço nunca foi usado.

        Chamada por: ``mutual_learning.train_mutual`` (relatório final).
        """
        result = {}
        for index, arm in enumerate(self.arms):
            rewards = [item["reward"] for item in self.history if item["arm"] == index]
            result[str(arm)] = {
                "uses": len(rewards),
                "mean_reward": round(sum(rewards) / len(rewards), 6) if rewards else None,
                "discounted_mean": round(self.sums[index] / self.counts[index], 6) if self.counts[index] > _UNTRIED else None,
            }
        return result

    def to_dict(self) -> dict:
        """Estado completo em dict serializável (para ``controller.json``)."""
        return {"arms": self.arms, "gamma": self.gamma, "exploration": self.exploration,
                "counts": self.counts, "sums": self.sums, "history": self.history}

    @classmethod
    def from_dict(cls, data: dict) -> "DiscountedUCB":
        """Recria o controlador gravado por ``to_dict`` (retomar treino).

        Erros:
            KeyError/ValueError: JSON incompleto ou inválido.
        """
        policy = cls(data["arms"], data["gamma"], data["exploration"])
        if len(data["counts"]) != len(policy.arms) or len(data["sums"]) != len(policy.arms):
            raise ValueError("controller.json inconsistente: contagens não batem com os braços.")
        policy.counts = [float(value) for value in data["counts"]]
        policy.sums = [float(value) for value in data["sums"]]
        policy.history = list(data.get("history", []))
        return policy


def reward_from_metrics(before: float | None, after: float | None) -> float:
    """Recompensa de uma época: Δ mAP50 de validação, limitada a [−1, 1].

    Parâmetros:
        before: mAP50 de validação antes da época (``None`` = ainda não medido).
        after: mAP50 depois da época.

    Retorna:
        ``after − before`` limitado; 0.0 quando falta uma das medidas (a época
        não conta a favor nem contra nenhum braço).

    Chamada por: ``mutual_learning.train_mutual``.
    """
    # EXTENSAO: recompensa composta (ex.: Δ recall − 0,5·Δ falsos positivos por
    # foto) entra aqui; mantenha o limite para um valor extremo não dominar o UCB.
    if before is None or after is None:
        return 0.0
    return max(-1.0, min(1.0, float(after) - float(before)))
