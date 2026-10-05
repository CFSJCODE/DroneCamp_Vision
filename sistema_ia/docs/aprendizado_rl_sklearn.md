# Aprendizado por reforço, scikit-learn e PyTorch no ciclo do YOLO — 05/10/2026

Origem: pedido do Cláudio para aplicar técnicas de *reinforcement learning*, scikit-learn, PyTorch e TensorFlow nos modelos e nos treinos reais com YOLO, para melhorar a detecção, o reconhecimento e a marcação de não conformidades de forma autônoma.

## Resumo técnico

O gargalo do DroneCamp não é poder de cálculo: são **37 fotos aprovadas de uma única edificação**, com `fixador_telha_frouxo` (1 caixa), `rufo_deslocado` (1) e `reparo_rufo` (0). Técnicas que precisam de milhares de exemplos (RL profundo, redes extras) só decorariam esses dados. Por isso cada técnica foi colocada onde ela ganha com poucos dados:

| Técnica | Onde entra | Módulo | Comando |
| --- | --- | --- | --- |
| RL (bandit contextual LinUCB) | Ordem da fila de revisão: qual foto o revisor vê primeiro | `active_learning.py` | `prioritize-review`, `learn-review` |
| RL (bandit successive halving) | Busca de hiperparâmetros do treino piloto com orçamento de épocas | `hparam_bandit.py` | `tune-pilot-bandit` |
| scikit-learn (regressão logística + `GroupKFold`) | Probabilidade de o revisor aceitar cada sugestão | `calibration.py` | `fit-calibrator`, `suggest --calibrator` |
| PyTorch (via Ultralytics) + reamostragem RFS | Treino real com fotos de classes raras repetidas | `sampling.py`, `training.py` | `train-pilot --repeat-factor-threshold` |
| Avaliação por classe | Adotar um modelo só com ganho medido e sem regressão | `model_gate.py`, `matching.py` | `compare-models` |

"Autônomo" aqui significa **a IA propõe, ordena e estima**, nunca aprova. Nenhum desses comandos altera status, caixas ou aprovações humanas, nem troca o modelo em uso; a regra do projeto (sugestões pendentes até decisão humana) continua valendo.

## 1. Fila de revisão por aprendizado por reforço (LinUCB)

**Modelagem.** Cada foto pendente é um contexto `x` com 10 características calculadas das sugestões do modelo: número de sugestões, confiança média e máxima, incerteza média `1 − |2c − 1|`, raridade da classe sugerida `1/√(1 + caixas de treino)`, conflito entre classes (caixas de classes diferentes com IoU > 0,5), edificação nova, ausência de sugestões e incerteza calibrada. A ação é escolher a próxima foto. Depois que o humano revisa, a recompensa é

```
r = (FP + FN) / (TP + FP + FN + 1)  +  0,5 · raridade máxima das caixas humanas
```

ou seja: foto em que o modelo errou ou que tem classe rara ensina mais ao próximo treino.

**Política.** LinUCB com vetor compartilhado:

```
A = λI + Σ x xᵀ        b = A θ₀ + Σ r x        θ = A⁻¹ b
pontuação(x) = θᵀx + α √(xᵀ A⁻¹ x)
```

O primeiro termo aproveita o que já foi aprendido; o segundo explora contextos pouco vistos. `θ₀` (`PRIOR_WEIGHTS`) é a priorização inicial explícita, usada enquanto não há recompensas. O estado (`A`, `b` e o histórico das fotos já contadas) fica em [`data/learning/review_policy.json`](../data/learning/review_policy.json); a mesma decisão humana nunca é contada duas vezes.

**Por que bandit e não RL profundo:** cada revisão gera uma recompensa e nenhuma transição de estado relevante. Um bandit linear aprende com dezenas de revisões; uma rede de política (PPO, DQN) precisaria de milhares.

**Execução real (05/10).** A política aprendeu com as **37 decisões humanas da v8** comparadas às sugestões que o revisor viu na v7 (pesos `dd284411`), com a raridade do treino v7:

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia learn-review `
  --registry data\reviews\ceasa_v8_revisao002_ff34e226416c\registry.json `
  --suggestions data\reviews\ceasa_v7_revisao002_ba8cc323c8a1\suggestions.json `
  --pilot-manifest data\pilot\ceasa_v7_piloto_s42\pilot.json `
  --policy data\learning\review_policy.json
```

Recompensa média 0,67. Pesos aprendidos: incerteza +0,49, ausência de sugestões +0,33, raridade +0,32, conflito entre classes +0,29; confiança máxima −0,09. Em seguida, a fila das 7 fotos de outras edificações:

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia prioritize-review `
  --registry data\reviews\internet_v1_sugestoes_modelo\registry.json `
  --policy data\learning\review_policy.json `
  --pilot-manifest data\pilot\ceasa_v8_piloto_s42\pilot.json
```

Resultado em [`review_priority.csv`](../data/reviews/internet_v1_sugestoes_modelo/review_priority.csv): primeiro as duas fotos em que o modelo não sugeriu nada (`detritos-calha`, `resto-tinta-sobre-telhado`), depois as de uma sugestão isolada; a foto com 13 sugestões confiantes de `residuos_telha` fica em 5º.

**Limite:** o modelo v7 já tinha visto 23 das 37 fotos no treino, então parte das recompensas reflete memorização. A política fica mais útil a cada campanha nova revisada; rode `learn-review` depois de cada `import-review`.

## 2. Calibração das sugestões com scikit-learn

**Problema.** A confiança do YOLO piloto não é probabilidade: 30% de `residuos_telha` e 30% de `fixador_telha_frouxo` não acertam na mesma taxa.

**Modelagem.** Cada sugestão numa foto com caixas humanas é um exemplo; `y = 1` se ela reencontra caixa humana (mesma classe, IoU ≥ 0,5). Características: confiança e logit, classe (one-hot), área e proporção da caixa, posição, posição no ranking, sugestões na foto e sobreposição com outras sugestões. Modelo: `StandardScaler` + `LogisticRegression(C=0,5)`, deliberadamente simples. Avaliação por `GroupKFold` agrupado por cena, comparando Brier, AUC-ROC e precisão média contra a confiança crua. Fotos do treino do detector ficam de fora (nelas a confiança parece melhor do que é).

**Execução real (05/10)** com os pesos piloto v7 no dataset v8:

| | Brier ↓ | AUC-ROC ↑ | Precisão média ↑ |
| --- | ---: | ---: | ---: |
| Confiança crua do YOLO | **0,022** | **0,799** | **0,353** |
| Calibrador (validação cruzada) | 0,025 | 0,743 | 0,295 |

375 sugestões em 14 fotos fora do treino, **só 10 acertos**. O calibrador **não superou** a confiança crua, e o código registra `improves_over_raw_confidence: false`. Por isso `suggest --calibrator` recusa esse calibrador (seria preciso `--allow-unproven-calibrator`, só para testes). É o resultado esperado com 10 acertos; o mecanismo está pronto para quando houver revisões de outras edificações. Calibrador gerado em [`runs/calibrator_20261005T143055Z_4758c5f5`](../runs/calibrator_20261005T143055Z_4758c5f5/calibrator.json).

Quando aprovado, cada sugestão ganha `acceptance_probability`, a lista é ordenada por ela e a página mostra "aceitação estimada N%" ao lado da confiança. Nenhuma sugestão é apagada.

Segurança: o calibrador é salvo com `joblib` (pickle). Só carregue arquivos gerados por `fit-calibrator`; o SHA-256 registrado em `calibrator.json` é conferido antes de abrir.

## 3. Treino real com classes raras repetidas (PyTorch/Ultralytics + RFS)

O treino continua sendo PyTorch pela Ultralytics. O que mudou é **o que o treino vê**: a reamostragem por fator de repetição (Gupta et al., LVIS 2019) repete na lista de treino as fotos com classes raras:

```
f_c = fração das fotos de treino com a classe c
r_c = max(1, √(t / f_c))        r_foto = max(r_c das classes da foto)
```

No dataset v8 com `t = 0,3`, as fotos de `fixador_telha_frouxo`, `rufo_deslocado` e `telha_ausente` (1 foto cada em 22) passam a aparecer cerca de 2,6 vezes por época (fator 2,57); a época vai de 22 para 33 entradas, com 7 fotos repetidas 2 vezes e 2 repetidas 3 vezes. Validação e teste não mudam. A lista vai para `runs/pilot_train_*/train_repeat_factor.txt` e o resumo para `sampling.json`; o dataset imutável não é alterado.

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia train-pilot --data data\pilot\ceasa_v8_piloto_s42\dataset.yaml --repeat-factor-threshold 0.3
```

Ou deixe fixo em `configs/project.yaml` (`pilot.sampling.repeat_factor_threshold`). **Limite:** repetir a mesma foto não cria variedade nova; com uma caixa a classe continua sem ser aprendida. Está desligado por padrão até um treino completo mostrar ganho no `compare-models`.

## 4. Hiperparâmetros por bandit (successive halving)

Cada combinação de `lr0`, `mosaic` e `repeat_factor_threshold` é um braço. Rodada 0: todos os braços com `e` épocas; rodada k: os melhores `⌈n/ηᵏ⌉` com `e·ηᵏ` épocas. Recompensa: melhor mAP50 de validação do `results.csv` do treino real. O teste nunca é usado; o vencedor é uma recomendação gravada em `runs/bandit_*/bandit.json`, e o `project.yaml` não é alterado.

Orçamento no Ryzen 5 4600G com o espaço padrão (12 braços, `--min-epochs 5`, `--eta 3`): 12×5 + 4×15 + 2×45 = 210 épocas, cerca de 2,7 treinos piloto completos (≈ 3 h). Para uma noite curta, sorteie menos braços:

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia tune-pilot-bandit --data data\pilot\ceasa_v8_piloto_s42\dataset.yaml --arms 6 --min-epochs 5 --eta 3
```

**Limite:** a validação tem 6 fotos de uma edificação; a recompensa é ruidosa. Confirme o vencedor com `compare-models` antes de adotar.

## 5. Adoção só com ganho medido por classe

`compare-models` roda o modelo atual e o candidato nas mesmas fotos, **excluindo as que estiveram no treino de qualquer um dos dois**, e grava em `runs/gate_*/gate.json`, por classe: TP, FP, omissões (FN), precisão, recall e AP50. Regra de adoção (todas obrigatórias): mAP50 maior em pelo menos 0,01; recall total não menor; nenhuma classe com 3 ou mais caixas humanas perdendo mais de 0,10 de recall. `production_ready` continua falso enquanto não houver edificações independentes e revisão técnica. O modelo anterior nunca é apagado.

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia compare-models --data data\pilot\ceasa_v8_piloto_s42\dataset.yaml `
  --baseline runs\pilot_train_20261004T170212Z_dd284411\fit\weights\best.pt --candidate CAMINHO\best.pt --split test
```

## Por que não TensorFlow

Foi avaliado e **não foi incluído**, porque não traria ganho concreto aqui:

- YOLO26 e toda a Ultralytics são PyTorch; não há YOLO26 em TensorFlow. Treinar o detector em TF significaria reescrever o modelo e perder os pesos pré-treinados.
- No Windows, o TensorFlow não usa GPU nativa desde a versão 2.10; na sua APU Radeon ele rodaria só na CPU, como o PyTorch.
- A aceleração da APU já existe pelo ONNX com `onnxruntime-directml` (124 ms/foto contra 372 ms na CPU), que é independente de framework.
- Somaria cerca de 500 MB de dependências e uma segunda pilha de tensores para manter, sem nenhuma métrica a ganhar.

Se um dia houver um modelo que só exista em TF (por exemplo, um classificador de terceiros), ele pode entrar exportado para ONNX, sem trazer o TensorFlow para o treino.

## Instalação

```powershell
uv pip install --python .venv\Scripts\python.exe -e ".[documents,learning]"
```

O extra `learning` instala scikit-learn (com joblib). Os comandos de RL usam só NumPy, que já vem com a Ultralytics.

## Validação desta entrega

Ver a seção correspondente em [validacao.md](validacao.md).

## Dependências para o próximo avanço

- **Fotos de outras edificações revisadas.** Sem elas, o calibrador não tem acertos suficientes (10 hoje) e nenhuma métrica mede generalização.
- **Exemplos de `fixador_telha_frouxo`, `rufo_deslocado` e `reparo_rufo`.** RFS e bandit não criam informação que não está nas fotos.
- **Um treino completo com `--repeat-factor-threshold 0.3` no PC**, comparado ao piloto com `compare-models`, para decidir se a RFS fica ligada por padrão.
