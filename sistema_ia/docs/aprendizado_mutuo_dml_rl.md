# Aprendizado mútuo YOLO26 + RF-DETR + D-FINE, com RL e rede local — 16/10/2026

Origem: pedido do Cláudio (16/10) para, depois do treino `614d070a`, reformular e implementar Deep Mutual Learning (DML) com YOLO26, RF-DETR e D-FINE treinando juntos, implementar Reinforcement Learning, e (mensagem seguinte) distribuir o treino por máquinas da rede local, testando primeiro no HP-Z230 e no 6006U.

Código: `detector_adapters.py`, `mutual_learning.py`, `dml_controller.py`, `dml_lan.py`. Configuração: `configs/dml.json` (treino real) e `configs/dml_smoke.json` (teste de fumaça). Testes: `tests/test_mutual_learning.py`.

## 1. Resumo técnico

- **O que muda em relação à proposta original.** A perda conceitual `L_i = L_detecção,i + λ·L_consistência,i` foi mantida e especificada: como alinhar previsões de arquiteturas diferentes (Hungarian matching), como compatibilizar classes (sigmoide por classe nas três cabeças → KL de Bernoulli), como filtrar previsões pouco confiáveis (τ + fusão por concordância + ancoragem nas caixas humanas) e como deixar λ comparável entre perdas de escalas muito diferentes (normalização por média móvel).
- **Três modos de treino**, do mais caro ao mais barato: `online` (os três na memória, DML original), `distill` (um aluno, professores congelados: é a "destilação do ensemble num modelo só") e `cached` (DML em rodadas, com as previsões dos pares gravadas em disco). O `cached` é o que cabe na CPU do 4600G e é o que a rede local distribui.
- **RL**: o peso da consistência de cada modelo, por época (ou rodada), é escolhido por um bandit UCB descontado. Um dos braços é λ = 0, então o próprio treino mede se a consistência ajuda ou atrapalha cada modelo.
- **Rede local**: cada máquina treina um modelo; a cada rodada trocam só previsões (~3 MB por modelo), não gradientes (~236 MB por passo no yolo26x).
- **Estado**: implementado e testado na nuvem com os três modelos reais (pesos aleatórios e pesos COCO do YOLO26n e do RF-DETR Nano). **Nenhum ganho foi demonstrado ainda**: isso depende do treino nas fotos da v4_rev002 e do `dml-evaluate` no teste contra o modelo atual. O código respeita a regra do projeto: pesos novos são candidatos, a adoção exige melhoria por classe.

## 2. Modelagem do problema

Para cada modelo `i ∈ {YOLO26, RF-DETR, D-FINE}`, imagem `x` e caixas humanas `G`:

```
L_i = L_det,i(f_i(x), G)  +  λ(t) · a_i(t) · s_i · L_cons,i
```

| Termo | Definição | Por quê |
| --- | --- | --- |
| `L_det,i` | Perda oficial da biblioteca: E2E da Ultralytics (one2many + one2one), `SetCriterion` do RF-DETR (focal + L1 + GIoU, com camadas auxiliares e encoder), perda do D-FINE no `transformers` (VFL + FGL + DDF) | Cada modelo continua aprendendo do jeito que foi projetado |
| `λ(t)` | `λ_max · exp(−5(1 − t/T)²)` nas `ramp_epochs` primeiras épocas | No começo os pares erram muito; ensinar com eles cedo atrapalha (Laine & Aila 2017) |
| `a_i(t)` | Multiplicador escolhido pelo RL em `{0,5; 0; 1; 0,25}` | Ver seção 5 |
| `s_i` | `EMA(L_det,i) / EMA(L_cons,i)`, sem gradiente | As perdas têm escalas de 10 (RF-DETR) a 1000 (D-FINE no começo); com `s_i`, λ = 0,5 quer dizer "a consistência pesa metade da detecção" nos três |

**Consistência para o modelo i**, por imagem:

1. **Pares sem gradiente.** Para cada par `j ≠ i`, pega as previsões da cabeça de inferência (one2one no YOLO26, as 300 queries do 1.º grupo no RF-DETR, as 300 queries sem denoising no D-FINE), com confiança `max_c p_c ≥ τ`, e remove duplicatas por NMS sem classe.
2. **Fusão por concordância (estilo WBF).** Previsões de pares diferentes com IoU ≥ 0,55 viram um alvo: caixa = média ponderada pela confiança; probabilidade por classe `p̄_c = Σ_j p_{j,c} / (N − 1)` (par que não viu conta 0); peso `w = confiança média × (pares que concordam / pares)`. Alvo que só um par viu pesa menos.
3. **Ancoragem nas caixas humanas** (`anchor: "gt"`, padrão). Ficam só os alvos com IoU ≥ 0,5 com alguma caixa humana. Nas fotos aprovadas, tudo que não é caixa humana é falso positivo por definição; o que vale aprender dos pares é a **distribuição entre classes** (o "conhecimento escuro" do DML: "parece objeto solto, mas lembra pedaço de telha") e o **ajuste fino da caixa**, não detecções extras. `anchor: "all"` existe para destilar em fotos sem revisão, e deve ser usado só com medição no teste.
4. **Alinhamento por Hungarian matching** entre todas as queries/âncoras do modelo i e os alvos, custo `BCE(classe) + 5·L1 + 2·(1 − GIoU)` (o matcher do DETR). Pares com sobreposição desprezível são descartados.
5. **Perda nos pares casados:**

```
L_cons,i = Σ_t w_t · [ Σ_c KL( p̄_{t,c} ‖ σ(z_{q,c}/T) ) · T²  +  β · ( ‖b_q − b̄_t‖₁ + 1 − GIoU(b_q, b̄_t) ) ] / Σ_t w_t
```

O KL de Bernoulli por classe é o análogo do KL entre softmax do DML de classificação (Zhang et al., CVPR 2018), porque as três cabeças usam **sigmoide por classe**, não softmax. Ele tem o mesmo gradiente da BCE com alvo suave, mas vale 0 quando o aluno já concorda.

**O que não entra, e por quê.** A distribuição de bordas do D-FINE (FDR) não é destilada para os outros: YOLO26 (reg_max = 1, sem DFL) e RF-DETR não têm essa distribuição, então a consistência de caixa é só no espaço das caixas. O termo negativo (punir o aluno por achar o que os pares não acharam) existe (`negative_weight`) mas fica em 0: com um par fraco ele apaga acertos.

## 3. Arquitetura do sistema

```
dataset.yaml (recortes 640 da v4_rev002, classes v7)
        │  TileDataset: mesma imagem e MESMO espelho para os três
        ▼
┌──────────────┬──────────────┬──────────────┐
│ YOLO26Adapter│ RFDETRAdapter│ DFINEAdapter │  detector_adapters.py
│ Ultralytics  │ rfdetr       │ transformers │  → RawPredictions (cxcywh 0-1, logits [B,Q,K])
└──────┬───────┴──────┬───────┴──────┬───────┘
       └── pares (sem gradiente) ────┘
                      ▼
   fuse_peer_predictions → anchor_to_truth → Hungarian → consistency_loss   mutual_learning.py
                      ▼
   L_det + λ·a·s·L_cons  → AdamW por modelo (lr e corte de gradiente por família)
                      ▼
   validação (mesma régua do compare-models: IoU 0,5, AP por classe) → recompensa → DiscountedUCB   dml_controller.py
```

| Família | Pacote | Entrada | Pesos salvos | Licença |
| --- | --- | --- | --- | --- |
| YOLO26 | `ultralytics==8.4.172` | 0-1 | `.pt` que `YOLO()` abre: funciona no `compare-models`, `suggest`, `export` | AGPL-3.0 |
| RF-DETR (nano…large) | `rfdetr` 1.11 | ImageNet | `.pth` (`model` + `model_config`) | Apache-2.0 (o XLarge é de outra licença; não usado) |
| D-FINE (s/m/l/x COCO) | `transformers` 5.x | 0-1 | pasta `save_pretrained` | Apache-2.0 |

Instalação no PC: `.\.venv\Scripts\python.exe -m pip install -e ".[dml]"`. Na 1.ª execução cada biblioteca baixa os pesos COCO (YOLO26 do GitHub da Ultralytics, RF-DETR do servidor da Roboflow, D-FINE do Hugging Face).

## 4. Pipeline de dados

- Entrada: o mesmo `dataset.yaml` que o `train-pilot --tile` gera (recortes de 640 em escala 1:1, splits por sequência de voo). Listas `.txt` (ex.: o arquivo do RFS) também funcionam.
- Aumento: só espelho horizontal (`hflip: 0.5`), igual nos três modelos; o espelho vertical fica desligado como no `project.yaml`. O laço próprio **não tem mosaico, EMA nem o MuSGD** do treino oficial da Ultralytics; por isso o controle justo do DML é o braço λ = 0 deste mesmo laço (ou a rodada 0 da rede local), não o `614d070a`.
- O split `test` nunca é usado no treino nem na escolha do `best` (que é o mAP50 de val).
- Cache dos pares (`dml-cache`): `.npz` sem pickle com até 100 previsões por recorte em float16 (≈ 3 MB para 1.234 recortes e 9 classes). Ele é espelhado junto com o lote.

## 5. Reinforcement Learning: onde entra e por que assim

O projeto já tinha dois bandits (PR #5): LinUCB para a fila de revisão e successive halving para hiperparâmetros. O novo ponto de decisão é o **peso da consistência de cada modelo ao longo do treino**:

- Ação: `a ∈ {0,5; 0; 1; 0,25}` (ordem de exploração: a 1.ª época com consistência usa 0,5, a 2.ª mede o controle 0).
- Recompensa: `Δ mAP50` de validação do modelo depois da época/rodada, limitada a [−1, 1].
- Política: **UCB descontado** (Garivier & Moulines 2011): `N_a ← γN_a + 1`, `S_a ← γS_a + r`, escolha `argmax S_a/N_a + c·√(ξ ln ΣN / N_a)`, com γ = 0,9. O desconto é necessário porque o problema muda durante o treino: o que ajuda na época 5 pode atrapalhar na 50.
- Por que não PPO/DQN: há uma decisão e uma recompensa por época e por modelo, dezenas no treino inteiro. Uma rede de política precisaria de milhares de transições.
- Leitura do resultado: `summary.json → models.<modelo>.rl` dá a recompensa média por braço. Se o braço 0 tiver a maior média num modelo, a consistência não está ajudando aquele modelo.
- Limite: a validação é de uma edificação e tem poucas fotos; a recompensa é ruidosa. O controlador escolhe pesos de treino, nunca adota modelo.

## 6. Estratégia de treino recomendada (em ordem)

1. **Esperar o `614d070a` terminar** (a thread do PC é dona dele). Depois, `yolo26.weights` no `configs/dml.json` passa a ser o `best.pt` dele (mesmas classes v7).
2. **Medir o tempo antes de decidir onde treinar**: `dml-plan` em cada máquina (seção 8). Na CPU do 4600G, o yolo26x sozinho leva 58 min a 1h40 por época; os três juntos no modo `online` devem levar várias vezes isso.
3. **Rodada 0 = os três sozinhos** (linha de base de cada um) e **`dml-evaluate`** com a complementaridade: se o recall da união não passar bem do melhor modelo, os três erram as mesmas coisas e DML/ensemble têm pouco a ganhar.
4. **DML em rodadas** (`cached`, local ou na rede), 3–4 rodadas de 5–10 épocas.
5. **Destilar no YOLO26** (`--mode distill --student yolo26`) se o ensemble for melhor que o melhor modelo sozinho, para ficar com um modelo rápido no ONNX/DirectML da plataforma.
6. **Adoção**: `compare-models` do `.pt` do YOLO26 contra o modelo atual nas fotos inteiras com fatiamento, e `dml-evaluate --split test` para os três. Só adotar com melhoria por classe e sem regressão, mantendo o anterior.

## 7. Rede local: protocolo e teste no HP-Z230 e no 6006U

**Protocolo.** O coordenador (um processo HTTP com token) guarda o estado das rodadas em `state.json`. Rodada 0: cada trabalhador treina seu modelo sozinho, grava o cache das previsões no split de treino e envia com o mAP50 de val. Quando os três chegaram, o coordenador abre a rodada seguinte com o multiplicador escolhido pelo RL; cada trabalhador baixa os caches dos outros dois e continua dos próprios pesos (`last`). Os trabalhadores consultam a cada 20 s; se uma máquina reiniciar, o coordenador retoma com `--resume` e o trabalhador volta a pedir a tarefa. Nenhum peso de modelo trafega; só JSON e `.npz` sem pickle.

**Pré-requisitos em cada máquina** (HP-Z230 192.168.0.82 e 6006U 192.168.0.15, ambos no 192.168.0.x do mapa de rede):

1. Clonar o repositório, criar o venv e instalar: `python -m pip install -e ".[dml]"`.
2. Copiar a **mesma** pasta do dataset para as duas máquinas (o coordenador recusa um dataset diferente). Para o teste de fumaça serve um recorte pequeno, como `data/datasets/e2e_test_v1`; o teste valida o fluxo, não a qualidade. O `data.yaml` desse recorte traz `path:` absoluto do 4600G: em outra máquina, apague a linha `path:` (a pasta do YAML passa a ser a raiz) ou ajuste o caminho.
3. Liberar a porta só para a rede local na máquina do coordenador (PowerShell como administrador; é uma mudança no sistema que você faz):

```powershell
New-NetFirewallRule -DisplayName "DroneCamp DML 8790" -Direction Inbound -Protocol TCP -LocalPort 8790 -RemoteAddress 192.168.0.0/24 -Action Allow
```

**Teste de fumaça** (`configs/dml_smoke.json`: YOLO26n, RF-DETR Nano, D-FINE Small, 1 época por rodada). Coordenador e dois modelos no HP-Z230, que deve ter mais núcleos; o modelo mais leve, o YOLO26n, no 6006U (i3-6006U, 2 núcleos, 7,9 GB). Medido no 6006U com `dml-plan` em 17/10 (batch 2, imgsz 640): YOLO26n 3,6 s por passo, D-FINE Small 6,9 s, RF-DETR Nano 17,8 s. O RF-DETR Nano é o mais pesado dos três, apesar do nome, e vai para a máquina mais forte.

```powershell
# HP-Z230 (192.168.0.82), terminal 1: coordenador (imprime o token)
.\.venv\Scripts\python.exe -m dronecamp_ia dml-lan-coordinator --models configs\dml_smoke.json --host 192.168.0.82 --rounds 2 --epochs-per-round 1

# HP-Z230, terminal 2: trabalhador do RF-DETR e do D-FINE
.\.venv\Scripts\python.exe -m dronecamp_ia dml-lan-worker --models configs\dml_smoke.json --data CAMINHO\dataset.yaml --model rfdetr --model dfine --coordinator http://192.168.0.82:8790 --token TOKEN --name HP-Z230 --max-batches 5

# 6006U (192.168.0.15): trabalhador do YOLO26
.\.venv\Scripts\python.exe -m dronecamp_ia dml-lan-worker --models configs\dml_smoke.json --data CAMINHO\dataset.yaml --model yolo26 --coordinator http://192.168.0.82:8790 --token TOKEN --name 6006U --max-batches 5
```

Se uma máquina reiniciar no meio: o coordenador volta com `--resume runs\dml_lan_<...>` (mesmo token) e o trabalhador com `--workdir runs\dml_lan_worker_<...>` (a pasta que ele imprimiu ao começar), porque a rodada seguinte continua dos pesos guardados nela.

Sucesso = `runs\dml_lan_*\state.json` com `"complete": true` e os três modelos em `uploads` das rodadas 0 e 1. Depois de validado, o 4600G entra como terceira máquina, e o treino real usa `configs\dml.json` sem `--max-batches`.

## 8. Comandos

```powershell
# Dependências e tempo medido por passo nesta máquina (decide onde treinar)
.\.venv\Scripts\python.exe -m dronecamp_ia dml-plan --data CAMINHO\dataset.yaml --output runs\dml_plan.json

# DML online (os três na memória) ou destilação em um aluno
.\.venv\Scripts\python.exe -m dronecamp_ia dml-train --data CAMINHO\dataset.yaml --mode online
.\.venv\Scripts\python.exe -m dronecamp_ia dml-train --data CAMINHO\dataset.yaml --mode distill --student yolo26

# DML em rodadas numa máquina só: cache de cada par, depois o aluno
.\.venv\Scripts\python.exe -m dronecamp_ia dml-cache --data CAMINHO\dataset.yaml --model rfdetr --weights runs\dml_X\weights\rfdetr\last.pth --output runs\pares_r1
.\.venv\Scripts\python.exe -m dronecamp_ia dml-train --data CAMINHO\dataset.yaml --mode cached --student yolo26 --peer-cache runs\pares_r1

# Medir os três, a complementaridade e o ensemble (weights do JSON apontando para os best)
.\.venv\Scripts\python.exe -m dronecamp_ia dml-evaluate --data CAMINHO\dataset.yaml --split test
```

Saídas: `runs/dml_*/history.jsonl` (por época e modelo: perdas, alvos casados, mAP50 de val, braço e recompensa do RL), `controller.json`, `weights/<modelo>/best*` e `last*`, `summary.json`; `runs/dml_eval_*/evaluation.json` (por classe: TP, FP, FN, precisão, recall, AP50; complementaridade; ensemble).

## 9. O que foi verificado e o que falta

Verificado na nuvem (CPU de 4 núcleos, sem GPU):

- Os três adaptadores calculam a perda oficial, devolvem previsões no formato comum com gradiente e gravam pesos que reabrem (o `.pt` do YOLO26 abre com `YOLO()`; o `.pth` do RF-DETR recarrega os 466 tensores idênticos).
- Os modos `online`, `distill`, `cached` e `solo`, o `dml-evaluate` e duas rodadas na rede local com um coordenador e dois trabalhadores (um com YOLO26 + D-FINE, outro com RF-DETR), no mesmo computador.
- 22 testes novos em `tests/test_mutual_learning.py`, incluindo a perda que puxa o aluno para os pares, a recusa de dataset diferente, de arquivo que não é `.npz` e de token errado.

Falta (depende do PC ou das outras máquinas):

- Pesos COCO do D-FINE: o Hugging Face está bloqueado na nuvem; no PC eles baixam na 1.ª execução.
- `dml-plan` em cada máquina, rodada 0 nas fotos da v4_rev002 e a comparação no teste. Só isso diz se o DML melhora alguma classe.
- O teste de fumaça no HP-Z230 e no 6006U.
