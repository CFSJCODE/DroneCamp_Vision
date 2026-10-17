# Métricas de treino e avaliação

Este documento descreve o que medir num treino e numa avaliação, como ler cada
número e de onde ele sai no código. Segue a ordem de prioridade pedida em 17/10/2026:
qualidade por classe, suporte dos dados, evolução contra o melhor checkpoint,
tempo por etapa e a ficha imutável de cada treino.

Nenhum comando daqui adota um modelo, muda severidade ou preenche diagnóstico.
Saída de IA continua como sugestão pendente.

## O que roda já e o que depende do próximo treino

| Comando | Treinos antigos (614d070a, 12a305a6) | Treinos novos |
|---|---|---|
| `training-progress RUN` | Sim. Lê só o `results.csv` e pode rodar com o treino em andamento. | Sim, e inclui também instrumentação e ficha. |
| `evaluate-detailed` | Sim, com qualquer `best.pt`/`last.pt`. | Sim, e a avaliação fica gravada na ficha. |
| `data-coverage` | Sim (fotos, objetos, janelas e repetições do treino). | Sim. |
| `run-card RUN` | Mostra `sem_ficha`. | Mostra as tentativas e confere os hashes. |
| Tempo por etapa, passos do otimizador, gradientes, recursos | Não: esses dados não ficam no `results.csv` e não podem ser reconstruídos. | Sim (`training_monitor.py`). |

`resume-pilot` só liga o monitor em treinos que já têm `ficha/`. O 614d070a continua
retomando exatamente como antes.

## Comandos

```powershell
# Evolução e tempo efetivo (só lê a pasta do treino; grava runs\progress_*)
python -m dronecamp_ia training-progress "C:\DroneCamp_runs\pilot_train_..._614d070a"

# Qualidade detalhada nas MESMAS fotos para vários pesos (grava runs\quality_*)
python -m dronecamp_ia evaluate-detailed --data data\pilot\...\dataset.yaml `
    --weights runs\...12a305a6\fit\weights\best.pt --label 12a305a6 `
    --weights C:\DroneCamp_runs\...614d070a\fit\weights\best.pt --label 614d070a `
    --split test --onnx-provider cpu

# Cobertura dos dados por classe (grava runs\coverage_*); --baseline compara com uma anterior
python -m dronecamp_ia data-coverage --data data\pilot\...\dataset.yaml --run runs\pilot_train_...

# Ficha do treino: tentativas, como terminaram e conferência dos hashes
python -m dronecamp_ia run-card runs\pilot_train_...
```

Opções do `evaluate-detailed`:
- `--conf`: confiança do ponto de operação. O padrão é `pilot.suggestion_conf`.
- `--regime janelas|inteira`: mesmas janelas das sugestões, ou a foto inteira.
- `--size-edges 32 96`: lados, em pixels, que separam pequeno, médio e grande.
- `--fp-budget`: falsos alarmes por foto para o recall com limite (repetível).
- `--conditions arquivo.csv`: etiquetas de captura. A primeira coluna é `image_sha256`; as outras são livres (material, angulo, iluminacao...).
- `--examples`: quantos erros guardar por classe e por tipo.
- `--no-crops`: não grava os recortes dos erros.

## 1. Qualidade por classe (`detection_metrics.py`, `quality_report.py`)

Unidade: **foto original**. As previsões das janelas sobrepostas são unidas na foto
(`suggestions.detect_boxes`, NMS por classe com `merge_iou`) antes de qualquer
contagem. Por isso, "falso alarme por foto" conta alarmes que o revisor de fato veria.

Por classe:
- **Suporte**: fotos e caixas humanas. Classe sem caixa aparece como `nao_avaliada`, com suporte 0 e métricas `null`. Ela nunca aparece como 0,0, porque 0,0 significaria "errou tudo".
- **AP50, AP75, AP50–95**:
  - AP é a área sob a curva precisão × recall, com interpolação em todos os pontos (`model_gate.average_precision`).
  - AP50–95 é a média de AP nos IoUs 0,50 a 0,95, de 0,05 em 0,05 (padrão COCO).
  - As previsões usadas no AP vão até a confiança `AP_CONF` (0,001).
- **TP/FP/FN, P, R e F1 no ponto de operação** (`conf`):
  - TP: caixa prevista casada com uma caixa humana (IoU ≥ 0,5, casamento húngaro por confiança).
  - P = TP/(TP+FP), R = TP/(TP+FN) e F1 = 2PR/(P+R).
- **Tipos de falso positivo**, testados nesta ordem:
  1. `duplicata`: casa com uma caixa humana já usada por uma previsão mais confiante.
  2. `classe_trocada`: IoU ≥ 0,5 com uma caixa humana de outra classe.
  3. `caixa_imprecisa`: mesma classe, com 0,1 ≤ IoU < 0,5.
  4. `fundo`: o resto. Pode ser também um defeito que o revisor não marcou.
- **Motivo de cada omissão**: `classe_trocada`, `confianca_baixa` (caixa certa abaixo de `conf`), `caixa_imprecisa` ou `nao_detectado`.
- **Localização**:
  - Quantis da IoU das caixas casadas (mediana e p95).
  - Recall e precisão por faixa de tamanho. A faixa usa a área em px² na foto original: abaixo de 32² é pequeno, de 32² a 96² é médio, acima de 96² é grande.
- **Curvas por confiança** (0,00 a 0,95, de 0,05 em 0,05), para o total e por classe: P, R, F1 e FP por foto, com a melhor F1 e a confiança em que ela ocorre.
- **Recall com limite de falsos alarmes**: o maior recall possível sem passar de N falsos positivos por foto (`--fp-budget`), com a confiança que o alcança. Previsões com a mesma confiança entram juntas, e no total contam também os alarmes de classes sem caixa humana na divisão.

**Condições de captura**:
- O relatório estratifica por `edificacao` e `cena` (vindas do `pilot.json`), por `periodo` (o mês da captura no EXIF, `AAAA-MM`) e pelas colunas do CSV de `--conditions`.
- Cada valor traz número de fotos, suporte, P, R, F1, mAP50 e FP por foto.
- Cada campo traz também `unknown_share`, a fração de fotos sem a etiqueta.
- Com poucas fotos por grupo, a diferença entre grupos é anedota, não evidência.

**Protocolo gravado**: dataset, hash do `pilot.json` e da taxonomia, divisões, regime,
fatiamento, imgsz, max_det, confiança, IoU, interpolação, consolidação, fotos excluídas
(vistas no treino de qualquer peso) e o **checkpoint** avaliado. Do checkpoint ficam
o hash, a época e o motivo: `best.pt` é a época de maior mAP50-95 de validação no
`results.csv`.

**Exemplos de erro**: `errors/<rótulo>/<classe>/` guarda um recorte por erro, com a
caixa humana em verde e a do modelo em vermelho. Em `quality.json`, cada erro aponta
para a foto, a caixa humana, a previsão, a classe, a confiança, o hash e a época do
checkpoint.

## 2. Cobertura dos dados (`data_coverage.py`)

Por divisão e por classe, o relatório conta:
- fotos originais, objetos, cenas e edificações;
- janelas (recortes) e objetos dentro delas. A janela é ligada à foto de origem pelo nome `<foto>_p<i>_<x>_<y>`;
- fotos negativas e janelas negativas;
- aparições na lista de reamostragem (`train_repeat_factor.txt`) e a repetição máxima.

`--baseline` compara com uma cobertura anterior e dá um veredito por classe no treino:
- `exemplos_novos`: mais objetos únicos;
- `so_repeticao`: mesmos objetos, mais aparições;
- `menos_exemplos`: menos objetos únicos;
- `igual`: nada mudou.

Repetir uma foto rara não equivale a ter mais exemplos.

## 3. Evolução do aprendizado (`training_progress.py`)

O comando lê do `results.csv`:
- melhor época, sua fitness (mAP50-95, a mesma régua do `best.pt`) e a diferença da última época até ela;
- épocas sem melhora e a paciência configurada;
- learning rate, perdas não finitas e a época da menor perda de validação.

Tempo efetivo:
- A coluna `time` da Ultralytics zera a cada retomada. A queda do relógio marca uma nova tentativa.
- Épocas refeitas depois de uma queda contam uma vez só.
- O resultado traz mediana e p95 por época, mais as médias das 5 primeiras e das 5 últimas épocas.

## 4. Instrumentação do treino (`training_monitor.py`, só treinos novos)

O monitor envolve métodos do treinador da Ultralytics e grava uma linha por época
em `runs/<treino>/instrumentation/epochs.jsonl`:
- **Etapas por lote**, com mediana, p95, média e soma de cada uma:
  - `data_wait`: espera pelo próximo lote;
  - `preprocess`;
  - `forward_loss`: até `scaler.scale`;
  - `backward`;
  - `optimizer_step`, que inclui o corte do gradiente;
  - `batch`: o lote inteiro.
- **Validação e checkpoint**: segundos de cada um.
- **Contadores**: lotes, imagens, passos efetivos do otimizador (com `accumulate`), passos pulados pelo GradScaler, perdas e gradientes não finitos e o `step` do estado do otimizador.
- **Gradiente**: norma L2 antes do corte, com mediana, p95 e máximo.
- **Também por época**: lr por grupo, métricas, fitness e best_fitness.

A validação final do `best.pt`, que a Ultralytics faz depois da última época, vai para
`final_eval.jsonl` e não conta como época extra.

**Recursos**: `instrumentation/resources.jsonl` recebe uma amostra a cada
`resource_seconds` com CPU total e do processo, RAM usada e disponível, RSS/VMS do
processo, swap, disco livre, leitura e gravação em disco (MB/s), rede (MB/s) e threads.
Os picos ficam no fim da tentativa.

**Custo**: dois relógios por etapa, mais a norma do gradiente uma vez por passo e
uma amostra a cada 30 s. Para desligar, use `pilot.instrumentation` em
`configs/project.yaml` (`enabled`, `resource_seconds`, `grad_norm`).

Ainda não implementado: o profiler do PyTorch em janelas curtas. O ponto de entrada
está marcado com `EXTENSAO:` em `on_train_batch_start`.

## 5. Ficha imutável (`run_record.py`)

Cada tentativa grava arquivos em `runs/<treino>/ficha/`, abertos em modo exclusivo:
nenhum é reescrito. O hash de cada arquivo vai para `index.jsonl`.

- `attempt_NNN_start.json`:
  - experimento, ID da tentativa, tipo (`novo`/`retomada`), hora, máquina e processo;
  - commit, branch e arquivos alterados localmente, mais as versões das bibliotecas;
  - hashes do manifesto, da taxonomia, dos grupos e um hash por divisão (imagens, anotações e a própria divisão);
  - peso inicial ou checkpoint pai, ambos com hash;
  - configuração completa e seeds;
  - as tentativas anteriores que terminaram sem registro, com o último sinal de vida de cada uma.
- `attempt_NNN_trainer.json`: o que a Ultralytics usou de fato:
  - argumentos efetivos, época inicial e best_fitness herdado;
  - lote, `accumulate`, otimizador e grupos de lr;
  - número de imagens e o hash do peso.
- `attempt_NNN_end.json`:
  - status (`concluida`, `interrompida_pelo_usuario` ou `falhou`) e o motivo;
  - época global alcançada, passos, lotes, imagens e tempo efetivo;
  - picos de recursos.
  - Uma queda de energia não grava este arquivo; a próxima tentativa registra a falta.
- `evaluation_*.json`: cada `evaluate-detailed` de um peso desse treino, com o checkpoint, o motivo da escolha e o protocolo.

`run-card` confere o índice e devolve código 2 se algum arquivo foi editado, apagado
ou acrescentado fora dele.

## 6. Registros do aprendizado mútuo (`mutual_learning.py`, `dml_controller.py`, `dml_lan.py`)

Cada linha de época do `history.jsonl` traz quatro registros.

**Influência dos pares**: perda de detecção, consistência bruta e aplicada, peso λ e
escala média.

**Cobertura da cooperação**:
- fotos totais, com pares e com orientação útil;
- motivos de não haver orientação;
- alvos fundidos, ancorados nas caixas humanas e casados.
- Os alvos casados são contados **uma vez por foto**: `matched_targets_final_head` conta a cabeça final e `matched_targets_any_head` conta a união das cabeças. Antes a contagem somava cabeças e contava o mesmo objeto várias vezes.
- Por par: o que foi recebido, usado e descartado, por classe.

**Decisões do UCB**:
- braço, multiplicador e motivo (`nunca_tentado` ou `maior_ucb`);
- contagem, média, bônus e escore de cada braço;
- recompensa (Δ mAP50 de validação), ou o motivo de não atualizar (`metrica_ausente_*`, `rl_desligado`).
- A recompensa sozinha não prova que o controlador causou o ganho.

**Tempo**: `seconds.model[<modelo>]` traz o tempo **próprio** de cada modelo (forward e
perda, consistência, backward e passo, validação, checkpoint). `seconds.shared` traz o
tempo compartilhado (dados, professores, cache dos pares). Antes, o campo acumulava o
relógio desde o início comum da época.

**Coordenação na rede local**:
- `state.json`: atribuições (máquina, hora de atribuição e conclusão), downloads (vezes e bytes) e falhas.
- `rounds/<r>/joint.json`: versões, hashes, multiplicadores e decisões da rodada.
- `worker_log.jsonl` de cada máquina: espera, download, treino e envio, com bytes e segundos, e falhas de rede.

## Leitura correta

- Validação com poucas fotos oscila. Antes de concluir algo da melhor época ou de um grupo de condição, confira o suporte (fotos e caixas).
- AP com suporte de 1 ou 2 caixas é anedota.
- Comparar pesos só nas mesmas fotos e com a mesma régua. O `evaluate-detailed` garante isso: tira de todos os pesos as fotos vistas no treino de qualquer um.
- Adoção continua passando pelo `compare-models`, com ganho por classe contra o 12a305a6.

## Convenção do código

Os módulos seguem o [guia de manutenção](guia_manutencao.md):
- cabeçalho com função, quem chama, o que lê e grava, e "Onde mexer";
- docstrings com Parâmetros/Retorna/Erros/Chamada por;
- passos numerados;
- marcadores `ENTRADA:`, `EXTENSAO:` e `ATENCAO:`.

Os testes ficam em `tests/test_training_metrics.py`. Um deles faz um treino real de
2 épocas com o monitor ligado e pode ser pulado com `DRONECAMP_SKIP_E2E=1`.
