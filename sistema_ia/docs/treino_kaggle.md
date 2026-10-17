# Treino piloto na GPU do Kaggle

O treino piloto roda hoje na CPU do Ryzen 5 4600G: cerca de 75 min por época com o yolo26x em janelas de 640. O Kaggle oferece GPUs grátis, com cerca de 30 h por semana e sessões de até 12 h. Este fluxo leva o treino para uma T4 **sem levar revisões, registros nem fotos de 20 MP**: vai só um pacote com as janelas já cortadas e os hashes de tudo.

Nada aqui aprova pesos. O run que volta do Kaggle é um `runs/pilot_train_*` como os outros e precisa do `compare-models` por classe antes de qualquer adoção.

> **Dados do cliente.** O pacote contém recortes das fotos da CEASA. Mesmo num Dataset privado, os arquivos ficam nos servidores do Kaggle (Google). Envie só com autorização do cliente ou da PUC TEC.

## Peças

| Onde | Comando / arquivo | O que faz |
|---|---|---|
| PC | `kaggle-package --data <piloto>/dataset.yaml --output <pasta>` | Reconfere o piloto contra as revisões humanas, gera as janelas de `pilot.tiling` (mesma semente do `train-pilot --tile`) e grava pasta + `.zip` |
| PC | `kaggle-package --resume-run runs\pilot_train_<id> --output <pasta>` | Copia as janelas que aquele treino usa, o `last.pt` e o `results.csv` (só leitura: o treino do PC segue intocado) |
| PC | `kaggle/enviar_kaggle.py --usuario <você> --run <run>` | Monta o pacote, sobe o Dataset privado e dispara o notebook na T4 (seção abaixo) |
| Kaggle | `kaggle/treino_t4.ipynb` | Clona o repositório, prepara Python 3.12 + CUDA se preciso e chama `kaggle-train` |
| Kaggle | `kaggle-train --package /kaggle/input --device 0 --max-hours 11` | Confere os hashes, treina (ou retoma) e grava `runs/pilot_train_*` |

Código: `src/dronecamp_ia/kaggle_package.py`. Testes: `tests/test_kaggle_package.py`, incluindo um treino real curto que para pelo limite de horas e continua numa segunda sessão.

## Conteúdo do pacote

```text
dronecamp_kaggle_<data>_<id>/
  package.json            tipo, commit, parâmetros, auditoria, contagens e SHA-256 de cada arquivo
  dataset/dataset.yaml    path: . (caminhos relativos)
  dataset/images/{train,val,test}/   dataset/labels/{train,val,test}/
  dataset/train_list.txt  só se o treino de origem repetia fotos de classes raras
  checkpoint/last.pt      só na continuação
  checkpoint/results.csv  só na continuação (o histórico continua no run novo)
  source_run/             execution.json, summary.json etc. do treino de origem
  configs/                project.yaml e taxonomia usados para montar o pacote
```

O `kaggle-train` recusa o pacote se algum arquivo listado mudou ou se aparece um arquivo a mais em `dataset/`.

## Passo a passo

1. **No PC** (na pasta `sistema_ia`):

   ```powershell
   # Treino novo com o piloto atual
   .\.venv\Scripts\python.exe -m dronecamp_ia kaggle-package --data "data\pilot\SEU_PILOTO\dataset.yaml" --output "C:\DroneCamp_kaggle"
   # ou: continuar o treino em andamento na GPU (o do PC não é parado)
   .\.venv\Scripts\python.exe -m dronecamp_ia kaggle-package --resume-run "runs\pilot_train_..._614d070a" --output "C:\DroneCamp_kaggle"
   ```

   Com a área de gravação (`DRONECAMP_FIT_STAGING=C:\DroneCamp_runs`), o `last.pt` mais novo é procurado lá, como no `resume-pilot`.

2. **No Kaggle:** crie um Dataset **privado** e envie o `.zip`. Importe o notebook `kaggle/treino_t4.ipynb` e configure:
   - Accelerator: **GPU T4 x1**. A P100 não serve, porque o PyTorch atual (rodas CUDA 12.8 e mais novas) não roda na arquitetura dela; o notebook para com uma mensagem se não houver GPU compatível.
   - Internet: **On**, para clonar o repositório e baixar o peso oficial no treino novo.
   - Add Input: o Dataset do pacote.

3. **Rode com Save Version → Save & Run All (Commit).** A execução segue em segundo plano e `runs/` vira a saída da versão.

4. **Se não terminar em uma sessão:** com `--max-hours 11` (padrão do notebook), o treino para depois da primeira época que passar de 11 h, grava `fit/weights/resume.pt` (com o otimizador) e marca o `summary.json` como `interrupted`. Numa nova versão, adicione a saída anterior como Input e aponte `CONTINUE_FROM` para esse `resume.pt`. O `results.csv` anterior é copiado e as épocas continuam a numeração.

5. **De volta ao PC:** baixe `runs/pilot_train_*` da aba Output, copie para `sistema_ia\runs\` e compare com o modelo de referência por classe, tamanho de caixa, falsos positivos e omissões.

## Envio automático pelo PC

`kaggle/enviar_kaggle.py` faz os passos 1 a 3 de uma vez: monta o pacote, sobe o `.zip` como Dataset **privado** (`dronecamp-pacote`; nas próximas vezes, como nova versão), espera o Kaggle processar e envia o notebook (`dronecamp-treino-t4`, privado, T4, internet ligada) com as opções da execução. O token não passa pelo script: o cliente `kaggle` o lê de `%USERPROFILE%\.kaggle\access_token` ou de `KAGGLE_API_TOKEN`.

```powershell
.\.venv\Scripts\python.exe -m pip install kaggle
# Teste: continua o treino do PC na T4 e para depois de ~15 min, o que mede o tempo por época.
.\.venv\Scripts\python.exe kaggle\enviar_kaggle.py --usuario SEU_USUARIO --run "runs\pilot_train_..._614d070a" --max-horas 0.25
# Acompanhar e baixar
kaggle kernels status SEU_USUARIO/dronecamp-treino-t4
kaggle kernels output SEU_USUARIO/dronecamp-treino-t4 -p C:\DroneCamp_kaggle\resultado
```

Antes do merge deste código na `main`, passe também `--branch claude/project-thread-sl8ytb`, porque o notebook clona o repositório no Kaggle. `--so-preparar` monta as pastas sem enviar nada, e `--pacote <pasta>` reaproveita um pacote já montado.

Com o treino do PC rodando, a cópia do `last.pt` pode coincidir com a gravação do fim de uma época. O pacote abre a cópia para conferir e, se ela não abrir, pede para rodar de novo.

## Ajustes de GPU

O `project.yaml` foi calibrado para a CPU (`amp: false`, `batch: 4`, `workers: 0`). Na GPU, o `kaggle-train` usa `amp: true`, `batch: 16` e `workers: 4` (`GPU_DEFAULTS`), e as opções `--batch`, `--workers` e `--no-amp` vencem esses valores. A Ultralytics acumula gradientes até um lote nominal de 64, então trocar o batch não muda o lote efetivo do otimizador.

Na continuação, a Ultralytics usa os argumentos gravados no `last.pt` (dataset em `E:/`, saída em `C:\DroneCamp_runs`, `device=cpu`, `amp=False`). O `kaggle-train` grava uma cópia temporária do checkpoint com `data`, `save_dir`, `device` e os ajustes de GPU trocados (`rebase_checkpoint`) e a apaga no fim. O checkpoint do pacote não muda. `--epochs` não vale na continuação: o total de épocas é o do treino original.

## Tempos esperados (estimativa, ainda não medida)

| Treino | 4600G (CPU) | T4 (estimado) |
|---|---|---|
| yolo26x, 120 épocas, janelas de 640 (1.234 treino / 304 val) | ~150 h | 4-10 h |
| Restante do 614d070a a partir da época 61 | ~74 h | 2-5 h |

A conta: o yolo26x faz cerca de 190 GFLOPs por imagem de 640; com o passo de volta, ~0,6 TFLOP por imagem, ou ~0,9 PFLOP por época. Uma T4 com AMP rende na prática 15-25 TFLOPs, o que dá menos de 1 min de GPU por época; as 4 vCPUs fazendo o aumento de dados devem ser o gargalo. Meça com um teste curto (`EXTRA_ARGS = ["--epochs", "2"]` num pacote de treino novo) antes de planejar.

## Limites e cuidados

- A cota semanal e o menu de GPUs do Kaggle mudam sem aviso; confira em Settings da conta.
- GPU e CPU não dão números idênticos (AMP, ordem das somas). As duas cópias de um treino continuado divergem a partir da época copiada; a escolha continua sendo pelo `compare-models`.
- O `execution.json` do run mostra `device_requested: cpu` (vem do `project.yaml` do clone); o dispositivo real está em `device` no mesmo arquivo.
- Conferido localmente apenas na CPU (testes acima). O caminho do notebook no Kaggle (clone, Python, CUDA) ainda não foi executado lá.
