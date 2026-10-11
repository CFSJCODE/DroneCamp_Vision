# Recortes, avaliação na GPU durante o treino e relatório comparativo

Este documento junta três peças do pipeline piloto voltadas às fotos de drone do
CEASA Lado A (DJI, 5472×3648 px) e à pergunta "o modelo novo é melhor que os
anteriores?". Nada aqui aprova pesos: a adoção continua dependendo do
`compare-models` em fotos fora do treino de ambos os modelos.

## 1. Recortes (tiling) — `tiling.py`, `pilot.tiling`

As caixas revisadas no Lado A têm mediana de ~40 px no lado menor. Reduzida
inteira para 640 px, uma foto de 5472 px encolhe 8,5× e a caixa vira ~5 px. A
solução (estilo SAHI) é trabalhar em janelas:

| chave de `pilot.tiling` | padrão | efeito |
|---|---|---|
| `enabled` | true | liga o fatiamento nas sugestões, no predict com pesos piloto, no gate e na medição |
| `min_side` | 2000 | fotos com lado maior abaixo disso (laudos de 1199 px) continuam inteiras |
| `patch` | 1280 | lado da janela em px; cada janela é reduzida ao `imgsz` do modelo (1280 → 640 = 2×) |
| `overlap` | 0,2 | sobreposição entre janelas vizinhas |
| `full_image` | true | também infere a foto inteira (rufo, calha: objetos maiores que uma janela) |
| `merge_iou` | 0,5 | NMS por classe entre janelas vizinhas |

- **Treino**: `train-pilot --tile` gera as janelas do dataset piloto auditado na
  mesma escala (`tiling_manifest.json` no run; `execution.json` → `tiling`,
  `label_unit: "janelas"`). Splits são preservados.
- **Inferência**: `suggest`, `predict` (pesos piloto), `compare-models`,
  `measure-models` e `scripts/evaluate_models.py` fatiam as fotos grandes e
  juntam as caixas. O `findings.jsonl` do predict registra `inference.regime`
  e o número de janelas.
- **Cenas de voo**: `build-pilot-data --sequence-block 10` agrupa fotos
  consecutivas (`DJI_0194…`) em blocos, para uma cena não cair em treino e teste.
- **Escala 1:1**: `patch: 640` com `imgsz 640` treina e infere na resolução
  original (4× mais janelas, ~4× mais CPU por época). Só vale medir depois do
  primeiro ciclo com 1280 → 640.

## 2. CPU + GPU em conjunto — `gpu_eval.py`, `train-pilot --gpu-eval`

A Radeon Vega da APU não treina (sem CUDA/ROCm no Windows), mas roda ONNX pelo
DirectML. Com `train-pilot --gpu-eval` (ou `pilot.gpu_eval.enabled: true`):

1. O treino continua na CPU (Ultralytics).
2. A cada `every` épocas, o callback `on_model_save` copia `last.pt` para a fila
   `runs/<treino>/gpu_eval/queue/` (cópia atômica; na área de gravação quando
   `DRONECAMP_FIT_STAGING` está definido). Se a fila ainda tem um checkpoint, a
   época é pulada (a última nunca é pulada).
3. Um processo à parte (`python -m dronecamp_ia gpu-eval-worker`, prioridade
   abaixo do normal, 2 threads) exporta o checkpoint para ONNX com o `imgsz` do
   treino, avalia o split `val` nas fotos **inteiras** com o fatiamento das
   sugestões na GPU e grava uma linha em `gpu_eval/eval_curve.jsonl` (mAP50,
   precisão, recall, por classe, estratos fatiadas/inteiras, segundos de
   exportação e inferência, provider realmente ativo). Checkpoints antigos na
   fila são descartados como `superseded`.
4. No fim do treino, `execution.json` → `gpu_eval` resume: épocas enfileiradas e
   puladas, melhor época por mAP50 e o bloco `parallel` com os segundos de GPU
   executados durante o treino (`gpu_busy_fraction`).

Segurança: o worker recebe um pipe em stdin e encerra sozinho quando o treino
morre; sem DirectML o treino segue e `worker.json` registra `skipped`. A curva
**não escolhe pesos**: `best.pt` continua vindo da validação da Ultralytics.
A plataforma lê `eval_curve.jsonl` e também `curve.jsonl` do
`scripts/gpu_eval_watcher.py` (vigia em segundo terminal), e usa `fit/heartbeat`
(tocado a cada minuto pelo treino) para não marcar épocas longas como
"interrompido".

## 3. Medição e relatório comparativo — `measure-models`, `report-models`

Só uma medição nas MESMAS fotos compara modelos. `measure-models` avalia N pesos
(`.pt` ou `.onnx`, na GPU com `--onnx-provider directml`) nas fotos de `test`
fora do treino de todos, nos regimes `janelas` e `inteira`, com estratos
(fotos fatiadas × inteiras), e grava `runs/measure_*/measurement.json`.
`report-models --measurement <arquivo>` gera `runs/report_*/report.html` (abre
do disco) com mAP50 e mAP50-95 por modelo, por classe (recall com intervalo de
Wilson 95 %), por estrato, as curvas de treino e a curva da GPU. Um modelo = uma
cor fixa. `compare-models` aceita `.onnx` (`--onnx-provider`) e grava também
`metrics_whole` (foto inteira) e `strata`.

## 4. Ciclo completo — `retrain-cycle`

```bat
.venv\Scripts\python -m dronecamp_ia retrain-cycle ^
  --registry data\reviews\ceasa_v9_rev001\registry.json ^
  --registry data\reviews\ceasa_lado_a_v1\registry.json ^
  --output data\pilot\ceasa_v11_lado_a_s42 --sequence-block 10 ^
  --baseline runs\pilot_train_<56038eb9>\fit\weights\best.pt ^
  --weights yolo26s.pt --epochs 80 --title "v11 Lado A"
```

Etapas: `build-pilot-data` → `train-pilot --tile --gpu-eval` → `export` →
`compare-models` contra cada `--baseline` → `measure-models` → `report-models`.
O resultado fica em `runs/cycle_*/cycle.json`, com `adopt` de cada gate.

## Limites

- O ganho dos recortes em fotos de drone ainda depende da revisão do Lado A
  concluída e de um treino completo no PC.
- `measure-models` só é comparável quando nenhuma foto de teste esteve no treino
  de qualquer modelo (o arquivo registra os hashes excluídos).
- A avaliação na GPU usa o split `val`; `test` fica reservado ao gate.
