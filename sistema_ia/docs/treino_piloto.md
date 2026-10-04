# Treino piloto, sugestões do modelo e APU — 04/10/2026

Origem: pedido do usuário para implementar treino, predição e exportação reais (sem mocks), sugerir caixas em fotos como a dos fixadores e melhorar o modelo com o arquivo de revisão `dronecamp-revisao-ceasa_v6_corrigida_a36ea7f67d08.json` (revisor `revisao002`). O conteúdo do arquivo foi tratado como dado de revisão.

## Revisão incorporada

O feedback está vinculado por SHA-256 ao registro da v6 (`ae200ee0…`). A importação gerou [a revisão v7](../data/reviews/ceasa_v7_revisao002_ba8cc323c8a1/registry.json), com cópia do feedback na mesma pasta: **34 fotos aprovadas, 4 ambíguas, 117 caixas**. As 6 caixas de `pedaco_telha_sobreposto` (ID 10) e a única de `fixador_telha_frouxo` (ID 11) vieram desta revisão. `reparo_rufo` (ID 12) continua sem nenhuma caixa.

## Por que um piloto

O treino de produção continua exigindo três edificações e exemplos das 13 classes. Com só o CEASA, ele é bloqueado de propósito. O piloto (`build-pilot-data` e `train-pilot`) treina de verdade com as fotos aprovadas para que o modelo **sugira caixas na revisão**, sem ser aprovado para uso:

- treino, validação e teste são separados por `scene_group`; recortes da mesma área nunca ficam em dois splits;
- classes com uma única foto ficam só no treino (não são medidas);
- `pilot.json` guarda os hashes de cada foto/label e do registro; antes de treinar, `validate_pilot_dataset` refaz a conferência e recalcula as labels a partir das caixas humanas;
- `summary.json` registra `pilot: true` e `production_ready: false`; `predict` grava os achados como `candidatos_modelo_piloto_nao_validado`.

## Treino executado

| Item | Valor |
| --- | --- |
| Dataset | [`data/pilot/ceasa_v7_piloto_s42`](../data/pilot/ceasa_v7_piloto_s42/pilot.json): 23 fotos treino (93 caixas), 6 val (9), 5 teste (7) |
| Modelo | YOLO26l (`models/yolo26l.pt`), `nms=False`, 640 px, batch 4, AdamW lr0 0,001, 80 épocas |
| Execução | `runs/pilot_train_20261004T170212Z_dd284411`, 1,03 h na CPU (Ryzen 5 4600G, 6 núcleos) |
| Auditoria | `complete: true`, contrato das 13 classes verificado, `best.pt`/`last.pt`/`results.csv` presentes |
| Validação (6 fotos, só 3 classes medidas) | precisão 0,566 · recall 0,25 · mAP50 0,231 · mAP50-95 0,077 |

Conferência das sugestões (confiança ≥ 0,15) contra as caixas humanas, mesma classe e IoU ≥ 0,5:

| Fotos | Caixas humanas reencontradas | Sugestões corretas |
| --- | ---: | ---: |
| Treino (23) | 75 / 93 | 109 / 143 |
| Validação (6) | 2 / 9 | 4 / 14 |
| Teste (5) | 0 / 7 | 0 / 4 |

**Leitura:** o modelo memorizou as fotos de treino e quase não generaliza para fotos que não viu. Com 23 fotos de uma edificação isso é o esperado. As sugestões já economizam trabalho nas fotos parecidas com as revisadas, mas não substituem a revisão. Na foto dos fixadores (`57321a0d…`, ambígua) a única sugestão foi `residuos_telha` com 34%; o modelo ainda não reconhece fixadores (há uma única caixa da classe 11 no treino).

## Sugestões na página de revisão

`suggest --registry` grava `suggestions.json` ao lado do registro (hash do registro, hash dos pesos, confiança mínima e tamanho da imagem) e regenera a página. O atalho [Abrir revisao DroneCamp.cmd](../../Abrir%20revisao%20DroneCamp.cmd) agora abre a v7, com **170 sugestões em 33 fotos**.

Na página, as sugestões aparecem tracejadas com “IA n” e a confiança. **Aceitar** copia a caixa para a revisão, com nota de origem, e volta a decisão da foto para pendente; **Descartar** esconde a sugestão nesta revisão. Exportar e importar o arquivo de revisão continua sendo o único caminho para uma caixa chegar ao dataset.

Para fotos novas de outra edificação: `suggest --source PASTA --output data\reviews\NOVA --group NOME`. As fotos são copiadas para `originals/` com hash, entram como ambíguas e sem caixas, e as sugestões viram o ponto de partida da revisão. Revisões novas somam-se com `build-pilot-data --registry A --registry B`; se a mesma foto aparecer em duas, vale a última.

Pesos piloto inferem no tamanho em que foram treinados (640 px), não nos 1024 px da produção.

## Exportação ONNX e paridade

`export` agora roda `.pt` e `.onnx` nas mesmas fotos e grava `parity.json`. Execução real: `runs/export_20261004T180523Z_14476aa6`, ONNX de 640 px, **27/27 caixas pareadas nas 5 fotos, diferença de confiança 0,0**, `runtime_parity_validated: true`. A primeira versão do teste revelou um erro real: o `.pt` usava letterbox retangular e o ONNX entrada quadrada fixa; a comparação agora força `rect=False` e lê o tamanho dos metadados do ONNX.

## CPU, APU e memória

- **Treino:** só na CPU. A Ultralytics treina em CUDA (NVIDIA) ou MPS (Apple); a Radeon integrada não tem PyTorch no Windows. O PyTorch usa os 6 núcleos físicos.
- **Predição ONNX na APU:** com autorização do usuário, `onnxruntime` 1.30.0 (CPU) foi trocado por `onnxruntime-directml` 1.24.4 (lista anterior em `.runtime/backup_20261004T1700_antes_treino_real/pip_freeze_antes_directml.txt`). Medido no ONNX do piloto, 640 px: **CPU 372 ms/imagem; APU (DirectML) 124 ms/imagem**. Nas 10 primeiras fotos do CEASA, CPU e APU deram as mesmas 609 caixas (confiança idêntica, IoU 1,0).
- A predição feita pela Ultralytics (`predict`, `suggest`, paridade) continua na CPU, porque a biblioteca só escolhe CUDA ou CPU para ONNX. O ganho da APU vale para quem rodar o `model.onnx` com `DmlExecutionProvider` diretamente (ex.: aplicação de campo).
- Os 16 GB livres permitem `cache: ram` e batch maior no próximo treino; não foram usados nesta execução.

## Testes

`tests/test_e2e_ultralytics.py` roda sem mocks: dataset piloto com as revisões reais → treino de 1 época → `predict` → exportação com paridade → sugestões aceitas num feedback chegando ao dataset seguinte. `tests/test_pilot.py` cobre a divisão por cena e o pareamento da paridade. Suíte completa: **170 testes OK** (com DirectML instalado).

O roteiro manual de integração pedido no projeto também foi executado: `scripts/create_e2e_dataset.py` (acervo v3, 33 fotos; corrigido para ler as subpastas `images/` e `labels/`), 1 época em 3 min 8 s, predição e exportação ONNX em `runs/detect/runs/train/e2e_test`. Esse dataset usa validação igual ao treino e não deve gerar sugestões.

## Próximos passos

1. Revisar a v7 com as sugestões (aceitar, corrigir, descartar), em especial as 4 fotos ambíguas e as classes 10–12; exportar o arquivo de revisão.
2. Fotografar outras edificações e abrir revisões com `suggest --source … --group …`; isso melhora o piloto e é o que libera o treino de produção.
3. Reconstruir o piloto com todas as revisões e treinar de novo (considerar `cache: ram`, batch 8 e 1024 px para objetos pequenos como fixadores).
4. Só considerar uso real depois de métricas por classe em edificações que não entraram no treino.

## Rodada v8 — 2ª revisão da `revisao002` (04/10/2026)

Origem: pedido do usuário para retreinar com o arquivo `dronecamp-revisao-ceasa_v7_revisao002_ba8cc323c8a1.json` (revisor `002`, exportado em 2026-10-04T19:46:45.427Z) e usar também as fotos antigas. Execução numa sessão na nuvem (4 vCPU Xeon 2,1 GHz, 15 GB), com `torch 2.14.1` e `onnxruntime 1.24.4` (CPU) do PyPI.

### Revisão incorporada

O feedback (SHA-256 `ff34e226…`) está vinculado ao registro v7 (`5cf15c7f…`). A importação gerou [a revisão v8](../data/reviews/ceasa_v8_revisao002_ff34e226416c/registry.json): **37 fotos aprovadas (36 positivas, 1 negativa), 1 ambígua (fixadores), 246 caixas**, das quais 129 são sugestões do modelo aceitas e 41 sugestões foram descartadas.

**Duplicatas:** 91 caixas eram sugestões aceitas sobre caixas já desenhadas no mesmo objeto (mesma classe, IoU ≥ 0,7). Em três fotos havia mais de 10 duplicatas cada. O registro humano foi preservado; no dataset piloto cada objeto vira um único alvo (`pilot.duplicate_iou: 0.7`), ficando a caixa do revisor ou, entre sugestões, a de maior confiança: **246 → 155 caixas**. A página agora oferece **Substituir caixa N** quando a sugestão cobre uma caixa existente, para a duplicata não voltar.

### Dataset e treinos

| Item | Valor |
| --- | --- |
| Dataset | [`data/pilot/ceasa_v8_piloto_s42`](../data/pilot/ceasa_v8_piloto_s42/pilot.json): treino 22 fotos (114 caixas), val 6 (8, uma negativa), teste 9 (33). Todas as 34 fotos do piloto v7 estão incluídas |
| Treino 1 | `runs/pilot_train_20261004T195922Z_5cdfa63b`: **parou na época 66** por `patience 25`; melhor época 41 (val mAP50 0,072 · mAP50-95 0,033). Pesos não publicados: reproduzíveis pelo treino 2 |
| Treino 2 | `runs/pilot_train_20261004T204113Z_973650d1`: mesmos dados, `patience 0`, **80 épocas em 47 min**. Determinístico: idêntico ao treino 1 até a época 66 |
| Pesos | `best.pt` = época 41 (SHA-256 `a7a4e0bd…`); `last.pt` = época 80 (`9c1fd92c…`); 13 classes, `complete: true` |

A validação do piloto tem 6 fotos e 8 caixas: um acerto a mais ou a menos muda o mAP50 de 0,003 para 0,07. Por isso a parada antecipada escolheu a época 41, ainda subtreinada, e foi desligada no piloto (`patience: 0`).

### Comparação com o piloto anterior

Contagem com [`scripts/compare_pilot_models.py`](../scripts/compare_pilot_models.py): caixas humanas reencontradas (mesma classe, IoU ≥ 0,5, confiança ≥ 0,15), separando as fotos que cada modelo viu no treino.

| Modelo | Fotos não vistas | Humanas reencontradas | Sugestões corretas | Fotos de treino |
| --- | ---: | ---: | ---: | ---: |
| Piloto v7 (80 épocas) | 14 | 7/29 (24%) | 7/26 | 75/93 |
| v8 época 41 (`best.pt`) | 15 | 9/41 (22%) | 9/25 | 56/114 |
| v8 época 80 (`last.pt`) | 15 | 6/41 (15%) | 6/16 | 88/114 |

Nas 7 fotos da internet, contra as 29 caixas propostas pela IA: v7 reencontra 5, v8 `best.pt` 4, v8 `last.pt` 4.

**Leitura:** o retreino com as anotações novas **não melhorou de forma mensurável** a generalização. As diferenças estão dentro do ruído de 14–15 fotos. O modelo memoriza as fotos de treino (até 77%) e acerta 15–24% nas outras. O gargalo é a quantidade e a diversidade de fotos (22 fotos de treino, uma edificação, 13 classes), não o treino. Para as sugestões foi escolhido o v8 `best.pt`: aprendeu com as caixas corrigidas e é o v8 com mais acertos em fotos não vistas.

### Sugestões, fotos da internet e exportação

- [Página v8](../data/reviews/ceasa_v8_revisao002_ff34e226416c/index.html): 179 sugestões em 27 fotos (o atalho da raiz abre esta página).
- Fotos da internet: [`internet_v1_sugestoes_modelo`](../data/reviews/internet_v1_sugestoes_modelo/index.html) (originais congeladas + 19 sugestões em 5 fotos) e [`internet_v2_propostas_claude`](../data/reviews/internet_v2_propostas_claude/index.html) (as 29 caixas propostas pela IA + as mesmas sugestões). Nenhuma foto aprovada; elas só entram no treino depois da sua revisão. Classes sugeridas: [novas classes](novas_classes_propostas.md).
- ONNX: `runs/export_20261004T213131Z_e52142ab`, 640 px, **paridade 11/11 caixas, IoU 1,0, Δconfiança 0,0**.
- **Pesos ainda fora do repositório:** a rede da sessão na nuvem bloqueou `lfs.github.com`, então `best.pt`, `last.pt` e `model.onnx` desta rodada não foram enviados ao Git LFS. Metadados, métricas, gráficos e hashes estão versionados. Para reproduzir: `train-pilot --data data\pilot\ceasa_v8_piloto_s42\dataset.yaml` (mesmo seed; ~1 h no 4600G).
- Testes: **181 OK**, incluindo os 5 e2e reais (treino, predição, ONNX e sugestões).

### Próximos passos

1. Revisar `internet_v2_propostas_claude` (aceitar, corrigir ou descartar as caixas da IA) e exportar o arquivo de revisão: são 7 fotos de outras coberturas, o tipo de dado que mais falta.
2. Decidir quais classes sugeridas ativar.
3. Fotografar outras edificações reais; com 50–100 fotos por classe ([plano de coleta](plano_coleta_imagens.md)) a validação passa a medir algo.
4. Opcional: treino final com as 37 fotos (sem validação própria) só para gerar sugestões.
