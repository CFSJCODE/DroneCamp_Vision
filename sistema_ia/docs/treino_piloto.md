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
