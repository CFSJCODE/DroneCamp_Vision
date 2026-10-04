# DroneCamp — núcleo inicial de visão computacional

> **Publicação DroneCamp Vision — 04/10/2026:** já existe treinamento piloto real e a revisão atual é a [v7](data/reviews/ceasa_v7_revisao002_ba8cc323c8a1/index.html), aberta pelo atalho na raiz. Os resultados e limites estão em [Treino piloto](docs/treino_piloto.md); o modelo permanece com `production_ready: false`. As seções iniciais abaixo preservam o histórico de 03/10. Consulte também o [guia da raiz](../README.md) para clonagem com Git LFS e limites dos caminhos absolutos nos registros históricos.

Base local de **detecção assistida** para inspeção de telhados industriais. As categorias iniciais vêm do laudo CEASA; cinco categorias foram acrescentadas ou ativadas a pedido do usuário. O resultado são caixas candidatas, imagens demarcadas e metadados para revisão. Estado atual em 03/10/2026: **taxonomia v5 com 13 classes ativas; nenhum detector especializado treinado**.

**Revisão humana incorporada:** o arquivo fornecido pelo usuário preserva 33 fotos positivas com 108 caixas aprovadas para as dez classes da v3. A [página corrigida atual](data/reviews/ceasa_v6_corrigida_a36ea7f67d08/index.html) mantém essas correções e solicita a conferência das três categorias posteriores. A fila completa tem 38 fotos e 115 caixas, incluindo sete propostas das duas fotos sem decisão; três fotos continuam ambíguas. A aprovação histórica não libera treinamento nas 13 classes. Consulte o [resultado da importação](docs/resultado_revisao_humana_ceasa.md), o [plano de coleta](docs/plano_coleta_imagens.md) e o [protocolo de novas imagens e anomalias](docs/novas_imagens_e_anomalias.md).

**YOLO27l é o alvo futuro, ainda sem lançamento público na consulta desta data.** O modelo executável inicial é YOLO26l, selecionado em `configs/project.yaml`. Não há troca silenciosa: solicitar `yolo27l.pt` sem checkpoint disponível falha com explicação. Pesos gerais COCO não conhecem as 13 categorias do projeto; `--demo` identifica explicitamente testes de infraestrutura.

## Material entregue

- [Plataforma de operações YOLO](docs/plataforma_operacoes.md): a página de revisão com visão geral, revisão e classificação, treinamento e fine-tuning, monitoramento ao vivo e comparação de modelos. Abra pelo [Abrir plataforma DroneCamp.cmd](../Abrir%20plataforma%20DroneCamp.cmd) (servidor local) ou, offline, pelo [Abrir revisao DroneCamp.cmd](../Abrir%20revisao%20DroneCamp.cmd).
- [Análise documental e categorias](docs/analise_documental.md): pesquisa inicial do laudo, planos e apresentação, com oito classes ativas de origem documental; a taxonomia atual está em `configs/taxonomy.json`.
- [Pedaço de telha](docs/categoria_pedaco_telha.md): critério da nova classe ativa ID 8 e distinção entre fragmento, dano instalado e resíduos.
- [Rufo deslocado/desalinhado](docs/categoria_rufo_deslocado.md): critério da classe ativa ID 9, contexto de alinhamento e incerteza sobre a identidade do elemento.
- [Pedaço de telha sobreposto à telha](docs/categoria_pedaco_telha_sobreposto.md): classe ativa ID 10, preferida ao ID 8 quando o fragmento está apoiado sobre telha instalada.
- [Elemento de fixação de telha solto/frouxo](docs/categoria_fixador_telha_frouxo.md): classe ativa ID 11, com evidência visual de soltura e limites da interpretação de aperto.
- [Reparo em rufo](docs/categoria_reparo_rufo.md): classe ativa ID 12, com intervenção identificável no próprio rufo e distinção de reparo em telha.
- [Pesquisa Ultralytics](docs/pesquisa_ultralytics.md): todas as páginas solicitadas, exemplos, API atual e boas práticas.
- [Fine-tuning e preparação dos dados](docs/fine_tuning_e_anotacao.md): roteiro de anotações, splits e avaliação.
- [Plano de revisão e melhoria contínua](docs/plano_revisao_e_melhoria_continua.md): correções humanas, diversidade de dados, comparação de versões e evolução dos relatórios.
- [Execução da revisão do CEASA](docs/execucao_revisao_ceasa.md): histórico das propostas e fluxo de revisão.
- [Resultado da revisão humana](docs/resultado_revisao_humana_ceasa.md): correções incorporadas, cobertura por classe e próximo treinamento.
- [Página local de revisão corrigida — v6](data/reviews/ceasa_v6_corrigida_a36ea7f67d08/index.html): correções anteriores preservadas, 13 categorias e exportação das decisões.
- [Plano de coleta de imagens](docs/plano_coleta_imagens.md): metas por etapa, diversidade e avaliação independente.
- [Novas imagens e anomalias](docs/novas_imagens_e_anomalias.md): como incorporar exemplos e tratar defeitos inéditos.
- [Decisões do projeto](docs/decisoes.md): diretriz registrada de melhoria progressiva.
- [Validações desta entrega](docs/validacao.md): verificações executadas e limites reais.
- `configs/taxonomy.json`: catálogo com IDs, critérios, páginas e gravidade histórica.
- `configs/project.yaml`: modelo, dispositivo, inferência e hiperparâmetros de treino.
- `src/dronecamp_ia/`: funções separadas para configuração, validação, detecção, treino, avaliação, exportação e extração de fotos.

**Abertura da revisão corrigida:** abra `data/reviews/ceasa_v6_corrigida_a36ea7f67d08/index.html` pelo Explorer no seu navegador. O servidor temporário em `http://127.0.0.1:8765/` serve a v3; sua atualização automática foi bloqueada na etapa anterior, e o navegador integrado recusou arquivos locais. A nova tela corrigida não recebeu teste visual no navegador nesta execução. Os detalhes estão em [Validação](docs/validacao.md).

Na pasta principal do projeto, dê dois cliques em [Abrir revisao DroneCamp.cmd](../Abrir%20revisao%20DroneCamp.cmd) para acessar essa versão. O arquivo `src/dronecamp_ia/review_templates/index.html` é o modelo interno usado na geração; abri-lo diretamente exibe a estrutura vazia porque seus dados ainda não foram preenchidos. O atalho não importa decisões, não altera anotações e não inicia treinamento.

## Ambiente local

Foi criado `.venv` isolado com Python 3.12, PyTorch CPU e Ultralytics fixado em **8.4.172**. Nenhum Python global foi alterado. A GPU AMD identificada nesta máquina não é CUDA; o dispositivo inicial é `cpu`. Treinar YOLO26l em CPU pode ser demorado; medir em ambiente apropriado antes de definir orçamento.

No PowerShell, abra a pasta `sistema_ia` e use os comandos abaixo. Ativar o ambiente é opcional: o executável explícito evita confundir versões.

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia --help
.\.venv\Scripts\python.exe -m dronecamp_ia categories
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`tests/test_e2e_ultralytics.py` roda Ultralytics de verdade, sem mocks: monta o dataset piloto com as revisões reais, treina 1 época, faz `predict`, exporta ONNX conferindo a paridade e gera sugestões que, aceitas num feedback, chegam ao dataset seguinte. Leva cerca de 1–2 minutos na CPU; defina `DRONECAMP_SKIP_E2E=1` para pulá-lo.

Para recriar em outra máquina com Python 3.12 e `uv` disponível:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe -e ".[documents]"
```

O arquivo `requirements-lock.txt` registra versões desta execução. A reprodução exata de Torch CPU requer seu índice oficial, como acima; não trocar automaticamente por wheel CUDA. Instalação em GPU diferente é uma decisão própria de ambiente.

## Teste de detecção com as fotos do relatório

As imagens extraídas ficam em `data/reference/ceasa/`, com `manifest.json` (página, dimensões, hash e duplicatas) e `review_queue.csv` para revisão. Esta pasta **não é dataset de treino**. A extração preserva bytes incorporados no PDF; marcações gravadas nos pixels continuam presentes, e desenhos vetoriais da página podem ficar de fora.

Para extrair novamente, escolha uma pasta de saída nova:

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia extract-report --pdf "..\Documentação Do Projeto\CEASA Minas - Pavilão A - Laudo.pdf" --output "data\reference\ceasa_nova"
```

Escolha uma foto e execute a demonstração:

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia predict --source "data\reference\ceasa\p005_img01_80528144.jpg" --demo
```

O exemplo usa uma foto real extraída da p. 5. Também é possível apontar `--source` para uma pasta local ou vídeo. O primeiro uso baixa o checkpoint oficial YOLO26 para `models/`. Operações são locais; não há envio de fotos a serviços externos. Vídeos são processados frame a frame, sem consolidar defeitos físicos repetidos. As configurações operacionais e a opção `sync=False` ficam em `.runtime/` no projeto.

Cada execução cria uma pasta única em `runs/`, com:

- `evidence/`: imagens demarcadas.
- `findings.jsonl`: caixas em pixels, classe, confiança, fonte, índice processado e revisão pendente.
- `execution.json`: versão, modelo, hash, parâmetros e origem dos arquivos.
- `summary.json`: conclusão do processamento; não conta defeitos físicos únicos.

**`--demo` pode identificar carro, pessoa e outros objetos gerais, mas não demonstra reconhecimento de telhas quebradas, resíduos ou rufos.** Sem `--demo`, o sistema exige que nomes/IDs do checkpoint correspondam à taxonomia especializada. Nenhuma detecção nunca significa certificado de conformidade.

## Revisar e preparar dados aprovados

Na versão inicial com oito classes, as 38 imagens receberam duas leituras visuais por IA, resultando em 108 caixas propostas. Esse histórico permanece preservado em `data/reviews/ceasa_v1/`. As 48 ocorrências do PDF continuam ligadas aos 38 conteúdos únicos.

A revisão v3 acrescentou um candidato às 108 caixas históricas. A v5 preservou essas 109 propostas e ampliou a taxonomia para 13 classes. Após a importação do arquivo humano da v3, a revisão atual está em `data/reviews/ceasa_v6_corrigida_a36ea7f67d08/registry.json`: **108 caixas humanas históricas em 33 fotos**, mais sete propostas das duas fotos sem decisão; total da fila **115 caixas, 35 fotos positivas na triagem e três ambíguas**. A aprovação anterior cobriu dez classes. Agora, 33 fotos pedem conferência complementar das classes 10–12; cinco ainda precisam de revisão completa ou exclusão justificada. Há dois fragmentos ID 8 e um rufo deslocado ID 9 na revisão humana histórica; IDs 10, 11 e 12 continuam sem exemplos. A migração preserva os rótulos e não converte automaticamente reparos ou fragmentos.

A 2ª rodada da `revisao002` já foi importada em `data/reviews/ceasa_v8_revisao002_ff34e226416c/` (37 fotos aprovadas, 1 ambígua, 246 caixas; 155 após consolidar duplicatas no treino). As fotos da internet têm revisão própria em `data/reviews/internet_v2_propostas_claude/`, com caixas propostas pela IA. Abra [a página v8](data/reviews/ceasa_v8_revisao002_ff34e226416c/index.html), que também mostra as sugestões do modelo piloto, confira a foto inteira, corrija caixas/classes e registre a decisão. **Salvar arquivo de revisão** baixa o JSON das suas decisões. Editar uma caixa invalida a confirmação anterior. A página distingue correções humanas históricas das propostas e da aprovação atual pendente. Não são resultados de um detector YOLO especializado. Consulte o [resultado da importação e próximo treinamento](docs/resultado_revisao_humana_ceasa.md).

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia import-review --registry "data\reviews\ceasa_v6_corrigida_a36ea7f67d08\registry.json" --feedback "dronecamp-revisao-ceasa_v6_corrigida_a36ea7f67d08.json" --output "data\reviews\ceasa_v7_humana\registry.json"
.\.venv\Scripts\python.exe -m dronecamp_ia build-reviewed-data --registry "data\reviews\ceasa_v7_humana\registry.json" --assignments "configs\review_groups.json" --output "data\versions\ceasa_v7_humana"
```

O primeiro comando exige o arquivo exportado pela página corrigida e preserva o registro anterior. O segundo gera uma versão nova com fotos, labels, grupos, proveniência e prontidão. Se a saída já existir, escolha outro nome de versão. **Sem aprovação compatível com a taxonomia atual, nenhum dataset é criado.** O acervo humano histórico em `data/annotation_corpus/ceasa_v3_a36ea7f67d08/` é uma referência de dez classes, sem YAML de treino. O arquivo de grupos fornecido reserva todo o CEASA para treino. Depois da revisão, esse único caso pode gerar um rascunho, com retorno de erro explicando a falta de prontidão. Validação e teste dependem de outras edificações reais.

Os IDs ativos originais 0–7 mantiveram seus significados; as adições são 8 (`pedaco_telha`), 9 (`rufo_deslocado`), 10 (`pedaco_telha_sobreposto`) e 12 (`reparo_rufo`); ID 11 (`fixador_telha_frouxo`) foi promovido da fase 2, preservando seu significado. Os antigos IDs 8, 9 e 10 do catálogo de fase 2 foram deslocados para 19, 20 e 21, respectivamente. O antigo ID 12 de `reparo_selante_fixador_telha` passa ao ID 22. O catálogo atual tem 23 categorias, com 13 ativas. Snapshots das taxonomias anteriores e metadados de migração preservam essa distinção; não reinterpretar um ID antigo sem sua versão. Acrescentar nomes não ensina o modelo: reconhecimento das 13 classes exige dados aprovados, novo treino e comparação.

## Treino piloto, sugestões e ciclo de melhoria

Enquanto só houver fotos do CEASA, o treino de produção continua bloqueado (exige três edificações e exemplos das 13 classes). O **piloto** treina de verdade com as fotos que têm aprovação humana, para o modelo passar a **sugerir caixas** na página de revisão. Detalhes, resultados e limites: [treino piloto](docs/treino_piloto.md).

```powershell
# 1. Importar o arquivo exportado pela página (gera uma nova versão da revisão)
.\.venv\Scripts\python.exe -m dronecamp_ia import-review --registry "data\reviews\ceasa_v8_revisao002_ff34e226416c\registry.json" --feedback "SEU_ARQUIVO.json" --output "data\reviews\ceasa_v9_ID\registry.json"
# 2. Dataset piloto a partir das revisões humanas (repita --registry para somar revisões)
.\.venv\Scripts\python.exe -m dronecamp_ia build-pilot-data --registry "data\reviews\ceasa_v9_ID\registry.json" --output "data\pilot\ceasa_v9_piloto_s42"
# 3. Treino real (YOLO26l na CPU; parâmetros na seção pilot de configs/project.yaml)
.\.venv\Scripts\python.exe -m dronecamp_ia train-pilot --data "data\pilot\ceasa_v9_piloto_s42\dataset.yaml"
# 4. Comparar o modelo novo com o anterior nas mesmas fotos
.\.venv\Scripts\python.exe scripts\compare_pilot_models.py --data "data\pilot\ceasa_v9_piloto_s42\dataset.yaml" --weights "runs\PILOTO_ANTERIOR\fit\weights\best.pt" --weights "runs\PILOTO_NOVO\fit\weights\best.pt" --output "runs\comparacao_v9.json"
# 5. Sugestões na revisão existente ou em fotos novas de outra edificação
.\.venv\Scripts\python.exe -m dronecamp_ia suggest --weights "runs\SEU_PILOTO\fit\weights\best.pt" --registry "data\reviews\SUA_REVISAO\registry.json"
.\.venv\Scripts\python.exe -m dronecamp_ia suggest --weights "runs\SEU_PILOTO\fit\weights\best.pt" --source "PASTA_FOTOS_NOVAS" --output "data\reviews\galpao_b_v1" --group "galpao_b"
# 6. Opcional: caixas propostas por uma leitura visual de IA, como candidatas (nunca aprovação)
.\.venv\Scripts\python.exe -m dronecamp_ia add-ai-proposals --registry "data\reviews\galpao_b_v1\registry.json" --proposals "data\proposals_ai\SUAS_PROPOSTAS.json" --output "data\reviews\galpao_b_v2_propostas\registry.json"
```

- O piloto separa treino, validação e teste por **cena** (`scene_group`): recortes da mesma área ficam no mesmo split. Classes com uma única foto ficam só no treino e não são medidas.
- **Duplicatas:** caixas da mesma classe com IoU ≥ `pilot.duplicate_iou` (0,7) viram um único alvo no dataset piloto; vale a caixa desenhada/editada pelo revisor, depois a sugestão aceita de maior confiança. O registro humano não muda; `pilot.json` registra a política e as contagens.
- `summary.json` do piloto registra `pilot: true` e `production_ready: false`; `predict` marca os achados como `candidatos_modelo_piloto_nao_validado`.
- Na página, as sugestões aparecem tracejadas com a confiança. **Aceitar** copia a caixa para a sua revisão e exige nova confirmação da foto; **Descartar** a esconde. Se a sugestão cobre uma caixa que já existe (mesma classe, IoU ≥ 0,7), o botão vira **Substituir caixa N** e troca a geometria em vez de duplicar. Só o arquivo de revisão exportado e importado com `import-review` entra no próximo dataset.
- Propostas visuais de IA (`add-ai-proposals`) usam a classe ativa mais próxima e podem sugerir uma classe nova em `proposed_new_class`; a página mostra “nova classe sugerida”. Ver [novas classes propostas](docs/novas_classes_propostas.md).
- Caminhos absolutos gravados em outra máquina (ex.: `E:\...\sistema_ia\...`) são reancorados neste clone pela pasta `sistema_ia`, sempre com conferência do SHA-256. O `dataset.yaml` do piloto usa `path: .`, então o dataset funciona em qualquer pasta.
- Ciclo: revisar → `import-review` → `build-pilot-data` com todas as revisões → `train-pilot` → comparar → `suggest`. Fotos novas de outras edificações (`--group`) também abrem caminho para o dataset de produção.

## Treinar e comparar

Uma versão pronta precisa de revisão humana, exemplos positivos das **13 classes** no treino e grupos independentes nos três splits. A pasta segue esta estrutura:

```text
data/versions/SUA_VERSAO/
  groups.csv
  dataset.yaml   provenance.json   readiness.json
  images/train/   images/val/   images/test/
  labels/train/   labels/val/   labels/test/
```

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia validate-data --data "data\versions\SUA_VERSAO\dataset.yaml" --output "runs\dataset_review.json"
.\.venv\Scripts\python.exe -m dronecamp_ia train --data "data\versions\SUA_VERSAO\dataset.yaml"
```

O dataset vazio entregue é reprovado de propósito. `validate-data` verifica estrutura e não baixa conteúdo nem executa `download` do YAML. Antes de carregar pesos, treino, tuning e avaliação também reconciliam a proveniência com a revisão humana: autores, classes, caixas, hashes, inventário completo e grupos. A prontidão é recalculada; editar apenas `readiness.json` não libera dados. O fluxo cria YAML resolvido com caminhos absolutos e salva a verificação na execução.

Labels com BOM, imagens menores que 10 pixels, JPEG sem marcador final e caminhos de label incompatíveis com o loader são reprovados sem reparar originais. A revisão recusa rotação/espelhamento por EXIF até que uma cópia normalizada, com novo hash, seja preparada e anotada. Os parâmetros iniciais (`AdamW`, taxa 0.001, 100 épocas, imagem 1024, batch 2) são uma proposta editável, **não um resultado otimizado ou testado no acervo**. A validação que seleciona checkpoints e o tuning usam a mesma cabeça `nms=False` escolhida para inferência.

Depois do treino, use o caminho real de `best.pt` em:

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia evaluate --data "data\versions\SUA_VERSAO\dataset.yaml" --weights "runs\SEU_TREINO\fit\weights\best.pt" --split val
.\.venv\Scripts\python.exe -m dronecamp_ia predict --weights "runs\SEU_TREINO\fit\weights\best.pt" --source "SUA_FOTO.jpg"
```

Tuning é uma busca com vários treinos. Execute após a primeira avaliação; o número de tentativas e épocas deve ser explícito:

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia tune --data "data\versions\SUA_VERSAO\dataset.yaml" --iterations 10 --epochs 30
```

O comando verifica o histórico NDJSON, métricas e artefatos depois da busca. Se as tentativas falharem ou faltarem resultados, registra o resumo e retorna erro. Fitness zero com métricas válidas é uma execução válida, sem significar que o modelo foi aprovado.

Reserve `evaluate --split test` para a avaliação final, depois de escolher parâmetros com `val`. Este comando não impede o pesquisador de repetir o teste; a disciplina experimental continua necessária.

## Exportação

ONNX FP32 tem função própria e depende do extra de exportação:

```powershell
uv pip install --python .venv\Scripts\python.exe -e ".[export]"
.\.venv\Scripts\python.exe -m dronecamp_ia export --weights "runs\SEU_TREINO\fit\weights\best.pt"
```

O exportador trabalha sobre cópia do checkpoint e registra hash do artefato. Em seguida roda o `.pt` e o `.onnx` (ONNX Runtime) nas mesmas fotos — padrão: 5 fotos de `data/reference/ceasa`, ou `--parity-source PASTA` — e grava `parity.json`: caixas pareadas por classe com IoU ≥ 0,9 (ou até 1 px), diferença de confiança ≤ 0,02. `runtime_parity_validated` só é verdadeiro se nenhuma caixa clara faltar ou sobrar. Qualidade e tempo no hardware de destino continuam sendo etapas próprias. TensorRT, quantização INT8, segmentação, classificação ordinal/CORAL, regras normativas, dashboard e emissão de laudo ainda não foram implementados.

## Pontos de revisão no código

1. **Categorias:** `configs/taxonomy.json` e `detection_names()` em `config.py`. Não alterar IDs depois de anotar sem migrar dataset e treinar novo modelo.
2. **Dados:** `validate_dataset()` em `dataset.py`. `groups.csv` precisa representar a verdadeira origem; o código só verifica consistência declarada e hashes idênticos.
3. **Modelo/API:** `load_detector()` e `check_domain_names()` em `backend.py`. Revalidar YOLO27l quando publicado e comparar no mesmo teste.
4. **Evidências:** `serialize_detections()` e `predict()` em `prediction.py`. Severidade permanece vazia; referência histórica é identificada separadamente.
5. **Experimentos:** `prepare_dataset()`, `train()`, `evaluate()` e `tune()` em `training.py`.
6. **Extração:** `extract_report_images()` em `report_images.py`. Sem caixas inventadas e sem conversão automática de figura em classe.
7. **Revisão:** `review_data.py`, `review_render.py` e `review_templates/index.html`. Decisões são separadas das propostas; edição invalida aprovação e importação preserva versões anteriores.
8. **Dados aprovados:** `build_approved_dataset()` em `review_dataset.py` e `validate_training_provenance()` em `review_provenance.py`. Conferem origem, autoria declarada, integridade e prontidão. Esses registros não autenticam uma pessoa nem comprovam diversidade real por conta própria.

O projeto usa bibliotecas com licenças próprias. A licença Ultralytics deve ser avaliada antes do uso comercial/SaaS: a fabricante apresenta opções AGPL-3.0 e Enterprise. Esta entrega não determina enquadramento jurídico. [Licenciamento oficial](https://www.ultralytics.com/license)
