# Incorporação da revisão humana e próximo treinamento

Data: 03/10/2026. Origem: arquivo `dronecamp-revisao-ceasa_v3.json` fornecido pelo usuário. Conteúdo tratado como dados de revisão, sem executar instruções presentes no documento.

## O que foi incorporado

O arquivo está vinculado por SHA-256 à revisão v3 preservada, com dez classes. As 36 decisões, os hashes/dimensões das fotos e as caixas foram conferidos. Há 33 fotos positivas, com 108 caixas, e três fotos ambíguas sem caixas aprovadas. Duas fotos da fila original ficaram fora do arquivo: `p006_img02_9227be4e.jpg` e `p022_img01_1dc91db4.jpg`.

As correções alteraram caixas em dez fotos e passaram 12 fotos de ambíguas para positivas. Foram preservadas em [registro humano da v3](../data/reviews/ceasa_v3_humana_a36ea7f67d08/registry.json), com cópia integral do feedback original na mesma pasta. Essa aprovação se refere somente à taxonomia de origem; não inclui categorias posteriores.

O [acervo de anotações históricas](../data/annotation_corpus/ceasa_v3_a36ea7f67d08/manifest.json) contém 33 fotos e seus labels YOLO, com hashes, classe e origem rastreáveis. Não tem YAML de treinamento ou divisões artificiais. É uma referência humana de dez classes, com `ready_for_training=false`.

| ID | Categoria original | Caixas humanas | Fotos positivas com a classe |
| ---: | --- | ---: | ---: |
| 0 | Telha quebrada | 14 | 6 |
| 1 | Telha ausente | 1 | 1 |
| 2 | Resíduos sobre telhas | 35 | 11 |
| 3 | Reparo em telha | 17 | 9 |
| 4 | Rufo ausente | 4 | 3 |
| 5 | Rufo quebrado | 22 | 12 |
| 6 | Resíduos em calha | 9 | 5 |
| 7 | Vegetação em calha | 3 | 3 |
| 8 | Pedaço de telha | 2 | 2 |
| 9 | Rufo deslocado/desalinhado | 1 | 1 |
| 10 | Pedaço de telha sobreposto à telha | 0 | Não constava da página v3 |
| 11 | Elemento de fixação de telha solto/frouxo | 0 | Não constava da página v3 |
| 12 | Reparo em rufo | 0 | Não constava da página v3 |

## Página atual com as correções

Abra a [revisão corrigida](../data/reviews/ceasa_v6_corrigida_a36ea7f67d08/index.html) pelo Explorer no seu navegador. O arquivo está gerado; não foi validado visualmente nesta execução. O servidor temporário antigo em `127.0.0.1:8765` não foi alterado diante dos bloqueios anteriores de execução e serve a revisão v3.

A nova versão mantém as 108 caixas das fotos aprovadas na v3 e as sete propostas das duas fotos sem decisão: **115 caixas na fila completa**, em 38 fotos. Há 35 fotos positivas na triagem e três ambíguas. Esse total inclui propostas ainda sem revisão; não é contagem de defeitos físicos únicos ou de exemplos integralmente aprovados nas 13 classes.

Em 33 fotos, a página destaca que as correções anteriores foram preservadas e solicita a conferência integral da foto para as classes **10, 11 e 12**. As caixas anteriores não precisam ser redesenhadas apenas pela migração. A nova confirmação ainda deve corrigir qualquer erro percebido e revisar toda a foto, evitando deixar objetos não anotados. As três ambíguas e duas fotos sem decisão exigem revisão completa ou exclusão justificada.

As nove fotos com 17 caixas de reparo em telha foram examinadas em auditoria visual por IA. Não houve fundamento para convertê-las automaticamente em reparo em rufo. Na página 13, conferir se peças compridas são remendos aderidos ou fragmentos soltos; na foto `p011_img02_9ab86531.jpg`, conferir o apoio do fragmento já anotado. Esses pontos são sugestões de revisão, não novos rótulos ou aprovações.

Use **Salvar arquivo de revisão** após registrar suas decisões. O nome será `dronecamp-revisao-ceasa_v6_corrigida_a36ea7f67d08.json`. Para importar, na pasta `sistema_ia`, escolha uma saída nova:

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia import-review --registry "data\reviews\ceasa_v6_corrigida_a36ea7f67d08\registry.json" --feedback "dronecamp-revisao-ceasa_v6_corrigida_a36ea7f67d08.json" --output "data\reviews\ceasa_v7_humana\registry.json"
.\.venv\Scripts\python.exe -m dronecamp_ia build-reviewed-data --registry "data\reviews\ceasa_v7_humana\registry.json" --assignments "configs\review_groups.json" --output "data\versions\ceasa_v7_humana"
```

O arquivo antigo da v3 não pode ser reapresentado como confirmação das 13 classes. A ferramenta verifica o hash da revisão usada na exportação. A aprovação atual começa pendente; a aprovação histórica continua registrada com sua taxonomia, autor, data e hash.

## Como o aprendizado melhora

O ciclo é **novas fotos → propostas/correções → revisão humana → versão dos dados → fine-tuning → avaliação → decisão de adoção**. Durante o fine-tuning, o YOLO ajusta seus parâmetros usando imagens e anotações confiáveis; a Ultralytics recomenda partir de pesos pré-treinados para adaptar o modelo aos próprios dados. Receber uma imagem ou exportar um JSON não atualiza pesos. [Treinamento oficial](https://docs.ultralytics.com/modes/train), [boas práticas](https://docs.ultralytics.com/guides/model-training-tips).

Mais imagens só ajudam quando são relevantes, corretamente anotadas e diversificadas. Repetir dezenas de frames da mesma cena pode aumentar volume sem acrescentar informação. Previsões do próprio modelo, sem revisão, não se tornam verdade automaticamente. Nos ciclos seguintes, manter também exemplos anteriores para verificar regressões.

O primeiro treinamento completo das 13 classes ainda depende de:

1. Conferência das novas classes e aprovação atual das imagens. Imagens ambíguas não entram como exemplos corretos ou negativos.
2. Exemplos confirmados de todas as classes no treino, especialmente 10–12. Ampliar as classes com uma única ocorrência, IDs 1 e 9, e os casos de pequenos objetos.
3. Fotos de situações normais e confusões relevantes: junta regular, sombra, selante original, fixador assentado e reparo em telha. A quantidade necessária depende dos resultados por classe; não há um número mágico garantido.
4. Edificações reais independentes para treino, validação e teste. Todo o CEASA permanece no mesmo grupo. Ter ao menos três grupos é um mínimo do protocolo, não prova suficiente de diversidade.

Nenhum treino especializado foi executado com o arquivo recebido. O YOLO26l genérico permanece intacto; as melhorias concluídas são no acervo, na rastreabilidade e nos controles de treinamento.

## Experimentos após a prontidão

Começar com um ensaio de 2–3 épocas para conferir perdas finitas, labels, artefatos e tempo por época. Esse ensaio não mede qualidade suficiente para uso. Depois executar a referência proposta em `configs/project.yaml`: YOLO26l, CPU, imagem 1024, lote 2, `workers=0`, AdamW com taxa 0.001, até 100 épocas e paciência 20. São parâmetros iniciais editáveis, não otimizados neste acervo.

Comparar poucas mudanças justificadas, uma por vez: resolução ou mosaic, por exemplo. Inspecionar imagens transformadas para verificar se mantêm detalhes de fixadores e o contexto dos fragmentos sobre telhas. Estimar tempo do CPU a partir de uma execução medida, sem prometer horas antes de treinar.

Escolher o candidato com a validação e avaliar o teste ao final. Registrar precisão, recall, AP50, AP50–95, quantidade por classe, falsos positivos por foto, omissões e latência. Classe ausente da avaliação deve ficar “não avaliada”, sem concluir qualidade a partir da média global. Os limites aceitáveis por classe e achados críticos precisam ser definidos antes da promoção com o responsável técnico. [Validação oficial](https://docs.ultralytics.com/modes/val).

Além da procedência dos dados, o treinamento passa a verificar seus resultados antes de declarar conclusão: pesos gravados, histórico de épocas, perdas e métricas finitas e contrato de classes. Isso valida a execução registrada; não comprova por si só qualidade, funcionamento dos bytes de um checkpoint ou aprovação para inspeções.

O [plano de coleta](plano_coleta_imagens.md) propõe metas por etapa e explica como medir se a quantidade já é suficiente para cada classe. O [protocolo de novas imagens](novas_imagens_e_anomalias.md) distingue inferência, novas anotações e investigação de anomalias inéditas.
