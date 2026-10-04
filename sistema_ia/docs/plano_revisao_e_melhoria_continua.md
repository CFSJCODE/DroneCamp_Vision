# Plano de revisão e melhoria contínua do DroneCamp

Data: 03/10/2026. Estado: **13 classes ativas; 33 fotos com correções humanas históricas incorporadas, conferência complementar das classes posteriores e treinamento especializado pendentes**. Resultado atual e comandos em [resultado_revisao_humana_ceasa.md](resultado_revisao_humana_ceasa.md). Metas propostas de coleta em [plano_coleta_imagens.md](plano_coleta_imagens.md).

## 1. Objetivo e limite atual

Construir um ciclo em que novas imagens e correções humanas melhorem, de forma mensurável, as detecções e a qualidade dos relatórios. O aprendizado acontecerá em novos treinamentos com datasets revisados; receber imagens ou realizar inferência, isoladamente, não atualiza o modelo.

A entrega fornece configuração, taxonomia, extração de fotos, inferência, validação e comandos de treinamento/avaliação, além de revisão gráfica, importação de correções e preparação de dados versionados. Foram incorporadas 36 decisões do arquivo humano da v3: 33 fotos positivas, 108 caixas e três fotos ambíguas; duas fotos ficaram fora do arquivo. Há um acervo humano histórico de dez classes, mas ainda não há dataset pronto nas 13 classes, modelo especializado promovido ou relatório técnico final.

O primeiro acervo tem 48 ocorrências de imagens no PDF e 38 conteúdos binários únicos. Elas representam um único caso CEASA, com possíveis cenas semelhantes, recortes e marcações. Duplicatas exatas foram identificadas; a identificação de cenas quase iguais ainda depende de revisão visual.

A rodada original de duas leituras por IA cobriu as oito classes iniciais e permanece preservada em `data/reviews/ceasa_v1/`. A v3 acrescentou um candidato de rufo deslocado; a v4 acrescentou ID 10 e ativou ID 11; a v5 acrescentou ID 12, Reparo em rufo. A importação humana foi preservada separadamente. A revisão atual está em `data/reviews/ceasa_v6_corrigida_a36ea7f67d08/`, com as 108 caixas humanas das 33 fotos e sete propostas das duas fotos sem decisão: 115 caixas na fila completa. Nenhuma migração converteu rótulos automaticamente ou inventou aprovação das classes posteriores. Nenhuma das 38 fotos foi liberada como integralmente revisada para as 13 classes.

## 2. Como analisar os resultados já existentes

Começar por comparar a fotografia original com a imagem em `evidence/`, depois ler `findings.jsonl`:

| Caso | Original e resultado | Leitura da revisão |
| --- | --- | --- |
| Página 30 / figura 41 | `data/reference/ceasa/p030_img01_50f69e0c.jpg`; resultado em `runs/predict_demo_20261003T232315Z_c8b62618/` | O canal da calha contém detritos visíveis. O modelo genérico marcou a cabine do caminhão como `oven` e `microwave`; rejeitar essas previsões como evidência das categorias do projeto. |
| Página 5 / figura 5 | `data/reference/ceasa/p005_img01_80528144.jpg`; resultado em `runs/predict_demo_20261003T232634Z_f4e86d45/` | O laudo documenta telhas quebradas, mas a demo não desenhou caixas. O resultado não certifica conformidade; os pesos genéricos não foram treinados nas classes DroneCamp. |

`confidence` é a pontuação do detector, não gravidade nem comprovação de diagnóstico. `review_status=pendente` indica ausência de decisão humana registrada. `severity=null` preserva a avaliação técnica. `complete=true` em `summary.json` significa término do processamento, não aprovação das detecções. Não há precisão/recall do domínio calculáveis sem as anotações de referência.

## 3. Primeira rodada: estabelecer convenções com um piloto

Revalidar um conjunto pequeno representativo das 13 classes: telha quebrada, telha ausente, resíduos sobre telha, reparo de telha, rufo ausente, rufo quebrado, resíduos em calha, vegetação em calha, pedaço de telha, rufo deslocado/desalinhado, pedaço de telha sobreposto à telha, elemento de fixação de telha solto/frouxo e reparo em rufo. A taxonomia atual está em `configs/taxonomy.json`.

As classes novas atendem aos pedidos e exemplos discutidos com o usuário em 03/10/2026. [Pedaço de telha](categoria_pedaco_telha.md), ID 8, descreve um fragmento solto reconhecível; ainda não há exemplos propostos dessa classe no registro atual. [Rufo deslocado/desalinhado](categoria_rufo_deslocado.md), ID 9, descreve desalinhamento observável em relação ao elemento adjacente; há um candidato na extremidade esquerda da foto da página 8, com identidade do elemento ainda não confirmada. [Pedaço de telha sobreposto à telha](categoria_pedaco_telha_sobreposto.md), ID 10, distingue o fragmento solto apoiado sobre telha instalada e é preferido ao ID 8 nesse contexto. [Elemento de fixação de telha solto/frouxo](categoria_fixador_telha_frouxo.md), ID 11, exige evidência visual de soltura, deslocamento ou folga e vínculo com a telha. [Reparo em rufo](categoria_reparo_rufo.md), ID 12, exige intervenção identificável no próprio rufo, sem confundir selante original ou reparo em telha. IDs 8, 10, 11 e 12 têm zero caixas propostas no registro atual. A criação ou ativação de categorias não aprova ocorrências nem ensina o detector.

Comparar deliberadamente situações que confundem a anotação: quebra versus junta/sombra; ausência versus abertura prevista; reparo versus detrito solto; resíduo sobre telha versus dentro da calha; vegetação na calha versus no entorno; fragmento de telha reconhecível versus mistura de detritos; rufo deslocado versus junta normal ou ruptura; fragmento sobre telha versus sobreposição normal ou remendo aderido; fixador frouxo versus fixador assentado, sombra ou objeto avulso sem vínculo; reparo em rufo versus reparo em telha, selante regular de montagem ou sujeira. Reparos e danos devem ter evidências próprias antes de coexistirem em caixas, sem duplicar a mesma caixa idêntica por inferência. A seleção histórica em `piloto_revisao.csv` continua sendo referência das oito classes originais e precisa ser complementada para as cinco classes adicionadas/ativadas; seus nomes sugeridos não são anotações aprovadas.

Para cada imagem, o anotador deverá:

1. Conferir origem, nitidez, oclusões e marcações incorporadas.
2. Localizar todas as ocorrências visíveis das classes ativas, admitindo várias classes na mesma foto.
3. Desenhar caixas com a menor área que contém a evidência e contexto suficiente para distingui-la de uma junta ou sombra.
4. Registrar dúvidas, elementos fora das 13 classes e informações desconhecidas, sem inferir fatos ausentes.
5. Solicitar segunda revisão do piloto e das ambiguidades antes de consolidar as convenções.

### Exemplo discutido: detritos dentro de uma calha

O recorte mostrado pelo usuário corresponde à foto `p030_img01_50f69e0c.jpg`, figura 41, página 30. O laudo identifica fragmentos de fibrocimento dentro da calha. A categoria visual proposta é `residuos_calha`, ID 6.

Anotar o acúmulo de detritos dentro do canal. Evitar incluir a cabine do caminhão, o piso e resíduos que estejam sobre telhas em vez de dentro da calha. Fixar no piloto se cada acúmulo contíguo será uma ocorrência; agrupamentos separados precisam de uma convenção consistente.

As classes `telha_quebrada` e `rufo_quebrado` descrevem dano no elemento instalado; não devem ser atribuídas a cada fragmento solto apenas pela sua origem provável. Na taxonomia v5, um fragmento de telha reconhecível recebe ID 10 quando seu apoio sobre telha instalada estiver confirmado; recebe ID 8 em outra superfície ou sem esse contexto confirmado. O mesmo objeto não recebe as duas classes. IDs 2 e 6 continuam descrevendo acúmulos de resíduos e seu contexto. Não duplicar a mesma caixa do mesmo objeto como fragmento e resíduo; uma caixa do acúmulo e outra menor de um fragmento identificável exigem evidências e escalas distintas, registradas nas observações.

Na execução foram propostas quatro caixas de resíduos nessa foto, além de candidatos a dano no elemento instalado; todas aguardam revisão humana. A observação visual confirma a pertinência da categoria, mas não mede vazão, percentual de obstrução, quantidade física de material ou gravidade de uma nova inspeção. O material é descrito pelo laudo; sua composição não foi verificada fisicamente nesta execução.

## 4. Segunda rodada: revisar todo o acervo disponível

Após aprovar as convenções do piloto, completar a revisão dos 38 arquivos para as 13 classes. Manter ligação com todas as páginas em que uma imagem se repete e agrupar recortes/cenas semelhantes. Nas 33 fotos aprovadas na taxonomia de dez classes, preservar as correções e conferir a foto inteira para as três classes posteriores, corrigindo qualquer erro percebido. Nas três ambíguas e nas duas sem decisão, realizar revisão completa ou exclusão justificada. Fotos normais só se tornam negativas depois de conferir todas as classes atuais.

Cada imagem deverá terminar com uma decisão explícita:

- **Aprovada com ocorrências:** caixas e classes revisadas.
- **Negativa revisada:** imagem examinada integralmente sem nenhuma classe ativa; somente então criar `.txt` vazio.
- **Ambígua:** dúvida documentada e pendente de decisão técnica; excluída do treino enquanto não resolvida.
- **Inadequada/excluída:** baixa qualidade, marcação que prejudica a aprendizagem ou outra razão documentada.

Uma imagem com item fora da taxonomia pode ser negativa para as classes ativas apenas após revisão completa; registrar esse item nos metadados. O título da página não define automaticamente todos os objetos da fotografia.

## 5. Registro necessário para cada revisão

| Campo | Regra |
| --- | --- |
| Imagem / identificador / hash | Preservar vínculo com o arquivo de origem e a página do laudo. |
| Edificação / campanha / grupo de cena | Preencher somente quando conhecido; recortes e frames relacionados ficam juntos. |
| Material / condições de captura | Informação confirmada ou `desconhecido`; não preencher por suposição. |
| Qualidade / oclusão / marcações | Indicar fatores que tornam a anotação confiável ou inconclusiva. |
| Objetos | Classe, caixa e observações por ocorrência, sem copiar rótulos genéricos da demo. |
| Decisão da imagem | Aprovada, negativa revisada, ambígua ou excluída, com motivo. |
| Anotador / revisor / data | Usar identificadores internos, evitando dados pessoais desnecessários. |
| Versão das regras e revisão | Permitir rastrear mudanças de taxonomia e critérios. |
| Severidade e recomendação | Separadas da classe visual; pendentes de validação técnica. |

`review_queue.csv` organiza arquivos e páginas. A [interface corrigida atual](../data/reviews/ceasa_v6_corrigida_a36ea7f67d08/index.html) permite ajustar caixas/classes e exportar decisões. O [registro atual](../data/reviews/ceasa_v6_corrigida_a36ea7f67d08/registry.json) preserva origem, autores, correções humanas históricas, escopo da taxonomia aprovada e necessidade de conferência complementar. Aprovação atual exige decisão explícita sobre a foto inteira e importação vinculada ao hash da versão. Nenhum desses passos treina o modelo automaticamente.

Os IDs ativos 0–11 foram preservados na v5, que acrescenta `reparo_rufo`, ID 12. Na v4, ID 10 passou a `pedaco_telha_sobreposto` e ID 11 de `fixador_telha_frouxo` manteve seu significado ao passar à fase ativa. As categorias antes catalogadas nos IDs 8, 9 e 10 da fase posterior passaram a 19, 20 e 21, respectivamente. O antigo ID 12 de `reparo_selante_fixador_telha` passa ao ID 22. Os snapshots das taxonomias anteriores e os metadados de migração do registro atual permitem interpretar cada ID conforme a versão de origem. Não reutilizar pesos ou anotações antigas como se já representassem as 13 classes.

## 6. Diversificar os próximos dados

Buscar novas edificações, campanhas e condições representativas do uso real: materiais de cobertura, ângulos, alturas, distâncias, iluminação, sombras, envelhecimento, reparos, tamanhos de defeito, positivos e negativos revisados.

Priorizar lacunas de classe e erros recorrentes, em vez de apenas acumular frames parecidos. Registrar somente características conhecidas. Aumento artificial de dados complementa a diversidade real; não substitui novos locais nem corrige rótulos errados.

Todas as fotos deste CEASA pertencem ao mesmo grupo de edificação para avaliar generalização a outros locais. Não dividir aleatoriamente suas figuras para alegar desempenho independente. Se o conjunto continuar limitado a esse caso, qualquer treino deverá ser declarado como exploração do caso, sem conclusão de generalização.

## 7. Versionar dados, treinar e comparar

1. Completar a conferência na página corrigida e importar sua exportação para uma versão nova, por exemplo `data/reviews/ceasa_v7_humana/registry.json`. Gerar dados versionados apenas com revisões aprovadas para a taxonomia atual, preservando imagens e anotações anteriores. O fluxo e os comandos estão no [resultado da importação](resultado_revisao_humana_ceasa.md).
2. Registrar grupos, classes, hashes, origem e contagem por classe. Validar com `validate-data`.
3. Separar treino e validação por unidades reais de captura e reservar teste independente. O protocolo atual exige os três splits e exemplos positivos das 13 classes no treino. `configs/review_groups.json` reserva todo o CEASA para `train`; validação e teste dependem de outras edificações reais.
4. Treinar o primeiro detector especializado e usá-lo como referência inicial. As demos COCO não constituem uma referência de qualidade das 13 classes. O fine-tuning ajusta pesos pré-treinados às categorias do projeto usando exemplos aprovados; as classes novas exigem um novo ciclo de treino e avaliação.
5. Nos ciclos seguintes, testar um candidato com dados novos e retenção de exemplos anteriores, para verificar se a melhoria não sacrificou classes já aprendidas.
6. Comparar precisão, recall e mAP por classe, além de falsos positivos, omissões, objetos pequenos, memória e tempo no ambiente de destino.
7. Selecionar parâmetros usando validação; avaliar o teste após fechar a decisão. Se o teste orientar novas correções, ele deixa de ser independente para aquela decisão e precisa ser renovado/separado adequadamente.
8. Adotar o candidato somente depois da revisão dos erros e critérios de aceite, mantendo a versão anterior recuperável.

Não aprovar automaticamente um modelo porque a média global subiu. A equipe deverá definir quais omissões são críticas e os limites aceitáveis por classe. Percentuais mínimos, prazos e orçamento de treinamento ainda não foram definidos.

## 8. Evolução dos relatórios

Primeiro estruturar relatórios de revisão com fotografia, localização em pixels, classe candidata, confiança, decisão humana, fonte e versão do modelo. Não transformar caixas em metros ou m² sem calibração/escala apropriadas nem contar a mesma ocorrência repetida em frames como defeitos distintos.

Depois acrescentar gravidade, prioridade e recomendação aprovadas pelo responsável técnico, com regras explícitas e fontes aplicáveis. Avaliar a qualidade do relatório separadamente da qualidade do detector: correspondência entre texto e imagem, cobertura de achados, duplicatas, omissões, rastreabilidade e ausência de conclusões sem suporte.

Segmentação, CORAL e regras de engenharia devem ser avaliados quando houver anotações e dados específicos. A geração de texto não deverá preencher lacunas factuais ou criar enquadramentos normativos sem validação.

## 9. Critérios para encerrar a primeira revisão

- Os 38 arquivos únicos possuem decisão humana documentada após conferência integral das 13 classes, com vínculo às páginas.
- O piloto e as dúvidas receberam segunda revisão; ambiguidades foram resolvidas ou excluídas com motivo.
- Positivos têm caixas reais; negativos têm revisão completa; formato, coordenadas e IDs passaram no validador.
- Cobertura de classes, situações sub-representadas e necessidade de novos dados estão registradas.
- Não há aprovação de generalização baseada apenas nas fotos desta edificação.
- Nenhum novo peso, implantação ou relatório técnico final foi apresentado como aprovado antes das avaliações necessárias.

## 10. Responsabilidades e estado

Papéis a designar: anotação/triagem, revisão técnica, preparação dos experimentos e aprovação de uso. Não foram atribuídos nomes ou prazos sem confirmação.

**Histórico preservado:** duas leituras de IA das 38 imagens para as oito classes originais, com 108 caixas propostas, 23 fotos positivas, 2 negativas e 13 ambíguas. Não houve aprovação humana.

**Estado atual após importação humana:** 13 classes ativas; fila de 38 fotos com 115 caixas, 35 fotos positivas na triagem e três ambíguas. Há 33 aprovações humanas históricas nas dez classes da v3, com 108 caixas; zero aprovações integrais atuais nas 13 classes. IDs 10–12 ainda não têm exemplos, enquanto IDs 1 e 9 têm uma foto humana histórica cada. Estão disponíveis página corrigida, acervo histórico rastreável, relatório de revisão e controles de procedência e conclusão do treinamento.

**Pendente:** conferência das classes posteriores, resolução ou exclusão das ambiguidades e fotos sem decisão, novos exemplos diversificados das 13 classes, edificações independentes, primeiro treino especializado, comparação mensurável e aprovação dos relatórios técnicos. O acervo histórico não foi promovido a dataset pronto de 13 classes e nenhum treinamento especializado foi iniciado. A descoberta assistida de alvos fora do contrato e a entrada geral de novas campanhas são evoluções propostas em [novas imagens e anomalias](novas_imagens_e_anomalias.md), sem implementação ou validação atual.

Referência externa de método: a Ultralytics descreve coleta/anotação consistente, treinamento com rótulos e uso de pesos pré-treinados como etapas do processo. As regras específicas deste plano são decisões propostas para o projeto. [Dicas oficiais de treinamento](https://docs.ultralytics.com/guides/model-training-tips)
