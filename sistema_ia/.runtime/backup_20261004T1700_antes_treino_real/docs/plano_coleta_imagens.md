# Plano de coleta e melhoria das detecções

Proposta de trabalho registrada em 03/10/2026. As quantidades abaixo orientam a coleta; não garantem desempenho nem substituem a avaliação em fotos de outras edificações.

## Como a revisão ajuda o sistema

Criar uma categoria define o que queremos ensinar. Revisar as fotos fornece exemplos corretos: qual objeto aparece, onde está e quando uma aparência parecida deve ser ignorada. O detector passa a usar esse conhecimento após um novo treinamento com os exemplos aprovados. Cadastrar a categoria, enviar uma foto ou corrigir uma caixa não altera os pesos do modelo imediatamente.

Fotos diversificadas ajudam o modelo a reconhecer a mesma condição em outros materiais, distâncias, ângulos e iluminações. Fotos incorretamente classificadas também ensinam, mas ensinam o erro. Dúvidas devem permanecer pendentes; hipóteses da IA não viram aprovação humana automaticamente.

O ponto de partida é pequeno: existem 38 fotos únicas do CEASA, e a revisão histórica da taxonomia com 10 classes contém 33 fotos positivas e 108 caixas. Essa revisão não comprova avaliação integral das 13 classes atuais. Não há pesos especializados treinados. As classes 10 — Pedaço de telha sobreposto à telha, 11 — Elemento de fixação de telha solto/frouxo e 12 — Reparo em rufo ainda não têm exemplos revisados; Telha ausente e Rufo deslocado têm apenas uma foto humana cada.

## Quantas imagens coletar

Não existe um número universal que assegure o objetivo. A proposta deste projeto é começar com **50–100 fotos distintas contendo cada classe** para um piloto assistido e, depois, ampliar para **200–500 por classe**, buscando os casos em que o modelo erra. São metas de coleta propostas, não mínimos obrigatórios nem promessa de qualidade para as 13 categorias.

A Ultralytics apresenta **1.500 imagens por classe e 10.000 instâncias anotadas por classe** como regra de planejamento para produção. A mesma fonte informa que tarefas estreitas, como uma classe com uma câmera, podem funcionar com 200–500 imagens. Esse cenário estreito não equivale ao nosso conjunto de 13 classes em inspeções de diferentes coberturas. [Fonte: especificação de conjuntos de dados da Ultralytics Academy](https://academy.ultralytics.com/courses/dataset-readiness-for-yolo/define-the-dataset-spec).

| Fase | Meta proposta e prioridade | Critério para avançar |
|---|---|---|
| Resolver as lacunas atuais | Revisar as classes novas 10, 11 e 12; completar as fotos pendentes; buscar mais exemplos de Telha ausente e Rufo deslocado. | Critérios de anotação claros, dúvidas resolvidas por pessoa habilitada e exemplos reais das categorias. |
| Piloto assistido | Buscar 50–100 fotos distintas contendo cada uma das 13 classes, com diversidade e revisão humana. | Dados e origem validados; conjunto independente disponível; primeiro treinamento avaliado por classe. Usar as sugestões com conferência humana. |
| Expansão orientada aos erros | Buscar 200–500 fotos por classe, priorizando categorias raras, omissões, falsas detecções e condições ainda mal representadas. | Comparar novos treinamentos na mesma validação; confirmar melhora nas classes e cenários importantes. |
| Preparação para uso amplo | Planejar volume, variedade e instâncias com a referência de produção acima, ajustando a coleta aos resultados reais. | Cumprir as metas de desempenho que serão pactuadas e verificar em edificações reservadas para o teste final. |

**Foto e caixa são medidas diferentes.** Uma foto pode conter cinco ocorrências, com cinco caixas, e pode conter várias classes. Ela conta uma vez na quantidade total de fotos e uma vez na cobertura de cada classe presente. Portanto, somar as metas das 13 classes não determina automaticamente a quantidade de fotos únicas necessária. Também acompanhe quantas caixas e quantas edificações representam cada classe.

Repetir a mesma imagem, recortá-la, clareá-la ou usar vários quadros quase idênticos de um vídeo não produz novas observações independentes. As transformações usadas no treinamento podem ajudar, mas não substituem a coleta em cenários diferentes.

## O que precisa variar e como separar os dados

Registrar o prédio, a campanha e a origem de cada foto permite manter juntos os registros relacionados. Diversificar materiais e formatos de cobertura, alturas de voo, ângulos, luz, sombra, reflexos, distância, resolução e tamanho aparente das ocorrências. Incluir situações normais e parecidas com defeitos: sobreposição regular, junta, sombra, fixador bem assentado e reparo cuja natureza ainda precisa ser confirmada.

Reservar fotos de fundo sem nenhuma das classes presentes ajuda a examinar falsas detecções. A referência da Ultralytics propõe **0–10% do total**; a proporção adotada será ajustada ao uso real. Uma imagem só é negativa após revisão completa das classes atuais: ausência de caixas ou baixa qualidade não provam ausência de ocorrência. [Fonte: Ultralytics Academy](https://academy.ultralytics.com/courses/dataset-readiness-for-yolo/define-the-dataset-spec).

Separar **edificações** entre treino, validação e teste, mantendo todas as campanhas relacionadas ao mesmo prédio no mesmo grupo. O sistema atual exige pelo menos três grupos para essas três divisões. Essa é uma barreira técnica, não uma garantia de representatividade; planejar mais edificações e verificar a cobertura de cada classe em cada divisão. As fotos do CEASA formam um único grupo e não devem ser repartidas para simular independência.

## Como saber se a coleta está melhorando o modelo

Manter uma validação fixa e independente. A cada rodada, comparar a precisão, a sensibilidade (*recall*, proporção das ocorrências reais encontradas) e as falsas detecções por 100 fotos, discriminadas por classe e cenário. Registrar também omissões, categorias confundidas e correções de caixas. As metas finais de desempenho precisam ser pactuadas conforme a importância dos erros; não foi definido um percentual garantido.

Construir uma curva de aprendizagem: treinar com quantidades crescentes de fotos revisadas, usando parâmetros comparáveis e a mesma validação. Se uma categoria parar de melhorar, investigar rótulos, critérios e cenários ausentes antes de apenas acumular imagens semelhantes. Reservar o teste final para a decisão de uso; não repetir ajustes com base nele.

Relatórios precisam de outra revisão além das caixas: correspondência entre texto e foto, localização, descrição fiel da evidência, clareza das dúvidas e confirmação técnica das conclusões. Mais fotos podem melhorar a detecção, mas não asseguram por si só um relatório correto nem comprovam causa, gravidade ou aperto de fixadores.

## Checklist de cada rodada

- Coletar fotos originais, distintas e com origem identificada, priorizando as classes novas e raras.
- Anotar as ocorrências visíveis; confirmar toda a imagem e registrar dúvidas sem transformá-las em rótulos aprovados.
- Conferir quantidade de fotos, caixas, edificações e cenários por classe, incluindo negativos revisados.
- Separar os prédios entre treino, validação e teste e preservar os arquivos e revisões anteriores.
- Treinar somente após validar o conjunto; comparar os resultados por classe e escolher a próxima coleta a partir dos erros.
- Revisar os relatórios e registrar se cada conclusão está apoiada em evidência suficiente.

Este plano não importa revisões, não cria aprovações, não inicia treinamento e não altera o modelo. O próximo treinamento depende de dados com revisão compatível com as 13 classes e de conjuntos independentes suficientes para avaliação.
