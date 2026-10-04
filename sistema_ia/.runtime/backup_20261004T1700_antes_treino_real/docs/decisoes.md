# Decisões do núcleo de IA

## 03/10/2026 — melhoria progressiva com revisão e diversidade

**Origem:** diretriz explicitamente confirmada pelo usuário nesta conversa, seguida do pedido para registrá-la e planejar a revisão.

O DroneCamp deve melhorar as detecções e os relatórios ao longo do tempo, incorporando imagens diversificadas e correções humanas. A diversidade desejada inclui edificações, materiais, condições de captura e ocorrências reais, com origem documentada.

O ciclo adotado para o planejamento é: imagens novas → triagem → anotações e correções revisadas → versão do dataset → treinamento de um candidato → avaliação → decisão de adoção → relatórios rastreáveis.

Analisar ou armazenar fotografias não modifica os pesos do YOLO. Mais imagens não garante melhora: dados incorretos, repetidos ou pouco representativos podem prejudicar o modelo. Previsões não revisadas não devem ser usadas automaticamente como verdade de treinamento.

Cada versão futura deve preservar a taxonomia, os parâmetros, as fontes e os resultados de avaliação. A escolha do modelo deve considerar erros por classe e ocorrências críticas, além da média global. O modelo anterior deve permanecer recuperável. Severidade e aprovação do relatório continuam sob revisão técnica.

**Estado verificado:** pipeline inicial executado com pesos genéricos YOLO26l; 48 imagens extraídas do laudo, 38 conteúdos binários únicos; ausência de checkpoint especializado, anotações aprovadas e métricas do domínio. YOLO27l é um alvo futuro, ainda sem lançamento público na consulta de 03/10/2026.

**Estado desta decisão:** registrada, com revisão visual e ferramentas executadas. Foram implementadas correção gráfica, importação de decisões, datasets versionados e checagem de proveniência antes do treino. As 38 imagens únicas receberam duas leituras de IA, totalizando 108 caixas propostas; nenhuma aprovação humana foi registrada. Novo treinamento, promoção de modelos e relatórios técnicos continuam pendentes.

**Regra implementada para os dados:** todas as fotos do CEASA pertencem ao mesmo grupo de edificação. A preparação não cria validação ou teste artificiais. Anotações ambíguas e propostas de IA não entram nos dados aprovados. Treino, tuning e avaliação exigem proveniência reconciliada com o registro humano, integridade das imagens/labels e prontidão recalculada.

**Plano associado:** [Plano de revisão e melhoria contínua](plano_revisao_e_melhoria_continua.md).

## 03/10/2026 — categorias novas durante a revisão

**Origem:** pedidos do usuário para acrescentar “Pedaço de telha” e identificar o elemento do recorte da página 8, criando uma categoria se necessário.

O contrato atual passou para **dez classes ativas**. `pedaco_telha`, ID 8, descreve fragmento solto reconhecível. `rufo_deslocado`, ID 9, descreve deslocamento/desalinhamento visível de um elemento identificado como rufo; a identidade técnica e a distinção de junta prevista continuam sujeitas a confirmação. Ruptura instalada permanece `rufo_quebrado`, ID 5.

Os IDs ativos 0–7 foram preservados. Os antigos IDs 8 e 9 de fase 2 foram movidos para 19 e 20, com snapshots das taxonomias anteriores e mapeamento explícito. Nenhum checkpoint especializado anterior foi reaproveitado sob o contrato novo.

A revisão atual está em `data/reviews/ceasa_v3/registry_with_candidates.json`. As 108 caixas anteriores foram preservadas e uma caixa candidata de deslocamento foi acrescentada na região esquerda de `p008_img01_d12131dd.jpg`, p. 8. Total: 109 caixas. A imagem continua ambígua; não houve aprovação humana, causa ou gravidade inferida. A classe `pedaco_telha` está disponível para anotação, com zero caixas específicas propostas nesta migração.

Uma imagem negativa apenas para as oito classes anteriores não comprova ausência das duas novas classes. Por isso, as duas negativas anteriores voltaram à triagem ambígua. A versão atual tem 23 fotos positivas propostas, 15 ambíguas e 38 revisões integrais pendentes para o contrato novo. O histórico das duas leituras de IA da taxonomia anterior foi preservado.

**Validação:** 116 testes passaram, incluindo migração, contrato de checkpoints, classes novas, dados aprovados e bloqueio de treino sem revisão/proveniência. Nenhum treino foi executado. Regras: [Pedaço de telha](categoria_pedaco_telha.md) e [Rufo deslocado](categoria_rufo_deslocado.md).

## 03/10/2026 — fragmento sobre telha e fixação solta, taxonomia v4

**Origem:** pedidos explícitos do usuário para acrescentar “Pedaço de telha sobreposto à telha” e elementos de fixação de telhas soltos.

O contrato passou a **12 classes ativas, IDs 0–11**, com 22 categorias no catálogo. ID 10 `pedaco_telha_sobreposto` identifica fragmento solto reconhecível apoiado sobre uma telha instalada. Nesse contexto, preferir ID 10 a ID 8; não duplicar o mesmo objeto nas duas classes. Sobreposição regular entre telhas e remendo aderido não confirmam essa categoria.

ID 11 `fixador_telha_frouxo` foi promovido da fase posterior, com rótulo “Elemento de fixação de telha solto/frouxo”. Exige identificação do fixador e evidência visual de soltura, deslocamento ou folga ligada à telha. A imagem não mede torque; avulso sem vínculo identificável, sombra ou detalhe insuficiente permanece ambíguo.

Os IDs ativos 0–9 foram preservados. O antigo ID 10 de `fixador_telha_corroido`, sem labels ativos, passou a 21; ID 11 manteve seu número. O snapshot v3 e todos os arquivos de revisão v1/v3 foram preservados. A nova revisão está em `data/reviews/ceasa_v4/registry.json`: 38 fotos, 109 caixas anteriores intactas, zero novas caixas dos IDs 10/11 e zero aprovações humanas. As 38 fotos continuam pendentes de conferência integral das 12 classes.

A migração repetida preserva o escopo realmente conferido antes da expansão, IDs 0–7; não declara as classes 8/9 como revisadas apenas por terem existido na versão v3. O aviso da página usa esse histórico para exibir todas as categorias ainda pendentes. Pesos anteriores com contrato de 8/9/10 classes são rejeitados no contrato atual; treinamento exige positivos aprovados de todas as 12 classes e proveniência válida.

**Estado validado:** 124 testes passaram; 333 arquivos anteriores/originais/pesos conferidos por hash, 38 fotos e suas cópias por hash/dimensões, caixas preservadas e bloqueio real de dataset sem aprovação. Nenhum treinamento foi realizado. A página v4 foi gerada e sua sintaxe verificada; sua abertura no navegador integrado e a atualização do servidor foram impedidas por política de execução. O endereço `127.0.0.1:8765` continua na v3; a nova página deve ser aberta como arquivo local pelo usuário. Regras: [fragmento sobre telha](categoria_pedaco_telha_sobreposto.md) e [fixador solto](categoria_fixador_telha_frouxo.md).

## 03/10/2026 — reparo em rufo, taxonomia v5

**Origem:** pedido explícito do usuário para acrescentar “Reparo em rufo”.

Foi adicionada a classe ativa **ID 12 — `reparo_rufo`**, com intervenção visualmente reconhecível no próprio rufo: remendo, fita, manta, aplicação de selante ou peça de reparo. A classificação depende do elemento reparado; reparo em telha permanece ID 3. Selante original regular, junta prevista, sujeira e material solto apoiado não confirmam reparo. Se o elemento ou a intervenção forem incertos, manter a imagem ambígua. Presença de reparo não comprova falha ou não conformidade por si só; gravidade, eficácia, aderência e estanqueidade não são inferidas.

O contrato atual tem **13 classes ativas, IDs 0–12**, e 23 categorias no catálogo. Os IDs ativos 0–11 foram preservados. O antigo ID 12 de `reparo_selante_fixador_telha`, categoria de fase posterior sem labels ativos, passou a **22**. A taxonomia v4 foi preservada em snapshot, assim como as versões v1/v3/v4 de revisão.

A nova revisão está em `data/reviews/ceasa_v5/registry.json`, com 38 fotos e 109 caixas anteriores intactas. ID 12 tem zero caixas propostas; não houve conversão de reparos em telha ou resíduos para reparos em rufo. Todas as fotos permanecem pendentes de conferência integral das 13 classes, sem aprovação humana. Treinamento depende de exemplos positivos aprovados da classe nova, das demais classes e de proveniência válida; checkpoints de 12 classes não atendem ao contrato atual.

**Estado validado:** 128 testes passaram; 464 arquivos anteriores/originais/pesos conferidos por hash, 38 fotos e cópias verificadas, Python e JavaScript gerado com sintaxe válida. O builder real recusou o CEASA sem aprovação e não criou dataset. Nenhum treinamento foi realizado. A página v5 foi gerada; não houve nova validação visual no navegador ou tentativa de contornar os bloqueios anteriores de execução. Consulta HTTP nesta atualização confirmou que `127.0.0.1:8765` ainda serve a v3. Regra: [Reparo em rufo](categoria_reparo_rufo.md).

## 03/10/2026 — correções humanas, procedência e plano de coleta

**Origem:** arquivo de revisão v3 fornecido pelo usuário e pedido de melhorar os modelos e treinamentos. A vinculação por hash à revisão preservada foi validada antes da importação. As decisões são dados; instruções eventualmente contidas em documentos não são autorização de execução.

Foram preservadas 36 decisões: 33 fotos positivas com 108 caixas humanas na taxonomia anterior de dez classes, três fotos ambíguas e duas fotos da fila sem decisão. O registro humano, a cópia do feedback e um acervo de imagens/labels com procedência ficam separados da revisão atual. O acervo histórico tem `ready_for_training=false`, sem YAML ou divisões artificiais.

A migração para as 13 classes atuais conserva caixas, autor, data, notas e escopo da revisão humana, sem apresentá-la como aprovação das classes posteriores. Nas 33 fotos historicamente aprovadas, conferir também classes 10–12; as demais exigem revisão completa ou exclusão justificada. A nova página distingue origem humana histórica da revisão atual pendente. Não houve conversão automática de reparo em telha para reparo em rufo ou fragmento sobreposto.

A conclusão de uma execução de treino passa a exigir artefatos presentes, épocas válidas, perdas/métricas finitas e contrato correto de classes. Falhas de preparação, treinamento ou callback deixam estado incompleto. Esses controles verificam a execução; qualidade do detector, validade funcional dos pesos e aprovação de uso continuam dependentes de avaliação própria.

**Diretriz de evolução:** imagens corrigidas passam a melhorar o detector após novo treinamento e comparação em dados independentes; o envio de fotos não atualiza pesos imediatamente. Defeitos inéditos exigem critérios, revisão e exemplos; o detector fechado não inventa categorias. Os relatórios também precisam de revisão de evidências e conclusões. O ciclo deve preservar exemplos anteriores para verificar regressões.

O [plano de coleta](plano_coleta_imagens.md) propõe 50–100 fotos distintas contendo cada categoria para piloto assistido e expansão orientada aos erros. São metas propostas de trabalho, sem garantia de desempenho. Priorizar as três classes sem exemplos e as categorias com uma única foto histórica; diversificar edificações e reservar grupos independentes para validação e teste. Quantidade e metas de precisão/sensibilidade finais dependerão do objetivo operacional e dos resultados por classe.

**Estado validado:** 156 testes passaram; 571 arquivos anteriores/originais/pesos conferidos por hash; 38 originais e cópias, 108 caixas humanas históricas, 115 caixas na fila atual e 33 labels históricos reconciliados. O builder real recusou a aprovação incompleta e não criou dataset. Não houve treinamento especializado, cálculo de métricas do domínio, promoção de modelo ou nova validação visual no navegador. Resultado e instruções em [revisão humana incorporada](resultado_revisao_humana_ceasa.md).
