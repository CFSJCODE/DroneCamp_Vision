# Categoria: Elemento de fixação de telha solto/frouxo

Data: 03/10/2026. **Origem:** pedido explícito do usuário para adicionar elementos de fixação de telhas soltos. A categoria `fixador_telha_frouxo` já existia na fase posterior do catálogo e passa à detecção ativa, preservando seu ID e seu significado visual.

## Identificação e evidência necessária

| Campo | Definição |
| --- | --- |
| ID ativo | **11** |
| Nome no sistema | `fixador_telha_frouxo` |
| Rótulo de revisão | Elemento de fixação de telha solto/frouxo |
| Unidade de anotação | Um elemento ou conjunto de fixação visivelmente solto/deslocado, delimitável e ligado à telha pelo contexto |
| Evidência exigida | Detalhe suficiente para reconhecer o fixador da telha e uma condição de soltura, deslocamento ou folga aparente |
| Gravidade | `null`, pendente de avaliação técnica |

A foto deve permitir distinguir **o elemento de fixação da telha** e **a condição visual que motivou o rótulo**. Exemplos possíveis são um conjunto afastado da posição de apoio, uma peça de fixação deslocada ou uma folga claramente visível. O fato de haver parafuso, porca, arruela ou gancho na foto não demonstra frouxidão. Uma sombra, ressalto, cor ou orientação isolados também não bastam.

Delimite o elemento ou conjunto afetado, mantendo apenas o contexto necessário para reconhecer sua relação com a telha e o apoio. Se a resolução ou oclusão impedir a confirmação, registre a dúvida e mantenha a imagem ambígua. Ampliar digitalmente a foto não recupera detalhe ausente.

## Escolha entre condições próximas

| Situação observável | Categoria ou decisão |
| --- | --- |
| Fixador reconhecível da telha com soltura, deslocamento ou folga aparente suficientemente visível | **11 — Elemento de fixação de telha solto/frouxo** |
| Parafuso, porca ou arruela instalado, sem evidência de soltura | Não criar ID 11 apenas pela presença do elemento |
| Posição esperada sem fixador, confirmada com contexto e detalhe | `fixador_telha_ausente`, **ID 20, fase posterior**; registrar a observação sem forçar ID 11 |
| Oxidação aparente, sem evidência própria de soltura | `fixador_telha_corroido`, **ID 21, fase posterior**; não inferir folga a partir da cor |
| Parafuso ou outro objeto avulso sobre a cobertura, sem vínculo reconhecível com a fixação da telha | **Ambígua** quanto ao ID 11; registrar a identidade e o vínculo pendentes |
| Fixador relacionado ao rufo, sem relação com a telha | Não usar ID 11; registrar a ocorrência fora desta classe |
| Sombra, oclusão ou resolução insuficiente para distinguir assentamento e folga | **Ambígua**, com o limite documentado |

Um fixador avulso não comprova qual telha o perdeu nem permite criar uma caixa de fixador ausente em uma posição presumida. Se o contexto permitir reconhecer sua relação com a fixação da telha, registre a evidência da soltura para revisão; se esse vínculo não estiver estabelecido, mantenha a dúvida. Não classifique todo objeto metálico solto como fixador de telha.

Não copie a mesma caixa com dois rótulos para contar condições simultâneas. Se soltura e outro achado forem visíveis e distintos, documente as evidências para adjudicação. Classes de fase posterior não entram automaticamente nos labels do detector ativo.

## Referência documental e limites

O laudo **CEASA Minas — Pavilhão A, item 17, página 17, figura 23**, registra fixador frouxo em telha. Essa referência orienta a discussão da categoria; a legenda não comprova a condição de cada fixador na foto nem fornece automaticamente suas caixas.

**A imagem não mede torque de aperto.** O sistema não conclui força de fixação, estabilidade estrutural, perda de estanqueidade, causa da movimentação, risco ou urgência a partir do rótulo. A gravidade histórica do CEASA permanece como referência do caso; `severity` continua `null` e precisa de avaliação técnica própria.

## Revisão, versionamento e aprendizado

A taxonomia atual v5 possui **13 classes ativas, IDs 0–12**, e 23 categorias no catálogo. Os IDs ativos 0–11 foram preservados e `reparo_rufo` foi acrescentado no ID 12; o antigo ID 12 de `reparo_selante_fixador_telha` passa ao ID 22. Na v4, o ID 11 de `fixador_telha_frouxo` foi promovido da fase posterior para a fase ativa, os IDs ativos 0–9 foram preservados, ID 10 passou a `pedaco_telha_sobreposto` e o antigo ID 10 de `fixador_telha_corroido` passou ao ID 21. Registros antigos devem ser interpretados com o snapshot correspondente.

Na [revisão v5](../data/reviews/ceasa_v5/index.html), ID 11 está disponível para seleção e desenho. **Há zero caixas propostas dessa classe na migração.** As 109 caixas anteriores permanecem preservadas e todas as 38 fotos aguardam conferência integral das 13 classes e aprovação humana. Ativar a categoria não confirma que as fotos disponíveis têm detalhe suficiente para anotá-la.

O novo treinamento depende de exemplos aprovados, incluindo fotos de detalhe, fixadores bem assentados como negativos pertinentes, variação de tipos e materiais conhecidos, iluminação e ângulos. O protocolo exige positivos de todas as 13 classes no treino e edificações independentes nos splits. Nenhum modelo foi treinado ou aprovado por essa inclusão; medir desempenho de objetos pequenos e revisar falsas interpretações de sombra/arruela será parte da avaliação.
