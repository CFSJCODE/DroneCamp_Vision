# Categoria: Reparo em rufo

Data: 03/10/2026. **Origem:** pedido explícito do usuário para acrescentar uma categoria de reparo no próprio rufo. Esta definição orienta a anotação; seu cadastro não acrescenta caixas ou aprova fotografias.

## Identificação e evidência necessária

| Campo | Definição |
| --- | --- |
| ID ativo | **12** |
| Nome no sistema | `reparo_rufo` |
| Rótulo de revisão | Reparo em rufo |
| Unidade de anotação | Uma região de intervenção visualmente identificável no rufo |
| Evidência exigida | Identidade do rufo e presença de fita, manta, remendo ou aplicação de selante reconhecível como intervenção sobre esse elemento |
| Gravidade | `null`, pendente de avaliação técnica |

Use ID 12 quando a fotografia permitir reconhecer tanto o rufo quanto uma região de intervenção nele. Delimite a menor caixa que contém o reparo visível. Não inclua todo o rufo ou a telha vizinha apenas para aumentar a caixa; a foto inteira mantém o contexto. Reparos contíguos e separados precisam seguir uma convenção aprovada no piloto para evitar contagens diferentes entre anotadores.

O rótulo registra **uma intervenção visível**. Não declara que o reparo falhou, que há vazamento ou que a cobertura está em não conformidade. Não permite concluir causa, data, material exato, aderência, resistência, urgência ou gravidade. Aparência de fita ou manta não comprova sua composição física.

## Escolha entre categorias próximas

| Situação observável | Categoria ou decisão |
| --- | --- |
| Fita, manta, remendo ou aplicação de selante identificável como intervenção no rufo | **12 — Reparo em rufo** |
| Intervenção identificável sobre o corpo da telha | **3 — Reparo em telha** |
| Ruptura ou perda parcial visível do rufo instalado | **5 — Rufo quebrado** |
| Desalinhamento ou deslocamento do rufo reconhecível | **9 — Rufo deslocado/desalinhado** |
| Selante regular de montagem, junta prevista ou acabamento original sem evidência de reparo | Não criar ID 12 apenas pela presença do material |
| Objeto solto, sujeira, sombra ou faixa de material sem vínculo reconhecível com uma intervenção no rufo | **Ambígua**, com a dúvida documentada |
| Elemento de apoio cuja identidade como rufo ou telha não é distinguível | **Ambígua**; não escolher ID 3 ou 12 pela proximidade |

**Reparo em rufo e reparo em telha são classes distintas pelo elemento que recebeu a intervenção.** Uma faixa junto ao encontro das peças exige contexto suficiente para decidir onde está aplicada. Quando uma mesma intervenção atravessar os dois elementos e não puder ser delimitada de modo consistente, registre a dúvida para decisão técnica do piloto, em vez de duplicar uma caixa idêntica nos dois IDs.

Reparo, ruptura e deslocamento podem coexistir na fotografia. Anote condições distintas somente quando cada uma tiver evidência própria e delimitação justificável. Não copie a mesma caixa idêntica do mesmo objeto para afirmar reparo e dano sem demonstração adicional. O reparo não prova ruptura escondida; sua presença também não elimina uma ruptura visível.

Selante existente pode pertencer à montagem original. Uma aplicação regular, sem evidência contextual de intervenção, não confirma reparo. Se a distinção não puder ser feita pela fotografia ou por informação verificável da inspeção, mantenha a ocorrência ambígua. Não atribua gravidade histórica de reparos em telhas a esta categoria.

## Revisão, versionamento e aprendizado

A taxonomia v5 tem **13 classes ativas, IDs 0–12**, e 23 categorias no catálogo. Os IDs ativos 0–11 mantêm seus significados. O antigo ID 12 de fase posterior, `reparo_selante_fixador_telha`, passa ao **ID 22**, preservando seu significado. O snapshot da taxonomia v4 e os metadados da migração permitem interpretar os registros anteriores; números iguais em versões diferentes não são equivalentes por si só.

Na [revisão v5](../data/reviews/ceasa_v5/index.html), ID 12 está disponível para seleção e desenho. **A migração tem zero caixas propostas desta classe.** As 109 caixas anteriores permanecem preservadas, sem conversão automática de `reparo_telha` para `reparo_rufo`. Todas as 38 fotos aguardam conferência integral das 13 classes e aprovação humana. A revisão anterior de oito classes por IA não substitui essa conferência ampliada.

Cadastrar a classe não modifica os pesos do YOLO. O reconhecimento exigirá exemplos diversificados de reparos em rufos, negativos revisados com juntas e selantes originais, anotações aprovadas, novo treinamento e avaliação por classe. Compare especialmente confusões com reparo em telha, rufo quebrado e rufo deslocado, além de regressões nas classes já existentes. O detector especializado e a avaliação técnica continuam pendentes.
