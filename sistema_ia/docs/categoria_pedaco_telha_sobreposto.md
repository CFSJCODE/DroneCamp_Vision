# Categoria: Pedaço de telha sobreposto à telha

Data: 03/10/2026. **Origem:** pedido explícito do usuário para distinguir o fragmento solto apoiado sobre uma telha instalada. O critério abaixo orienta a revisão; nenhuma fotografia recebeu aprovação ou nova caixa por esse cadastro.

## Identificação e evidência necessária

| Campo | Definição |
| --- | --- |
| ID ativo | **10** |
| Nome no sistema | `pedaco_telha_sobreposto` |
| Rótulo de revisão | Pedaço de telha sobreposto à telha |
| Unidade de anotação | Um fragmento solto reconhecível, apoiado sobre uma telha instalada |
| Evidência exigida | Identidade do fragmento e superfície de apoio reconhecíveis na fotografia, com limites que permitam distingui-lo da telha instalada |
| Gravidade | `null`, pendente de avaliação técnica |

Use ID 10 quando houver evidência conjunta de **um pedaço solto de telha** e **seu apoio sobre outra telha instalada**. Perfil, borda fraturada, contorno separado e relação entre planos podem ajudar; um desses sinais isolado não confirma a condição. Delimite o fragmento visível com a menor caixa adequada. A foto inteira e as observações preservam o contexto da telha de apoio.

“Sobreposto” descreve aqui a posição do fragmento solto. Não corresponde a toda sobreposição entre telhas, à presença de outra peça instalada nem a uma conclusão de falta de travamento.

## Escolha entre categorias próximas

| Situação observável | Categoria ou decisão |
| --- | --- |
| Fragmento solto reconhecível apoiado sobre telha instalada | **10 — Pedaço de telha sobreposto à telha** |
| Fragmento solto reconhecível em calha, sobre outra superfície ou sem superfície de apoio identificável | **8 — Pedaço de telha** |
| Ruptura no corpo de uma telha ainda instalada | **0 — Telha quebrada** |
| Mistura ou acúmulo de resíduos sobre telha, sem fragmento individual reconhecível | **2 — Resíduos sobre telhas** |
| Pedaço usado como remendo, com intervenção aderida ou fixada visualmente reconhecível | **3 — Reparo em telha** |
| Sobreposição prevista entre telhas instaladas | Não criar ID 10 apenas pela sobreposição |
| Objeto cuja natureza ou condição solta não é distinguível | **Ambígua**, com a dúvida documentada |

**Prefira ID 10 ao ID 8 quando o apoio sobre uma telha instalada estiver confirmado visualmente. Não duplique a mesma peça com IDs 8 e 10.** Se o fragmento for identificável e a superfície estiver desconhecida, ID 8 pode registrar a evidência conhecida sem inventar o contexto. Se também houver dúvida sobre a identidade ou sobre o fragmento estar solto, mantenha a ocorrência ambígua.

Uma caixa de acúmulo de resíduos e outra menor de um fragmento identificável podem existir quando representarem evidências e escalas distintas; registre essa relação para evitar contagem física duplicada. Não copie uma caixa idêntica do mesmo objeto para rotulá-lo também como resíduo.

O catálogo de fase posterior inclui `telha_sobreposta_sem_travamento`, ID 19. Ele descreve outra hipótese, referente a peça sobreposta e travamento; não aprova nem substitui automaticamente ID 10. Travamento oculto ou condição de aderência indefinida exigem confirmação própria. Não deduza ausência de fixação, vento, origem da quebra, composição, vazamento, risco ou gravidade pela posição do fragmento.

## Revisão, versionamento e aprendizado

A taxonomia atual v5 possui **13 classes ativas, IDs 0–12**, preservando os IDs ativos 0–11 e acrescentando `reparo_rufo`, ID 12. A taxonomia v4 tinha doze classes ativas: preservou IDs 0–9, reservou o novo ID 10 para o fragmento sobre telha e moveu a categoria de fase posterior `fixador_telha_corroido`, anteriormente catalogada como ID 10, para **ID 21**. O ID 11 de `fixador_telha_frouxo` manteve seu significado ao passar à fase ativa. Na v5, o antigo ID 12 de `reparo_selante_fixador_telha` passa ao ID 22. Snapshots e metadados de migração preservam a interpretação dos registros antigos.

Na [revisão v5](../data/reviews/ceasa_v5/index.html), ID 10 está disponível para seleção e desenho. **Há zero caixas propostas dessa classe na migração.** As 109 caixas anteriores permanecem preservadas; nenhuma foi convertida automaticamente. Todas as 38 fotos precisam de conferência integral das 13 classes e aprovação humana.

Cadastrar a classe não modifica os pesos do YOLO. O reconhecimento exigirá fotos diversificadas, negativos revisados que incluam sobreposições normais e reparos, caixas aprovadas, novo treinamento e comparação por classe. A avaliação deve verificar também confusões entre IDs 8 e 10 e regressões nas classes anteriores. O detector especializado e a gravidade técnica continuam pendentes.
