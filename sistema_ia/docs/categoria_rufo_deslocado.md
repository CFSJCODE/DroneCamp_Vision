# Categoria: Rufo deslocado/desalinhado

Data: 03/10/2026. **Origem:** pedido do usuário para classificar o elemento mostrado na fotografia e criar uma categoria adequada caso as existentes não descrevam sua condição visual. A distinção adotada é deslocamento/desalinhamento observável, sem exigir ou presumir ruptura.

## Identificação e critério de anotação

| Campo | Definição |
| --- | --- |
| ID ativo | **9** |
| Nome no sistema | `rufo_deslocado` |
| Rótulo de revisão | Rufo deslocado/desalinhado |
| Evidência principal | Segmento de rufo reconhecível fora do alinhamento ou da posição esperada em relação às peças adjacentes e à junção da cobertura |
| Unidade de anotação | Um segmento deslocado/desalinhado visualmente delimitável |
| Gravidade | `null`, pendente de avaliação técnica |

Use a categoria quando a imagem apresentar um segmento cuja identidade como rufo seja suficientemente reconhecível e cujo desalinhamento possa ser observado em relação à continuidade da junção e aos elementos vizinhos. Inclinação, afastamento ou mudança de posição podem servir de evidência visual quando houver contexto adequado.

Delimite o segmento visível e inclua apenas a margem de contexto necessária para demonstrar sua relação com o alinhamento adjacente. Não desenhe uma caixa sobre toda a faixa transversal se somente uma parte estiver deslocada. Não há limiar universal em pixels ou graus definido nesta etapa; a convenção precisa ser revisada com o responsável técnico e aplicada consistentemente.

A identificação do elemento também é uma condição de revisão. Uma peça linear na junção pode ser rufo, outro acabamento, suporte ou material solto. Se sua função não puder ser distinguida pela imagem, registre **identidade do elemento pendente** e mantenha a imagem ambígua. O rótulo candidato não transforma essa hipótese em fato confirmado.

## Diferenças entre as condições visuais

| Situação | Categoria ou decisão |
| --- | --- |
| Rufo identificável com segmento fora do alinhamento observado na junção | **9 — Rufo deslocado/desalinhado** |
| Ruptura, borda de fratura ou perda irregular de material no rufo instalado | **5 — Rufo quebrado**, quando o dano for efetivamente visível |
| Junta ou sobreposição prevista, sem evidência de deslocamento indevido | Não criar ocorrência de deslocamento apenas pela interrupção entre peças |
| Ruptura no corpo de uma telha instalada | **0 — Telha quebrada** |
| Fragmento solto de telha em outra superfície ou sem apoio identificável | **8 — Pedaço de telha** |
| Fragmento solto de telha apoiado sobre telha instalada | **10 — Pedaço de telha sobreposto à telha** |
| Peça linear solta, sem identidade ou posição original reconhecível | **Ambígua**; não presumir rufo deslocado ou pedaço de telha |

**Desalinhamento não comprova quebra.** Uma ponta reta ou um espaço entre peças pode corresponder a uma junta. Só atribua ID 5 quando houver evidência própria de ruptura, diferenciada da posição alterada.

Se uma mesma peça mostrar deslocamento e uma ruptura distinta, registre claramente as duas evidências para adjudicação. Não copie uma caixa idêntica e altere apenas seu ID para contar automaticamente dois problemas. A convenção para condições simultâneas deve ser definida na revisão; enquanto houver dúvida, mantenha o caso ambíguo.

Também não conclua ausência de rufo apenas porque uma peça aparece afastada. A categoria de ausência requer contexto suficiente para localizar um trecho onde o elemento deveria estar presente.

## Referência visual verificada

O recorte enviado pelo usuário em 03/10/2026 foi comparado ao original e corresponde ao **trecho esquerdo de `p008_img01_d12131dd.jpg`, página 8 do laudo CEASA Minas — Pavilão A**. O alinhamento das ondas e do elemento transversal permite reconhecer a mesma cena.

Nesse trecho aparece uma peça linear inclinada/desalinhada em relação à faixa adjacente. A interpretação como rufo e a hipótese de ruptura na ponta continuam sujeitas a confirmação técnica. A categoria nova permite registrar o deslocamento observado sem obrigar a classificação como “Rufo quebrado” ou “Pedaço de telha”.

Na revisão anterior consultada, a área esquerda mostrada no recorte **não possuía uma caixa própria**. A caixa de `rufo_quebrado` `[802, 105, 872, 171]` estava na extremidade direita do original e descrevia outra região; ela não aprova nem representa o segmento esquerdo. A inclusão de uma proposta de ID 9 deve ser registrada em nova revisão e manter explícita a incerteza quanto à identidade do componente.

A fonte visual é referência para o critério, não uma categoria histórica já definida no laudo. O pedido do usuário dá origem ao novo rótulo; a página, a legenda e a localização não aprovam automaticamente a classe ou uma caixa.

## Limites das conclusões

A posição observável não comprova falta de travamento, aperto inadequado, fixador ausente, falha de selante, ação do vento, causa da movimentação ou perda de estanqueidade. Nenhuma dessas conclusões deve ser incluída automaticamente no relatório pela presença de ID 9.

O sistema também não determina material, composição, dimensão física, urgência ou gravidade apenas pela fotografia. **`severity` permanece `null`.** Confiança do detector e intensidade do desalinhamento visual não substituem avaliação técnica.

## Evolução dos IDs e do modelo

A inclusão de `rufo_deslocado` reservou **ID 9** e ampliou a taxonomia v3 para **dez classes**, de 0 a 9. A v4 acrescentou `pedaco_telha_sobreposto`, ID 10, e promoveu `fixador_telha_frouxo`, ID 11, à fase ativa, chegando a doze classes. O contrato atual v5 possui **13 classes**, de 0 a 12, preservando os IDs ativos 0–11 e acrescentando `reparo_rufo`, ID 12.

Para evitar colisão com o catálogo anterior:

- `fixador_telha_ausente`, categoria de fase 2 anteriormente catalogada com ID 9, passa para **ID 20**.
- `telha_sobreposta_sem_travamento`, antiga categoria de fase 2 com ID 8, permanece no **ID 19**, conforme a evolução anterior.
- `fixador_telha_corroido`, antiga categoria de fase 2 com ID 10, passa ao **ID 21** para reservar ID 10 ao fragmento sobreposto.
- `fixador_telha_frouxo` mantém o **ID 11** e passa à fase ativa.
- `reparo_selante_fixador_telha`, antiga categoria de fase 2 com ID 12, passa ao **ID 22** na v5 para reservar ID 12 a reparo em rufo.

Registros anteriores precisam permanecer rastreáveis à sua própria versão. Não interprete um ID 9 de um catálogo antigo como deslocamento de rufo sem conferir a versão da taxonomia. Atualizações de rótulos, revisões, dados e modelos devem registrar a migração correspondente.

Cadastrar a categoria e salvar uma caixa candidata **não atualiza os pesos do YOLO**. Um checkpoint com oito ou nove classes não adquire a nova classe por alteração do nome. O reconhecimento exige exemplos diversificados, anotações revisadas e aprovadas, novo treinamento e comparação dos resultados por categoria, incluindo verificação de regressões nas classes anteriores.

**Estado atual:** categoria disponível na taxonomia e na página de revisão v5. A v3 acrescentou uma caixa candidata `[0, 133, 223, 202]` para o segmento esquerdo de `p008_img01_d12131dd.jpg`, preservando a caixa anterior de ruptura à direita. As duas permanecem nas migrações v4 e v5, sem nova aprovação. A imagem permanece ambígua, com identidade do componente e aprovação técnica/humana pendentes. Todas as 38 fotos aguardam conferência integral das 13 classes. Não houve treinamento, medição de desempenho da classe ou emissão de laudo técnico.
