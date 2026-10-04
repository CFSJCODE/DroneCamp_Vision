# Categoria: Pedaço de telha

Data: 03/10/2026. **Origem da decisão:** pedido explícito do usuário para incluir a categoria “Pedaço de telha”. Este documento define seu critério de anotação; não registra treinamento ou aprovação de fotografias.

## Identificação e critério visual

| Campo | Definição |
| --- | --- |
| ID ativo | **8** |
| Nome no sistema | `pedaco_telha` |
| Rótulo de revisão | Pedaço de telha |
| Unidade de anotação | Um fragmento solto individualmente reconhecível |
| Evidência exigida | Forma, perfil, bordas e contexto suficientes para reconhecer visualmente uma parte de telha separada do elemento instalado |
| Gravidade | `null`, pendente de avaliação técnica |

Use essa categoria quando a foto permitir reconhecer um **fragmento solto de telha** dentro de uma calha, sobre outra superfície ou sem superfície de apoio identificável. Quando o fragmento solto estiver visivelmente apoiado sobre uma telha instalada, prefira a classe específica [10 — Pedaço de telha sobreposto à telha](categoria_pedaco_telha_sobreposto.md). Delimite a menor caixa que contém a peça visível. Inclua somente o contexto necessário para distinguir a peça de uma mancha, sombra ou parte da cobertura instalada.

O perfil ondulado, uma borda de fratura e uma sombra própria podem ajudar nessa distinção, mas nenhum desses sinais isoladamente confirma a categoria. Não atribua “Pedaço de telha” a toda peça cinza ou a qualquer material dentro da calha. Se o objeto estiver parcialmente encoberto ou a resolução não permitir reconhecer sua natureza, registre a dúvida e mantenha a imagem ambígua até revisão.

## Como distinguir das categorias existentes

| Situação observável | Categoria ou decisão |
| --- | --- |
| Fragmento solto reconhecível em outra superfície ou sem apoio identificável | **8 — Pedaço de telha** |
| Fragmento solto reconhecível apoiado sobre telha instalada | **10 — Pedaço de telha sobreposto à telha**, preferido ao ID 8 |
| Ruptura no corpo de uma telha que continua instalada | **0 — Telha quebrada** |
| Acúmulo contíguo de detritos sobre a cobertura | **2 — Resíduos sobre telhas** |
| Acúmulo de detritos no canal da calha | **6 — Resíduos em calha** |
| Remendo, fita ou outra intervenção aderida à telha | **3 — Reparo em telha**, quando a intervenção for visualmente identificável |
| Junta prevista entre peças, sombra ou variação de textura | Não criar uma ocorrência de pedaço de telha apenas por esse sinal |
| Peça possivelmente solta, sem distinção entre fragmento, remendo e elemento instalado | **Ambígua**, com a dúvida documentada |

Um fragmento solto não comprova onde ocorreu a ruptura original. Só desenhe uma caixa de “Telha quebrada” quando o dano no elemento instalado também estiver visível e for uma ocorrência distinta. Não atribua ruptura à cobertura apenas porque existe uma peça solta próxima.

Pedaços de rufo ou outros componentes não devem ser rotulados como telha por semelhança de cor. Se a identidade do componente não for distinguível, registre a incerteza. A presença de um remendo também não comprova falha de execução.

## Relação com resíduos e regras para não duplicar objetos

“Pedaço de telha” identifica **um objeto reconhecível**. As classes de resíduos descrevem **um acúmulo e seu contexto**, sobre telha ou dentro de calha. A localização pode ser registrada nas observações do fragmento; não é necessário criar outra caixa de resíduos para renomear o mesmo objeto isolado.

Adote estas regras de revisão:

1. Anote cada fragmento individual quando seus limites e sua natureza forem reconhecíveis. Não desenhe uma caixa de ID 8 sobre uma mistura inteira de materiais indistinguíveis.
2. Use ID 2 ou 6 para um acúmulo contíguo de resíduos, conforme a superfície visível. Uma caixa de acúmulo não presume que todos os seus componentes são pedaços de telha.
3. Pode existir uma caixa para o acúmulo e outra, menor, para um fragmento reconhecível dentro dele, **desde que as caixas representem escalas distintas ou objetos distintos**. Registre essa relação nas observações; não conte a peça novamente como um defeito físico independente do acúmulo.
4. **Nunca duplique o mesmo fragmento com IDs 8 e 10, nem uma caixa idêntica do mesmo objeto com ID 8/10 e ID 2 ou 6.** Para um único fragmento isolado reconhecível, use a categoria do fragmento. Se houver outros resíduos separáveis, anote-os conforme a convenção do acúmulo.
5. Se não for possível distinguir fragmento de telha, material de reparo, componente instalado ou resíduo genérico, mantenha a decisão ambígua. Não converta dúvida em imagem negativa nem em anotação aprovada.

Exemplo conceitual, sem criar anotação para uma foto: uma calha contém diversos detritos e uma peça ondulada claramente reconhecível. Uma caixa de ID 6 pode delimitar o acúmulo; outra caixa de ID 8 pode delimitar apenas a peça, com área menor e relação registrada. Copiar a mesma caixa do acúmulo e mudar o ID para 8 não representa esse critério.

## Fontes e limites da interpretação

As fotografias do **laudo CEASA Minas — Pavilão A**, páginas **5, 7 e 30**, são referências visuais para a discussão de fragmentos soltos, detritos sobre cobertura e material no canal da calha. O relatório descreve alguns fragmentos como fibrocimento; essa descrição documental não equivale a uma verificação física realizada pelo sistema.

O nome “Pedaço de telha” e seu ID ativo decorrem do pedido do usuário em 03/10/2026. **Não são apresentados como uma categoria histórica já definida no laudo.** A leitura da página, sua legenda ou a origem provável do material não determina automaticamente a classe ou os limites da caixa.

A classificação visual não confirma composição, presença de amianto, local de origem, causa da ruptura, quantidade física, risco, vazão ou obstrução. Esses fatos precisam de evidência própria. A gravidade permanece `null`; nenhuma classificação de resíduos ou fragmento herda automaticamente a gravidade histórica do caso CEASA.

## Versionamento e efeito sobre o treinamento

Os **IDs ativos 0–11 permanecem preservados** na v5. A inclusão inicial de `pedaco_telha` como ID 8 ampliou a taxonomia para nove classes; a v3 acrescentou `rufo_deslocado`, ID 9, chegando a dez. A v4 chegou a doze classes com `pedaco_telha_sobreposto`, ID 10, e a promoção de `fixador_telha_frouxo`, ID 11, preservando seu significado. O contrato atual v5 possui **13 classes ativas**, acrescentando `reparo_rufo`, ID 12. No catálogo de fase posterior, `telha_sobreposta_sem_travamento`, antigo ID 8, está no **ID 19**; `fixador_telha_ausente`, antigo ID 9, no **ID 20**; `fixador_telha_corroido`, antigo ID 10, no **ID 21**; e `reparo_selante_fixador_telha`, antigo ID 12, passa ao **ID 22**. Os demais IDs de fase posterior mantêm seus significados.

Essa mudança exige uma nova versão de taxonomia, revisões e dados, preservando os registros anteriores. As fotografias precisam ser revisadas segundo o novo critério; a existência anterior de uma caixa de resíduos não autoriza convertê-la automaticamente em pedaço de telha. Registros e arquivos de feedback continuam vinculados à versão de taxonomia e à origem correspondente.

Adicionar o nome à configuração ou selecionar a categoria na página de revisão **não ensina o modelo**. Um checkpoint preparado para oito ou nove categorias não adquire as saídas do contrato atual sem adaptação e novo treinamento. Para reconhecer a nova categoria, será necessário reunir exemplos diversificados, delimitar e aprovar seus rótulos, treinar um candidato com as 13 classes atuais e comparar os resultados por categoria, verificando também as oito categorias anteriores.

**Estado atual:** categoria disponível na taxonomia e na página de revisão v5; critério proposto e rastreável, pendente de aplicação e aprovação nas imagens. Não foram convertidas automaticamente caixas de resíduos em pedaços de telha. Há zero caixas específicas de IDs 8 e 10 nesta migração; todas as 38 fotos aguardam conferência integral das 13 classes e aprovação humana. Não houve treinamento da nova classe.
