# Novas classes sugeridas pela IA — 04/10/2026

Origem: pedido do usuário para usar as fotos de `Fotos_InsperçãoDeTelhadosIndustriais_Internet` e “pedir para a IA sugerir novas classes como Telha Trincada e outros”, e para que a IA marcasse sozinha as caixas de `resto-manutencao-telhado.webp`.

**Estado:** proposta para decisão humana. Nenhuma classe foi ativada; a taxonomia ativa (13 classes, SHA-256 `8e269463…`) não mudou. Ativar uma classe exige nova versão da taxonomia, `migrate-review` e revisão das caixas existentes.

## Como as propostas foram feitas

Leitura visual por IA (Claude) de cada foto inteira e de recortes ampliados com grade de coordenadas. As 29 caixas estão em [`data/proposals_ai/internet_v1_claude_visual.json`](../data/proposals_ai/internet_v1_claude_visual.json), identificadas pelo SHA-256 de cada foto. Cada caixa usa a **classe ativa mais próxima** em `class_id` e, quando nenhuma classe atual descreve bem o achado, a **classe nova sugerida** em `proposed_new_class`. O comando `add-ai-proposals` levou essas caixas para uma revisão própria como `first_pass_ai`, `human_approved: false`; a página mostra “nova classe sugerida: …” ao lado da caixa.

As 7 fotos vêm da internet: não são do CEASA, não têm autoria/licença registradas e não contam como edificações independentes para liberar o treino de produção.

## Classes sugeridas

| Classe sugerida | Achado visual | Diferenciar de | Evidência nas fotos |
| --- | --- | --- | --- |
| `telha_trincada` — Telha trincada | Fissura linear no corpo da telha, **sem separação nem perda de material** | `telha_quebrada` (ruptura com lacuna ou parte destacada), sombra de sobreposição, junta, risco superficial, cabo/fio | `telhado-fibrocimento.webp`: trinca longitudinal contínua (caixa 1) |
| `objeto_estranho_cobertura` — Objeto estranho sobre a cobertura (**ativar o ID 17 do catálogo**, hoje fase 2) | Material alheio ao sistema esquecido sobre a cobertura: tábuas, sacaria, perfis metálicos, baldes, cabos, vergalhões, pipa | `residuos_telha` (acúmulo de detritos, sedimentos, folhas), `pedaco_telha` (fragmento de telha), equipamento instalado (antena, linha de vida, tubulação fixada) | 9 caixas em 4 fotos: tábuas e sacaria (`resto-manutencao`), perfis e cabos (`detritos-sobre-telhado`), recipiente (`resto-tinta`), barra metálica (`telhado-fibrocimento`) |
| `residuo_tinta_argamassa` — Material de obra solidificado | Tinta, argamassa, cimento ou selante **derramado e aderido** sobre a telha | `reparo_telha` (intervenção intencional e delimitada: remendo, fita, manta), `residuos_telha` (material solto), mancha ou envelhecimento | `resto-tinta-sobre-telhado.webp` (placas e grumos); faixa branca de `resto-manutencao` (dúvida: reflexo) |
| `crescimento_biologico` — Musgo, líquen ou bolor | Cobertura biológica contínua sobre telha ou calha | `vegetacao_calha` (plantas, galhos, folhas), sujeira, mancha de envelhecimento | `galhos-liquen-calha.webp`: faixa contínua de musgo (caixa 1); manchas difusas no fibrocimento, sem caixa delimitável |

Critérios físicos comuns: a classe descreve o achado visual. Não inferir profundidade da trinca, infiltração, estanqueidade, causa, risco, severidade ou urgência pela foto. Umidade persistente sugerida por musgo não comprova infiltração; água acumulada continua excluída (`excluded_positive_claims`).

### Por que `objeto_estranho_cobertura` e não uma classe nova

O catálogo já tem o ID 17 (fase 2) com o critério “objeto alheio ao sistema; o laudo registra pipa e parafuso solto”. Ativá-lo evita duas classes com o mesmo significado. A proposta só amplia os exemplos do critério (material de obra e manutenção esquecido).

### Candidatas sem evidência suficiente nestas fotos

- `corrosao_telha` (oxidação em chapa metálica): comum em coberturas industriais, mas nenhuma das 7 fotos mostra corrosão delimitável.
- Chapa levantada/deslocada e fixadores elevados em `vegetacao-dentro-calha.webp` (borda da chapa em y≈390–495): a foto não permite confirmar; ficou só na nota da imagem. Fixadores já têm a classe 11 e as classes de fase 2 (IDs 20–22).

## Caixas propostas por foto

| Foto | Caixas (classe ativa → classe nova sugerida) |
| --- | --- |
| `resto-manutencao-telhado.webp` | 1 pilha de tábuas, 2 tábua longa, 3 sacaria (resíduos → `objeto_estranho_cobertura`); 4 peça ondulada sobre o rufo (`pedaco_telha_sobreposto`, dúvida com `rufo_deslocado`); 5–7 fragmentos (resíduos); 8 faixa branca (resíduos → `residuo_tinta_argamassa`, dúvida) |
| `telhado-fibrocimento.webp` | 1 trinca (`telha_quebrada` → `telha_trincada`); 2 barra metálica (→ `objeto_estranho_cobertura`) |
| `resto-tinta-sobre-telhado.webp` | 1 recipiente (→ `objeto_estranho_cobertura`); 2 material solidificado (→ `residuo_tinta_argamassa`); 3 pó/areia (resíduos) |
| `detritos-sobre-telhado.webp` | 1 perfis metálicos, 2 cabos (→ `objeto_estranho_cobertura`); 3 detritos junto à parede; 4 fragmentos escuros planos; 5–8 folhas/galhos (resíduos) |
| `detritos-calha.webp` | 1 folhas secas (`vegetacao_calha`); 2 sedimento (`residuos_calha`) |
| `galhos-liquen-calha.webp` | 1 musgo (`vegetacao_calha` → `crescimento_biologico`); 2 capim/galhos; 3 sedimento (`residuos_calha`); 4 galho na telha (resíduos) |
| `vegetacao-dentro-calha.webp` | 1 planta viva (`vegetacao_calha`); 2 terra e detritos (`residuos_calha`) |

## Para ativar uma classe

1. Decidir quais classes e slugs ativar (as quatro acima ou parte delas).
2. Criar a próxima taxonomia preservando os IDs 0–12 e acrescentando as novas ao fim; os IDs de fase 2 seguintes são renumerados, como na v5.
3. `migrate-review` nas revisões mais recentes: caixas preservadas, `taxonomy_recheck_pending` para as classes novas.
4. Na página, revisar as caixas com “nova classe sugerida” e trocar a classe quando confirmada; caixas antigas de `telha_quebrada` precisam ser reconferidas (trinca × quebra).
5. Coletar mais exemplos e negativos de cada classe nova antes de esperar sugestões úteis do modelo: com 1–2 exemplos o detector não aprende a classe.
