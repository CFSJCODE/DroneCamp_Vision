# Taxonomia v7: classes sem sobreposição (13/10/2026)

Pedido do Cláudio: "melhore, recrie e remaneje as classes para não ter sobreposição nem classes muito próximas, quero classes bem demarcadas e que cubram todas as não conformidades, mas sem sobreposição de classes como pedaço de telha e pedaço de telha sobre telha". A proposta foi aprovada como está (9 classes ativas).

**Resumo.** As 14 classes ativas e as 19 do catálogo da v6 viraram **9 ativas e 6 de catálogo**. A regra que acaba com as sobreposições: **a classe diz o que o objeto é e em que estado está, nunca onde está apoiado**. Tipo, material, apoio e elemento viram **atributos opcionais da caixa**, então a cobertura das não conformidades do laudo continua completa. Toda classe antiga cabe inteira numa classe nova (só fusões), então **nenhuma caixa é redesenhada**: a migração troca o ID e guarda a classe anterior.

## 1. Classes

### Ativas (o detector aprende)

Caixas: v8 (38 fotos aprovadas) + pré-revisão do Lado A guardada no repositório, contadas na v7 depois da fusão. A revisão do Lado A no PC tem mais caixas.

| ID | Slug | Nome | Entra | Não entra | Atributos | Caixas |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `telha_danificada` | Telha danificada | Ruptura, trinca, furo ou trecho faltando numa telha que continua instalada (opaca ou translúcida). | Telha inteira faltando (1); fragmento solto (3); oxidação (11); furo previsto. | `tipo` quebra/trinca/furo; `material` opaca/translucida | 17 |
| 1 | `telha_ausente` | Telha ausente | Lacuna da largura de uma telha inteira. | Se resta parte da telha, é 0. Abertura prevista. | — | 4 |
| 2 | `residuos_cobertura` | Resíduos sobre a cobertura | Acúmulo difuso, sem forma individual (terra, folhas, detritos miúdos). | Objeto individual (3); dentro da calha (4); musgo aderido (14). | — | 98 |
| 3 | `objeto_solto_cobertura` | Objeto solto sobre a cobertura | Peça individual solta: pedaço de telha, de calha ou de rufo, tábua, pipa, parafuso. **O apoio não muda a classe.** | Peça presa no lugar (0, 6, 9); acúmulo (2); remendo aderido (7). | `material` telha/calha/rufo_cumeeira/outro; `apoio` sobre_telha_instalada/outro | 32 |
| 4 | `calha_obstruida` | Calha com resíduos ou vegetação | Qualquer material no canal da calha ou na boca do ralo. | Água sem material (13); calha deformada (9). | `tipo` detritos/vegetacao; `local` canal/saida_ralo | 33 |
| 5 | `rufo_ausente` | Rufo ou cumeeira ausente | Trecho sem rufo ou cumeeira onde a peça é esperada. | Peça presente mas danificada (6). | `elemento` rufo/cumeeira | 24 |
| 6 | `rufo_danificado` | Rufo ou cumeeira danificado | Quebra, amassado, levantamento, deslocamento ou desalinhamento. | Trecho inteiro faltando (5); peça já solta (3). | `tipo` quebra/deslocamento; `elemento` rufo/cumeeira | 129 |
| 7 | `reparo_material_aplicado` | Reparo ou material aplicado | Remendo, fita, manta, selante, argamassa ou tinta aderidos (telha, rufo, calha, fixador). | Material solto (2, 3). Reparo não comprova falha. | `elemento`; `intencional` sim/nao | 55 |
| 8 | `fixador_irregular` | Fixador irregular | Parafuso ou gancho frouxo, solto, faltando ou corroído. | Parafuso caído longe do ponto (3); selante (7, 12). | `tipo` frouxo/ausente/corroido; `elemento` telha/rufo | 1 |

### Catálogo (fase 2, `requires_training`)

9 `calha_danificada`, 10 `telha_deslocada`, 11 `corrosao_metalica`, 12 `vedacao_falha`, 13 `agua_empocada`, 14 `crescimento_biologico`. Critérios completos em `configs/taxonomy.json` (`criterion`, `excludes`).

### Precedência (mesma região, uma caixa)

Ausente > danificado/deslocado > fixador > corrosão > vedação > reparo > objeto solto > resíduos > calha obstruída > crescimento biológico > água (lista `precedence` na taxonomia). Objetos diferentes na mesma foto têm caixas próprias.

## 2. Mapa v6 → v7

| v6 | v7 | Atributos gravados na caixa |
| --- | --- | --- |
| 0 telha_quebrada | 0 telha_danificada | tipo: quebra |
| 1 telha_ausente | 1 telha_ausente | — |
| 2 residuos_telha | 2 residuos_cobertura | — |
| 3 reparo_telha | 7 reparo_material_aplicado | elemento: telha |
| 4 rufo_ausente | 5 rufo_ausente | elemento: rufo |
| 5 rufo_quebrado | 6 rufo_danificado | tipo: quebra, elemento: rufo |
| 6 residuos_calha | 4 calha_obstruida | tipo: detritos |
| 7 vegetacao_calha | 4 calha_obstruida | tipo: vegetacao |
| 8 pedaco_telha | 3 objeto_solto_cobertura | material: telha |
| 9 rufo_deslocado | 6 rufo_danificado | tipo: deslocamento, elemento: rufo |
| 10 pedaco_telha_sobreposto | 3 objeto_solto_cobertura | material: telha, apoio: sobre_telha_instalada |
| 11 fixador_telha_frouxo | 8 fixador_irregular | tipo: frouxo, elemento: telha |
| 12 reparo_rufo | 7 reparo_material_aplicado | elemento: rufo |
| 13 pedaco_calha_sobre_telha | 3 objeto_solto_cobertura | material: calha |

O catálogo v6 também tem destino (`migration.catalog_class_map`): trinca/furo e translúcida → 0; objeto estranho → 3; ralo → 4; cumeeira → 5 e 6; selante em fixador e tinta/argamassa → 7; fixadores ausentes/corroídos → 8; selantes de junta, fixador e passagem → 12.

## 3. Migração das revisões (no PC, depois do merge)

O mesmo `migrate-review` faz a fusão. A revisão original fica intacta; a saída é uma pasta nova.

```
python -m dronecamp_ia migrate-review ^
  --registry data\reviews\<revisão v6>\registry.json ^
  --previous-taxonomy configs\taxonomy_ceasa_v6_pedaco_calha_sobre_telha.json ^
  --output data\reviews\<revisão>_taxonomia_v7\registry.json
python -m dronecamp_ia render-review --registry data\reviews\<revisão>_taxonomia_v7\registry.json
```

Revisões ainda na v5 vão direto para a v7 com `--previous-taxonomy configs\taxonomy_ceasa_v5_reparo_rufo.json` (as classes v5 são o início da lista v6 que o mapa cobre).

O que a migração grava:

- Cada caixa: `class_id` novo, `previous_class_id`, `previous_class_slug` e `attributes`. Coordenadas, notas e hashes não mudam.
- Cada foto: `human_approved: false` e `taxonomy_recheck_pending: true`, com a aprovação anterior no histórico. Só `telha_ausente` e `residuos_cobertura` contam como já revistas (escopo igual ao da v6); a página pede conferência das outras 7, que ficaram mais amplas (ex.: uma trinca não era classe na v6). **Enter aceita a foto** como está.
- `taxonomy_merge_overlaps`: pares de caixas que a fusão juntou na mesma classe sobre o mesmo objeto (IoU ≥ `pilot.duplicate_iou`). Nada é apagado. Nas revisões do repositório (v8 e pré-revisão do Lado A) não houve nenhum.

Ensaio na nuvem com a v8 (38 fotos, taxonomia v5): 246 caixas preservadas (17, 4, 79, 20, 27, 9, 47, 42, 1 por classe v7), 0 sobreposições criadas pela fusão, `render-review` sem erro e a página listando "Objeto solto sobre a cobertura · material: telha, apoio: sobre telha instalada".

## 4. Pesos antigos e comparação

- A v7 muda o número de classes (14 → 9): **é preciso treinar de novo** antes de adotar um modelo v7 (`build-pilot-data` → `train-pilot`).
- `compare-models`, `measure-models` e `suggest` aceitam pesos v6 (e v5) pelo mapa: cada previsão antiga vira a classe v7 que a contém, e duas previsões antigas fundidas sobre o mesmo objeto viram uma (fica a mais confiante). O `gate.json`/`measurement.json` registra `class_remap`. Assim o treino v6 de 13/10 (`12a305a6`) é comparável com o primeiro treino v7 nas mesmas fotos e classes.
- `predict`, exportação ONNX, calibrador e treino continuam exigindo pesos da taxonomia atual.
- `fixador_irregular` tem 1 exemplo: segue ativa para não perder a caixa, mas não terá desempenho útil sem fotos próximas ou recortes.

## 5. O que mudou no código

| Onde | O quê |
| --- | --- |
| `configs/taxonomy.json` | v7 (`ceasa-v7-classes-sem-sobreposicao`), com `criterion`, `excludes`, `attributes`, `precedence` e `migration` (mapa, hash e classes de origem). |
| `configs/taxonomy_ceasa_v6_pedaco_calha_sobre_telha.json` | Snapshot da v6, byte a byte (SHA-256 `a4350ab8…`). |
| `config.active_class_remap` | Lê e valida o mapa (só fusão; atributos dentro dos valores da classe de destino). |
| `review_data.migrate_review_taxonomy` | Expansão (como antes) ou fusão declarada. |
| `backend.check_domain_names(..., taxonomy=)` e `suggestions.detect_boxes` | Pesos antigos traduzidos para as classes v7. |
| `review_dataset.MVP_NAMES`, `configs/dataset.yaml` | 9 classes. |
| `review_render.COLORS` | 15 cores (cada classe ativa herdou a cor da classe v6 dominante); acima disso as cores se repetem em vez de quebrar o `render-review`. |
| `laudo_abnt.CHECKLIST` | Perguntas da v7, citando os itens do laudo CEASA fundidos. |
| `configs/zero_shot_prompts.json` | Frases da busca aberta reagrupadas; "hole in roof", "damaged skylight" e "bird nest" passaram a classes. A primeira busca aberta recria o checkpoint YOLOE. |
| `platform.js` | Lista de caixas mostra os atributos; editar uma caixa sem trocar a categoria mantém atributos e classe anterior. |

Pendente: a página ainda não tem um campo para o revisor escolher atributos em caixas novas (eles vêm da migração). Cores e nomes de exibição continuam editáveis na seção Classes.
