# Categoria: Pedaço de calha sobre telha

Data: 11/10/2026. **Origem:** pedido explícito do usuário, com um recorte de exemplo de um segmento alongado e claro de calha apoiado sobre as telhas. Esta definição orienta a anotação; seu cadastro não acrescenta caixas nem aprova fotografias.

## Identificação e evidência necessária

| Campo | Definição |
| --- | --- |
| ID ativo | **13** |
| Nome no sistema | `pedaco_calha_sobre_telha` |
| Rótulo de revisão | Pedaço de calha sobre telha |
| Unidade de anotação | Um segmento solto de calha, individual |
| Evidência exigida | Perfil reconhecível como calha (metálica ou de PVC: peça alongada, bordas dobradas ou seção em U) apoiado sobre a superfície das telhas, fora da posição de instalação |
| Gravidade | `null`, pendente de avaliação técnica |

Use ID 13 quando a fotografia permitir reconhecer o objeto como um pedaço de calha e a superfície de apoio como telha. Delimite a menor caixa que contém o segmento visível, sem incluir as telhas de apoio nem a calha instalada próxima. Segmentos separados recebem caixas separadas; segmentos sobrepostos que não se distinguem formam uma única caixa.

Na escala das fotos de drone o segmento aparece como uma faixa estreita e alongada, clara ou metálica, frequentemente paralela às ondas da telha. Por isso a caixa costuma ser fina e comprida: confira a orientação e não alargue a caixa para "dar contexto"; a foto inteira já mantém o contexto.

O rótulo registra **um objeto solto visível**. Não declara de onde a peça saiu, que falta calha em algum trecho, que há falha de escoamento ou risco de queda. Não permite concluir material exato, causa, data, urgência ou gravidade.

## Escolha entre categorias próximas

| Situação observável | Categoria ou decisão |
| --- | --- |
| Segmento solto reconhecível como calha, apoiado sobre telhas | **13 — Pedaço de calha sobre telha** |
| Calha instalada com detritos ou vegetação no canal | **6 — Resíduos em calha** / **7 — Vegetação em calha** |
| Rufo identificável, deslocado ou levantado em relação ao arremate | **9 — Rufo deslocado/desalinhado** |
| Fragmento de telha solto, sobre telha ou em outro local | **10 — Pedaço de telha sobreposto** / **8 — Pedaço de telha** |
| Acúmulo de detritos sem objeto identificável como calha | **2 — Resíduos sobre telhas** |
| Faixa aderida (fita, manta, remendo) sobre a telha | **3 — Reparo em telha** |
| Perfil metálico solto que não se distingue entre calha, rufo, cumeeira ou outro arremate | **Ambígua**, com a dúvida documentada |
| Reflexo, sombra ou junta entre telhas com aparência de faixa | Não criar ID 13 |

**Não duplique o mesmo objeto em duas classes.** Prefira ID 13 a ID 2 quando a peça for reconhecível como calha; um amontoado que contenha um pedaço de calha identificável e outros detritos distintos pode ter as duas caixas, cada uma delimitando sua própria evidência, com nota na revisão.

## Revisão, versionamento e aprendizado

A taxonomia v6 (`ceasa-v6-proposta-pedaco-calha-sobre-telha`) tem **14 classes ativas, IDs 0–13**, e 24 categorias no catálogo. Os IDs ativos 0–12 mantêm seus significados. O antigo ID 13 de fase posterior, `fixador_rufo_ausente`, passa ao **ID 23**, preservando seu significado. O snapshot `configs/taxonomy_ceasa_v5_reparo_rufo.json` guarda os bytes exatos da v5, para que registros e pesos anteriores continuem verificáveis pelo hash.

Consequências práticas da troca para a v6:

- **Pesos piloto atuais (13 classes) deixam de ser aceitos** por `suggest`, `predict`, `export` e comparações, porque o contrato de classes mudou (`check_domain_names`). As sugestões da IA voltam depois de um novo `train-pilot` com 14 classes.
- **Registros de revisão da v5 precisam ser migrados** com `migrate-review`, que preserva fotos, hashes e todas as caixas, mas deixa cada foto pendente de conferência da nova classe (aprovações anteriores ficam no histórico, não valem para a v6):

```
python -m dronecamp_ia migrate-review ^
  --registry data\reviews\ceasa_v9_rev001\registry.json ^
  --previous-taxonomy configs\taxonomy_ceasa_v5_reparo_rufo.json ^
  --output data\reviews\ceasa_v10_pedaco_calha\registry.json
```

Ensaio na nuvem com `ceasa_v9_rev001` (38 fotos): 226 caixas preservadas, zero caixas da classe 13 inventadas, 37 aprovações anteriores movidas para o histórico e 38 fotos marcadas para reconferência.

Cadastrar a classe não modifica os pesos do YOLO. O reconhecimento exigirá exemplos aprovados de pedaços de calha (de preferência de mais de uma edificação), negativos revisados com calhas instaladas, rufos e reflexos, novo treinamento e avaliação por classe. Compare especialmente confusões com resíduos sobre telhas, rufo deslocado e pedaço de telha, além de regressões nas classes já existentes.
