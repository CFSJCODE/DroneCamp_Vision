# Análise da taxonomia de classes e cadastro pela plataforma — 11/10/2026

> **Substituída em parte pela [taxonomia v7](taxonomia_v7.md) (13/10/2026).** Os pares sobrepostos apontados aqui (8/10, 2/8/17, 4/5/9/12) foram fundidos; IDs e slugs abaixo são os da v6.

Pedido: "faça uma análise geral; caso haja muitas classes redundantes ou faltantes, adicione-as e crie um CRUD para as classes".

**Resumo.** A taxonomia v6 tem 14 classes ativas (IDs 0–13, as que o detector aprende) e um catálogo de fase 2 (IDs 14–23) herdado do laudo CEASA. Não há classe redundante a ponto de justificar remoção: os pares parecidos (8/10, 2/8/17, 4/5/9/12) descrevem evidências visuais distintas e as políticas de anotação já dizem qual usar. O problema real é de **cobertura**: faltavam categorias comuns em telhados industriais (calha danificada, telha trincada, corrosão, água empoçada, telha translúcida, ralo obstruído, cumeeira) e as classes 10–13 quase não têm exemplos. Nove categorias entraram no catálogo (IDs 24–32, fase 2, `requires_training: true`) e a plataforma ganhou a seção **Classes** com rotas `/api/classes`. **Nenhum ID existente mudou e o detector atual continua com 14 classes**: só um novo ciclo de dados aprovados + nova versão da taxonomia + treino comparado leva uma classe nova à detecção.

## 1. As 14 classes ativas, uma a uma

| ID | Classe | Elemento | Avaliação |
| --- | --- | --- | --- |
| 0 | `telha_quebrada` | telha | Núcleo do laudo (p. 5). Critério claro: ruptura visível. |
| 1 | `telha_ausente` | telha | Núcleo. Depende de contexto (lacuna onde se espera telha). |
| 2 | `residuos_telha` | telha | Acúmulo delimitável. Fronteira com 8 e 17 descrita na política do ID 8. |
| 3 | `reparo_telha` | telha | Intervenção visível. Não comprova falha. |
| 4 | `rufo_ausente` | rufo | Núcleo. Exige conhecer a junção esperada. |
| 5 | `rufo_quebrado` | rufo | Núcleo. |
| 6 | `residuos_calha` | calha | Núcleo (gravíssima no laudo). |
| 7 | `vegetacao_calha` | calha | Núcleo. |
| 8 | `pedaco_telha` | fragmento | Acrescentada em 03/10. Fragmento solto reconhecível, em qualquer lugar. |
| 9 | `rufo_deslocado` | rufo | Acrescentada em 03/10. Deslocamento sem ruptura. |
| 10 | `pedaco_telha_sobreposto` | fragmento | Acrescentada em 03/10. **Subcaso do ID 8** (fragmento apoiado sobre telha instalada). |
| 11 | `fixador_telha_frouxo` | fixador | Promovida da fase 2 em 03/10. Objeto de poucos pixels em foto de drone. |
| 12 | `reparo_rufo` | rufo | Acrescentada em 03/10. Espelho do ID 3 para o rufo. |
| 13 | `pedaco_calha_sobre_telha` | fragmento | Acrescentada em 11/10 (PR #12). Segmento de calha solto sobre as telhas. |

Exemplos disponíveis (registro v7 de 04/10, 117 caixas): classes 0–9 têm caixas; **ID 10 tinha 6 caixas em 4 fotos, ID 11 uma caixa, ID 12 nenhuma e ID 13 ainda nenhuma**. Conferir no registro atual do PC antes de treinar.

## 2. Redundâncias e sobreposições

| Par | Diagnóstico | Decisão |
| --- | --- | --- |
| 8 `pedaco_telha` × 10 `pedaco_telha_sobreposto` | 10 é um subcaso de 8 que só muda o apoio do fragmento. Com 6 caixas em 4 fotos, o detector dificilmente separa os dois; o erro mais comum vai ser 8↔10. | **Manter os dois IDs** (estão nos pesos `56038eb9`). Candidato a consolidação na próxima versão da taxonomia: uma classe `pedaco_telha` com atributo "sobre telha instalada" no registro, em vez de duas classes. Decisão humana; não converter caixas automaticamente. |
| 2 `residuos_telha` × 8 `pedaco_telha` × 17 `objeto_estranho_cobertura` (fase 2) | Três graus de "coisa sobre a telha": acúmulo sem identificação, fragmento de telha reconhecível, objeto alheio ao sistema. | Não é redundância: a política do ID 8 já proíbe caixa dupla. Falta ativar o 17 quando houver exemplos (a proposta de 04/10 em [novas_classes_propostas.md](novas_classes_propostas.md) encontrou tábuas, sacaria e perfis em fotos da internet). |
| 4/5/9/12 (rufo ausente, quebrado, deslocado, reparado) | Conjunto consistente por elemento e por tipo de evidência. | Manter. |
| 3 `reparo_telha` × 12 `reparo_rufo` × 22 `reparo_selante_fixador_telha` | Mesma evidência ("intervenção") em três elementos. | Manter: o elemento muda a recomendação no laudo. |
| 11 `fixador_telha_frouxo` × 20/21/22 (ausente, corroído, selante) | Fixadores exigem vista próxima; em fotos de 20 MP a 30–40 m um parafuso tem ~10 px. | Manter os nomes, mas registrar a dependência: só a inferência fatiada (`tiling`) e voos baixos dão exemplos úteis. |
| 13 `pedaco_calha_sobre_telha` × 9 `rufo_deslocado` × 24 `calha_danificada` | Perfis metálicos soltos ou deformados se confundem na foto. | Política do ID 13 já separa; 24 cobre a calha **instalada** deformada, o que nenhuma classe cobria. |

Nomes: os slugs seguem o padrão `elemento_condicao` em português snake_case e foram mantidos. A única inconsistência é de estilo entre `pedaco_telha_sobreposto` e `pedaco_calha_sobre_telha`; trocar slug de classe treinada invalida a conferência `check_domain_names` dos pesos, então a correção fica para o nome de exibição (editável na plataforma) e para a próxima versão da taxonomia.

## 3. Lacunas e o que foi acrescentado

Fontes: lista de verificação do laudo CEASA que o sistema ainda não cobria (`MANUAL_CHECKLIST` em `laudo_abnt.py`), termos "não catalogados" da busca aberta (`configs/zero_shot_prompts.json`: `rust on roof sheet`, `standing water on roof`, `hole in roof`, `damaged skylight`) e a proposta de 04/10.

| ID | Classe | Nome de exibição | Origem da lacuna |
| --- | --- | --- | --- |
| 24 | `calha_danificada` | Calha danificada | Item "As calhas estão danificadas?" do laudo. |
| 25 | `telha_trincada_ou_perfurada` | Telha trincada ou perfurada | Item "Existem trincas ou furos nas telhas?"; `telha_trincada` da proposta de 04/10; `hole in roof`. |
| 26 | `corrosao_telha` | Corrosão em telha metálica | `rust on roof sheet`; candidata de 04/10 sem foto. |
| 27 | `agua_empocada_cobertura` | Água empoçada sobre a cobertura | Item "Há sinais de acúmulo de água?"; `standing water on roof`. Só lâmina de água **visível**; a alegação excluída `agua_acumulada_calha` continua excluída porque o laudo a inferiu. |
| 28 | `telha_translucida_danificada` | Telha translúcida danificada | `damaged skylight`. |
| 29 | `ralo_obstruido` | Ralo ou bocal de descida obstruído | Saída da calha sem classe própria; resíduos no canal continuam ID 6. |
| 30 | `cumeeira_danificada` | Cumeeira danificada ou deslocada | Cumeeiras só cabiam por aproximação nas classes de rufo. |
| 31 | `residuo_tinta_argamassa` | Resíduo de tinta ou argamassa aderido | Proposta de 04/10 com evidência em foto da internet. |
| 32 | `crescimento_biologico` | Musgo, líquen ou bolor | Proposta de 04/10. |

Todas entram como **fase 2 com `requires_training: true`**: aparecem na seção Classes e no catálogo, não entram em `detection_names`, em `dataset.yaml`, nos pesos nem nas caixas da revisão. `bird nest on roof` ficou de fora (cabe em `objeto_estranho_cobertura`). Sem evidência não entrou classe de "telha deslocada" (ID 19 `telha_sobreposta_sem_travamento` cobre o caso verificável).

### Como levar uma classe à detecção (não muda com este trabalho)

1. Fotos aprovadas com a classe em pelo menos uma edificação (idealmente duas, para a divisão por cena).
2. Nova versão de `configs/taxonomy.json` com snapshot da anterior (como `taxonomy_ceasa_v5_reparo_rufo.json`), promovendo a classe para `phase: 1` **no fim da lista ativa** (o próximo ID ativo é 14) e deslocando o ID de catálogo que ocupava a posição, com `migration` preenchida; `dataset.yaml`, `MVP_NAMES` em `review_dataset.py`, `COLORS`, `CHECKLIST` do laudo e os prompts da busca aberta acompanham.
3. `migrate-review` em cada revisão, `render-review`, revisão humana das fotos para a classe nova.
4. `build-pilot-data`, `train-pilot`, `compare-models` contra `56038eb9`; adoção só com ganho demonstrado e sem regressão por classe.

## 4. Cadastro de classes na plataforma (CRUD)

Seção **Classes** na barra lateral (nona seção) e rotas do servidor local, com o aviso fixo: *"Mudanças nas classes exigem novo ciclo de treinamento para ter efeito nos modelos."*

| Rota | O que faz |
| --- | --- |
| `GET /api/classes` | Todas as classes (taxonomia + propostas): `id`, `name`, `display_name`, `color`, `active`, `in_detector`, `requires_training`, `source`, `criterion`. |
| `POST /api/classes` | Cria classe **proposta** (`name` snake_case único, `display_name`, `color` `#RRGGBB`, `criterion`) com o próximo ID livre. Só grava; não retreina. |
| `PUT /api/classes/<id>` | Edita `display_name`, `color`, `active`; `name` e `criterion` só nas propostas. |
| `DELETE /api/classes/<id>` | Exclusão lógica: `active: false`. Nada é apagado; `PUT {"active": true}` reativa. |

Decisões de projeto:

- **`taxonomy.json` nunca é reescrito pela plataforma.** Revisões, datasets piloto, calibradores, pesos e laudos guardam o SHA-256 desse arquivo; mudar uma cor dentro dele invalidaria tudo de uma vez. As edições ficam em `configs/taxonomy_overrides.json` (`classes` por ID, `proposed`, `history` com autor e data) e são aplicadas por cima ao servir a página.
- IDs são atribuídos uma vez e nunca reaproveitados; o slug das classes da taxonomia não é editável (é o nome que `check_domain_names` confere nos pesos).
- As edições das classes da taxonomia ficam guardadas **pelo slug**, não pelo ID: uma migração desloca IDs do catálogo (o antigo 13 virou 23 na v6) e a edição precisa seguir a classe. As propostas são conciliadas com a taxonomia a cada leitura: proposta cujo slug entrou na taxonomia é dada como absorvida e sai da lista; proposta cujo ID uma expansão da taxonomia ocupou recebe o próximo ID livre (tudo registrado no `history`).
- `in_detector` só é verdadeiro para classe ativa **sem** `requires_training`: o ID 13 da v6 é ativo e aparece como "ativa · exige treino" enquanto os pesos piloto tiverem 13 classes.
- "Inativa" só tira a classe das opções de caixa nova na página servida (a opção fica desabilitada); caixas existentes dessa classe mantêm a categoria ao serem editadas, e `browser_data.json` em disco e os pesos não mudam.
- Aberta direto do disco, a seção lista as classes embutidas e desativa a edição (mesmo padrão das seções Importar e Laudos).
- Testes: `tests/test_class_catalog.py` (regras do cadastro e as quatro rotas, com token).

Prévia da seção (claro, escuro e celular) na pasta compartilhada `plataforma_yolo/classes/`.

## 5. O que este trabalho não faz

- Não treina, não muda `dataset.yaml` nem `MVP_NAMES`; o detector continua com 14 classes e os pesos `56038eb9` seguem válidos.
- Não consolida 8/10 nem converte caixas: decisão humana para a próxima versão da taxonomia.
- Não migra revisões existentes: o PR #12 já exige `migrate-review` para páginas da v5, e isto continua.
- `review_templates/index.html` das inspeções no PC precisa de `refresh-page` para ganhar a seção Classes (como nas versões anteriores da plataforma).
