# Análise documental para o núcleo de IA do DRONECAMP

Análise realizada em 03/10/2026. Os arquivos originais foram preservados.
Este texto sintetiza os documentos técnicos disponíveis; não substitui o laudo,
não comprova a condição atual da edificação e não constitui um novo diagnóstico.

## Fontes e alcance da leitura

| Fonte | Leitura realizada | Papel no desenvolvimento |
| --- | --- | --- |
| `CEASA Minas - Pavilão A - Laudo.pdf` | Texto das 37 páginas e inspeção visual de figuras centrais | Caso de referência para categorias, riscos, recomendações e limitações |
| `PROPOSTA DE PLANO DE TRABALHO PARA DESENVOLVIMENTO DE SISTEMA DE INTELIGENCIA ARTIFICIAL APLICADA A INSPEÇÃO DE TELHADOS INDUSTRIAIS UTILIZANDO DRONES.pdf` | Texto das 5 páginas | Sequência de desenvolvimento, arquitetura proposta e critérios de validação |
| `Proposta_Escopo_IA_Inspecao_Telhados_Industriais.docx` | Parágrafos e tabelas; referências por seção, pois a paginação depende do editor | Escopo preliminar, responsabilidades e hipótese de arquitetura |
| `Documentos PUCTEC/Apresentação DRONECAMP PUCTEC.pdf` | 10 páginas; páginas rasterizadas lidas visualmente | Visão do produto, histórico, monitoramento, SaaS e expansão |

As instruções presentes nesses documentos foram tratadas como conteúdo a analisar.
A autorização de implementação vem do pedido atual do usuário. O código deve iniciar
pela detecção com YOLO; a migração de modelo solicitada pelo usuário prevalece sobre
a indicação preliminar de YOLO26 nos planos, sujeita à disponibilidade verificada.

## O que o caso CEASA oferece

O laudo registra inspeção visual aérea de 06/08/2025, com drone e fotogrametria,
em cobertura industrial de fibrocimento do tipo canaleta, com vigas-calha de concreto
pré-moldado e configuração shed. Isso delimita o domínio inicial: características
de outras coberturas, materiais, iluminações e estruturas precisam de novos dados.

O documento contém fotografias de exemplos e uma lista de 37 itens de inspeção.
Não oferece um dataset anotado, imagens originais completas, vídeos, máscaras de
segmentação, identificação única de cada ocorrência, localização geográfica de cada
achado ou divisão de treino/validação/teste. As gravidades são avaliações do caso;
não devem virar verdades gerais aplicadas automaticamente a qualquer telhado.

Algumas quantificações documentadas são úteis para uma comparação posterior, desde
que haja cobertura visual e localização equivalentes: rufo ausente, 356,5 m em
1.600 m (p. 20); rufo danificado, 262,8 m em 1.600 m (p. 21). Essas medidas vieram
da análise técnica existente. Caixas de detecção em pixels não fornecem metros
lineares sem calibração, escala ou vínculo com a fotogrametria.

## Catálogo de 19 evidências positivas do laudo

Este catálogo fundamenta a taxonomia inicial. As classes do modelo são evidências
visuais; a conformidade, a gravidade, o risco e a prioridade são campos separados.
A coluna de gravidade reproduz a referência contextual do CEASA, sem automatizar
sua adoção. Todos os resultados de IA começam como candidatos a revisão humana.

| Categoria proposta | Fonte no laudo | Gravidade no caso | Critério visual para anotar | Confusões e limites |
| --- | --- | --- | --- | --- |
| Telha quebrada | item 7, p. 5, fig. 5 | Gravíssima | Ruptura da peça instalada, perda parcial de material, borda fraturada ou abertura com parte da telha remanescente | Fragmento solto sobre telha íntegra; trinca sem abertura; sombra entre planos; delimitar o dano e registrar contexto |
| Telha ausente | item 8, p. 5–7, figs. 6–7 | Gravíssima | Trecho onde deveria existir a peça e há abertura completa, com sequência de cobertura interrompida | Calha, vão projetado, borda do telhado e grande quebra; a ausência exige contexto de geometria esperada |
| Resíduos sobre telha | item 10, p. 7–8, figs. 8–10 | Média | Acúmulo ou fragmentos soltos apoiados sobre a telha, distinguíveis da peça instalada | Remendos colados/parafusados, telha quebrada instalada, manchas e fuligem; legenda não substitui revisão do objeto |
| Reparo de telha | item 13, p. 9–13, figs. 11–17 | Baixa | Área de intervenção com fita, manta, selante ou remendo de outra peça | Resíduo solto, sombra, mudança de material; presença de reparo não prova vazamento nem reprovação geral |
| Rufo ausente | item 20, p. 20, fig. 27 | Grave | Interrupção do fechamento esperado no encontro/borda do plano, com trecho exposto | Abertura projetada de ventilação, sombra e rufo quebrado; exige contexto da linha de rufos |
| Rufo quebrado | item 21, p. 21–24, figs. 28–35 | Gravíssima | Fratura ou perda parcial do rufo instalado, com parte da peça ainda reconhecível | Telha quebrada, fragmento de rufo solto e rufo completamente ausente; casos mistos precisam de regra de anotação |
| Resíduos em calha | item 28, p. 30–31, figs. 41–42 | Gravíssima | Detritos acumulados dentro da região reconhecível de calha | Resíduos sobre telha, vegetação, fundo escuro ou sombra; localizar a calha e não afirmar obstrução hidráulica medida |
| Vegetação em calha | item 29, p. 31, fig. 43 | Grave | Folhas, ramos ou vegetação seca identificável dentro da calha | Cabo, sombra, detritos finos, planta no entorno; a contagem do laudo está pendente |
| Telha sobreposta sem travamento | item 4, p. 3, fig. 2 | Média | Peça nova sobre peça antiga, com sobreposição anormal visível | Sobreposição normal de projeto e remendo; ausência de travamento não é demonstrável quando o fixador está oculto |
| Fixador ausente em telha | item 15, p. 15–16, figs. 20–21 | Média | Posição esperada com furo exposto ou reparo que substitui a fixação, confirmada por especialista | Fixador pequeno/oculto, furo vedado com fita, resolução insuficiente; não inferir ausência de qualquer ponto sem referência |
| Fixador corroído em telha | item 16, p. 16, fig. 22 | Média | Oxidação aparente no fixador, com detalhe suficiente para distingui-la | Sujeira, pintura, ferrugem de componente vizinho e balanço de branco; foto não determina resistência residual |
| Fixador aparentemente frouxo em telha | item 17, p. 17, fig. 23 | Média | Levantamento, inclinação ou afastamento visível do fixador em relação à peça | Parafuso solto sobre a superfície e perspectiva; torque/fixação exigem verificação física, não apenas RGB |
| Reparo de selante em fixador de telha | item 18, p. 18, fig. 24 | Baixa | Camada de selante sobre fixador, com região visível | Arruela, fita sobre furo sem fixador e pintura; reparo não comprova falha de estanqueidade |
| Fixador ausente em rufo | item 22, p. 25, fig. 36 | Média | Furo ou posição esperada sem parafuso/rebite, validada com detalhe suficiente | Fixador oculto, material quebrado e padrão variável; requer recorte aproximado |
| Junta de rufo sem selante | item 23, p. 26, fig. 37 | Média | Junta aberta ou linha de encontro em que a falta de selante é visualmente confirmável | Selante transparente/oculto, junta projetada e sombra; RGB distante pode ser inconclusivo |
| Falha/ausência de selante em fixador de rufo | item 24, p. 27, fig. 38 | Média | Fixador exposto ou selante visivelmente interrompido | Selante transparente, arruela e sujeira; ausência só com visibilidade suficiente |
| Fixador corroído em rufo | item 26, p. 28, fig. 39 | Média | Oxidação aparente no fixador do rufo | Confusões de cor/material; não prova rompimento nem risco estrutural quantitativo |
| Objeto estranho sobre cobertura | item 32, p. 33, figs. 44–45 | Baixa | Objeto solto que não pertence à cobertura; exemplos documentados: pipa e parafuso solto | Fixador instalado, cabo, antena funcional e fragmentos; distinguir de resíduos para evitar dupla contagem |
| Passagem/antena sem vedação aparente | item 34, p. 35, figs. 47–48 | Média | Elemento atravessando a telha e abertura sem vedação visível | Passagem selada, fixador e selante oculto; o mesmo item documenta uma passagem selada e outra não selada |

Os oito primeiros itens compõem o MVP de detecção, por apresentarem áreas visuais
maiores e exemplos no laudo. Isso é uma escolha de implementação, não uma prova
de que haverá amostras suficientes. Os outros 11 entram em fase posterior, após
obter fotografias de detalhe, critérios de anotação e validação de resolução.

### Ordem inicial de classes no MVP

Esta seção registra a análise documental inicial. A taxonomia atual v5 preserva os IDs ativos 0–11 e acrescenta `reparo_rufo` (12). A v4 acrescentou `pedaco_telha_sobreposto` (10) e promoveu `fixador_telha_frouxo` (11) da fase posterior, preservando seu significado. O contrato atual tem 13 classes ativas e 23 categorias no catálogo. Os antigos IDs de fase 2 8, 9 e 10 foram movidos para 19, 20 e 21; o antigo ID 12 de `reparo_selante_fixador_telha` passa ao ID 22, com snapshots e migração explícita. As decisões do usuário e seus critérios estão em [Pedaço de telha](categoria_pedaco_telha.md), [Rufo deslocado/desalinhado](categoria_rufo_deslocado.md), [Pedaço de telha sobreposto à telha](categoria_pedaco_telha_sobreposto.md), [Elemento de fixação de telha solto/frouxo](categoria_fixador_telha_frouxo.md) e [Reparo em rufo](categoria_reparo_rufo.md). Reparo em rufo é uma intervenção visual, cadastrada por pedido do usuário, sem aprovação de ocorrência ou gravidade herdada do laudo. A tabela abaixo preserva os oito IDs da análise inicial; não representa o contrato completo atual.

| ID | Nome técnico | Rótulo |
| --- | --- | --- |
| 0 | `telha_quebrada` | Telha quebrada |
| 1 | `telha_ausente` | Telha ausente |
| 2 | `residuos_telha` | Resíduos sobre telha |
| 3 | `reparo_telha` | Reparo de telha |
| 4 | `rufo_ausente` | Rufo ausente |
| 5 | `rufo_quebrado` | Rufo quebrado |
| 6 | `residuos_calha` | Resíduos em calha |
| 7 | `vegetacao_calha` | Vegetação em calha |

A ordem precisa ser idêntica no dataset, configuração e checkpoint. Alterar IDs
depois de anotar ou treinar exige uma migração explícita das etiquetas. Classes
de fase posterior não devem receber IDs improvisados no modelo do MVP.

### Subtipos úteis sem multiplicar classes cedo demais

O item 13, p. 9–13, registra reparos com pedaços de telha e PU, pedaços parafusados,
fita aluminizada, manta asfáltica e pedaços sem fixação por parafuso. No MVP, anotar
uma classe `reparo_telha` e manter o subtipo como metadado revisável evita fragmentar
um conjunto pequeno. Só separar os subtipos como classes após avaliar a quantidade,
visibilidade e benefício operacional de cada um.

Para resíduos, registrar composição aparente e agrupamento; a regra de anotação deve
definir quando uma pilha é uma ocorrência e quando os fragmentos são separados.
Para rufos, definir como dividir um trecho contínuo e como tratar quebras parciais
junto a ausência completa. Sem isso, contagens entre anotadores divergem.

## Itens que não viram positivos de treinamento com este laudo

| Item | Fonte | Estado documental | Consequência para IA |
| --- | --- | --- | --- |
| Inclinação da cobertura | p. 2–3 | Conforme segundo medição fotogramétrica do caso | Não deduzir inclinação normativa a partir de uma caixa RGB; guardar medição externa |
| Tirantes/contraventamento | p. 4–5 | Conforme | Exemplos de contexto; não há falha positiva documentada |
| Incompatibilidade de peças | p. 5 | Não identificada | Não inventar exemplos positivos |
| Trinca ou furo aberto em telha | p. 7 | Não identificado no item específico | Novas categorias podem ser úteis, mas este item não fornece positivos; figura com remendo não equivale a trinca exposta |
| Vegetação diretamente sobre telha | p. 9 | Não identificada | Vegetação em calha é categoria distinta |
| Água acumulada sobre telha | p. 9 | Não identificada | Não usar mancha/sombra como água sem validação |
| Posição dos fixadores de telha | p. 14 | Conforme | Caso negativo/contextual; conformidade depende da especificação aplicável |
| Fixador frouxo em rufo | p. 27 | Não identificado | Separar do achado positivo de fixador frouxo em telha |
| Água acumulada em calha | p. 32 | Inferência indireta por resíduos/vegetação; poça não observada diretamente | Não criar rótulo positivo de água nem converter resíduos em prova visual de água |
| Calha danificada | p. 32 | Não identificada | Não há exemplo positivo |
| Chaminé/juntas | p. 34 | Conformidade não constatável, por sujidade/fuligem | Registrar visibilidade insuficiente e necessidade de limpeza/inspeção; não inferir falha na junta |
| Árvores e vizinhança elevadas | p. 36 | Não se aplica no caso | Contexto externo, não defeito da cobertura |

Os estados `conforme`, `não conformidade`, `inconclusivo`, `não avaliado` e `não se
aplica` precisam permanecer separados. Não detectar um defeito significa apenas
ausência de detecção naquela imagem e configuração; não certifica conformidade.

## Gravidade, risco e prioridade

O laudo utiliza baixa, média, grave e gravíssima. A ordenação desses quatro níveis
pode apoiar futuramente um classificador ordinal, mas não há aqui um conjunto
de ocorrências anotadas com variação de gravidade suficiente para treinar CORAL.
Classificar toda ocorrência de uma classe com a gravidade do laudo tornaria o
classificador redundante e ignoraria o contexto do imóvel e da consequência.

No núcleo inicial, manter:

- a classe visual e a confiança numérica do detector;
- a gravidade do caso de referência com fonte/página;
- a gravidade revisada pelo responsável, inicialmente não avaliada;
- risco e prioridade separados, com justificativa e origem da decisão;
- o estado de revisão humana, o identificador da evidência e a correção de classe.

Confiança do modelo não é probabilidade de segurança, gravidade ou conformidade.
A prioridade de trabalho de IA também difere da prioridade de manutenção: o MVP
começa pelas classes com maior viabilidade visual; o responsável técnico define
as ações de manutenção conforme risco, contexto e inspeção complementar.

O laudo recomenda correção imediata das falhas e, especificamente, substituição dos
rufos de fibrocimento por perfis metálicos (p. 37). Isso é recomendação histórica
do caso CEASA. Não foram definidos aqui prazos de atendimento, orçamento ou
intervenções atuais. Estanqueidade, torque, capacidade estrutural e causa raiz
não são demonstrados automaticamente por imagens RGB.

## Plano de trabalho e arquitetura proposta

O PDF do plano, p. 1–2, e o DOCX, seções 1–3, propõem receber fotos/vídeos, detectar,
segmentar e classificar ocorrências, apresentar evidências demarcadas/metadados,
associar riscos e critérios, ajudar na elaboração de laudos e registrar revisão
humana. O DOCX descreve a solução como apoio à decisão técnica profissional.

A arquitetura documental é composta por quatro estágios:

| Estágio | Proposta original | Decisão para a primeira implementação |
| --- | --- | --- |
| 1. Visão computacional | YOLO26 com fine-tuning; DOCX cita YOLO26-seg | Iniciar detecção com caixas e configuração de modelo verificável; modelo do pedido atual tratado explicitamente |
| 2. Severidade | Classificador dedicado e regressão ordinal CORAL | Adiar treinamento; armazenar revisão e gravidade referenciada sem simular modelo treinado |
| 3. Engenharia | Regras técnicas, contexto, criticidade e requisitos | Manter critérios/fonte revisáveis; não emitir conclusão normativa automática |
| 4. Relatório | Evidências consolidadas e validação humana | Produzir saída rastreável para revisão, ainda sem emissão de laudo profissional |

O DOCX, seção 3 e observação técnica final, chama a arquitetura de hipótese inicial,
a confirmar após analisar os dados, as classes, a qualidade, as métricas e os
critérios de aceite. Assim, segmentação e CORAL não são pré-requisitos para provar
primeiro a detecção, e a seleção definitiva do modelo deve vir de experimentos.

### Sequência documentada

O plano PDF, p. 3, e o DOCX, seção 5, apresentam esta progressão:

1. Reunir acervo; estabelecer taxonomia, amostras, anotação e controle de qualidade.
2. Definir requisitos, arquitetura, resolução e critérios de severidade.
3. Treinar/ajustar a visão computacional, inferir sobre imagens/vídeos e marcar evidências.
4. Evoluir classificação de severidade, CORAL e regras de engenharia.
5. Estruturar registros técnicos e laudo assistido.
6. Comparar contra o caso CEASA, com achados e omissões rastreáveis.
7. Testar em ambiente controlado e executar piloto real.
8. Evoluir infraestrutura, APIs, armazenamento, segurança e atendimento em escala.

Os documentos não fornecem cronograma detalhado, orçamento, quantidade de imagens,
meta numérica de desempenho, VRAM exigida, política de captura ou escolha definitiva
de ambiente de execução. Esses campos permanecem pendentes; não foram preenchidos
por suposição.

### Validação prevista e critérios a especificar

O plano PDF, p. 4, e o DOCX, seção 6, pedem qualidade de detecção/segmentação,
cobertura de categorias, consistência de severidade, rastreabilidade, concordâncias
e divergências contra o laudo, custo/tempo de processamento e limites explícitos.

Para materializar isso, o desenvolvimento precisa documentar métricas por classe,
recall e falsos negativos nas ocorrências críticas, precisão/falsos positivos,
matriz de confusão, latência com hardware identificado e resultados em dados
separados por inspeção. As metas numéricas precisam ser pactuadas com o responsável
técnico após estabelecer uma referência inicial. Nenhuma meta foi inventada.

Comparar apenas com o laudo não mede exaustivamente o detector: várias ocorrências
não foram contadas, algumas figuras se repetem e parte das conclusões depende de
medições ou inferências. É necessário revisar imagens completas e construir
anotações de referência independentes, incluindo regiões sem defeito.

## Visão de produto da apresentação

| Páginas | Conteúdo observado | Implicação de escopo |
| --- | --- | --- |
| 2 | Risco, custo e tempo das inspeções; problemas em relatórios e manutenção corretiva | Benefícios pretendidos; não metas quantitativas comprovadas |
| 3 | Drone, fotogrametria, visão computacional, IA, dashboard e cliente | Integração futura de captura, análise e tomada de decisão |
| 4 | Histórico rastreável, comparação, gestão de risco e plataforma escalável | Preservar IDs de inspeção, evidências e versões para evolução |
| 5 | Setores industriais e posicionamento de mercado | Visão comercial ampla; o dataset inicial continua limitado ao CEASA |
| 6 | Assinatura, banco histórico, monitoramento e renovação; serviços e SaaS | Comparação longitudinal requer localização/identidade de achados entre inspeções |
| 7 | Rede de operadores, padrão de captura e nuvem | Padronização de aquisição e controle de acesso antes de expansão |
| 8 | MVP em desenvolvimento; roteiro MVP → IA → SaaS → expansão | Construção por etapas; o núcleo atual precisa de validação própria |
| 9–10 | Nuvem, manutenção preditiva e benefícios de negócio | Manutenção preditiva exige histórico temporal; detecção isolada não demonstra previsão |

A expressão de tecnologia validada na p. 8 é uma afirmação da apresentação.
Ela não vem acompanhada de dataset, resultados por classe, teste independente
ou checkpoint nesta pasta. Não comprova um modelo de IA já treinado ou validado.

## Cuidados de dados antes do fine-tuning

- Solicitar/organizar fotografias e vídeos originais do drone em sua resolução
  original e com identificador de voo/inspeção. As figuras do PDF não são o acervo.
- Não transformar círculos e setas vermelhos do laudo em dados de treino: o modelo
  pode aprender a marcação gráfica. Usar originais limpos e anotações separadas.
- Evitar vazamento de dados: várias figuras mostram as mesmas cenas em perguntas
  diferentes. Frames consecutivos, recortes da mesma imagem e vistas da mesma
  ocorrência devem ficar no mesmo grupo ao separar treino, validação e teste.
- Não extrapolar o tamanho do defeito em metros a partir de pixels sem escala,
  calibração ou vínculo fotogramétrico verificável.
- Medir se parafusos, vedantes e trincas têm pixels suficientes. Objetos pequenos
  podem exigir imagens próximas, recortes ou segmentação; redimensionar uma foto
  de contexto não recupera detalhe inexistente.
- Anotar todas as classes do escopo em cada imagem usada; não rotular só o círculo
  do relatório quando há outras ocorrências visíveis. Não anotar suposição causal.
- Separar categoria visual de subtipo, material, gravidade, teste físico e causa.
  Marcar `inconclusivo` quando faltarem visibilidade ou contexto.
- Revisar uma amostra entre anotadores e registrar divergências antes de ampliar
  o acervo. Preservar as correções humanas para nova versão dos rótulos e do modelo.

## Lacunas e pendências concretas

1. Não há imagens/vídeos originais nem dataset YOLO anotado na pasta inventariada.
2. Não há checkpoint especializado, métricas de modelo ou evidência de treino prévio.
3. O item vegetação em calha contém `X` em vez de uma contagem na p. 31; o total não
   foi inferido nem substituído pelo número de figuras.
4. Água em calha, p. 32, é inferência indireta; faltam positivos de poças observadas.
5. Fixação, vedação e estanqueidade exigem detalhe ou inspeção complementar; não
   são conclusões gerais certificadas pelo detector.
6. Falta protocolo de anotação detalhado para dano parcial, ausência, resíduos,
   reparos e limites de trechos de rufos/calhas.
7. Faltam imagens negativas e outros imóveis para medir generalização.
8. Faltam metas de aceite, hardware alvo, política de retenção e integração de
   revisão humana com o responsável técnico.
9. As normas citadas no plano são referências propostas; não foram auditadas aqui
   quanto a edição vigente, conteúdo licenciado e aplicabilidade. Não foi criado
   mapeamento automático entre defeito e infração normativa.
10. As recomendações do laudo e as afirmações comerciais não demonstram resultados
    atuais do novo sistema. Validação de código não substitui desempenho do modelo.

## Inventário administrativo sem leitura de conteúdo pessoal

Os arquivos abaixo foram apenas inventariados por nome e tipo, sem extrair
identificadores, dados bancários, notas acadêmicas, assinaturas ou dados pessoais.
Eles não participam da taxonomia, do treinamento ou da implementação de IA.

| Arquivo em `Documentos PUCTEC` | Tipo/uso aparente pelo nome |
| --- | --- |
| `CarteiraDeIdentidade&CPF.pdf` | Identificação pessoal, PDF |
| `Curriculo_Claudio_Francisco_Auxiliar_TI.pdf` | Currículo, PDF |
| `HistoricoEscolar.pdf` | Histórico escolar, PDF |
| `Formulário de abertura da conta SANTANDER_PROPPg.pdf` | Formulário bancário, PDF |
| `Formulario_de_abertura_da_conta_SANTANDER_PROPPg_assinado.pdf` | Formulário bancário assinado, PDF |
| `Termo de Compromisso Bolsista startup - 2026.docx` | Termo de compromisso, DOCX |
| `Termo de Compromisso Bolsista startup - 2026.pdf` | Termo de compromisso, PDF |
| `Termo_de_Compromisso_Bolsista_startup_-_2026_assinado.pdf` | Termo de compromisso assinado, PDF |

Nenhum conteúdo administrativo foi convertido em resumo técnico ou exemplo de IA.
