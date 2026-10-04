# Validação da entrega inicial — 03/10/2026

Este registro separa código entregue, execução observada e etapas pendentes. Não há checkpoint especializado ou métrica de qualidade de detecção de não conformidades nesta entrega.

## Ambiente efetivamente utilizado

- Windows; AMD Ryzen 5 4600G / Radeon Graphics; aproximadamente 31,8 GB de RAM.
- `.venv` local, Python 3.12.14, Ultralytics 8.4.172, Torch 2.14.1+cpu, torchvision 0.29.1+cpu.
- Dispositivo CPU; CUDA indisponível. ONNX 1.23.1 e ONNX Runtime 1.30.0 instalados para exportação.
- Dependências completas em `requirements-lock.txt`. Não houve instalação de pacotes no Python global.
- O projeto configura `.runtime/Ultralytics/settings.json`, caminhos locais e `sync=False` antes de usar o detector. Uma consulta inicial de versão feita antes desse isolamento levou a biblioteca a criar configurações padrão em `%APPDATA%/Ultralytics/settings.json`; a operação normal deste projeto usa o arquivo local. Configurações temporárias criadas dentro do workspace foram removidas.

## Verificações concluídas

| Verificação | Resultado observado | O que permite afirmar |
| --- | --- | --- |
| Leitura documental | Laudo de 37 páginas, plano de 5, DOCX com tabelas, apresentação de 10; administrativos só inventariados | Categorias e escopo rastreados às fontes, sem anotações inventadas |
| Pesquisa oficial | Todas as páginas solicitadas consultadas; YOLO27l em pré-lançamento | YOLO26l disponível como base; não há execução pública YOLO27l verificada |
| Extração do laudo | 48 imagens elegíveis, 38 conteúdos binários únicos; todas as 48 verificadas por Pillow | Fotografias extraídas/decodificáveis e rastreáveis, sem aprovação de uso em treino |
| Testes automatizados | **32 testes aprovados** no `.venv` | Contratos de dataset, classes, revisão e falhas de tuning exercitados |
| Checagem de sintaxe | `compileall` concluído em `src` e `tests` | Arquivos Python compiláveis neste ambiente |
| Dataset vazio | `validate-data` retorna erro e relatório; `train` interrompe antes de carregar modelo | Não há treino silencioso sem dados anotados |
| YOLO27l solicitado | Rejeitado antes de download | Não existe substituição silenciosa de modelo |
| Peso COCO sem demo | Rejeitado por divergência das classes | Objetos gerais não são apresentados como categorias do projeto |
| YOLO26l em foto CEASA p30 | Inferência CPU e evidência/JSONL concluídos | Pipeline de mídia→modelo→evidências funciona; previsões gerais foram inadequadas |
| YOLO26l em foto CEASA p5 | Inferência concluída, sem caixas acima do limiar | Fluxo registra ausência de detecção sem afirmar conformidade |
| Exportação ONNX | Executada em cópia do YOLO26l genérico; batch 1, FP32, 1024, `nms=False`, opset 18, saída `[1,300,6]`; grafo aprovado por `onnx.checker` | Artefato criado; paridade/inferência do runtime de destino ainda não foram medidas |

Na foto p30, os pesos gerais classificaram incorretamente uma região como `oven` e `microwave`. Na foto p5 de telhas quebradas, não produziram caixas no limiar de 0,25. Ambos os resultados têm `mode=demo_pesos_genericos`, severidade vazia e nenhuma associação à taxonomia do projeto. Esses testes mostram a necessidade de ajuste fino e não avaliam um detector especializado.

Na foto p5, o tempo registrado de inferência foi aproximadamente **2.173 ms** em CPU, com entrada de 1024 pixels. É uma única observação local; não é benchmark, latência ponta a ponta ou promessa de operação em vídeo em tempo real.

## Artefatos reais de demonstração

- Fotos e proveniência: `data/reference/ceasa/manifest.json` e `review_queue.csv`.
- Foto p30: `runs/predict_demo_20261003T232315Z_c8b62618/`.
- Foto p5: `runs/predict_demo_20261003T232634Z_f4e86d45/`.
- Exportação: `runs/export_demo_20261003T232634Z_840d545c/model.onnx`, com `export.json`.
- Reprovação do dataset: `runs/dataset_review.json` e `runs/train_20261003T232842Z_7a2f3d12/dataset_validation.json`.

Os nomes de pastas usam UTC; a data deste registro segue o contexto do usuário em America/Sao_Paulo. Demos e modelos ficam fora do versionamento por `.gitignore`, mas estão salvos localmente. Nenhum commit, push ou publicação foi feito.

## Escopo dos testes

Os 24 testes do dataset cobrem IDs/ordem, imagens corruptas/checksum, normalização e limites de caixas, NaN/inf, negativos explícitos, label ausente/órfã/BOM, colisões de stem, diretórios sobrepostos, mapeamento literal `images`→`labels`, tamanho mínimo, JPEG sem marcador final, hashes e grupos entre splits, Unicode/espaços e ausência de escrita/rede.

Os 8 testes dos contratos de saída cobrem pesos/classes gerais, ordem de classes, modelo futuro indisponível, severidade pendente mesmo em confiança alta, separação da demo, ausência de caixas e verificação de resultados de tuning. Os históricos de tuning nesses testes são fixtures; **nenhuma busca real foi executada**.

Uma integração adicional usou dataset temporário com caminhos absolutos Windows, Unicode/espaços e negativo explícito: `prepare_dataset` criou YAML normalizado sem `download`, preservou labels e registrou hashes/contagens. Não carregou modelo nem executou treino.

## Implementado, mas ainda sem execução com dados especializados

Os comandos `train`, `evaluate` e `tune` estão implementados e passaram por revisão de compatibilidade com o código instalado. Não houve fine-tuning real, avaliação por classe ou tuning com o acervo. O código de vídeo usa gerador e índice de frames processados; **não houve teste de vídeo** por falta de arquivo disponível.

As categorias são proposta baseada no laudo. Ainda exigem aprovação de critérios de anotação e dados diversificados. O grupo informado no CSV precisa corresponder de verdade à captura; hashes só capturam duplicatas exatas, não todos os recortes ou frames semelhantes.

## Pendências para o próximo ciclo

1. Revisar fotografias e taxonomia com responsável técnico; obter fotos originais quando disponíveis.
2. Conferir as caixas propostas e tratar ambiguidades; reunir outras edificações/campanhas para um teste independente.
3. Definir ambiente/orçamento de treino, metas por classe e critérios de erro tolerável.
4. Executar fine-tuning; selecionar com `val`; avaliar `test` após fechar parâmetros.
5. Validar generalização, vídeo, paridade ONNX, memória/tempo e eventual processamento em recortes.
6. Após os dados necessários, implementar segmentação, severidade ordinal/CORAL, regras técnicas e relatório técnico. A edição visual e o rascunho de revisão foram entregues na execução abaixo.
7. Reavaliar YOLO27l após lançamento e licenciamento antes de implantação comercial.

As referências normativas citadas nos planos foram registradas, sem verificar vigência/aplicabilidade nem automatizar conclusões normativas. Ferramentas `ai-memory` não estavam expostas nesta execução; `activity_memory_sync.search_memory` foi consultada especificamente e não encontrou histórico pertinente. A execução do hook Stop não foi verificada, e não houve envio manual duplicado de resumo.

## Execução do plano de revisão — 03/10/2026

Esta seção registra a execução posterior ao núcleo inicial acima. [Registro automatizado](../runs/review_validation_20261004T002953Z_a73f01c9/validation.json), [log dos testes](../runs/review_validation_20261004T002953Z_a73f01c9/tests.log) e [guia de execução](execucao_revisao_ceasa.md).

| Verificação | Resultado atual | Limite |
| --- | --- | --- |
| Revisão do CEASA | 38 imagens únicas com duas leituras visuais por IA, feitas por revisores diferentes | Não é aprovação humana nem medição do desempenho do YOLO |
| Propostas | 108 caixas; 23 fotos positivas, 2 negativas e 13 ambíguas | Inclui hipóteses; contagens por foto podem repetir um defeito físico |
| Integridade | Hashes das 48 ocorrências originais e das 38 cópias de visualização conferidos | Hash não identifica toda semelhança visual entre recortes |
| Testes completos | **98 testes passaram**, sem falha, erro ou teste pulado | Usam fixtures e mocks; não provam qualidade das anotações CEASA |
| Sintaxe | Python em `src`/`tests` e JavaScript da página gerada aprovados | Não é avaliação de modelo |
| Importação de feedback | Hash do registro, taxonomia, foto, dimensões, caixas e confirmação integral verificados em testes | Autor é um identificador declarado; não há autenticação de identidade |
| Dados aprovados | Builder testado com versões imutáveis, positivos, negativos, exclusões e grupos | Nenhum dataset real aprovado foi produzido |
| Falta de aprovação real | `build-reviewed-data` recusou o CEASA e não criou `data/versions/ceasa_v1` | A revisão humana continua pendente |
| Guarda de treinamento | `train`, `tune` e `evaluate` bloqueiam dataset estruturalmente válido sem proveniência antes de chamar `load_detector`, verificado por mocks | Nenhum treino, tuning ou avaliação especializada foi executado |
| EXIF e IDs | Rotação/espelhamento EXIF e IDs ativos descontínuos são recusados | Normalização de fotos de celular ainda requer cópia explícita e nova anotação |
| Tela local | Navegação, foto da calha, edição com teclado, rejeição de caixa sem área e confirmação integral verificadas no navegador | Desenho por arraste e download da interface não foram testados nesta rodada; importador foi exercitado por testes |
| Largura estreita | Viewport de 390×844; conteúdo sem rolagem horizontal (375px úteis, conteúdo 375px) | Não houve teste em aparelho físico ou auditoria completa de acessibilidade |
| Modelo preservado | YOLO26l manteve SHA-256 `9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26` | Nenhum peso especializado foi criado |

A página foi aberta em `http://127.0.0.1:8765/`, com servidor estático limitado à pasta de revisão e à própria máquina. O servidor é temporário; `data/reviews/ceasa_v1/index.html` permanece utilizável como arquivo local. A [evidência da tela](../data/reviews/ceasa_v1/review_screen.jpg) mostra a foto da calha e suas propostas.

Os 66 testes adicionais cobrem revisão/importação/renderização (37), dados aprovados (10) e proveniência/integração de treinamento (19). A estrutura pronta com três grupos nos testes é sintética; nenhum grupo real novo foi inventado. O CEASA real continua sendo uma única edificação, com zero aprovações humanas e sem validação/teste independentes.

## Atualização das categorias e revisão v3 — 03/10/2026

Os resultados da seção anterior pertencem à primeira revisão com oito classes. Após os pedidos do usuário, foram adicionadas **Pedaço de telha (ID 8)** e **Rufo deslocado/desalinhado (ID 9)**. [Verificação atual](../runs/review_validation_v3_20261004T004627Z_a2aa9742/validation.json) e [log de 116 testes](../runs/review_validation_v3_20261004T004627Z_a2aa9742/tests.log).

- **116 testes passaram**, sem falha, erro ou teste pulado; Python e JavaScript gerado compiláveis.
- Contrato atual com dez classes ativas e 21 categorias no catálogo; snapshots de oito e nove classes preservados.
- Os registros e as 108 caixas anteriores foram preservados, comparados integralmente; os hashes/dimensões das 38 fotos originais e suas cópias foram novamente verificados.
- Uma caixa candidata ID 9 foi acrescentada no segmento esquerdo da foto p. 8; total atual **109 caixas**. A identidade do elemento e sua condição precisam de validação técnica. A imagem continua ambígua.
- A categoria ID 8 está disponível na interface, com **zero caixas específicas propostas** nesta atualização. Não foram convertidos automaticamente detritos em fragmentos de telha.
- As duas negativas anteriores voltaram à triagem ambígua até que as novas classes sejam examinadas. Estado atual: **23 fotos positivas propostas, 15 ambíguas, zero aprovações humanas**.
- O histórico das duas leituras de IA das oito classes foi preservado. Todas as 38 fotos exigem nova conferência para o contrato ampliado; não foram declaradas integralmente revisadas nas dez classes.
- Os testes confirmam rejeição de checkpoints com oito/nove classes no contrato atual, bloqueio quando falta qualquer classe nova no treino e migração sem transferir aprovações antigas.
- Lotes históricos sem taxonomia declarada são recusados para o contrato ampliado; não é possível consolidá-los silenciosamente como revisão das dez classes.
- O builder real continua bloqueado sem aprovação humana e não criou `data/versions/ceasa_v3`. Os pesos YOLO26l mantiveram o hash anterior; não houve treino especializado.

A tela atual no mesmo endereço local mostra [a revisão v3](../data/reviews/ceasa_v3/index.html), e a [evidência da tela](../data/reviews/ceasa_v3/review_screen.jpg) registra a foto 9, página 8, com a nova caixa candidata. O servidor temporário anterior, criado nesta tarefa, foi substituído pelo da pasta v3 após conferir seu processo; continua limitado a `127.0.0.1`.

## Expansão para 12 classes e revisão v4 — 03/10/2026

Foram cadastradas **Pedaço de telha sobreposto à telha (ID 10)** e **Elemento de fixação de telha solto/frouxo (ID 11)**. O catálogo atual tem 22 categorias, 12 ativas. ID 11 foi promovido da fase 2 e o antigo ID 10 de fixador corroído passou a 21. [Auditoria desta atualização](../runs/review_validation_v4_20261004T013645Z_cff20330/validation.json) e [log dos testes](../runs/review_validation_v4_20261004T013645Z_cff20330/tests.log).

- **124 testes passaram**, sem falhas, erros ou testes pulados. Python e JavaScript da página gerada passaram na verificação de sintaxe.
- Foram conferidos 333 arquivos anteriores/originais/pesos por hash, usando o snapshot v3 para a taxonomia substituída. Revisões v1/v3, fotos originais e checkpoint mantiveram seus bytes.
- As 38 fotos e cópias da nova página foram conferidas por hash e dimensões. **109 caixas preservadas**, 23 fotos positivas propostas, 15 ambíguas, zero negativas, zero aprovações humanas. Novas classes 10/11 têm zero caixas propostas.
- A migração repetida preserva a conferência anterior efetiva dos IDs 0–7 e as duas leituras históricas de IA. Todas as 38 imagens exigem conferência integral das 12 classes; a existência de classes 8/9 na versão anterior não simula uma revisão delas.
- O builder real recusou o registro sem aprovação e não criou o dataset. Os testes verificaram positivos das novas classes, exportação com 12 classes, procedência e rejeição de checkpoints anteriores incompatíveis. Lotes históricos sem taxonomia declarada continuam recusados.
- Pesos YOLO26l preservados, com SHA-256 anterior. Nenhum treinamento especializado, métricas do domínio ou relatório técnico aprovado foi produzido.
- A [página v4](../data/reviews/ceasa_v4/index.html) foi gerada com 12 opções e aviso dinâmico das classes pendentes. **Não houve validação visual nova dessa página no navegador:** a execução automática bloqueou a atualização do servidor local e a tentativa de criar outro servidor. O navegador integrado recusou a abertura `file:` por permitir apenas HTTP/HTTPS; nenhuma alternativa para contornar esse bloqueio foi aplicada.

O processo anterior permaneceu ativo em `127.0.0.1:8765`, servindo a v3, verificado por consulta HTTP. Para usar a v4 agora, o usuário pode abrir seu `index.html` pelo Explorer em seu navegador. Isso não foi executado automaticamente nem declarado como teste de interface. As validações visuais anteriores pertencem à v1/v3, conforme documentadas acima.

## Reparo em rufo e revisão v5 — 03/10/2026

Foi adicionada **Reparo em rufo (ID 12, `reparo_rufo`)**, totalizando 13 classes ativas e 23 categorias no catálogo. IDs ativos 0–11 preservados; o antigo ID 12 de `reparo_selante_fixador_telha`, fase posterior, passou a 22. [Auditoria desta atualização](../runs/review_validation_v5_20261004T020100Z_61a9423b/validation.json) e [log dos testes](../runs/review_validation_v5_20261004T020100Z_61a9423b/tests.log).

- **128 testes passaram**, sem falhas, erros ou testes pulados. Cobrem exportação YOLO/proveniência do ID 12, bloqueio quando falta positivo de reparo em rufo, contrato de checkpoints, snapshots e severidade pendente.
- Foram conferidos **464 arquivos anteriores/originais/pesos por hash**, incluindo revisão v4 e snapshots. Os 38 originais e cópias de visualização foram verificados por hash e dimensões.
- **109 caixas preservadas**, 23 fotos positivas propostas e 15 ambíguas. Zero novas caixas de reparo em rufo, zero aprovações humanas. Todas as 38 fotos aguardam conferência integral das 13 classes; o escopo efetivamente revisado antes das expansões continua IDs 0–7.
- O builder real recusou o registro não aprovado e não criou dataset. Lotes históricos sem taxonomia continuam bloqueados. Pesos genéricos YOLO26l mantiveram o hash anterior; nenhum treinamento especializado ou avaliação do domínio foi realizado.
- A [página v5](../data/reviews/ceasa_v5/index.html) foi gerada com 13 opções e aviso de categorias pendentes. Python e JavaScript da página passaram na verificação de sintaxe. **Não houve nova validação visual no navegador**; os bloqueios anteriores de execução e de protocolo local não foram contornados.

Uma consulta HTTP nesta execução confirmou `ceasa_v3` em `127.0.0.1:8765`. A nova página v5 pode ser aberta pelo usuário via Explorer em seu navegador. A adição da categoria não deve ser confundida com reconhecimento aprendido pelo detector ou aprovação de reparos como não conformidades.

## Incorporação do feedback humano e controles de treinamento — 03/10/2026

As contagens anteriores pertencem a suas versões históricas. A importação atual foi conferida em [validation.json](../runs/feedback_validation_20261004T023559Z_838172e7/validation.json), com resultado de 156 testes aprovados em 10,623 segundos. Os testes exercitam importação, migração da procedência humana, bloqueios de dados e falhas/artefatos de treinamento; fixtures não constituem novas edificações reais ou treinamento do domínio.

| Verificação | Resultado desta execução | Limite |
| --- | --- | --- |
| Feedback e origem | Hash ligado ao registro v3; 36 decisões e 38 originais reconciliados | Dez classes constavam da página antiga |
| Correções humanas históricas | 33 fotos positivas, 108 caixas e 33 labels YOLO preservados | Conferência das três classes posteriores pendente |
| Arquivos anteriores e pesos | 571 arquivos conferidos com hashes anteriores | Pesos genéricos intactos; não houve fine-tuning |
| Fila corrigida de 13 classes | 38 fotos, 115 caixas, 35 positivas na triagem, três ambíguas; 33 correções humanas históricas | Sete caixas das duas fotos sem decisão seguem como propostas; zero aprovações integrais atuais |
| Procedência das aprovações | Autor, data, notas, taxonomia de origem e escopo preservados | Aprovação histórica não libera categorias posteriores |
| Acervo de anotações históricas | 33 imagens/labels com hashes e coordenadas reconciliados | `ready_for_training=false`; sem YAML ou splits fictícios |
| Builder real | Recusou o registro atual sem aprovação compatível; nenhuma pasta de dataset criada | Classes 10–12 sem exemplos e uma única edificação continuam sendo lacunas |
| Página corrigida | Dados, 38 cópias e evidências conferidos; JavaScript e fontes Python com sintaxe válida | Sem nova validação visual no navegador; servidor não atualizado |
| Conclusão de treino | Testes cobrem ausência de pesos/CSV, contrato errado, métricas/perdas inválidas e falhas de execução | Arquivos de pesos presentes não comprovam funcionamento ou qualidade; nenhuma execução real especializada foi realizada |

A [página corrigida](../data/reviews/ceasa_v6_corrigida_a36ea7f67d08/index.html) pode ser aberta como arquivo local pelo usuário. A atualização do servidor não foi tentada diante dos bloqueios anteriores; o endereço temporário continua vinculado à revisão v3. O [resultado da revisão humana](resultado_revisao_humana_ceasa.md), o [plano de coleta](plano_coleta_imagens.md) e o [protocolo de novas imagens](novas_imagens_e_anomalias.md) descrevem os próximos passos e limites.

**Não validado:** detecção aprendida nas 13 categorias, métricas em edificações independentes, entrada automatizada geral de novas campanhas, descoberta de anomalias inéditas, checkpoint especializado funcional, exportação/paridade e geração de laudo técnico final. As melhorias concluídas são no acervo revisado, no registro das aprovações e nos controles de treinamento.

## Acesso à página de revisão — 03/10/2026

A captura de tela vazia foi investigada. O inventário de abas confirmou abertura direta de `src/dronecamp_ia/review_templates/index.html`. Esse modelo contém o marcador `__REVIEW_DATA__`; sua execução direta gera `ReferenceError` antes de preencher fila e fotografia. O arquivo gerado da revisão tem os dados preenchidos.

Foi aberto no Edge `data/reviews/ceasa_v6_corrigida_a36ea7f67d08/index.html`, e a URL da nova aba foi confirmada. Na pasta principal foi criado [Abrir revisao DroneCamp.cmd](../../Abrir%20revisao%20DroneCamp.cmd), com caminho relativo à sua própria localização e verificação de existência da página. O README passou a orientar esse acesso.

[Verificação desta correção de acesso](../runs/review_opening_fix_20261004T024452Z_040b147c/validation.json): 38 imagens presentes com hashes corretos, 13 categorias, registro e pesos iguais à verificação anterior, e 173 links locais conferidos antes desta nota. Não foi necessário alterar a interface, regenerar a revisão ou reimportar decisões. A abertura da aba foi confirmada; a interação visual com a página local não foi automatizada nesta execução.
