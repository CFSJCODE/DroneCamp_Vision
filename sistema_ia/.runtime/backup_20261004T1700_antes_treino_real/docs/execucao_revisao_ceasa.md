# Execução da revisão do CEASA

Data: 03/10/2026. A taxonomia atual tem **13 classes ativas**. A importação do arquivo humano da v3 preservou 33 fotos positivas com 108 caixas nas dez classes anteriores. A [página corrigida atual](../data/reviews/ceasa_v6_corrigida_a36ea7f67d08/index.html) pede a conferência complementar das três classes posteriores. O resultado, cobertura por classe e comandos atuais estão em [incorporação da revisão humana](resultado_revisao_humana_ceasa.md); as metas propostas estão no [plano de coleta](plano_coleta_imagens.md). **O detector ainda não foi treinado nas categorias do DroneCamp.**

As seções abaixo documentam a execução histórica até a revisão v5, antes dessa importação. Seus totais e comandos pertencem àquela versão; para trabalhar sobre as correções humanas já incorporadas, use o guia atual acima.

## Histórico da execução até a revisão v5

As **48 ocorrências de imagens incorporadas ao PDF** correspondem a **38 arquivos de conteúdo único**, identificados por hash. As repetições continuam vinculadas às páginas do laudo; não viraram novas amostras independentes.

Na taxonomia original, com oito classes, cada uma das 38 imagens recebeu uma primeira leitura visual por IA e uma segunda leitura por outro revisor de IA. Esse histórico permanece em [registry_second_pass.json da v1](../data/reviews/ceasa_v1/registry_second_pass.json): 108 caixas, 23 fotos positivas propostas, 2 negativas e 13 ambíguas, sem aprovação humana. Os originais e registros antigos foram preservados. **Duas leituras de IA não substituem aprovação humana ou avaliação de engenharia e não comprovam revisão das novas classes.**

O registro dessa etapa é [registry.json da v5](../data/reviews/ceasa_v5/registry.json), e suas contagens estão em [review_summary.json da v5](../data/reviews/ceasa_v5/review_summary.json). A v3 acrescentou um candidato de rufo deslocado/desalinhado às 108 caixas históricas. A v4 acrescentou ID 10 e ativou ID 11. A migração para v5 acrescentou Reparo em rufo, ID 12, preservando as 109 propostas, sem conversão de classes ou novas caixas:

| Decisão visual proposta | Imagens |
| --- | ---: |
| Positiva: há ocorrências delimitadas | 23 |
| Negativa revisada para as 13 classes | 0 |
| Ambígua: alguma classe, limite ou interpretação ainda precisa de decisão | 15 |
| Excluída | 0 |
| Total | **38** |
| Com aprovação humana | **0** |
| Aguardando conferência integral da taxonomia v5 | **38** |

Há **109 caixas propostas**, incluindo hipóteses nas imagens ambíguas:

| ID | Categoria | Caixas propostas |
| ---: | --- | ---: |
| 0 | Telha quebrada | 15 |
| 1 | Telha ausente | 3 |
| 2 | Resíduos sobre telhas | 33 |
| 3 | Reparo em telha | 19 |
| 4 | Rufo ausente | 3 |
| 5 | Rufo quebrado | 21 |
| 6 | Resíduos em calha | 10 |
| 7 | Vegetação em calha | 4 |
| 8 | Pedaço de telha | 0 |
| 9 | Rufo deslocado/desalinhado | 1 |
| 10 | Pedaço de telha sobreposto à telha | 0 |
| 11 | Elemento de fixação de telha solto/frouxo | 0 |
| 12 | Reparo em rufo | 0 |
| | Total | **109** |

Esses números contam caixas por fotografia. Fotos e recortes relacionados podem mostrar o mesmo defeito físico. Não são medições de precisão, recall ou desempenho do YOLO; as caixas foram propostas pela revisão visual por IA e ainda não constituem dados aprovados para treino. As gravidades permanecem vazias.

As duas fotos antes propostas como negativas passaram a ambíguas até que se confira também a ausência das novas classes. Todas as 38 fotos, inclusive as positivas, precisam ser examinadas integralmente para a taxonomia v5. O registro conserva as duas leituras históricas das oito classes e sinaliza essa conferência pendente.

[Pedaço de telha](categoria_pedaco_telha.md), ID 8, foi criado a pedido do usuário para fragmentos soltos visualmente reconhecíveis. Nenhuma caixa dessa classe foi proposta no registro atual; é necessário procurar ocorrências e revisar as caixas antigas de resíduos sem converter todas automaticamente. [Rufo deslocado/desalinhado](categoria_rufo_deslocado.md), ID 9, atende ao outro recorte discutido. Seu único candidato está na extremidade esquerda de `p008_img01_d12131dd.jpg`, página 8, em pixels `[0, 133, 223, 202]`. A identidade como rufo e a distinção entre junta, desprendimento e ruptura ainda precisam de confirmação; a imagem permanece ambígua.

[Pedaço de telha sobreposto à telha](categoria_pedaco_telha_sobreposto.md), ID 10, identifica o fragmento solto apoiado sobre telha instalada. Prefira-o ao ID 8 quando esse apoio estiver reconhecível e não anote o mesmo objeto com as duas classes. [Elemento de fixação de telha solto/frouxo](categoria_fixador_telha_frouxo.md), ID 11, exige detalhe suficiente para reconhecer o fixador da telha e soltura, deslocamento ou folga aparente; não mede torque. Um fixador avulso sem vínculo reconhecível com a telha permanece ambíguo. Ambas têm zero caixas propostas no registro atual.

[Reparo em rufo](categoria_reparo_rufo.md), ID 12, descreve uma intervenção identificável no próprio rufo: fita, manta, remendo ou aplicação de selante reconhecível como reparo. Delimite a intervenção; não converta automaticamente caixas de reparo em telha, ID 3. Selante regular de montagem não comprova reparo. Reparo não declara falha, vazamento, causa, aderência ou gravidade. Ruptura e deslocamento exigem evidências próprias, sem duplicar caixa idêntica do mesmo objeto por inferência. A nova classe tem zero caixas propostas nesta migração.

Os IDs ativos 0–11 mantiveram seus significados na v5. Na v4, os antigos IDs 8, 9 e 10 do catálogo de fase 2 já haviam sido deslocados para 19, 20 e 21; ID 11 de `fixador_telha_frouxo` manteve seu significado e foi promovido à fase ativa. Para reservar o novo ID 12, a categoria de fase posterior `reparo_selante_fixador_telha` passa ao ID 22. O catálogo tem 23 categorias e 13 ativas. Os snapshots de taxonomia e os metadados de migração do registro permitem interpretar cada versão. Não reinterpretar anotações ou pesos anteriores apenas pelos IDs numéricos.

As dúvidas recorrentes envolvem fragmento solto versus dano instalado, reparo versus objeto sobreposto, depósito versus textura e ausência de rufo versus terminação prevista. Fita sobre rufo não é automaticamente reparo de telha. Misturas genéricas de resíduos não devem receber ID 8 sem um fragmento reconhecível; a mesma caixa do mesmo objeto não deve ser duplicada como ID 8 e ID 2 ou 6. Fixadores frouxos da telha entram em ID 11 somente com evidência suficiente. Presença de porca, arruela ou sombra não comprova soltura. Fixadores ausentes, corroídos, relacionados ao rufo e outros itens fora das 13 classes continuam registrados nas observações.

## Como abrir e revisar

**Estado da abertura nesta execução:** a consulta HTTP confirmou que `http://127.0.0.1:8765/` ainda serve a v3. A atualização automática do servidor foi bloqueada na etapa anterior, quando o navegador integrado também recusou a abertura direta de arquivos locais por restrição de protocolo. Use o Explorer e seu navegador para abrir a página v5 abaixo; a tela v5 não recebeu teste visual no navegador nesta execução.

Abra [a página de revisão v5](../data/reviews/ceasa_v5/index.html) no navegador. No Explorer, ela está em `sistema_ia\data\reviews\ceasa_v5\index.html`. Também é possível abrir pelo PowerShell, a partir da pasta `sistema_ia`:

```powershell
Start-Process ".\data\reviews\ceasa_v5\index.html"
```

A página reúne as 38 fotos, suas páginas de origem, propostas, notas históricas e o candidato novo. Para trabalhar:

1. Preencha **Seu identificador de revisão** com um identificador interno, como `revisor_01`.
2. Selecione uma foto na fila. Use **Mostrar caixas propostas** para alternar a visualização e **Abrir foto original** para examinar detalhes. Leia também **Observações da revisão visual**; a proposta inicial fica disponível em uma seção expansível.
3. Confira a foto inteira, procurando todas as 13 categorias. Não se limite às caixas existentes ou ao título da página do laudo. Examine também as categorias novas e registre dúvida se a identidade do elemento não estiver clara.
4. Para corrigir, selecione uma caixa em **Ocorrências na foto**, escolha a **Categoria da caixa**, ajuste as quatro coordenadas em pixels e clique em **Atualizar caixa**. Para remover, use **Excluir selecionada**.
5. Para uma ocorrência omitida, escolha a categoria e use **Desenhar nova caixa**, arrastando sobre a foto. Você também pode usar **Nova caixa**, preencher as coordenadas e clicar em **Adicionar caixa**.
6. Em **Situação da revisão**, escolha a decisão adequada e descreva correções ou dúvidas. Para aprovar uma imagem positiva ou negativa, marque **Conferi a foto inteira e confirmei as classes e caixas registradas** e clique em **Registrar minha decisão**.
7. Use **Salvar arquivo de revisão** para baixar o JSON com as decisões registradas. O nome sugerido é `dronecamp-revisao-ceasa_v5.json`. Guarde esse arquivo: ele é o resultado transferível da revisão.

Uma imagem negativa precisa estar sem caixas e ter sido examinada integralmente para as 13 classes. Se houver uma região indefinida, escolha **Há dúvida: precisa de análise técnica**; se a foto for inadequada para treinamento, use **Excluir do treinamento** e registre o motivo. Não transforme incerteza em negativo.

Alterar, adicionar ou excluir uma caixa devolve a decisão da foto ao estado pendente e desmarca a confirmação. Depois de editar, registre novamente sua decisão. O navegador mantém um rascunho local; o arquivo exportado é necessário para importar as correções no sistema. O download inclui as fotos com decisão registrada, permitindo revisar em etapas.

## Como registrar uma nova versão

Os comandos abaixo correspondem à interface de comandos atual. Execute-os na pasta `sistema_ia`, usando o ambiente local.

`prepare-review` consolida propostas e auditorias de um lote novo. Use-o somente quando as leituras desse lote estiverem completas para a taxonomia vigente; as duas leituras da v1 cobriam oito classes. Não regenere o histórico antigo com a taxonomia ampliada como se ele já tivesse sido revisado. Para um futuro lote, substitua o caminho abaixo pela sua pasta de propostas e auditorias:

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia prepare-review --directory "data\reviews\NOVO_LOTE_REVISADO" --audits
```

A página atual já foi gerada. Para regenerá-la a partir do registro v5, use:

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia render-review --registry "data\reviews\ceasa_v5\registry.json"
```

Após a revisão humana, copie o JSON baixado para a pasta `sistema_ia` e importe-o em **um arquivo de saída novo**, preservando a proposta anterior:

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia import-review --registry "data\reviews\ceasa_v5\registry.json" --feedback "dronecamp-revisao-ceasa_v5.json" --output "data\reviews\ceasa_v6_humana\registry.json"
.\.venv\Scripts\python.exe -m dronecamp_ia render-review --registry "data\reviews\ceasa_v6_humana\registry.json"
```

Substitua o caminho do feedback pelo local real do arquivo. Se o arquivo de saída já existir, use outro nome de versão. Cada exportação precisa ser importada contra o registro e a taxonomia que a originaram; não use feedback da v1 contra a v5. O importador confere essa ligação, hashes das fotos, classes, coordenadas e confirmação de revisão integral. Fotos positivas/negativas explicitamente confirmadas recebem aprovação visual humana; as fotos não revisadas e as ambiguidades continuam pendentes. Aprovação visual de rótulos não determina gravidade ou conformidade de engenharia. Os comandos de preparação e renderização também não aprovam imagens nem treinam modelos.

Os pontos de implementação ficam em `review_data.py` para consolidação/importação, `review_render.py` e `review_templates/index.html` para a página, `review_dataset.py` para os dados aprovados, `review_provenance.py` para a checagem anterior ao treino e `cli.py` para os comandos. A exportação da página grava as decisões em arquivo; ela não altera diretamente o modelo nem o registro anterior.

## Como preparar dados sem inventar independência

Todas as fotos deste acervo pertencem ao grupo de edificação **`ceasa_pavilhao_a`**. Cenas, recortes e páginas diferentes deste CEASA devem permanecer no mesmo split quando o objetivo for medir generalização entre edificações.

Depois de importar decisões humanas, pode-se construir um rascunho de dados apenas com as imagens aprovadas. O arquivo fornecido `configs/review_groups.json` reserva todo o CEASA para `train`. Escolha uma pasta nova para a versão dos dados:

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia build-reviewed-data --registry "data\reviews\ceasa_v6_humana\registry.json" --assignments "configs\review_groups.json" --output "data\versions\ceasa_v6_humana"
```

O comando seleciona somente imagens com aprovação humana explícita e decisão positiva/negativa, copia fotos e rótulos YOLO e registra origem, hashes, aprovação e grupos. Sem aprovação humana, ele recusa a construção. Versões existentes não são sobrescritas.

Antes de qualquer carregamento de modelo, treino, tuning e avaliação reconciliam o dataset com o registro humano e os arquivos de origem. Mudanças em imagens, caixas, labels, autores, taxonomia ou inventário exigem revisão/versionamento; o indicador salvo de prontidão não basta para liberar os dados.

Com apenas o CEASA em `train`, a versão produzida será um **rascunho**, com `ready_for_training=false` em `readiness.json`. O retorno de erro nesse caso informa falta de prontidão, mesmo que o rascunho tenha sido salvo. Não serão criadas amostras artificiais em `val` ou `test` para satisfazer o protocolo. Imagens ambíguas e excluídas ficam fora dos dados aprovados.

Para o protocolo atual, são necessários treino, validação e teste com grupos de edificações independentes e exemplos positivos das 13 classes no treino. O mínimo estrutural de três grupos não garante diversidade suficiente: devem ser avaliados materiais, condições de captura, tamanhos de ocorrência, positivos e negativos. Essa separação e a qualidade dos rótulos precisam corresponder aos dados reais.

## O que falta para melhorar detecções e relatórios

**Receber imagens, corrigir caixas ou executar detecção não atualiza os pesos do modelo.** O aprendizado acontecerá após preparar uma nova versão aprovada dos dados, executar fine-tuning e comparar o candidato com a versão anterior em avaliação adequada.

O próximo ciclo depende da revisão integral das 38 fotos para as 13 classes, da aprovação humana, da resolução ou exclusão das 15 propostas ambíguas e da coleta de fotos de outras edificações e condições. É necessário obter exemplos positivos das classes novas; criar seus nomes não modifica os pesos. Depois será possível realizar fine-tuning, medir precisão, recall e mAP por classe, revisar falsos positivos/negativos e decidir a adoção do candidato. Mais imagens ajudam quando acrescentam diversidade relevante e rótulos confiáveis; repetir cenas ou incorporar previsões não revisadas pode reforçar erros.

O [relatório de revisão atual](../data/reviews/ceasa_v5/review_report.md) é um rascunho rastreável de fotos e ocorrências propostas. **Não é laudo técnico.** Caixas não fornecem metros, área, vazão, causa, urgência ou gravidade. Recomendações e severidade exigem avaliação técnica separada; a gravidade histórica do laudo CEASA não deve ser transferida automaticamente para novas inspeções.

O histórico preserva as duas leituras de IA das oito classes, as adições de IDs 8 e 9 e o candidato de rufo da v3. A v4 acrescentou ID 10 e ativou ID 11. Nesta atualização, ID 12 foi acrescentado e as 109 propostas foram migradas de v4 para v5, sem aprovação humana ou novas caixas. As evidências e a página de correção acompanham a nova taxonomia. Estão disponíveis mecanismos de versionamento/importação/preparação dos dados. **Revisão integral das 13 classes, aprovação humana, fine-tuning, comparação de modelos especializados e aprovação de uso seguem pendentes.** As verificações desta revisão não mudam o caráter das demonstrações anteriores com pesos genéricos, descritas em [validacao.md](validacao.md).
