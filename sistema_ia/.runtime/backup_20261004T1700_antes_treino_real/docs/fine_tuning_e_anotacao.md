# Fine-tuning e anotação do primeiro detector

## O que significa fine-tuning

Fine-tuning, ou ajuste fino, adapta os pesos de um modelo que já aprendeu padrões visuais a uma tarefa específica. YOLO26l pré-treinado reconhece classes gerais do COCO; ele ainda não tem as categorias de inspeção DroneCamp. Fornecer nomes de classes ou um YAML não ensina o modelo. São necessárias imagens e anotações corretas, e o treinamento altera os pesos para aprender essas ocorrências.

Exemplo: um especialista delimita uma telha quebrada, associa a caixa ao ID 0 e revisa ambiguidades. O treinamento compara a previsão com essa anotação e ajusta os pesos. Ao final, `best.pt` é um novo checkpoint especializado, que precisa passar por avaliação independente. Não se garante aprendizagem útil pelo simples término do treino. [Train oficial](https://docs.ultralytics.com/modes/train)

Hyperparameter tuning é outra operação: executa vários treinamentos para comparar parâmetros como taxa de aprendizado e aumentos. Fine-tuning adapta o conhecimento; tuning seleciona configurações. Não iniciar uma busca cara antes de conferir dados, rótulos e primeiro resultado. [Tuning oficial](https://docs.ultralytics.com/guides/hyperparameter-tuning)

## Categorias e gravidade

O contrato atual v5 possui **13 classes ativas**, IDs 0–12 em `configs/dataset.yaml`. As oito categorias iniciais foram mantidas; foram acrescentadas [Pedaço de telha, ID 8](categoria_pedaco_telha.md), [Rufo deslocado/desalinhado, ID 9](categoria_rufo_deslocado.md), [Pedaço de telha sobreposto à telha, ID 10](categoria_pedaco_telha_sobreposto.md) e [Reparo em rufo, ID 12](categoria_reparo_rufo.md). [Elemento de fixação de telha solto/frouxo, ID 11](categoria_fixador_telha_frouxo.md), foi promovido da fase posterior. O catálogo tem outras dez categorias de fase posterior. Sua inclusão exige dados apropriados, revisão e versionamento do dataset/modelo.

Não existe classe `conforme`: uma imagem sem os objetos de interesse é um negativo explícito. Uma imagem ambígua, desfocada ou com elemento escondido não deve ser registrada como negativo confiável; marque a pendência na fila de revisão e retire-a do conjunto enquanto estiver indefinida.

A gravidade do laudo é contexto histórico do CEASA. Confiança do detector não mede gravidade, causa, urgência, estanqueidade ou estabilidade estrutural. `severity` permanece `null`, `review_status` permanece `pendente` e a referência histórica tem nome próprio. O estágio CORAL previsto nos planos depende de níveis ordinais consistentes e exemplos suficientes; não foi implementado agora.

## Fotos do PDF

As fotos incorporadas permitem testar o fluxo e iniciar uma oficina de anotação. Elas podem conter compressão, recortes, setas e elipses; uma mesma cena aparece em páginas diferentes. Extração de bytes não recupera resolução perdida nem garante remoção de marcações. O manifesto identifica duplicatas exatas, mas fotos quase iguais e recortes precisam de revisão visual.

Não usar toda figura de uma página como caixa positiva. Uma foto de resíduos em calha pode também conter telha quebrada; a anotação deve localizar cada objeto e admitir mais de uma classe na foto. A palavra do laudo identifica uma evidência documental, não uma caixa de treino.

O acervo do CEASA representa uma única edificação/caso. Dividir suas figuras aleatoriamente entre treino e teste produz uma avaliação otimista. Mesmo uma prova de conceito com esses recortes não sustenta generalização para novos telhados. Buscar novos locais/campanhas e originais limpos antes de alegar qualidade operacional.

## Regras de anotação propostas para revisão

1. Use ferramenta de anotação local ou um ambiente aprovado pela equipe, exportando **YOLO detection**. Este projeto não envia imagens automaticamente.
2. Anote somente a ocorrência que você consegue localizar visualmente. Registre dúvidas e adjudique divergências com o responsável técnico.
3. Delimite a menor caixa que contém a evidência. Em quebra, inclua contexto suficiente para distinguir junta normal; em ausência, delimite a lacuna esperada. Essas convenções precisam de aprovação antes de criar o acervo inteiro.
4. Em resíduos, uma caixa cobre um acúmulo contíguo. Fixe essa convenção para evitar que cada anotador conte de modo diferente. Anote material na calha como `residuos_calha`, e sobre a telha como `residuos_telha`.
5. Reparos são evidência de intervenção. Não inferir que o material está mal executado só pela sua presença. Subtipos manta/fita/PU/remendo podem ser metadados futuros.
6. Para detectar falta de rufo, conhecer onde o elemento deveria existir; uma borda exposta sem contexto pode ser inconclusiva.
7. Registre um `.txt` vazio apenas quando a imagem tiver sido revisada e não contiver classes ativas. Ausência de `.txt` é falta de anotação, não negativo.
8. Faça dupla revisão de uma amostra e registre o critério adotado nos conflitos. Não há número mínimo universal que garanta qualidade; medir cobertura, diversidade e curvas de aprendizagem.
9. Reconheça um pedaço de telha por evidência do fragmento solto, sem converter todo detrito em telha. Evite duplicar uma caixa idêntica entre fragmento e resíduos.
10. Diferencie deslocamento de rufo, ruptura visível e junta prevista. Confirme a identidade do elemento; dúvida permanece fora do treino.
11. Prefira ID 10 quando o fragmento solto de telha e seu apoio sobre uma telha instalada forem reconhecíveis. Use ID 8 quando a peça for identificável e esse contexto não estiver confirmado; nunca duplique o mesmo fragmento com IDs 8 e 10. Sobreposição normal e reparo aderido não são ID 10.
12. Use ID 11 somente quando o fixador da telha e a soltura, o deslocamento ou a folga aparente forem distinguíveis. A presença de porca, arruela ou sombra não mede torque. Um fixador avulso sem vínculo reconhecível com a telha permanece ambíguo.
13. Use ID 12 quando o próprio rufo e uma intervenção nele forem identificáveis: fita, manta, remendo ou aplicação de selante reconhecível como reparo. Delimite a intervenção; reparo em telha continua em ID 3. Selante regular de montagem não comprova reparo. Reparo não prova falha, ruptura escondida, aderência ou estanqueidade. Não duplique caixa idêntica de reparo e dano sem evidências próprias; dúvidas sobre o elemento ou a intervenção permanecem ambíguas.

Formato de cada linha: `class_id x_center y_center width height`, coordenadas relativas à largura/altura, no intervalo `[0,1]`. Exemplo apenas matemático: `0 0.50 0.50 0.20 0.10` representa uma caixa central com largura de 20% e altura de 10%; **não é anotação de nenhuma foto real**.

## Separar dados sem vazamento

Defina `group_id` pela unidade que deve permanecer junta: mesma edificação, captura/campanha e seus frames/recortes. Para medir generalização entre edificações, todas as campanhas da mesma edificação devem compartilhar o grupo. Não usar número do frame como grupo.

O arquivo `groups.csv` fica em `data/dataset`:
Na preparação versionada, o builder escreve esse arquivo dentro da nova versão, junto de `provenance.json`; consulte [o fluxo de revisão atual](execucao_revisao_ceasa.md). Treino e avaliação exigem revisão humana, integridade reconciliada e positivos de todas as 13 classes no treino. Criar somente a estrutura abaixo não libera o treinamento.

```csv
image,group_id,split
images/train/foto_local_a.jpg,edificacao_a,train
images/val/foto_local_b.jpg,edificacao_b,val
images/test/foto_local_c.jpg,edificacao_c,test
```

Os nomes acima são exemplos fictícios de estrutura. Não coloque o mesmo grupo em splits diferentes. A validação local exige imagens nos três splits, labels, IDs coerentes, caixas válidas, manifesto completo e nenhuma imagem com bytes duplicados. Ela não detecta todos os recortes/frames semelhantes nem prova a veracidade do grupo declarado.

## Avaliar e corrigir

- Acompanhar precisão, recall, mAP50 e mAP50-95 por classe, além dos valores globais.
- Revisar falsos positivos e falsos negativos em sombras, sujeira, emendas, material envelhecido, aberturas previstas e objetos pequenos.
- Comparar resolução 640/1024 e recortes somente com metodologia consistente. Aumentar resolução tem custo; não recupera detalhe que não existe na fonte.
- Não transformar pixels em metros/m² sem calibração, escala e geometria apropriadas.
- Medir tempo por imagem/frame e consumo de memória no destino real.
- Selecionar modelo e limiares com validação; consultar o teste final após fechar a decisão.
- Registrar correções humanas como nova versão do dataset. Não fazer aprendizagem automática a partir de previsões não revisadas.

Metas de aceite e pesos para penalizar omissões críticas ainda precisam ser definidos com a equipe técnica. Esta entrega não inventa percentuais mínimos ou prova de qualidade.
