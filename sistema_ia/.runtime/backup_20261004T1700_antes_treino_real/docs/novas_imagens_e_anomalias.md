# Novas imagens, defeitos conhecidos e anomalias inéditas

Data: 03/10/2026. Estado: protocolo de evolução do DroneCamp; ainda não existe detector especializado aprovado, entrada automatizada geral de novas campanhas ou módulo de descoberta de anomalias em produção.

## Novas fotos de classes conhecidas

Depois de treinado e avaliado, o detector procura as categorias do seu contrato e retorna caixas, classes e confiança. Uma nova foto serve primeiro para inferência; receber a foto não modifica pesos. O modelo pode deixar de detectar ocorrências e confundir sombras, juntas, reparos e detritos. Falta de uma caixa não comprova ausência de defeitos. [Detecção Ultralytics](https://docs.ultralytics.com/tasks/detect).

Para transformar uma nova foto em aprendizado, registrar origem e edificação, conferir qualidade e anotações, marcar a decisão humana e criar uma versão dos dados. Fotos ou frames relacionados permanecem no mesmo grupo de divisão. O usuário pode fornecer novas imagens para análise assistida nesta conversa; propostas visuais feitas pelo assistente também precisam de confirmação antes de entrar no treinamento.

Priorizar detalhes nítidos e contexto do elemento: fotografia geral mais aproximação, quando disponíveis; ângulos e iluminação diferentes; coberturas, materiais e edificações diversos; exemplos de defeitos e situações normais. Registrar só características conhecidas. Recortes não recuperam detalhes que não foram capturados e não se tornam novas edificações independentes.

## Defeitos que não pertencem às categorias atuais

O detector fechado, treinado nas 13 classes atuais, não inventa uma nova classe ao ver algo diferente. Pode ignorar uma anomalia ou classificá-la incorretamente como uma classe conhecida. Confiança baixa ou nenhuma detecção não identificam automaticamente uma anomalia inédita.

O procedimento é preservar a foto, registrar a dúvida, identificar visualmente o elemento e a condição com revisão adequada, procurar uma categoria equivalente e criar uma nova somente quando o significado for distinto. Definir critérios, incluir casos de confusão e negativos pertinentes, anotar exemplos diversificados e executar novo treino e avaliação. A fotografia não resolve propriedades invisíveis como torque, aderência ou estanqueidade.

Para “Reparo em rufo”, por exemplo, uma aplicação identificável no próprio rufo pode receber a classe 12. Reparo sobre telha continua classe 3. Selante original regular não prova reparo; se o elemento ou a intervenção forem incertos, o registro permanece ambíguo. A classe descreve o achado visual; conformidade, severidade, urgência e recomendação exigem avaliação própria.

## Apoio futuro à descoberta

Há modelos como YOLOE que recebem categorias por texto ou exemplo visual, ou usam um vocabulário amplo pré-definido. Eles podem apoiar propostas de anotação para alvos que não constam do detector fechado. O modo sem prompt continua usando um vocabulário próprio; isso não equivale a diagnosticar qualquer defeito novo. [Documentação oficial do YOLOE](https://docs.ultralytics.com/models/yoloe).

Como hipótese de evolução, uma camada separada poderia propor regiões/objetos suspeitos e encaminhá-los à revisão. Esta camada está planejada, não implementada ou validada nas inspeções do DroneCamp. Suas propostas não deverão aprovar rótulos, criar não conformidades técnicas, retrainar ou substituir um modelo automaticamente.

## Como verificar se melhorou

Comparar candidatos em fotos independentes do treinamento, com a mesma referência de revisão e métricas por classe. Melhorar uma média global pode esconder piora em uma classe rara. Retenção de exemplos anteriores, novas imagens representativas e análise das falsas detecções/omissões orientam o próximo ciclo. Adotar o candidato somente com critérios previamente definidos e conservar a versão anterior para recuperação.

O [plano de coleta](plano_coleta_imagens.md) organiza metas iniciais por classe, exemplos de situações normais e diversidade entre campanhas. Essas metas são propostas de planejamento; a aprovação de uso depende das medições, não apenas da contagem de fotos.
