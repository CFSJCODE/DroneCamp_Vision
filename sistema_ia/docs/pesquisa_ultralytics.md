# Pesquisa oficial Ultralytics para o Drone Camp

Data da consulta: **03/10/2026**. Escopo: documentação oficial solicitada, repositório público Ultralytics e metadados do pacote publicado. Esta pesquisa verifica suporte e APIs; não comprova desempenho nas imagens deste projeto.

## 1. YOLO27l: prioridade futura com bloqueio público confirmado

**YOLO27l ainda não pode ser executado com o pacote público.** A página oficial informa que pesos, configurações e código de implementação não foram liberados; seus exemplos são para depois do lançamento. A recomendação atual da própria Ultralytics é usar YOLO26. A família planejada tem tamanhos N/S/M/L; a detecção L prevê backbone UltraViT e decodificador por consultas, sem NMS. Resultados preliminares: 72,3 milhões de parâmetros, 60,4 mAP COCO em 640 pixels; 61,2 em 800. Os tempos divulgados usam RTX PRO 6000/TensorRT 11/FP16 e não estimam o desempenho deste computador. [YOLO27 oficial](https://docs.ultralytics.com/models/yolo27)

Verificação independente da árvore completa de `main`, commit `4a536fd41d5cde788a136324546a800cac111916`: o único caminho contendo `yolo27` era `docs/en/models/yolo27.md`; nenhum arquivo de arquitetura YOLO27 foi encontrado. Isso confirma a ausência de configurações nominadas no estado público consultado, sem provar a inexistência de pesquisa interna. [Árvore oficial consultada](https://api.github.com/repos/ultralytics/ultralytics/git/trees/4a536fd41d5cde788a136324546a800cac111916?recursive=1)

**Decisão de implementação:** manter YOLO27l identificado como objetivo, rejeitar sua execução enquanto não houver suporte verificado e usar YOLO26l como base inicial explicitamente registrada. Quando YOLO27l sair, repetir a avaliação no mesmo conjunto de teste antes de trocar o modelo.

## 2. YOLO26 e a imagem anexada

O Markdown solicitado foi acessado diretamente; a ferramenta de navegação não aceitou seu tipo `text/markdown`, então a leitura foi complementada por acesso HTTP e pela página HTML equivalente. A documentação atual confirma os sete itens do print: detecção, instâncias, segmentação semântica, profundidade monocular, classificação, pose e OBB. São **pesos especializados**, com sufixos `-seg`, `-sem`, `-depth`, `-cls`, `-pose`, `-obb`, e tamanhos n/s/m/l/x; um arquivo de detecção não realiza todas as tarefas simultaneamente.

YOLO26 remove DFL da arquitetura de regressão e combina Progressive Loss, STAL e MuSGD. Na detecção, `nms=False` seleciona a saída um-para-um; o caminho padrão usa NMS. YOLO26l apresenta 55,0 mAP COCO no caminho padrão e 54,4 no caminho sem NMS; esses resultados não medem não conformidades de edificações. Variantes P2/P6 são arquiteturas YAML, sem pesos oficiais específicos. [Markdown YOLO26](https://docs.ultralytics.com/models/yolo26.md), [exemplos YOLO26](https://docs.ultralytics.com/models/yolo26#usage-examples)

Exemplo adaptado para uma imagem local:

```python
from ultralytics import YOLO

detector = YOLO("yolo26l.pt")
predicoes = detector.predict(source="imagem_inspecao.jpg", device="cpu")
```

## 3. Receita de treinamento YOLO26

Os pesos oficiais passaram por pré-treino Objects365v1 e ajuste COCO. A receita é uma referência para entender os checkpoints, não uma configuração universal para inspeção. Para YOLO26l, o estágio COCO documenta MuSGD, `lr0=0.00038`, `lrf=0.882`, `momentum=0.948`, `weight_decay=0.00027`, 60 épocas, batch 128 e imagem 640. Copiar esse batch sem medir memória pode inviabilizar a execução.

É possível examinar `checkpoint.ckpt["train_args"]` e a revisão em `checkpoint.ckpt["git"]`. Os argumentos experimentais `muon_w`, `sgd_w`, `o2m` e `cls_w` não são opções públicas comuns e podem ser rejeitados. Apesar do nome, o ganho `dfl` do YOLO26 passa a ponderar regressão L1 de distâncias; removê-lo por causa de “DFL-free” altera a receita. [Receita oficial](https://docs.ultralytics.com/guides/yolo26-training-recipe)

## 4. YOLOE para explorar classes novas

YOLOE realiza detecção e segmentação de instâncias com nomes textuais, exemplos visuais ou vocabulário interno. Para prompts, usar `*-seg.pt`; a variante `*-seg-pf.pt` não aceita `set_classes()`. Prompts visuais retornam nomes genéricos como `object0`, exigindo mapeamento próprio. A primeira preparação textual pode instalar CLIP e baixar encoder, por isso exige rede e preparação antecipada para operação offline. YOLOE-26 exige Ultralytics 8.4.0 ou posterior.

Uso possível: ajudar na triagem e na criação inicial de anotações, sempre revisadas. A documentação de vocabulário aberto não demonstra reconhecimento confiável de fissura, infiltração ou corrosão deste projeto; uma frase fornecida ao modelo não equivale a treinamento especializado. [YOLOE oficial](https://docs.ultralytics.com/models/yoloe)

Exemplo conceitual adaptado, ainda sujeito a validação no projeto:

```python
from ultralytics import YOLOE

explorador = YOLOE("yoloe-26s-seg.pt")
explorador.set_classes(["wall", "roof", "window"])
explorador.predict(source="imagem_inspecao.jpg", device="cpu")
```

## 5. Predict: imagens, vídeos e resultados revisáveis

`predict()` aceita arquivos, pastas, vídeos e streams. Para muitos arquivos ou vídeo, `stream=True` retorna um gerador e evita acumular todos os resultados na memória. Cada resultado de detecção fornece caixas, IDs de classe e confiança; a confiança não é garantia de veracidade. `classes` filtra IDs existentes e não cria classes.

`conf` regula o filtro, `imgsz` a resolução e `vid_stride` o salto de frames. `rect=False` mantém tamanho de entrada fixo, útil ao comparar exportações. Em vídeo ao vivo, `stream_buffer=True` acumula atraso quando a inferência é mais lenta que a fonte; o padrão descarta frames antigos. NumPy/OpenCV usa imagem BGR; tensores seguem RGB. A precisão atual usa `quantize`, substituindo `half` legado. [Predict oficial](https://docs.ultralytics.com/modes/predict)

Exemplo adaptado: `detector.predict(source="voo.mp4", stream=True, device="cpu", imgsz=640, conf=0.25)`. O limiar precisa ser escolhido com a validação do projeto.

## 6. Train: fine-tuning e retomada

Carregar `.pt` pré-treinado e treinar com `data` próprio adapta o modelo; carregar `.yaml` inicia uma arquitetura e pode exigir transferência explícita. Detecção usa YAML de dataset; classificação usa diretório organizado por classe. Em Windows, o ponto de entrada deve ficar protegido por `if __name__ == "__main__":` para evitar falhas de multiprocessamento.

`epochs`, `batch`, `imgsz`, `patience`, `seed`, `device` e `workers` precisam estar visíveis na configuração. `optimizer="auto"` escolhe MuSGD em treinos longos e AdamW nos curtos, ignorando `lr0` manual; escolher o otimizador explicitamente torna esse controle revisável. Retomar `last.pt` com `resume=True` restaura o estado do treino; iniciar novo ajuste com `best.pt` é uma operação diferente. [Train oficial](https://docs.ultralytics.com/modes/train)

Exemplo adaptado: `detector.train(data="dataset.yaml", epochs=50, imgsz=640, batch=2, device="cpu", optimizer="AdamW", lr0=0.001)`. Esses valores ilustram a API e não foram otimizados para o Drone Camp.

## 7. Ajuste de hiperparâmetros

Fine-tuning ajusta pesos aprendidos; **hyperparameter tuning** procura configurações de treinamento. `model.tune()` faz várias tentativas, mutando taxa de aprendizado, perdas e aumentos. O tuner padrão usa fitness da tarefa; para detecção, avaliar também precisão, recall e desempenho por classe. O espaço padrão não procura arquitetura, batch ou número de épocas.

`iterations=20, epochs=30` representa vinte treinamentos de trinta épocas, não um treino único. Definir orçamento antes de executar. A documentação alerta que buscas muito curtas ou em dados pequenos podem escolher parâmetros que não transferem ao treino completo. Guardar `best_hyperparameters.yaml` e os resultados NDJSON. [Guia oficial de tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning)

Exemplo adaptado para uma etapa posterior: `detector.tune(data="dataset.yaml", iterations=10, epochs=20, optimizer="AdamW", space={"lr0": (0.0001, 0.001)})`.

## 8. Export: validar o artefato no destino

Exportar `best.pt` para ONNX permite execução com outros runtimes; TensorRT depende do destino NVIDIA. Selecionar `imgsz`, `batch`, formato e precisão explicitamente. `quantize=16` indica FP16; `quantize=8` exige calibração representativa para pós-treino. `half` e `int8` antigos seguem aceitos com avisos de depreciação.

No YOLO26, exportar com `nms=None` deixa saídas brutas do caminho um-para-muitos, `nms=True` incorpora NMS quando suportado e `nms=False` seleciona o caminho sem NMS. Isso modifica formato e interpretação da saída. Shape dinâmico amplia flexibilidade, mas a compatibilidade depende do runtime. Comparar o resultado exportado ao PyTorch nas mesmas imagens e medir o hardware real. [Export oficial](https://docs.ultralytics.com/modes/export)

Exemplo adaptado: `detector.export(format="onnx", imgsz=640, batch=1, nms=False, quantize=32, device="cpu")`. Exportação concluída não comprova qualidade nem velocidade em produção.

## 9. Boas práticas de treinamento

Pré-treino oferece um ponto de partida para adaptar uma tarefa relacionada. Batch deve caber na memória; em CPU, AutoBatch pode recair em 16, portanto um batch explícito facilita testes pequenos. Cache em RAM consome memória; cache em disco consome armazenamento. AMP é ignorado na CPU e Apple silicon; na GPU CUDA há verificação inicial.

`scale` modifica o zoom da imagem, enquanto `multi_scale` muda a resolução de entrada entre batches. Aumentos devem representar a captura real; aumentar dados artificialmente não substitui diversidade real nem corrige rótulos incorretos. Um subconjunto representativo ajuda a validar o fluxo antes do treinamento completo. [Dicas oficiais](https://docs.ultralytics.com/guides/model-training-tips)

## 10. Segmentação de instâncias

Instâncias fornecem máscara própria para cada objeto, além de caixa, classe e confiança. Pesos usam `-seg.pt`, com anotações de contorno. `result.masks.xy`, `.xyn` e `.data` oferecem polígonos e máscaras; estes podem estar ausentes quando não há detecção. Avaliar métricas de caixa e máscara separadamente.

Para delimitar corrosão, área descascada ou manchas, segmentação pode ser uma próxima etapa após obter anotações adequadas. Pixel medido não é automaticamente área física: medições em m² exigem escala, geometria e calibração. Segmentação semântica atribui uma classe por pixel e não separa ocorrências individuais. [Segment oficial](https://docs.ultralytics.com/tasks/segment)

Exemplo adaptado: `YOLO("yolo26l-seg.pt").train(data="dataset_segmentacao.yaml", epochs=50, imgsz=640)`. Executar somente com polígonos revisados.

## 11. Classificação de imagens

Classificação atribui uma classe à imagem inteira, sem localizar a ocorrência. Os pesos `-cls.pt` partem de ImageNet. Dataset usa pastas por classe; saída apresenta probabilidades/top-1/top-5. Para fotos com várias patologias, classificar apenas a foto pode esconder a coexistência de defeitos. Uma alternativa a avaliar é classificar recortes localizados pelo detector.

O pré-processamento padrão usa recortes na aprendizagem e na inferência; em imagens muito alongadas pode remover uma região essencial. A documentação permite customizar transformações para preservar o conteúdo. Avaliar a mesma transformação no treino, validação e destino. [Classify oficial](https://docs.ultralytics.com/tasks/classify)

Exemplo adaptado: `YOLO("yolo26l-cls.pt").train(data="dataset_classificacao", epochs=50, imgsz=224)`. Esse classificador é uma etapa distinta do detector inicial.

## 12. Encaminhamento técnico para este projeto

As orientações abaixo são decisões de engenharia propostas; devem ser confrontadas com o relatório e as imagens do projeto:

1. Definir categorias observáveis e um manual de anotação. Severidade, urgência e causa devem ficar em campos independentes, preenchidos ou confirmados por responsável técnico.
2. Usar imagens do drone e do contexto de inspeção, com positivos, negativos e situações ambíguas. Anotar a evidência visível; não rotular uma causa oculta apenas pela aparência.
3. Separar treino, validação e teste por edificação/voo/captura. Frames consecutivos e recortes da mesma foto devem ficar no mesmo split para evitar vazamento.
4. Validar estrutura e rótulos antes de treinar. Versionar dataset, classes, parâmetros, modelo e relatório de métricas.
5. Executar YOLO26l com pesos genéricos para verificar leitura, inferência e artefatos. Este teste confirma infraestrutura e não reconhecimento especializado de não conformidades.
6. Fazer fine-tuning com o dataset anotado. Medir falsos positivos, falsos negativos, recall por classe e desempenho em objetos pequenos; preservar um teste final que não participou da escolha de parâmetros.
7. Entregar evidência localizável com status de revisão humana. Ausência de detecção significa “nenhuma ocorrência identificada pelo modelo”, sem certificar conformidade.
8. Avaliar YOLO27l quando seus pesos e suporte público estiverem disponíveis. Usar o mesmo teste e comparar qualidade, memória e tempo antes da migração.

## 13. Versão e limites desta pesquisa

O PyPI oficial respondeu **Ultralytics 8.4.172**, publicado em 03/10/2026, com Python `>=3.8`; o metadata exclui Torch 2.4.0 em Windows. Uma instalação isolada e uma versão fixada reduzem variações entre máquinas. [Metadados oficiais PyPI](https://pypi.org/pypi/ultralytics/8.4.172/json)

Todos os links solicitados foram consultados. YOLO27l foi verificado como indisponível; YOLO26-sem/depth estão documentados e seus YAMLs existem no repositório consultado. Esta pesquisa não executou modelos, treinamento, tuning ou exportação. O suporte real da instalação local e o desempenho do sistema são verificados separadamente no registro de validação do projeto.
