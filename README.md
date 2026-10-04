# DroneCamp Vision

Sistema de inteligencia artificial da DroneCamp para analisar imagens de inspecoes com drones, identificar possiveis nao conformidades em telhados e apoiar relatorios tecnicos, com revisao humana e rastreabilidade das evidencias.

## Estado do projeto

Esta publicacao preserva o projeto local e seus artefatos em 04/10/2026. O sistema possui taxonomia de 13 classes, revisoes de imagens, codigo de treinamento e inferencia, pesos de um modelo piloto e exportacoes ONNX.

O treinamento piloto esta documentado em [Treino piloto, sugestoes e APU](sistema_ia/docs/treino_piloto.md). Os artefatos o identificam como `production_ready: false`: o acervo atual nao comprova generalizacao para outras edificacoes. As sugestoes do modelo exigem revisao humana; a publicacao dos arquivos nao representa aprovacao do detector para uso em producao.

## Navegacao

| Pasta ou arquivo | Conteudo |
| --- | --- |
| [sistema_ia/README.md](sistema_ia/README.md) | Instalacao, comandos e fluxo de trabalho |
| [sistema_ia/src/dronecamp_ia](sistema_ia/src/dronecamp_ia) | Codigo de deteccao, revisao, dados, treinamento e exportacao |
| [sistema_ia/configs](sistema_ia/configs) | Taxonomia e configuracoes |
| [sistema_ia/data](sistema_ia/data) | Imagens, proveniencia, anotacoes, revisoes e datasets |
| [sistema_ia/models](sistema_ia/models) e [sistema_ia/runs](sistema_ia/runs) | Pesos, resultados, evidencias e exportacoes |
| [sistema_ia/docs](sistema_ia/docs) | Metodo, historico, validacoes e limites |
| [Documentacao do projeto](Documentação%20Do%20Projeto) | Laudo de referencia e propostas de trabalho |
| [Apresentacao DroneCamp PUCTEC](Documentos%20PUCTEC/Apresentação%20DRONECAMP%20PUCTEC.pdf) | Apresentacao institucional |
| [Fotos de referencia da internet](Fotos_InsperçãoDeTelhadosIndustriais_Internet) | Referencias e suas origens |
| [PUBLICACAO.md](PUBLICACAO.md) | Escopo publicado, exclusoes e verificacao |

## Obter os arquivos completos

Os arquivos `.pt`, `.onnx` e `.engine` usam [Git Large File Storage](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage). Instale Git e Git LFS antes de clonar:

```powershell
git lfs install
git clone https://github.com/CFSJCODE/DroneCamp_Vision.git
cd DroneCamp_Vision
git lfs pull
```

Para instalar o sistema, siga [Ambiente local](sistema_ia/README.md#ambiente-local). O ambiente `.venv` da maquina de origem nao faz parte do repositorio; as dependencias estao em `pyproject.toml` e `requirements-lock.txt`.

## Abrir a revisao de imagens

No Windows, abra [Abrir revisao DroneCamp.cmd](Abrir%20revisao%20DroneCamp.cmd), na raiz do clone. Ele usa caminhos relativos e abre a [revisao v7](sistema_ia/data/reviews/ceasa_v7_revisao002_ba8cc323c8a1/index.html) no navegador local. O GitHub exibe o codigo do HTML; a interface de revisao deve ser aberta a partir do clone.

## Limites de portabilidade

Os registros historicos, manifestos e YAMLs de datasets preservam caminhos absolutos da maquina original. Eles sao mantidos para auditoria. Treinar ou reconstruir datasets em outra maquina exige remapear os caminhos em uma nova versao e validar novamente hashes e proveniencia; a clonagem nao faz essa migracao automaticamente. A pagina de revisao utiliza imagens relativas e o atalho utiliza a localizacao do proprio clone.

As imagens de referencia e os documentos mantem sua origem e autoria. Esta publicacao nao concede uma nova licenca a materiais de terceiros nem define a licenca comercial das bibliotecas usadas.
