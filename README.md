# DroneCamp Vision

**Visão computacional para apoiar inspeções de telhados industriais com imagens de drones.**

O DroneCamp Vision reúne análise de imagens, sugestões de possíveis não conformidades, revisão humana e organização de evidências para apoiar relatórios técnicos. O projeto conecta cada anotação à imagem de origem, registra as decisões de revisão e preserva a procedência dos dados utilizados no treinamento.

[![Python 3.12–3.13](https://img.shields.io/badge/Python-3.12%E2%80%933.13-3776AB?style=flat-square&logo=python&logoColor=white)](sistema_ia/pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics 8.4.172](https://img.shields.io/badge/Ultralytics-8.4.172-111F68?style=flat-square)](sistema_ia/pyproject.toml)
[![YOLO26](https://img.shields.io/badge/YOLO26-detec%C3%A7%C3%A3o-2563EB?style=flat-square)](sistema_ia/configs/project.yaml)
[![ONNX](https://img.shields.io/badge/ONNX-exporta%C3%A7%C3%A3o-005CED?style=flat-square&logo=onnx&logoColor=white)](https://onnx.ai/)
[![OpenCV](https://img.shields.io/badge/OpenCV-vis%C3%A3o_computacional-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)

[![HTML5](https://img.shields.io/badge/HTML5-revis%C3%A3o_local-E34F26?style=flat-square&logo=html5&logoColor=white)](sistema_ia/src/dronecamp_ia/review_templates/index.html)
[![CSS](https://img.shields.io/badge/CSS-interface-663399?style=flat-square&logo=css&logoColor=white)](sistema_ia/src/dronecamp_ia/review_templates/index.html)
[![JavaScript](https://img.shields.io/badge/JavaScript-revis%C3%A3o-F7DF1E?style=flat-square&logo=javascript&logoColor=black)](sistema_ia/src/dronecamp_ia/review_templates/index.html)
[![Git LFS](https://img.shields.io/badge/Git_LFS-modelos-F05032?style=flat-square&logo=gitlfs&logoColor=white)](.gitattributes)
[![Conventional Commits](https://img.shields.io/badge/Conventional_Commits-1.0.0-FE5196?style=flat-square&logo=conventionalcommits&logoColor=white)](CONTRIBUTING.md#mensagens-de-commit)

> **Estado atual: piloto experimental.** O modelo gera sugestões para revisão e permanece com `production_ready: false`. Os resultados disponíveis ainda não comprovam generalização para outras edificações. Consulte o [relatório de treinamento e seus limites](sistema_ia/docs/treino_piloto.md).

## Comece por aqui

| Quero… | Onde encontrar |
| --- | --- |
| Entender o funcionamento e os comandos | [Guia do sistema](sistema_ia/README.md) |
| Conferir e corrigir as imagens | [Revisão v7](sistema_ia/data/reviews/ceasa_v7_revisao002_ba8cc323c8a1/index.html), aberta localmente pelo [atalho Windows](Abrir%20revisao%20DroneCamp.cmd) |
| Conhecer as categorias de inspeção | [Taxonomia](sistema_ia/configs/taxonomy.json) e [análise documental](sistema_ia/docs/analise_documental.md) |
| Consultar o treinamento e as métricas | [Treino piloto](sistema_ia/docs/treino_piloto.md) e [resultados das execuções](sistema_ia/runs) |
| Localizar o código | [Módulos Python](sistema_ia/src/dronecamp_ia) |
| Entender a coleta e a evolução dos dados | [Plano de coleta](sistema_ia/docs/plano_coleta_imagens.md) e [melhoria contínua](sistema_ia/docs/plano_revisao_e_melhoria_continua.md) |
| Contribuir com o projeto | [Guia de contribuição e Conventional Commits](CONTRIBUTING.md) |
| Conferir o que foi publicado | [Escopo da publicação](PUBLICACAO.md) e [inventário com hashes](inventario-publicacao.json) |

## O que o sistema oferece

- **Inspeção assistida:** caixas candidatas, classes e níveis de confiança para orientar a conferência de possíveis problemas nas imagens.
- **Revisão humana:** interface local para aceitar, corrigir ou descartar sugestões e exportar as decisões do revisor.
- **Rastreabilidade:** registros com hashes, origem das imagens, categorias, anotações e histórico das decisões.
- **Dados versionados:** preparação de conjuntos de imagens e rótulos a partir das revisões, com verificações de integridade e separação por grupos.
- **Treinamento e avaliação:** comandos para treinamento piloto, treinamento condicionado à prontidão dos dados e avaliação do detector.
- **Evidências e exportação:** imagens demarcadas, resultados estruturados em JSON/JSONL e exportação dos pesos para ONNX.

## Tecnologias utilizadas

| Tecnologia | Aplicação no projeto | Referência |
| --- | --- | --- |
| **Python 3.12–3.13** | Código do sistema, CLI, processamento e testes | [pyproject.toml](sistema_ia/pyproject.toml) |
| **Ultralytics 8.4.172 / YOLO26l** | Detecção, treinamento e geração de sugestões | [Configuração do projeto](sistema_ia/configs/project.yaml) |
| **PyTorch** | Execução e treinamento do modelo; ambiente de referência com CPU | [Dependências registradas](sistema_ia/requirements-lock.txt) |
| **OpenCV, Pillow e NumPy** | Leitura e processamento das imagens, inclusive no fluxo da Ultralytics | [Dependências registradas](sistema_ia/requirements-lock.txt) |
| **ONNX / ONNX Runtime** | Exportação e execução do modelo, com conferência de paridade | [Exportador](sistema_ia/src/dronecamp_ia/exporting.py) |
| **DirectML — opcional** | Execução direta do ONNX em GPU/APU compatível no Windows | [Medições e limites](sistema_ia/docs/treino_piloto.md#cpu-apu-e-memória) |
| **HTML, CSS e JavaScript** | Página de revisão das fotos, caixas e decisões | [Template da interface](sistema_ia/src/dronecamp_ia/review_templates/index.html) |
| **PyYAML e pypdf** | Configurações em YAML e extração de imagens de relatórios PDF | [Dependências e extras](sistema_ia/pyproject.toml) |
| **Git / Git LFS** | Histórico do projeto e armazenamento dos pesos e exportações | [.gitattributes](.gitattributes) |

As versões do [arquivo de dependências](sistema_ia/requirements-lock.txt) descrevem o ambiente de origem. Os extras disponíveis estão em `pyproject.toml`. A aceleração DirectML é opcional: os comandos atuais `predict`, `suggest` e a conferência de paridade seguem usando CPU.

## Estrutura do repositório

```text
DroneCamp_Vision/
├── README.md                          # Apresentação e caminhos principais
├── CONTRIBUTING.md                    # Contribuições e mensagens de commit
├── PUBLICACAO.md                      # Escopo, exclusões e integridade
├── inventario-publicacao.json          # Arquivos, tamanhos e hashes SHA-256
├── Abrir revisao DroneCamp.cmd         # Abre a revisão v8 no Windows (offline)
├── Abrir plataforma DroneCamp.cmd      # Plataforma com servidor local (treino e monitoramento ao vivo)
├── Documentação Do Projeto/            # Laudo técnico e propostas
├── Documentos PUCTEC/                  # Apresentação institucional publicada
├── Fotos_InsperçãoDeTelhadosIndustriais_Internet/
├── .githooks/                         # Validação de commits e hooks do Git LFS
└── sistema_ia/
    ├── README.md                      # Guia operacional detalhado
    ├── pyproject.toml                 # Dependências, extras e CLI
    ├── requirements-lock.txt          # Versões do ambiente de referência
    ├── configs/                       # Taxonomia e parâmetros do projeto
    ├── data/
    │   ├── reference/                 # Imagens extraídas e procedência
    │   ├── reviews/                   # Páginas e registros de revisão
    │   ├── annotation_corpus/         # Anotações humanas históricas
    │   ├── pilot/                     # Datasets do piloto
    │   └── dataset/ e datasets/       # Estruturas de dados e integração
    ├── docs/                          # Método, decisões e validações
    ├── models/                        # Pesos de referência via Git LFS
    ├── runs/                          # Resultados, evidências e exportações
    ├── scripts/                       # Utilitários de integração
    ├── src/dronecamp_ia/              # Implementação modular
    └── tests/                         # Testes automatizados
```

Os nomes de pastas existentes foram preservados. Todos os links acima e abaixo partem da raiz do repositório, sem depender da localização da pasta original na máquina do autor.

## Clonar e instalar

O exemplo abaixo usa **Windows, Git, Git LFS e Python 3.12**. O projeto declara suporte a Python `>=3.12,<3.14`.

### 1. Obter o projeto e os modelos

```powershell
git lfs install
git clone -c core.longpaths=true https://github.com/CFSJCODE/DroneCamp_Vision.git
cd DroneCamp_Vision
git lfs pull
```

O Git LFS recebe os arquivos `.pt`, `.onnx` e `.engine` completos. A opção `core.longpaths=true` fica restrita ao clone e permite receber as pastas de anotação com nomes baseados em hashes no Windows.

### 2. Criar um ambiente Python isolado

Execute a partir da raiz do clone:

```powershell
cd sistema_ia
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[documents,export]"
```

Esse exemplo prepara execução em CPU e os extras de PDF e ONNX. Para outras opções de ambiente, consulte o [guia operacional](sistema_ia/README.md#ambiente-local) e os [extras disponíveis](sistema_ia/pyproject.toml). O ambiente `.venv` é criado localmente e não é versionado.

### 3. Conferir os comandos e as categorias

A partir da pasta `sistema_ia`:

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia --help
.\.venv\Scripts\python.exe -m dronecamp_ia categories
```

Os exemplos completos de predição, revisão, treinamento e exportação estão no [README do sistema](sistema_ia/README.md). Os registros históricos precisam dos ajustes de caminhos descritos em [Portabilidade](#portabilidade-e-limites).

## Abrir a revisão das imagens

Na raiz do clone, dê dois cliques em **[Abrir revisao DroneCamp.cmd](Abrir%20revisao%20DroneCamp.cmd)**. O atalho encontra a pasta do próprio projeto e abre a [revisão v8](sistema_ia/data/reviews/ceasa_v8_revisao002_ff34e226416c/index.html) no navegador local, sem servidor.

Para acompanhar treinos ao vivo e iniciar `train-pilot` ou `suggest` pela própria página, use **[Abrir plataforma DroneCamp.cmd](Abrir%20plataforma%20DroneCamp.cmd)**, que sobe um servidor local em `127.0.0.1:8765` com o `.venv` do `sistema_ia`. As seções da plataforma estão descritas em [Plataforma de operações YOLO](sistema_ia/docs/plataforma_operacoes.md).

Também é possível abrir diretamente:

```text
sistema_ia/data/reviews/ceasa_v7_revisao002_ba8cc323c8a1/index.html
```

No GitHub, o link para um arquivo HTML mostra seu código. A interface de revisão deve ser aberta a partir do clone, junto das imagens e dos arquivos de dados. O arquivo em `src/dronecamp_ia/review_templates/` é o template de geração; a página com fotos e dados está em `data/reviews/`.

## Estado documentado do piloto

| Item | Estado em 04/10/2026 |
| --- | --- |
| Taxonomia | 13 classes ativas |
| Revisão atual | v7, com 38 fotos e 117 caixas registradas |
| Decisões das fotos | 34 aprovadas e 4 ambíguas |
| Acervo usado no piloto | Uma edificação: CEASA |
| Modelo efetivo | YOLO26l, com treinamento piloto de 80 épocas |
| Prontidão operacional | `production_ready: false` |

Os resultados, as classes sem cobertura suficiente e as métricas estão em [Treino piloto, sugestões do modelo e APU](sistema_ia/docs/treino_piloto.md). A evolução depende de novas imagens, revisão humana e avaliação em edificações que não participaram do treinamento.

## Contribuir e registrar mudanças

As novas contribuições adotam [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/), com mensagens como:

```text
docs(readme): melhorar apresentação e navegação do projeto
feat(review): acrescentar uma opção de revisão
fix(dataset): corrigir validação de uma anotação
```

Consulte [CONTRIBUTING.md](CONTRIBUTING.md) para tipos, escopos, mudanças incompatíveis e ativação da validação local. Os hooks versionados preservam os eventos utilizados pelo Git LFS. Um novo clone precisa ativá-los explicitamente, seguindo o guia.

## Portabilidade e limites

- **Caminhos históricos:** registros, manifestos e YAMLs de datasets preservam caminhos absolutos da máquina original. Reconstruir datasets em outra pasta exige remapear os caminhos em uma nova versão e validar novamente os hashes e a procedência. A clonagem não faz essa migração automaticamente.
- **Interpretação das detecções:** caixas e classes geradas pelo piloto são sugestões. Elas exigem revisão humana e não certificam a conformidade de uma estrutura.
- **Acervo e licenças:** os documentos e imagens mantêm suas referências e autoria. A publicação não concede nova licença a materiais de terceiros. As bibliotecas têm condições próprias, descritas na [documentação do sistema](sistema_ia/README.md).
- **Escopo público:** documentos pessoais, credenciais, ambientes locais e caches ficam fora do conteúdo público. As exclusões estão registradas em [PUBLICACAO.md](PUBLICACAO.md).
