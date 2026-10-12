# Guia de manutenção do código

Este guia é o ponto de partida para corrigir ou implementar algo no `sistema_ia` sem
precisar ler o projeto inteiro. Ele responde três perguntas:

1. **Onde fica cada coisa?** (mapa das pastas e dos módulos)
2. **Se quebrar X, onde eu olho?** (tabela de sintomas)
3. **Para implementar Z, por onde eu começo?** (receitas passo a passo)

Todo o código está comentado arquivo por arquivo (módulos, plataforma, scripts e testes):

- **Cabeçalho de cada arquivo:** a função no projeto, a lista do que o arquivo contém, quem
  o chama, o que ele lê e grava no disco e uma seção **"Onde mexer"** com casos concretos.
- **Toda função, classe e método** tem docstring com o que faz, `Parâmetros:`, `Retorna:`,
  `Erros:` e `Chamada por:`. No `platform.js`, o mesmo vem em comentários `/** ... */`.
- **Cada bloco do corpo** de uma função tem um comentário dizendo o que faz e por quê; as
  funções longas têm etapas numeradas (`# 1.`, `# 2.`...).
- **Constantes** dizem a unidade e o efeito de mudar o valor.

Para entender um arquivo: leia o cabeçalho, procure `ENTRADA:` para achar onde o fluxo
começa e siga as etapas numeradas. Para alterar algo: procure `EXTENSAO:` no arquivo e leia
o `ATENCAO:` mais próximo.

## 1. Marcadores no código

Os pontos estratégicos do código têm um destes três marcadores em comentário:

| Marcador | Significado |
| --- | --- |
| `ENTRADA:` | Onde um fluxo começa (um comando da CLI, uma rota do servidor, um botão da página). Comece a ler por aqui. |
| `EXTENSAO:` | Onde mexer para **acrescentar** algo (comando, rota, seção, campo do laudo, classe, formato de arquivo). Diz todos os lugares que precisam mudar juntos. |
| `ATENCAO:` | Armadilha conhecida ou regra que, se quebrada, gera erro ou dado errado. Leia antes de alterar o trecho. |

Os marcadores são em ASCII (sem acento) para a busca funcionar em qualquer terminal.

Para listar todos, na pasta `sistema_ia`:

```powershell
# PowerShell (Windows)
Select-String -Path src\dronecamp_ia\*.py, src\dronecamp_ia\review_templates\* -Pattern "EXTENSAO:"
```

```bash
# Git Bash / Linux
grep -rn "EXTENSAO:" src/dronecamp_ia
```

No VS Code: `Ctrl+Shift+F` e procure `EXTENSAO:` (ou `ATENCAO:`, `ENTRADA:`).

Os arquivos grandes também já são divididos em seções com linhas assim, fáceis de achar
pelo título:

```python
# ---------------------------------------------------------------------------
# Laudo técnico preliminar
# ---------------------------------------------------------------------------
```

No `platform.js` e no `index.html` as seções da página seguem o mesmo nome da barra
lateral: `/* ======================= Revisão ... */` e `<!-- ===== REVISÃO ===== -->`.

## 2. Mapa das pastas

```
sistema_ia/
├── configs/            parâmetros e classes (é aqui que se ajusta quase tudo sem programar)
│   ├── project.yaml          treino, inferência, piloto, fatiamento, gpu_eval
│   ├── taxonomy.json         classes (IDs, nomes, critérios) — contrato com pesos e revisões
│   ├── taxonomy_overrides.json  edições feitas pela seção Classes (nome, cor, ativo)
│   ├── taxonomy_ceasa_v*.json   versões anteriores da taxonomia (usadas pelo migrate-review)
│   ├── review_groups.json    edificação de cada grupo (dataset de produção)
│   ├── vlm_assist.json       modelo de visão gratuito (OpenRouter), sem chave
│   └── zero_shot_prompts.json  frases da busca aberta (YOLOE)
├── data/
│   ├── reviews/<inspeção>/   uma pasta por inspeção/revisão (ver abaixo)
│   ├── pilot/<versão>/       datasets do treino piloto (build-pilot-data)
│   ├── versions/<versão>/    datasets de produção (build-reviewed-data)
│   └── reference/ceasa/      fotos extraídas do laudo de referência
├── models/              pesos base (yolo26l.pt, Git LFS)
├── runs/                tudo o que um comando executa: treinos, comparações, exportações
├── src/dronecamp_ia/    o código (pacote Python)
│   └── review_templates/  index.html + platform.css + platform.js (a plataforma)
├── tests/               testes unittest (um arquivo por módulo, em geral)
├── scripts/             utilitários avulsos (comparação e avaliação de modelos)
├── docs/                documentação (este guia está aqui)
└── .runtime/jobs/       logs das tarefas iniciadas pela plataforma
```

Dentro de `data/reviews/<inspeção>/`:

| Arquivo | Quem grava | O que é |
| --- | --- | --- |
| `registry.json` | prepare-review, import-media, suggest --source, import-review, migrate-review | **Fonte da verdade**: fotos (por SHA-256), caixas, status e quem decidiu. |
| `originals/` | import-media, suggest --source | Cópia congelada das fotos originais. |
| `suggestions.json` | suggest | Caixas propostas pelo modelo; presas ao hash do `registry.json`. |
| `browser_data.json` | render-review | Dados que a página embute (fotos, caixas, sugestões, classes). |
| `index.html` | render-review / refresh-page | A página gerada. **Não edite à mão**; ela é gerada no PC do projeto. |
| `images/`, `evidence/` | render-review | Cópia das fotos e fotos com as caixas desenhadas. |
| `laudos/`, `exportacoes/` | plataforma (`/api/report`, `/api/export/evidence`) | Laudos (PDF/HTML/JSON) e ZIP de evidências. |
| `ai_notes.json`, `laudo_apoio.json` | describe-occurrences, report-support | Texto do VLM gratuito, sempre pendente de validação. |

## 3. A jornada dos dados (do voo ao laudo)

```
fotos/vídeos ──import-media──▶ data/reviews/<nova>/registry.json        (ingest.py)
                │
                └─suggest────▶ suggestions.json (caixas candidatas)      (suggestions.py)
                                   │
render-review / refresh-page ──▶ index.html (página de revisão)          (review_render.py)
                                   │  revisor aceita/corrige/descarta e
                                   │  clica em "Salvar revisão"
                                   ▼
                     arquivo de revisão (JSON exportado pela página)
                                   │
import-review ─────────────────▶ registry.json novo (aprovações humanas)  (review_data.py)
                                   │
build-pilot-data ──────────────▶ data/pilot/<versão>/ (só fotos aprovadas)(pilot.py)
                                   │
train-pilot ───────────────────▶ runs/pilot_train_*/ (pesos + auditoria)  (training.py)
                                   │
compare-models / measure-models ▶ runs/gate_*/, runs/measure_*/          (model_gate.py)
report-models ─────────────────▶ runs/report_*/ (gráficos + HTML)         (model_report.py)
                                   │  adota só com ganho demonstrado
                                   ▼
plataforma → Laudos ───────────▶ laudos/ (PDF/HTML ABNT) e exportacoes/  (exports.py, laudo_abnt.py)
```

`retrain-cycle` (`cycle.py`) roda de `build-pilot-data` até `report-models` numa tacada só.

Regras que atravessam toda a jornada (não as quebre):

- **A IA nunca aprova nada.** Sugestão do modelo, busca aberta (YOLOE) e texto do VLM são
  candidatos; só o `import-review` de uma decisão humana muda o que entra no treino.
- **Fotos originais são identificadas por SHA-256.** Quase todo comando confere o hash;
  se a foto mudar, o comando para.
- **Nada é sobrescrito.** Comandos que geram dados gravam numa saída nova (`--output`)
  e recusam saída existente; `runs/` ganha uma pasta nova por execução.
- **Gravidade, diagnóstico e recomendação nunca são inferidos.** No laudo eles ficam
  "pendente de validação técnica" até um responsável técnico preencher.
- **`taxonomy.json` muda só por versão**, com `migrate-review` para as revisões antigas.

## 4. Mapa dos módulos

| Camada | Arquivo | Para que serve | Comandos |
| --- | --- | --- | --- |
| Entrada | `cli.py` | Lê os argumentos e chama o módulo certo. | todos |
| Entrada | `__main__.py` | Permite `python -m dronecamp_ia`. | — |
| Configuração | `config.py` | Lê `project.yaml` e `taxonomy.json`. | — |
| Configuração | `class_catalog.py` | Seção Classes (nomes, cores, ativo) sem tocar na taxonomia. | `/api/classes` |
| Utilidades | `io.py` | Hash, pastas em `runs/`, JSON, caminhos de outra máquina. | — |
| Utilidades | `naming.py` | Nomes legíveis ("Treino Piloto F5E5BA8B"). Gêmeo do bloco "Nomes legíveis" do `platform.js`. | — |
| Mídia | `ingest.py` | Fotos e vídeos enviados viram inspeção nova. | `import-media` |
| Mídia | `report_images.py` | Extrai as fotos do PDF do laudo de referência. | `extract-report` |
| Revisão | `review_data.py` | Registros, importação de decisões, migração de taxonomia. | `prepare-review`, `import-review`, `add-ai-proposals`, `migrate-review` |
| Revisão | `review_render.py` | Gera a página (`index.html`), evidências e resumo. | `render-review`, `refresh-page` |
| Revisão | `suggestions.py` | Caixas sugeridas pelo modelo (com fatiamento das fotos grandes). | `suggest` |
| Revisão | `autolabel.py` | Busca aberta YOLOE (classes sem exemplos). | `suggest --zero-shot`, `evaluate-autolabel` |
| Revisão | `active_learning.py` | Ordem da fila de revisão (bandit LinUCB). | `prioritize-review`, `learn-review` |
| Revisão | `calibration.py` | Probabilidade de o revisor aceitar uma sugestão (scikit-learn). | `fit-calibrator` |
| Dados | `pilot.py` | Dataset piloto a partir das revisões aprovadas. | `build-pilot-data` |
| Dados | `review_dataset.py`, `review_provenance.py` | Dataset e guarda de procedência de produção. | `build-reviewed-data` |
| Dados | `dataset.py` | Validação somente leitura de um dataset YOLO. | `validate-data` |
| Dados | `tiling.py`, `sampling.py` | Fatiamento de fotos de 20 MP; repetição de classes raras. | `tile-dataset` |
| Modelo | `backend.py` | Único arquivo que importa a Ultralytics. | — |
| Modelo | `model_family.py` | YOLO26 → YOLO27 como configuração. | `model-info` |
| Modelo | `training.py` | Treino piloto/produção, avaliação, auditoria do resultado. | `train-pilot`, `train`, `evaluate`, `tune`, `resume-pilot` |
| Modelo | `gpu_eval.py` | Avaliação na GPU (DirectML) durante o treino na CPU. | `train-pilot --gpu-eval` |
| Modelo | `hparam_bandit.py` | Busca de hiperparâmetros por successive halving. | `tune-pilot-bandit` |
| Modelo | `exporting.py`, `prediction.py` | ONNX com paridade; inferência com evidências. | `export`, `predict` |
| Avaliação | `matching.py` | Régua comum: TP/FP/FN por classe (IoU ≥ 0,5). | — |
| Avaliação | `model_gate.py` | Comparação por classe e regra de adoção. | `compare-models`, `measure-models` |
| Avaliação | `model_report.py` | Relatório com gráficos dos modelos. | `report-models` |
| Avaliação | `cycle.py` | Ciclo completo de retreino. | `retrain-cycle` |
| IA de apoio | `vlm_assist.py` | VLM gratuito descreve recortes e rascunha apoio ao laudo. | `describe-occurrences`, `report-support`, `list-free-models` |
| IA de apoio | `ai_panel.py`, `ai_usage.py` | Assistente IA da plataforma: chave/modelo no `.env`, chat com o instantâneo do projeto, consumo local e limites do OpenRouter. | `/api/ai/...` |
| Plataforma | `platform_server.py` | Servidor local (127.0.0.1): página, API, uploads, tarefas. | `platform` |
| Plataforma | `operations.py` | Treinos, curvas, comparações e datasets para a página. | — |
| Plataforma | `review_templates/` | A página: `index.html` (estrutura), `platform.css` (visual), `platform.js` (comportamento). | — |
| Laudo | `exports.py` | ZIP de evidências, CSV e montagem do laudo (`laudo.json`). | `/api/report`, `/api/export/evidence` |
| Laudo | `laudo_abnt.py` | Texto fixo (normas, metodologia) e desenho do laudo em HTML e PDF. | — |

### Como a página da plataforma é montada

1. `review_render.build_page` lê `index.html`, embute `platform.css` e `platform.js` e
   coloca os dados (`browser_data.json` + instantâneo de `operations.py`) dentro do JS.
   O resultado é **um único HTML** que abre até direto do disco, sem servidor.
2. Pelo comando `platform`, `platform_server.py` serve essa mesma página com um token na
   meta `dronecamp-server`. Com o token, o `platform.js` passa a chamar a API
   (`api('/api/...')`) para dados ao vivo, uploads, laudos e tarefas.
3. Cada seção da barra lateral é um `<section class="view" id="view-<nome>">` no
   `index.html`, uma entrada em `VIEWS` no `platform.js` e uma função `render<Nome>()`.

| Seção da página | Função no `platform.js` | Rotas do servidor | Python |
| --- | --- | --- | --- |
| Início | `renderHome` | `/api/workspace`, `/api/operations` | `platform_server.Workspace`, `operations.py` |
| Inspeções | `renderInspections` | `/api/workspace` | `platform_server.Workspace` |
| Importar mídia | `renderUpload` | `/api/uploads...` | `platform_server.UploadManager`, `ingest.py` |
| Revisão | `paint` (bloco "Revisão") | `/api/feedback` | `platform_server._save_feedback` |
| Laudos e downloads | `renderReports` | `/api/report`, `/api/export/evidence`, `/api/reviews/<pasta>/files` | `exports.py`, `laudo_abnt.py` |
| Treinamento | `renderTraining` | `/api/jobs` | `platform_server.JobRunner`, `_job_arguments` |
| Monitoramento | `renderMonitor` | `/api/operations`, `/api/jobs` | `operations.py` |
| Modelos | `renderModels` | `/api/operations` | `operations.py` |
| Classes | `renderClasses` | `/api/classes` | `class_catalog.py` |

## 5. Se quebrar X, olhe Y

Primeiro passo sempre: **leia a mensagem inteira**. A CLI transforma erros previstos em
`Etapa não concluída: <motivo>` (código de saída 2); o motivo diz o que fazer. Erros fora
desse padrão (traceback completo) são falhas de código: a última linha do traceback que
aponta para `src/dronecamp_ia/` é onde começar.

| Sintoma | Causa provável | Onde olhar / o que fazer |
| --- | --- | --- |
| Depois de `git pull`, a página continua com o visual ou os botões antigos. | O `index.html` de cada inspeção é gerado; ele não muda sozinho. | Rode `python -m dronecamp_ia refresh-page --registry data\reviews\<pasta>\registry.json` em cada inspeção. `review_render.refresh_review_page`. |
| "Taxonomia mudou: migre a revisão antes..." ou "usa uma taxonomia anterior à atual". | `taxonomy.json` mudou depois que a revisão foi criada. | `migrate-review` (cria cópia na taxonomia nova) e depois `render-review`. `review_data.migrate_review_taxonomy`. |
| "suggestions.json pertence a outra versão do registro". | O `registry.json` mudou (import-review, migração) e as sugestões ficaram velhas. | Gere de novo com `suggest --registry ... --weights ...`. |
| "browser_data.json pertence a outra versão do registro". | Registro mudou sem gerar a página. | `render-review --registry ...`. |
| "Feedback não corresponde à versão original da revisão". | O arquivo exportado pela página veio de outra versão do registro. | Abra a página da versão atual e exporte de novo. `review_data.import_human_feedback`. |
| "Saída já existe; use uma nova versão da revisão". | Proteção contra sobrescrever. | Use outro `--output` (ex.: `..._v11`). |
| "A foto original mudou" / "Imagem original diverge". | A foto em disco não tem mais o mesmo SHA-256 (editada, girada, recomprimida). | Restaure a original; nunca edite fotos dentro de `data/reviews/`. `io.file_hash`. |
| "Checkpoint não corresponde às N classes ativas". | Pesos treinados com outra lista de classes. | Use pesos treinados com a taxonomia atual ou treine de novo. `backend.check_domain_names`. |
| Treino parou no meio (queda de energia, Ctrl+C). | — | `train-pilot --resume runs\pilot_train_<...>` ou `resume-pilot runs\pilot_train_<...>`. `training.resume_pilot`. |
| Treino terminou mas a plataforma o mostra como falho. | A auditoria final reprovou o resultado. | `runs/<treino>/summary.json`, campo `reasons`. `training.review_training_result`. |
| Tarefa iniciada pela página falhou. | Erro no comando filho. | `.runtime/jobs/<data>_<comando>.log` (a seção Monitoramento mostra o fim do log). |
| "Já existe uma tarefa em andamento". | O servidor roda uma tarefa por vez. | Espere ou clique em interromper. `platform_server.JobRunner`. |
| "Token ausente ou inválido" (403). | Página aberta de outro servidor ou de uma execução anterior. | Recarregue a página pelo endereço que o comando `platform` imprimiu. |
| Plataforma não abre: endereço em uso. | Outra plataforma já ocupa a porta 8765. | Feche a outra janela ou use `platform --port 8766`. |
| "A página desta revisão ainda não foi gerada (render-review)". | Inspeção sem `browser_data.json`. | `render-review --registry ...`. |
| Laudo sai só em HTML, sem PDF. | `reportlab` não instalado. | `pip install reportlab` no `.venv` (extra `reports`). O motivo vem em `pdf_reason`. |
| `describe-occurrences` diz para definir `OPENROUTER_API_KEY`. | Sem chave do OpenRouter. | Assistente IA → Configuração da IA (testa e grava em `sistema_ia\.env`) ou crie o `.env` com `OPENROUTER_API_KEY=...` (nunca no repositório). `vlm_assist.load_vlm_settings`. |
| `--gpu-eval` falha ao iniciar. | `onnxruntime-directml` ausente. | Instale o extra `directml` (no lugar do `onnxruntime` comum). `gpu_eval.py`. |
| Classe nova não aparece no menu da caixa. | Classe criada na seção Classes fica "proposta" (fase 2); a revisão usa só as classes ativas da taxonomia. | Receita "Ativar uma classe" abaixo. |
| Botão ou seção da página não reage. | Erro de JavaScript. | No navegador, `F12` → Console. A linha indicada está dentro do `index.html` gerado; procure o mesmo trecho em `review_templates/platform.js`. |
| Erro "platform.css/platform.js não podem conter a tag de fechamento". | Algum texto `</script>` ou `</style>` entrou no JS/CSS. | Quebre a string (ex.: `'<' + '/script>'`). `review_render.build_page`. |

## 6. Para implementar Z, comece por W

Cada receita lista os arquivos na ordem em que costuma ser mais fácil mexer. Depois de
qualquer receita, rode os testes (seção 8).

### Mudar um parâmetro de treino ou de inferência
Só `configs/project.yaml`. Piloto: `pilot.training`; produção: `training`; sugestões:
`pilot.suggestion_conf` e `pilot.tiling`; inferência: `prediction`. Não precisa de código.
As opções `--epochs/--imgsz/--batch` do `train-pilot` sobrescrevem o YAML numa execução.

### Criar um comando novo na CLI
1. A lógica vai num módulo (novo ou existente), numa função que recebe `config` e devolve
   a pasta/arquivo gerado.
2. `cli.py`, `build_parser` (marcador `EXTENSAO:`): `commands.add_parser(...)` e as opções.
3. `cli.py`, `main`: um `elif args.command == "..."` com o `import` dentro do ramo.
   Atenção ao `else` final (marcador `ATENCAO:`): comando sem `elif` cai nele.
4. Uma linha na lista do docstring do topo de `cli.py`.
5. Um teste em `tests/` chamando `main([...])` ou a função direto.

### Liberar um comando para a página iniciar (seção Treinamento)
1. `platform_server._job_arguments`: um ramo `if kind == "..."` validando cada opção.
2. `platform.js`: um botão que chama `startJob('<comando>', { ...opções })`.
3. `index.html`: o botão dentro da seção Treinamento.

### Criar uma rota nova na API
1. Leitura: `do_GET` em `platform_server.py`; alteração: `do_POST` (o token já é conferido).
2. A função de verdade fica em outro módulo; a rota só valida e chama.
3. Na página: `api('/api/...')` (GET) ou `api('/api/...', corpo)` (POST).
4. Teste em `tests/test_platform.py` (há exemplos que sobem o servidor numa porta livre).

### Criar uma seção nova na plataforma
1. `index.html`: botão `.nav-item` com `data-view="<nome>"` e `<section class="view" id="view-<nome>">`.
2. `platform.js`: entrada em `VIEWS`, `if (view === '<nome>') render<Nome>()` em `showView`
   e um bloco `/* ===== <Nome> ===== */` com `render<Nome>()`.
3. `platform.css`: estilos novos só se os existentes (cartões, tabelas, chips) não servirem;
   use as variáveis de `:root` para manter a identidade visual.
4. Rode `refresh-page` numa inspeção e abra pela plataforma para conferir.

### Acrescentar um campo ao laudo
1. `laudo_abnt.DETAIL_FIELDS`: chave e rótulo.
2. `index.html`, formulário "Dados técnicos" (`reportDetails`): `<input data-detail="<chave>">`.
   O `platform.js` já envia todos os `data-detail` preenchidos.
3. `laudo_abnt.render_html` **e** `laudo_abnt.write_pdf`: onde o valor aparece (os dois!).
4. Teste em `tests/test_exports.py`.

### Acrescentar uma coluna ao CSV de ocorrências ou de fotos
`exports.OCCURRENCE_COLUMNS`/`PHOTO_COLUMNS` e o valor na mesma posição em
`_occurrence_rows`/`_photo_rows`.

### Ativar uma classe nova para o detector
É a mudança mais delicada, porque os IDs são o contrato entre labels, pesos e revisões.
1. Copie `configs/taxonomy.json` para `configs/taxonomy_ceasa_v<N>_<descrição>.json`
   (snapshot da versão atual, usado pela migração).
2. Em `taxonomy.json`: suba `version`, passe a classe para `phase: 1` com o **próximo ID
   ativo** (as ativas precisam ser 0, 1, 2... sem buraco; as de fase 2 vêm depois) e
   registre a mudança em `migration`. Nunca renumere nem apague classes antigas.
3. `review_render.COLORS`: acrescente uma cor para cada ID ativo novo (hoje há 14 cores,
   IDs 0–13; marcador `ATENCAO:`).
4. `migrate-review --registry <revisão> --previous-taxonomy <snapshot> --output <nova>` em
   cada revisão que continuará em uso, e `render-review` na nova.
5. Revise as fotos (a página marca as classes que precisam de conferência).
6. Só depois de haver caixas aprovadas: `build-pilot-data` → `train-pilot` →
   `compare-models`. Pesos antigos são recusados com a taxonomia nova (é esperado).

Mudar só o nome de exibição, a cor ou desativar uma classe: use a seção Classes da
plataforma (grava em `taxonomy_overrides.json`, sem treino).

### Aceitar um formato de foto ou vídeo novo
Foto: `dataset.IMAGE_EXTENSIONS` (e `ingest.WEB_IMAGE_SUFFIXES` se o navegador já exibe o
formato). Vídeo: `ingest.VIDEO_EXTENSIONS`.

### Dar nome legível a um tipo novo de execução em `runs/`
`naming.STAGE_NAMES` **e** o bloco "Nomes legíveis" do `platform.js`.
`tests/test_naming.py` roda os dois e exige o mesmo resultado (precisa do Node instalado).

### Mostrar uma métrica nova nas curvas de treino
`operations.CURVE_COLUMNS` (coluna do `results.csv` → nome curto) e o gráfico em
`platform.js` (`renderTraining`/`renderMonitor`), que lê `OPS.runs[].curves`.

### Trocar a versão do YOLO (ex.: YOLO27)
`model_family.FAMILIES` (se a família ainda não existir), `model:` em
`configs/project.yaml`, `python -m dronecamp_ia model-info` para conferir o suporte,
treino e `compare-models` antes de adotar. Detalhes em `docs/modelos_e_ia_gratuita.md`.

### Mudar a regra de adoção de modelos
`model_gate.decide` (limites `min_map_gain`, `max_class_recall_drop`, `min_class_boxes`).
Registre o motivo em `docs/decisoes.md`: afrouxar a regra faz modelo pior parecer melhor.

## 7. Armadilhas conhecidas

- **Finais de linha.** O repositório guarda os bytes como estão (`.gitattributes` com
  `* -text`). `pilot.py`, `suggestions.py` e `training.py` usam CRLF; os demais, LF.
  Edite com um editor que preserve o final de linha do arquivo (o VS Code mostra `CRLF`/`LF`
  na barra de status). Misturar os dois gera diffs enormes.
- **Template da página.** `index.html` e `platform.js` têm marcadores de substituição que
  `build_page` troca por texto. Não escreva o nome desses marcadores em comentários do
  template; o marcador de dados precisa aparecer uma única vez no `platform.js`.
- **Página gerada ≠ template.** Editar `data/reviews/*/index.html` não adianta (é
  regravado); edite `review_templates/` e rode `refresh-page`.
- **Paridade Python × JavaScript.** Os nomes legíveis existem em `naming.py` e no
  `platform.js`; mude os dois juntos.
- **HTML e PDF do laudo** são desenhados separadamente (`render_html` e `write_pdf`).
- **Paleta de 14 cores** em `review_render.COLORS` para as classes ativas.
- **A Ultralytics só é importada dentro de cada comando** (`cli.main`) e só por
  `backend.py`; comandos leves (`categories`, `platform`) abrem rápido por isso.
- **Windows.** O `if __name__ == "__main__"` do `__main__.py` é obrigatório (a Ultralytics
  cria subprocessos). Treino com `workers: 0` é o estável no Windows.
- **Segredos.** O repositório é público: chave do OpenRouter só em `.env`; documentos
  pessoais nunca entram no git.

## 8. Testes

Os testes usam `unittest` (não pytest). Na pasta `sistema_ia`:

```powershell
# todos (alguns minutos; os de ponta a ponta treinam de verdade)
.\.venv\Scripts\python.exe -m unittest discover -s tests

# só um arquivo ou um teste
.\.venv\Scripts\python.exe -m unittest tests.test_exports
.\.venv\Scripts\python.exe -m unittest tests.test_platform.ServerTests

# pular os de ponta a ponta (mais rápido)
$env:DRONECAMP_SKIP_E2E = "1"; .\.venv\Scripts\python.exe -m unittest discover -s tests
```

Alguns testes são pulados (aparecem como `skipped`) quando falta algo opcional: Node.js
(paridade dos nomes e gráfico de evolução), `reportlab` (PDF do laudo), pesos do Git LFS
(ponta a ponta) e ONNX exportado (DirectML). Pulado não é falha.

| Mexeu em | Rode |
| --- | --- |
| `cli.py`, `config.py`, `backend.py` | `tests.test_safety`, `tests.test_learning_loop`, `tests.test_ingest` |
| `platform_server.py`, `operations.py`, `review_templates/` | `tests.test_platform`, `tests.test_naming` |
| `exports.py`, `laudo_abnt.py` | `tests.test_exports` |
| `ingest.py` | `tests.test_ingest` |
| `class_catalog.py` | `tests.test_class_catalog` |
| `review_data.py`, `review_render.py` | `tests.test_review_feedback`, `tests.test_taxonomy_migration` |
| `pilot.py`, `training.py` | `tests.test_pilot`, `tests.test_training_artifacts`, `tests.test_fit_staging` |
| `tiling.py`, `suggestions.py` | `tests.test_tiling`, `tests.test_tiled_pipeline` |
| `model_gate.py`, `model_report.py`, `cycle.py` | `tests.test_learning_methods`, `tests.test_model_report`, `tests.test_cycle` |
| `vlm_assist.py`, `model_family.py` | `tests.test_vlm_assist`, `tests.test_model_family` |
| `ai_panel.py`, `ai_usage.py` | `tests.test_ai_panel` |

No fim, rode a suíte inteira antes de dar a mudança por pronta.

## 9. Roteiro seguro para qualquer mudança

1. `git pull` e crie um ramo: `git switch -c minha-mudanca`.
2. Ache o ponto de partida neste guia (seções 5 e 6) e os marcadores no arquivo.
3. Leia o docstring do topo do arquivo e o comentário `ATENCAO:` mais próximo.
4. Faça a mudança pequena, rode o teste do arquivo, depois a suíte inteira.
5. Se mudou o template da página, rode `refresh-page` numa inspeção e confira no navegador.
6. Se mudou dados (taxonomia, registros), confira que nada foi sobrescrito: os comandos
   gravam saídas novas, e a versão anterior precisa continuar existindo.
7. Commit com mensagem que diga o porquê; abra um PR para a `main`.
