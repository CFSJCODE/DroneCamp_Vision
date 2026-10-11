# Plataforma DroneCamp Vision

A página de revisão (`data/reviews/<inspeção>/index.html`) é a plataforma do produto: uma única página, sem dependências externas, com nove seções navegáveis pela barra lateral. Com o servidor local ela recebe fotos e vídeos, gera laudos e pacotes de evidências e acompanha treinos ao vivo; aberta do disco, a revisão continua funcionando inteira.

| Seção | O que faz | De onde vêm os dados |
| --- | --- | --- |
| Início | Indicadores da inspeção aberta (fotos, decisões, ocorrências, sugestões da IA pendentes, classes com poucos exemplos), ações rápidas, cobertura por classe, triagem, modelo das sugestões, último treino e inspeções recentes | Registro da revisão, rascunho local do navegador e `runs/` |
| Inspeções | Todas as inspeções do espaço de trabalho com fotos, progresso das decisões, sugestões pendentes e atalhos para revisar, gerar laudo ou abrir | `data/reviews/*/registry.json` (servidor) ou só a inspeção da página (disco) |
| Importar mídia | Arrastar fotos (JPG, PNG, WEBP, TIFF, BMP) e vídeos (MP4, MOV, AVI, MKV); vídeos viram quadros no intervalo escolhido; cria uma inspeção nova e, se pedido, gera as sugestões do modelo em seguida | `ingest.py` pelo servidor local |
| Revisão | Fila com miniaturas e filtros, foto com caixas e sugestões tracejadas, limiar de confiança, editor de caixas (desenho ou coordenadas), decisão por foto e salvamento do arquivo de revisão | Registro + `suggestions.json` (mesma lógica de antes) |
| Laudos e downloads | Laudo técnico no formato ABNT em HTML e PDF com as evidências, pacote ZIP das fotos demarcadas (com CSV e manifesto), CSV das ocorrências, arquivo de revisão e histórico do que já foi gerado | `exports.py` pelo servidor local; CSV e JSON também offline |
| Treinamento | Ciclo de fine-tuning com os comandos prontos, execuções de `runs/`, curvas por época e formulário de novo treino piloto com estimativa de duração | `runs/*/execution.json`, `summary.json`, `fit/results.csv`, `data/pilot/*` |
| Monitoramento | Progresso do treino em andamento, métricas da última época, curvas ao vivo e a tarefa do servidor com o log | `fit/results.csv` relido a cada 5 s pelo servidor local |
| Modelos | Critério de adoção, comparação entre modelos por divisão, caixas por classe no dataset de cada treino e melhor mAP50 por treino | `runs/compare_*.json`, `execution.json` |
| Classes | Cadastro das categorias de ocorrência: nome de exibição, cor, ativo/inativo e classes propostas (sempre marcadas como "exige treino"); aviso fixo de que mudanças nas classes exigem novo ciclo de treinamento | `configs/taxonomy.json` (só leitura) + `configs/taxonomy_overrides.json` pelo servidor local ([análise da taxonomia](analise_taxonomia_classes.md)) |

Em toda seção a página repete o limite do produto: sugestões da IA aguardam decisão humana; gravidade, diagnóstico e recomendações dependem de validação técnica e nunca são inferidos. O laudo sai como "preliminar, sujeito a validação por responsável técnico", com esses campos em branco.

## Dois modos de abrir

1. **Direto do disco** (o `index.html` da inspeção, sem servidor): funciona offline e sem Python. Revisão, CSV e arquivo de revisão funcionam; importação, laudos e pacotes mostram um aviso pedindo o servidor; treinamento, monitoramento e modelos exibem o instantâneo de quando a página foi gerada.
2. **Servidor local** (`Abrir plataforma DroneCamp.cmd`, ou `python -m dronecamp_ia platform`): servidor da biblioteca padrão preso a `127.0.0.1:8765`. Sem `--registry`, trabalha com todas as inspeções de `data/reviews/` (modo espaço de trabalho) e abre a mais recente; `--registry` (repetível) limita a lista. Acrescenta:
   - importação de fotos e vídeos pela página (uma importação por vez; os arquivos passam por `.runtime/uploads/<lote>/` e, criada a inspeção, ficam guardados em `data/reviews/<inspeção>/midia_enviada/`, vídeos inteiros inclusive);
   - laudo preliminar em `data/reviews/<inspeção>/laudos/<data>_laudo_<inspeção>/` (`laudo.html`, `laudo.json`, `manifesto.json` e `laudo.pdf` quando o extra `reports` está instalado) e pacote ZIP em `exportacoes/`;
   - monitoramento ao vivo de um treino em andamento (lê `results.csv` enquanto a Ultralytics escreve);
   - **Iniciar treino** (`train-pilot`) e **Gerar sugestões** (`suggest` na inspeção aberta), uma tarefa por vez, com log em `.runtime/jobs/`;
   - ao salvar o arquivo de revisão, uma cópia vai para `data/reviews/<inspeção>/feedback_inbox/` (além do download).

O servidor não aprova dados, não adota modelos, não apaga inspeções e não infere gravidade. Só aceita os comandos `train-pilot` e `suggest`, com cada opção validada (datasets só de `data/pilot/`, pesos só de `models/` ou `runs/`, `imgsz` múltiplo de 32), serve arquivos apenas da pasta de cada inspeção e de `runs/`, e exige um token sorteado na partida para qualquer alteração ou envio, o que impede que outro site aberto no navegador dispare treinos ou envie arquivos.

### Rotas do servidor

| Rota | Uso |
| --- | --- |
| `GET /`, `GET /r/<pasta>/` | Página da inspeção padrão ou de uma inspeção específica |
| `GET /api/workspace?review=<pasta>` | Inspeções com fotos, decisões, sugestões pendentes e edificações; capacidades deste PC (`pdf`, `video_frames`) |
| `GET /api/operations?review=<pasta>` | Treinos, curvas, comparações, datasets, pesos e arquivos gerados da inspeção |
| `GET /api/jobs`, `POST /api/jobs`, `POST /api/jobs/stop` | Tarefa em andamento, iniciar e interromper |
| `POST /api/uploads` → `PUT /api/uploads/<lote>/files` → `POST /api/uploads/<lote>/ingest` → `GET /api/uploads/<lote>` | Lote de envio, um arquivo por requisição (cabeçalho `X-File-Name`), importação em segundo plano e acompanhamento |
| `POST /api/export/evidence` | ZIP das fotos demarcadas (`only`: all, decided, positive; `max_side`; `include_suggestions`) |
| `POST /api/report` | Laudo técnico ABNT (título, cliente, local, responsável técnico, observações, `details` com os dados técnicos opcionais, `pdf`, `all_images`) |
| `GET /api/reviews/<pasta>/files` | Laudos e pacotes já gerados |
| `POST /api/feedback` | Cópia do arquivo de revisão exportado |
| `GET /api/classes` | Classes da taxonomia e propostas, com `id`, `name` (slug), `display_name`, `color`, `active`, `in_detector`, `requires_training` |
| `POST /api/classes` | Nova classe proposta (`name` em snake_case, `display_name`, `color`, `criterion`); grava em `configs/taxonomy_overrides.json`, nunca em `taxonomy.json`, e não retreina nada |
| `PUT /api/classes/<id>` | Edita `display_name`, `color` e `active`; `name` e `criterion` só nas propostas (o slug das classes da taxonomia é o que os pesos conferem) |
| `DELETE /api/classes/<id>` | Exclusão lógica (`active: false`); a classe some das opções de caixa nova e pode ser reativada |

As rotas que alteram algo exigem o cabeçalho `X-DroneCamp-Token`. Decisões enviadas aos laudos e pacotes são as mesmas do arquivo de revisão (`image_sha256`, `status`, `boxes`, `notes`, `confirmed_complete`, `dismissed_suggestions`): valem sobre o registro, foto a foto.

## Importar mídia pelo terminal

O mesmo fluxo da página existe como comando, útil para pastas grandes já no disco:

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia import-media --source "D:\voo\fotos" --source "D:\voo\video.mp4" --output "data\reviews\galpao_b_2026_10" --group galpao_b --every-seconds 2 --max-frames 200
```

Cada foto é congelada por SHA-256 em `originals/` (girada pelo EXIF antes, quando preciso; TIFF e BMP são regravadas como JPEG porque o navegador não as exibe); cada vídeo vira quadros JPEG amostrados a cada `--every-seconds` segundos (no máximo `--max-frames` por vídeo, todos na mesma cena); fotos repetidas entram uma vez. A inspeção nasce com todas as fotos "ambíguas", sem caixas e com `visual_review_status = pending_model_suggestions`; as sugestões vêm de `suggest --registry ...` (ou da opção "Gerar sugestões da IA após importar" na página). Quadros de vídeo exigem OpenCV (vem com a Ultralytics) ou `ffmpeg` no PATH. Inspeções com muitas fotos ocupam GB em `originals/`, `images/`, `evidence/` e, nas importadas pela página, `midia_enviada/` (os arquivos tal como enviados); `midia_enviada/` já está no `.gitignore`, as outras pastas acrescente como foi feito com `ceasa_lado_a_*`.

## Laudos e pacotes de evidências

- **Pacote de evidências (ZIP)**: `evidencias/NNN_<nome>.jpg` (fotos com as caixas desenhadas; sugestões pendentes tracejadas só se marcado), `ocorrencias.csv` (uma linha por caixa, com classe, coordenadas e origem: revisão humana ou registro), `fotos.csv`, `LEIA-ME.txt` e `manifesto.json` com o SHA-256 de cada arquivo e do registro.
- **Laudo técnico (ABNT)**: capa, folha de rosto, sumário, seções numeradas (identificação, objetivo, normas aplicáveis, metodologia, descrição do imóvel, registro fotográfico e análise de conformidades por categoria no estilo do laudo CEASA, quadro-resumo, recomendações gerais, considerações finais, responsabilidade técnica), referências e anexos (fotos sem ocorrência, sugestões pendentes, subsídios de IA, demarcações por foto, apoio ao responsável técnico). Conformidade, risco, diagnóstico, recomendação e prioridade saem como "pendente de validação técnica". Os dados do imóvel, do voo e do responsável técnico vêm do bloco opcional "Dados técnicos do laudo"; o que ficar vazio sai como "não informado". Detalhes em [laudo_abnt.md](laudo_abnt.md). O PDF usa `reportlab` (`pip install -e .[reports]`); sem ele, HTML e JSON saem normalmente.
- **CSV de ocorrências** e **arquivo de revisão (JSON)**: montados no navegador com o estado atual (rascunho local sobre o registro), também offline.

## Nomes legíveis

Tudo o que a plataforma mostra tem um nome em português seguido de um código curto (os 8 primeiros caracteres do hash, em maiúsculas); o id cru fica na dica (`title`) do elemento e nos comandos, porque é ele que nomeia pastas e arquivos. Os nomes são derivados dos ids por `naming.py` (e pelo bloco "Nomes legíveis" de `platform.js`, que `tests/test_naming.py` mantém idêntico); nada é renomeado no disco.

| Em disco | Na plataforma |
|---|---|
| `runs/pilot_train_20261007T010057Z_f5e5ba8b` | Treino Piloto F5E5BA8B · 07/10/2026 01:00 |
| `runs/.../fit/weights/best.pt`, `models/yolo26l.pt` | Treino Piloto F5E5BA8B · melhor época, Modelo Base YOLO26l |
| `data/reviews/ceasa_v8_revisao002_ff34e226416c` | CEASA Pavilhão A v8 · Revisão 002 FF34E226 |
| `data/reviews/ceasa_lado_a_v1`, `internet_v3_busca_aberta` | CEASA Lado A v1, Fotos da Internet v3 · Busca Aberta |
| `data/pilot/ceasa_v8_piloto_s42` | Dataset Piloto CEASA Pavilhão A v8 · Semente 42 |
| `runs/compare_v9_f5e5ba8b_vs_56038eb9_20261007.json` | Comparação F5E5BA8B × 56038EB9 · 07/10/2026 |
| `laudos/20261011T140301Z_laudo_<inspeção>/laudo.pdf` | Laudo em PDF · Gerado em 11/10/2026 11:03 |

Datas aparecem como `dd/mm/aaaa hh:mm` no fuso do navegador. Uma inspeção enviada pela plataforma guarda o nome digitado (`display_name` no `registry.json`, com acentos) e ele vence o nome derivado; as derivações (`import-review`, `migrate-review`) não herdam esse nome. Para dar a um treino o nome que a equipe usa, crie `runs/nomes.json` com o id completo ou só o sufixo de 8 caracteres como chave:

```json
{ "56038eb9": "v7.4 referência", "f5e5ba8b": "v9 Lado A parcial" }
```

O laudo escreve "Inspeção: <nome legível> (<pasta>)" e grava `registry_display_name` ao lado de `registry_version`.

Atalhos na revisão: `←`/`→` fotos, `B` desenhar, `A` aceitar a primeira sugestão, `X` descartar a primeira sugestão, `Del` excluir a caixa selecionada.

## Design

Identidade comercial clara, pensada para o cliente ver a tela ao lado do inspetor: barra lateral escura com rótulos, conteúdo em cartões brancos sobre fundo cinza-azulado, uma cor primária e uma cor própria para tudo o que veio da IA. O tema escuro é opcional (botão no rodapé da barra lateral; a escolha fica no navegador).

| Elemento | Como é |
| --- | --- |
| Tipografia | Manrope (títulos) e Inter (texto), com Segoe UI e fontes do sistema como reserva; sem internet, a página usa a reserva. Mono: Cascadia/Consolas |
| Cores | Primária azul `#2457F5`; violeta `#7C3AED` marca sugestões e tarefas da IA; verde, âmbar e vermelho indicam estado. Superfícies brancas sobre `#F4F6F9`, bordas `#E3E8EF`; no escuro, os mesmos papéis com tokens próprios |
| Superfícies | Cartões planos com borda de 1 px e sombra discreta; sem vidro nem desfoque |
| Cantos | 8 px em controles, 12 px em cartões, cápsula nas etiquetas |
| Navegação | Barra lateral fixa com dois grupos (Operação, Inteligência artificial) e contadores; até 860 px vira menu lateral deslizante com barra superior |
| Rótulos nas fotos | Caixas: rótulo na cor da classe com texto escuro. Sugestões da IA: placa escura com texto na cor da classe e contorno tracejado |
| Foco | Anel duplo em todos os controles, inclusive caixas de seleção, controle de confiança e foto |
| Movimento | 100–300 ms; "reduzir movimento" desliga animações |

Os arquivos ficam em `review_templates/`: `index.html` (estrutura), `platform.css` (tokens no início, em `:root` e `[data-theme="dark"]`) e `platform.js` (comportamento). `build_page` embute os três em um único HTML, então a página continua abrindo do disco. Os nomes `--primary`, `--ai`, `--ok`, `--warn`, `--bad`, `--muted`, `--line-strong` e `--chart-grid` também são lidos pelo script dos gráficos.

## Atualizar a página sem regerar as evidências

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia refresh-page --registry "data\reviews\ceasa_v8_revisao002_ff34e226416c\registry.json"
```

Regrava só o `index.html` com o modelo atual e o instantâneo de `runs/`. Fotos, evidências, rótulos propostos e `browser_data.json` ficam como estão; o comando recusa o registro se `browser_data.json` pertencer a outra versão. Rode uma vez por inspeção depois de atualizar o código para que as páginas em disco ganhem o visual novo (o servidor local já monta a página atual a cada abertura).

## Limites

- Métricas do piloto vêm de uma única edificação. A seção Modelos declara que nenhuma versão pode ser adotada enquanto não houver avaliação em edificação independente, comparação salva e análise de regressões por classe.
- A estimativa de duração usa a média de segundos por época das execuções anteriores no mesmo PC (escala com imgsz²); é uma ordem de grandeza.
- Um treino sem `summary.json` e sem nova época há mais de 15 minutos aparece como interrompido.
- A comparação depende de `scripts/compare_pilot_models.py`; a página só lê o JSON gerado.
- O laudo em HTML carrega as evidências dentro do arquivo (alguns MB por dezenas de fotos); para centenas de fotos, prefira o PDF ou o pacote ZIP.
- Repetição de fotos é detectada dentro da mesma importação, não entre inspeções.

## Código

- `src/dronecamp_ia/platform_server.py`: servidor local, espaço de trabalho, lotes de envio, laudos/pacotes e lista fechada de tarefas.
- `src/dronecamp_ia/class_catalog.py`: cadastro das classes (rotas `/api/classes`), edições em `configs/taxonomy_overrides.json` com histórico.
- `src/dronecamp_ia/ingest.py`: importação de fotos e vídeos (hash, EXIF, quadros) para um registro novo; comando `import-media`.
- `src/dronecamp_ia/exports.py`: pacote de evidências, dados do laudo (`laudo.json`) e CSV de ocorrências.
- `src/dronecamp_ia/laudo_abnt.py`: estrutura ABNT, normas, lista de verificação e desenho do laudo em HTML/PDF ([laudo_abnt.md](laudo_abnt.md)).
- `src/dronecamp_ia/operations.py`: coleta treinos, curvas, comparações, datasets e arquivos gerados (só leitura).
- `src/dronecamp_ia/review_render.py`: `build_page`, `empty_page_payload` e `refresh_review_page`.
- `src/dronecamp_ia/review_templates/`: `index.html`, `platform.css`, `platform.js`.
- `tests/test_platform.py`, `tests/test_ingest.py`, `tests/test_exports.py`: painel, página gerada, servidor (espaço de trabalho, envio, laudos), importação e exportações.
