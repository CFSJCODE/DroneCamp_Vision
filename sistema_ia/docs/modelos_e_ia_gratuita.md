# Modelos de visão: YOLO26 → YOLO27 e IA gratuita de apoio — 10/10/2026

Origem: pedido do Cláudio, dentro da reestruturação do DroneCamp Vision em produto, para aprimorar e refatorar o código e os modelos de visão computacional com YOLO26 **preparado para receber o YOLO27**, e usar modelos gratuitos de IA adequados a cada parte do fluxo (ele indicou o roteador `openrouter/free` e a coleção de modelos de visão do OpenRouter). A plataforma (interface, upload, download, laudos prontos) é tratada à parte; este documento cobre os modelos e o código de treino, inferência e avaliação.

## 1. Família e versão do detector (`model_family.py`)

Antes, "qual YOLO" estava espalhado em expressões regulares (`backend.py`) e em nomes de colunas do `results.csv` (`training.py`). Agora um único arquivo responde por isso:

| Pergunta | Onde | Como |
| --- | --- | --- |
| Que peso é este? | `parse_model_name("yolo26l.pt")` | família 26, tamanho l, tarefa detect, cabeça sem NMS, perda de distância `l1_loss` |
| Posso baixá-lo? | `ensure_downloadable` (usado por `load_detector`) | só nomes oficiais de detecção de família publicada; `yolo27l.pt` falha com "YOLO27 ainda não foi publicado (Ultralytics 8.4.172)" |
| A biblioteca instalada já tem a família? | `family_support` | confere, **sem baixar nada**, a pasta `ultralytics/cfg/models/<versão>` e o nome na lista de ativos oficiais (`GITHUB_ASSETS_NAMES`) |
| De que família é um `best.pt` treinado aqui? | `describe_checkpoint` (gravado em `execution.json`/`export.json` → `runtime.model_family`) | lê o peso de partida em `train_args` (ex.: `yolo26l.pt`) |
| Que colunas de perda auditar? | `distance_loss_columns` (usado por `review_training_result`) | `l1_loss` no YOLO26; família desconhecida aceita `l1_loss` ou `dfl_loss` |

Comando novo:

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia model-info
.\.venv\Scripts\python.exe -m dronecamp_ia model-info --model yolo27l.pt --output runs\model_info.json
```

Ele mostra o modelo configurado, o alvo (`requested_model: yolo27l.pt`), o suporte na Ultralytics instalada e a **lista de migração**, nesta ordem obrigatória:

1. **Pesos publicados** — `model-info` precisa mostrar `downloadable: true` para `yolo27l.pt` (depois de atualizar a Ultralytics para a versão que publicar o YOLO27).
2. **Treino piloto no mesmo dataset** — `retrain-cycle --weights yolo27l.pt ...` (ou `train-pilot --weights yolo27l.pt`) em `data/pilot/<versão>`, a mesma versão usada pelo YOLO26 de referência.
3. **Exportação ONNX com paridade** — `export`; `parity.json` com `runtime_parity_validated: true` (a cabeça do YOLO27 é diferente: a paridade `.pt × .onnx` é o que prova que a exportação continua correta).
4. **Comparação por classe** — `compare-models --baseline <YOLO26 atual> --candidate <YOLO27>`: `adopt: true` e leitura das regressões por classe; `measure-models`/`report-models` para o relatório.
5. **Modelo anterior preservado** — já é a regra: nenhum comando apaga ou sobrescreve pesos em `runs/`.
6. **Configuração** — só depois do gate: `model: yolo27l.pt` e `pilot.model: yolo27l.pt` em `configs/project.yaml`. Nenhuma linha de Python muda.

**Estado verificado em 10/10/2026:** a Ultralytics 8.4.172 instalada e a 8.4.175 (última no PyPI, pacote baixado e inspecionado) **não trazem** arquitetura (`cfg/models/27`) nem pesos oficiais do YOLO27. Os dados de `FAMILIES[27]` (tamanhos n/s/m/l, cabeça sem NMS) vêm da prévia oficial registrada em [pesquisa Ultralytics](pesquisa_ultralytics.md) e devem ser confirmados no lançamento; a perda de distância do YOLO27 é desconhecida, por isso a auditoria aceita `l1_loss` ou `dfl_loss` para essa família. O teste `test_installed_library_reports_support_without_downloading` passa a documentar a mudança quando a biblioteca publicar o modelo.

## 2. IA gratuita de apoio: onde faz diferença e onde não

| Etapa do fluxo | Modelo gratuito | Decisão |
| --- | --- | --- |
| Achar e classificar caixas | YOLO26l piloto (treinado aqui) | continua sendo o detector; é o único que conhece as 13 classes |
| Procurar o que o piloto não viu | YOLOE-26l (`suggest --zero-shot`, já existente) | opcional: ajudou em calhas de outras edificações, 0/29 nas fotos do CEASA ([marcação automática](marcacao_automatica.md)) |
| **Explicar o recorte de uma caixa** | **VLM gratuito via OpenRouter (novo, `describe-occurrences`)** | descrição do que está visível, classe sugerida de forma independente, legibilidade; tudo pendente |
| **Rascunhar o texto de apoio ao laudo** | **`report-support` (novo)** | ocorrências revisadas + descrições pendentes; gravidade, diagnóstico e recomendação ficam em branco |
| Segmentar/refinar contornos (SAM), detecção aberta (Grounding DINO, OWLv2) | não implementado | sem ganho mensurável agora: a revisão usa caixas, não máscaras, e a busca aberta já existente foi medida como fraca no CEASA; voltar a avaliar quando houver fotos de outras edificações revisadas (mesmo protocolo do `evaluate-autolabel`) |

### 2.1 Como funciona o `describe-occurrences`

```powershell
# uma vez: copie .env.example para .env e preencha OPENROUTER_API_KEY (o .env é ignorado pelo git)
.\.venv\Scripts\python.exe -m dronecamp_ia describe-occurrences --registry data\reviews\ceasa_v8_revisao002_ff34e226416c\registry.json --limit 40
.\.venv\Scripts\python.exe -m dronecamp_ia describe-occurrences --registry data\reviews\ceasa_lado_a_v1\registry.json --include-suggestions --limit 60
```

1. Para cada caixa de foto com aprovação humana (`source: humana`; caixas de fotos sem aprovação, como propostas `first_pass_ai`, entram como `proposta` e ficam fora da concordância) e, com `--include-suggestions`, cada sugestão pendente do modelo, o comando recorta a foto com 30% de margem, lado mínimo 96 px e máximo 768 px (`configs/vlm_assist.json`), e confere o SHA-256 do original antes.
2. Envia o recorte ao OpenRouter (`openrouter/free` por padrão: o roteador sorteia um modelo gratuito que aceita imagem) com as 13 classes e seus critérios. **A classe humana ou sugerida não vai no prompt**: a resposta é uma leitura independente.
3. O VLM responde em JSON: `descricao` (1–3 frases), `classe_sugerida` (um slug, `nenhuma` ou `incerto`), `evidencias` e `legibilidade`. O prompt proíbe gravidade, causa, risco e recomendação; se o modelo responder fora do formato, a nota fica como `incerto` com o texto bruto.
4. Grava `ai_notes.json` ao lado do registro, preso ao hash do registro e ao hash do prompt (`PROMPT_VERSION`), com o modelo que de fato respondeu (`model_used`), tokens, `status: "pendente"` e `severity/diagnosis/recommendation: null`.
5. É **retomável**: notas já gravadas para o mesmo recorte, mesmo prompt e mesmo modelo pedido (`resume_sha256`; trocar `--model`/`OPENROUTER_MODEL` reenvia tudo) não são pedidas de novo; `--limit N` limita as chamadas da execução (os modelos gratuitos têm cota). Chave recusada (401/403), cota (429) ou rede fora interrompem o comando sem gravar nada inventado; erros depois da primeira resposta ficam registrados por caixa.
6. `agreement` no mesmo arquivo mede, por classe, se a classe sugerida pelo VLM coincide com a humana (`agree`), se ele não viu ocorrência (`none_or_uncertain`) ou confundiu com outra (`confusions`). **É essa medida que decide se o VLM ajuda na triagem**; sem ela, o texto é só descrição.

Outros comandos:

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia list-free-models            # modelos gratuitos com entrada de imagem
.\.venv\Scripts\python.exe -m dronecamp_ia describe-occurrences --registry ... --model qwen/qwen2.5-vl-72b-instruct:free   # slug fixo (exemplo; confira na lista)
.\.venv\Scripts\python.exe -m dronecamp_ia report-support --registry data\reviews\ceasa_v8_revisao002_ff34e226416c\registry.json
```

**Cadeia de modelos gratuitos (12/10/2026).** `--model` aceita vários slugs separados por vírgula: o OpenRouter usa o primeiro e cai para o próximo quando ele falha ou está limitado (campo `models` da API). `--model free-vision` monta essa cadeia com todos os modelos `:free` com entrada de imagem do catálogo atual, Gemma 4 primeiro, sem o roteador `openrouter/free` e sem classificadores de segurança: no teste de 12/10 o roteador mandou 36 de 40 recortes para modelos que não descreveram a foto. A cadeia **não** aumenta a cota: o limite diário de chamadas gratuitas vale para a conta inteira (no teste, HTTP 429 depois de ~40 chamadas). Exemplo: `describe-occurrences --registry ... --include-suggestions --model free-vision`. Com `--include-suggestions`, a opinião do VLM sobre cada sugestão do detector aparece na página de revisão ("VLM: não vê ocorrência", "VLM: <classe> (concorda)"), só como informação; aceitar ou descartar continua sendo decisão humana.

`report-support` grava `laudo_apoio.json` e `laudo_apoio.md` na pasta da revisão (o registro precisa estar na taxonomia atual, conferida pelo hash, como no `suggest`): só fotos com aprovação humana, uma ocorrência por caixa, com classe, critério da taxonomia, nota do revisor, gravidade histórica do laudo de referência (só como referência) e a descrição da IA quando existir. Gravidade, diagnóstico e recomendação saem como `null`/"a definir pelo responsável técnico". A plataforma pode consumir `draft_report_support()` e `describe_occurrences()` de `vlm_assist.py`.

### 2.2 Segurança da chave

- A chave é lida de `OPENROUTER_API_KEY` (ambiente) ou de `sistema_ia/.env` (ignorado pelo git; modelo em `.env.example`). Ela nunca é gravada em `ai_notes.json`, nos logs ou na configuração; `settings.describe()` registra só `api_key_present: true`.
- Uma chave colada em conversa, página ou arquivo compartilhado deve ser considerada exposta: gere outra no painel do OpenRouter e substitua a do `.env`.
- As fotos (recortes) saem do computador para o provedor escolhido pelo roteador. Para campanhas com restrição de sigilo, não use o comando ou fixe um provedor com política conhecida.

### 2.3 O que ainda não foi medido

A nuvem onde este código foi escrito não alcança `openrouter.ai` (proxy responde 403), então **nenhuma chamada real foi feita**: a integração foi validada com um transporte falso que reproduz as respostas da API (JSON puro, com cerca ```` ```json ````, com texto em volta, 401, 429, 500). A primeira execução real, no PC do Cláudio com a chave no `.env`, deve começar com `--limit 20` na v8 e ler `agreement` antes de usar as descrições na revisão ou no laudo.

## 3. Validação desta entrega

- 34 testes novos (`tests/test_model_family.py`, `tests/test_vlm_assist.py`), todos sem rede; suíte completa executada na nuvem com a Ultralytics 8.4.172 e o `yolo26l.pt` real (ver `docs/validacao.md` e o PR para a contagem final).
- `model-info` executado na nuvem: YOLO26L `downloadable: true`; YOLO27L `downloadable: false`, com a lista de migração.
- `report-support` executado na revisão v7 real: 34 fotos aprovadas, 117 ocorrências, todas com os campos técnicos em branco.
- **Não validado:** chamadas reais ao OpenRouter (sem rede na nuvem), qualidade das descrições nas fotos do CEASA e do Lado A, YOLO27 (não publicado).

## 4. Dependências para o próximo avanço

1. Rodar `describe-occurrences --limit 20` na v8 no PC e ler `agreement`; se a concordância por classe for baixa, manter o VLM só como descrição de apoio e não como triagem.
2. Quando a Ultralytics publicar o YOLO27: `pip install -U ultralytics`, `model-info`, e seguir a lista de migração acima no mesmo `data/pilot/<versão>` do YOLO26 de referência (`56038eb9`).
3. Fotos revisadas de outras edificações continuam sendo o maior ganho possível para o detector; nenhum modelo gratuito substitui isso.
