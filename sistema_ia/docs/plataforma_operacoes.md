# Plataforma de operações YOLO

A página de revisão (`data/reviews/<registro>/index.html`) virou uma plataforma com cinco seções, navegáveis pelo trilho à esquerda:

| Seção | O que mostra | De onde vêm os dados |
| --- | --- | --- |
| Visão geral | Fotos, decisões, caixas atuais, sugestões da IA pendentes, cobertura por classe, triagem, modelo das sugestões e último treino | Registro da revisão, rascunho local do navegador e `runs/` |
| Revisão e classificação | Fila com miniaturas, foto com caixas e sugestões tracejadas, filtro de confiança, editor de caixas e decisão por foto | Registro + `suggestions.json` (mesma lógica de antes) |
| Treinamento e fine-tuning | Ciclo de seis etapas com os comandos prontos, execuções em `runs/`, curvas de perda e métricas por época, gráficos da Ultralytics e formulário de novo treino piloto com estimativa de duração em CPU | `runs/*/execution.json`, `summary.json`, `fit/results.csv`, `data/pilot/*` |
| Monitoramento | Progresso do treino (época, tempo decorrido, tempo restante), métricas da última época, curvas e a tarefa em andamento no servidor local com o log | `fit/results.csv` relido a cada 5 s pelo servidor local |
| Modelos e comparação | Critério de adoção, comparação entre modelos por divisão (recall e precisão), caixas por classe no dataset de cada treino e melhor mAP50 por treino | `runs/compare_*.json`, `execution.json` |

## Nomes dos treinos

Cada execução de `runs/` aparece com um nome legível no lugar do id (`pilot_train_<data>_<hash>`): por padrão "Treino piloto N", numerado em ordem cronológica dentro da etapa, seguido de um resumo (modelo base, épocas feitas, treino em janelas e melhor mAP50 na validação). Para dar o nome que a equipe usa, crie `runs/nomes.json` com o id completo ou só o sufixo de 8 caracteres como chave:

```json
{ "56038eb9": "v7.4 referência", "f5e5ba8b": "v9 Lado A parcial" }
```

O id continua visível em letra pequena, porque é ele que aparece nos comandos e nas pastas.

O tema escuro é o padrão; o botão da lua alterna para o tema claro, que mantém a paleta original da revisão. Atalhos na revisão: `←`/`→` fotos, `B` desenhar, `A` aceitar a primeira sugestão, `X` descartar a primeira sugestão, `Del` excluir a caixa selecionada.

## Design

Uma única linguagem visual: estrutura **Fluent 2** com **Liquid Glass** (vidro acrílico) e paleta escura inspirada no Moonlight (Deep Navy e Aqua Cyan). Os dois temas têm a mesma forma e mudam só os tokens de cor; o escuro é o padrão.

| Elemento | Como é |
| --- | --- |
| Tipografia | Poppins e Roboto (Google Fonts) com Segoe UI Variable e fontes do sistema como reserva; sem internet, a página usa a reserva |
| Cores | Base Deep Navy `#08111F` / Midnight Slate `#0D1726`; destaque Aqua Cyan `--primary`; âmbar `--ai` marca o que veio da IA; verde, âmbar e vermelho indicam estado. O trilho e o dock usam `--bg-secondary`, então ficam claros no tema claro |
| Cantos | 7px em controles, 14px em cartões, cápsula nos botões táteis |
| Botões | Cápsula com ícone (salvar), botões neon de vidro com raio de luz (ações das caixas) e botão iridescente; desabilitados ficam apagados e sem brilho |
| Navegação | Trilho compacto no computador; até 1000px vira dock com lupa de vidro na base (com o ponto verde de treino em andamento), e o botão de tema flutua no topo |
| Vidro | Translucidez com `backdrop-filter` (desfoque 28px e saturação), borda especular de 1px e fundo com gradientes estáticos |
| Rótulos nas fotos | Caixas: rótulo na cor da classe com texto escuro. Sugestões da IA: placa escura com texto na cor da classe e contorno tracejado |
| Foco | Anel duplo em todos os controles, inclusive caixas de seleção, controle de confiança e foto |
| Movimento | 100–600ms; "reduzir movimento" desliga animações, inclusive o brilho da decisão |

Sem suporte a `backdrop-filter`, ou com "reduzir transparência" ativado, o vidro vira superfície opaca. Tudo fica em `review_templates/index.html`, com os tokens no início do `<style>`; os nomes `--primary`, `--ai`, `--ok`, `--warn`, `--muted`, `--line-strong` e `--chart-grid` também são lidos pelo script dos gráficos.

## Dois modos de abrir

1. **Direto do disco** (`Abrir revisao DroneCamp.cmd`): funciona offline e sem Python. A revisão funciona inteira; treino, monitoramento e modelos mostram um instantâneo de quando a página foi gerada.
2. **Servidor local** (`Abrir plataforma DroneCamp.cmd`, ou `python -m dronecamp_ia platform --registry ...`): servidor da biblioteca padrão preso a `127.0.0.1:8765`. Acrescenta:
   - monitoramento ao vivo de um treino em andamento (lê `results.csv` enquanto a Ultralytics escreve);
   - botão **Iniciar treino** (`train-pilot`) e **Gerar sugestões** (`suggest` nesta revisão), uma tarefa por vez, com log em `.runtime/jobs/`;
   - ao salvar o arquivo de revisão, uma cópia vai para `data/reviews/<registro>/feedback_inbox/` (além do download).

A escolha por um servidor leve, e não por um backend permanente, mantém a página estática como fonte principal: o navegador não permite que um arquivo aberto do disco leia `runs/` durante um treino nem inicie processos. O servidor não importa revisões, não aprova dados, não adota modelos e não apaga arquivos. Ele só aceita os comandos `train-pilot` e `suggest`, com cada opção validada (datasets só de `data/pilot/`, pesos só de `models/` ou `runs/`, `imgsz` múltiplo de 32) e exige um token sorteado na partida para qualquer alteração, o que impede que outro site aberto no navegador dispare treinos.

## Atualizar a página sem regerar as evidências

```powershell
.\.venv\Scripts\python.exe -m dronecamp_ia refresh-page --registry "data\reviews\ceasa_v8_revisao002_ff34e226416c\registry.json"
```

Regrava só o `index.html` com o modelo atual e o instantâneo de `runs/`. Fotos, evidências, rótulos propostos e `browser_data.json` ficam como estão; o comando recusa o registro se `browser_data.json` pertencer a outra versão.

## Limites

- Métricas do piloto vêm de uma única edificação. A seção Modelos declara que nenhuma versão pode ser adotada enquanto não houver avaliação em edificação independente, comparação salva e análise de regressões por classe.
- A estimativa de duração usa a média de segundos por época das execuções anteriores no mesmo PC (escala com imgsz²); é uma ordem de grandeza.
- Um treino sem `summary.json` e sem nova época há mais de 15 minutos aparece como interrompido.
- A comparação depende de `scripts/compare_pilot_models.py`; a página só lê o JSON gerado.

## Código

- `src/dronecamp_ia/operations.py`: coleta treinos, curvas, comparações e datasets (só leitura).
- `src/dronecamp_ia/platform_server.py`: servidor local, lista fechada de tarefas e gravação do arquivo de revisão.
- `src/dronecamp_ia/review_render.py`: `build_page` e `refresh_review_page`.
- `src/dronecamp_ia/review_templates/index.html`: interface (HTML, CSS e JavaScript sem dependências externas).
- `tests/test_platform.py`: testes do painel, da página gerada e do servidor (caminhos, token, comandos permitidos).
