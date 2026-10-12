/* ---------------------------------------------------------------------------
   DroneCamp Vision — comportamento da plataforma (sem dependências externas).

   A página recebe os dados embutidos pelo marcador de dados (review_render.build_page):
   fotos, caixas, sugestões, classes e o instantâneo de operações (treinos,
   comparações, datasets). Aberta do disco, tudo funciona com esse instantâneo.
   Servida pelo comando ``platform``, a página ainda consulta o servidor local
   (token na meta ``dronecamp-server``) para monitoramento ao vivo, tarefas,
   importação de mídia, laudos e pacotes de evidências.

   Regras que o código preserva: rascunhos ficam no navegador até o revisor
   salvar o arquivo de revisão; sugestões da IA nunca viram caixa sozinhas; toda
   edição invalida a confirmação anterior da foto.
   --------------------------------------------------------------------------- */
/* ---------------------------------------------------------------------------
   MAPA DO ARQUIVO (de cima para baixo; procure pelos títulos "======="):
     1. Estado global e utilitários (DATA, OPS, TOKEN, SERVER, state, active, fmt*)
     2. Nomes legíveis (NAMING): cópia em JS de naming.py
     3. Construtores de DOM (el, icon), mensagens (message, toast) e rascunho (persist)
     4. Navegação entre seções (VIEWS, showView, menu lateral)
     5. Revisão: fila de fotos, caixas, sugestões da IA, desenho, zoom, decisão, exportação
     6. Início (painel), 7. Inspeções, 8. Importar mídia, 9. Laudos e downloads
    10. Gráficos SVG (lineChart) e rótulos de execuções/pesos/métricas
    11. Treinamento, 12. Monitoramento, 13. Modelos, 14. Classes (cadastro), 14b. Assistente IA (configuração, uso e limites, chat)
    15. Servidor local (api, refreshWorkspace, refreshOperations, tarefas)
    16. Inicialização (roda uma vez quando a página abre)

   QUEM USA ESTE ARQUIVO: review_render.build_page lê platform.js e o embute no
   index.html (no lugar do marcador do JS), trocando o marcador de dados da linha
   abaixo pelo JSON da revisão. O comando render-review grava a página em
   data/reviews/<inspeção>/index.html; o comando platform (platform_server.py) a
   serve em http://127.0.0.1 com um token na meta "dronecamp-server".

   O QUE LÊ E GRAVA: só lê o JSON embutido e, com servidor, as rotas /api/...;
   grava rascunhos no localStorage do navegador (chave dronecamp-review-<hash>),
   o tema (dronecamp-theme) e a seção aberta (sessionStorage dronecamp-view).
   Arquivos (revisão JSON, CSV) são baixados pelo navegador; no servidor, as
   rotas gravam em data/reviews/<inspeção>/ e runs/.

   ONDE MEXER (procure os marcadores EXTENSAO:, ENTRADA: e ATENCAO:):
     - seção nova na plataforma: bloco "Navegação entre seções" (VIEWS e showView);
     - rota nova do servidor: função api() no bloco "Servidor local" + platform_server.py;
     - coluna nova no CSV de ocorrências: handler de downloadCsv (bloco "Laudos");
     - métrica nova nos gráficos: LOSS_SERIES / METRIC_SERIES (bloco "Gráficos SVG");
     - nome legível novo (palavra, local, etapa): bloco NAMING e naming.py juntos;
     - atalho de teclado novo: listener de keydown no fim do bloco "Revisão";
     - estado novo de treino: stateLabel / stateClass (bloco "Treinamento").
   ATENCAO: não escreva nos comentários deste arquivo os nomes dos marcadores que
   review_render troca (dados, JS, CSS) nem a tag de fechamento de script: a
   troca é feita por texto e quebraria a página gerada.
   --------------------------------------------------------------------------- */
// DATA: todo o conteúdo da inspeção embutido pelo review_render (fotos em DATA.images,
// classes em DATA.classes, sugestões em DATA.suggestions, hash do registro em
// DATA.registry_sha256, versão em DATA.version, instantâneo de operações em DATA.operations).
// ATENCAO: esta linha precisa conter exatamente um marcador de dados; review_render.build_page
// recusa o template se houver zero ou mais de um.
const DATA = __REVIEW_DATA__;
// $: atalho para document.getElementById. Ex.: $('canvas') devolve o <canvas id="canvas"> do index.html.
const $ = id => document.getElementById(id);
// OPS_EMPTY: forma mínima do instantâneo de operações, usada quando a página não trouxe nenhum.
// ATENCAO: o resto do código acessa OPS.runs, OPS.weights etc. sem checar; mantenha as listas aqui.
const OPS_EMPTY = { runs: [], comparisons: [], datasets: [], weights: [], pilot_defaults: {} };
// OPS: instantâneo de operações (treinos em runs/, comparações, datasets, pesos, padrões do piloto).
// Vem embutido na página e, com servidor, é trocado a cada 5 s por refreshOperations (/api/operations).
let OPS = DATA.operations || OPS_EMPTY;
// Token presente = página servida pelo comando platform (monitoramento ao vivo e tarefas).
const TOKEN = (document.querySelector('meta[name="dronecamp-server"]') || {}).content || '';
// SERVER: verdadeiro só quando há token E a página veio por http(s). Aberta do disco (file://), é falso
// e nenhuma rota /api/ é chamada.
const SERVER = Boolean(TOKEN) && location.protocol.startsWith('http');
// Revisões abertas pelo servidor: { current, items: [{ name, href, images }] }; null quando aberta do disco.
const REVIEWS = (() => {
  try { return JSON.parse((document.querySelector('meta[name="dronecamp-reviews"]') || {}).content || 'null') } catch { return null }
})();
// Espaço de trabalho (lista de inspeções e capacidades do servidor), carregado de /api/workspace.
let WORKSPACE = null;
// HAS_IMAGES: falso quando a página foi aberta sem inspeção (espaço vazio); várias funções saem cedo nesse caso.
const HAS_IMAGES = Array.isArray(DATA.images) && DATA.images.length > 0;
// names: rótulo do status PROPOSTO no registro (o que a triagem sugeriu).
// EXTENSAO: status novo de foto: acrescente aqui, em decisionNames, no <select id="decision"> do
// index.html, em reviewTotals (objeto triage) e no Python que valida o arquivo de revisão.
const names = { positive: 'Positiva proposta', negative: 'Negativa proposta', ambiguous: 'Ambígua', excluded: 'Excluída' };
// decisionNames: rótulo da decisão HUMANA registrada no rascunho (mesmas chaves de names).
const decisionNames = { positive: 'Há ocorrências', negative: 'Negativa', ambiguous: 'Ambígua', excluded: 'Excluída' };
// storageKey: chave do rascunho no localStorage. Inclui o hash do registro: se o registro mudar,
// o rascunho antigo não é aplicado sobre fotos diferentes.
// ATENCAO: não troque a fórmula da chave, senão os rascunhos já guardados no navegador somem da página.
const storageKey = 'dronecamp-review-' + DATA.registry_sha256;
// saved: rascunho lido do navegador ({ images: { <sha da foto>: decisão }, reviewer }); vazio se não houver.
let saved = {};
try { saved = JSON.parse(localStorage.getItem(storageKey) || '{}') } catch {}
// Zoom visual do visor (escala e deslocamento em px de tela); volta a 100% a cada foto.
const zoom = { scale: 1, x: 0, y: 0, pan: null };
// state: estado da tela de revisão e da navegação.
//   index: posição da foto aberta em DATA.images; selected: índice da caixa selecionada (-1 = nenhuma);
//   drawing: modo desenho ligado; start: ponto inicial do arraste [x, y] em pixels da foto;
//   img: Image carregada da foto aberta; drafts: rascunhos por sha da foto; reviewer: identificador;
//   view: seção aberta (chave de VIEWS); run: id da execução escolhida na tabela de Treinamento.
let state = { index: 0, selected: -1, drawing: false, start: null, img: new Image(), drafts: saved.images || {}, reviewer: saved.reviewer || '', view: 'home', run: null };
// clone: cópia profunda via JSON (suficiente: os dados são só objetos, listas, números e textos).
const clone = value => JSON.parse(JSON.stringify(value));
// item: foto aberta (objeto de DATA.images).
const item = () => DATA.images[state.index];
// draft: rascunho da foto aberta ou, se ainda não houver, um rascunho novo a partir das caixas do registro.
const draft = () => state.drafts[item().image_sha256] || { boxes: clone(item().boxes), status: 'pending', notes: item().human_notes || '', confirmed_complete: false };
// active: rascunho em edição da foto aberta (cópia; só vai a state.drafts quando persist() roda).
let active = HAS_IMAGES ? draft() : { boxes: [], status: 'pending', notes: '', confirmed_complete: false };
// classOf: categoria pelo id (cor e rótulo). Id desconhecido vira "classe N" em cinza para não quebrar a tela.
const classOf = id => DATA.classes.find(value => value.id === id) || { id, label: `classe ${id}`, color: '#64748B' };
// pct: fração 0..1 em porcentagem com 1 casa ("12.3%"); null/undefined vira "—".
const pct = value => value == null ? '—' : `${(value * 100).toFixed(1)}%`;
// fmtBytes: tamanho em bytes legível (GB/MB/kB/B, base 1000).
const fmtBytes = n => n == null ? '—' : n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB` : n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : n >= 1e3 ? `${Math.round(n / 1e3)} kB` : `${n} B`;
// Data legível dd/mm/aaaa hh:mm no fuso do navegador (nunca o carimbo ISO cru).
const fmtDate = value => { if (!value) return '—'; const date = new Date(value); return isNaN(date) ? String(value) : date.toLocaleString('pt-BR', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' }).replace(',', '') };
// Quantas edificações a inspeção cobre, sem expor os códigos internos (building_group).
const buildingCount = n => n ? `${n} edificaç${n > 1 ? 'ões' : 'ão'}` : 'edificação não informada';
// ATENCAO: tests/test_naming.py recorta este bloco pelo texto dos dois títulos (início e fim)
// e o executa no Node; não renomeie os títulos nem use aqui nada definido fora do bloco.
/* ======================= Nomes legíveis ======================= */
// Mesmo algoritmo de naming.py (tests/test_naming.py roda os dois e exige resultados iguais).
// Ids e pastas em disco nunca mudam; aqui só se deriva o nome mostrado.
/**
 * NAMING: gerador de nomes legíveis (inspeções, execuções, pesos, comparações, datasets, arquivos gerados).
 * Função auto-executada que devolve um objeto com as funções públicas listadas no return.
 * ATENCAO: paridade Python × JS. Qualquer mudança aqui tem de ser repetida em naming.py
 * (e vice-versa); o teste test_page_uses_the_same_names compara os dois resultados.
 * Chamada por: reviewLabel, runName, weightLabel, renderTraining, renderModels, fileRow etc.
 */
const NAMING = (() => {
  // STAGES: nome da etapa pelo prefixo do id da execução (ex.: pilot_train_<data>_<hash>).
  // EXTENSAO: etapa nova de execução: acrescente "prefixo: 'Nome'" aqui e em naming.py.
  const STAGES = { pilot_train: 'Treino Piloto', train: 'Treino de Produção', tune: 'Ajuste de Hiperparâmetros', export: 'Exportação', export_demo: 'Exportação de Demonstração', predict_demo: 'Predição de Demonstração', autolabel_eval: 'Avaliação da Marcação Automática', calibrator: 'Calibrador', feedback_validation: 'Validação de Revisão', review_validation: 'Validação de Revisão', review_opening_fix: 'Correção de Abertura' };
  // RUN_ID: id de execução = <etapa>_<AAAAMMDDTHHMMSSZ>_<8 hex>. Grupos: 1 etapa, 2 carimbo UTC, 3 hash curto.
  const RUN_ID = /^([a-z0-9_]+?)_(\d{8}T\d{6}Z)_([0-9a-f]{8})$/;
  // PLACES: nome do local pela parte da pasta antes da versão (ex.: ceasa_v8 → "CEASA Pavilhão A").
  // EXTENSAO: local novo: acrescente aqui e em naming.py.
  const PLACES = { ceasa: 'CEASA Pavilhão A', ceasa_lado_a: 'CEASA Lado A', ceasa_ladoa: 'CEASA Lado A', internet: 'Fotos da Internet' };
  // WORDS: tradução de palavras soltas do nome da pasta para a forma exibida.
  // EXTENSAO: palavra nova: acrescente aqui e em naming.py (mesma chave e mesmo texto).
  const WORDS = { revisao: 'Revisão', rev: 'Revisão', prerevisao: 'Pré-revisão', sugestoes: 'Sugestões', humana: 'Humana', corrigida: 'Corrigida', taxonomia: 'Taxonomia', parcial: 'Parcial', piloto: 'Piloto', propostas: 'Propostas', busca: 'Busca', aberta: 'Aberta', modelo: 'Modelo', ladoa: 'Lado A', galpao: 'Galpão', pavilhao: 'Pavilhão', inspecao: 'Inspeção', ceasa: 'CEASA', claude: 'Claude', base: 'Base', demo: 'Demonstração', evidencias: 'Evidências', laudo: 'Laudo' };
  // PHRASES: sufixos inteiros que viram uma expressão pronta (testados antes de traduzir palavra a palavra).
  const PHRASES = { humana: 'Revisão Humana', corrigida: 'Revisão Corrigida', sugestoes_modelo: 'Sugestões do Modelo' };
  // CONNECTORS: preposições/artigos que ficam minúsculos quando não são a primeira palavra.
  const CONNECTORS = new Set(['de', 'da', 'do', 'das', 'dos', 'e', 'a', 'o', 'em']);
  // Utilitários internos: has (chave própria do objeto), shortHash (8 primeiros caracteres em maiúsculas),
  // isHash (8+ caracteres hexadecimais) e isVersion (v8, v12, v3b...).
  const has = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
  const shortHash = value => String(value).slice(0, 8).toUpperCase();
  const isHash = token => token.length >= 8 && /^[0-9a-f]+$/.test(token);
  const isVersion = token => /^v\d+[a-z]?$/.test(token);
  /**
   * Traduz uma palavra (token) do nome técnico para a forma exibida, testando as regras em ordem.
   * Parâmetros: token (string minúscula, ex.: "rev2"); first (boolean: é a primeira palavra?).
   * Retorna: string legível (ex.: "Revisão 2", "Semente 42", "Lado A").
   * ATENCAO: a ORDEM das regras importa e é a mesma de naming.py.
   */
  const word = (token, first) => {
    // 1. palavra conhecida; 2. versão (v8) fica como está; 3. s<num> = semente do sorteio; 4. número puro.
    if (has(WORDS, token)) return WORDS[token];
    if (isVersion(token)) return token;
    if (/^s\d+$/.test(token)) return `Semente ${token.slice(1)}`;
    if (/^\d+$/.test(token)) return token;
    // 5. palavra conhecida colada a um número (ex.: rev2 → "Revisão 2").
    const match = /^([a-z]+?)(\d+)$/.exec(token);
    if (match && has(WORDS, match[1])) return `${WORDS[match[1]]} ${match[2]}`;
    // 6. conector minúsculo (fora da 1ª posição); 7. sigla de até 2 letras em maiúsculas; 8. só a inicial maiúscula.
    if (CONNECTORS.has(token) && !first) return token;
    if (token.length <= 2 && /^[a-z]+$/.test(token)) return token.toUpperCase();
    return token.slice(0, 1).toUpperCase() + token.slice(1)
  };
  /**
   * humanize: transforma um identificador (slug) em texto: separa por "_", "-" ou espaço, encurta hashes
   * e traduz cada palavra com word(). Parâmetros: slug (string). Retorna: string ("" para vazio).
   */
  const humanize = slug => String(slug || '').trim().toLowerCase().split(/[_\-\s]+/).filter(Boolean).map((token, index) => isHash(token) ? shortHash(token) : word(token, index === 0)).join(' ');
  /**
   * Divide o nome de pasta de uma revisão em partes: local, versão, resto e hash final.
   * Parâmetros: name (string, ex.: "ceasa_v8_humana_1a2b3c4d").
   * Retorna: [local legível, "v8", ["humana"], "1a2b3c4d"]; sem versão (ou versão na 1ª posição):
   * [null, null, tokens, null].
   * Chamada por: review e dataset.
   */
  const splitReview = name => {
    const tokens = String(name).toLowerCase().split('_').filter(Boolean);
    const position = tokens.findIndex(isVersion);
    if (position <= 0) return [null, null, tokens, null];
    const key = tokens.slice(0, position).join('_');
    const rest = tokens.slice(position + 1);
    const digest = rest.length && isHash(rest[rest.length - 1]) ? rest.pop() : null;
    return [has(PLACES, key) ? PLACES[key] : humanize(key), tokens[position], rest, digest]
  };
  /**
   * Nome legível de uma inspeção/revisão. Parâmetros: name (pasta em data/reviews), display (nome
   * personalizado gravado no registro; quando preenchido, vence). Retorna: ex. "CEASA Pavilhão A v8 · Revisão Humana".
   */
  const review = (name, display) => {
    if (display && String(display).trim()) return String(display).trim();
    if (!name) return '';
    const [place, version, rest, digest] = splitReview(name);
    if (version == null) return humanize(name);
    const qualifier = PHRASES[rest.join('_')] || rest.map((token, index) => word(token, index === 0)).join(' ');
    const tail = [qualifier, digest ? shortHash(digest) : ''].filter(Boolean).join(' ');
    return tail ? `${place} ${version} · ${tail}` : `${place} ${version}`
  };
  /**
   * Nome legível de uma execução de treino. Parâmetros: id (pasta em runs/), custom (nome personalizado).
   * Retorna: "<Etapa> <HASH8>" (ex.: "Treino Piloto 56038EB9") ou o id humanizado se não seguir RUN_ID.
   */
  const run = (id, custom) => {
    if (custom && String(custom).trim()) return String(custom).trim();
    const match = RUN_ID.exec(String(id || ''));
    if (!match) return humanize(id || '');
    return `${STAGES[match[1]] || humanize(match[1])} ${shortHash(match[3])}`
  };
  /**
   * Nome legível de um arquivo de pesos (.pt). Parâmetros: path (caminho com / ou \).
   * Retorna: "<execução> · melhor época/última época" quando o caminho passa por uma pasta de execução,
   * "Modelo Base YOLO..." para pesos oficiais (yolo26l.pt), senão o nome do arquivo humanizado.
   */
  const weights = path => {
    const parts = String(path || '').replace(/\\/g, '/').split('/');
    const owner = parts.find(part => RUN_ID.test(part));
    const file = parts[parts.length - 1] || '';
    const stem = file.replace(/\.[^.]*$/, '');
    if (owner) return `${run(owner)} · ${{ best: 'melhor época', last: 'última época' }[stem] || file}`;
    // Pesos oficiais: yolo + versão (com ou sem "v") + tamanho opcional n/s/m/l/x.
    const family = /^yolo(v?\d+)([nsmlx]?)$/i.exec(stem);
    if (family) return `Modelo Base YOLO${family[1].toUpperCase()}${family[2].toLowerCase()}`;
    return stem ? humanize(stem) : ''
  };
  // day: "20261004" → "04/10/2026"; outro formato → null.
  const day = stamp => /^\d{8}$/.test(stamp) ? `${stamp.slice(6, 8)}/${stamp.slice(4, 6)}/${stamp.slice(0, 4)}` : null;
  /**
   * Nome legível do arquivo de comparação/avaliação em runs/ (compare_<...>.json ou eval_por_classe_<...>.json).
   * Parâmetros: file (nome ou caminho). Retorna: ex. "Comparação v8 · 04/10/2026" ou "Comparação 1A2B3C4D × 5E6F7A8B".
   */
  const comparison = file => {
    let stem = String(file || '').split('/').pop();
    if (stem.endsWith('.json')) stem = stem.slice(0, -5);
    let tokens = stem.split('_').filter(Boolean);
    if (!tokens.length || !['compare', 'eval'].includes(tokens[0])) return humanize(stem);
    const kind = tokens[0] === 'compare' ? 'Comparação' : 'Avaliação por Classe';
    // Remove o prefixo (compare_ / eval_por_classe_ / eval_) para ficar só com o assunto e a data.
    tokens = tokens[0] === 'compare' ? tokens.slice(1) : tokens[1] === 'por' && tokens[2] === 'classe' ? tokens.slice(3) : tokens.slice(1);
    // Último token com 8 dígitos é a data; hashes viram "A × B"; senão as palavras (sem "vs").
    const date = tokens.length && day(tokens[tokens.length - 1]) ? day(tokens.pop()) : null;
    const hashes = tokens.filter(isHash).map(shortHash);
    const subject = hashes.length ? hashes.join(' × ') : (tokens.includes('vs') ? [] : tokens).map((token, index) => word(token, index === 0)).join(' ');
    return [`${kind} ${subject}`.trim(), date].filter(Boolean).join(' · ')
  };
  /**
   * Nome legível de um dataset (pasta em data/pilot). Parâmetros: name (ex.: "ceasa_v8_piloto_s42").
   * Retorna: ex. "Dataset Piloto CEASA Pavilhão A v8 · Semente 42".
   */
  const dataset = name => {
    const [place, version, rest, digest] = splitReview(name);
    if (version == null) return `Dataset ${humanize(name)}`;
    const seed = rest.find(token => /^s\d+$/.test(token)) || null;
    const words = rest.filter(token => token !== 'piloto' && token !== seed);
    const kind = rest.includes('piloto') ? 'Dataset Piloto' : 'Dataset';
    const qualifier = words.map((token, index) => word(token, index === 0)).join(' ');
    return [`${kind} ${place} ${version}`, qualifier, seed ? `Semente ${seed.slice(1)}` : '', digest ? shortHash(digest) : ''].filter(Boolean).join(' · ')
  };
  /**
   * Nome e data de um arquivo gerado (laudo ou pacote de evidências): <AAAAMMDDTHHMMSSZ>_laudo_<...>.
   * Parâmetros: file (nome do arquivo/pasta). Retorna: [nome legível, data ISO em UTC ou null].
   */
  const generated = file => {
    const match = /^(\d{8}T\d{6}Z)_(laudo|evidencias)_/.exec(String(file || ''));
    if (!match) return [humanize(String(file).replace(/\.[^.]*$/, '')), null];
    const s = match[1];
    // Remonta o carimbo compacto como ISO 8601 em UTC (+00:00) para fmtDate converter ao fuso local.
    const iso = `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}T${s.slice(9, 11)}:${s.slice(11, 13)}:${s.slice(13, 15)}+00:00`;
    return [match[2] === 'laudo' ? 'Laudo Técnico' : 'Pacote de Evidências', iso]
  };
  // EXTENSAO: tipo novo de nome: crie a função acima, exporte aqui e crie a equivalente em naming.py.
  return { shortHash, humanize, review, run, weights, comparison, dataset, generated }
})();
/* ======================= fim dos nomes legíveis ======================= */
/* ======================= Utilitários de DOM, mensagens e rascunho ======================= */
// Elementos do index.html usados aqui: message, toast, toastTitle, toastText, reviewer. Nenhuma rota da API.
/**
 * Cria um elemento HTML com propriedades e filhos (substitui um framework de interface).
 * Parâmetros: tag (ex.: 'div'); props (objeto: 'class' vira className, 'style' vira atributo,
 * on* vira manipulador de evento, o resto vira atributo); children (nós ou textos; null é ignorado,
 * listas são achatadas). Retorna: o elemento criado.
 * Chamada por: praticamente todas as funções render*.
 */
const el = (tag, props = {}, ...children) => {
  const node = document.createElement(tag);
  Object.entries(props).forEach(([key, value]) => {
    // Despacho por tipo de propriedade.
    if (key === 'class') node.className = value;
    else if (key === 'style') node.setAttribute('style', value);
    else if (key.startsWith('on')) node[key] = value;
    else node.setAttribute(key, value)
  });
  children.flat().forEach(child => child != null && node.append(child));
  return node
};
/**
 * Cria um ícone SVG que referencia um <symbol> do sprite do index.html (ex.: 'i-video', 'i-report').
 * Parâmetros: name (id do símbolo). Retorna: elemento <svg>.
 */
const icon = name => { const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); const use = document.createElementNS('http://www.w3.org/2000/svg', 'use'); use.setAttribute('href', '#' + name); svg.append(use); return svg };

/**
 * Mostra um texto na barra de mensagens da revisão (#message).
 * Parâmetros: text (string); error (boolean: aplica o estilo de erro). Retorna: nada.
 */
function message(text, error = false) { $('message').textContent = text; $('message').classList.toggle('error', error) }
/**
 * Mostra o aviso flutuante (#toast) por 3,6 s. Parâmetros: title, text (strings). Retorna: nada.
 * O temporizador fica em toast.timer para que um aviso novo reinicie a contagem.
 */
function toast(title, text) {
  $('toastTitle').textContent = title; $('toastText').textContent = text; $('toast').classList.add('show');
  clearTimeout(toast.timer); toast.timer = setTimeout(() => $('toast').classList.remove('show'), 3600)
}

// Guardar o rascunho no navegador não aprova dados nem atualiza pesos do modelo.
/**
 * Guarda o rascunho da foto aberta e o identificador do revisor no localStorage.
 * Parâmetros: nenhum. Retorna: nada. Erro de armazenamento (cota cheia, modo privado) vira mensagem.
 * Chamada por: openPhoto, commitBox, deleteSelected, invalidateDecision, saveDecision, campo reviewer.
 * ATENCAO: o rascunho nunca sai do navegador sozinho; só vira dado quando o revisor salva o arquivo.
 */
function persist() {
  // 1. copia o rascunho em edição para a coleção de rascunhos (por sha da foto).
  if (HAS_IMAGES) state.drafts[item().image_sha256] = clone(active);
  state.reviewer = $('reviewer').value;
  // 2. grava tudo no navegador.
  try { localStorage.setItem(storageKey, JSON.stringify({ images: state.drafts, reviewer: state.reviewer })) }
  catch { message('O navegador não conseguiu guardar o rascunho. Salve um arquivo de revisão.', true) }
}

/* ======================= Navegação entre seções ======================= */
// Elementos do index.html: sidebar, scrim, menuToggle, viewTitle, viewSubtitle, as <section id="view-...">,
// os botões .nav-item[data-view] e os botões [data-goto]. Nenhuma rota da API.
// EXTENSAO: seção nova na plataforma = (1) <section class="view" id="view-<nome>"> no index.html;
// (2) botão .nav-item com data-view="<nome>" na barra lateral; (3) entrada aqui em VIEWS;
// (4) if (view === '<nome>') render<Nome>() em showView; (5) bloco próprio com render<Nome>().
// VIEWS: chave da seção → [título do cabeçalho, subtítulo]. A chave é também o hash da URL (#review).
const VIEWS = {
  home: ['Início', 'Visão geral da inspeção aberta e do espaço de trabalho'],
  inspections: ['Inspeções', 'Campanhas, edificações e revisões registradas'],
  upload: ['Importar mídia', 'Fotos e vídeos do voo viram uma nova inspeção'],
  review: ['Revisão', 'Confira, corrija e decida foto a foto'],
  reports: ['Laudos e downloads', 'Laudo técnico (ABNT), evidências e arquivos da inspeção'],
  training: ['Treinamento', 'Ciclo de fine-tuning com as correções humanas'],
  monitor: ['Monitoramento', 'Treino em andamento e tarefas do servidor local'],
  models: ['Modelos', 'Comparação, cobertura e critério de adoção'],
  classes: ['Classes', 'Categorias de ocorrência: nomes, cores e classes propostas'],
  // Assistente IA: três chaves, uma só <section id="view-assistant"> (ver AI_VIEWS no bloco Assistente IA).
  'ai-settings': ['Assistente IA', 'Configuração da IA: provedor, chave e modelo'],
  'ai-usage': ['Assistente IA', 'Uso e limites: consumo local e limites do provedor'],
  'ai-chat': ['Assistente IA', 'Chat sobre as inspeções, classes, treinos e comparações']
};
// VIEW_SECTION: chaves de VIEWS que dividem a mesma <section> (submenus). As demais usam view-<chave>.
// EXTENSAO: seção nova com submenus = mapeie cada chave para o nome da section aqui.
const VIEW_SECTION = { 'ai-settings': 'assistant', 'ai-usage': 'assistant', 'ai-chat': 'assistant' };
/**
 * Abre uma seção: marca a <section> ativa, o item do menu, o título, a URL (#view) e chama o render dela.
 * Parâmetros: view (chave de VIEWS; desconhecida vira 'home'). Retorna: nada.
 * Chamada por: cliques em [data-view]/[data-goto], hashchange, inicialização, applyTheme, refreshWorkspace.
 */
function showView(view) {
  if (!VIEWS[view]) view = 'home';
  state.view = view;
  // 1. mostra só a seção escolhida e marca o item do menu (aria-current) para leitores de tela.
  document.querySelectorAll('.view').forEach(section => section.classList.toggle('active', section.id === 'view-' + (VIEW_SECTION[view] || view)));
  document.querySelectorAll('.nav-item[data-view]').forEach(button => button.setAttribute('aria-current', button.dataset.view === view ? 'page' : 'false'));
  // Item principal do Assistente IA fica marcado em qualquer um dos três submenus.
  document.querySelectorAll('.nav-item[data-group="assistant"]').forEach(button => button.setAttribute('aria-current', VIEW_SECTION[view] === 'assistant' ? 'page' : 'false'));
  // 2. título, subtítulo e título da aba do navegador.
  $('viewTitle').textContent = VIEWS[view][0];
  $('viewSubtitle').textContent = DATA.version ? `${VIEWS[view][1]} · ${reviewLabel(DATA.version, DATA.display_name)}` : VIEWS[view][1];
  document.title = [VIEWS[view][0], DATA.version ? reviewLabel(DATA.version, DATA.display_name) : '', 'DroneCamp Vision'].filter(Boolean).join(' · ');
  // 3. lembra a seção (sessionStorage) e atualiza o hash sem criar entrada no histórico.
  try { sessionStorage.setItem('dronecamp-view', view) } catch {}
  if (location.hash !== '#' + view) history.replaceState(null, '', '#' + view);
  closeMenu();
  // 4. desenha a seção aberta. EXTENSAO: seção nova = mais um if aqui.
  if (view === 'home') renderHome();
  if (view === 'inspections') renderInspections();
  if (view === 'upload') renderUpload();
  if (view === 'review') paint();
  if (view === 'reports') renderReports();
  if (view === 'training') renderTraining();
  if (view === 'monitor') renderMonitor();
  if (view === 'models') renderModels();
  if (view === 'classes') renderClasses();
  if (VIEW_SECTION[view] === 'assistant') renderAssistant(view);
  window.scrollTo({ top: 0 })
}
// ENTRADA: cliques no menu lateral ([data-view]) e nos atalhos internos ([data-goto]).
document.querySelectorAll('[data-view]').forEach(node => node.onclick = event => { event.preventDefault(); showView(node.dataset.view) });
document.querySelectorAll('[data-goto]').forEach(button => button.onclick = () => showView(button.dataset.goto));
/** Fecha o menu lateral (telas estreitas). Parâmetros: nenhum. Retorna: nada. */
function closeMenu() { $('sidebar').classList.remove('open'); $('scrim').classList.add('hidden'); $('menuToggle').setAttribute('aria-expanded', 'false') }
// ENTRADA: botão de menu (telas estreitas) abre/fecha a barra lateral e o fundo escurecido.
$('menuToggle').onclick = () => { const open = !$('sidebar').classList.contains('open'); $('sidebar').classList.toggle('open', open); $('scrim').classList.toggle('hidden', !open); $('menuToggle').setAttribute('aria-expanded', String(open)) };
$('scrim').onclick = closeMenu;

/* ======================= Revisão (lógica preservada) ======================= */
// Elementos do index.html: queue, filter, stats, queueProgress, navSuggestions, canvas, viewer, photoTitle,
// photoMeta, proposalStatus, proposalNotes, auditNotes, historyNotice, taxonomyNotice, originalLink,
// classId, x1, y1, x2, y2, applyBox, newBox, deleteBox, boxes, boxCount, drawToggle, showBoxes,
// showSuggestions, showSuggestionsLabel, confFilter, confFilterLabel, confValue, suggestionPanel,
// suggestionMeta, suggestions, suggestionCount, decision, notes, confirmed, saveDecision, export,
// previous, next, reviewer, reviewPicker, zoomIn, zoomOut, zoomFit, zoomValue, message.
// Rotas da API: POST /api/feedback (cópia do arquivo de revisão, só com servidor).
// ATENCAO: regras de negócio desta seção: sugestão da IA nunca vira caixa sem clique humano; toda
// edição volta a foto para "pending" e desmarca a confirmação; positiva exige caixa, negativa não aceita caixa.
/**
 * Decisão humana registrada de uma foto, se houver.
 * Parâmetros: photo (objeto de DATA.images). Retorna: status ('positive', 'negative', ...) ou null se pendente.
 * Chamada por: renderCycle.
 */
function decidedStatus(photo) {
  const decision = state.drafts[photo.image_sha256];
  return decision && decision.status !== 'pending' ? decision.status : null
}
// currentBoxes: caixas valendo para a foto (rascunho, se existir; senão as do registro).
const currentBoxes = photo => { const decision = state.drafts[photo.image_sha256]; return decision ? decision.boxes : photo.boxes };

/**
 * Redesenha a fila de fotos da coluna esquerda, aplicando o filtro escolhido, e atualiza contadores.
 * Parâmetros: nenhum (lê #filter e state). Retorna: nada.
 * Chamada por: openPhoto, commitBox, saveDecision, dismissSuggestion, invalidateDecision, #filter.
 */
function queue() {
  if (!HAS_IMAGES) return;
  const filter = $('filter').value;
  $('queue').replaceChildren();
  let decided = 0;
  // Um botão por foto; o laço também conta as decisões já tomadas.
  DATA.images.forEach((photo, index) => {
    const decision = state.drafts[photo.image_sha256];
    if (decision && decision.status !== 'pending') decided++;
    // Filtro: 'suggested' = com sugestão pendente; 'pending' = sem decisão minha; 'all'; senão compara com
    // o status PROPOSTO da foto. EXTENSAO: filtro novo = <option> em #filter (index.html) + condição aqui.
    if (filter === 'suggested') {
      if (!pendingSuggestionsFor(photo, decision || { boxes: photo.boxes }).length) return
    } else if (filter !== 'all' && (filter === 'pending' ? decision && decision.status !== 'pending' : photo.status !== filter)) return;
    const button = el('button', { type: 'button', class: 'q-item', 'aria-current': String(index === state.index) });
    const thumb = el('img', { src: photo.image_url, alt: '', loading: 'lazy' });
    const title = el('div', { class: 'q-title' }, photo.pages && photo.pages.length ? `${index + 1}. Páginas ${photo.pages.join(', ')}` : `${index + 1}. ${photo.filename}`);
    const pendingAi = pendingSuggestionsFor(photo, decision || { boxes: photo.boxes }).length;
    // Linha de baixo: minha decisão; ou aviso de migração de taxonomia; ou status proposto + sugestões pendentes.
    const small = el('small', {}, decision && decision.status !== 'pending'
      ? `Sua decisão: ${decisionNames[decision.status]}`
      : photo.previous_human_approved
        ? 'Correções anteriores preservadas · conferir novas categorias'
        : names[photo.status] + (pendingAi ? ` · ${pendingAi} sugest${pendingAi > 1 ? 'ões' : 'ão'} IA` : ''));
    const dot = el('span', { class: 'q-state ' + (decision && decision.status !== 'pending' ? decision.status : ''), 'aria-hidden': 'true' });
    button.append(thumb, el('div', {}, title, small), dot);
    button.onclick = () => openPhoto(index);
    $('queue').append(button)
  });
  // Contadores do topo da fila, barra de progresso e selo de sugestões no menu.
  $('stats').textContent = `${DATA.images.length} fotos · ${decided} decisões`;
  $('queueProgress').style.width = `${DATA.images.length ? decided / DATA.images.length * 100 : 0}%`;
  const pendingAi = DATA.images.reduce((sum, photo) => sum + pendingSuggestionsFor(photo, state.drafts[photo.image_sha256] || { boxes: photo.boxes }).length, 0);
  $('navSuggestions').textContent = pendingAi;
  $('navSuggestions').classList.toggle('hidden', !pendingAi)
}

/** Limpa a seleção de caixa e os campos de coordenadas. Parâmetros: nenhum. Retorna: nada. */
function resetSelection() {
  state.selected = -1;
  ['x1', 'y1', 'x2', 'y2'].forEach(id => $(id).value = '');
  $('applyBox').textContent = 'Adicionar caixa';
  $('deleteBox').disabled = true
}

/**
 * Seleciona uma caixa para edição: preenche categoria e coordenadas e troca o botão para "Atualizar".
 * Parâmetros: index (posição em active.boxes). Retorna: nada. Chamada por: clique na lista de caixas.
 */
function selectBox(index) {
  const box = active.boxes[index];
  state.selected = index;
  $('classId').value = box.class_id;
  ['x1', 'y1', 'x2', 'y2'].forEach((id, position) => $(id).value = box.bbox_xyxy[position]);
  $('applyBox').textContent = 'Atualizar caixa';
  $('deleteBox').disabled = false;
  renderBoxes()
}

/**
 * Redesenha a lista de caixas (#boxes), o contador, o painel de sugestões e o canvas.
 * Parâmetros: nenhum. Retorna: nada. Chamada por: quase toda edição na revisão.
 */
function renderBoxes() {
  const list = $('boxes');
  list.replaceChildren();
  // Um botão por caixa, na cor da categoria, com as coordenadas arredondadas.
  active.boxes.forEach((box, index) => {
    const category = classOf(box.class_id);
    const button = el('button', { type: 'button', class: 'box-item' + (index === state.selected ? ' selected' : '') });
    button.style.setProperty('--box-color', category.color);
    button.textContent = `${index + 1}. ${category.label}` + (box.proposed_new_class ? ` · nova classe sugerida: ${box.proposed_new_class}` : '');
    button.append(el('div', { class: 'box-caption' }, box.bbox_xyxy.map(Math.round).join(', ') + ' pixels'));
    button.onclick = () => selectBox(index);
    list.append(button)
  });
  if (!active.boxes.length) list.textContent = 'Nenhuma caixa registrada nesta foto.';
  $('boxCount').textContent = `${active.boxes.length} caixa${active.boxes.length === 1 ? '' : 's'}`;
  // Atualiza também as sugestões (dependem das caixas) e o desenho.
  renderSuggestions();
  paint()
}

// Sugestões do modelo são apenas candidatas: aceitar uma copia a caixa para
// a revisão e exige nova confirmação da foto inteira.
// sameBox: mesma classe e mesmas coordenadas arredondadas (usado para saber se uma sugestão já foi aceita).
const sameBox = (a, b) => a.class_id === b.class_id && a.bbox_xyxy.every((value, i) => Math.round(value) === Math.round(b.bbox_xyxy[i]));
/**
 * IoU (interseção sobre união) entre duas caixas [x1, y1, x2, y2] em pixels.
 * Parâmetros: a, b (listas de 4 números). Retorna: número de 0 a 1 (0 se a união for vazia).
 */
function boxIou(a, b) {
  const inter = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0
}
// Aceitar sobre uma caixa existente do mesmo objeto duplicaria o alvo de treino.
// coveringBox: índice da caixa existente da mesma classe com IoU >= DATA.duplicate_iou (configs/project.yaml,
// pilot.duplicate_iou); -1 se não houver ou se o limite não veio na página.
const coveringBox = suggestion => DATA.duplicate_iou
  ? active.boxes.findIndex(box => box.class_id === suggestion.class_id && boxIou(box.bbox_xyxy, suggestion.bbox_xyxy) >= DATA.duplicate_iou)
  : -1;
// Limite de exibição: esconder sugestões fracas não as descarta nem as aceita.
// confThreshold: limite de exibição escolhido no controle #confFilter (0..1); 0 quando não há sugestões.
const confThreshold = () => DATA.suggestions ? Number($('confFilter').value) / 100 : 0;
/**
 * Sugestões da IA ainda pendentes numa foto: nem aceitas (caixa igual) nem descartadas.
 * Parâmetros: photo (DATA.images[i]); decision (rascunho com boxes e dismissed_suggestions).
 * Retorna: lista de { suggestion, index } (index = posição original, usado no rótulo "IA n").
 */
function pendingSuggestionsFor(photo, decision) {
  return (photo.model_suggestions || [])
    .map((suggestion, index) => ({ suggestion, index }))
    .filter(({ suggestion }) => !(decision.boxes || []).some(box => sameBox(box, suggestion)) && !(decision.dismissed_suggestions || []).includes(suggestion.id))
}
/**
 * Sugestões pendentes da foto aberta acima do limite de confiança da tela.
 * Parâmetros: nenhum. Retorna: lista de { suggestion, index }. O -1e-9 evita erro de arredondamento.
 */
function pendingSuggestions() {
  if (!HAS_IMAGES) return [];
  return pendingSuggestionsFor(item(), active).filter(({ suggestion }) => suggestion.confidence >= confThreshold() - 1e-9)
}

/**
 * Aceita uma sugestão: copia a caixa (coordenadas arredondadas) para o rascunho ou substitui a caixa
 * que cobre o mesmo objeto, e invalida a decisão da foto.
 * Parâmetros: suggestion (objeto de photo.model_suggestions). Retorna: nada.
 * Chamada por: botão Aceitar/Substituir e tecla "a".
 * ATENCAO: a IA nunca aprova: aceitar só cria caixa; a foto continua pendente até a confirmação humana.
 */
function acceptSuggestion(suggestion) {
  const covered = coveringBox(suggestion);
  const box = { class_id: suggestion.class_id, bbox_xyxy: suggestion.bbox_xyxy.map(Math.round), note: `Sugestão do modelo aceita (confiança ${Math.round(suggestion.confidence * 100)}%); conferida na revisão local.` };
  // Substituir troca só a geometria e a nota; campos como proposed_new_class ficam.
  if (covered >= 0) active.boxes[covered] = { ...active.boxes[covered], ...box }; else active.boxes.push(box);
  invalidateDecision(covered >= 0
    ? `Caixa ${covered + 1} substituída pela geometria da sugestão; nenhuma duplicata criada. Confira a foto inteira antes de confirmar.`
    : 'Sugestão aceita como caixa. Ajuste se necessário e confira a foto inteira antes de confirmar.')
}

/**
 * Descarta uma sugestão nesta revisão (guarda o id em dismissed_suggestions do rascunho).
 * Parâmetros: suggestion. Retorna: nada. Chamada por: botão Descartar e tecla "x".
 */
function dismissSuggestion(suggestion) {
  active.dismissed_suggestions = [...(active.dismissed_suggestions || []), suggestion.id];
  persist(); renderBoxes(); queue();
  message('Sugestão descartada nesta revisão.')
}

/**
 * Desenha o painel de sugestões da IA da foto aberta (lista com confiança, aceitar e descartar).
 * Parâmetros: nenhum. Retorna: nada. Chamada por: renderBoxes.
 */
function renderSuggestions() {
  // Sem DATA.suggestions (nenhum modelo aplicado) os controles de sugestão ficam escondidos.
  const meta = DATA.suggestions;
  $('suggestionPanel').classList.toggle('hidden', !meta);
  $('showSuggestionsLabel').classList.toggle('hidden', !meta);
  $('confFilterLabel').classList.toggle('hidden', !meta);
  if (!meta || !HAS_IMAGES) return;
  $('suggestionMeta').textContent = `${meta.model_label} · confiança mínima ${Math.round(meta.conf * 100)}%. ${meta.warning}`;
  const list = $('suggestions');
  list.replaceChildren();
  const pending = pendingSuggestions();
  const hidden = pendingSuggestionsFor(item(), active).length - pending.length;
  $('suggestionCount').textContent = `${pending.length} pendente${pending.length === 1 ? '' : 's'}`;
  // Uma linha por sugestão pendente.
  pending.forEach(({ suggestion, index }) => {
    const category = classOf(suggestion.class_id);
    const covered = coveringBox(suggestion);
    const row = el('div', { class: 'suggestion' });
    row.style.setProperty('--box-color', category.color);
    const percent = Math.round(suggestion.confidence * 100);
    const head = el('div', { class: 's-head' }, el('span', { class: 's-index' }, `IA ${index + 1}`), el('span', { class: 's-dot', 'aria-hidden': 'true' }), el('span', { class: 's-class' }, category.label));
    const metaRow = el('div', { class: 's-meta' }, el('span', {}, 'Confiança ', el('strong', {}, `${percent}%`)));
    if (covered >= 0) metaRow.append(el('span', { class: 's-overlap' }, `Mesmo objeto da caixa ${covered + 1}`));
    // acceptance_probability vem do calibrador; source/prompt vêm de suggest --zero-shot (busca aberta, YOLOE).
    if (suggestion.acceptance_probability != null) metaRow.append(el('span', {}, `Aceitação estimada ${Math.round(suggestion.acceptance_probability * 100)}%`));
    if (suggestion.prompt) metaRow.append(el('span', {}, `Busca aberta: “${suggestion.prompt}”`));
    // EXTENSAO: ação nova por sugestão = outro botão aqui, acrescentado na div .actions abaixo.
    const accept = el('button', { type: 'button', class: 'small ai' }, covered >= 0 ? `Substituir caixa ${covered + 1}` : 'Aceitar');
    accept.onclick = () => acceptSuggestion(suggestion);
    const dismiss = el('button', { type: 'button', class: 'small' }, 'Descartar');
    dismiss.onclick = () => dismissSuggestion(suggestion);
    const bar = el('div', { class: 'conf', 'aria-hidden': 'true' }, el('i', { style: `width:${percent}%` }));
    row.append(head, metaRow, bar, el('div', { class: 'actions' }, accept, dismiss));
    list.append(row)
  });
  // Lista vazia: explica o motivo (abaixo do limite, todas tratadas ou nenhuma sugestão).
  if (!pending.length) list.textContent = (item().model_suggestions || []).length
    ? (hidden > 0 ? `${hidden} sugest${hidden > 1 ? 'ões abaixo' : 'ão abaixo'} do limite de confiança escolhido.` : 'Todas as sugestões desta foto foram aceitas ou descartadas.')
    : 'O modelo não sugeriu caixas nesta foto acima da confiança mínima.';
  else if (hidden > 0) list.append(el('p', { class: 'muted small-text' }, `${hidden} oculta${hidden > 1 ? 's' : ''} abaixo de ${Math.round(confThreshold() * 100)}%.`))
}

/**
 * Volta a foto aberta para "pendente", desmarca a confirmação, guarda e redesenha.
 * Parâmetros: text (mensagem a mostrar). Retorna: nada. Chamada por: acceptSuggestion.
 */
function invalidateDecision(text) {
  active.status = 'pending'; active.confirmed_complete = false;
  $('decision').value = 'pending'; $('confirmed').checked = false;
  persist(); resetSelection(); renderBoxes(); queue(); message(text)
}

/**
 * Desenha a foto no canvas com as caixas, as sugestões (tracejadas) e o retângulo em desenho.
 * Parâmetros: preview (opcional, [x1, y1, x2, y2] do arraste em andamento). Retorna: nada.
 * O canvas tem o tamanho real da foto; a escala na tela é feita por CSS e pelo zoom.
 */
function paint(preview) {
  if (!HAS_IMAGES) return;
  const canvas = $('canvas'), ctx = canvas.getContext('2d');
  // A imagem ainda não carregou: o onload de openPhoto chamará renderBoxes → paint de novo.
  if (!state.img.complete || !state.img.naturalWidth) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.img, 0, 0);
  // 1. caixas do rascunho (borda sólida, rótulo "n: categoria"; a selecionada fica mais grossa).
  if ($('showBoxes').checked) {
    active.boxes.forEach((box, index) => {
      const [x1, y1, x2, y2] = box.bbox_xyxy;
      const category = classOf(box.class_id);
      ctx.strokeStyle = category.color; ctx.lineWidth = index === state.selected ? 5 : 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.font = `600 ${Math.max(14, Math.round(canvas.width / 70))}px system-ui, sans-serif`;
      const text = `${index + 1}: ${category.label}`;
      const top = Math.max(0, y1 - 24);
      ctx.fillStyle = category.color; ctx.fillRect(x1, top, ctx.measureText(text).width + 12, 24);
      // Texto escuro: com a paleta atual passa de 4,5:1 nas 13 cores de classe.
      ctx.fillStyle = '#08111F'; ctx.fillText(text, x1 + 5, top + 18)
    })
  }
  // 2. sugestões pendentes acima do limite (borda tracejada, placa no canto de baixo).
  if (DATA.suggestions && $('showSuggestions').checked) {
    pendingSuggestions().forEach(({ suggestion, index }) => {
      const [x1, y1, x2, y2] = suggestion.bbox_xyxy;
      const category = classOf(suggestion.class_id);
      ctx.setLineDash([10, 6]); ctx.strokeStyle = category.color; ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); ctx.setLineDash([]);
      ctx.font = `${Math.max(13, Math.round(canvas.width / 80))}px system-ui, sans-serif`;
      const text = `IA ${index + 1}: ${category.label} ${Math.round(suggestion.confidence * 100)}%`;
      const bottom = Math.min(canvas.height - 22, y2);
      // Placa escura opaca com texto na cor da classe: difere do rótulo sólido das caixas.
      ctx.fillStyle = '#08111F'; ctx.fillRect(x1, bottom, ctx.measureText(text).width + 12, 22);
      ctx.fillStyle = category.color; ctx.fillText(text, x1 + 5, bottom + 16)
    })
  }
  // 3. retângulo provisório enquanto o revisor arrasta.
  if (preview) {
    ctx.setLineDash([8, 5]); ctx.strokeStyle = '#2457F5'; ctx.lineWidth = 3;
    ctx.strokeRect(preview[0], preview[1], preview[2] - preview[0], preview[3] - preview[1]); ctx.setLineDash([])
  }
}

/**
 * Abre a foto de posição index: guarda o rascunho atual, carrega o rascunho da nova foto e preenche
 * todos os painéis (título, metadados, status proposto, notas, avisos, decisão, canvas).
 * Parâmetros: index (posição em DATA.images). Retorna: nada.
 * Chamada por: fila, botões anterior/próxima, setas do teclado e inicialização (foto 0).
 */
function openPhoto(index) {
  if (!HAS_IMAGES) return;
  // 1. salva a foto anterior e troca o rascunho ativo.
  persist();
  state.index = index;
  active = clone(draft());
  resetSelection();
  state.drawing = false;
  $('drawToggle').setAttribute('aria-pressed', 'false');
  $('viewer').classList.remove('drawing');
  // 2. cabeçalho e metadados (arquivo, páginas, dimensões, quadro de vídeo).
  const photo = item();
  $('photoTitle').textContent = `Foto ${index + 1} de ${DATA.images.length}`;
  $('photoMeta').textContent = [photo.filename, photo.pages && photo.pages.length ? `páginas ${photo.pages.join(', ')}` : null, `${photo.width} × ${photo.height}`, photo.ingest && photo.ingest.kind === 'video_frame' ? `quadro de vídeo em ${photo.ingest.frame_time_s}s` : null].filter(Boolean).join(' · ');
  // 3. status proposto (chip colorido) e notas da proposta/segunda leitura.
  $('proposalStatus').textContent = names[photo.status];
  $('proposalStatus').className = 'chip ' + ({ positive: 'bad', negative: 'ok', ambiguous: 'warn', excluded: 'neutral' }[photo.status] || '');
  $('proposalNotes').textContent = photo.notes;
  $('auditNotes').textContent = [photo.human_notes, photo.second_review_notes, photo.additional_candidate_note, photo.review_focus_note].filter(Boolean).join('\n\n') || 'Segunda leitura ainda não registrada.';
  // A migração conserva a correção antiga e exige apenas a conferência faltante.
  const reviewedClasses = photo.classes_reviewed_before_migration || [];
  const pendingClasses = DATA.classes.filter(category => !reviewedClasses.includes(category.id)).map(category => category.label);
  $('historyNotice').classList.toggle('hidden', !photo.previous_human_approved);
  $('historyNotice').textContent = photo.previous_human_approved
    ? `Suas caixas e categorias da revisão anterior foram preservadas. Confira a foto inteira para as categorias acrescentadas: ${pendingClasses.join('; ')}. Depois registre uma nova confirmação.` : '';
  $('taxonomyNotice').classList.toggle('hidden', !photo.taxonomy_recheck_pending);
  $('taxonomyNotice').textContent = `Categorias que ainda precisam de conferência integral nesta foto: ${pendingClasses.join('; ')}.`;
  $('originalLink').href = photo.image_url;
  // 4. campos de decisão, tamanho do canvas (= pixels da foto) e limites dos campos de coordenada.
  $('decision').value = active.status; $('notes').value = active.notes; $('confirmed').checked = active.confirmed_complete;
  $('canvas').width = photo.width; $('canvas').height = photo.height;
  resetZoom();
  ['x1', 'x2'].forEach(id => $(id).max = photo.width); ['y1', 'y2'].forEach(id => $(id).max = photo.height);
  // 5. carrega a imagem; ao terminar, redesenha as caixas.
  state.img = new Image();
  state.img.onload = () => renderBoxes();
  state.img.src = photo.image_url;
  renderBoxes(); queue();
  // 6. rola a fila até a foto aberta.
  const current = $('queue').querySelector('[aria-current="true"]');
  if (current) current.scrollIntoView({ block: 'nearest' });
  message('Ajuste as propostas e registre sua decisão depois de conferir a foto inteira.')
}

/**
 * Confere se uma caixa [x1, y1, x2, y2] tem área e cabe na foto aberta.
 * Parâmetros: coordinates (4 números em pixels). Retorna: boolean.
 */
function validBox(coordinates) {
  return coordinates.every(Number.isFinite) && coordinates[0] >= 0 && coordinates[1] >= 0
    && coordinates[2] <= item().width && coordinates[3] <= item().height && coordinates[2] > coordinates[0] && coordinates[3] > coordinates[1]
}

// Toda edição invalida a confirmação anterior: a foto deve ser revista novamente.
/**
 * Grava uma caixa (nova ou a selecionada) a partir de coordenadas, validando área e categoria.
 * Parâmetros: coordinates ([x1, y1, x2, y2], podem vir com decimais). Retorna: nada.
 * Chamada por: botão Adicionar/Atualizar caixa e fim do arraste no canvas.
 * Erros: mostra mensagem (não lança) para caixa inválida, sem categoria ou categoria inativa.
 */
function commitBox(coordinates) {
  coordinates = coordinates.map(Math.round);
  if (!validBox(coordinates)) { message('A caixa precisa ter área e ficar dentro da foto.', true); return }
  const classId = $('classId').value === '' ? NaN : Number($('classId').value);
  if (!Number.isInteger(classId)) { message('Escolha a categoria da caixa.', true); return }
  // Classe inativa (seção Classes): caixas já existentes mantêm a categoria; caixas novas não a recebem.
  if (state.selected < 0 && classOf(classId).active === false) { message('Categoria inativa: reative-a na seção Classes para usá-la em caixas novas.', true); return }
  // Caixa válida: insere ou substitui, e a foto volta a pendente (exige nova confirmação).
  const box = { class_id: classId, bbox_xyxy: coordinates, note: 'Caixa conferida/editada na revisão local.' };
  if (state.selected < 0) active.boxes.push(box); else active.boxes[state.selected] = box;
  active.confirmed_complete = false; active.status = 'pending';
  $('confirmed').checked = false; $('decision').value = 'pending';
  persist(); resetSelection(); renderBoxes(); queue();
  message('Caixa registrada. Confira a foto inteira antes de confirmar a decisão.')
}
// ENTRADA: botões e campos do editor de caixas (Adicionar/Atualizar, Nova caixa, notas da foto).
$('applyBox').onclick = () => commitBox(['x1', 'y1', 'x2', 'y2'].map(id => $(id).value === '' ? NaN : Number($(id).value)));
$('newBox').onclick = () => { resetSelection(); renderBoxes() };
$('notes').oninput = () => { active.notes = $('notes').value; persist() };

// Excluir uma caixa também exige nova decisão humana sobre a foto inteira.
/** Exclui a caixa selecionada e invalida a decisão. Parâmetros: nenhum. Retorna: nada. */
function deleteSelected() {
  if (state.selected < 0) return;
  active.boxes.splice(state.selected, 1);
  active.status = 'pending'; active.confirmed_complete = false;
  $('decision').value = 'pending'; $('confirmed').checked = false;
  persist(); resetSelection(); renderBoxes(); queue();
  message('Caixa excluída. Confira a foto inteira antes de confirmar a decisão.')
}
$('deleteBox').onclick = deleteSelected;

/**
 * Liga/desliga o modo desenho (arrastar sobre a foto cria caixa).
 * Parâmetros: nenhum. Retorna: nada. Chamada por: botão #drawToggle e tecla "b".
 */
function toggleDrawing() {
  if (!HAS_IMAGES) return;
  state.drawing = !state.drawing;
  resetSelection();
  $('drawToggle').setAttribute('aria-pressed', state.drawing);
  $('viewer').classList.toggle('drawing', state.drawing);
  renderBoxes();
  message(state.drawing ? 'Arraste sobre a foto para criar uma caixa da categoria selecionada.' : 'Desenho desativado.')
}
$('drawToggle').onclick = toggleDrawing;

// Converta o toque/clique para pixels da foto original, sem depender do tamanho na tela.
/**
 * Posição do ponteiro em pixels da foto, limitada às bordas.
 * Parâmetros: event (PointerEvent). Retorna: [x, y].
 * Regra de três: (posição na tela - borda) × largura real / largura exibida.
 */
function point(event) {
  const rect = $('canvas').getBoundingClientRect();
  return [Math.min(item().width, Math.max(0, (event.clientX - rect.left) * item().width / rect.width)), Math.min(item().height, Math.max(0, (event.clientY - rect.top) * item().height / rect.height))]
}
/* Zoom e movimento da foto. A escala é só visual (transform no canvas): point()
   lê o retângulo já transformado, então as caixas continuam em pixels da foto. */
/**
 * Aplica zoom.scale/x/y ao canvas (CSS transform) e atualiza o rótulo de porcentagem.
 * Abaixo de 100% volta ao enquadramento original. Parâmetros: nenhum. Retorna: nada.
 */
function applyZoom() {
  if (zoom.scale <= 1) { zoom.scale = 1; zoom.x = 0; zoom.y = 0 }
  $('canvas').style.transform = zoom.scale === 1 ? '' : `translate(${zoom.x}px, ${zoom.y}px) scale(${zoom.scale})`;
  $('viewer').classList.toggle('zoomed', zoom.scale > 1);
  $('zoomValue').textContent = `${Math.round(zoom.scale * 100)}%`;
}
// Mantém fixo o ponto da foto sob (clientX, clientY); sem coordenadas, usa o centro do visor.
/**
 * Muda o zoom para scale (limitado entre 1 e 12) mantendo fixo o ponto sob o cursor.
 * Parâmetros: scale (número); clientX, clientY (px de tela; opcionais = centro do visor). Retorna: nada.
 */
function zoomTo(scale, clientX, clientY) {
  const rect = $('canvas').getBoundingClientRect();
  if (clientX == null) { const box = $('viewer').getBoundingClientRect(); clientX = box.left + box.width / 2; clientY = box.top + box.height / 2 }
  const next = Math.min(12, Math.max(1, scale));
  // (u, v): ponto sob o cursor em coordenadas sem escala; recalcula o deslocamento para ele não sair do lugar.
  const u = (clientX - rect.left) / zoom.scale, v = (clientY - rect.top) / zoom.scale;
  zoom.x += clientX - u * next - rect.left; zoom.y += clientY - v * next - rect.top;
  zoom.scale = next; applyZoom();
}
/** Volta o zoom a 100%. Parâmetros: nenhum. Retorna: nada. */
function resetZoom() { zoom.scale = 1; applyZoom() }
// ENTRADA: roda do mouse sobre o visor = zoom (deltaMode 1 = linhas, mais sensível que pixels).
$('viewer').addEventListener('wheel', event => {
  if (!HAS_IMAGES) return;
  event.preventDefault();
  zoomTo(zoom.scale * Math.exp(-event.deltaY * (event.deltaMode === 1 ? 0.05 : 0.0015)), event.clientX, event.clientY);
}, { passive: false });
// ENTRADA: botões de zoom (+, -, ajustar).
$('zoomIn').onclick = () => zoomTo(zoom.scale * 1.25);
$('zoomOut').onclick = () => zoomTo(zoom.scale / 1.25);
$('zoomFit').onclick = resetZoom;
// Arrastar move a foto ampliada (botão do meio sempre; botão esquerdo fora do modo desenho).
$('viewer').addEventListener('pointerdown', event => {
  if (zoom.scale <= 1 || !(event.button === 1 || (event.button === 0 && !state.drawing))) return;
  event.preventDefault(); event.stopPropagation();
  zoom.pan = { id: event.pointerId, x: event.clientX - zoom.x, y: event.clientY - zoom.y };
  $('viewer').setPointerCapture(event.pointerId); $('viewer').classList.add('panning');
}, true);
$('viewer').addEventListener('pointermove', event => {
  if (!zoom.pan || event.pointerId !== zoom.pan.id) return;
  zoom.x = event.clientX - zoom.pan.x; zoom.y = event.clientY - zoom.pan.y; applyZoom();
});
// endPan: fim do arraste da foto ampliada (soltar ou cancelar o ponteiro).
const endPan = event => { if (!zoom.pan || event.pointerId !== zoom.pan.id) return; zoom.pan = null; $('viewer').classList.remove('panning') };
$('viewer').addEventListener('pointerup', endPan); $('viewer').addEventListener('pointercancel', endPan);
$('viewer').addEventListener('auxclick', event => { if (event.button === 1) event.preventDefault() });

// ENTRADA: desenho de caixa no canvas: pressionar marca o início, mover mostra a prévia, soltar grava.
$('canvas').onpointerdown = event => { if (!state.drawing) return; event.preventDefault(); state.start = point(event); $('canvas').setPointerCapture(event.pointerId) };
$('canvas').onpointermove = event => { if (!state.start) return; const end = point(event); paint([Math.min(state.start[0], end[0]), Math.min(state.start[1], end[1]), Math.max(state.start[0], end[0]), Math.max(state.start[1], end[1])]) };
$('canvas').onpointerup = event => { if (!state.start) return; const end = point(event), start = state.start; state.start = null; commitBox([Math.min(start[0], end[0]), Math.min(start[1], end[1]), Math.max(start[0], end[0]), Math.max(start[1], end[1])]) };
$('canvas').onpointercancel = () => { state.start = null; paint() };
// ENTRADA: opções de exibição (mostrar caixas, mostrar sugestões, limite de confiança).
$('showBoxes').onchange = () => paint();
$('showSuggestions').onchange = () => paint();
$('confFilter').oninput = () => { $('confValue').textContent = `${$('confFilter').value}%`; renderBoxes() };

// A aprovação visual exige confirmação da revisão integral e caixas coerentes.
// ENTRADA: botão "Registrar decisão" da foto.
// EXTENSAO: regra nova de validação da decisão = mais um if com message(..., true); return, antes da gravação.
$('saveDecision').onclick = () => {
  if (!HAS_IMAGES) return;
  const status = $('decision').value, confirmed = $('confirmed').checked;
  if (['positive', 'negative'].includes(status) && !confirmed) { message('Confirme que revisou a foto inteira antes de aprovar a anotação.', true); return }
  if (status === 'positive' && !active.boxes.length) { message('Uma foto com ocorrências precisa de pelo menos uma caixa.', true); return }
  if (status === 'negative' && active.boxes.length) { message('Uma foto negativa não pode ter caixas. Corrija a anotação ou escolha outra decisão.', true); return }
  // Validações passaram: grava no rascunho (ainda não é o arquivo de revisão).
  active.status = status; active.notes = $('notes').value; active.confirmed_complete = confirmed;
  persist(); queue();
  message(status === 'pending' ? 'Rascunho guardado; decisão ainda pendente.' : 'Sua decisão foi registrada no rascunho local. Salve o arquivo de revisão para importá-la.');
  if (status !== 'pending') toast('Decisão registrada', `Foto ${state.index + 1}: ${decisionNames[status]}. Fica no rascunho local até salvar o arquivo.`)
};

// Abrir uma foto já guarda um rascunho igual ao registro. Esse rascunho "intocado" não é decisão humana:
// não vai ao servidor (senão uma foto positiva do registro viraria pendente no laudo) nem ao CSV.
// untouched: o rascunho é igual ao registro (nenhuma edição real).
const untouched = (photo, value) => !value || (value.status === 'pending' && !value.confirmed_complete && !(value.dismissed_suggestions || []).length
  && (value.notes || '') === (photo.human_notes || '') && JSON.stringify(value.boxes) === JSON.stringify(photo.boxes));
// touchedDraft: rascunho da foto só se foi editado; senão null.
const touchedDraft = photo => { const value = state.drafts[photo.image_sha256]; return untouched(photo, value) ? null : value };
// Decisões do rascunho local (as editadas, inclusive pendentes com caixas mexidas), para o servidor desenhar.
const allDrafts = () => DATA.images.flatMap(photo => { const value = touchedDraft(photo); return value ? [{ image_sha256: photo.image_sha256, ...value }] : [] });
// decidedDrafts: só as fotos com decisão (vão ao arquivo de revisão).
const decidedDrafts = () => allDrafts().filter(value => value.status !== 'pending');
/**
 * Monta o arquivo de revisão (o JSON que import-review lê).
 * Parâmetros: nenhum (lê #reviewer e os rascunhos). Retorna: objeto
 * { schema_version, registry_sha256, reviewer_id, exported_at, images: [decisões] } ou null se faltar
 * identificador ou decisão (abre a revisão e mostra a mensagem).
 * Chamada por: botão Exportar (#export) e #downloadReview.
 * ATENCAO: registry_sha256 liga o arquivo ao registro exibido; import-review recusa hash diferente.
 * EXTENSAO: campo novo no arquivo de revisão = acrescente no objeto do return e trate em review_data.import_human_feedback (Python).
 */
function reviewFile() {
  const reviewer = $('reviewer').value.trim();
  if (!reviewer) { showView('review'); message('Preencha seu identificador de revisão para salvar o arquivo.', true); $('reviewer').focus(); return null }
  const decisions = decidedDrafts();
  if (!decisions.length) { showView('review'); message('Registre pelo menos uma decisão antes de salvar.', true); return null }
  return { schema_version: 1, registry_sha256: DATA.registry_sha256, reviewer_id: reviewer, exported_at: new Date().toISOString(), images: decisions }
}
/**
 * Faz o navegador baixar um Blob com o nome dado (link temporário).
 * Parâmetros: blob (Blob), name (nome do arquivo). Retorna: nada.
 */
function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a'); link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

// O hash vincula o feedback ao registro exibido; exportar não executa treinamento.
// ENTRADA: botão "Salvar arquivo de revisão" (#export): baixa o JSON e, com servidor, envia uma cópia.
$('export').onclick = async () => {
  const feedback = reviewFile();
  if (!feedback) return;
  downloadBlob(new Blob([JSON.stringify(feedback, null, 2)], { type: 'application/json' }), 'dronecamp-revisao-' + DATA.version + '.json');
  let where = '';
  // Pelo servidor local, uma cópia também fica na pasta da revisão (feedback_inbox/).
  if (SERVER) {
    try { const result = await api('/api/feedback', feedback); where = ` Cópia salva em ${result.saved}.` }
    catch (error) { where = ` Cópia no projeto não foi salva: ${error.message}` }
  }
  message(`${feedback.images.length} decisões incluídas no arquivo de revisão.${where}`);
  toast('Arquivo de revisão salvo', `${feedback.images.length} decisões. Importe com import-review (etapa 2 do ciclo).${where}`)
};
// ENTRADA: navegação entre fotos, filtro da fila e campo do identificador do revisor.
$('previous').onclick = () => openPhoto(Math.max(0, state.index - 1));
$('next').onclick = () => openPhoto(Math.min(DATA.images.length - 1, state.index + 1));
$('filter').onchange = queue;
$('reviewer').value = state.reviewer;
$('reviewer').oninput = persist;

// Seletor de inspeções: cada revisão abre na sua própria página (/r/<pasta>/), com fotos,
// decisões salvas e arquivo exportado separados; trocar não mistura registros.
// Nome legível (NAMING.review); a pasta crua fica na dica (title) para rastreabilidade.
/**
 * Nome legível de uma inspeção. Parâmetros: name (pasta), display (nome personalizado). Retorna: string.
 */
function reviewLabel(name, display) { return NAMING.review(name, display) }
/**
 * Preenche o seletor de inspeções (#reviewPicker) e liga a troca de página.
 * Parâmetros: items (lista de { name, display_name, href, images }); current (pasta aberta). Retorna: nada.
 * Chamada por: inicialização (meta dronecamp-reviews) e refreshWorkspace (/api/workspace).
 */
function fillReviewPicker(items, current) {
  const picker = $('reviewPicker');
  picker.replaceChildren();
  (items || []).forEach(value => {
    const count = value.images == null ? '' : ` · ${value.images} fotos`;
    const option = el('option', { value: value.href }, `${reviewLabel(value.name, value.display_name)}${count}`);
    option.title = value.name;
    if (value.name === current) option.selected = true;
    picker.append(option)
  });
  // Sem lista: uma opção só com a inspeção desta página; com menos de 2 opções o seletor fica desabilitado.
  if (!picker.options.length) picker.append(el('option', { value: '' }, DATA.version ? reviewLabel(DATA.version, DATA.display_name) : 'Nenhuma inspeção'));
  picker.disabled = picker.options.length < 2;
  picker.onchange = event => { if (event.target.value) location.href = event.target.value + location.hash }
}
// Preenchimento inicial a partir da meta dronecamp-reviews (null quando aberta do disco).
fillReviewPicker(REVIEWS && REVIEWS.items, REVIEWS && REVIEWS.current);

// Atalhos de teclado só na revisão e fora de campos de texto.
// ENTRADA: atalhos de teclado da revisão (setas, b, a, x, Delete, +, -, 0).
// EXTENSAO: atalho novo = mais um "else if (event.key === ...)" antes do "else return".
document.addEventListener('keydown', event => {
  if (state.view !== 'review' || !HAS_IMAGES || event.ctrlKey || event.metaKey || event.altKey) return;
  const tag = (event.target.tagName || '').toLowerCase();
  if (['input', 'select', 'textarea'].includes(tag)) return;
  const first = pendingSuggestions()[0];
  if (event.key === 'ArrowLeft') openPhoto(Math.max(0, state.index - 1));
  else if (event.key === 'ArrowRight') openPhoto(Math.min(DATA.images.length - 1, state.index + 1));
  else if (event.key.toLowerCase() === 'b') toggleDrawing();
  else if (event.key.toLowerCase() === 'a' && first) acceptSuggestion(first.suggestion);
  else if (event.key.toLowerCase() === 'x' && first) dismissSuggestion(first.suggestion);
  else if (event.key === 'Delete') deleteSelected();
  else if (event.key === '+' || event.key === '=') zoomTo(zoom.scale * 1.25);
  else if (event.key === '-' || event.key === '_') zoomTo(zoom.scale / 1.25);
  else if (event.key === '0') resetZoom();
  else return;
  event.preventDefault()
});

/* ======================= Início ======================= */
// Elementos do index.html: kpiPhotos, kpiPhotosHint, kpiDecisions, kpiDecisionsBar, kpiBoxes, kpiSuggestions,
// kpiSuggestionsHint, kpiSparse, homeContext, qaReview, classBars, triageBars, modelChip, modelSummary,
// lastRunSummary, homeInspections. Nenhuma rota própria: usa DATA e OPS (atualizado pelo bloco Servidor).
/**
 * Cria uma linha de gráfico de barras horizontal (parte humana sólida + parte IA).
 * Parâmetros: color (CSS), label (nome), human (valor humano), ai (valor IA), max (valor de 100%),
 * countText (texto à direita). Retorna: elemento .bar-row. Chamada por: renderHome e renderModels.
 */
function barRow(color, label, human, ai, max, countText) {
  const row = el('div', { class: 'bar-row' });
  row.style.setProperty('--c', color);
  const bar = el('div', { class: 'bar', role: 'img', 'aria-label': `${label}: ${countText}` },
    el('i', { class: 'human', style: `width:${max ? human / max * 100 : 0}%` }), el('i', { class: 'ai', style: `width:${max ? ai / max * 100 : 0}%` }));
  row.append(el('div', { class: 'name' }, el('span', { class: 'swatch' }), el('span', { title: label }, label)), bar, el('div', { class: 'count' }, countText));
  return row
}

/**
 * Totais da inspeção a partir das caixas valendo (rascunho ou registro).
 * Parâmetros: nenhum. Retorna: { totals: {classe: {boxes, ai}}, boxes, pendingAi, decided, triage }.
 * Chamada por: renderHome e inspectionItems.
 * ATENCAO: triage.pending soma as fotos SEM decisão minha mesmo quando também contam no status
 * proposto (positive etc.); por isso a soma das barras de triagem pode passar do total de fotos.
 */
function reviewTotals() {
  const totals = Object.fromEntries(DATA.classes.map(category => [category.id, { boxes: 0, ai: 0 }]));
  let boxes = 0, pendingAi = 0, decided = 0;
  const triage = { positive: 0, negative: 0, ambiguous: 0, excluded: 0, pending: 0 };
  DATA.images.forEach(photo => {
    const decision = state.drafts[photo.image_sha256];
    // Triagem: sua decisão quando existe; senão o status proposto no registro. "pending" conta as fotos sem decisão sua.
    const status = decision && decision.status !== 'pending' ? decision.status : (triage[photo.status] == null ? 'pending' : photo.status);
    triage[status]++;
    if (decision && decision.status !== 'pending') decided++; else triage.pending += status === 'pending' ? 0 : 1;
    // Caixas por classe (só classes conhecidas entram em totals) e sugestões pendentes por classe.
    currentBoxes(photo).forEach(box => { if (totals[box.class_id]) totals[box.class_id].boxes++; boxes++ });
    pendingSuggestionsFor(photo, decision || { boxes: photo.boxes }).forEach(({ suggestion }) => { if (totals[suggestion.class_id]) totals[suggestion.class_id].ai++; pendingAi++ })
  });
  return { totals, boxes, pendingAi, decided, triage }
}

/**
 * Desenha a seção Início: indicadores, barras por classe e de triagem, modelo das sugestões,
 * último treino e as três primeiras inspeções. Parâmetros: nenhum. Retorna: nada.
 * Chamada por: showView('home') e refreshOperations.
 */
function renderHome() {
  const { totals, boxes, pendingAi, decided, triage } = reviewTotals();
  // sparse: classes com menos de 5 caixas (pouca amostra para treinar). O limite 5 se repete em
  // renderCoverage e na coloração das barras logo abaixo; mude nos três lugares juntos.
  const sparse = DATA.classes.filter(category => totals[category.id].boxes < 5);
  // 1. indicadores (cartões do topo).
  $('kpiPhotos').textContent = DATA.images.length;
  const groups = [...new Set(DATA.images.map(photo => photo.building_group).filter(Boolean))];
  $('kpiPhotosHint').textContent = !HAS_IMAGES ? 'nenhuma inspeção aberta' : buildingCount(groups.length);
  $('kpiDecisions').textContent = `${decided}/${DATA.images.length}`;
  $('kpiDecisionsBar').style.width = `${DATA.images.length ? decided / DATA.images.length * 100 : 0}%`;
  $('kpiBoxes').textContent = boxes;
  $('kpiSuggestions').textContent = pendingAi;
  $('kpiSuggestionsHint').textContent = DATA.suggestions ? `limite do modelo ${Math.round(DATA.suggestions.conf * 100)}%` : 'nenhum modelo aplicado';
  $('kpiSparse').textContent = sparse.length;
  $('homeContext').textContent = DATA.version ? `Inspeção aberta: ${reviewLabel(DATA.version, DATA.display_name)}` : '';
  $('homeContext').title = DATA.version || '';
  $('qaReview').textContent = HAS_IMAGES ? `${DATA.images.length - decided} fotos sem decisão · ${pendingAi} sugestões` : 'Nenhuma inspeção aberta';
  // 2. barras por classe (vermelho = 0 caixas, amarelo = menos de 5).
  const max = Math.max(1, ...Object.values(totals).map(value => value.boxes + value.ai));
  $('classBars').replaceChildren(...DATA.classes.map(category => {
    const total = totals[category.id];
    const row = barRow(category.color, category.label, total.boxes, total.ai, max, `${total.boxes}${total.ai ? ' +' + total.ai : ''}`);
    if (total.boxes === 0) row.querySelector('.count').classList.add('cell-bad');
    else if (total.boxes < 5) row.querySelector('.count').classList.add('cell-warn');
    return row
  }));
  // 3. barras de triagem. EXTENSAO: status novo = cor aqui em triageColors (e rótulo em decisionNames).
  const triageColors = { positive: 'var(--bad)', negative: 'var(--ok)', ambiguous: 'var(--warn)', excluded: 'var(--muted)', pending: 'var(--line-strong)' };
  const triageLabels = { ...decisionNames, pending: 'Aguardando sua decisão' };
  $('triageBars').replaceChildren(...Object.keys(triage).map(key => barRow(triageColors[key], triageLabels[key], triage[key], 0, DATA.images.length, String(triage[key]))));
  // Modelo das sugestões: liga o SHA dos pesos à execução que os gerou (via comparações).
  const meta = DATA.suggestions;
  if (!meta) {
    $('modelChip').replaceChildren(el('span', { class: 'chip neutral' }, 'sem sugestões'));
    $('modelSummary').textContent = 'Esta inspeção não recebeu sugestões de um modelo treinado. Use “Gerar sugestões” no Monitoramento (servidor local) ou o comando suggest.'
  } else {
    const run = runForWeights(meta.weights_sha256);
    $('modelChip').replaceChildren(el('span', { class: 'chip ai' }, meta.pilot ? 'piloto · não validado' : 'não validado'));
    $('modelSummary').replaceChildren(
      el('p', {}, `${meta.model_label}. Confiança mínima ${Math.round(meta.conf * 100)}%.`),
      run ? el('p', { title: run.id }, el('b', {}, runName(run)), ` · iniciado em ${fmtDate(run.started_at)}`) : null,
      el('p', { class: 'mono muted small-text', title: meta.weights_sha256 || '' }, `SHA-256 dos pesos ${String(meta.weights_sha256 || '').slice(0, 12)}…`),
      run && run.best_epoch ? el('p', {}, `Melhor mAP50 de validação dessa execução: ${pct(run.best_epoch.map50)} (época ${run.best_epoch.epoch}).`) : null)
  }
  // 4. último treino com curvas (OPS.runs vem do mais novo para o mais antigo).
  const last = OPS.runs.find(run => run.curves.length);
  if (!last) $('lastRunSummary').textContent = 'Nenhum treino encontrado em runs/.';
  else {
    const chart = el('div', { class: 'chart' });
    $('lastRunSummary').replaceChildren(
      el('p', {}, el('span', { class: 'chip ' + stateClass(last.state) }, stateLabel(last.state)), ' ', el('b', { title: last.id }, runName(last)), ' ', el('span', { class: 'muted small-text' }, fmtDate(last.started_at))),
      el('p', { class: 'muted' }, `${last.epochs_done}/${last.epochs_planned || '?'} épocas · melhor mAP50 ${pct(last.best_epoch && last.best_epoch.map50)}`), chart);
    lineChart(chart, last.curves, [{ key: 'map50', label: 'mAP50', color: '#16A34A' }, { key: 'recall', label: 'Recall', color: '#2457F5' }], { height: 120, percent: true })
  }
  // 5. cartões das inspeções (compactos: só 3).
  $('homeInspections').replaceChildren(inspectionCards(true))
}

/* ======================= Inspeções ======================= */
// Elementos do index.html: inspectionList, inspectionsHint, navInspections (contador no menu).
// Rotas da API: nenhuma direta; usa WORKSPACE, carregado por refreshWorkspace (GET /api/workspace).
/**
 * Lista de inspeções para os cartões.
 * Parâmetros: nenhum. Retorna: lista de { name, display_name, images, decided, boxes, suggestions_pending,
 * current, href, buildings, page_ready, ... }. Ordem de preferência: WORKSPACE do servidor; meta
 * dronecamp-reviews (servidor antigo); só a inspeção desta página.
 */
// Lista do servidor (/api/workspace) ou, aberta do disco, só a inspeção desta página.
function inspectionItems() {
  if (WORKSPACE && Array.isArray(WORKSPACE.reviews) && WORKSPACE.reviews.length) return WORKSPACE.reviews;
  if (!DATA.version) return [];
  const { decided, pendingAi, boxes } = reviewTotals();
  const mine = { name: DATA.version, display_name: DATA.display_name, version: DATA.version, images: DATA.images.length, decided, boxes, suggestions_pending: pendingAi, current: true, href: '', buildings: [...new Set(DATA.images.map(photo => photo.building_group).filter(Boolean))], page_ready: true, offline: true };
  // Servidor antigo (sem /api/workspace) com várias revisões: a lista vem da meta dronecamp-reviews.
  if (REVIEWS && Array.isArray(REVIEWS.items) && REVIEWS.items.length) {
    return REVIEWS.items.map(item => item.name === REVIEWS.current ? { ...item, ...mine, href: item.href } : { ...item, current: false, page_ready: item.images != null })
  }
  return [mine]
}
/**
 * Monta os cartões das inspeções (progresso, contagens e botões).
 * Parâmetros: compact (true = só as 3 primeiras, usado no Início). Retorna: elemento com os cartões.
 * Chamada por: renderHome e renderInspections.
 */
function inspectionCards(compact) {
  const items = inspectionItems();
  $('navInspections').textContent = items.length || '';
  if (!items.length) return el('div', { class: 'empty' }, SERVER ? 'Nenhuma inspeção em data/reviews. Importe fotos ou vídeos para começar.' : 'Nenhuma inspeção aberta.');
  const list = el('div', { class: 'inspection-list' });
  // Um cartão por inspeção: cabeçalho, edificações, números, barra de progresso e ações.
  (compact ? items.slice(0, 3) : items).forEach(review => {
    const progress = review.images ? Math.round((review.decided || 0) / review.images * 100) : 0;
    const card = el('div', { class: 'inspection' + (review.current ? ' current' : '') });
    card.append(
      el('div', { class: 'head' }, el('b', { title: review.name }, reviewLabel(review.name, review.display_name)), el('span', { class: 'chip ' + (review.current ? 'primary' : review.page_ready === false ? 'warn' : 'neutral') }, review.current ? 'aberta' : review.page_ready === false ? 'página não gerada' : review.scope_label || 'revisão')),
      // Só a contagem: o código interno da edificação (ex.: ceasa_pavilhao_a) não é mostrado.
      el('div', { class: 'meta' }, el('span', {}, buildingCount((review.buildings || []).length)), review.updated_at ? el('span', {}, `atualizada ${fmtDate(review.updated_at)}`) : null),
      el('div', { class: 'nums' }, el('div', {}, el('b', {}, review.images ?? '—'), 'fotos'), el('div', {}, el('b', {}, `${progress}%`), 'decididas'), el('div', {}, el('b', { class: review.suggestions_pending ? 'cell-ai' : '' }, review.suggestions_pending ?? 0), 'sugestões IA')),
      el('div', { class: 'progress ok' }, el('i', { style: `width:${progress}%` })));
    // Ações: inspeção aberta = continuar/laudos; outra pronta = abrir; outra sem página = instrução.
    const actions = el('div', { class: 'actions' });
    if (review.current) actions.append(el('button', { type: 'button', class: 'small primary', onclick: () => showView('review') }, 'Continuar revisão'), el('button', { type: 'button', class: 'small', onclick: () => showView('reports') }, 'Laudos e downloads'));
    else if (review.href && review.page_ready !== false) actions.append(el('a', { class: 'btn small primary', href: review.href + '#review' }, 'Abrir'));
    else if (review.href) actions.append(el('span', { class: 'muted small-text' }, 'Gere a página com render-review para abrir.'));
    card.append(actions);
    list.append(card)
  });
  return list
}
/** Desenha a seção Inspeções (lista completa). Parâmetros: nenhum. Retorna: nada. */
function renderInspections() {
  $('inspectionsHint').textContent = SERVER ? 'Cada inspeção guarda as fotos originais congeladas por hash, o registro de revisão e o histórico das decisões.' : 'Aberta direto do disco, a página mostra só esta inspeção. Pelo servidor local, todas as inspeções de data/reviews aparecem aqui.';
  // replaceWith troca o nó inteiro, então o novo recebe de volta o id inspectionList.
  $('inspectionList').replaceWith(Object.assign(inspectionCards(false), { id: 'inspectionList' }))
}

/* ======================= Importar mídia ======================= */
// Elementos do index.html: dropzone, fileInput, fileList, fileSummary, startUpload, cancelUpload,
// uploadName, uploadGroup, uploadEvery, uploadMaxFrames, uploadNotes, uploadWeights, uploadOffline,
// uploadProgress, uploadStatus, uploadResult, reviewer.
// Rotas da API: POST /api/uploads (cria lote), PUT /api/uploads/<lote>/files (um arquivo por vez),
// POST /api/uploads/<lote>/ingest (inicia a importação), GET /api/uploads/<lote> (andamento).
// upload: estado do envio. files: [{ file, state ('pronto'|'enviando'|'enviado'|'falhou'), sent (0..100) }];
// running: envio em andamento; batch: id do lote no servidor; request/cancelled: para o botão Cancelar.
const upload = { files: [], running: false, batch: null };
// Extensões aceitas. EXTENSAO: formato novo = acrescente aqui E em ingest.VIDEO_EXTENSIONS (vídeo) ou dataset.IMAGE_EXTENSIONS
// (foto), senão a importação no servidor recusa o arquivo.
const IMAGE_EXT = /\.(jpe?g|png|webp|bmp|tiff?)$/i, VIDEO_EXT = /\.(mp4|mov|avi|mkv|m4v|webm)$/i;
/**
 * Acrescenta arquivos escolhidos/soltos à lista, ignorando formatos não aceitos e repetidos (nome+tamanho).
 * Parâmetros: list (FileList ou lista de File). Retorna: nada.
 */
function addFiles(list) {
  [...list].forEach(file => {
    if (!IMAGE_EXT.test(file.name) && !VIDEO_EXT.test(file.name)) return;
    if (upload.files.some(entry => entry.file.name === file.name && entry.file.size === file.size)) return;
    upload.files.push({ file, state: 'pronto', sent: 0 })
  });
  renderFiles()
}
/**
 * Redesenha a lista de arquivos a enviar, o resumo (fotos, vídeos, bytes) e o botão Enviar.
 * Parâmetros: nenhum. Retorna: nada.
 */
function renderFiles() {
  const list = $('fileList');
  list.replaceChildren(...upload.files.map((entry, index) => {
    const video = VIDEO_EXT.test(entry.file.name);
    const row = el('div', { class: 'file' }, icon(video ? 'i-video' : 'i-image'),
      el('div', {}, el('div', { class: 'name', title: entry.file.name }, entry.file.name), el('small', {}, `${fmtBytes(entry.file.size)}${video ? ' · vídeo → quadros' : ''}`), entry.sent ? el('div', { class: 'progress' }, el('i', { style: `width:${entry.sent}%` })) : null),
      upload.running ? el('span', { class: 'state ' + (entry.state === 'enviado' ? 'ok' : entry.state === 'falhou' ? 'bad' : '') }, entry.state) : el('button', { type: 'button', class: 'small', onclick: () => { upload.files.splice(index, 1); renderFiles() } }, 'Remover'));
    return row
  }));
  // Resumo e habilitação do botão (exige servidor, nada em andamento e pelo menos um arquivo).
  const photos = upload.files.filter(entry => IMAGE_EXT.test(entry.file.name)).length, videos = upload.files.length - photos;
  const bytes = upload.files.reduce((sum, entry) => sum + entry.file.size, 0);
  $('fileSummary').textContent = upload.files.length ? `${photos} foto${photos === 1 ? '' : 's'} · ${videos} vídeo${videos === 1 ? '' : 's'} · ${fmtBytes(bytes)}` : '';
  $('startUpload').disabled = !SERVER || upload.running || !upload.files.length
}
/**
 * Prepara a seção Importar mídia: aviso offline, lista de pesos para sugestões e nome padrão.
 * Parâmetros: nenhum. Retorna: nada. Chamada por: showView('upload').
 */
function renderUpload() {
  $('uploadOffline').classList.toggle('hidden', SERVER);
  const weights = OPS.weights.filter(path => path.startsWith('runs/') || path.startsWith('models/'));
  fillSelect($('uploadWeights'), [['', 'Não gerar agora (revisar sem modelo)'], ...weights.map(path => [path, weightLabel(path)])], (OPS.runs.find(value => value.weights && value.weights.best) || { weights: {} }).weights.best || '');
  if (!$('uploadName').value) $('uploadName').value = `Inspeção ${new Date().toLocaleDateString('pt-BR')}`;
  renderFiles()
}
// ENTRADA: área de soltar (clique, Enter/Espaço, escolher arquivos, arrastar e soltar).
$('dropzone').onclick = () => $('fileInput').click();
$('dropzone').onkeydown = event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); $('fileInput').click() } };
$('fileInput').onchange = () => { addFiles($('fileInput').files); $('fileInput').value = '' };
['dragenter', 'dragover'].forEach(type => $('dropzone').addEventListener(type, event => { event.preventDefault(); $('dropzone').classList.add('over') }));
['dragleave', 'drop'].forEach(type => $('dropzone').addEventListener(type, event => { event.preventDefault(); $('dropzone').classList.remove('over') }));
$('dropzone').addEventListener('drop', event => addFiles(event.dataTransfer.files));
// Nome livre ("Galpão B · outubro"): o servidor deriva a pasta (sem acentos) e guarda o nome digitado no registro.

/**
 * Envia um arquivo ao servidor com PUT cru, informando o progresso.
 * Parâmetros: url (rota do lote), file (File), onProgress (função que recebe a fração 0..1).
 * Retorna: Promise com o JSON da resposta. Erros: rejeita com a mensagem do servidor, falha de rede ou cancelamento.
 * ATENCAO: o token vai no cabeçalho X-DroneCamp-Token e o nome em X-File-Name (codificado); o servidor confere os dois.
 */
function putFile(url, file, onProgress) {
  // XMLHttpRequest para ter progresso do envio; o corpo vai cru (sem multipart) direto para o disco.
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('PUT', url);
    request.setRequestHeader('X-DroneCamp-Token', TOKEN);
    request.setRequestHeader('X-File-Name', encodeURIComponent(file.name));
    request.setRequestHeader('Content-Type', 'application/octet-stream');
    request.upload.onprogress = event => { if (event.lengthComputable) onProgress(event.loaded / event.total) };
    request.onload = () => { let value = {}; try { value = JSON.parse(request.responseText) } catch {} request.status < 300 ? resolve(value) : reject(new Error(value.error || `HTTP ${request.status}`)) };
    request.onerror = () => reject(new Error('Falha de rede ao enviar o arquivo.'));
    request.onabort = () => reject(new Error('Envio cancelado.'));
    upload.request = request;
    request.send(file)
  })
}
// ENTRADA: botão "Enviar e importar" (#startUpload). Fluxo em etapas numeradas abaixo.
$('startUpload').onclick = async () => {
  const name = $('uploadName').value.trim(), group = $('uploadGroup').value.trim();
  // 1. validação do formulário.
  if (!name) { $('uploadStatus').textContent = 'Informe o nome da inspeção.'; $('uploadName').focus(); return }
  if (!group) { $('uploadStatus').textContent = 'Informe a edificação ou campanha.'; $('uploadGroup').focus(); return }
  // 2. estado de envio e barra de progresso.
  upload.running = true; upload.cancelled = false;
  $('cancelUpload').classList.remove('hidden'); $('uploadResult').classList.add('hidden'); $('uploadProgress').classList.remove('hidden');
  renderFiles();
  const total = upload.files.reduce((sum, entry) => sum + entry.file.size, 0);
  let done = 0;
  try {
    // 3. cria o lote e envia arquivo por arquivo (progresso total ponderado pelos bytes).
    const { batch } = await api('/api/uploads', { name });
    upload.batch = batch;
    for (const entry of upload.files) {
      if (upload.cancelled) throw new Error('Envio cancelado.');
      entry.state = 'enviando';
      $('uploadStatus').textContent = `Enviando ${entry.file.name}…`;
      await putFile(`/api/uploads/${batch}/files`, entry.file, fraction => { entry.sent = Math.round(fraction * 100); $('uploadProgress').firstElementChild.style.width = `${(done + fraction * entry.file.size) / total * 100}%`; renderFiles() });
      entry.state = 'enviado'; entry.sent = 100; done += entry.file.size; renderFiles()
    }
    // 4. pede a importação (hash, quadros dos vídeos, sugestões opcionais).
    // EXTENSAO: opção nova de importação = campo no index.html + chave neste objeto + leitura em start_ingest (platform_server.py).
    $('uploadStatus').textContent = 'Arquivos enviados. Congelando fotos por hash e extraindo quadros dos vídeos…';
    await api(`/api/uploads/${batch}/ingest`, { name, group, every_seconds: Number($('uploadEvery').value) || 2, max_frames: Number($('uploadMaxFrames').value) || 200, notes: $('uploadNotes').value, uploaded_by: $('reviewer').value.trim() || null, weights: $('uploadWeights').value || null });
    // 5. consulta o andamento a cada 1,5 s até sair de "running".
    let status;
    do {
      await new Promise(resolve => setTimeout(resolve, 1500));
      status = await api(`/api/uploads/${batch}`);
      $('uploadStatus').textContent = status.message || 'Processando…'
    } while (status.state === 'running');
    if (status.state !== 'finished') throw new Error(status.error || 'A importação falhou.');
    // 6. mostra o resultado com links para a nova inspeção.
    const result = status.result;
    $('uploadStatus').textContent = '';
    $('uploadResult').classList.remove('hidden');
    $('uploadResult').replaceChildren(el('div', { class: 'verdict ok' },
      el('b', { title: result.review }, `Inspeção “${reviewLabel(result.review, result.display_name)}” criada com ${result.images} fotos`),
      el('p', { class: 'small-text', style: 'margin:4px 0 8px' }, `${result.photos} fotos enviadas · ${result.video_frames} quadros de ${result.videos} vídeo(s)` + (result.skipped && result.skipped.length ? ` · ${result.skipped.length} repetidas ignoradas` : '') + (result.suggest_job ? ' · gerando sugestões da IA (veja Monitoramento)' : '')),
      el('div', { class: 'row-actions', style: 'margin:0' }, el('a', { class: 'btn primary small', href: result.href + '#review' }, 'Abrir revisão'), el('a', { class: 'btn small', href: result.href + '#monitor' }, 'Acompanhar sugestões'))));
    toast('Inspeção criada', `${result.images} fotos prontas para revisão.`);
    upload.files = [];
    refreshWorkspace()
  // Falha: mostra o motivo e marca o arquivo em envio como falho. Sempre: libera a tela.
  } catch (error) {
    $('uploadStatus').textContent = `Não concluído: ${error.message}`;
    upload.files.forEach(entry => { if (entry.state === 'enviando') entry.state = 'falhou' })
  } finally {
    upload.running = false; $('cancelUpload').classList.add('hidden'); $('uploadProgress').classList.add('hidden'); renderFiles()
  }
};
// ENTRADA: botão Cancelar envio (aborta o PUT em andamento e para o laço).
$('cancelUpload').onclick = () => { upload.cancelled = true; if (upload.request) upload.request.abort() };

/* ======================= Laudos e downloads ======================= */
// Elementos do index.html: reportsOffline, makeReport, makeEvidence, downloadCsv, downloadReview,
// reportTitle, reportClient, reportSite, reportInspector, reportNotes, reportPdf, reportAllImages,
// inputs [data-detail], reportStatus, evidenceSize, evidenceOnly, evidenceSuggestions, evidenceStatus,
// fileHistory, refreshFiles, reviewer.
// Rotas da API: POST /api/report (laudo), POST /api/export/evidence (ZIP), GET /api/reviews/<pasta>/files.
// ATENCAO: laudo e evidências recebem allDrafts() (rascunhos editados), não só o registro salvo.
/**
 * Prepara a seção Laudos: habilita botões conforme servidor/fotos, título padrão, PDF e histórico.
 * Parâmetros: nenhum. Retorna: nada. Chamada por: showView('reports').
 */
function renderReports() {
  $('reportsOffline').classList.toggle('hidden', SERVER);
  $('makeReport').disabled = !SERVER || !HAS_IMAGES;
  $('makeEvidence').disabled = !SERVER || !HAS_IMAGES;
  $('downloadCsv').disabled = !HAS_IMAGES;
  if (!$('reportTitle').value && DATA.version) $('reportTitle').value = 'Laudo técnico de inspeção de cobertura com uso de drone';
  // Servidor sem o extra "reports" (sem gerador de PDF): desmarca e desabilita a opção PDF.
  if (WORKSPACE && WORKSPACE.capabilities && WORKSPACE.capabilities.pdf === false) { $('reportPdf').checked = false; $('reportPdf').disabled = true; $('reportPdf').nextElementSibling.textContent = 'PDF indisponível neste PC: instale o extra "reports" (pip install -e .[reports]). O HTML é gerado sempre.' }
  renderFileHistory()
}
/**
 * Linha do histórico de arquivos gerados (ícone, tipo, data e botão Baixar).
 * Parâmetros: file ({ name, kind, href, bytes, modified }). Retorna: elemento .download.
 */
function fileRow(file) {
  // EXTENSAO: tipo novo de arquivo no histórico = acrescente "kind: [ícone, rótulo]" aqui.
  const kinds = { pdf: ['i-report', 'Laudo em PDF'], html: ['i-report', 'Laudo em HTML'], zip: ['i-zip', 'Pacote de evidências'], json: ['i-save', 'Dados do laudo'], csv: ['i-report', 'Planilha'] };
  const [ico, label] = kinds[file.kind] || ['i-download', file.kind];
  // Pastas e ZIPs são <carimbo UTC>_laudo_<inspeção>: a data vira texto legível, o caminho fica na dica.
  const generatedOn = NAMING.generated(String(file.name || '').split('/')[1] || file.name)[1];
  return el('div', { class: 'download' }, el('div', { class: 'ico' }, icon(ico)),
    el('div', {}, el('b', { title: file.name }, label), el('small', {}, `${generatedOn ? 'Gerado em ' + fmtDate(generatedOn) : fmtDate(file.modified)} · ${fmtBytes(file.bytes)}`)),
    el('a', { class: 'btn small', href: file.href, target: '_blank', rel: 'noopener', download: file.kind === 'zip' || file.kind === 'pdf' ? '' : null }, 'Baixar'))
}

// Dados técnicos opcionais do laudo (seção "Dados técnicos"): só os campos preenchidos vão ao servidor.
/**
 * Coleta os campos técnicos opcionais do laudo (inputs com data-detail no index.html).
 * Parâmetros: nenhum. Retorna: objeto { chave: valor } só com os campos preenchidos.
 * EXTENSAO: campo novo = input data-detail="<chave>" no index.html + a mesma chave em laudo_abnt.DETAIL_FIELDS.
 */
function reportDetails() {
  const details = {};
  document.querySelectorAll('[data-detail]').forEach((input) => { const value = input.value.trim(); if (value) details[input.dataset.detail] = value });
  return details;
}

/**
 * Atualiza o histórico de laudos e pacotes (do servidor, se houver; senão do instantâneo OPS.review_files).
 * Parâmetros: nenhum. Retorna: Promise<void>. Erro do servidor é ignorado (fica a lista embutida).
 */
async function renderFileHistory() {
  const history = $('fileHistory');
  let files = (OPS.review_files || []);
  if (SERVER && DATA.version) { try { files = (await api(`/api/reviews/${encodeURIComponent(DATA.version)}/files`)).files } catch {} }
  history.replaceChildren(...(files.length ? files.map(fileRow) : [el('div', { class: 'empty' }, 'Nenhum laudo ou pacote gerado ainda para esta inspeção.')]))
}
// ENTRADA: botão Atualizar histórico.
$('refreshFiles').onclick = renderFileHistory;
// ENTRADA: botão "Gerar laudo" (#makeReport).
// EXTENSAO: campo novo do laudo enviado ao servidor = chave neste objeto + leitura em _build_report (platform_server.py).
$('makeReport').onclick = async () => {
  $('makeReport').disabled = true;
  $('reportStatus').textContent = 'Gerando o laudo (as evidências são desenhadas uma a uma)…';
  try {
    const result = await api('/api/report', { review: DATA.version, decisions: allDrafts(), reviewer: $('reviewer').value.trim(), title: $('reportTitle').value.trim(), client: $('reportClient').value.trim(), site: $('reportSite').value.trim(), inspector: $('reportInspector').value.trim(), notes: $('reportNotes').value.trim(), details: reportDetails(), pdf: $('reportPdf').checked, all_images: $('reportAllImages').checked });
    $('reportStatus').replaceChildren(`Laudo gerado: ${result.images} fotos, ${result.occurrences} ocorrências. `, el('a', { href: result.html, target: '_blank', rel: 'noopener' }, 'Abrir HTML'), result.pdf ? ' · ' : '', result.pdf ? el('a', { href: result.pdf, target: '_blank', rel: 'noopener' }, 'Abrir PDF') : (result.pdf_reason ? ` · PDF não gerado: ${result.pdf_reason}` : ''));
    toast('Laudo técnico gerado', `${result.occurrences} ocorrências em ${result.images} fotos. Campos técnicos aguardam validação.`);
    renderFileHistory()
  } catch (error) { $('reportStatus').textContent = `Não concluído: ${error.message}` }
  finally { $('makeReport').disabled = !SERVER || !HAS_IMAGES }
};
// ENTRADA: botão "Gerar pacote de evidências" (#makeEvidence); tamanho vazio = resolução original.
$('makeEvidence').onclick = async () => {
  $('makeEvidence').disabled = true;
  $('evidenceStatus').textContent = 'Desenhando as caixas e compactando…';
  try {
    const size = $('evidenceSize').value;
    const result = await api('/api/export/evidence', { review: DATA.version, decisions: allDrafts(), max_side: size === '' ? null : Number(size), only: $('evidenceOnly').value, include_suggestions: $('evidenceSuggestions').checked });
    $('evidenceStatus').replaceChildren(`${result.images} imagens · ${fmtBytes(result.bytes)} · `, el('a', { href: result.href, download: '' }, 'Baixar o ZIP'));
    toast('Pacote de evidências pronto', `${result.images} imagens com caixas (${fmtBytes(result.bytes)}).`);
    renderFileHistory()
  } catch (error) { $('evidenceStatus').textContent = `Não concluído: ${error.message}` }
  finally { $('makeEvidence').disabled = !SERVER || !HAS_IMAGES }
};
// CSV montado no navegador com o estado atual (rascunho local sobre o registro).
// ENTRADA: botão "Baixar CSV" (#downloadCsv). Uma linha por caixa; foto sem caixa ganha uma linha vazia.
$('downloadCsv').onclick = () => {
  // EXTENSAO: coluna nova no CSV = nome aqui E o valor na MESMA posição nos dois rows.push abaixo.
  const rows = [['numero_foto', 'image_sha256', 'filename', 'status', 'origem', 'caixa', 'classe_id', 'classe', 'x1', 'y1', 'x2', 'y2', 'nota']];
  DATA.images.forEach((photo, index) => {
    const decision = touchedDraft(photo);  // rascunho editado vale como está (até "pending"); intocado, vale o registro
    const boxes = decision ? decision.boxes : photo.boxes;
    const status = decision ? decision.status : photo.status;
    if (!boxes.length) rows.push([index + 1, photo.image_sha256, photo.filename, status, decision ? 'revisao_humana' : 'registro', '', '', '', '', '', '', '', '']);
    boxes.forEach((box, position) => rows.push([index + 1, photo.image_sha256, photo.filename, status, decision ? 'revisao_humana' : 'registro', position + 1, box.class_id, classOf(box.class_id).label, ...box.bbox_xyxy.map(Math.round), box.note || '']))
  });
  // ATENCAO: o texto começa com o BOM do UTF-8 (caractere invisível entre as aspas) para o Excel
  // reconhecer os acentos; separador ";" e quebras CRLF, aspas duplicadas para escapar.
  const csv = '﻿' + rows.map(row => row.map(value => `"${String(value ?? '').replaceAll('"', '""')}"`).join(';')).join('\r\n');
  downloadBlob(new Blob([csv], { type: 'text/csv;charset=utf-8' }), `dronecamp-ocorrencias-${DATA.version}.csv`)
};
// ENTRADA: botão "Baixar arquivo de revisão" (#downloadReview): mesmo JSON do #export, sem cópia no servidor.
$('downloadReview').onclick = () => { const feedback = reviewFile(); if (feedback) downloadBlob(new Blob([JSON.stringify(feedback, null, 2)], { type: 'application/json' }), 'dronecamp-revisao-' + DATA.version + '.json') };

/* ======================= Gráficos SVG (sem bibliotecas externas) ======================= */
// Elementos do index.html: nenhum fixo (desenha dentro do contêiner recebido). Nenhuma rota da API.
// drawnCharts: contêiner → argumentos do último desenho, para redesenhar ao mudar a largura da janela.
const drawnCharts = new Map();
// chartViewport/chartTimer: largura atual e temporizador (150 ms) que agrupa vários eventos de resize.
let chartViewport = innerWidth, chartTimer;
// Só redesenha quando a LARGURA muda; gráficos que saíram da página são esquecidos.
addEventListener('resize', () => {
  if (innerWidth === chartViewport) return;
  chartViewport = innerWidth;
  clearTimeout(chartTimer);
  chartTimer = setTimeout(() => drawnCharts.forEach((args, node) => { if (!node.isConnected) drawnCharts.delete(node); else if (node.offsetParent) lineChart(node, ...args) }), 150)
});
// Por padrão o eixo X é a época (curvas de um treino). Opções para eixos de categorias
// (ex.: um ponto por treino na Evolução dos modelos): xKey, xRange [mín, máx], xTicks
// [{ x, label, sub }], tipTitle(ponto), dots, highlight (x com anel), hollow (x com
// ponto vazado) e dividers [{ x, label }] (linha vertical tracejada antes do ponto x).
/**
 * Desenha um gráfico de linhas em SVG dentro de container (eixos, grade, linhas, pontos e dica ao passar o mouse).
 * Parâmetros: container (elemento vazio); points (lista de objetos, ex.: curvas por época com epoch,
 * map50...); series (lista de { key, label, color, dashed }); options (ver comentário acima: height,
 * percent, label, empty, marker, xKey, xRange, xTicks, tipTitle, dots, highlight, highlightLabel,
 * hollow, dividers). Retorna: nada.
 * Chamada por: renderHome, renderRunDetail, renderMonitor e renderEvolution.
 */
function lineChart(container, points, series, options = {}) {
  drawnCharts.set(container, [points, series, options]);
  container.replaceChildren();
  // 1. dimensões e margens (pad); a largura acompanha o contêiner (mínimo 280 px).
  const xKey = options.xKey || 'epoch';
  const height = options.height || 220, width = Math.max(280, Math.round(container.clientWidth) || 640), pad = { l: 44, r: 12, t: options.highlight != null ? 22 : 10, b: options.xTicks ? 40 : 24 };
  // Só séries com pelo menos um valor; sem dados mostra a mensagem de vazio.
  const usable = series.filter(serie => points.some(point => point[serie.key] != null));
  if (!points.length || !usable.length) { container.append(el('div', { class: 'empty' }, options.empty || 'Sem dados de épocas ainda.')); return }
  // 2. faixas dos eixos (com 5% de folga no topo) e funções de escala sx/sy (valor → px do SVG).
  const xs = points.map(point => point[xKey]);
  const [xMin, xMax] = options.xRange || [Math.min(...xs), Math.max(...xs, Math.min(...xs) + 1)];
  const values = usable.flatMap(serie => points.map(point => point[serie.key]).filter(value => value != null));
  let yMax = options.percent ? Math.max(0.05, ...values) : Math.max(...values);
  let yMin = options.percent ? 0 : Math.min(...values);
  if (yMax === yMin) yMax = yMin + 1;
  yMax += (yMax - yMin) * .05;
  const sx = x => pad.l + (x - xMin) / (xMax - xMin) * (width - pad.l - pad.r);
  const sy = y => pad.t + (1 - (y - yMin) / (yMax - yMin)) * (height - pad.t - pad.b);
  // 3. cria o <svg> e o atalho add(tag, atributos) que acrescenta um elemento a ele.
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`); svg.setAttribute('preserveAspectRatio', 'none'); svg.style.height = height + 'px';
  svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', options.label || usable.map(serie => serie.label).join(', '));
  const add = (tag, attrs) => { const node = document.createElementNS(ns, tag); Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v)); svg.append(node); return node };
  // 4. grade horizontal com 5 linhas e rótulos do eixo Y.
  for (let i = 0; i <= 4; i++) {
    const value = yMin + (yMax - yMin) * i / 4, y = sy(value);
    add('line', { x1: pad.l, x2: width - pad.r, y1: y, y2: y, stroke: 'var(--chart-grid)', 'stroke-width': 1 });
    const label = add('text', { x: pad.l - 6, y: y + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--muted)' });
    label.textContent = options.percent ? `${Math.round(value * 100)}%` : value.toFixed(value >= 10 ? 0 : 2)
  }
  // 5. rótulos do eixo X: categorias (xTicks) ou início/meio/fim em épocas.
  if (options.xTicks) {
    // Rótulos de categoria: com muitos pontos, só um a cada "step" para não sobrepor.
    const step = Math.max(1, Math.ceil(options.xTicks.length / Math.max(1, Math.floor((width - pad.l - pad.r) / 76))));
    options.xTicks.forEach((tick, index) => {
      if (index % step && index !== options.xTicks.length - 1) return;
      const label = add('text', { x: sx(tick.x), y: height - 22, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--ink-2)', 'font-weight': 600 });
      label.textContent = tick.label;
      if (tick.sub) { const sub = add('text', { x: sx(tick.x), y: height - 8, 'text-anchor': 'middle', 'font-size': 10.5, fill: 'var(--muted)' }); sub.textContent = tick.sub }
    })
  } else [xMin, Math.round((xMin + xMax) / 2), xMax].forEach(x => {
    const anchor = x === xMax ? 'end' : x === xMin ? 'start' : 'middle';
    const label = add('text', { x: sx(x), y: height - 6, 'text-anchor': anchor, 'font-size': 11, fill: 'var(--muted)' });
    label.textContent = `ép. ${x}`
  });
  // 6. divisórias verticais (ex.: troca de dataset) e destaque do ponto "em uso".
  (options.dividers || []).forEach(divider => {
    const x = sx(divider.x - .5);
    add('line', { x1: x, x2: x, y1: pad.t, y2: height - pad.b, stroke: 'var(--line-strong)', 'stroke-dasharray': '4 4', 'vector-effect': 'non-scaling-stroke' });
    if (divider.label) { const text = add('text', { x: x + 4, y: height - pad.b - 6, 'font-size': 10.5, fill: 'var(--muted)' }); text.textContent = divider.label }
  });
  if (options.highlight != null) {
    const x = sx(options.highlight);
    add('rect', { x: x - 14, y: pad.t, width: 28, height: height - pad.t - pad.b, rx: 6, fill: 'var(--primary-soft)' });
    const text = add('text', { x, y: pad.t - 8, 'text-anchor': 'middle', 'font-size': 10.5, 'font-weight': 600, fill: 'var(--primary)' });
    text.textContent = options.highlightLabel || 'em uso'
  }
  // 7. uma linha (path) por série e, se options.dots, os pontos (vazados para os x em hollow).
  const hollow = new Set(options.hollow || []);
  usable.forEach(serie => {
    const drawn = points.filter(point => point[serie.key] != null);
    const path = drawn.map((point, i) => `${i ? 'L' : 'M'}${sx(point[xKey]).toFixed(1)},${sy(point[serie.key]).toFixed(1)}`).join('');
    add('path', { d: path, fill: 'none', stroke: serie.color, 'stroke-width': 2, 'vector-effect': 'non-scaling-stroke', 'stroke-dasharray': serie.dashed ? '5 4' : '' });
    if (options.dots) drawn.forEach(point => add('circle', { cx: sx(point[xKey]), cy: sy(point[serie.key]), r: point[xKey] === options.highlight ? 5 : 3.5, fill: hollow.has(point[xKey]) ? 'var(--surface)' : serie.color, stroke: serie.color, 'stroke-width': 2 }))
  });
  // 8. marcador vertical (ex.: melhor época) e cursor que segue o mouse.
  if (options.marker != null) { const x = sx(options.marker); add('line', { x1: x, x2: x, y1: pad.t, y2: height - pad.b, stroke: 'var(--ai)', 'stroke-dasharray': '3 3', 'vector-effect': 'non-scaling-stroke' }) }
  const cursor = add('line', { x1: 0, x2: 0, y1: pad.t, y2: height - pad.b, stroke: 'var(--line-strong)', visibility: 'hidden', 'vector-effect': 'non-scaling-stroke' });
  const tip = el('div', { class: 'tooltip hidden' });
  // 9. dica: acha o ponto mais próximo do x do mouse e mostra os valores das séries.
  svg.addEventListener('pointermove', event => {
    const rect = svg.getBoundingClientRect();
    const vx = (event.clientX - rect.left) / rect.width * width;
    const at = xMin + (vx - pad.l) / (width - pad.l - pad.r) * (xMax - xMin);
    const nearest = points.reduce((best, point) => Math.abs(point[xKey] - at) < Math.abs(best[xKey] - at) ? point : best, points[0]);
    const x = sx(nearest[xKey]);
    cursor.setAttribute('x1', x); cursor.setAttribute('x2', x); cursor.setAttribute('visibility', 'visible');
    // Com tipTitle (eixo de categorias) a dica tem uma linha por métrica.
    const glue = options.tipTitle ? '\n' : ' · ';
    tip.textContent = (options.tipTitle ? options.tipTitle(nearest) : `Época ${nearest.epoch}`) + glue + usable.map(serie => `${serie.label} ${nearest[serie.key] == null ? '—' : options.percent ? pct(nearest[serie.key]) : nearest[serie.key].toFixed(3)}`).join(glue);
    tip.classList.toggle('multi', Boolean(options.tipTitle));
    tip.classList.remove('hidden'); tip.style.left = '0px';
    const left = Math.max(0, Math.min(rect.width - tip.offsetWidth - 4, x / width * rect.width + 8));
    tip.style.left = left + 'px'; tip.style.top = '4px'
  });
  // 10. some a dica ao sair; acrescenta SVG, dica e legenda ao contêiner.
  svg.addEventListener('pointerleave', () => { cursor.setAttribute('visibility', 'hidden'); tip.classList.add('hidden') });
  container.append(svg, tip, el('div', { class: 'legend' }, ...usable.map(serie => { const node = el('span', {}, el('i'), serie.label); node.firstChild.style.setProperty('--c', serie.color); return node })))
}
/* Nome legível da execução (operations.py grava display_name/display_detail; runs/nomes.json personaliza). */
// runName: nome exibido da execução (nome personalizado gravado ou NAMING.run).
const runName = run => run ? (run.display_name || NAMING.run(run.id)) : '';
// Pasta do dataset de uma comparação ("data/pilot/ceasa_v8_piloto_s42/dataset.yaml" → "ceasa_v8_piloto_s42"); sem pasta, o nome do arquivo.
const datasetFolder = path => { const parts = String(path).replace(/\\/g, '/').split('/').filter(Boolean); const folder = parts.length > 1 ? parts[parts.length - 2] : (parts[0] || '').replace(/\.ya?ml$/, ''); return folder || String(path) };
// runById: execução pelo id (ou só { id } se não estiver no instantâneo); runLabel: nome dela.
const runById = id => OPS.runs.find(run => run.id === id) || { id };
const runLabel = id => runName(runById(id));
// Pesos: "Treino Piloto 56038EB9 · melhor época" (nome personalizado da execução, se houver) ou "Modelo Base YOLO26l".
/**
 * Rótulo de um arquivo de pesos para os seletores.
 * Parâmetros: path (ex.: 'runs/<id>/fit/weights/best.pt'). Retorna: string.
 */
const weightLabel = path => {
  const run = OPS.runs.find(value => Object.values(value.weights || {}).includes(path));
  if (!run) return NAMING.weights(path);
  const stem = path.split('/').pop().replace(/\.pt$/, '');
  return `${runName(run)} · ${{ best: 'melhor época', last: 'última época' }[stem] || stem}`
};
// Rótulos das divisões (train/val/test) nas tabelas de comparação e cobertura.
// splitLabel acrescenta um aviso quando a divisão contém fotos vistas no treino ("visto" no nome).
const SPLIT_LABELS = { train: 'Treino', val: 'Validação', test: 'Teste' };
const splitLabel = split => { const base = String(split || ''); const seen = base.includes('visto'); const key = base.split(' ')[0].split('(')[0]; return (SPLIT_LABELS[key] || key) + (seen ? ' ⚠ fotos vistas no treino' : '') };
// Séries dos gráficos de perda (erro) por época: chaves das curvas gravadas por operations.py.
// EXTENSAO: série nova = { key, label, color } aqui, com a chave já presente em run.curves (operations.py).
const LOSS_SERIES = [
  { key: 'train_box', label: 'Erro de posição (treino)', color: '#2457F5' },
  { key: 'train_cls', label: 'Erro de classe (treino)', color: '#7C3AED' },
  { key: 'val_box', label: 'Erro de posição (validação)', color: '#2457F5', dashed: true },
  { key: 'val_cls', label: 'Erro de classe (validação)', color: '#7C3AED', dashed: true }
];
// Séries de métricas de validação: usadas nos gráficos, nos cartões do Monitoramento e na Evolução.
// ATENCAO: tests/test_platform.py recorta a lista abaixo do início até o primeiro fechamento de
// colchete seguido de ponto e vírgula; não ponha comentários dentro dela.
// EXTENSAO: métrica nova = mais um objeto { key, label, color } na lista (a chave deve existir em run.curves e run.best_epoch).
const METRIC_SERIES = [
  { key: 'precision', label: 'Precisão (sugestões certas)', color: '#D97706' },
  { key: 'recall', label: 'Recall (caixas encontradas)', color: '#2457F5' },
  { key: 'map50', label: 'mAP50 (acerto geral)', color: '#16A34A' },
  { key: 'map50_95', label: 'mAP50-95 (acerto com encaixe exato)', color: '#7C3AED' }
];

/* ======================= Treinamento ======================= */
// Elementos do index.html: cycleSteps, deviceChip, trainData, trainWeights, trainEpochs, trainImgsz,
// trainBatch, trainTile, trainGpuEval, trainCommand, trainEstimate, startTrain, trainServerHint,
// runsTable, runsHint, runDetailTitle, runDetailChips, runLossChart, runMetricChart, runPlots.
// Rotas da API: nenhuma direta; o botão Iniciar treino usa startJob (POST /api/jobs, bloco Servidor).
// stateLabel/stateClass: texto e estilo do chip de estado de uma execução.
// EXTENSAO: estado novo (gravado por operations.py) = acrescente nos dois objetos.
const stateLabel = value => ({ completed: 'concluído', running: 'em andamento', interrupted: 'interrompido', failed: 'falhou', prepared: 'preparado' }[value] || value);
const stateClass = value => ({ completed: 'ok', running: 'live ok', interrupted: 'warn', failed: 'bad', prepared: 'neutral' }[value] || 'neutral');
// winPath: troca "/" por "\" para os comandos copiados no PowerShell do Windows.
const winPath = path => String(path || '').replaceAll('/', '\\');
// PY: prefixo dos comandos copiáveis (Python do venv do projeto no Windows).
const PY = '.\\.venv\\Scripts\\python.exe -m dronecamp_ia';
/**
 * Descobre a execução que gerou um arquivo de pesos, pelo SHA-256, procurando nas comparações salvas.
 * Parâmetros: sha (SHA-256 dos pesos). Retorna: objeto da execução (ou { id }) ou null.
 * Chamada por: renderHome e renderEvolution.
 */
function runForWeights(sha) {
  if (!sha) return null;
  for (const comparison of OPS.comparisons) {
    const model = comparison.models.find(value => value.weights_sha256 === sha);
    if (model && model.run_id) return OPS.runs.find(run => run.id === model.run_id) || { id: model.run_id }
  }
  return null
}
/**
 * Caixa com um comando de terminal e botão Copiar.
 * Parâmetros: text (comando). Retorna: elemento .command.
 */
function commandBox(text) {
  const box = el('div', { class: 'command' }, text);
  const copy = el('button', { type: 'button' }, 'Copiar');
  copy.onclick = async () => { try { await navigator.clipboard.writeText(text); copy.textContent = 'Copiado' } catch { copy.textContent = 'Selecione e copie' } setTimeout(() => copy.textContent = 'Copiar', 1500) };
  box.append(copy);
  return box
}
/**
 * Sugere o nome da próxima versão de revisão (ex.: ceasa_v8... → ceasa_v9_ID).
 * Parâmetros: nenhum (lê DATA.version). Retorna: string.
 */
function nextVersionName() {
  const match = String(DATA.version || '').match(/^([a-z]+)_v(\d+)/i);
  return match ? `${match[1]}_v${Number(match[2]) + 1}_ID` : (DATA.version ? `${DATA.version}_rev1` : 'nova_versao_ID')
}
/**
 * Desenha os 6 passos do ciclo de fine-tuning com os comandos prontos para copiar.
 * Parâmetros: nenhum. Retorna: nada. Chamada por: renderTraining.
 */
function renderCycle() {
  const directory = winPath(OPS.review_directory || `data/reviews/${DATA.version}`);
  const next = nextVersionName();
  const decided = DATA.images.filter(photo => decidedStatus(photo)).length;
  const latest = OPS.runs.find(run => run.weights && run.weights.best);
  const previous = latest ? winPath(latest.weights.best) : 'runs\\PILOTO_ANTERIOR\\fit\\weights\\best.pt';
  // EXTENSAO: passo novo do ciclo = mais um objeto { title, text, command | action, done } nesta lista.
  // ATENCAO: os comandos precisam bater com os argumentos aceitos por cli.py.
  const steps = [
    { title: 'Revisar e salvar', text: `${decided}/${DATA.images.length} fotos decididas nesta inspeção. Salve o arquivo de revisão.`, done: decided > 0 && decided === DATA.images.length, action: () => showView('review') },
    { title: 'Importar decisões', text: 'Cria um novo registro; o atual fica preservado.', command: `${PY} import-review --registry "${directory}\\registry.json" --feedback "SEU_ARQUIVO.json" --output "data\\reviews\\${next}\\registry.json"` },
    { title: 'Montar dataset piloto', text: 'Só fotos com decisão humana completa; divisão por cena.', command: `${PY} build-pilot-data --registry "data\\reviews\\${next}\\registry.json" --output "data\\pilot\\${next}_piloto_s42"` },
    { title: 'Treinar (fine-tuning)', text: 'Treino em CPU; acompanhe em Monitoramento.', command: `${PY} train-pilot --data "data\\pilot\\${next}_piloto_s42\\dataset.yaml"` },
    { title: 'Comparar com o atual', text: 'Mesmo dataset, mesmas divisões; olhe classe por classe.', command: `.\\.venv\\Scripts\\python.exe scripts\\compare_pilot_models.py --data "data\\pilot\\${next}_piloto_s42\\dataset.yaml" --weights "${previous}" --weights "runs\\PILOTO_NOVO\\fit\\weights\\best.pt" --output "runs\\compare_${next}.json"` },
    { title: 'Sugerir na revisão', text: 'Sugestões ficam pendentes até decisão humana.', command: `${PY} suggest --weights "runs\\PILOTO_NOVO\\fit\\weights\\best.pt" --registry "data\\reviews\\${next}\\registry.json"` }
  ];
  $('cycleSteps').replaceChildren(...steps.map(step => {
    const node = el('div', { class: 'step' + (step.done ? ' done' : '') }, el('h3', {}, step.title), el('p', {}, step.text));
    if (step.command) node.append(commandBox(step.command));
    if (step.action) node.append(el('button', { type: 'button', class: 'small', onclick: step.action }, 'Abrir revisão'));
    return node
  }))
}
/**
 * Preenche um <select> com opções [valor, rótulo], preservando a escolha atual quando ela ainda existe.
 * Parâmetros: select (elemento), options (lista de [valor, rótulo]), selected (valor padrão). Retorna: nada.
 */
function fillSelect(select, options, selected) {
  const previous = select.value;
  select.replaceChildren(...options.map(([value, label]) => el('option', { value }, label)));
  select.value = options.some(([value]) => value === previous) ? previous : (selected != null && options.some(([value]) => value === selected) ? selected : (options[0] || [''])[0])
}
/**
 * Estima segundos por época para um tamanho de imagem, a partir das execuções anteriores (mais de 3 épocas).
 * Parâmetros: imgsz (px). Retorna: { perEpoch, runs } ou null sem histórico.
 */
function estimateSeconds(imgsz) {
  // Base: s/época das execuções concluídas no mesmo PC; escala ~ imgsz² quando muda o tamanho.
  const done = OPS.runs.filter(run => run.seconds_per_epoch && run.parameters.imgsz && run.epochs_done > 3);
  if (!done.length) return null;
  const same = done.filter(run => run.parameters.imgsz === imgsz);
  const base = same.length ? same : done;
  const perEpoch = base.reduce((sum, run) => sum + run.seconds_per_epoch * (imgsz / run.parameters.imgsz) ** 2, 0) / base.length;
  return { perEpoch, runs: base.length }
}
// duration: segundos em texto curto ("1.5 h", "12 min", "40 s"); null vira "—".
const duration = seconds => seconds == null ? '—' : seconds >= 3600 ? `${(seconds / 3600).toFixed(1)} h` : seconds >= 60 ? `${Math.round(seconds / 60)} min` : `${Math.round(seconds)} s`;
/**
 * Lê o formulário de treino. Parâmetros: nenhum.
 * Retorna: { data, weights, epochs, imgsz, batch, tile, gpu_eval } (vazios numéricos = null).
 * EXTENSAO: opção nova de treino = campo no index.html, chave aqui, flag em updateTrainCommand,
 * id na lista de listeners abaixo de renderRunDetail e leitura no servidor (POST /api/jobs).
 */
function trainOptions() {
  const integer = id => $(id).value === '' ? null : Number($(id).value);
  return { data: $('trainData').value, weights: $('trainWeights').value, epochs: integer('trainEpochs'), imgsz: integer('trainImgsz'), batch: integer('trainBatch'), tile: $('trainTile').checked, gpu_eval: $('trainGpuEval').checked }
}
/**
 * Monta o comando train-pilot equivalente ao formulário e a estimativa de duração.
 * Parâmetros: nenhum. Retorna: nada. Chamada por: renderTraining e mudanças no formulário.
 */
function updateTrainCommand() {
  const options = trainOptions();
  let command = `${PY} train-pilot --data "${winPath(options.data || 'data/pilot/SEU_DATASET/dataset.yaml')}"`;
  if (options.weights) command += ` --weights "${winPath(options.weights)}"`;
  ['epochs', 'imgsz', 'batch'].forEach(key => { if (options[key] != null) command += ` --${key} ${options[key]}` });
  if (options.tile) command += ' --tile';
  if (options.gpu_eval) command += ' --gpu-eval';
  $('trainCommand').replaceWith(Object.assign(commandBox(command), { id: 'trainCommand' }));
  const estimate = estimateSeconds(options.imgsz || OPS.pilot_defaults.imgsz);
  $('trainEstimate').textContent = estimate
    ? `Estimativa em CPU: ≈ ${duration(estimate.perEpoch)} por época, ≈ ${duration(estimate.perEpoch * (options.epochs || OPS.pilot_defaults.epochs || 0))} no total (média de ${estimate.runs} execução(ões) anteriores).`
    : 'Sem execuções anteriores para estimar a duração.'
}
/**
 * Desenha a seção Treinamento: ciclo, formulário, tabela de execuções e detalhes da escolhida.
 * Parâmetros: nenhum. Retorna: nada. Chamada por: showView('training'), refreshOperations, clique na tabela.
 */
function renderTraining() {
  renderCycle();
  $('deviceChip').textContent = `dispositivo: ${OPS.device || 'cpu'}`;
  // 1. seletores de dataset e pesos; padrões do piloto (configs/project.yaml) nos campos vazios.
  fillSelect($('trainData'), OPS.datasets.map(dataset => [dataset.path, `${NAMING.dataset(dataset.name)} · ${Object.entries(dataset.splits || {}).map(([k, v]) => `${k} ${v}`).join(' / ')}`]), (OPS.datasets[OPS.datasets.length - 1] || {}).path);
  fillSelect($('trainWeights'), [['', `Padrão do projeto (${NAMING.weights(OPS.base_model || 'yolo26l.pt')})`], ...OPS.weights.map(path => [path, weightLabel(path)])], '');
  const defaults = OPS.pilot_defaults || {};
  ['epochs', 'imgsz', 'batch'].forEach(key => { const input = $('train' + key[0].toUpperCase() + key.slice(1)); if (input.value === '' && defaults[key] != null) input.value = defaults[key] });
  updateTrainCommand();
  $('startTrain').disabled = !SERVER || jobRunning();
  $('trainServerHint').textContent = SERVER
    ? 'O treino roda no servidor local deste PC; feche a página à vontade, o processo continua.'
    : 'Aberta direto do disco, a página não inicia processos: copie o comando no terminal da pasta sistema_ia ou use o atalho Abrir plataforma DroneCamp.cmd.';
  // 2. tabela de execuções (clique ou Enter escolhe a execução mostrada nos detalhes).
  const body = $('runsTable');
  body.replaceChildren();
  $('runsHint').textContent = `${OPS.runs.length} execuções · atualizado ${fmtDate(OPS.generated_at || Date.now())}`;
  if (!OPS.runs.length) body.append(el('tr', {}, el('td', { colspan: 5, class: 'empty' }, 'Nenhum treino encontrado em runs/.')));
  OPS.runs.forEach(run => {
    const row = el('tr', { class: 'clickable', tabindex: 0, 'aria-selected': String(state.run === run.id) },
      el('td', { title: run.id }, el('div', {}, el('b', {}, runName(run))), el('small', { class: 'muted' }, `${fmtDate(run.started_at)} · ${run.display_detail || run.checkpoint || ''}`)),
      el('td', {}, el('span', { class: 'chip ' + stateClass(run.state) }, el('span', { class: 'dot' }), stateLabel(run.state))),
      el('td', { class: 'num' }, `${run.epochs_done}/${run.epochs_planned || '?'}`),
      el('td', { class: 'num' }, run.best_epoch ? `${pct(run.best_epoch.map50)} (ép. ${run.best_epoch.epoch})` : '—'),
      el('td', { class: 'num' }, run.seconds_per_epoch ? Math.round(run.seconds_per_epoch) : '—'));
    const select = () => { state.run = run.id; renderTraining() };
    row.onclick = select;
    row.onkeydown = event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select() } };
    body.append(row)
  });
  // 3. detalhes: a escolhida, senão a mais nova com curvas, senão a primeira.
  const run = OPS.runs.find(value => value.id === state.run) || OPS.runs.find(value => value.curves.length) || OPS.runs[0];
  renderRunDetail(run)
}
/**
 * Detalhes de uma execução: parâmetros, gráficos de perda/métricas e imagens geradas pelo Ultralytics.
 * Parâmetros: run (objeto de OPS.runs ou undefined). Retorna: nada.
 */
function renderRunDetail(run) {
  $('runDetailTitle').textContent = run ? `Detalhes · ${runName(run)}` : 'Detalhes da execução';
  $('runDetailChips').replaceChildren(...(run ? Object.entries(run.parameters).map(([key, value]) => el('span', { class: 'chip neutral', style: 'margin:2px' }, `${key} ${value}`)) : []));
  lineChart($('runLossChart'), run ? run.curves : [], LOSS_SERIES, { label: 'Perdas por época' });
  lineChart($('runMetricChart'), run ? run.curves : [], METRIC_SERIES, { percent: true, marker: run && run.best_epoch ? run.best_epoch.epoch : null, label: 'Métricas de validação por época' });
  // EXTENSAO: gráfico novo do Ultralytics = chave (como operations.py a grava em run.plots) e nome aqui.
  const plotNames = { results: 'Curvas (Ultralytics)', confusion: 'Matriz de confusão normalizada', pr_curve: 'Curva precisão × recall', val_pred: 'Predições na validação', labels: 'Distribuição dos rótulos' };
  // Caminhos relativos à pasta da revisão; pelo servidor, /runs/... é servido de sistema_ia/runs.
  const plotUrl = path => SERVER ? '/' + path.replace(/^(\.\.\/)+/, '') : path;
  $('runPlots').replaceChildren(...(run ? Object.entries(run.plots).map(([key, path]) => el('a', { href: plotUrl(path), target: '_blank', rel: 'noopener' }, el('img', { src: plotUrl(path), alt: plotNames[key] || key, loading: 'lazy' }), el('span', {}, plotNames[key] || key))) : []))
}
// ENTRADA: qualquer mudança no formulário de treino atualiza o comando.
['trainData', 'trainWeights', 'trainEpochs', 'trainImgsz', 'trainBatch', 'trainTile', 'trainGpuEval'].forEach(id => $(id).addEventListener('input', updateTrainCommand));

/* ======================= Monitoramento ======================= */
// Elementos do index.html: monitorOffline, liveDot, monitorTitle, monitorState, monitorFacts, monitorTiles,
// ringValue, ringText, ringSub, liveLossChart, liveMetricChart, suggestWeights, suggestConf, jobState,
// jobHint, jobLog, startSuggest, stopJob, startTrain.
// Rotas da API: nenhuma direta; os dados vêm de refreshOperations (GET /api/operations e GET /api/jobs).
// job: tarefa do servidor ({ state: 'idle'|'running'|'finished'|'failed', kind, log, log_tail }).
let job = { state: 'idle' };
const jobRunning = () => job.state === 'running';
/**
 * Desenha o Monitoramento: treino em andamento (ou o mais recente), anel de progresso, fatos,
 * cartões de métricas, gráficos ao vivo e o formulário de sugestões.
 * Parâmetros: nenhum. Retorna: nada. Chamada por: showView('monitor') e refreshOperations (a cada 5 s).
 */
function renderMonitor() {
  $('monitorOffline').classList.toggle('hidden', SERVER);
  // Treino recém-iniciado pela página ainda sem época gravada: mostra a execução mais nova.
  const starting = jobRunning() && job.kind === 'train-pilot' ? OPS.runs[0] : null;
  const run = OPS.runs.find(value => value.state === 'running') || starting || OPS.runs.find(value => value.curves.length) || OPS.runs[0];
  $('liveDot').classList.toggle('hidden', !OPS.runs.some(value => value.state === 'running') && !jobRunning());
  if (!run) {
    $('monitorTitle').textContent = 'Nenhum treino encontrado';
    $('monitorState').replaceChildren(); $('monitorFacts').replaceChildren(); $('monitorTiles').replaceChildren()
  } else {
    $('monitorTitle').textContent = `${runName(run)} · iniciado em ${fmtDate(run.started_at)}`;
    $('monitorTitle').title = run.id;
    $('monitorState').replaceChildren(el('span', { class: 'chip ' + stateClass(run.state) }, el('span', { class: 'dot' }), stateLabel(run.state)));
    // Anel de progresso: fração de épocas concluídas.
    const planned = run.epochs_planned || run.epochs_done || 1;
    const fraction = Math.min(1, run.epochs_done / planned);
    // ATENCAO: 339.3 = circunferência do círculo do anel (2 × π × 54); se mudar o raio r="54" no index.html, mude aqui.
    $('ringValue').setAttribute('stroke-dashoffset', (339.3 * (1 - fraction)).toFixed(1));
    $('ringText').textContent = `${Math.round(fraction * 100)}%`;
    $('ringSub').textContent = `época ${run.epochs_done}/${planned}`;
    // Fatos da execução (tempo decorrido, restante estimado, s/época, dispositivo, pesos iniciais, última escrita).
    const elapsed = run.last && run.last.time;
    const remaining = run.state === 'running' && run.seconds_per_epoch ? run.seconds_per_epoch * (planned - run.epochs_done) : null;
    const facts = [['Tempo decorrido', duration(elapsed)], ['Tempo restante', run.state === 'running' ? duration(remaining) : '—'], ['Por época', duration(run.seconds_per_epoch)], ['Dispositivo', run.device || OPS.device || '—'], ['Pesos iniciais', run.checkpoint || '—'], ['Última escrita', run.updated_at ? new Date(run.updated_at).toLocaleTimeString('pt-BR') : '—']];
    $('monitorFacts').replaceChildren(...facts.flatMap(([key, value]) => [el('dt', {}, key), el('dd', {}, value)]));
    // Cartões: valor da última época e variação em pontos percentuais contra a anterior.
    const last = run.curves[run.curves.length - 1] || {}, before = run.curves[run.curves.length - 2] || {};
    $('monitorTiles').replaceChildren(...METRIC_SERIES.map(serie => {
      const value = last[serie.key], delta = value != null && before[serie.key] != null ? value - before[serie.key] : null;
      return el('div', { class: 'tile' }, el('span', {}, serie.label), el('b', {}, pct(value)), el('span', { class: 'delta ' + (delta > 0 ? 'up' : delta < 0 ? 'down' : '') }, delta == null ? '—' : `${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(1)} p.p.`))
    }))
  }
  lineChart($('liveLossChart'), run ? run.curves : [], LOSS_SERIES);
  lineChart($('liveMetricChart'), run ? run.curves : [], METRIC_SERIES, { percent: true, marker: run && run.best_epoch ? run.best_epoch.epoch : null });
  // ATENCAO: aqui value.weights.best é lido sem checar se value.weights existe (em renderUpload há a checagem);
  // uma execução sem o campo weights no instantâneo quebraria esta seção.
  fillSelect($('suggestWeights'), OPS.weights.filter(path => path.startsWith('runs/')).map(path => [path, weightLabel(path)]), (OPS.runs.find(value => value.weights.best) || { weights: {} }).weights.best);
  if ($('suggestConf').value === '' && OPS.suggestion_conf) $('suggestConf').value = OPS.suggestion_conf;
  renderJob()
}
/**
 * Mostra o estado da tarefa do servidor (chip, dica, fim do log) e habilita/desabilita os botões.
 * Parâmetros: nenhum (lê job). Retorna: nada.
 */
function renderJob() {
  // EXTENSAO: tipo novo de tarefa = nome no objeto da dica abaixo (train-pilot, suggest) e no servidor (POST /api/jobs).
  const labels = { idle: 'sem tarefa', running: 'em andamento', finished: 'concluída', failed: 'falhou' };
  $('jobState').textContent = labels[job.state] || job.state;
  $('jobState').className = 'chip ' + ({ running: 'live ok', finished: 'ok', failed: 'bad' }[job.state] || 'neutral');
  $('jobHint').textContent = SERVER
    ? (job.kind ? `${{ 'train-pilot': 'Treino piloto', suggest: 'Geração de sugestões' }[job.kind] || job.kind} · log em ${job.log}` + (job.kind === 'suggest' && job.state === 'finished' ? ' · recarregue a página para ver as novas sugestões.' : '') : 'Uma tarefa por vez. Pesos novos não são adotados automaticamente.')
    : 'Disponível quando a plataforma é aberta pelo servidor local (Abrir plataforma DroneCamp.cmd).';
  $('jobLog').textContent = (job.log_tail || []).join('\n') || '—';
  $('jobLog').scrollTop = $('jobLog').scrollHeight;
  $('startSuggest').disabled = !SERVER || jobRunning() || !$('suggestWeights').value || !HAS_IMAGES;
  $('stopJob').disabled = !SERVER || !jobRunning();
  $('startTrain').disabled = !SERVER || jobRunning()
}

/* ======================= Modelos ======================= */
// Elementos do index.html: adoptionVerdict, compareSelect, compareTable, coverageRun, coverageTable,
// leaderboard, evolutionFilter, evolutionChart, evolutionHint, evolutionSummary.
// Rotas da API: nenhuma direta; usa OPS (instantâneo ou GET /api/operations).
/**
 * Desenha a seção Modelos: veredito de adoção, comparação, cobertura por classe, ranking e evolução.
 * Parâmetros: nenhum. Retorna: nada. Chamada por: showView('models') e refreshOperations.
 */
function renderModels() {
  const pilots = OPS.runs.filter(run => run.stage === 'pilot_train');
  const independent = OPS.runs.some(run => run.independent_evaluation === true);
  const buildings = [...new Set(DATA.images.map(photo => photo.building_group).filter(Boolean))];
  // 1. motivos que impedem adotar um modelo em produção.
  // EXTENSAO: critério novo de adoção = reasons.push(...) aqui e a mesma condição em ok logo abaixo.
  // ATENCAO: o veredito nunca adota sozinho; mesmo "ok" exige análise por classe e revisão técnica.
  const reasons = [];
  if (!independent) reasons.push('Nenhuma execução tem avaliação em edificação independente (todas as divisões vêm do mesmo prédio).');
  if (buildings.length < 3) reasons.push(`A inspeção cobre ${buildings.length || 0} edificaç${buildings.length === 1 ? 'ão' : 'ões'}; o treino de produção exige 3 independentes.`);
  if (!OPS.comparisons.length) reasons.push('Nenhuma comparação salva em runs/compare_*.json.');
  reasons.push('Toda nova versão precisa de análise de regressões por classe e revisão técnica antes do uso em inspeções.');
  const ok = independent && buildings.length >= 3 && OPS.comparisons.length;
  $('adoptionVerdict').replaceChildren(el('div', { class: 'verdict' + (ok ? ' ok' : '') },
    el('div', {}, el('b', {}, ok ? 'Critérios básicos atendidos; falta a análise por classe.' : 'Nenhuma versão pode ser adotada como modelo de produção.'),
      el('ul', {}, ...reasons.map(reason => el('li', {}, reason))),
      el('p', { class: 'muted small-text' }, `${pilots.length} treinos piloto registrados. O modelo atual das sugestões continua só como apoio à revisão.`))));
  // 2. comparação, cobertura, ranking por mAP50 e evolução.
  fillSelect($('compareSelect'), OPS.comparisons.map((comparison, index) => [String(index), NAMING.comparison(comparison.file)]), String(OPS.comparisons.length - 1));
  renderCompare();
  const withCounts = OPS.runs.filter(run => Object.keys(run.class_counts || {}).length);
  fillSelect($('coverageRun'), withCounts.map(run => [run.id, `${runName(run)} · ${fmtDate(run.started_at)}`]), (withCounts[0] || {}).id);
  renderCoverage();
  const ranked = OPS.runs.filter(run => run.best_epoch).sort((a, b) => b.best_epoch.map50 - a.best_epoch.map50);
  const max = Math.max(0.01, ...ranked.map(run => run.best_epoch.map50));
  $('leaderboard').replaceChildren(...(ranked.length ? ranked.map(run => barRow('var(--ok)', run.display_detail ? `${runName(run)} · ${run.display_detail}` : runName(run), run.best_epoch.map50, 0, max, pct(run.best_epoch.map50))) : [el('div', { class: 'empty' }, 'Sem métricas.')]))
  renderEvolution()
}
/* Evolução dos modelos: um ponto por treino (ordem cronológica), lido de OPS.runs a cada
   atualização do instantâneo; nada fixo no código, então cada treino novo aparece sozinho. */
// codeOf: 8 últimos caracteres do id da execução (o hash) em maiúsculas.
const codeOf = id => String(id).slice(-8).toUpperCase();
/**
 * Um ponto por treino com validação, em ordem de início, para o gráfico de evolução.
 * Parâmetros: filter ('all' ou 'completed'). Retorna: lista de { x (1..n), run, precision, recall, map50, map50_95 }.
 * ATENCAO: tests/test_platform.py recorta e executa esta função isolada (só com a lista de métricas e OPS);
 * não use aqui nada além disso.
 */
function evolutionPoints(filter) {
  return OPS.runs.filter(run => run.best_epoch && run.best_epoch.map50 != null && (filter !== 'completed' || run.state === 'completed'))
    .slice().sort((a, b) => String(a.started_at).localeCompare(String(b.started_at)))
    .map((run, index) => ({ x: index + 1, run, ...Object.fromEntries(METRIC_SERIES.map(serie => [serie.key, run.best_epoch[serie.key]])) }))
}
/**
 * Desenha o gráfico "Evolução dos modelos" e o resumo do último treino contra o anterior e o modelo em uso.
 * Parâmetros: nenhum (lê #evolutionFilter). Retorna: nada. Chamada por: renderModels e #evolutionFilter.
 */
function renderEvolution() {
  const points = evolutionPoints($('evolutionFilter').value);
  // Modelo das sugestões desta inspeção: id da execução gravado na página ou, em páginas
  // antigas, o SHA dos pesos procurado nas comparações salvas.
  const meta = DATA.suggestions || {};
  const inUse = (meta.weights_run_id && { id: meta.weights_run_id }) || runForWeights(meta.weights_sha256);
  const used = inUse && points.find(point => point.run.id === inUse.id);
  // Divisórias: entre dois treinos consecutivos com dataset_sha256 diferente (validação em outro conjunto).
  // points.slice(1) desloca os índices: points[index] é o ponto ANTERIOR a point.
  const dividers = points.slice(1).filter((point, index) => point.run.dataset_sha256 && points[index].run.dataset_sha256 && point.run.dataset_sha256 !== points[index].run.dataset_sha256)
    .map(point => ({ x: point.x, label: 'novo dataset' }));
  $('evolutionHint').textContent = points.length ? `${points.length} treino${points.length > 1 ? 's' : ''} com validação · atualizado ${fmtDate(OPS.generated_at || Date.now())}` : '';
  // Opções do gráfico: eixo de categorias (um x por treino), rótulo curto (nome ou hash) e data dd/mm.
  lineChart($('evolutionChart'), points, METRIC_SERIES, {
    xKey: 'x', xRange: [.5, points.length + .5], percent: true, dots: true, height: 260,
    xTicks: points.map(point => ({ x: point.x, label: point.run.display_name && !point.run.display_name.endsWith(codeOf(point.run.id)) ? point.run.display_name.slice(0, 14) : codeOf(point.run.id), sub: fmtDate(point.run.started_at).slice(0, 5) })),
    tipTitle: point => `${runName(point.run)} · ${fmtDate(point.run.started_at)} · ${stateLabel(point.run.state)} · melhor época ${point.run.best_epoch.epoch}/${point.run.epochs_planned || point.run.epochs_done}`,
    highlight: used ? used.x : null, highlightLabel: 'em uso', hollow: points.filter(point => point.run.state === 'running').map(point => point.x), dividers,
    label: 'Evolução das métricas de validação por treino', empty: 'Nenhum treino com métricas de validação em runs/ ainda.'
  });
  // Resumo: último treino contra o anterior e contra o modelo em uso (pontos percentuais de mAP50).
  const last = points[points.length - 1];
  const delta = (a, b) => { const value = (a - b) * 100; return `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(1)} p.p.` };
  const crosses = (a, b) => dividers.some(divider => divider.x > Math.min(a.x, b.x) && divider.x <= Math.max(a.x, b.x));
  const versus = (other, text) => el('span', {}, ` · ${delta(last.map50, other.map50)} ${text}`, crosses(last, other) ? el('span', { class: 'muted' }, ' (outro dataset de validação)') : null);
  const parts = [];
  if (last) parts.push(el('span', {}, 'Último: ', el('b', {}, runName(last.run)), ` · mAP50 ${pct(last.map50)}`));
  if (last && points.length > 1) parts.push(versus(points[points.length - 2], 'vs. o anterior'));
  if (last && used && used !== last) parts.push(versus(used, `vs. o modelo em uso (${runName(used.run)})`));
  $('evolutionSummary').replaceChildren(...parts)
}
// ENTRADA: troca do filtro da evolução.
$('evolutionFilter').onchange = renderEvolution;
/**
 * Tabela da comparação escolhida: recall e precisão por modelo e por divisão.
 * Parâmetros: nenhum (lê #compareSelect). Retorna: nada.
 */
function renderCompare() {
  const comparison = OPS.comparisons[Number($('compareSelect').value)];
  if (!comparison) { $('compareTable').replaceChildren(el('div', { class: 'empty' }, 'Nenhuma comparação salva. Etapa 5 do ciclo de fine-tuning.')); return }
  const splits = [...new Set(comparison.models.flatMap(model => Object.keys(model.splits)))];
  const head = el('tr', {}, el('th', {}, 'Modelo'), ...splits.map(split => el('th', { class: 'num' }, splitLabel(split))));
  const rows = comparison.models.map(model => el('tr', {}, el('td', { title: model.weights }, el('div', {}, el('b', {}, model.run_id ? runLabel(model.run_id) : NAMING.weights(model.weights))), el('small', { class: 'muted' }, `${model.run_id && runById(model.run_id).display_detail ? runById(model.run_id).display_detail + ' · ' : ''}${model.file} · ${String(model.weights_sha256 || '').slice(0, 10)}…`)),
    // Recall = encontradas / verdadeiras; precisão = certas / sugeridas (null quando o denominador é 0).
    ...splits.map(split => {
      const value = model.splits[split];
      if (!value) return el('td', { class: 'num muted' }, '—');
      const recall = value.truth ? value.found / value.truth : null, precision = value.suggestions ? value.correct / value.suggestions : null;
      return el('td', { class: 'num' }, el('div', {}, `Recall ${pct(recall)}`, el('small', { class: 'muted' }, ` · ${value.found} de ${value.truth} caixas encontradas`)), el('div', {}, `Precisão ${pct(precision)}`, el('small', { class: 'muted' }, ` · ${value.correct} de ${value.suggestions} sugestões certas`)))
    })));
  $('compareTable').replaceChildren(el('table', {}, el('thead', {}, head), el('tbody', {}, ...rows)),
    el('p', { class: 'muted small-text', style: 'padding:8px 12px' }, `${comparison.dataset ? NAMING.dataset(datasetFolder(comparison.dataset)) : 'Dataset —'} · confiança mínima ${pct(comparison.conf)} · uma sugestão conta como certa com sobreposição (IoU) ≥ ${comparison.iou_match}.`))
}
/**
 * Tabela de cobertura: caixas por classe e por divisão da execução escolhida.
 * Parâmetros: nenhum (lê #coverageRun). Retorna: nada.
 */
function renderCoverage() {
  const run = OPS.runs.find(value => value.id === $('coverageRun').value);
  if (!run) { $('coverageTable').replaceChildren(el('div', { class: 'empty' }, 'Sem contagens por classe registradas.')); return }
  const splits = Object.keys(run.class_counts);
  const classes = [...new Set(splits.flatMap(split => Object.keys(run.class_counts[split])))];
  // class_counts segue a ordem dos IDs da taxonomia (mesma ordem de DATA.classes).
  // ATENCAO: o rótulo é achado pela POSIÇÃO do slug na lista de classes presentes, e não pelo slug. Se alguma
  // classe da taxonomia não aparecer em nenhuma divisão, as posições se deslocam e o rótulo fica errado.
  const label = slug => (DATA.classes[classes.indexOf(slug)] || {}).label || slug;
  const rows = classes.map(slug => el('tr', {}, el('td', {}, label(slug)), ...splits.map(split => { const value = run.class_counts[split][slug] || 0; return el('td', { class: 'num ' + (value === 0 ? 'cell-bad' : value < 5 ? 'cell-warn' : '') }, String(value)) })));
  $('coverageTable').replaceChildren(el('table', {}, el('thead', {}, el('tr', {}, el('th', {}, 'Classe'), ...splits.map(split => el('th', { class: 'num' }, splitLabel(split))))), el('tbody', {}, ...rows)),
    el('p', { class: 'muted small-text', style: 'padding:8px 12px' }, 'Vermelho: sem exemplos (a classe não é aprendida). Amarelo: menos de 5 caixas.'))
}
// ENTRADA: seletores de comparação e de execução da cobertura.
$('compareSelect').onchange = renderCompare;
$('coverageRun').onchange = renderCoverage;

/* ======================= Classes (cadastro) ======================= */
// Elementos do index.html: classesTable, classesHint, classesVersion, classesStatus, classesOffline,
// createClass, newClassName, newClassDisplay, newClassColor, newClassCriterion, reviewer.
// Rotas da API: GET /api/classes, POST /api/classes (proposta nova), PUT /api/classes/<id> (editar ou
// reativar), DELETE /api/classes/<id>?author=... (desativar).
// ATENCAO: IDs de classe são contrato com pesos, labels e revisões: a página nunca apaga nem renumera.
// Aberta do disco, a lista vem das classes embutidas; pelo servidor, de /api/classes (editável).
// CLASSES: resposta de /api/classes ({ classes, summary, taxonomy_version }); null sem servidor.
let CLASSES = null;
// editingClass: id da classe em edição na tabela (null = nenhuma).
let editingClass = null;
/**
 * Linhas da tabela de classes: do servidor, se carregadas; senão derivadas de DATA.classes (só leitura).
 * Parâmetros: nenhum. Retorna: lista de { id, name, display_name, color, active, phase, in_detector, ... }.
 */
function classRows() {
  if (CLASSES) return CLASSES.classes;
  return DATA.classes.map(category => ({ id: category.id, name: category.slug || '', display_name: category.label, color: category.color, active: category.active !== false, phase: 1, in_detector: !category.requires_training, requires_training: Boolean(category.requires_training), source: 'taxonomia', name_editable: false }))
}
/**
 * Chip de situação de uma classe (inativa, no detector, proposta/ativa/catálogo exigindo treino).
 * Parâmetros: entry (linha de classRows). Retorna: elemento <span>.
 */
function classSituation(entry) {
  if (!entry.active) return el('span', { class: 'chip neutral' }, 'inativa');
  if (entry.in_detector) return el('span', { class: 'chip ok' }, 'no detector');
  const title = 'Só passa a ser detectada depois de dados aprovados e novo treino.';
  if (entry.source === 'proposta') return el('span', { class: 'chip warn', title }, 'proposta · exige treino');
  return el('span', { class: 'chip warn', title }, entry.phase === 1 ? 'ativa · exige treino' : 'catálogo · exige treino')
}
/**
 * Chamada genérica às rotas de classes (aceita GET/POST/PUT/DELETE, ao contrário de api()).
 * Parâmetros: method, path, body (opcional). Retorna: Promise com o JSON. Erros: lança Error com a mensagem do servidor.
 */
async function classRequest(method, path, body) {
  const response = await fetch(path, { method, headers: { 'Content-Type': 'application/json', 'X-DroneCamp-Token': TOKEN }, body: body === undefined ? undefined : JSON.stringify(body) });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value
}
/** Carrega CLASSES do servidor (sem servidor não faz nada). Parâmetros: nenhum. Retorna: Promise<void>. */
async function loadClasses() {
  if (!SERVER) return;
  try { CLASSES = await classRequest('GET', '/api/classes') } catch (error) { $('classesStatus').textContent = error.message }
}
/** Mensagem da seção Classes. Parâmetros: text, error (boolean). Retorna: nada. */
function classStatus(text, error = false) { $('classesStatus').textContent = text; $('classesStatus').classList.toggle('error', error) }
/**
 * Salva uma alteração de classe (PUT), recarrega e redesenha.
 * Parâmetros: id (classe), body (campos alterados), done (texto do aviso de sucesso). Retorna: Promise<void>.
 */
async function saveClass(id, body, done) {
  try {
    await classRequest('PUT', `/api/classes/${id}`, { ...body, author: $('reviewer').value || undefined });
    await loadClasses(); editingClass = null; renderClasses();
    toast('Classe atualizada', done)
  } catch (error) { classStatus(error.message, true) }
}
/**
 * Desenha a tabela de classes (modo leitura ou edição em linha) e o resumo.
 * Parâmetros: nenhum. Retorna: nada. Chamada por: showView('classes'), saveClass, botões da tabela.
 */
function renderClasses() {
  const rows = classRows();
  $('classesOffline').classList.toggle('hidden', SERVER);
  $('classesHint').textContent = CLASSES ? `${CLASSES.summary.in_detector} no detector atual · ${CLASSES.summary.requires_training} exigem treino · ${CLASSES.summary.inactive} inativas` : `${rows.length} classes embutidas nesta página`;
  $('classesVersion').textContent = CLASSES ? CLASSES.taxonomy_version || '' : '';
  $('createClass').disabled = !SERVER;
  const canEdit = SERVER && CLASSES;
  // Uma linha por classe: id, nome (com cor), situação e ações.
  $('classesTable').replaceChildren(...rows.map(entry => {
    const editing = canEdit && editingClass === entry.id;
    const name = el('div', { class: 'class-name', style: `--c:${entry.color}` });
    let displayInput, colorInput, slugInput;
    if (editing) {
      colorInput = el('input', { type: 'color', value: entry.color, 'aria-label': 'Cor da classe' });
      displayInput = el('input', { type: 'text', value: entry.display_name, maxlength: '80', 'aria-label': 'Nome de exibição' });
      name.append(colorInput, displayInput)
    } else name.append(el('span', { class: 'swatch' }), el('span', {}, entry.display_name));
    // Nome técnico (slug) sob o nome de exibição; editável só nas classes propostas (os pesos conferem o slug da taxonomia).
    if (editing && entry.name_editable) slugInput = el('input', { type: 'text', value: entry.name, maxlength: '60', 'aria-label': 'Nome técnico', class: 'mono' });
    const nameCell = el('td', {}, name, el('div', { class: 'class-slug' }, slugInput || el('span', { class: 'mono' }, entry.name)));
    // Ações: em edição = Salvar/Cancelar; senão Editar e Desativar/Reativar (só com servidor).
    // EXTENSAO: campo editável novo = input no bloco "if (editing)" acima, chave em body do Salvar e no PUT do servidor.
    const actions = el('div', { class: 'class-actions' });
    if (editing) {
      actions.append(
        el('button', { type: 'button', class: 'small primary', onclick: () => {
          const body = { display_name: displayInput.value.trim(), color: colorInput.value.toUpperCase() };
          if (slugInput) body.name = slugInput.value.trim();
          saveClass(entry.id, body, `${body.display_name} salva. Mudanças nas classes exigem novo ciclo de treinamento para ter efeito nos modelos.`)
        } }, 'Salvar'),
        el('button', { type: 'button', class: 'small', onclick: () => { editingClass = null; renderClasses() } }, 'Cancelar'))
    } else if (canEdit) {
      actions.append(el('button', { type: 'button', class: 'small', onclick: () => { editingClass = entry.id; renderClasses() } }, 'Editar'));
      if (entry.active) actions.append(el('button', { type: 'button', class: 'small bad', onclick: async () => {
        if (!confirm(`Desativar a classe "${entry.display_name}"? Ela some das opções de caixa nova; nada é apagado e pode ser reativada.`)) return;
        try { await classRequest('DELETE', `/api/classes/${entry.id}` + ($('reviewer').value ? '?author=' + encodeURIComponent($('reviewer').value) : '')); await loadClasses(); renderClasses(); toast('Classe desativada', `${entry.display_name} ficou inativa.`) } catch (error) { classStatus(error.message, true) }
      } }, 'Desativar'));
      else actions.append(el('button', { type: 'button', class: 'small', onclick: () => saveClass(entry.id, { active: true }, `${entry.display_name} reativada.`) }, 'Reativar'))
    }
    return el('tr', { class: entry.active ? '' : 'inactive' },
      el('td', { class: 'num mono' }, String(entry.id)),
      nameCell,
      el('td', {}, classSituation(entry)),
      el('td', {}, actions))
  }))
}
/**
 * Confere o formulário de classe nova: nome técnico em snake_case começando por letra e nome de exibição.
 * Parâmetros: nenhum. Retorna: boolean.
 * ATENCAO: a mesma regra do nome técnico é validada de novo no servidor.
 */
function newClassValid() { return SERVER && /^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$/.test($('newClassName').value) && $('newClassDisplay').value.trim().length > 0 }
// ENTRADA: digitar no formulário de classe nova habilita/desabilita o botão; o clique envia a proposta.
['newClassName', 'newClassDisplay'].forEach(id => $(id).oninput = () => { $('createClass').disabled = !newClassValid() });
$('createClass').onclick = async () => {
  if (!newClassValid()) return;
  try {
    const created = await classRequest('POST', '/api/classes', { name: $('newClassName').value, display_name: $('newClassDisplay').value.trim(), color: $('newClassColor').value.toUpperCase(), criterion: $('newClassCriterion').value.trim() || undefined, author: $('reviewer').value || undefined });
    $('newClassName').value = ''; $('newClassDisplay').value = ''; $('newClassCriterion').value = ''; $('createClass').disabled = true;
    await loadClasses(); renderClasses();
    classStatus(`Classe ${created.id} (${created.display_name}) registrada como proposta. Mudanças nas classes exigem novo ciclo de treinamento para ter efeito nos modelos.`);
    toast('Classe adicionada', `${created.display_name} aguarda fotos aprovadas e novo treino.`)
  } catch (error) { classStatus(error.message, true) }
};

/* ======================= Assistente IA ======================= */
// Mesma estrutura do painel de IA do NetInventory: três submenus (Configuração da IA, Uso e limites, Chat)
// numa só seção, com abas no topo e subitens no menu lateral.
// Elementos do index.html: aiOffline, aiTabs, aiState, aiLocked, aiKey, aiKeyLabel, aiModel, aiModelList,
// aiModelHelp, aiListModels, aiSave, aiRemoveKey, aiSettingsStatus, aiUsagePeriod, aiUsageRefresh,
// aiProviderLine, aiProviderStatus, aiProviderKpis, aiProviderDetails, aiProviderChecked, aiDashboard,
// aiUsageScope, aiUsageStatus, aiUsageKpis, aiUsageTable, aiUsageFoot, aiLog, aiChatError, aiChatForm,
// aiDraft, aiSend, aiChatClear.
// Rotas da API: GET /api/ai/status, GET /api/ai/usage?period=; POST /api/ai/settings, /api/ai/key/delete,
// /api/ai/models, /api/ai/provider-usage e /api/ai/chat (platform_server.py → ai_panel.py / ai_usage.py).
// ATENCAO: a chave digitada só vai no POST de /api/ai/settings e o campo é limpo em seguida; o servidor
// nunca a devolve. A conversa fica só em AI.chat (memória desta aba): nada vai para o localStorage.
// AI_VIEWS: chave de VIEWS → data-panel do painel mostrado. EXTENSAO: submenu novo = entrada aqui.
const AI_VIEWS = { 'ai-settings': 'ai-settings', 'ai-usage': 'ai-usage', 'ai-chat': 'ai-chat' };
// AI_FEATURES: nome legível do recurso que chamou o modelo (mesma tabela de ai_usage.FEATURES).
const AI_FEATURES = { chat: 'Chat do assistente', describe: 'Descrição de ocorrências', 'connection-test': 'Teste da chave' };
// Perguntas prontas do chat (botões do estado vazio). EXTENSAO: pergunta nova = texto aqui.
const AI_SUGGESTIONS = [
  'Quais classes têm menos caixas em fotos confirmadas na inspeção aberta?',
  'Quantas sugestões do modelo ainda estão pendentes e em quais classes?',
  'Resuma os treinos e diga qual comparação está registrada.',
  'Quais inspeções ainda têm fotos sem decisão?'
];
// AI: estado do painel. status = /api/ai/status; usage/provider = consultas do Uso e limites;
// chat = conversa em memória (messages, pending, error), mantida ao trocar de submenu.
const AI = { status: null, usage: null, provider: null, providerError: null, usageError: null, loading: false, chat: { messages: [], pending: false, error: null } };
// Números e valores em USD no padrão brasileiro; ausente vira "Não informado" (nunca zero inventado).
const aiNumber = value => value == null ? 'Não informado' : Number(value).toLocaleString('pt-BR', { maximumFractionDigits: 0 });
const aiMoney = value => value == null ? 'Não informado' : Number(value).toLocaleString('pt-BR', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 6 });
/**
 * Abre um submenu do Assistente IA: marca a aba, mostra o painel e carrega os dados dele.
 * Parâmetros: view ('ai-settings', 'ai-usage' ou 'ai-chat'). Retorna: nada. Chamada por: showView.
 */
function renderAssistant(view) {
  $('aiOffline').classList.toggle('hidden', SERVER);
  document.querySelectorAll('#aiTabs [data-view]').forEach(button => button.setAttribute('aria-current', button.dataset.view === view ? 'page' : 'false'));
  document.querySelectorAll('.ai-panel').forEach(panel => panel.classList.toggle('active', panel.dataset.panel === AI_VIEWS[view]));
  if (view === 'ai-settings') renderAiSettings();
  if (view === 'ai-usage') renderAiUsage();
  if (view === 'ai-chat') renderAiChat();
  if (SERVER && !AI.status) loadAiStatus()
}
/** Lê /api/ai/status e redesenha o submenu aberto. Parâmetros: nenhum. Retorna: Promise<void>. */
async function loadAiStatus() {
  try { AI.status = await api('/api/ai/status') } catch (error) { AI.status = { error: error.message } }
  if (VIEW_SECTION[state.view] === 'assistant') {
    if (state.view === 'ai-settings') renderAiSettings();
    if (state.view === 'ai-chat') renderAiChat();
    if (state.view === 'ai-usage') renderAiProvider()
  }
}
/** Texto de estado ("● Ativa — OpenRouter, chave terminada em …"), sem nunca mostrar a chave. */
function aiStateLine(status) {
  if (!SERVER) return [el('span', { class: 'off' }, '● Indisponível.'), ' Abra a plataforma pelo servidor local.'];
  if (!status) return ['Verificando…'];
  if (status.error && !status.configured) return [el('span', { class: 'off' }, '● Desativada.'), ` ${status.error}`];
  if (!status.configured) return [el('span', { class: 'off' }, '● Desativada.'), ' Os recursos de IA ficam indisponíveis até você cadastrar a chave (crie em ', el('a', { href: status.key_help_url, target: '_blank', rel: 'noreferrer' }, 'openrouter.ai/keys'), ').'];
  const source = status.key_source === 'ambiente' ? 'variável de ambiente do sistema' : 'arquivo .env deste PC';
  return [el('span', { class: 'on' }, '● Ativa'), ` — ${status.provider_name}, chave terminada em `, el('code', {}, `…${status.key_hint}`), ` (${source}), modelo `, el('code', {}, status.model), '.']
}
/** Submenu Configuração da IA: estado, campos e botões conforme a origem da chave. */
function renderAiSettings() {
  const status = AI.status;
  $('aiState').replaceChildren(...aiStateLine(status));
  const locked = Boolean(status && status.key_locked);
  $('aiLocked').classList.toggle('hidden', !locked);
  $('aiLocked').textContent = locked ? 'A chave vem da variável de ambiente OPENROUTER_API_KEY do Windows e vence o arquivo .env. Para trocá-la ou removê-la, altere a variável do sistema; aqui dá para escolher o modelo.' : '';
  const keeps = Boolean(status && status.configured);
  $('aiKeyLabel').textContent = locked ? 'Chave da API (definida no sistema)' : keeps ? 'Nova chave (deixe vazio para manter)' : 'Chave da API';
  $('aiKey').disabled = !SERVER || locked;
  // O modelo só é preenchido na primeira vez (não apaga o que o usuário está digitando).
  if (status && status.model && !$('aiModel').dataset.touched) $('aiModel').value = status.model;
  // OPENROUTER_MODEL no ambiente do sistema vence o .env: o campo fica travado (o servidor também recusa).
  $('aiModel').disabled = !SERVER || Boolean(status && status.model_locked);
  $('aiListModels').disabled = !SERVER || !keeps;
  // Com chave malformada no .env a IA fica desativada, mas a chave ainda pode ser removida.
  $('aiRemoveKey').classList.toggle('hidden', !(status && status.key_source === '.env'));
  updateAiSave()
}
/** Habilita "Testar e salvar" só com modelo preenchido e chave disponível (nova ou já cadastrada). */
function updateAiSave() {
  const status = AI.status || {};
  $('aiSave').disabled = !SERVER || AI.loading || !$('aiModel').value.trim() || (!status.configured && !$('aiKey').value.trim())
}
/** Mensagem do submenu Configuração da IA. Parâmetros: text, error (boolean). */
function aiSettingsStatus(text, error = false) { $('aiSettingsStatus').textContent = text; $('aiSettingsStatus').classList.toggle('error', error); $('aiSettingsStatus').style.color = error ? 'var(--bad)' : '' }
// ENTRADA: campos e botões da Configuração da IA.
$('aiKey').oninput = updateAiSave;
$('aiModel').oninput = () => { $('aiModel').dataset.touched = '1'; updateAiSave() };
$('aiSave').onclick = async () => {
  AI.loading = true; updateAiSave();
  $('aiSave').lastElementChild.textContent = 'Testando…';
  aiSettingsStatus('Testando a chave e o modelo no provedor (1 token)…');
  try {
    AI.status = await api('/api/ai/settings', { api_key: $('aiKey').value.trim() || undefined, model: $('aiModel').value.trim() });
    $('aiKey').value = '';
    delete $('aiModel').dataset.touched;
    AI.usage = null; AI.provider = null;
    aiSettingsStatus(`Configuração salva. Respondeu: ${AI.status.tested_model || AI.status.model}.`);
    toast('Configuração da IA salva', 'Chave testada no provedor e gravada no .env deste PC.')
  } catch (error) { aiSettingsStatus(error.message, true) }
  finally { AI.loading = false; $('aiSave').lastElementChild.textContent = 'Testar e salvar'; renderAiSettings() }
};
$('aiRemoveKey').onclick = async () => {
  if (!confirm('Remover a chave do arquivo .env? Os recursos de IA ficam desativados até cadastrar outra.')) return;
  try { AI.status = await api('/api/ai/key/delete', {}); AI.provider = null; aiSettingsStatus('Chave removida do .env.'); toast('Chave removida', 'Os recursos de IA estão desativados.') }
  catch (error) { aiSettingsStatus(error.message, true) }
  renderAiSettings()
};
$('aiListModels').onclick = async () => {
  $('aiListModels').disabled = true; aiSettingsStatus('Consultando modelos gratuitos com entrada de imagem…');
  try {
    const { models } = await api('/api/ai/models', {});
    $('aiModelList').replaceChildren(...models.map(model => el('option', { value: model.id }, model.name || model.id)));
    aiSettingsStatus(`${models.length} modelos gratuitos com visão. Clique no campo Modelo para escolher.`)
  } catch (error) { aiSettingsStatus(error.message, true) }
  finally { $('aiListModels').disabled = !SERVER || !(AI.status && AI.status.configured) }
};

/** Cartão de métrica (mesmo componente .kpi do Início). Parâmetros: label, value, hint, long (valor longo). */
function aiMetric(label, value, hint, long = false) {
  return el('div', { class: 'card kpi' }, el('div', { class: 'label' }, label), el('div', { class: long ? 'value long' : 'value' }, value), el('div', { class: 'hint', title: hint }, hint))
}
/** Submenu Uso e limites: dispara as duas consultas (consumo local e provedor) e desenha o que já tem. */
function renderAiUsage() {
  renderAiProvider(); renderAiLocal();
  if (SERVER && !AI.usage && !AI.usageError) loadAiUsage();
  if (SERVER && !AI.provider && !AI.providerError) loadAiProvider()
}
/** GET /api/ai/usage com o período escolhido. Retorna: Promise<void>. */
async function loadAiUsage() {
  $('aiUsageStatus').textContent = 'Carregando métricas…';
  try { AI.usage = await api('/api/ai/usage?period=' + encodeURIComponent($('aiUsagePeriod').value)); AI.usageError = null }
  catch (error) { AI.usage = null; AI.usageError = error.message }
  renderAiLocal()
}
/** POST /api/ai/provider-usage (consulta o OpenRouter com a chave, sem inferência). Retorna: Promise<void>. */
async function loadAiProvider() {
  $('aiProviderStatus').textContent = 'Consultando o provedor…';
  try { AI.provider = await api('/api/ai/provider-usage', {}); AI.providerError = null }
  catch (error) { AI.provider = null; AI.providerError = error.message }
  renderAiProvider()
}
/** Cartão "Créditos e limites do provedor". */
function renderAiProvider() {
  const status = AI.status, provider = AI.provider;
  $('aiProviderLine').replaceChildren(status ? status.provider_name || 'OpenRouter' : 'Carregando provedor…', status && status.model ? el('span', { class: 'mono', style: 'display:block;margin-top:2px' }, status.model) : '');
  $('aiProviderStatus').textContent = !SERVER ? 'Disponível só com o servidor local.' : AI.providerError || (provider ? provider.message : 'Consultando o provedor…');
  $('aiProviderStatus').style.color = AI.providerError || (provider && provider.state === 'error') ? 'var(--bad)' : '';
  const available = Boolean(provider && provider.state === 'available');
  $('aiProviderKpis').classList.toggle('hidden', !available);
  $('aiProviderDetails').classList.toggle('hidden', !available);
  if (available) {
    $('aiProviderKpis').replaceChildren(
      aiMetric('Créditos restantes na conta', aiMoney(provider.account_balance), 'Saldo da conta no provedor, em USD.', true),
      aiMetric('Disponível no teto da chave', aiMoney(provider.key_remaining), provider.key_limit == null ? 'Chave sem teto definido ou teto não informado.' : `Teto da chave: ${aiMoney(provider.key_limit)}.`, true),
      aiMetric('Consumo acumulado da chave', aiMoney(provider.key_usage), 'Inclui uso da mesma chave fora deste PC.', true),
      aiMetric('Camada gratuita', provider.is_free_tier == null ? 'Não informado' : provider.is_free_tier ? 'Sim' : 'Não', 'Conta sem créditos comprados usa os limites diários dos modelos :free.', true));
    const resets = { daily: 'Diária', weekly: 'Semanal', monthly: 'Mensal' };
    $('aiProviderDetails').replaceChildren(
      el('div', {}, el('dt', {}, 'Consumo da chave hoje (UTC)'), el('dd', {}, aiMoney(provider.daily_usage))),
      el('div', {}, el('dt', {}, 'Consumo da chave no mês (UTC)'), el('dd', {}, aiMoney(provider.monthly_usage))),
      el('div', {}, el('dt', {}, 'Renovação do teto da chave'), el('dd', {}, resets[provider.reset] || 'Não informada')))
  }
  $('aiProviderChecked').textContent = provider && provider.checked_at ? `Consulta em ${fmtDate(provider.checked_at)}. O filtro de período abaixo vale só para o consumo local.` : '';
  const dashboard = status && status.dashboard_url;
  $('aiDashboard').classList.toggle('hidden', !dashboard);
  if (dashboard) $('aiDashboard').href = dashboard
}
/** Métricas e tabela "Consumo nesta instalação". */
function renderAiLocal() {
  const data = AI.usage;
  $('aiUsageScope').textContent = `${$('aiUsagePeriod').selectedOptions[0].textContent}. Inclui descrição de ocorrências, chat e testes da chave feitos neste PC.`;
  $('aiUsageStatus').textContent = !SERVER ? 'Disponível só com o servidor local.' : AI.usageError ? `Não foi possível carregar as métricas: ${AI.usageError}` : data ? '' : 'Carregando métricas…';
  if (!data) { $('aiUsageKpis').replaceChildren(); $('aiUsageTable').replaceChildren(el('tr', {}, el('td', { colspan: 5 }, el('div', { class: 'empty' }, 'Sem dados.')))); $('aiUsageFoot').textContent = ''; return }
  const seconds = data.average_duration_ms == null ? 'Não informado' : `${(data.average_duration_ms / 1000).toLocaleString('pt-BR', { maximumFractionDigits: 2 })} s`;
  $('aiUsageKpis').replaceChildren(
    aiMetric('Tokens gastos', aiNumber(data.total_tokens), `${aiNumber(data.input_tokens)} de entrada · ${aiNumber(data.output_tokens)} de saída`),
    aiMetric('Chamadas ao modelo', aiNumber(data.requests), `${aiNumber(data.successful)} concluídas · ${aiNumber(data.errors)} com falha`),
    aiMetric('Custo informado', aiMoney(data.cost_usd), `${aiNumber(data.cost_reports)} de ${aiNumber(data.requests)} chamadas informaram custo.`, true),
    aiMetric('Tempo médio por chamada', seconds, `${aiNumber(data.token_reports)} de ${aiNumber(data.requests)} chamadas informaram tokens.`, true));
  $('aiUsageTable').replaceChildren(...(data.features.length ? data.features.map(row => el('tr', {},
    el('td', {}, el('b', {}, AI_FEATURES[row.feature] || row.feature_label || row.feature), el('small', { class: 'mono', style: 'display:block;color:var(--muted);margin-top:2px;overflow-wrap:anywhere' }, `${row.provider} · ${row.model}`)),
    el('td', { class: 'num' }, aiNumber(row.events)), el('td', { class: 'num' }, aiNumber(row.errors)),
    el('td', { class: 'num' }, aiNumber(row.total_tokens)), el('td', { class: 'num' }, aiMoney(row.cost_usd))))
    : [el('tr', {}, el('td', { colspan: 5 }, el('div', { class: 'empty' }, 'Nenhum uso registrado neste período. A coleta começa nesta versão da plataforma.')))]));
  $('aiUsageFoot').textContent = `${data.tracking_since ? `Coleta iniciada em ${fmtDate(data.tracking_since)}.` : 'A coleta começa na primeira chamada à IA depois desta atualização.'} Não há recuperação do consumo anterior. Tokens e custos que o provedor não informa aparecem como "Não informado"; totais podem ser parciais. O registro (.runtime/ai_usage.jsonl) guarda só metadados: nenhuma pergunta, resposta, foto ou chave.`
}
// ENTRADA: filtro de período e botão Atualizar do Uso e limites.
$('aiUsagePeriod').onchange = () => { AI.usage = null; AI.usageError = null; renderAiLocal(); if (SERVER) loadAiUsage() };
$('aiUsageRefresh').onclick = () => { if (!SERVER) return; AI.usageError = null; AI.providerError = null; loadAiUsage(); loadAiProvider() };

/**
 * Markdown mínimo das respostas (parágrafos, listas, **negrito**, `código`) montado com el():
 * nunca usa innerHTML, então texto malicioso vindo do modelo não vira script na página.
 * Parâmetros: text (resposta do modelo). Retorna: lista de nós.
 */
function aiMarkdown(text) {
  const inline = line => line.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean).map(part =>
    part.startsWith('**') && part.endsWith('**') ? el('strong', {}, part.slice(2, -2)) : part.startsWith('`') && part.endsWith('`') ? el('code', {}, part.slice(1, -1)) : part);
  const blocks = []; let list = null;
  String(text).split(/\r?\n/).forEach(raw => {
    const line = raw.trim();
    const bullet = /^(?:[-*•]|\d+[.)])\s+(.*)$/.exec(line);
    if (bullet) { if (!list) { list = el('ul'); blocks.push(list) } list.append(el('li', {}, inline(bullet[1]))); return }
    list = null;
    if (!line) return;
    const heading = /^#{1,4}\s+(.*)$/.exec(line);
    blocks.push(heading ? el('p', {}, el('strong', {}, inline(heading[1]))) : el('p', {}, inline(line)))
  });
  return blocks
}
/** Submenu Chat: conversa em memória, sugestões no estado vazio e indicador de resposta. */
function renderAiChat() {
  const chat = AI.chat, ready = SERVER && AI.status && AI.status.configured;
  const nodes = [];
  if (!chat.messages.length) {
    nodes.push(el('div', { class: 'ai-empty' },
      el('p', {}, !SERVER ? 'O chat precisa do servidor local.' : !AI.status ? 'Verificando a configuração da IA…' : ready ? 'Comece com uma pergunta sobre o projeto ou escolha uma sugestão:' : 'IA desativada. Cadastre a chave em Configuração da IA para conversar.'),
      ready ? el('div', { class: 'ai-suggestions' }, AI_SUGGESTIONS.map(text => el('button', { type: 'button', class: 'small', onclick: () => sendAiMessage(text) }, text)))
        : SERVER && AI.status ? el('div', {}, el('button', { type: 'button', class: 'small', 'data-goto': 'ai-settings', onclick: () => showView('ai-settings') }, 'Abrir Configuração da IA')) : null))
  }
  chat.messages.forEach(message => nodes.push(message.role === 'user'
    ? el('div', { class: 'ai-msg user' }, message.content)
    : el('div', { class: 'ai-msg assistant' }, aiMarkdown(message.content), message.model ? el('span', { class: 'meta' }, `Resposta da IA · ${message.model}`) : null)));
  if (chat.pending) nodes.push(el('div', { class: 'ai-msg assistant pending' }, el('span', { class: 'dots', 'aria-hidden': 'true' }, el('i'), el('i'), el('i')), 'Consultando os dados da plataforma…'));
  $('aiLog').replaceChildren(...nodes);
  $('aiLog').scrollTop = $('aiLog').scrollHeight;
  $('aiChatError').textContent = chat.error || '';
  $('aiChatError').classList.toggle('hidden', !chat.error);
  $('aiDraft').disabled = !ready || chat.pending;
  $('aiSend').disabled = !ready || chat.pending;
  $('aiChatClear').classList.toggle('hidden', !chat.messages.length);
  $('aiChatClear').disabled = chat.pending
}
/**
 * Envia a pergunta com a conversa inteira; o servidor junta o instantâneo do projeto e responde.
 * Parâmetros: text (pergunta). Retorna: Promise<void>.
 */
async function sendAiMessage(text) {
  const chat = AI.chat, question = String(text || '').trim();
  if (!question || chat.pending || !(AI.status && AI.status.configured)) return;
  chat.messages.push({ role: 'user', content: question });
  chat.pending = true; chat.error = null; $('aiDraft').value = '';
  renderAiChat();
  try {
    const result = await api('/api/ai/chat', { review: REVIEWS ? REVIEWS.current : DATA.version, messages: chat.messages.map(({ role, content }) => ({ role, content })) });
    chat.messages.push({ role: 'assistant', content: result.answer, model: result.model_used })
  } catch (error) { chat.error = error.message }
  finally { chat.pending = false; AI.usage = null; renderAiChat(); if (state.view === 'ai-chat') $('aiDraft').focus() }
}
// ENTRADA: envio da pergunta (Enter ou botão) e limpeza da conversa.
$('aiChatForm').onsubmit = event => { event.preventDefault(); sendAiMessage($('aiDraft').value) };
$('aiChatClear').onclick = () => { AI.chat.messages = []; AI.chat.error = null; renderAiChat() };

/* ======================= Servidor local (opcional) ======================= */
// Elementos do index.html: modeChip, navInspections, liveDot, jobLog, startTrain, startSuggest, stopJob,
// suggestWeights, suggestConf.
// Rotas da API: GET /api/workspace, GET /api/operations, GET /api/jobs, POST /api/jobs, POST /api/jobs/stop.
// EXTENSAO: chamada nova ao servidor = api('/api/...') para ler (GET) ou api('/api/...', corpo)
// para alterar (POST com o token). A rota correspondente fica em platform_server.py (do_GET/do_POST).
// ATENCAO: aberta do disco (SERVER falso) não há servidor; teste SERVER antes de chamar.
/**
 * Chamada ao servidor local: sem corpo = GET; com corpo = POST JSON com o token.
 * Parâmetros: path (ex.: '/api/jobs'), body (objeto opcional). Retorna: Promise com o JSON da resposta.
 * Erros: lança Error com o campo error da resposta (ou "HTTP <código>").
 */
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-DroneCamp-Token': TOKEN }, body: JSON.stringify(body) });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value
}
/** Atualiza o chip de conexão com o servidor. Parâmetros: connected (boolean). Retorna: nada. */
function setMode(connected) {
  $('modeChip').className = 'status ' + (connected ? 'ok' : 'bad');
  $('modeChip').lastElementChild.textContent = connected ? 'Servidor local conectado' : 'Servidor local desconectado'
}
/**
 * Recarrega o espaço de trabalho (lista de inspeções e capacidades), o seletor e o contador do menu.
 * Parâmetros: nenhum. Retorna: Promise<void>. Chamada por: inicialização, a cada 30 s e após importar mídia.
 * Erros são ignorados (a próxima rodada tenta de novo).
 */
async function refreshWorkspace() {
  if (!SERVER) return;
  try {
    WORKSPACE = await api('/api/workspace' + (DATA.version ? '?review=' + encodeURIComponent(DATA.version) : ''));
    fillReviewPicker(WORKSPACE.reviews.filter(review => review.page_ready !== false).map(review => ({ name: review.name, display_name: review.display_name, href: review.href, images: review.images })), DATA.version);
    $('navInspections').textContent = WORKSPACE.reviews.length || '';
    if (state.view === 'home' || state.view === 'inspections') showView(state.view)
  } catch {}
}
/**
 * Recarrega o instantâneo de operações e a tarefa atual e redesenha a seção aberta.
 * Parâmetros: nenhum. Retorna: Promise<void>. Chamada por: inicialização e a cada 5 s.
 * Falha marca o chip como "desconectado".
 */
async function refreshOperations() {
  try {
    OPS = await api('/api/operations' + (DATA.version ? '?review=' + encodeURIComponent(DATA.version) : ''));
    job = await api('/api/jobs');
    // EXTENSAO: seção nova que dependa de OPS = mais um "else if (state.view === ...)" aqui.
    if (state.view === 'monitor') renderMonitor();
    else if (state.view === 'training') renderTraining();
    else if (state.view === 'models') renderModels();
    else if (state.view === 'home') renderHome();
    renderJob();
    $('liveDot').classList.toggle('hidden', !OPS.runs.some(value => value.state === 'running') && !jobRunning());
    setMode(true)
  } catch { setMode(false) }
}
/**
 * Inicia uma tarefa no servidor (uma por vez) e abre o Monitoramento.
 * Parâmetros: kind ('train-pilot' ou 'suggest'), options (objeto repassado ao servidor). Retorna: Promise<void>.
 */
async function startJob(kind, options) {
  try {
    job = await api('/api/jobs', { kind, options });
    renderJob(); showView('monitor');
    toast('Tarefa iniciada', `${kind} rodando no servidor local. Acompanhe o log e as curvas.`)
  } catch (error) { message(error.message, true); $('jobLog').textContent = error.message; alert(error.message) }
}
// ENTRADA: botão "Iniciar treino" (pede confirmação; o treino ocupa o PC até terminar).
$('startTrain').onclick = () => {
  const options = trainOptions();
  if (!confirm(`Iniciar treino piloto com ${options.epochs || '?'} épocas em ${OPS.device || 'cpu'}? O processo usa o PC até terminar.`)) return;
  startJob('train-pilot', options)
};
// ENTRADA: botão "Gerar sugestões" (pesos e confiança do formulário do Monitoramento).
$('startSuggest').onclick = () => startJob('suggest', { review: REVIEWS ? REVIEWS.current : DATA.version, weights: $('suggestWeights').value, conf: $('suggestConf').value === '' ? null : Number($('suggestConf').value) });
// ENTRADA: botão "Interromper tarefa".
$('stopJob').onclick = async () => {
  if (!confirm('Interromper a tarefa em andamento?')) return;
  try { job = await api('/api/jobs/stop', {}); renderJob() } catch (error) { alert(error.message) }
};
$('suggestWeights').onchange = renderJob;

/* ======================= Inicialização ======================= */
// Elementos do index.html: themeToggle, versionLabel, classId, confFilter, confValue, taxonomyNotice,
// reviewEmpty, reviewGrid, aiNotice, export. Rotas da API (só com servidor): as do bloco Servidor local.
// ENTRADA: daqui para baixo roda uma vez quando a página abre (tema, classes, primeira foto,
// seção inicial e, com servidor, consultas a cada 5 s/30 s). Erro de JavaScript aparece no
// console do navegador (F12 → Console) com a linha deste arquivo dentro do index.html.
/**
 * Aplica o tema ('light' ou 'dark'), guarda a escolha e redesenha (os gráficos usam variáveis de cor).
 * Parâmetros: theme. Retorna: nada. Chamada por: botão #themeToggle.
 */
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('dronecamp-theme', theme) } catch {}
  $('themeToggle').replaceChildren(icon(theme === 'dark' ? 'i-sun' : 'i-moon'), el('span', {}, theme === 'dark' ? 'Claro' : 'Escuro'));
  if (state.view !== 'review') showView(state.view); else paint()
}
// ENTRADA: botão de tema.
$('themeToggle').onclick = () => applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
// Tema salvo aplicado já na abertura (sem redesenhar, porque nada foi desenhado ainda).
(() => { let theme = 'light'; try { theme = localStorage.getItem('dronecamp-theme') || 'light' } catch {} document.documentElement.dataset.theme = theme; $('themeToggle').replaceChildren(icon(theme === 'dark' ? 'i-sun' : 'i-moon'), el('span', {}, theme === 'dark' ? 'Claro' : 'Escuro')) })();

// 1. nome da inspeção no topo.
$('versionLabel').textContent = DATA.version ? reviewLabel(DATA.version, DATA.display_name) : '';
$('versionLabel').title = DATA.version || '';
// Inativas continuam na lista (desabilitadas) para que uma caixa existente dessa classe mostre e preserve a categoria.
// 2. opções de categoria do editor de caixas.
DATA.classes.forEach(category => $('classId').append(el('option', category.active === false ? { value: category.id, disabled: 'disabled' } : { value: category.id }, category.active === false ? `${category.label} (inativa)` : category.label)));
// 3. limite de confiança: começa (e não desce abaixo) da confiança usada ao gerar as sugestões.
if (DATA.suggestions) {
  $('confFilter').value = Math.round(DATA.suggestions.conf * 100);
  $('confFilter').min = Math.round(DATA.suggestions.conf * 100);
  $('confValue').textContent = `${$('confFilter').value}%`
}
// 4. aviso de taxonomia nova a conferir.
if (DATA.taxonomy_recheck_pending) {
  // Os nomes vêm do catálogo e do histórico; não manter categorias antigas fixas no HTML.
  const categories = (DATA.classes_requiring_review || DATA.classes.map(value => value.label)).join('; ');
  $('taxonomyNotice').textContent = `Categorias que ainda precisam de conferência integral: ${categories}. Confira essas ocorrências em todas as fotos antes de registrar uma aprovação.`;
  $('taxonomyNotice').classList.remove('hidden')
}
// 5. revisão vazia vs. com fotos; abre a primeira foto.
$('reviewEmpty').classList.toggle('hidden', HAS_IMAGES);
$('reviewGrid').classList.toggle('hidden', !HAS_IMAGES);
$('aiNotice').classList.toggle('hidden', !HAS_IMAGES);
$('export').disabled = !HAS_IMAGES;
if (HAS_IMAGES) openPhoto(0);
// 6. seção inicial: hash da URL, senão a última aberta nesta aba, senão Início.
let initialView = location.hash.slice(1);
if (!VIEWS[initialView]) { try { initialView = sessionStorage.getItem('dronecamp-view') || 'home' } catch { initialView = 'home' } }
if (!HAS_IMAGES && initialView === 'review') initialView = SERVER ? 'inspections' : 'home';
showView(initialView);
// ENTRADA: mudança manual do hash (#review etc.) abre a seção correspondente.
addEventListener('hashchange', () => { const view = location.hash.slice(1); if (VIEWS[view] && view !== state.view) showView(view) });
// 7. com servidor: carrega dados ao vivo e agenda as atualizações (operações a cada 5 s, inspeções a cada 30 s).
if (SERVER) { refreshOperations(); refreshWorkspace(); loadClasses().then(() => { if (state.view === 'classes') renderClasses() }); setInterval(refreshOperations, 5000); setInterval(refreshWorkspace, 30000) }
