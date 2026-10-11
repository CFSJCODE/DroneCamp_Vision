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
const DATA = __REVIEW_DATA__;
const $ = id => document.getElementById(id);
const OPS_EMPTY = { runs: [], comparisons: [], datasets: [], weights: [], pilot_defaults: {} };
let OPS = DATA.operations || OPS_EMPTY;
// Token presente = página servida pelo comando platform (monitoramento ao vivo e tarefas).
const TOKEN = (document.querySelector('meta[name="dronecamp-server"]') || {}).content || '';
const SERVER = Boolean(TOKEN) && location.protocol.startsWith('http');
// Revisões abertas pelo servidor: { current, items: [{ name, href, images }] }; null quando aberta do disco.
const REVIEWS = (() => {
  try { return JSON.parse((document.querySelector('meta[name="dronecamp-reviews"]') || {}).content || 'null') } catch { return null }
})();
// Espaço de trabalho (lista de inspeções e capacidades do servidor), carregado de /api/workspace.
let WORKSPACE = null;
const HAS_IMAGES = Array.isArray(DATA.images) && DATA.images.length > 0;
const names = { positive: 'Positiva proposta', negative: 'Negativa proposta', ambiguous: 'Ambígua', excluded: 'Excluída' };
const decisionNames = { positive: 'Há ocorrências', negative: 'Negativa', ambiguous: 'Ambígua', excluded: 'Excluída' };
const storageKey = 'dronecamp-review-' + DATA.registry_sha256;
let saved = {};
try { saved = JSON.parse(localStorage.getItem(storageKey) || '{}') } catch {}
let state = { index: 0, selected: -1, drawing: false, start: null, img: new Image(), drafts: saved.images || {}, reviewer: saved.reviewer || '', view: 'home', run: null };
const clone = value => JSON.parse(JSON.stringify(value));
const item = () => DATA.images[state.index];
const draft = () => state.drafts[item().image_sha256] || { boxes: clone(item().boxes), status: 'pending', notes: item().human_notes || '', confirmed_complete: false };
let active = HAS_IMAGES ? draft() : { boxes: [], status: 'pending', notes: '', confirmed_complete: false };
const classOf = id => DATA.classes.find(value => value.id === id) || { id, label: `classe ${id}`, color: '#64748B' };
const pct = value => value == null ? '—' : `${(value * 100).toFixed(1)}%`;
const fmtBytes = n => n == null ? '—' : n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB` : n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : n >= 1e3 ? `${Math.round(n / 1e3)} kB` : `${n} B`;
// Data legível dd/mm/aaaa hh:mm no fuso do navegador (nunca o carimbo ISO cru).
const fmtDate = value => { if (!value) return '—'; const date = new Date(value); return isNaN(date) ? String(value) : date.toLocaleString('pt-BR', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' }).replace(',', '') };
// Quantas edificações a inspeção cobre, sem expor os códigos internos (building_group).
const buildingCount = n => n ? `${n} edificaç${n > 1 ? 'ões' : 'ão'}` : 'edificação não informada';
/* ======================= Nomes legíveis ======================= */
// Mesmo algoritmo de naming.py (tests/test_naming.py roda os dois e exige resultados iguais).
// Ids e pastas em disco nunca mudam; aqui só se deriva o nome mostrado.
const NAMING = (() => {
  const STAGES = { pilot_train: 'Treino Piloto', train: 'Treino de Produção', tune: 'Ajuste de Hiperparâmetros', export: 'Exportação', export_demo: 'Exportação de Demonstração', predict_demo: 'Predição de Demonstração', autolabel_eval: 'Avaliação da Marcação Automática', calibrator: 'Calibrador', feedback_validation: 'Validação de Revisão', review_validation: 'Validação de Revisão', review_opening_fix: 'Correção de Abertura' };
  const RUN_ID = /^([a-z0-9_]+?)_(\d{8}T\d{6}Z)_([0-9a-f]{8})$/;
  const PLACES = { ceasa: 'CEASA Pavilhão A', ceasa_lado_a: 'CEASA Lado A', ceasa_ladoa: 'CEASA Lado A', internet: 'Fotos da Internet' };
  const WORDS = { revisao: 'Revisão', rev: 'Revisão', prerevisao: 'Pré-revisão', sugestoes: 'Sugestões', humana: 'Humana', corrigida: 'Corrigida', taxonomia: 'Taxonomia', parcial: 'Parcial', piloto: 'Piloto', propostas: 'Propostas', busca: 'Busca', aberta: 'Aberta', modelo: 'Modelo', ladoa: 'Lado A', galpao: 'Galpão', pavilhao: 'Pavilhão', inspecao: 'Inspeção', ceasa: 'CEASA', claude: 'Claude', base: 'Base', demo: 'Demonstração', evidencias: 'Evidências', laudo: 'Laudo' };
  const PHRASES = { humana: 'Revisão Humana', corrigida: 'Revisão Corrigida', sugestoes_modelo: 'Sugestões do Modelo' };
  const CONNECTORS = new Set(['de', 'da', 'do', 'das', 'dos', 'e', 'a', 'o', 'em']);
  const has = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
  const shortHash = value => String(value).slice(0, 8).toUpperCase();
  const isHash = token => token.length >= 8 && /^[0-9a-f]+$/.test(token);
  const isVersion = token => /^v\d+[a-z]?$/.test(token);
  const word = (token, first) => {
    if (has(WORDS, token)) return WORDS[token];
    if (isVersion(token)) return token;
    if (/^s\d+$/.test(token)) return `Semente ${token.slice(1)}`;
    if (/^\d+$/.test(token)) return token;
    const match = /^([a-z]+?)(\d+)$/.exec(token);
    if (match && has(WORDS, match[1])) return `${WORDS[match[1]]} ${match[2]}`;
    if (CONNECTORS.has(token) && !first) return token;
    if (token.length <= 2 && /^[a-z]+$/.test(token)) return token.toUpperCase();
    return token.slice(0, 1).toUpperCase() + token.slice(1)
  };
  const humanize = slug => String(slug || '').trim().toLowerCase().split(/[_\-\s]+/).filter(Boolean).map((token, index) => isHash(token) ? shortHash(token) : word(token, index === 0)).join(' ');
  const splitReview = name => {
    const tokens = String(name).toLowerCase().split('_').filter(Boolean);
    const position = tokens.findIndex(isVersion);
    if (position <= 0) return [null, null, tokens, null];
    const key = tokens.slice(0, position).join('_');
    const rest = tokens.slice(position + 1);
    const digest = rest.length && isHash(rest[rest.length - 1]) ? rest.pop() : null;
    return [has(PLACES, key) ? PLACES[key] : humanize(key), tokens[position], rest, digest]
  };
  const review = (name, display) => {
    if (display && String(display).trim()) return String(display).trim();
    if (!name) return '';
    const [place, version, rest, digest] = splitReview(name);
    if (version == null) return humanize(name);
    const qualifier = PHRASES[rest.join('_')] || rest.map((token, index) => word(token, index === 0)).join(' ');
    const tail = [qualifier, digest ? shortHash(digest) : ''].filter(Boolean).join(' ');
    return tail ? `${place} ${version} · ${tail}` : `${place} ${version}`
  };
  const run = (id, custom) => {
    if (custom && String(custom).trim()) return String(custom).trim();
    const match = RUN_ID.exec(String(id || ''));
    if (!match) return humanize(id || '');
    return `${STAGES[match[1]] || humanize(match[1])} ${shortHash(match[3])}`
  };
  const weights = path => {
    const parts = String(path || '').replace(/\\/g, '/').split('/');
    const owner = parts.find(part => RUN_ID.test(part));
    const file = parts[parts.length - 1] || '';
    const stem = file.replace(/\.[^.]*$/, '');
    if (owner) return `${run(owner)} · ${{ best: 'melhor época', last: 'última época' }[stem] || file}`;
    const family = /^yolo(v?\d+)([nsmlx]?)$/i.exec(stem);
    if (family) return `Modelo Base YOLO${family[1].toUpperCase()}${family[2].toLowerCase()}`;
    return stem ? humanize(stem) : ''
  };
  const day = stamp => /^\d{8}$/.test(stamp) ? `${stamp.slice(6, 8)}/${stamp.slice(4, 6)}/${stamp.slice(0, 4)}` : null;
  const comparison = file => {
    let stem = String(file || '').split('/').pop();
    if (stem.endsWith('.json')) stem = stem.slice(0, -5);
    let tokens = stem.split('_').filter(Boolean);
    if (!tokens.length || !['compare', 'eval'].includes(tokens[0])) return humanize(stem);
    const kind = tokens[0] === 'compare' ? 'Comparação' : 'Avaliação por Classe';
    tokens = tokens[0] === 'compare' ? tokens.slice(1) : tokens[1] === 'por' && tokens[2] === 'classe' ? tokens.slice(3) : tokens.slice(1);
    const date = tokens.length && day(tokens[tokens.length - 1]) ? day(tokens.pop()) : null;
    const hashes = tokens.filter(isHash).map(shortHash);
    const subject = hashes.length ? hashes.join(' × ') : (tokens.includes('vs') ? [] : tokens).map((token, index) => word(token, index === 0)).join(' ');
    return [`${kind} ${subject}`.trim(), date].filter(Boolean).join(' · ')
  };
  const dataset = name => {
    const [place, version, rest, digest] = splitReview(name);
    if (version == null) return `Dataset ${humanize(name)}`;
    const seed = rest.find(token => /^s\d+$/.test(token)) || null;
    const words = rest.filter(token => token !== 'piloto' && token !== seed);
    const kind = rest.includes('piloto') ? 'Dataset Piloto' : 'Dataset';
    const qualifier = words.map((token, index) => word(token, index === 0)).join(' ');
    return [`${kind} ${place} ${version}`, qualifier, seed ? `Semente ${seed.slice(1)}` : '', digest ? shortHash(digest) : ''].filter(Boolean).join(' · ')
  };
  const generated = file => {
    const match = /^(\d{8}T\d{6}Z)_(laudo|evidencias)_/.exec(String(file || ''));
    if (!match) return [humanize(String(file).replace(/\.[^.]*$/, '')), null];
    const s = match[1];
    const iso = `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}T${s.slice(9, 11)}:${s.slice(11, 13)}:${s.slice(13, 15)}+00:00`;
    return [match[2] === 'laudo' ? 'Laudo Técnico' : 'Pacote de Evidências', iso]
  };
  return { shortHash, humanize, review, run, weights, comparison, dataset, generated }
})();
/* ======================= fim dos nomes legíveis ======================= */
const el = (tag, props = {}, ...children) => {
  const node = document.createElement(tag);
  Object.entries(props).forEach(([key, value]) => {
    if (key === 'class') node.className = value;
    else if (key === 'style') node.setAttribute('style', value);
    else if (key.startsWith('on')) node[key] = value;
    else node.setAttribute(key, value)
  });
  children.flat().forEach(child => child != null && node.append(child));
  return node
};
const icon = name => { const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); const use = document.createElementNS('http://www.w3.org/2000/svg', 'use'); use.setAttribute('href', '#' + name); svg.append(use); return svg };

function message(text, error = false) { $('message').textContent = text; $('message').classList.toggle('error', error) }
function toast(title, text) {
  $('toastTitle').textContent = title; $('toastText').textContent = text; $('toast').classList.add('show');
  clearTimeout(toast.timer); toast.timer = setTimeout(() => $('toast').classList.remove('show'), 3600)
}

// Guardar o rascunho no navegador não aprova dados nem atualiza pesos do modelo.
function persist() {
  if (HAS_IMAGES) state.drafts[item().image_sha256] = clone(active);
  state.reviewer = $('reviewer').value;
  try { localStorage.setItem(storageKey, JSON.stringify({ images: state.drafts, reviewer: state.reviewer })) }
  catch { message('O navegador não conseguiu guardar o rascunho. Salve um arquivo de revisão.', true) }
}

/* ======================= Navegação entre seções ======================= */
const VIEWS = {
  home: ['Início', 'Visão geral da inspeção aberta e do espaço de trabalho'],
  inspections: ['Inspeções', 'Campanhas, edificações e revisões registradas'],
  upload: ['Importar mídia', 'Fotos e vídeos do voo viram uma nova inspeção'],
  review: ['Revisão', 'Confira, corrija e decida foto a foto'],
  reports: ['Laudos e downloads', 'Laudo técnico (ABNT), evidências e arquivos da inspeção'],
  training: ['Treinamento', 'Ciclo de fine-tuning com as correções humanas'],
  monitor: ['Monitoramento', 'Treino em andamento e tarefas do servidor local'],
  models: ['Modelos', 'Comparação, cobertura e critério de adoção'],
  classes: ['Classes', 'Categorias de ocorrência: nomes, cores e classes propostas']
};
function showView(view) {
  if (!VIEWS[view]) view = 'home';
  state.view = view;
  document.querySelectorAll('.view').forEach(section => section.classList.toggle('active', section.id === 'view-' + view));
  document.querySelectorAll('.nav-item[data-view]').forEach(button => button.setAttribute('aria-current', button.dataset.view === view ? 'page' : 'false'));
  $('viewTitle').textContent = VIEWS[view][0];
  $('viewSubtitle').textContent = DATA.version ? `${VIEWS[view][1]} · ${reviewLabel(DATA.version, DATA.display_name)}` : VIEWS[view][1];
  document.title = [VIEWS[view][0], DATA.version ? reviewLabel(DATA.version, DATA.display_name) : '', 'DroneCamp Vision'].filter(Boolean).join(' · ');
  try { sessionStorage.setItem('dronecamp-view', view) } catch {}
  if (location.hash !== '#' + view) history.replaceState(null, '', '#' + view);
  closeMenu();
  if (view === 'home') renderHome();
  if (view === 'inspections') renderInspections();
  if (view === 'upload') renderUpload();
  if (view === 'review') paint();
  if (view === 'reports') renderReports();
  if (view === 'training') renderTraining();
  if (view === 'monitor') renderMonitor();
  if (view === 'models') renderModels();
  if (view === 'classes') renderClasses();
  window.scrollTo({ top: 0 })
}
document.querySelectorAll('[data-view]').forEach(node => node.onclick = event => { event.preventDefault(); showView(node.dataset.view) });
document.querySelectorAll('[data-goto]').forEach(button => button.onclick = () => showView(button.dataset.goto));
function closeMenu() { $('sidebar').classList.remove('open'); $('scrim').classList.add('hidden'); $('menuToggle').setAttribute('aria-expanded', 'false') }
$('menuToggle').onclick = () => { const open = !$('sidebar').classList.contains('open'); $('sidebar').classList.toggle('open', open); $('scrim').classList.toggle('hidden', !open); $('menuToggle').setAttribute('aria-expanded', String(open)) };
$('scrim').onclick = closeMenu;

/* ======================= Revisão (lógica preservada) ======================= */
function decidedStatus(photo) {
  const decision = state.drafts[photo.image_sha256];
  return decision && decision.status !== 'pending' ? decision.status : null
}
const currentBoxes = photo => { const decision = state.drafts[photo.image_sha256]; return decision ? decision.boxes : photo.boxes };

function queue() {
  if (!HAS_IMAGES) return;
  const filter = $('filter').value;
  $('queue').replaceChildren();
  let decided = 0;
  DATA.images.forEach((photo, index) => {
    const decision = state.drafts[photo.image_sha256];
    if (decision && decision.status !== 'pending') decided++;
    if (filter === 'suggested') {
      if (!pendingSuggestionsFor(photo, decision || { boxes: photo.boxes }).length) return
    } else if (filter !== 'all' && (filter === 'pending' ? decision && decision.status !== 'pending' : photo.status !== filter)) return;
    const button = el('button', { type: 'button', class: 'q-item', 'aria-current': String(index === state.index) });
    const thumb = el('img', { src: photo.image_url, alt: '', loading: 'lazy' });
    const title = el('div', { class: 'q-title' }, photo.pages && photo.pages.length ? `${index + 1}. Páginas ${photo.pages.join(', ')}` : `${index + 1}. ${photo.filename}`);
    const pendingAi = pendingSuggestionsFor(photo, decision || { boxes: photo.boxes }).length;
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
  $('stats').textContent = `${DATA.images.length} fotos · ${decided} decisões`;
  $('queueProgress').style.width = `${DATA.images.length ? decided / DATA.images.length * 100 : 0}%`;
  const pendingAi = DATA.images.reduce((sum, photo) => sum + pendingSuggestionsFor(photo, state.drafts[photo.image_sha256] || { boxes: photo.boxes }).length, 0);
  $('navSuggestions').textContent = pendingAi;
  $('navSuggestions').classList.toggle('hidden', !pendingAi)
}

function resetSelection() {
  state.selected = -1;
  ['x1', 'y1', 'x2', 'y2'].forEach(id => $(id).value = '');
  $('applyBox').textContent = 'Adicionar caixa';
  $('deleteBox').disabled = true
}

function selectBox(index) {
  const box = active.boxes[index];
  state.selected = index;
  $('classId').value = box.class_id;
  ['x1', 'y1', 'x2', 'y2'].forEach((id, position) => $(id).value = box.bbox_xyxy[position]);
  $('applyBox').textContent = 'Atualizar caixa';
  $('deleteBox').disabled = false;
  renderBoxes()
}

function renderBoxes() {
  const list = $('boxes');
  list.replaceChildren();
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
  renderSuggestions();
  paint()
}

// Sugestões do modelo são apenas candidatas: aceitar uma copia a caixa para
// a revisão e exige nova confirmação da foto inteira.
const sameBox = (a, b) => a.class_id === b.class_id && a.bbox_xyxy.every((value, i) => Math.round(value) === Math.round(b.bbox_xyxy[i]));
function boxIou(a, b) {
  const inter = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0
}
// Aceitar sobre uma caixa existente do mesmo objeto duplicaria o alvo de treino.
const coveringBox = suggestion => DATA.duplicate_iou
  ? active.boxes.findIndex(box => box.class_id === suggestion.class_id && boxIou(box.bbox_xyxy, suggestion.bbox_xyxy) >= DATA.duplicate_iou)
  : -1;
// Limite de exibição: esconder sugestões fracas não as descarta nem as aceita.
const confThreshold = () => DATA.suggestions ? Number($('confFilter').value) / 100 : 0;
function pendingSuggestionsFor(photo, decision) {
  return (photo.model_suggestions || [])
    .map((suggestion, index) => ({ suggestion, index }))
    .filter(({ suggestion }) => !(decision.boxes || []).some(box => sameBox(box, suggestion)) && !(decision.dismissed_suggestions || []).includes(suggestion.id))
}
function pendingSuggestions() {
  if (!HAS_IMAGES) return [];
  return pendingSuggestionsFor(item(), active).filter(({ suggestion }) => suggestion.confidence >= confThreshold() - 1e-9)
}

function acceptSuggestion(suggestion) {
  const covered = coveringBox(suggestion);
  const box = { class_id: suggestion.class_id, bbox_xyxy: suggestion.bbox_xyxy.map(Math.round), note: `Sugestão do modelo aceita (confiança ${Math.round(suggestion.confidence * 100)}%); conferida na revisão local.` };
  // Substituir troca só a geometria e a nota; campos como proposed_new_class ficam.
  if (covered >= 0) active.boxes[covered] = { ...active.boxes[covered], ...box }; else active.boxes.push(box);
  invalidateDecision(covered >= 0
    ? `Caixa ${covered + 1} substituída pela geometria da sugestão; nenhuma duplicata criada. Confira a foto inteira antes de confirmar.`
    : 'Sugestão aceita como caixa. Ajuste se necessário e confira a foto inteira antes de confirmar.')
}

function dismissSuggestion(suggestion) {
  active.dismissed_suggestions = [...(active.dismissed_suggestions || []), suggestion.id];
  persist(); renderBoxes(); queue();
  message('Sugestão descartada nesta revisão.')
}

function renderSuggestions() {
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
    const accept = el('button', { type: 'button', class: 'small ai' }, covered >= 0 ? `Substituir caixa ${covered + 1}` : 'Aceitar');
    accept.onclick = () => acceptSuggestion(suggestion);
    const dismiss = el('button', { type: 'button', class: 'small' }, 'Descartar');
    dismiss.onclick = () => dismissSuggestion(suggestion);
    const bar = el('div', { class: 'conf', 'aria-hidden': 'true' }, el('i', { style: `width:${percent}%` }));
    row.append(head, metaRow, bar, el('div', { class: 'actions' }, accept, dismiss));
    list.append(row)
  });
  if (!pending.length) list.textContent = (item().model_suggestions || []).length
    ? (hidden > 0 ? `${hidden} sugest${hidden > 1 ? 'ões abaixo' : 'ão abaixo'} do limite de confiança escolhido.` : 'Todas as sugestões desta foto foram aceitas ou descartadas.')
    : 'O modelo não sugeriu caixas nesta foto acima da confiança mínima.';
  else if (hidden > 0) list.append(el('p', { class: 'muted small-text' }, `${hidden} oculta${hidden > 1 ? 's' : ''} abaixo de ${Math.round(confThreshold() * 100)}%.`))
}

function invalidateDecision(text) {
  active.status = 'pending'; active.confirmed_complete = false;
  $('decision').value = 'pending'; $('confirmed').checked = false;
  persist(); resetSelection(); renderBoxes(); queue(); message(text)
}

function paint(preview) {
  if (!HAS_IMAGES) return;
  const canvas = $('canvas'), ctx = canvas.getContext('2d');
  if (!state.img.complete || !state.img.naturalWidth) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.img, 0, 0);
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
  if (preview) {
    ctx.setLineDash([8, 5]); ctx.strokeStyle = '#2457F5'; ctx.lineWidth = 3;
    ctx.strokeRect(preview[0], preview[1], preview[2] - preview[0], preview[3] - preview[1]); ctx.setLineDash([])
  }
}

function openPhoto(index) {
  if (!HAS_IMAGES) return;
  persist();
  state.index = index;
  active = clone(draft());
  resetSelection();
  state.drawing = false;
  $('drawToggle').setAttribute('aria-pressed', 'false');
  $('viewer').classList.remove('drawing');
  const photo = item();
  $('photoTitle').textContent = `Foto ${index + 1} de ${DATA.images.length}`;
  $('photoMeta').textContent = [photo.filename, photo.pages && photo.pages.length ? `páginas ${photo.pages.join(', ')}` : null, `${photo.width} × ${photo.height}`, photo.ingest && photo.ingest.kind === 'video_frame' ? `quadro de vídeo em ${photo.ingest.frame_time_s}s` : null].filter(Boolean).join(' · ');
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
  $('decision').value = active.status; $('notes').value = active.notes; $('confirmed').checked = active.confirmed_complete;
  $('canvas').width = photo.width; $('canvas').height = photo.height;
  ['x1', 'x2'].forEach(id => $(id).max = photo.width); ['y1', 'y2'].forEach(id => $(id).max = photo.height);
  state.img = new Image();
  state.img.onload = () => renderBoxes();
  state.img.src = photo.image_url;
  renderBoxes(); queue();
  const current = $('queue').querySelector('[aria-current="true"]');
  if (current) current.scrollIntoView({ block: 'nearest' });
  message('Ajuste as propostas e registre sua decisão depois de conferir a foto inteira.')
}

function validBox(coordinates) {
  return coordinates.every(Number.isFinite) && coordinates[0] >= 0 && coordinates[1] >= 0
    && coordinates[2] <= item().width && coordinates[3] <= item().height && coordinates[2] > coordinates[0] && coordinates[3] > coordinates[1]
}

// Toda edição invalida a confirmação anterior: a foto deve ser revista novamente.
function commitBox(coordinates) {
  coordinates = coordinates.map(Math.round);
  if (!validBox(coordinates)) { message('A caixa precisa ter área e ficar dentro da foto.', true); return }
  const classId = $('classId').value === '' ? NaN : Number($('classId').value);
  if (!Number.isInteger(classId)) { message('Escolha a categoria da caixa.', true); return }
  // Classe inativa (seção Classes): caixas já existentes mantêm a categoria; caixas novas não a recebem.
  if (state.selected < 0 && classOf(classId).active === false) { message('Categoria inativa: reative-a na seção Classes para usá-la em caixas novas.', true); return }
  const box = { class_id: classId, bbox_xyxy: coordinates, note: 'Caixa conferida/editada na revisão local.' };
  if (state.selected < 0) active.boxes.push(box); else active.boxes[state.selected] = box;
  active.confirmed_complete = false; active.status = 'pending';
  $('confirmed').checked = false; $('decision').value = 'pending';
  persist(); resetSelection(); renderBoxes(); queue();
  message('Caixa registrada. Confira a foto inteira antes de confirmar a decisão.')
}
$('applyBox').onclick = () => commitBox(['x1', 'y1', 'x2', 'y2'].map(id => $(id).value === '' ? NaN : Number($(id).value)));
$('newBox').onclick = () => { resetSelection(); renderBoxes() };
$('notes').oninput = () => { active.notes = $('notes').value; persist() };

// Excluir uma caixa também exige nova decisão humana sobre a foto inteira.
function deleteSelected() {
  if (state.selected < 0) return;
  active.boxes.splice(state.selected, 1);
  active.status = 'pending'; active.confirmed_complete = false;
  $('decision').value = 'pending'; $('confirmed').checked = false;
  persist(); resetSelection(); renderBoxes(); queue();
  message('Caixa excluída. Confira a foto inteira antes de confirmar a decisão.')
}
$('deleteBox').onclick = deleteSelected;

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
function point(event) {
  const rect = $('canvas').getBoundingClientRect();
  return [Math.min(item().width, Math.max(0, (event.clientX - rect.left) * item().width / rect.width)), Math.min(item().height, Math.max(0, (event.clientY - rect.top) * item().height / rect.height))]
}
$('canvas').onpointerdown = event => { if (!state.drawing) return; event.preventDefault(); state.start = point(event); $('canvas').setPointerCapture(event.pointerId) };
$('canvas').onpointermove = event => { if (!state.start) return; const end = point(event); paint([Math.min(state.start[0], end[0]), Math.min(state.start[1], end[1]), Math.max(state.start[0], end[0]), Math.max(state.start[1], end[1])]) };
$('canvas').onpointerup = event => { if (!state.start) return; const end = point(event), start = state.start; state.start = null; commitBox([Math.min(start[0], end[0]), Math.min(start[1], end[1]), Math.max(start[0], end[0]), Math.max(start[1], end[1])]) };
$('canvas').onpointercancel = () => { state.start = null; paint() };
$('showBoxes').onchange = () => paint();
$('showSuggestions').onchange = () => paint();
$('confFilter').oninput = () => { $('confValue').textContent = `${$('confFilter').value}%`; renderBoxes() };

// A aprovação visual exige confirmação da revisão integral e caixas coerentes.
$('saveDecision').onclick = () => {
  if (!HAS_IMAGES) return;
  const status = $('decision').value, confirmed = $('confirmed').checked;
  if (['positive', 'negative'].includes(status) && !confirmed) { message('Confirme que revisou a foto inteira antes de aprovar a anotação.', true); return }
  if (status === 'positive' && !active.boxes.length) { message('Uma foto com ocorrências precisa de pelo menos uma caixa.', true); return }
  if (status === 'negative' && active.boxes.length) { message('Uma foto negativa não pode ter caixas. Corrija a anotação ou escolha outra decisão.', true); return }
  active.status = status; active.notes = $('notes').value; active.confirmed_complete = confirmed;
  persist(); queue();
  message(status === 'pending' ? 'Rascunho guardado; decisão ainda pendente.' : 'Sua decisão foi registrada no rascunho local. Salve o arquivo de revisão para importá-la.');
  if (status !== 'pending') toast('Decisão registrada', `Foto ${state.index + 1}: ${decisionNames[status]}. Fica no rascunho local até salvar o arquivo.`)
};

// Abrir uma foto já guarda um rascunho igual ao registro. Esse rascunho "intocado" não é decisão humana:
// não vai ao servidor (senão uma foto positiva do registro viraria pendente no laudo) nem ao CSV.
const untouched = (photo, value) => !value || (value.status === 'pending' && !value.confirmed_complete && !(value.dismissed_suggestions || []).length
  && (value.notes || '') === (photo.human_notes || '') && JSON.stringify(value.boxes) === JSON.stringify(photo.boxes));
const touchedDraft = photo => { const value = state.drafts[photo.image_sha256]; return untouched(photo, value) ? null : value };
// Decisões do rascunho local (as editadas, inclusive pendentes com caixas mexidas), para o servidor desenhar.
const allDrafts = () => DATA.images.flatMap(photo => { const value = touchedDraft(photo); return value ? [{ image_sha256: photo.image_sha256, ...value }] : [] });
const decidedDrafts = () => allDrafts().filter(value => value.status !== 'pending');
function reviewFile() {
  const reviewer = $('reviewer').value.trim();
  if (!reviewer) { showView('review'); message('Preencha seu identificador de revisão para salvar o arquivo.', true); $('reviewer').focus(); return null }
  const decisions = decidedDrafts();
  if (!decisions.length) { showView('review'); message('Registre pelo menos uma decisão antes de salvar.', true); return null }
  return { schema_version: 1, registry_sha256: DATA.registry_sha256, reviewer_id: reviewer, exported_at: new Date().toISOString(), images: decisions }
}
function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a'); link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

// O hash vincula o feedback ao registro exibido; exportar não executa treinamento.
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
$('previous').onclick = () => openPhoto(Math.max(0, state.index - 1));
$('next').onclick = () => openPhoto(Math.min(DATA.images.length - 1, state.index + 1));
$('filter').onchange = queue;
$('reviewer').value = state.reviewer;
$('reviewer').oninput = persist;

// Seletor de inspeções: cada revisão abre na sua própria página (/r/<pasta>/), com fotos,
// decisões salvas e arquivo exportado separados; trocar não mistura registros.
// Nome legível (NAMING.review); a pasta crua fica na dica (title) para rastreabilidade.
function reviewLabel(name, display) { return NAMING.review(name, display) }
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
  if (!picker.options.length) picker.append(el('option', { value: '' }, DATA.version ? reviewLabel(DATA.version, DATA.display_name) : 'Nenhuma inspeção'));
  picker.disabled = picker.options.length < 2;
  picker.onchange = event => { if (event.target.value) location.href = event.target.value + location.hash }
}
fillReviewPicker(REVIEWS && REVIEWS.items, REVIEWS && REVIEWS.current);

// Atalhos de teclado só na revisão e fora de campos de texto.
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
  else return;
  event.preventDefault()
});

/* ======================= Início ======================= */
function barRow(color, label, human, ai, max, countText) {
  const row = el('div', { class: 'bar-row' });
  row.style.setProperty('--c', color);
  const bar = el('div', { class: 'bar', role: 'img', 'aria-label': `${label}: ${countText}` },
    el('i', { class: 'human', style: `width:${max ? human / max * 100 : 0}%` }), el('i', { class: 'ai', style: `width:${max ? ai / max * 100 : 0}%` }));
  row.append(el('div', { class: 'name' }, el('span', { class: 'swatch' }), el('span', { title: label }, label)), bar, el('div', { class: 'count' }, countText));
  return row
}

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
    currentBoxes(photo).forEach(box => { if (totals[box.class_id]) totals[box.class_id].boxes++; boxes++ });
    pendingSuggestionsFor(photo, decision || { boxes: photo.boxes }).forEach(({ suggestion }) => { if (totals[suggestion.class_id]) totals[suggestion.class_id].ai++; pendingAi++ })
  });
  return { totals, boxes, pendingAi, decided, triage }
}

function renderHome() {
  const { totals, boxes, pendingAi, decided, triage } = reviewTotals();
  const sparse = DATA.classes.filter(category => totals[category.id].boxes < 5);
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
  const max = Math.max(1, ...Object.values(totals).map(value => value.boxes + value.ai));
  $('classBars').replaceChildren(...DATA.classes.map(category => {
    const total = totals[category.id];
    const row = barRow(category.color, category.label, total.boxes, total.ai, max, `${total.boxes}${total.ai ? ' +' + total.ai : ''}`);
    if (total.boxes === 0) row.querySelector('.count').classList.add('cell-bad');
    else if (total.boxes < 5) row.querySelector('.count').classList.add('cell-warn');
    return row
  }));
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
  const last = OPS.runs.find(run => run.curves.length);
  if (!last) $('lastRunSummary').textContent = 'Nenhum treino encontrado em runs/.';
  else {
    const chart = el('div', { class: 'chart' });
    $('lastRunSummary').replaceChildren(
      el('p', {}, el('span', { class: 'chip ' + stateClass(last.state) }, stateLabel(last.state)), ' ', el('b', { title: last.id }, runName(last)), ' ', el('span', { class: 'muted small-text' }, fmtDate(last.started_at))),
      el('p', { class: 'muted' }, `${last.epochs_done}/${last.epochs_planned || '?'} épocas · melhor mAP50 ${pct(last.best_epoch && last.best_epoch.map50)}`), chart);
    lineChart(chart, last.curves, [{ key: 'map50', label: 'mAP50', color: '#16A34A' }, { key: 'recall', label: 'Recall', color: '#2457F5' }], { height: 120, percent: true })
  }
  $('homeInspections').replaceChildren(inspectionCards(true))
}

/* ======================= Inspeções ======================= */
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
function inspectionCards(compact) {
  const items = inspectionItems();
  $('navInspections').textContent = items.length || '';
  if (!items.length) return el('div', { class: 'empty' }, SERVER ? 'Nenhuma inspeção em data/reviews. Importe fotos ou vídeos para começar.' : 'Nenhuma inspeção aberta.');
  const list = el('div', { class: 'inspection-list' });
  (compact ? items.slice(0, 3) : items).forEach(review => {
    const progress = review.images ? Math.round((review.decided || 0) / review.images * 100) : 0;
    const card = el('div', { class: 'inspection' + (review.current ? ' current' : '') });
    card.append(
      el('div', { class: 'head' }, el('b', { title: review.name }, reviewLabel(review.name, review.display_name)), el('span', { class: 'chip ' + (review.current ? 'primary' : review.page_ready === false ? 'warn' : 'neutral') }, review.current ? 'aberta' : review.page_ready === false ? 'página não gerada' : review.scope_label || 'revisão')),
      // Só a contagem: o código interno da edificação (ex.: ceasa_pavilhao_a) não é mostrado.
      el('div', { class: 'meta' }, el('span', {}, buildingCount((review.buildings || []).length)), review.updated_at ? el('span', {}, `atualizada ${fmtDate(review.updated_at)}`) : null),
      el('div', { class: 'nums' }, el('div', {}, el('b', {}, review.images ?? '—'), 'fotos'), el('div', {}, el('b', {}, `${progress}%`), 'decididas'), el('div', {}, el('b', { class: review.suggestions_pending ? 'cell-ai' : '' }, review.suggestions_pending ?? 0), 'sugestões IA')),
      el('div', { class: 'progress ok' }, el('i', { style: `width:${progress}%` })));
    const actions = el('div', { class: 'actions' });
    if (review.current) actions.append(el('button', { type: 'button', class: 'small primary', onclick: () => showView('review') }, 'Continuar revisão'), el('button', { type: 'button', class: 'small', onclick: () => showView('reports') }, 'Laudos e downloads'));
    else if (review.href && review.page_ready !== false) actions.append(el('a', { class: 'btn small primary', href: review.href + '#review' }, 'Abrir'));
    else if (review.href) actions.append(el('span', { class: 'muted small-text' }, 'Gere a página com render-review para abrir.'));
    card.append(actions);
    list.append(card)
  });
  return list
}
function renderInspections() {
  $('inspectionsHint').textContent = SERVER ? 'Cada inspeção guarda as fotos originais congeladas por hash, o registro de revisão e o histórico das decisões.' : 'Aberta direto do disco, a página mostra só esta inspeção. Pelo servidor local, todas as inspeções de data/reviews aparecem aqui.';
  $('inspectionList').replaceWith(Object.assign(inspectionCards(false), { id: 'inspectionList' }))
}

/* ======================= Importar mídia ======================= */
const upload = { files: [], running: false, batch: null };
const IMAGE_EXT = /\.(jpe?g|png|webp|bmp|tiff?)$/i, VIDEO_EXT = /\.(mp4|mov|avi|mkv|m4v|webm)$/i;
function addFiles(list) {
  [...list].forEach(file => {
    if (!IMAGE_EXT.test(file.name) && !VIDEO_EXT.test(file.name)) return;
    if (upload.files.some(entry => entry.file.name === file.name && entry.file.size === file.size)) return;
    upload.files.push({ file, state: 'pronto', sent: 0 })
  });
  renderFiles()
}
function renderFiles() {
  const list = $('fileList');
  list.replaceChildren(...upload.files.map((entry, index) => {
    const video = VIDEO_EXT.test(entry.file.name);
    const row = el('div', { class: 'file' }, icon(video ? 'i-video' : 'i-image'),
      el('div', {}, el('div', { class: 'name', title: entry.file.name }, entry.file.name), el('small', {}, `${fmtBytes(entry.file.size)}${video ? ' · vídeo → quadros' : ''}`), entry.sent ? el('div', { class: 'progress' }, el('i', { style: `width:${entry.sent}%` })) : null),
      upload.running ? el('span', { class: 'state ' + (entry.state === 'enviado' ? 'ok' : entry.state === 'falhou' ? 'bad' : '') }, entry.state) : el('button', { type: 'button', class: 'small', onclick: () => { upload.files.splice(index, 1); renderFiles() } }, 'Remover'));
    return row
  }));
  const photos = upload.files.filter(entry => IMAGE_EXT.test(entry.file.name)).length, videos = upload.files.length - photos;
  const bytes = upload.files.reduce((sum, entry) => sum + entry.file.size, 0);
  $('fileSummary').textContent = upload.files.length ? `${photos} foto${photos === 1 ? '' : 's'} · ${videos} vídeo${videos === 1 ? '' : 's'} · ${fmtBytes(bytes)}` : '';
  $('startUpload').disabled = !SERVER || upload.running || !upload.files.length
}
function renderUpload() {
  $('uploadOffline').classList.toggle('hidden', SERVER);
  const weights = OPS.weights.filter(path => path.startsWith('runs/') || path.startsWith('models/'));
  fillSelect($('uploadWeights'), [['', 'Não gerar agora (revisar sem modelo)'], ...weights.map(path => [path, weightLabel(path)])], (OPS.runs.find(value => value.weights && value.weights.best) || { weights: {} }).weights.best || '');
  if (!$('uploadName').value) $('uploadName').value = `Inspeção ${new Date().toLocaleDateString('pt-BR')}`;
  renderFiles()
}
$('dropzone').onclick = () => $('fileInput').click();
$('dropzone').onkeydown = event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); $('fileInput').click() } };
$('fileInput').onchange = () => { addFiles($('fileInput').files); $('fileInput').value = '' };
['dragenter', 'dragover'].forEach(type => $('dropzone').addEventListener(type, event => { event.preventDefault(); $('dropzone').classList.add('over') }));
['dragleave', 'drop'].forEach(type => $('dropzone').addEventListener(type, event => { event.preventDefault(); $('dropzone').classList.remove('over') }));
$('dropzone').addEventListener('drop', event => addFiles(event.dataTransfer.files));
// Nome livre ("Galpão B · outubro"): o servidor deriva a pasta (sem acentos) e guarda o nome digitado no registro.

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
$('startUpload').onclick = async () => {
  const name = $('uploadName').value.trim(), group = $('uploadGroup').value.trim();
  if (!name) { $('uploadStatus').textContent = 'Informe o nome da inspeção.'; $('uploadName').focus(); return }
  if (!group) { $('uploadStatus').textContent = 'Informe a edificação ou campanha.'; $('uploadGroup').focus(); return }
  upload.running = true; upload.cancelled = false;
  $('cancelUpload').classList.remove('hidden'); $('uploadResult').classList.add('hidden'); $('uploadProgress').classList.remove('hidden');
  renderFiles();
  const total = upload.files.reduce((sum, entry) => sum + entry.file.size, 0);
  let done = 0;
  try {
    const { batch } = await api('/api/uploads', { name });
    upload.batch = batch;
    for (const entry of upload.files) {
      if (upload.cancelled) throw new Error('Envio cancelado.');
      entry.state = 'enviando';
      $('uploadStatus').textContent = `Enviando ${entry.file.name}…`;
      await putFile(`/api/uploads/${batch}/files`, entry.file, fraction => { entry.sent = Math.round(fraction * 100); $('uploadProgress').firstElementChild.style.width = `${(done + fraction * entry.file.size) / total * 100}%`; renderFiles() });
      entry.state = 'enviado'; entry.sent = 100; done += entry.file.size; renderFiles()
    }
    $('uploadStatus').textContent = 'Arquivos enviados. Congelando fotos por hash e extraindo quadros dos vídeos…';
    await api(`/api/uploads/${batch}/ingest`, { name, group, every_seconds: Number($('uploadEvery').value) || 2, max_frames: Number($('uploadMaxFrames').value) || 200, notes: $('uploadNotes').value, uploaded_by: $('reviewer').value.trim() || null, weights: $('uploadWeights').value || null });
    let status;
    do {
      await new Promise(resolve => setTimeout(resolve, 1500));
      status = await api(`/api/uploads/${batch}`);
      $('uploadStatus').textContent = status.message || 'Processando…'
    } while (status.state === 'running');
    if (status.state !== 'finished') throw new Error(status.error || 'A importação falhou.');
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
  } catch (error) {
    $('uploadStatus').textContent = `Não concluído: ${error.message}`;
    upload.files.forEach(entry => { if (entry.state === 'enviando') entry.state = 'falhou' })
  } finally {
    upload.running = false; $('cancelUpload').classList.add('hidden'); $('uploadProgress').classList.add('hidden'); renderFiles()
  }
};
$('cancelUpload').onclick = () => { upload.cancelled = true; if (upload.request) upload.request.abort() };

/* ======================= Laudos e downloads ======================= */
function renderReports() {
  $('reportsOffline').classList.toggle('hidden', SERVER);
  $('makeReport').disabled = !SERVER || !HAS_IMAGES;
  $('makeEvidence').disabled = !SERVER || !HAS_IMAGES;
  $('downloadCsv').disabled = !HAS_IMAGES;
  if (!$('reportTitle').value && DATA.version) $('reportTitle').value = 'Laudo técnico de inspeção de cobertura com uso de drone';
  if (WORKSPACE && WORKSPACE.capabilities && WORKSPACE.capabilities.pdf === false) { $('reportPdf').checked = false; $('reportPdf').disabled = true; $('reportPdf').nextElementSibling.textContent = 'PDF indisponível neste PC: instale o extra "reports" (pip install -e .[reports]). O HTML é gerado sempre.' }
  renderFileHistory()
}
function fileRow(file) {
  const kinds = { pdf: ['i-report', 'Laudo em PDF'], html: ['i-report', 'Laudo em HTML'], zip: ['i-zip', 'Pacote de evidências'], json: ['i-save', 'Dados do laudo'], csv: ['i-report', 'Planilha'] };
  const [ico, label] = kinds[file.kind] || ['i-download', file.kind];
  // Pastas e ZIPs são <carimbo UTC>_laudo_<inspeção>: a data vira texto legível, o caminho fica na dica.
  const generatedOn = NAMING.generated(String(file.name || '').split('/')[1] || file.name)[1];
  return el('div', { class: 'download' }, el('div', { class: 'ico' }, icon(ico)),
    el('div', {}, el('b', { title: file.name }, label), el('small', {}, `${generatedOn ? 'Gerado em ' + fmtDate(generatedOn) : fmtDate(file.modified)} · ${fmtBytes(file.bytes)}`)),
    el('a', { class: 'btn small', href: file.href, target: '_blank', rel: 'noopener', download: file.kind === 'zip' || file.kind === 'pdf' ? '' : null }, 'Baixar'))
}

// Dados técnicos opcionais do laudo (seção "Dados técnicos"): só os campos preenchidos vão ao servidor.
function reportDetails() {
  const details = {};
  document.querySelectorAll('[data-detail]').forEach((input) => { const value = input.value.trim(); if (value) details[input.dataset.detail] = value });
  return details;
}

async function renderFileHistory() {
  const history = $('fileHistory');
  let files = (OPS.review_files || []);
  if (SERVER && DATA.version) { try { files = (await api(`/api/reviews/${encodeURIComponent(DATA.version)}/files`)).files } catch {} }
  history.replaceChildren(...(files.length ? files.map(fileRow) : [el('div', { class: 'empty' }, 'Nenhum laudo ou pacote gerado ainda para esta inspeção.')]))
}
$('refreshFiles').onclick = renderFileHistory;
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
$('downloadCsv').onclick = () => {
  const rows = [['numero_foto', 'image_sha256', 'filename', 'status', 'origem', 'caixa', 'classe_id', 'classe', 'x1', 'y1', 'x2', 'y2', 'nota']];
  DATA.images.forEach((photo, index) => {
    const decision = touchedDraft(photo);  // rascunho editado vale como está (até "pending"); intocado, vale o registro
    const boxes = decision ? decision.boxes : photo.boxes;
    const status = decision ? decision.status : photo.status;
    if (!boxes.length) rows.push([index + 1, photo.image_sha256, photo.filename, status, decision ? 'revisao_humana' : 'registro', '', '', '', '', '', '', '', '']);
    boxes.forEach((box, position) => rows.push([index + 1, photo.image_sha256, photo.filename, status, decision ? 'revisao_humana' : 'registro', position + 1, box.class_id, classOf(box.class_id).label, ...box.bbox_xyxy.map(Math.round), box.note || '']))
  });
  const csv = '﻿' + rows.map(row => row.map(value => `"${String(value ?? '').replaceAll('"', '""')}"`).join(';')).join('\r\n');
  downloadBlob(new Blob([csv], { type: 'text/csv;charset=utf-8' }), `dronecamp-ocorrencias-${DATA.version}.csv`)
};
$('downloadReview').onclick = () => { const feedback = reviewFile(); if (feedback) downloadBlob(new Blob([JSON.stringify(feedback, null, 2)], { type: 'application/json' }), 'dronecamp-revisao-' + DATA.version + '.json') };

/* ======================= Gráficos SVG (sem bibliotecas externas) ======================= */
const drawnCharts = new Map();
let chartViewport = innerWidth, chartTimer;
addEventListener('resize', () => {
  if (innerWidth === chartViewport) return;
  chartViewport = innerWidth;
  clearTimeout(chartTimer);
  chartTimer = setTimeout(() => drawnCharts.forEach((args, node) => { if (!node.isConnected) drawnCharts.delete(node); else if (node.offsetParent) lineChart(node, ...args) }), 150)
});
function lineChart(container, points, series, options = {}) {
  drawnCharts.set(container, [points, series, options]);
  container.replaceChildren();
  const height = options.height || 220, width = Math.max(280, Math.round(container.clientWidth) || 640), pad = { l: 44, r: 12, t: 10, b: 24 };
  const usable = series.filter(serie => points.some(point => point[serie.key] != null));
  if (!points.length || !usable.length) { container.append(el('div', { class: 'empty' }, options.empty || 'Sem dados de épocas ainda.')); return }
  const xs = points.map(point => point.epoch);
  const xMin = Math.min(...xs), xMax = Math.max(...xs, xMin + 1);
  const values = usable.flatMap(serie => points.map(point => point[serie.key]).filter(value => value != null));
  let yMax = options.percent ? Math.max(0.05, ...values) : Math.max(...values);
  let yMin = options.percent ? 0 : Math.min(...values);
  if (yMax === yMin) yMax = yMin + 1;
  yMax += (yMax - yMin) * .05;
  const sx = x => pad.l + (x - xMin) / (xMax - xMin) * (width - pad.l - pad.r);
  const sy = y => pad.t + (1 - (y - yMin) / (yMax - yMin)) * (height - pad.t - pad.b);
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`); svg.setAttribute('preserveAspectRatio', 'none'); svg.style.height = height + 'px';
  svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', options.label || usable.map(serie => serie.label).join(', '));
  const add = (tag, attrs) => { const node = document.createElementNS(ns, tag); Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v)); svg.append(node); return node };
  for (let i = 0; i <= 4; i++) {
    const value = yMin + (yMax - yMin) * i / 4, y = sy(value);
    add('line', { x1: pad.l, x2: width - pad.r, y1: y, y2: y, stroke: 'var(--chart-grid)', 'stroke-width': 1 });
    const label = add('text', { x: pad.l - 6, y: y + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--muted)' });
    label.textContent = options.percent ? `${Math.round(value * 100)}%` : value.toFixed(value >= 10 ? 0 : 2)
  }
  [xMin, Math.round((xMin + xMax) / 2), xMax].forEach(x => {
    const anchor = x === xMax ? 'end' : x === xMin ? 'start' : 'middle';
    const label = add('text', { x: sx(x), y: height - 6, 'text-anchor': anchor, 'font-size': 11, fill: 'var(--muted)' });
    label.textContent = `ép. ${x}`
  });
  usable.forEach(serie => {
    const path = points.filter(point => point[serie.key] != null).map((point, i) => `${i ? 'L' : 'M'}${sx(point.epoch).toFixed(1)},${sy(point[serie.key]).toFixed(1)}`).join('');
    add('path', { d: path, fill: 'none', stroke: serie.color, 'stroke-width': 2, 'vector-effect': 'non-scaling-stroke', 'stroke-dasharray': serie.dashed ? '5 4' : '' })
  });
  if (options.marker != null) { const x = sx(options.marker); add('line', { x1: x, x2: x, y1: pad.t, y2: height - pad.b, stroke: 'var(--ai)', 'stroke-dasharray': '3 3', 'vector-effect': 'non-scaling-stroke' }) }
  const cursor = add('line', { x1: 0, x2: 0, y1: pad.t, y2: height - pad.b, stroke: 'var(--line-strong)', visibility: 'hidden', 'vector-effect': 'non-scaling-stroke' });
  const tip = el('div', { class: 'tooltip hidden' });
  svg.addEventListener('pointermove', event => {
    const rect = svg.getBoundingClientRect();
    const vx = (event.clientX - rect.left) / rect.width * width;
    const epoch = xMin + (vx - pad.l) / (width - pad.l - pad.r) * (xMax - xMin);
    const nearest = points.reduce((best, point) => Math.abs(point.epoch - epoch) < Math.abs(best.epoch - epoch) ? point : best, points[0]);
    const x = sx(nearest.epoch);
    cursor.setAttribute('x1', x); cursor.setAttribute('x2', x); cursor.setAttribute('visibility', 'visible');
    tip.textContent = `Época ${nearest.epoch} · ` + usable.map(serie => `${serie.label} ${nearest[serie.key] == null ? '—' : options.percent ? pct(nearest[serie.key]) : nearest[serie.key].toFixed(3)}`).join(' · ');
    tip.classList.remove('hidden'); tip.style.left = '0px';
    const left = Math.max(0, Math.min(rect.width - tip.offsetWidth - 4, x / width * rect.width + 8));
    tip.style.left = left + 'px'; tip.style.top = '4px'
  });
  svg.addEventListener('pointerleave', () => { cursor.setAttribute('visibility', 'hidden'); tip.classList.add('hidden') });
  container.append(svg, tip, el('div', { class: 'legend' }, ...usable.map(serie => { const node = el('span', {}, el('i'), serie.label); node.firstChild.style.setProperty('--c', serie.color); return node })))
}
/* Nome legível da execução (operations.py grava display_name/display_detail; runs/nomes.json personaliza). */
const runName = run => run ? (run.display_name || NAMING.run(run.id)) : '';
// Pasta do dataset de uma comparação ("data/pilot/ceasa_v8_piloto_s42/dataset.yaml" → "ceasa_v8_piloto_s42"); sem pasta, o nome do arquivo.
const datasetFolder = path => { const parts = String(path).replace(/\\/g, '/').split('/').filter(Boolean); const folder = parts.length > 1 ? parts[parts.length - 2] : (parts[0] || '').replace(/\.ya?ml$/, ''); return folder || String(path) };
const runById = id => OPS.runs.find(run => run.id === id) || { id };
const runLabel = id => runName(runById(id));
// Pesos: "Treino Piloto 56038EB9 · melhor época" (nome personalizado da execução, se houver) ou "Modelo Base YOLO26l".
const weightLabel = path => {
  const run = OPS.runs.find(value => Object.values(value.weights || {}).includes(path));
  if (!run) return NAMING.weights(path);
  const stem = path.split('/').pop().replace(/\.pt$/, '');
  return `${runName(run)} · ${{ best: 'melhor época', last: 'última época' }[stem] || stem}`
};
const SPLIT_LABELS = { train: 'Treino', val: 'Validação', test: 'Teste' };
const splitLabel = split => { const base = String(split || ''); const seen = base.includes('visto'); const key = base.split(' ')[0].split('(')[0]; return (SPLIT_LABELS[key] || key) + (seen ? ' ⚠ fotos vistas no treino' : '') };
const LOSS_SERIES = [
  { key: 'train_box', label: 'Erro de posição (treino)', color: '#2457F5' },
  { key: 'train_cls', label: 'Erro de classe (treino)', color: '#7C3AED' },
  { key: 'val_box', label: 'Erro de posição (validação)', color: '#2457F5', dashed: true },
  { key: 'val_cls', label: 'Erro de classe (validação)', color: '#7C3AED', dashed: true }
];
const METRIC_SERIES = [
  { key: 'precision', label: 'Precisão (sugestões certas)', color: '#D97706' },
  { key: 'recall', label: 'Recall (caixas encontradas)', color: '#2457F5' },
  { key: 'map50', label: 'mAP50 (acerto geral)', color: '#16A34A' },
  { key: 'map50_95', label: 'mAP50-95 (acerto com encaixe exato)', color: '#7C3AED' }
];

/* ======================= Treinamento ======================= */
const stateLabel = value => ({ completed: 'concluído', running: 'em andamento', interrupted: 'interrompido', failed: 'falhou', prepared: 'preparado' }[value] || value);
const stateClass = value => ({ completed: 'ok', running: 'live ok', interrupted: 'warn', failed: 'bad', prepared: 'neutral' }[value] || 'neutral');
const winPath = path => String(path || '').replaceAll('/', '\\');
const PY = '.\\.venv\\Scripts\\python.exe -m dronecamp_ia';
function runForWeights(sha) {
  if (!sha) return null;
  for (const comparison of OPS.comparisons) {
    const model = comparison.models.find(value => value.weights_sha256 === sha);
    if (model && model.run_id) return OPS.runs.find(run => run.id === model.run_id) || { id: model.run_id }
  }
  return null
}
function commandBox(text) {
  const box = el('div', { class: 'command' }, text);
  const copy = el('button', { type: 'button' }, 'Copiar');
  copy.onclick = async () => { try { await navigator.clipboard.writeText(text); copy.textContent = 'Copiado' } catch { copy.textContent = 'Selecione e copie' } setTimeout(() => copy.textContent = 'Copiar', 1500) };
  box.append(copy);
  return box
}
function nextVersionName() {
  const match = String(DATA.version || '').match(/^([a-z]+)_v(\d+)/i);
  return match ? `${match[1]}_v${Number(match[2]) + 1}_ID` : (DATA.version ? `${DATA.version}_rev1` : 'nova_versao_ID')
}
function renderCycle() {
  const directory = winPath(OPS.review_directory || `data/reviews/${DATA.version}`);
  const next = nextVersionName();
  const decided = DATA.images.filter(photo => decidedStatus(photo)).length;
  const latest = OPS.runs.find(run => run.weights && run.weights.best);
  const previous = latest ? winPath(latest.weights.best) : 'runs\\PILOTO_ANTERIOR\\fit\\weights\\best.pt';
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
function fillSelect(select, options, selected) {
  const previous = select.value;
  select.replaceChildren(...options.map(([value, label]) => el('option', { value }, label)));
  select.value = options.some(([value]) => value === previous) ? previous : (selected != null && options.some(([value]) => value === selected) ? selected : (options[0] || [''])[0])
}
function estimateSeconds(imgsz) {
  // Base: s/época das execuções concluídas no mesmo PC; escala ~ imgsz² quando muda o tamanho.
  const done = OPS.runs.filter(run => run.seconds_per_epoch && run.parameters.imgsz && run.epochs_done > 3);
  if (!done.length) return null;
  const same = done.filter(run => run.parameters.imgsz === imgsz);
  const base = same.length ? same : done;
  const perEpoch = base.reduce((sum, run) => sum + run.seconds_per_epoch * (imgsz / run.parameters.imgsz) ** 2, 0) / base.length;
  return { perEpoch, runs: base.length }
}
const duration = seconds => seconds == null ? '—' : seconds >= 3600 ? `${(seconds / 3600).toFixed(1)} h` : seconds >= 60 ? `${Math.round(seconds / 60)} min` : `${Math.round(seconds)} s`;
function trainOptions() {
  const integer = id => $(id).value === '' ? null : Number($(id).value);
  return { data: $('trainData').value, weights: $('trainWeights').value, epochs: integer('trainEpochs'), imgsz: integer('trainImgsz'), batch: integer('trainBatch'), tile: $('trainTile').checked, gpu_eval: $('trainGpuEval').checked }
}
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
function renderTraining() {
  renderCycle();
  $('deviceChip').textContent = `dispositivo: ${OPS.device || 'cpu'}`;
  fillSelect($('trainData'), OPS.datasets.map(dataset => [dataset.path, `${NAMING.dataset(dataset.name)} · ${Object.entries(dataset.splits || {}).map(([k, v]) => `${k} ${v}`).join(' / ')}`]), (OPS.datasets[OPS.datasets.length - 1] || {}).path);
  fillSelect($('trainWeights'), [['', `Padrão do projeto (${NAMING.weights(OPS.base_model || 'yolo26l.pt')})`], ...OPS.weights.map(path => [path, weightLabel(path)])], '');
  const defaults = OPS.pilot_defaults || {};
  ['epochs', 'imgsz', 'batch'].forEach(key => { const input = $('train' + key[0].toUpperCase() + key.slice(1)); if (input.value === '' && defaults[key] != null) input.value = defaults[key] });
  updateTrainCommand();
  $('startTrain').disabled = !SERVER || jobRunning();
  $('trainServerHint').textContent = SERVER
    ? 'O treino roda no servidor local deste PC; feche a página à vontade, o processo continua.'
    : 'Aberta direto do disco, a página não inicia processos: copie o comando no terminal da pasta sistema_ia ou use o atalho Abrir plataforma DroneCamp.cmd.';
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
  const run = OPS.runs.find(value => value.id === state.run) || OPS.runs.find(value => value.curves.length) || OPS.runs[0];
  renderRunDetail(run)
}
function renderRunDetail(run) {
  $('runDetailTitle').textContent = run ? `Detalhes · ${runName(run)}` : 'Detalhes da execução';
  $('runDetailChips').replaceChildren(...(run ? Object.entries(run.parameters).map(([key, value]) => el('span', { class: 'chip neutral', style: 'margin:2px' }, `${key} ${value}`)) : []));
  lineChart($('runLossChart'), run ? run.curves : [], LOSS_SERIES, { label: 'Perdas por época' });
  lineChart($('runMetricChart'), run ? run.curves : [], METRIC_SERIES, { percent: true, marker: run && run.best_epoch ? run.best_epoch.epoch : null, label: 'Métricas de validação por época' });
  const plotNames = { results: 'Curvas (Ultralytics)', confusion: 'Matriz de confusão normalizada', pr_curve: 'Curva precisão × recall', val_pred: 'Predições na validação', labels: 'Distribuição dos rótulos' };
  // Caminhos relativos à pasta da revisão; pelo servidor, /runs/... é servido de sistema_ia/runs.
  const plotUrl = path => SERVER ? '/' + path.replace(/^(\.\.\/)+/, '') : path;
  $('runPlots').replaceChildren(...(run ? Object.entries(run.plots).map(([key, path]) => el('a', { href: plotUrl(path), target: '_blank', rel: 'noopener' }, el('img', { src: plotUrl(path), alt: plotNames[key] || key, loading: 'lazy' }), el('span', {}, plotNames[key] || key))) : []))
}
['trainData', 'trainWeights', 'trainEpochs', 'trainImgsz', 'trainBatch', 'trainTile', 'trainGpuEval'].forEach(id => $(id).addEventListener('input', updateTrainCommand));

/* ======================= Monitoramento ======================= */
let job = { state: 'idle' };
const jobRunning = () => job.state === 'running';
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
    const planned = run.epochs_planned || run.epochs_done || 1;
    const fraction = Math.min(1, run.epochs_done / planned);
    $('ringValue').setAttribute('stroke-dashoffset', (339.3 * (1 - fraction)).toFixed(1));
    $('ringText').textContent = `${Math.round(fraction * 100)}%`;
    $('ringSub').textContent = `época ${run.epochs_done}/${planned}`;
    const elapsed = run.last && run.last.time;
    const remaining = run.state === 'running' && run.seconds_per_epoch ? run.seconds_per_epoch * (planned - run.epochs_done) : null;
    const facts = [['Tempo decorrido', duration(elapsed)], ['Tempo restante', run.state === 'running' ? duration(remaining) : '—'], ['Por época', duration(run.seconds_per_epoch)], ['Dispositivo', run.device || OPS.device || '—'], ['Pesos iniciais', run.checkpoint || '—'], ['Última escrita', run.updated_at ? new Date(run.updated_at).toLocaleTimeString('pt-BR') : '—']];
    $('monitorFacts').replaceChildren(...facts.flatMap(([key, value]) => [el('dt', {}, key), el('dd', {}, value)]));
    const last = run.curves[run.curves.length - 1] || {}, before = run.curves[run.curves.length - 2] || {};
    $('monitorTiles').replaceChildren(...METRIC_SERIES.map(serie => {
      const value = last[serie.key], delta = value != null && before[serie.key] != null ? value - before[serie.key] : null;
      return el('div', { class: 'tile' }, el('span', {}, serie.label), el('b', {}, pct(value)), el('span', { class: 'delta ' + (delta > 0 ? 'up' : delta < 0 ? 'down' : '') }, delta == null ? '—' : `${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(1)} p.p.`))
    }))
  }
  lineChart($('liveLossChart'), run ? run.curves : [], LOSS_SERIES);
  lineChart($('liveMetricChart'), run ? run.curves : [], METRIC_SERIES, { percent: true, marker: run && run.best_epoch ? run.best_epoch.epoch : null });
  fillSelect($('suggestWeights'), OPS.weights.filter(path => path.startsWith('runs/')).map(path => [path, weightLabel(path)]), (OPS.runs.find(value => value.weights.best) || { weights: {} }).weights.best);
  if ($('suggestConf').value === '' && OPS.suggestion_conf) $('suggestConf').value = OPS.suggestion_conf;
  renderJob()
}
function renderJob() {
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
function renderModels() {
  const pilots = OPS.runs.filter(run => run.stage === 'pilot_train');
  const independent = OPS.runs.some(run => run.independent_evaluation === true);
  const buildings = [...new Set(DATA.images.map(photo => photo.building_group).filter(Boolean))];
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
  fillSelect($('compareSelect'), OPS.comparisons.map((comparison, index) => [String(index), NAMING.comparison(comparison.file)]), String(OPS.comparisons.length - 1));
  renderCompare();
  const withCounts = OPS.runs.filter(run => Object.keys(run.class_counts || {}).length);
  fillSelect($('coverageRun'), withCounts.map(run => [run.id, `${runName(run)} · ${fmtDate(run.started_at)}`]), (withCounts[0] || {}).id);
  renderCoverage();
  const ranked = OPS.runs.filter(run => run.best_epoch).sort((a, b) => b.best_epoch.map50 - a.best_epoch.map50);
  const max = Math.max(0.01, ...ranked.map(run => run.best_epoch.map50));
  $('leaderboard').replaceChildren(...(ranked.length ? ranked.map(run => barRow('var(--ok)', run.display_detail ? `${runName(run)} · ${run.display_detail}` : runName(run), run.best_epoch.map50, 0, max, pct(run.best_epoch.map50))) : [el('div', { class: 'empty' }, 'Sem métricas.')]))
}
function renderCompare() {
  const comparison = OPS.comparisons[Number($('compareSelect').value)];
  if (!comparison) { $('compareTable').replaceChildren(el('div', { class: 'empty' }, 'Nenhuma comparação salva. Etapa 5 do ciclo de fine-tuning.')); return }
  const splits = [...new Set(comparison.models.flatMap(model => Object.keys(model.splits)))];
  const head = el('tr', {}, el('th', {}, 'Modelo'), ...splits.map(split => el('th', { class: 'num' }, splitLabel(split))));
  const rows = comparison.models.map(model => el('tr', {}, el('td', { title: model.weights }, el('div', {}, el('b', {}, model.run_id ? runLabel(model.run_id) : NAMING.weights(model.weights))), el('small', { class: 'muted' }, `${model.run_id && runById(model.run_id).display_detail ? runById(model.run_id).display_detail + ' · ' : ''}${model.file} · ${String(model.weights_sha256 || '').slice(0, 10)}…`)),
    ...splits.map(split => {
      const value = model.splits[split];
      if (!value) return el('td', { class: 'num muted' }, '—');
      const recall = value.truth ? value.found / value.truth : null, precision = value.suggestions ? value.correct / value.suggestions : null;
      return el('td', { class: 'num' }, el('div', {}, `Recall ${pct(recall)}`, el('small', { class: 'muted' }, ` · ${value.found} de ${value.truth} caixas encontradas`)), el('div', {}, `Precisão ${pct(precision)}`, el('small', { class: 'muted' }, ` · ${value.correct} de ${value.suggestions} sugestões certas`)))
    })));
  $('compareTable').replaceChildren(el('table', {}, el('thead', {}, head), el('tbody', {}, ...rows)),
    el('p', { class: 'muted small-text', style: 'padding:8px 12px' }, `${comparison.dataset ? NAMING.dataset(datasetFolder(comparison.dataset)) : 'Dataset —'} · confiança mínima ${pct(comparison.conf)} · uma sugestão conta como certa com sobreposição (IoU) ≥ ${comparison.iou_match}.`))
}
function renderCoverage() {
  const run = OPS.runs.find(value => value.id === $('coverageRun').value);
  if (!run) { $('coverageTable').replaceChildren(el('div', { class: 'empty' }, 'Sem contagens por classe registradas.')); return }
  const splits = Object.keys(run.class_counts);
  const classes = [...new Set(splits.flatMap(split => Object.keys(run.class_counts[split])))];
  // class_counts segue a ordem dos IDs da taxonomia (mesma ordem de DATA.classes).
  const label = slug => (DATA.classes[classes.indexOf(slug)] || {}).label || slug;
  const rows = classes.map(slug => el('tr', {}, el('td', {}, label(slug)), ...splits.map(split => { const value = run.class_counts[split][slug] || 0; return el('td', { class: 'num ' + (value === 0 ? 'cell-bad' : value < 5 ? 'cell-warn' : '') }, String(value)) })));
  $('coverageTable').replaceChildren(el('table', {}, el('thead', {}, el('tr', {}, el('th', {}, 'Classe'), ...splits.map(split => el('th', { class: 'num' }, splitLabel(split))))), el('tbody', {}, ...rows)),
    el('p', { class: 'muted small-text', style: 'padding:8px 12px' }, 'Vermelho: sem exemplos (a classe não é aprendida). Amarelo: menos de 5 caixas.'))
}
$('compareSelect').onchange = renderCompare;
$('coverageRun').onchange = renderCoverage;

/* ======================= Classes (cadastro) ======================= */
// Aberta do disco, a lista vem das classes embutidas; pelo servidor, de /api/classes (editável).
let CLASSES = null;
let editingClass = null;
function classRows() {
  if (CLASSES) return CLASSES.classes;
  return DATA.classes.map(category => ({ id: category.id, name: category.slug || '', display_name: category.label, color: category.color, active: category.active !== false, phase: 1, in_detector: !category.requires_training, requires_training: Boolean(category.requires_training), source: 'taxonomia', name_editable: false }))
}
function classSituation(entry) {
  if (!entry.active) return el('span', { class: 'chip neutral' }, 'inativa');
  if (entry.in_detector) return el('span', { class: 'chip ok' }, 'no detector');
  const title = 'Só passa a ser detectada depois de dados aprovados e novo treino.';
  if (entry.source === 'proposta') return el('span', { class: 'chip warn', title }, 'proposta · exige treino');
  return el('span', { class: 'chip warn', title }, entry.phase === 1 ? 'ativa · exige treino' : 'catálogo · exige treino')
}
async function classRequest(method, path, body) {
  const response = await fetch(path, { method, headers: { 'Content-Type': 'application/json', 'X-DroneCamp-Token': TOKEN }, body: body === undefined ? undefined : JSON.stringify(body) });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value
}
async function loadClasses() {
  if (!SERVER) return;
  try { CLASSES = await classRequest('GET', '/api/classes') } catch (error) { $('classesStatus').textContent = error.message }
}
function classStatus(text, error = false) { $('classesStatus').textContent = text; $('classesStatus').classList.toggle('error', error) }
async function saveClass(id, body, done) {
  try {
    await classRequest('PUT', `/api/classes/${id}`, { ...body, author: $('reviewer').value || undefined });
    await loadClasses(); editingClass = null; renderClasses();
    toast('Classe atualizada', done)
  } catch (error) { classStatus(error.message, true) }
}
function renderClasses() {
  const rows = classRows();
  $('classesOffline').classList.toggle('hidden', SERVER);
  $('classesHint').textContent = CLASSES ? `${CLASSES.summary.in_detector} no detector atual · ${CLASSES.summary.requires_training} exigem treino · ${CLASSES.summary.inactive} inativas` : `${rows.length} classes embutidas nesta página`;
  $('classesVersion').textContent = CLASSES ? CLASSES.taxonomy_version || '' : '';
  $('createClass').disabled = !SERVER;
  const canEdit = SERVER && CLASSES;
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
function newClassValid() { return SERVER && /^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$/.test($('newClassName').value) && $('newClassDisplay').value.trim().length > 0 }
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

/* ======================= Servidor local (opcional) ======================= */
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-DroneCamp-Token': TOKEN }, body: JSON.stringify(body) });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value
}
function setMode(connected) {
  $('modeChip').className = 'status ' + (connected ? 'ok' : 'bad');
  $('modeChip').lastElementChild.textContent = connected ? 'Servidor local conectado' : 'Servidor local desconectado'
}
async function refreshWorkspace() {
  if (!SERVER) return;
  try {
    WORKSPACE = await api('/api/workspace' + (DATA.version ? '?review=' + encodeURIComponent(DATA.version) : ''));
    fillReviewPicker(WORKSPACE.reviews.filter(review => review.page_ready !== false).map(review => ({ name: review.name, display_name: review.display_name, href: review.href, images: review.images })), DATA.version);
    $('navInspections').textContent = WORKSPACE.reviews.length || '';
    if (state.view === 'home' || state.view === 'inspections') showView(state.view)
  } catch {}
}
async function refreshOperations() {
  try {
    OPS = await api('/api/operations' + (DATA.version ? '?review=' + encodeURIComponent(DATA.version) : ''));
    job = await api('/api/jobs');
    if (state.view === 'monitor') renderMonitor();
    else if (state.view === 'training') renderTraining();
    else if (state.view === 'models') renderModels();
    else if (state.view === 'home') renderHome();
    renderJob();
    $('liveDot').classList.toggle('hidden', !OPS.runs.some(value => value.state === 'running') && !jobRunning());
    setMode(true)
  } catch { setMode(false) }
}
async function startJob(kind, options) {
  try {
    job = await api('/api/jobs', { kind, options });
    renderJob(); showView('monitor');
    toast('Tarefa iniciada', `${kind} rodando no servidor local. Acompanhe o log e as curvas.`)
  } catch (error) { message(error.message, true); $('jobLog').textContent = error.message; alert(error.message) }
}
$('startTrain').onclick = () => {
  const options = trainOptions();
  if (!confirm(`Iniciar treino piloto com ${options.epochs || '?'} épocas em ${OPS.device || 'cpu'}? O processo usa o PC até terminar.`)) return;
  startJob('train-pilot', options)
};
$('startSuggest').onclick = () => startJob('suggest', { review: REVIEWS ? REVIEWS.current : DATA.version, weights: $('suggestWeights').value, conf: $('suggestConf').value === '' ? null : Number($('suggestConf').value) });
$('stopJob').onclick = async () => {
  if (!confirm('Interromper a tarefa em andamento?')) return;
  try { job = await api('/api/jobs/stop', {}); renderJob() } catch (error) { alert(error.message) }
};
$('suggestWeights').onchange = renderJob;

/* ======================= Inicialização ======================= */
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('dronecamp-theme', theme) } catch {}
  $('themeToggle').replaceChildren(icon(theme === 'dark' ? 'i-sun' : 'i-moon'), el('span', {}, theme === 'dark' ? 'Claro' : 'Escuro'));
  if (state.view !== 'review') showView(state.view); else paint()
}
$('themeToggle').onclick = () => applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
(() => { let theme = 'light'; try { theme = localStorage.getItem('dronecamp-theme') || 'light' } catch {} document.documentElement.dataset.theme = theme; $('themeToggle').replaceChildren(icon(theme === 'dark' ? 'i-sun' : 'i-moon'), el('span', {}, theme === 'dark' ? 'Claro' : 'Escuro')) })();

$('versionLabel').textContent = DATA.version ? reviewLabel(DATA.version, DATA.display_name) : '';
$('versionLabel').title = DATA.version || '';
// Inativas continuam na lista (desabilitadas) para que uma caixa existente dessa classe mostre e preserve a categoria.
DATA.classes.forEach(category => $('classId').append(el('option', category.active === false ? { value: category.id, disabled: 'disabled' } : { value: category.id }, category.active === false ? `${category.label} (inativa)` : category.label)));
if (DATA.suggestions) {
  $('confFilter').value = Math.round(DATA.suggestions.conf * 100);
  $('confFilter').min = Math.round(DATA.suggestions.conf * 100);
  $('confValue').textContent = `${$('confFilter').value}%`
}
if (DATA.taxonomy_recheck_pending) {
  // Os nomes vêm do catálogo e do histórico; não manter categorias antigas fixas no HTML.
  const categories = (DATA.classes_requiring_review || DATA.classes.map(value => value.label)).join('; ');
  $('taxonomyNotice').textContent = `Categorias que ainda precisam de conferência integral: ${categories}. Confira essas ocorrências em todas as fotos antes de registrar uma aprovação.`;
  $('taxonomyNotice').classList.remove('hidden')
}
$('reviewEmpty').classList.toggle('hidden', HAS_IMAGES);
$('reviewGrid').classList.toggle('hidden', !HAS_IMAGES);
$('aiNotice').classList.toggle('hidden', !HAS_IMAGES);
$('export').disabled = !HAS_IMAGES;
if (HAS_IMAGES) openPhoto(0);
let initialView = location.hash.slice(1);
if (!VIEWS[initialView]) { try { initialView = sessionStorage.getItem('dronecamp-view') || 'home' } catch { initialView = 'home' } }
if (!HAS_IMAGES && initialView === 'review') initialView = SERVER ? 'inspections' : 'home';
showView(initialView);
addEventListener('hashchange', () => { const view = location.hash.slice(1); if (VIEWS[view] && view !== state.view) showView(view) });
if (SERVER) { refreshOperations(); refreshWorkspace(); loadClasses().then(() => { if (state.view === 'classes') renderClasses() }); setInterval(refreshOperations, 5000); setInterval(refreshWorkspace, 30000) }
