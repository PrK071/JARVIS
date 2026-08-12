/**
 * T.R.I.A.D.E — Ternary Reasoning & Intelligent Adaptive Decision Engine
 * Interface estilo JARVIS com simulação de modelo ternário (−1, 0, +1)
 */

const TERNARY = { NEG: -1, NEU: 0, POS: 1 };

const state = {
  processing: false,
  history: [],
  distribution: { neg: 33, neu: 34, pos: 33 },
};

// ─── DOM refs ───
const $ = (sel) => document.querySelector(sel);
const logContainer = $('#log-container');
const userInput = $('#user-input');
const sendBtn = $('#send-btn');
const coreContainer = document.querySelector('.core-container');
const coreState = $('#core-state');
const coreConfidence = $('#core-confidence');
const ternaryGrid = $('#ternary-grid');
const inferenceHistory = $('#inference-history');

// ─── Init ───
function init() {
  buildTernaryGrid();
  buildParticles();
  initWaveform();
  initCoreCanvas();
  updateClock();
  setInterval(updateClock, 1000);
  setInterval(updateTelemetry, 2000);
  setInterval(randomizeGrid, 1500);

  sendBtn.addEventListener('click', handleSubmit);
  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') handleSubmit();
  });

  document.querySelectorAll('.quick-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      userInput.value = btn.dataset.cmd;
      handleSubmit();
    });
  });

  animateWaveform();
  animateCore();
}

// ─── Clock & telemetry ───
function updateClock() {
  const now = new Date();
  $('#clock').textContent = now.toLocaleTimeString('pt-BR');
}

function updateTelemetry() {
  const temp = (36.5 + Math.random() * 2).toFixed(1);
  $('#core-temp').textContent = `${temp}°C`;
  $('#convergence').textContent = `${(95 + Math.random() * 4.9).toFixed(1)}%`;
  $('#latency').textContent = `${Math.floor(8 + Math.random() * 15)}ms`;
}

// ─── Ternary grid ───
function buildTernaryGrid() {
  ternaryGrid.innerHTML = '';
  for (let i = 0; i < 64; i++) {
    const cell = document.createElement('div');
    cell.className = 'ternary-cell';
    ternaryGrid.appendChild(cell);
  }
  randomizeGrid();
}

function randomizeGrid() {
  if (state.processing) return;
  const cells = ternaryGrid.querySelectorAll('.ternary-cell');
  cells.forEach((cell) => {
    const r = Math.random();
    cell.className = 'ternary-cell ' + (r < 0.33 ? 'neg' : r < 0.66 ? 'neu' : 'pos');
  });
}

// ─── Particles ───
function buildParticles() {
  const container = $('#particles');
  for (let i = 0; i < 12; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const angle = (i / 12) * Math.PI * 2;
    const radius = 140 + Math.random() * 20;
    p.style.left = `${160 + Math.cos(angle) * radius}px`;
    p.style.top = `${160 + Math.sin(angle) * radius}px`;
    p.style.animation = `spin ${8 + Math.random() * 6}s linear infinite`;
    p.style.transformOrigin = `${160 - parseFloat(p.style.left) + 2}px ${160 - parseFloat(p.style.top) + 2}px`;
    container.appendChild(p);
  }
}

// ─── Waveform canvas ───
let waveCtx, waveData = [];

function initWaveform() {
  const canvas = $('#wave-canvas');
  waveCtx = canvas.getContext('2d');
  resizeWaveCanvas();
  window.addEventListener('resize', resizeWaveCanvas);
  waveData = new Array(100).fill(0).map(() => Math.random() * 0.5);
}

function resizeWaveCanvas() {
  const canvas = $('#wave-canvas');
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
}

function animateWaveform() {
  waveData.shift();
  waveData.push(state.processing ? Math.random() * 0.9 + 0.1 : Math.random() * 0.4);

  const w = $('#wave-canvas').width;
  const h = $('#wave-canvas').height;
  waveCtx.clearRect(0, 0, w, h);
  waveCtx.beginPath();
  waveCtx.strokeStyle = state.processing ? '#ffd700' : '#00d4ff';
  waveCtx.lineWidth = 1.5;

  waveData.forEach((v, i) => {
    const x = (i / waveData.length) * w;
    const y = h / 2 + (v - 0.25) * h * 0.8;
    i === 0 ? waveCtx.moveTo(x, y) : waveCtx.lineTo(x, y);
  });
  waveCtx.stroke();

  requestAnimationFrame(animateWaveform);
}

// ─── Core canvas (ternary visualization) ───
let coreCtx, coreAngle = 0;

function initCoreCanvas() {
  const canvas = $('#core-canvas');
  coreCtx = canvas.getContext('2d');
}

function animateCore() {
  const canvas = $('#core-canvas');
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const r = 100;

  coreCtx.clearRect(0, 0, canvas.width, canvas.height);

  // Three arcs representing ternary states
  const segments = [
    { start: 0, end: (Math.PI * 2) / 3, color: '#ff3b5c', label: '−1' },
    { start: (Math.PI * 2) / 3, end: (Math.PI * 4) / 3, color: '#8899aa', label: '0' },
    { start: (Math.PI * 4) / 3, end: Math.PI * 2, color: '#00ff9d', label: '+1' },
  ];

  const weights = [
    state.distribution.neg / 100,
    state.distribution.neu / 100,
    state.distribution.pos / 100,
  ];

  segments.forEach((seg, i) => {
    coreCtx.beginPath();
    coreCtx.arc(cx, cy, r, seg.start + coreAngle, seg.end + coreAngle);
    coreCtx.strokeStyle = seg.color;
    coreCtx.lineWidth = 6 + weights[i] * 10;
    coreCtx.globalAlpha = 0.5 + weights[i] * 0.5;
    coreCtx.stroke();
    coreCtx.globalAlpha = 1;
  });

  // Inner pulse
  const pulse = 0.5 + Math.sin(Date.now() / 500) * 0.2;
  const grad = coreCtx.createRadialGradient(cx, cy, 0, cx, cy, 60);
  grad.addColorStop(0, `rgba(0, 212, 255, ${0.3 * pulse})`);
  grad.addColorStop(1, 'rgba(0, 212, 255, 0)');
  coreCtx.fillStyle = grad;
  coreCtx.beginPath();
  coreCtx.arc(cx, cy, 60, 0, Math.PI * 2);
  coreCtx.fill();

  // Center symbol
  coreCtx.font = 'bold 28px Orbitron, monospace';
  coreCtx.fillStyle = '#00d4ff';
  coreCtx.textAlign = 'center';
  coreCtx.textBaseline = 'middle';
  coreCtx.fillText('◈', cx, cy);

  coreAngle += state.processing ? 0.04 : 0.008;
  requestAnimationFrame(animateCore);
}

// ─── Ternary inference engine (simulated) ───
function inferTernary(input) {
  const text = input.toLowerCase().trim();

  // Keyword-based sentiment mapping to ternary
  const negWords = ['não', 'erro', 'falha', 'problema', 'negativo', 'ruim', 'alerta', 'crítico', 'reiniciar'];
  const posWords = ['sim', 'ok', 'bom', 'positivo', 'sucesso', 'ativo', 'online', 'pronto', 'ótimo'];
  const neuWords = ['status', 'diagnóstico', 'info', 'analise', 'análise', 'consulta', 'verificar'];

  let score = 0;
  negWords.forEach((w) => { if (text.includes(w)) score -= 1; });
  posWords.forEach((w) => { if (text.includes(w)) score += 1; });
  neuWords.forEach((w) => { if (text.includes(w)) score *= 0.5; });

  // Add noise for realism
  score += (Math.random() - 0.5) * 0.6;

  let result, confidence;
  if (score < -0.3) {
    result = TERNARY.NEG;
    confidence = Math.min(95, 60 + Math.abs(score) * 20);
  } else if (score > 0.3) {
    result = TERNARY.POS;
    confidence = Math.min(95, 60 + Math.abs(score) * 20);
  } else {
    result = TERNARY.NEU;
    confidence = Math.min(90, 55 + Math.random() * 25);
  }

  // Update distribution
  const total = 100;
  const spread = confidence / 100;
  state.distribution = {
    neg: result === TERNARY.NEG ? Math.round(spread * total) : Math.round((1 - spread) * total * 0.4),
    neu: result === TERNARY.NEU ? Math.round(spread * total) : Math.round((1 - spread) * total * 0.3),
    pos: result === TERNARY.POS ? Math.round(spread * total) : Math.round((1 - spread) * total * 0.3),
  };
  const sum = state.distribution.neg + state.distribution.neu + state.distribution.pos;
  state.distribution.neu += total - sum;

  return { result, confidence: confidence.toFixed(1) };
}

function getResponse(input, inference) {
  const responses = {
    [TERNARY.NEG]: [
      'Detectei anomalias no fluxo ternário. Recomendo verificação dos vetores de inibição.',
      'Estado −1 confirmado. Sistemas de segurança em alerta. Aguardando correção.',
      'Análise negativa concluída. Parâmetros fora do intervalo ideal.',
    ],
    [TERNARY.NEU]: [
      'Consulta processada. Estado neutro — nenhuma ação imediata necessária.',
      'Diagnóstico completo. Todos os subsistemas operando dentro dos parâmetros normais.',
      'Informação registrada. Modelo ternário em equilíbrio.',
    ],
    [TERNARY.POS]: [
      'Confirmação positiva. Todos os indicadores dentro do esperado. Sistemas operacionais.',
      'Estado +1 atingido. Otimização neural concluída com sucesso.',
      'Processamento concluído. Resposta favorável detectada nos neurônios ternários.',
    ],
  };

  const cmd = input.toLowerCase();
  if (cmd.includes('status')) {
    return `Status do sistema: ONLINE. Neurônios ternários: 4.096 ativos. Convergência: ${$('#convergence').textContent}. Núcleo: ${$('#core-temp').textContent}.`;
  }
  if (cmd.includes('diagnóstico') || cmd.includes('diagnostico')) {
    return 'Diagnóstico completo: 12 camadas neurais OK. Memória ternária: 98% livre. Latência média: 12ms. Nenhuma falha detectada.';
  }
  if (cmd.includes('reiniciar')) {
    return 'Reiniciando núcleo ternário... Sequência de boot concluída. Sistema restaurado ao estado inicial.';
  }
  if (cmd.includes('analise') || cmd.includes('análise')) {
    return `Análise ternária: distribuição atual −1: ${state.distribution.neg}%, 0: ${state.distribution.neu}%, +1: ${state.distribution.pos}%. Confiança: ${inference.confidence}%.`;
  }

  const pool = responses[inference.result];
  return pool[Math.floor(Math.random() * pool.length)];
}

// ─── UI updates ───
function addLog(type, message) {
  const entry = document.createElement('div');
  entry.className = `log-entry ${type}`;
  const ts = new Date().toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  entry.innerHTML = `<span class="timestamp">[${ts}]</span><span class="message">${message}</span>`;
  logContainer.appendChild(entry);
  logContainer.scrollTop = logContainer.scrollHeight;
}

function updateDecisionBars() {
  $('#bar-negative').style.width = `${state.distribution.neg}%`;
  $('#bar-neutral').style.width = `${state.distribution.neu}%`;
  $('#bar-positive').style.width = `${state.distribution.pos}%`;
  $('#pct-negative').textContent = `${state.distribution.neg}%`;
  $('#pct-neutral').textContent = `${state.distribution.neu}%`;
  $('#pct-positive').textContent = `${state.distribution.pos}%`;
}

function updateCoreLabel(inference) {
  const labels = { [TERNARY.NEG]: 'INIBIÇÃO', [TERNARY.NEU]: 'NEUTRO', [TERNARY.POS]: 'ATIVAÇÃO' };
  coreState.textContent = labels[inference.result];
  coreConfidence.textContent = `Confiança: ${inference.confidence}%`;
}

function addInferenceHistory(input, inference) {
  const labels = { [TERNARY.NEG]: '−1', [TERNARY.NEU]: '0', [TERNARY.POS]: '+1' };
  const classes = { [TERNARY.NEG]: 'neg', [TERNARY.NEU]: 'neu', [TERNARY.POS]: 'pos' };

  const li = document.createElement('li');
  li.innerHTML = `
    <span class="query">${input.substring(0, 20)}${input.length > 20 ? '…' : ''}</span>
    <span class="result ${classes[inference.result]}">${labels[inference.result]}</span>
  `;
  inferenceHistory.prepend(li);
  if (inferenceHistory.children.length > 8) {
    inferenceHistory.removeChild(inferenceHistory.lastChild);
  }
}

function flashGrid(result) {
  const classMap = { [TERNARY.NEG]: 'neg', [TERNARY.NEU]: 'neu', [TERNARY.POS]: 'pos' };
  const cells = ternaryGrid.querySelectorAll('.ternary-cell');
  cells.forEach((cell, i) => {
    setTimeout(() => {
      cell.className = `ternary-cell ${classMap[result]}`;
    }, i * 15);
  });
}

// ─── Submit handler ───
async function handleSubmit() {
  const input = userInput.value.trim();
  if (!input || state.processing) return;

  state.processing = true;
  userInput.value = '';
  coreContainer.classList.add('processing');
  coreState.textContent = 'PROCESSANDO';

  addLog('user', input);
  addLog('processing', 'Executando inferência ternária...');

  // Simulate processing delay
  await new Promise((r) => setTimeout(r, 800 + Math.random() * 700));

  const inference = inferTernary(input);
  const response = getResponse(input, inference);

  // Remove processing log
  const processingEntry = logContainer.querySelector('.log-entry.processing');
  if (processingEntry) processingEntry.remove();

  addLog('response', response);
  updateDecisionBars();
  updateCoreLabel(inference);
  addInferenceHistory(input, inference);
  flashGrid(inference.result);

  state.processing = false;
  coreContainer.classList.remove('processing');
}

// Boot
init();
