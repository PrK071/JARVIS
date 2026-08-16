/**
 * JARVIS — interface com simulação de modelo ternário (−1, 0, +1)
 */

const TERNARY = { NEG: -1, NEU: 0, POS: 1 };

const AVATAR_FRAMES = [
  { src: 'assets/black-visor/head/black-visor-head-00.png', label: 'Vista frontal' },
  { src: 'assets/black-visor/head/black-visor-head-01.png', label: 'Três quartos direito' },
  { src: 'assets/black-visor/head/black-visor-head-06.png', label: 'Perfil direito' },
  { src: 'assets/black-visor/head/black-visor-head-03.png', label: 'Três quartos traseiro direito' },
  { src: 'assets/black-visor/head/black-visor-head-04.png', label: 'Vista traseira' },
  { src: 'assets/black-visor/head/black-visor-head-05.png', label: 'Três quartos traseiro esquerdo' },
  { src: 'assets/black-visor/head/black-visor-head-02.png', label: 'Perfil esquerdo' },
  { src: 'assets/black-visor/head/black-visor-head-07.png', label: 'Três quartos esquerdo' },
];

const FRONT_FRAME = 0;
const TURN_PIXELS_PER_FRAME = 52;
const TURN_KEY_DURATION = 280;
const TURN_SETTLE_DURATION = 180;
const MAX_CHAT_HISTORY = 16;
const MAX_INPUT_LENGTH = 4000;

const state = {
  processing: false,
  history: [],
  distribution: { neg: 33, neu: 34, pos: 33 },
  visualPhase: 'idle',
  avatarMode: 'core',
  viewRevision: 0,
  interactionId: 0,
  avatarReady: false,
  utterance: null,
  speechWatchdog: null,
  voices: [],
  messages: [],
  lastResponse: '',
  voiceEnabled: true,
  speechWave: {
    active: false,
    energy: 0,
    boundaryAt: 0,
    startedAt: 0,
    fallbackPulseAt: 0,
    seed: 0,
  },
  gestures: {
    active: false,
    generation: 0,
    timer: null,
    animations: [],
    boundaryAt: 0,
    lastAt: 0,
  },
  turntable: {
    frame: FRONT_FRAME,
    angle: FRONT_FRAME,
    targetAngle: null,
    animationFrame: null,
    pointerId: null,
    lastX: 0,
  },
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
const avatarRig = $('#avatar-rig');
const avatarHead = $('#avatar-head');
const avatarView = $('#avatar-view');
const avatarTurntable = $('#avatar-turntable');
const turntableFrames = $('#turntable-frames');
const showAvatarBtn = $('#show-avatar-btn');
const returnCoreBtn = $('#return-core-btn');
const coreSection = document.querySelector('.core-section');
const waveform = $('#waveform');
const quickButtons = [...document.querySelectorAll('.quick-btn')];
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const speechSupported = 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window;

// ─── Init ───
function init() {
  buildTernaryGrid();
  buildParticles();
  initWaveform();
  initCoreCanvas();
  initAvatar();
  initVoices();
  updateClock();
  setInterval(updateClock, 1000);
  setInterval(updateTelemetry, 2000);
  setInterval(randomizeGrid, 1500);

  sendBtn.addEventListener('click', handleSubmit);
  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') handleSubmit();
  });

  userInput.addEventListener('input', () => {
    if (userInput.value.trim() && state.avatarMode === 'core' && !state.processing) {
      openAvatarManually();
    }
  });

  quickButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      userInput.value = btn.dataset.cmd;
      handleSubmit();
    });
  });

  showAvatarBtn.addEventListener('click', openAvatarManually);
  returnCoreBtn.addEventListener('click', returnToCore);
  initTurntableControls();

  animateWaveform();
  animateCore();
  window.addEventListener('pagehide', stopActiveSpeech);
  reducedMotion.addEventListener('change', () => {
    if (reducedMotion.matches) stopSpeechGestures(false);
  });
}

// ─── Avatar & voice ───
function initAvatar() {
  const markReady = () => {
    state.avatarReady = true;
    coreContainer.classList.add('avatar-ready');
  };

  const markUnavailable = () => {
    state.avatarReady = false;
    coreContainer.classList.remove('avatar-ready');
  };

  if (avatarHead.complete) {
    avatarHead.naturalWidth > 0 ? markReady() : markUnavailable();
  } else {
    avatarHead.addEventListener('load', markReady, { once: true });
    avatarHead.addEventListener('error', markUnavailable, { once: true });
  }

  AVATAR_FRAMES.forEach((frame, index) => {
    const image = document.createElement('img');
    image.className = 'turntable-frame';
    image.src = frame.src;
    image.alt = '';
    image.draggable = false;
    image.dataset.frame = String(index);
    turntableFrames.appendChild(image);
  });

  updateTurntableFrame(FRONT_FRAME, false);
}

function syncAvatarView() {
  const avatarVisible = state.avatarMode !== 'core';
  coreContainer.dataset.view = avatarVisible ? 'avatar' : 'core';
  coreSection.dataset.focus = avatarVisible ? 'avatar' : 'core';
  avatarView.setAttribute('aria-hidden', String(!avatarVisible));
  showAvatarBtn.setAttribute('aria-expanded', String(avatarVisible));
  avatarTurntable.tabIndex = avatarVisible ? 0 : -1;
  returnCoreBtn.disabled = state.processing;
}

function normalizeTurntableAngle(angle) {
  return ((angle % AVATAR_FRAMES.length) + AVATAR_FRAMES.length) % AVATAR_FRAMES.length;
}

function cancelTurntableAnimation() {
  if (state.turntable.animationFrame !== null) cancelAnimationFrame(state.turntable.animationFrame);
  state.turntable.animationFrame = null;
  state.turntable.targetAngle = null;
}

function renderTurntableAngle(angle) {
  const normalized = normalizeTurntableAngle(angle);
  const lowerFrame = Math.floor(normalized);
  const upperFrame = (lowerFrame + 1) % AVATAR_FRAMES.length;
  const progress = normalized - lowerFrame;
  const blend = progress * progress * (3 - 2 * progress);
  const lowerOpacity = Math.cos(blend * Math.PI / 2);
  const upperOpacity = Math.sin(blend * Math.PI / 2);
  const primaryFrame = progress < 0.5 ? lowerFrame : upperFrame;
  const degrees = Math.round(normalized * (360 / AVATAR_FRAMES.length)) % 360;

  state.turntable.angle = angle;
  state.turntable.frame = primaryFrame;

  turntableFrames.querySelectorAll('.turntable-frame').forEach((frame, index) => {
    const isLower = index === lowerFrame;
    const isUpper = index === upperFrame && progress > 0.0001;
    const opacity = isLower ? lowerOpacity : (isUpper ? upperOpacity : 0);
    frame.style.opacity = opacity.toFixed(4);
    frame.classList.toggle('is-primary', index === primaryFrame);
    frame.classList.toggle('is-secondary', (isLower || isUpper) && index !== primaryFrame);
  });

  avatarTurntable.classList.toggle('is-turning', normalized > 0.0001 && normalized < AVATAR_FRAMES.length - 0.0001);
  avatarTurntable.dataset.angle = String(degrees);
  avatarTurntable.setAttribute('aria-valuenow', String(primaryFrame));
  avatarTurntable.setAttribute('aria-valuetext', `${AVATAR_FRAMES[primaryFrame].label}, ${degrees} graus`);

  if (primaryFrame !== FRONT_FRAME && state.gestures.active) stopSpeechGestures();
}

function animateTurntableTo(targetAngle, duration = TURN_KEY_DURATION) {
  const startAngle = state.turntable.angle;
  const delta = targetAngle - startAngle;
  cancelTurntableAnimation();

  if (reducedMotion.matches || Math.abs(delta) < 0.0001) {
    renderTurntableAngle(targetAngle);
    return;
  }

  const startedAt = performance.now();
  state.turntable.targetAngle = targetAngle;

  const tick = (now) => {
    const progress = Math.min(1, (now - startedAt) / duration);
    const eased = progress * progress * (3 - 2 * progress);
    renderTurntableAngle(startAngle + delta * eased);

    if (progress < 1) {
      state.turntable.animationFrame = requestAnimationFrame(tick);
      return;
    }

    state.turntable.animationFrame = null;
    state.turntable.targetAngle = null;
    renderTurntableAngle(targetAngle);
  };

  state.turntable.animationFrame = requestAnimationFrame(tick);
}

function nearestAngleForFrame(frameIndex) {
  const targetFrame = normalizeTurntableAngle(frameIndex);
  const currentFrame = normalizeTurntableAngle(state.turntable.angle);
  let delta = targetFrame - currentFrame;
  if (delta > AVATAR_FRAMES.length / 2) delta -= AVATAR_FRAMES.length;
  if (delta < -AVATAR_FRAMES.length / 2) delta += AVATAR_FRAMES.length;
  return state.turntable.angle + delta;
}

function updateTurntableFrame(frameIndex, animated = true) {
  const targetAngle = nearestAngleForFrame(frameIndex);
  if (animated) animateTurntableTo(targetAngle);
  else {
    cancelTurntableAnimation();
    renderTurntableAngle(targetAngle);
  }
}

function turnTurntableBy(frameDelta) {
  const baseAngle = state.turntable.targetAngle ?? state.turntable.angle;
  animateTurntableTo(baseAngle + frameDelta);
}

function resetTurntableToFront() {
  updateTurntableFrame(FRONT_FRAME, false);
  avatarTurntable.classList.remove('is-dragging');
}

async function openAvatarManually() {
  if (!state.avatarReady || state.processing || state.avatarMode !== 'core') return;

  const revision = ++state.viewRevision;
  state.avatarMode = 'manual';
  resetTurntableToFront();
  setVisualPhase('summoning');
  coreState.textContent = 'SYNTH-ALPHA';
  coreConfidence.textContent = '';
  syncAvatarView();

  await wait(reducedMotion.matches ? 140 : 760);
  if (revision !== state.viewRevision || state.avatarMode !== 'manual' || state.processing) return;
  setVisualPhase('present');
}

async function returnToCore() {
  if (state.processing || state.avatarMode === 'core') return;

  ++state.viewRevision;
  setVisualPhase('dismissing');
  coreState.textContent = 'STANDBY';
  coreConfidence.textContent = 'Retornando ao núcleo ternário...';
  await wait(reducedMotion.matches ? 120 : 430);

  state.avatarMode = 'core';
  resetTurntableToFront();
  setVisualPhase('idle');
  coreState.textContent = 'STANDBY';
  coreConfidence.textContent = '—';
  syncAvatarView();
}

function initTurntableControls() {
  const finishDrag = (event) => {
    if (state.turntable.pointerId === null) return;
    if (event?.pointerId !== undefined && event.pointerId !== state.turntable.pointerId) return;

    try {
      if (avatarTurntable.hasPointerCapture(state.turntable.pointerId)) {
        avatarTurntable.releasePointerCapture(state.turntable.pointerId);
      }
    } catch (_error) {
      // The browser can drop pointer capture when the pointer leaves the window.
    }

    state.turntable.pointerId = null;
    avatarTurntable.classList.remove('is-dragging');
    animateTurntableTo(Math.round(state.turntable.angle), TURN_SETTLE_DURATION);
  };

  avatarTurntable.addEventListener('pointerdown', (event) => {
    if (state.avatarMode === 'core' || state.processing) return;
    if (event.pointerType === 'mouse' && event.button !== 0) return;
    if (state.turntable.pointerId !== null) return;

    cancelTurntableAnimation();
    state.turntable.pointerId = event.pointerId;
    state.turntable.lastX = event.clientX;
    avatarTurntable.setPointerCapture(event.pointerId);
    avatarTurntable.classList.add('is-dragging');
  });

  avatarTurntable.addEventListener('pointermove', (event) => {
    if (event.pointerId !== state.turntable.pointerId) return;
    const deltaX = event.clientX - state.turntable.lastX;
    state.turntable.lastX = event.clientX;
    renderTurntableAngle(state.turntable.angle + deltaX / TURN_PIXELS_PER_FRAME);
  });

  avatarTurntable.addEventListener('pointerup', finishDrag);
  avatarTurntable.addEventListener('pointercancel', finishDrag);
  avatarTurntable.addEventListener('lostpointercapture', finishDrag);
  window.addEventListener('blur', () => finishDrag());

  avatarTurntable.addEventListener('keydown', (event) => {
    if (state.avatarMode === 'core' || state.processing) return;
    if (event.key === 'ArrowRight') {
      event.preventDefault();
      turnTurntableBy(1);
    } else if (event.key === 'ArrowLeft') {
      event.preventDefault();
      turnTurntableBy(-1);
    } else if (event.key === 'Home') {
      event.preventDefault();
      resetTurntableToFront();
    } else if (event.key === 'Escape') {
      event.preventDefault();
      returnToCore();
    }
  });
}

function initVoices() {
  if (!speechSupported) return;

  const refreshVoices = () => {
    state.voices = window.speechSynthesis.getVoices();
  };

  refreshVoices();
  window.speechSynthesis.addEventListener('voiceschanged', refreshVoices);
}

function selectPortugueseVoice() {
  return state.voices.find((voice) => voice.lang.toLowerCase() === 'pt-br')
    || state.voices.find((voice) => voice.lang.toLowerCase().startsWith('pt'))
    || state.voices.find((voice) => voice.default)
    || null;
}

function setVisualPhase(phase, ternaryResult = null) {
  const keys = { [TERNARY.NEG]: 'neg', [TERNARY.NEU]: 'neu', [TERNARY.POS]: 'pos' };
  state.visualPhase = phase;
  coreContainer.dataset.phase = phase;

  if (ternaryResult !== null) {
    coreContainer.dataset.ternary = keys[ternaryResult] || 'neu';
  } else if (phase === 'idle') {
    coreContainer.dataset.ternary = 'neu';
  }

  syncAvatarView();
}

function setControlsEnabled(enabled) {
  userInput.disabled = !enabled;
  sendBtn.disabled = !enabled;
  quickButtons.forEach((button) => { button.disabled = !enabled; });
}

function normalizeCommand(value) {
  return value
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9+\-\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function currentTelemetry() {
  return {
    convergence: $('#convergence').textContent,
    temperature: $('#core-temp').textContent,
    latency: $('#latency').textContent,
  };
}

function helpResponse() {
  return [
    'Comandos do JARVIS:',
    '• status, diagnóstico, análise, distribuição e telemetria',
    '• temperatura, latência, convergência, neurônios e estados ternários',
    '• histórico, limpar terminal, limpar histórico e reiniciar núcleo',
    '• ativar voz, desativar voz, repetir resposta e parar fala',
    '• mostrar Synth-Alpha, voltar ao círculo, olhar à direita/esquerda, mostrar costas ou frente',
    'Você também pode escrever perguntas livres para conversar com o modelo.',
  ].join('\n');
}

function resolveLocalCommand(input, inference) {
  const command = normalizeCommand(input);
  const telemetry = currentTelemetry();
  const negatedRestart = /\b(nao|nunca|jamais)\b.*\b(reinici|reset)/.test(command);

  if (/^(ajuda|comandos|help)$/.test(command) || /o que (voce )?(faz|pode fazer)/.test(command)) {
    return { reply: helpResponse(), updateInference: false };
  }

  if (/\b(limpar|apagar)\b.*\bterminal\b/.test(command)) {
    return {
      reply: 'Terminal limpo. Aguardando nova entrada.',
      beforeResponse() {
        logContainer.replaceChildren();
      },
      updateInference: false,
    };
  }

  if (/\b(limpar|apagar|zerar)\b.*\bhistorico\b/.test(command)) {
    return {
      reply: 'Histórico de inferências apagado.',
      beforeResponse() {
        inferenceHistory.replaceChildren();
        state.history = [];
      },
      updateInference: false,
    };
  }

  if (!negatedRestart && /\b(reinici|resetar|reset)\w*\b/.test(command)) {
    return {
      reply: 'Núcleo ternário reiniciado. Distribuição, histórico e estado visual restaurados.',
      beforeResponse() {
        state.history = [];
        state.messages = [];
        state.distribution = { neg: 33, neu: 34, pos: 33 };
        inferenceHistory.replaceChildren();
        updateDecisionBars();
        randomizeGrid();
      },
      updateInference: false,
    };
  }

  if (/\b(desativar|desligar|silencio)\b.*\bvoz\b|^silencio$/.test(command)) {
    return {
      reply: 'Voz desativada. As respostas continuarão aparecendo no Terminal de Resposta.',
      beforeResponse() {
        state.voiceEnabled = false;
        stopActiveSpeech();
      },
      speak: false,
      updateInference: false,
    };
  }

  if (/\b(ativar|ligar)\b.*\bvoz\b/.test(command)) {
    return {
      reply: 'Voz ativada. As próximas respostas serão faladas.',
      beforeResponse() { state.voiceEnabled = true; },
      updateInference: false,
    };
  }

  if (/^(pare|parar|cancele|cancelar)( a)? (fala|voz)$/.test(command)) {
    return {
      reply: 'Síntese de voz interrompida.',
      beforeResponse() { stopActiveSpeech(); },
      speak: false,
      updateInference: false,
    };
  }

  if (/\b(repetir|repita)\b/.test(command)) {
    return {
      reply: state.lastResponse || 'Ainda não há uma resposta anterior para repetir.',
      updateInference: false,
    };
  }

  if (/\b(mostrar|aparecer|apareca)\b.*\b(synth|robo|cabeca|avatar)\b/.test(command)) {
    return {
      reply: 'Synth-Alpha em exibição.',
      finalAvatarMode: 'manual',
      afterLog() {
        state.avatarMode = 'manual';
        resetTurntableToFront();
        setVisualPhase('present');
        syncAvatarView();
      },
      updateInference: false,
    };
  }

  if (/\b(voltar|ocultar|esconder|desaparecer)\b.*\b(circulo|nucleo|synth|robo|avatar)\b/.test(command)) {
    return {
      reply: 'Retornando ao núcleo ternário.',
      finalAvatarMode: 'core',
      afterLog() {
        state.avatarMode = 'core';
        resetTurntableToFront();
        setVisualPhase('idle');
        syncAvatarView();
      },
      updateInference: false,
    };
  }

  if (/\b(frente|frontal)\b/.test(command) && /\b(olhar|mostrar|vista|virar|girar)\w*\b/.test(command)) {
    return { reply: 'Vista frontal selecionada.', afterLog: () => updateTurntableFrame(0, false), updateInference: false };
  }
  if (/\b(costas|traseira)\b/.test(command) && /\b(olhar|mostrar|vista|virar|girar)\w*\b/.test(command)) {
    return { reply: 'Vista traseira selecionada.', afterLog: () => updateTurntableFrame(4, true), updateInference: false };
  }
  if (/\b(direita)\b/.test(command) && /\b(olhar|mostrar|vista|virar|girar)\w*\b/.test(command)) {
    return { reply: 'Perfil direito selecionado.', afterLog: () => updateTurntableFrame(2, true), updateInference: false };
  }
  if (/\b(esquerda)\b/.test(command) && /\b(olhar|mostrar|vista|virar|girar)\w*\b/.test(command)) {
    return { reply: 'Perfil esquerdo selecionado.', afterLog: () => updateTurntableFrame(6, true), updateInference: false };
  }

  if (/\b(status|sistema online|como esta o sistema)\b/.test(command)) {
    return { reply: `Status do sistema: ONLINE. Neurônios ternários: 4.096 ativos. Convergência: ${telemetry.convergence}. Núcleo: ${telemetry.temperature}.` };
  }
  if (/\b(diagnostico|falha|verificar o sistema)\b/.test(command)) {
    return { reply: `Diagnóstico completo: 12 camadas neurais OK. Memória ternária: 98% livre. Latência média: ${telemetry.latency}. Nenhuma falha detectada.` };
  }
  if (/\b(telemetria|metricas)\b/.test(command)) {
    return { reply: `Telemetria: núcleo ${telemetry.temperature}; convergência ${telemetry.convergence}; latência ${telemetry.latency}; 12 de 12 camadas e 4.096 neurônios ativos.` };
  }
  if (/\btemperatura\b/.test(command)) return { reply: `Temperatura atual do núcleo: ${telemetry.temperature}.`, updateInference: false };
  if (/\blatencia\b/.test(command)) return { reply: `Latência atual: ${telemetry.latency}.`, updateInference: false };
  if (/\bconvergencia\b/.test(command)) return { reply: `Taxa de convergência atual: ${telemetry.convergence}.`, updateInference: false };
  if (/\b(neuronios|camadas)\b/.test(command)) return { reply: 'Telemetria neural: 4.096 neurônios ternários ativos em 12 de 12 camadas.', updateInference: false };
  if (/\b(memoria)\b/.test(command)) return { reply: 'Memória ternária simulada: 98% livre.', updateInference: false };
  if (/\b(distribuicao|probabilidades|percentuais)\b/.test(command)) {
    return { reply: `Distribuição atual: −1 ${state.distribution.neg}%, 0 ${state.distribution.neu}%, +1 ${state.distribution.pos}%.`, updateInference: false };
  }
  if (/\b(estados ternarios|significa.*(\-1|\+1|neutro)|inibicao|ativacao)\b/.test(command)) {
    return { reply: 'Estados ternários: −1 representa inibição ou alerta; 0 representa neutralidade e equilíbrio; +1 representa ativação ou confirmação.', updateInference: false };
  }
  if (/\b(historico|ultimas inferencias)\b/.test(command)) {
    const entries = [...inferenceHistory.querySelectorAll('.query')].slice(0, 5).map((item) => `• ${item.textContent}`);
    return { reply: entries.length ? `Últimas inferências:\n${entries.join('\n')}` : 'O histórico de inferências está vazio.', updateInference: false };
  }
  if (/\b(versao|sobre o sistema)\b/.test(command)) return { reply: 'JARVIS v2.4.1-TER — interface Synth-Alpha e motor ternário experimental.', updateInference: false };
  if (/\b(horas|hora atual|que horas)\b/.test(command)) return { reply: `Hora atual: ${$('#clock').textContent}.`, updateInference: false };
  if (/\b(analise|analisar|inferir|inferencia)\b/.test(command)) {
    return {
      reply: `Análise ternária: −1 ${inference.distribution.neg}%, 0 ${inference.distribution.neu}%, +1 ${inference.distribution.pos}%. Confiança: ${inference.confidence}%.`,
    };
  }

  return null;
}

async function requestModelResponse(message) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 50_000);

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history: state.messages.slice(-MAX_CHAT_HISTORY) }),
      signal: controller.signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Falha HTTP ${response.status}.`);
    if (typeof data.reply !== 'string' || !data.reply.trim()) throw new Error('O modelo retornou uma resposta vazia.');
    return data.reply.trim();
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('O modelo demorou demais para responder.');
    if (location.protocol === 'file:') {
      throw new Error('Conversa livre requer o servidor web. Execute python web_server.py e abra http://localhost:8000.');
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function estimateSpeechDuration(text) {
  const words = text.trim().split(/\s+/).filter(Boolean).length;
  return Math.max(2200, Math.min(11500, words * 380));
}

function rememberGestureAnimation(animation) {
  state.gestures.animations.push(animation);
  animation.finished.catch(() => {}).finally(() => {
    state.gestures.animations = state.gestures.animations.filter((item) => item !== animation);
  });
  return animation;
}

function animateGesturePart(element, keyframes, options) {
  if (!element || reducedMotion.matches) return null;
  if (element.getAnimations().some((animation) => animation.playState === 'running')) return null;
  return rememberGestureAnimation(element.animate(keyframes, {
    duration: options.duration,
    easing: options.easing || 'cubic-bezier(0.22, 0.75, 0.2, 1)',
    fill: 'none',
  }));
}

function performSpeechGesture(intensity = 0.55, spontaneous = false) {
  if (!state.gestures.active || reducedMotion.matches || state.turntable.frame !== FRONT_FRAME) return;

  const direction = Math.random() < 0.5 ? -1 : 1;
  const tilt = (0.45 + intensity * 0.75) * direction;
  const lift = 0.55 + intensity * 0.9;
  const duration = 820 + Math.random() * 280;
  const gesture = Math.floor(Math.random() * 3);
  const patterns = [
    [
      { transform: 'none' },
      { transform: `translateY(${-lift}px) rotate(${tilt * 0.18}deg)`, offset: 0.38 },
      { transform: `translateY(0.2px) rotate(${-tilt * 0.08}deg)`, offset: 0.76 },
      { transform: 'none' },
    ],
    [
      { transform: 'none' },
      { transform: `rotate(${tilt}deg) translateX(${direction * 0.65}px)`, offset: 0.4 },
      { transform: `rotate(${tilt * 0.2}deg) translateY(-0.35px)`, offset: 0.78 },
      { transform: 'none' },
    ],
    [
      { transform: 'none' },
      { transform: `translateY(${-lift * 0.72}px) rotate(${-tilt * 0.38}deg)`, offset: 0.42 },
      { transform: `translateY(-0.15px) rotate(${tilt * 0.12}deg)`, offset: 0.78 },
      { transform: 'none' },
    ],
  ];

  const gestureTarget = turntableFrames.querySelector('.turntable-frame.is-primary');
  animateGesturePart(gestureTarget, patterns[gesture], {
    duration: spontaneous ? duration + 180 : duration,
    easing: 'cubic-bezier(0.25, 0.68, 0.24, 1)',
  });
}

function scheduleSpontaneousGesture(generation) {
  if (!state.gestures.active || generation !== state.gestures.generation) return;
  const delay = 1800 + Math.random() * 1500;
  state.gestures.timer = setTimeout(() => {
    if (!state.gestures.active || generation !== state.gestures.generation) return;
    if (performance.now() - state.gestures.boundaryAt > 900) {
      performSpeechGesture(0.42 + Math.random() * 0.34, true);
    }
    scheduleSpontaneousGesture(generation);
  }, delay);
}

function startSpeechGestures() {
  stopSpeechGestures(false);
  if (reducedMotion.matches) return;
  state.gestures.active = true;
  state.gestures.boundaryAt = performance.now();
  state.gestures.lastAt = 0;
  const generation = ++state.gestures.generation;
  performSpeechGesture(0.46, true);
  scheduleSpontaneousGesture(generation);
}

function pulseSpeechGesture(intensity) {
  if (!state.gestures.active) return;
  const now = performance.now();
  state.gestures.boundaryAt = now;
  if (now - state.gestures.lastAt < 440) return;
  state.gestures.lastAt = now;
  performSpeechGesture(intensity, false);
}

function stopSpeechGestures(settle = true) {
  state.gestures.active = false;
  state.gestures.generation += 1;
  if (state.gestures.timer) clearTimeout(state.gestures.timer);
  state.gestures.timer = null;
  state.gestures.lastAt = 0;
  state.gestures.animations.forEach((animation) => animation.cancel());
  state.gestures.animations = [];
  if (settle) {
    [turntableFrames.querySelector('.turntable-frame.is-primary')].forEach((element) => {
      if (!element || reducedMotion.matches) return;
      element.animate([
        { transform: getComputedStyle(element).transform },
        { transform: 'none' },
      ], { duration: 220, easing: 'ease-out' });
    });
  }
}

function startSpeechWave() {
  const now = performance.now();
  state.speechWave.active = true;
  state.speechWave.energy = 0.42;
  state.speechWave.boundaryAt = now;
  state.speechWave.startedAt = now;
  state.speechWave.fallbackPulseAt = now + 360;
  state.speechWave.seed = Math.random() * Math.PI * 2;
  waveform.classList.add('is-speaking');
}

function pulseSpeechWave(intensity = 0.8) {
  if (!state.speechWave.active) return;
  state.speechWave.energy = Math.min(1, Math.max(state.speechWave.energy, intensity));
  state.speechWave.boundaryAt = performance.now();
}

function stopSpeechWave() {
  state.speechWave.active = false;
  state.speechWave.energy = 0;
  state.speechWave.boundaryAt = 0;
  state.speechWave.startedAt = 0;
  state.speechWave.fallbackPulseAt = 0;
  waveform.classList.remove('is-speaking');
  waveData.fill(0);
}

function stopActiveSpeech() {
  if (state.speechWatchdog) {
    clearTimeout(state.speechWatchdog);
    state.speechWatchdog = null;
  }

  stopSpeechWave();
  stopSpeechGestures();
  state.utterance = null;
  if (speechSupported) window.speechSynthesis.cancel();
}

function speakResponse(text, interactionId) {
  const estimatedDuration = estimateSpeechDuration(text);

  return new Promise((resolve) => {
    let settled = false;
    const settle = () => {
      if (settled) return;
      settled = true;
      if (state.speechWatchdog) clearTimeout(state.speechWatchdog);
      state.speechWatchdog = null;
      stopSpeechWave();
      stopSpeechGestures();
      state.utterance = null;
      resolve();
    };

    if (!speechSupported) {
      startSpeechWave();
      startSpeechGestures();
      state.speechWatchdog = setTimeout(settle, estimatedDuration);
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    const voice = selectPortugueseVoice();
    if (voice) utterance.voice = voice;
    utterance.lang = voice?.lang || 'pt-BR';
    utterance.rate = 0.95;
    utterance.pitch = 0.88;
    utterance.volume = 1;

    utterance.onstart = () => {
      if (interactionId !== state.interactionId) {
        window.speechSynthesis.cancel();
        settle();
        return;
      }
      startSpeechWave();
      startSpeechGestures();
    };
    utterance.onboundary = (event) => {
      if (state.utterance !== utterance || interactionId !== state.interactionId) return;
      const nearbyWord = text.slice(event.charIndex, event.charIndex + 18).match(/^\S+/)?.[0] || '';
      const intensity = 0.45 + Math.min(0.5, nearbyWord.length * 0.04);
      pulseSpeechWave(0.58 + Math.min(0.38, nearbyWord.length * 0.035));
      pulseSpeechGesture(intensity);
    };
    utterance.onpause = () => {
      stopSpeechWave();
      stopSpeechGestures();
    };
    utterance.onresume = () => {
      startSpeechWave();
      startSpeechGestures();
    };
    utterance.onend = settle;
    utterance.onerror = settle;
    state.utterance = utterance;
    state.speechWatchdog = setTimeout(() => {
      window.speechSynthesis.cancel();
      settle();
    }, estimatedDuration + 3500);

    try {
      window.speechSynthesis.speak(utterance);
    } catch (_error) {
      clearTimeout(state.speechWatchdog);
      startSpeechWave();
      startSpeechGestures();
      state.speechWatchdog = setTimeout(settle, estimatedDuration);
    }
  });
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
  waveData = new Array(100).fill(0);
}

function resizeWaveCanvas() {
  const canvas = $('#wave-canvas');
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
}

function animateWaveform() {
  const now = performance.now();
  waveData.shift();

  if (state.speechWave.active) {
    const boundarySilentFor = now - state.speechWave.boundaryAt;
    if (boundarySilentFor > 450 && now >= state.speechWave.fallbackPulseAt) {
      pulseSpeechWave(0.56 + Math.random() * 0.34);
      state.speechWave.fallbackPulseAt = now + 250 + Math.random() * 210;
    }

    const cadence = Math.sin(now / 34 + state.speechWave.seed) * 0.28
      + Math.sin(now / 17 + state.speechWave.seed * 0.7) * 0.18;
    const noise = (Math.random() - 0.5) * 0.44;
    const sample = Math.max(-1, Math.min(1, (cadence + noise) * state.speechWave.energy));
    waveData.push(sample);
    state.speechWave.energy = Math.max(0.24, state.speechWave.energy * 0.955);
  } else {
    waveData.push(0);
  }

  const w = $('#wave-canvas').width;
  const h = $('#wave-canvas').height;
  waveCtx.clearRect(0, 0, w, h);
  waveCtx.beginPath();
  waveCtx.strokeStyle = state.speechWave.active ? '#00e5ff' : 'rgba(0, 212, 255, 0.38)';
  waveCtx.lineWidth = state.speechWave.active ? 1.8 : 1;
  waveCtx.shadowBlur = state.speechWave.active ? 7 : 0;
  waveCtx.shadowColor = '#00d4ff';

  waveData.forEach((v, i) => {
    const x = (i / waveData.length) * w;
    const y = h / 2 + v * h * 0.42;
    i === 0 ? waveCtx.moveTo(x, y) : waveCtx.lineTo(x, y);
  });
  waveCtx.stroke();
  waveCtx.shadowBlur = 0;

  requestAnimationFrame(animateWaveform);
}

// ─── Core canvas (ternary visualization) ───
let coreCtx, coreAngle = 0;

function initCoreCanvas() {
  const canvas = $('#core-canvas');
  coreCtx = canvas.getContext('2d');
}

function animateCore() {
  if (state.avatarReady && state.avatarMode !== 'core') {
    requestAnimationFrame(animateCore);
    return;
  }

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

  return {
    result,
    confidence: confidence.toFixed(1),
    distribution: { ...state.distribution },
  };
}

// ─── UI updates ───
function addLog(type, message) {
  const entry = document.createElement('div');
  entry.className = `log-entry ${type}`;
  const ts = new Date().toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const timestamp = document.createElement('span');
  timestamp.className = 'timestamp';
  timestamp.textContent = type === 'processing' ? '[PROCESSANDO]' : `[${ts}]`;
  const messageNode = document.createElement('span');
  messageNode.className = 'message';
  messageNode.textContent = message;
  entry.append(timestamp, messageNode);
  logContainer.appendChild(entry);
  logContainer.scrollTop = logContainer.scrollHeight;
  return entry;
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
  const query = document.createElement('span');
  query.className = 'query';
  query.textContent = `${input.substring(0, 20)}${input.length > 20 ? '…' : ''}`;
  const result = document.createElement('span');
  result.className = `result ${classes[inference.result]}`;
  result.textContent = labels[inference.result];
  li.append(query, result);
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
  if (input.length > MAX_INPUT_LENGTH) {
    addLog('error', `A mensagem deve ter no máximo ${MAX_INPUT_LENGTH} caracteres.`);
    return;
  }

  const keepAvatarOpen = state.avatarMode === 'manual';
  let localCommand = null;
  const interactionId = ++state.interactionId;
  ++state.viewRevision;
  state.processing = true;
  if (!keepAvatarOpen) state.avatarMode = 'request';
  userInput.value = '';
  setControlsEnabled(false);
  stopActiveSpeech();
  resetTurntableToFront();
  setVisualPhase('summoning');
  coreState.textContent = 'PROCESSANDO';
  coreConfidence.textContent = 'Materializando interface neural...';

  addLog('user', input);
  const processingEntry = addLog('processing', 'Interpretando comando e consultando o modelo...');

  try {
    await wait(reducedMotion.matches ? 140 : 560);
    if (interactionId !== state.interactionId) return;

    setVisualPhase('thinking');
    coreConfidence.textContent = 'Analisando vetores ternários...';
    await wait(320 + Math.random() * 480);
    if (interactionId !== state.interactionId) return;

    const previousDistribution = { ...state.distribution };
    const inference = inferTernary(input);
    state.distribution = previousDistribution;
    localCommand = resolveLocalCommand(input, inference);
    const response = localCommand ? localCommand.reply : await requestModelResponse(input);
    if (interactionId !== state.interactionId) return;

    if (!localCommand || localCommand.updateInference !== false) {
      state.distribution = { ...inference.distribution };
    }

    localCommand?.beforeResponse?.();
    processingEntry.remove();
    addLog('response', response);
    localCommand?.afterLog?.();
    state.lastResponse = response;

    if (!localCommand) {
      state.messages.push({ role: 'user', content: input }, { role: 'assistant', content: response });
      state.messages = state.messages.slice(-MAX_CHAT_HISTORY);
    }

    if (localCommand?.updateInference !== false) {
      updateDecisionBars();
      updateCoreLabel(inference);
      addInferenceHistory(input, inference);
      flashGrid(inference.result);
    }

    setVisualPhase('speaking', inference.result);
    if (state.voiceEnabled && localCommand?.speak !== false) {
      await speakResponse(response, interactionId);
    } else {
      await wait(reducedMotion.matches ? 120 : 360);
    }
    if (interactionId !== state.interactionId) return;

    const remainVisible = localCommand?.finalAvatarMode === 'manual'
      || (keepAvatarOpen && localCommand?.finalAvatarMode !== 'core');
    if (remainVisible) {
      setVisualPhase('present', inference.result);
    } else {
      setVisualPhase('dismissing', inference.result);
      coreState.textContent = 'CONCLUÍDO';
      coreConfidence.textContent = 'Retornando ao núcleo ternário...';
      await wait(reducedMotion.matches ? 140 : 560);
    }
  } catch (error) {
    processingEntry.remove();
    const message = error instanceof Error ? error.message : 'Não foi possível processar a mensagem.';
    addLog('error', message);
    state.lastResponse = message;
    coreState.textContent = 'ERRO';
    coreConfidence.textContent = 'Verifique o Terminal de Resposta';
    await wait(reducedMotion.matches ? 140 : 720);
  } finally {
    if (interactionId === state.interactionId) {
      state.processing = false;
      const remainVisible = localCommand?.finalAvatarMode === 'manual'
        || (keepAvatarOpen && localCommand?.finalAvatarMode !== 'core');
      if (remainVisible) {
        state.avatarMode = 'manual';
        setVisualPhase('present');
      } else {
        state.avatarMode = 'core';
        resetTurntableToFront();
        setVisualPhase('idle');
        coreState.textContent = 'STANDBY';
        coreConfidence.textContent = '—';
      }
      syncAvatarView();
      setControlsEnabled(true);
      userInput.focus();
    }
  }
}

// Boot
init();
