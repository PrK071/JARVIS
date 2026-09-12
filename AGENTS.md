# AGENTS.md — JARVIS

Guia de trabalho para agentes de IA (Codex, Claude, Gemini, etc.) e
contribuidores humanos no repositório JARVIS.

## O que é o JARVIS

Assistente Windows local: Qwen3.5-4B na GPU, arquivos controlados, Codex,
pesquisa web citada, STT faster-whisper e TTS (Microsoft Daniel pt-BR como
principal, Piper como fallback local). O pacote Python é `tern`; o orquestrador
fica em `tern/orchestrator/` e a interface HUD em `interface/`.

## Comandos essenciais

```powershell
# Configurar (uma vez)
python -m pip install --editable D:\JARVIS

# Sessões e perguntas
python -m tern.orchestrator start        # inicia llama-server (sem duplicar)
python -m tern.orchestrator ask "..."    # pergunta pontual ao Qwen
python -m tern.orchestrator voice        # sessão contínua de voz
python -m tern.orchestrator text         # sessão interativa digitada

# Codex (App Server local compartilhado)
python -m tern.orchestrator codex-shared-start
python -m tern.orchestrator codex-shared-tui
python -m tern.orchestrator codex-shared-status
python -m tern.orchestrator codex-jobs
python -m tern.orchestrator codex-history          # lista sessoes do Codex CLI
python -m tern.orchestrator codex-history 1 --turn-limit 5   # le a sessao 1

# Testes
python -m pytest -q
$env:RUN_VOICE_INTEGRATION_TESTS='true'; python -m pytest -q tests\test_voice_integration.py
```

A suíte completa deve permanecer verde antes de qualquer commit. Voz usa
`VOICE_*` no `.env` (ver `.env.example`). O modelo local fica em
`models/Qwen_Qwen3.5-4B-Q4_K_M.gguf` (ignorado pelo git).

## Arquitetura (pacote tern)

- `tern/orchestrator/cli.py` — todos os subcomandos (`build_parser` + dispatch).
- `tern/orchestrator/agent.py` — Supervisor: roteia o texto do usuário para
  Qwen, Codex, DeepSeek ou resposta direta; gates de histórico e de jobs.
- `tern/orchestrator/decision_policy.py` — política determinística de decisão:
  intents, tools, `TOOL_EFFECTS`, fast paths e restrições de execução.
- `tern/orchestrator/tools.py` — `ToolRegistry`: schemas OpenAI e handlers de
  todas as ferramentas allowlisted do Qwen.
- `tern/orchestrator/codex.py` — bridge Codex (App Server compartilhado, jobs,
  `review_session` via thread/read).
- `tern/orchestrator/codex_history.py` — leitura read-only das sessões do
  Codex CLI do usuário em `~/.codex/sessions/**.jsonl`.
- `tern/orchestrator/codex_sessions.py`, `codex_jobs.py`, `codex_state.py` —
  registro de threads, jobs e estado persistido do bridge.
- `tern/orchestrator/deepseek.py` — consultor DeepSeek opcional e stateless
  (sem chave, vira modo leitura).
- `tern/orchestrator/web.py`, `research.py` — pesquisa web com citação e
  validação de fontes.
- `tern/orchestrator/voice/` — STT faster-whisper, TTS Windows SAPI/Piper,
  sessão push-to-talk.
- `tern/orchestrator/projects.py`, `project_discovery.py`,
  `project_intelligence_v2.py` — descoberta e índice de projetos permitidos.
- `tern/orchestrator/predictive/` — análise preditiva de código: hipóteses de
  causa raiz, grounding de evidências, alvos de reparo tipados, grafo de
  imports e benchmarks congelados.
- `tern/orchestrator/execution_authority.py`, `execution_gate*.py`,
  `orchestration_*.py` — modos de orquestração (bounded live, shadow, replay)
  e matriz de risco das ações.
- `interface/` — HUD web/desktop (licença MIT própria; não confundir com o
  orquestrador).

## Convenções do código

- Strings de código, logs e mensagens em português **sem acentos** (ex.:
  `"sessao"`, `"nao foi possivel"`). Textos de interface e README usam acentos.
- Docstrings curtas explicam o "porquê"; comentários só quando o código não é
  autoexplicativo.
- Linhas razoavelmente curtas; sem dependências novas sem necessidade — o
  orquestrador roda CPU-only com o mínimo em `pyproject.toml`.
- Ferramentas novas entram em `ToolRegistry` e precisam de: efeito em
  `TOOL_EFFECTS`, política em `BoundedLiveRiskMatrix`, registro na rota de
  decisão e, quando cabível, orientação no `prompt.py`.
- Mudança de roteamento exige caso novo em `tests/test_decision_policy.py` e
  conferir `tests/data/agent_routing_cases.jsonl` (baseline congelado).

## O que já foi feito até aqui (log de funcionalidades)

1. **Núcleo tern** — quantização ternária GGUF e inferência local via
   llama.cpp (scripts de pack/quantize, kernel Vulkan opcional).
2. **Orquestrador Qwen** — CLI completa (`ask`, `voice`, `text`, `config`,
   `status`), decisão determinística + intérprete semântico Qwen, fast path
   e modos shadow/bounded-live com matriz de risco auditável.
3. **Bridge Codex compartilhado** — App Server local com thread persistida por
   projeto, jobs com status/steer/cancel, `review_session` (thread/read sem
   iniciar turn), TUI reaberta por `codex-shared-tui`.
4. **Histórico das conversas do usuário com o Codex CLI** — `codex-history`
   lista e lê `~/.codex/sessions` (ferramenta `read_codex_history` do Qwen;
   rota `codex_cli_history_query`). Somente leitura, sem iniciar o Codex.
5. **DeepSeek consultivo** — TUI persistente compartilhada, delegação com
   contexto, revisão de sessão sem chamar API.
6. **Pesquisa web citada** — intenção, expansão de consulta, pontuação,
   validação de páginas, correções e bloqueio de fontes inválidas.
7. **Voz** — STT faster-whisper local, TTS Windows (Daniel) com Piper
   fallback, comparador de vozes com CER/WER, dispositivos persistentes.
8. **Descoberta e inteligência de projetos** — allowlist de diretórios,
   índice leve de arquivos/símbolos, detecção de agentes externos instalados
   (Kiro, Claude, Gemini, GLM, Aider) com delegação registrada em runtime.
9. **Análise preditiva** — MVP de decisão preditiva com hipóteses de causa
   raiz, grounding de evidências, raciocínio causal, alvos de reparo tipados,
   grafo de imports com ciclos e benchmarks v4/v5 congelados em testes.
10. **Interface HUD** — web/desktop com avatar Synth-Alpha, entrada por voz e
    Terminal de Resposta (em `interface/`).

## Git e segurança

- Conventional Commits em inglês (`feat:`, `fix:`, `test:`, `docs:`),
  escopo quando fizer sentido (`feat(codex): ...`). Veja `git log --oneline`.
- **Nunca** commitar: `.env`, modelos (`*.gguf`), `runtime/`, `_arquivo/`
  (arquivo de recuperação), `interface/providers.json`, caches.
- **Autoria**: commits ficam com o autor configurado do usuário. Não adicionar
  trailers de coautoria de agentes e não alterar `git config`.
- Branch principal: `main` (remoto `origin`). `upstream` aponta para a
  interface original; push de código do orquestrador vai para `origin`.
- Toda ferramenta nova é somente leitura por padrão; mutação exige efeito
  declarado, política de risco e confirmação quando irreversível.
