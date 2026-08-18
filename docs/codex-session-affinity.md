# Codex Session Affinity — auditoria e validação

Data: 2026-08-12. Escopo: Jarvis ↔ Codex. O `explicit_agent_binding.py`
aprovado não foi alterado.

## 1. Root cause

A perda ocorria depois do agent binding:

1. `delegate_to_codex` expunha `task`, `project_path`, continuidade e espera, mas
   nenhuma identidade de thread.
2. `CodexRunner` não descobria threads do provider. Dependia de um único
   `.orchestrator/codex-session.json`.
3. `continue_session(session_id=...)` recebia identidade, mas não a encaminhava
   ao manager.
4. Manager, jobs, foco da conversa e UI guardavam identidades independentes.
5. Thread criada por App Server era persistida no ponteiro legado, mas não
   existia registry canônica nem exposição do ID na UI.

Classificação: A + B + E, com lacuna de lifecycle F/G. A API nativa aceita
resume; portanto C não é causa.

O trace histórico corrige uma percepção importante: o job
`4b046e6e-8010-4145-885c-c79dd479fa31` não criou thread nova. Ele usou a thread
Jarvis antiga `019fbbb0-7ba1-7631-835c-229147e9316c`. Ela não era a sessão
Codex atualmente focada pelo usuário e não estava bem exposta pela UI.

A sessão standalone atual `019ff619-1e8e-7e31-8d8f-a224df6ae550` tinha cwd
`C:\Users\User`, enquanto a tarefa apontava para `D:\JARVIS`, e ainda possuía
writer ativo. Reutilizá-la seria cross-project e dual-writer; foi corretamente
rejeitada. Uma sessão aberta por `jarvis codex` usa o mesmo App Server e pode
ser reutilizada com segurança.

## 2. Modelo real de identidade

| Identity | Created by | Stored in | UI knows? | Reusable? | Persistent? |
|---|---|---|---|---|---|
| `thread_id` | Codex `thread/start`, ou provider preexistente | provider, `codex-sessions.json`, job e ponteiro legado | sim; Jarvis mostra ID e Codex cli/vscode mostra suas fontes | sim, por `thread/resume` + novo `turn/start` | sim |
| `session_id` | Codex; `thread.sessionId` identifica raiz da árvore | provider + registry | não é o ID principal exibido | não é usado para seleção; metadado da árvore | sim |
| `turn_id` | Codex `turn/start` | provider, runtime e job | visível em status/eventos | retomável para monitorar execução; não equivale a sessão | sim |
| `job_id` | Jarvis `CodexJobStore` | `codex-jobs.json` | sim, status/CLI | não; representa uma delegação | sim |
| `request_id` | Jarvis resolver | job + observabilidade | logs | não | sim nos logs/job |
| `conversation_id` | `Supervisor` Jarvis | binding da registry | não exibido | chave de affinity enquanto a conversa existe | mapping persiste; ID de conversa não sobrevive reinício |
| UI session/conversation ID | inexistente como identidade compartilhada | — | — | — | — |
| `run_id` | inexistente neste bridge | — | — | — | — |

`client_message_id` é idempotência/transporte, não sessão. Neste backend,
thread é a conversa persistente, turn é uma solicitação e item é uma unidade
dentro do turn. Contrato oficial: <https://developers.openai.com/codex/app-server/>.

## 3. Estados reais

Provider: `notLoaded`, `idle`, `active`, `systemError`. O resolver acrescenta
`stale` quando uma identidade registrada deixa de ser encontrada. Threads
ephemeral e arquivadas não são candidatas.

Jobs Jarvis: `queued`, `starting`, `running`, `steering`, `cancelling`,
`disconnected`, `reconnecting`, `interrupted`, `completed`, `failed`.

Reutilizável: provider `idle` ou `notLoaded`, recoverable, não ephemeral e cwd
exatamente igual após normalização. `active` externo retorna `SESSION_BUSY`.
`active` com `active_job_id` do próprio Jarvis entra na fila serializada da
thread. `steer` continua operação distinta e nunca substitui delegate.

## 4. Arquitetura anterior

```text
USER -> explicit binding -> delegate(task, project)
     -> CodexRunner -> único codex-session.json
     -> ensure_thread(create/resume pointer) -> turn -> job

UI focus -------- estrutura independente
Provider threads - não enumeradas
```

## 5. Arquitetura final

```text
USER -> explicit binding (intacto)
     -> requested_agent=codex
     -> CodexSessionResolver
          ^ focused/conversation/project affinity
          ^ CodexSessionRegistry (canônica para Jarvis)
          ^ provider thread/list + thread/read
     -> thread/resume ou thread/start
     -> registrar/verificar recoverability
     -> job_id -> turn/start na thread exata
     -> UI mostra binding de thread
```

O ponteiro `codex-session.json` permanece só como compatibilidade do manager e
comando `jarvis codex`; não decide affinity no Jarvis.

## 6. Registry e lifecycle

`CodexSessionRegistry` persiste atomicamente em
`.orchestrator/codex-sessions.json`. Campos usados: `thread_id`, `session_id`,
`project`/`project_key`, endpoint, state, source, timestamps, origin,
visible, recoverable, ephemeral e active_job_id; bindings de conversa/projeto
ficam no mesmo arquivo.

Origens: `user_existing` e `jarvis_created`. Nenhum fluxo fecha, apaga, reseta
ou limpa história de uma sessão adotada.

Criação é transacional até o limite do provider:

```text
thread/start -> thread/read -> registry atomic write -> bind -> job -> turn/start
```

Falha de registry impede job e `turn/start`. O provider não expõe delete de
thread no contrato usado; logo não há rollback remoto seguro. A identidade é
logada, mas trabalho do usuário nunca é enviado para uma thread não registrada.

## 7. Resolver e precedência

1. `thread_id` explícito.
2. thread Codex focada no contexto Jarvis.
3. binding da mesma `conversation_id`.
4. binding persistido do mesmo projeto.
5. única thread reutilizável do projeto.
6. múltiplas equivalentes: `AMBIGUOUS_SESSION`, sem escolha e sem criação.
7. nenhuma reutilizável e criação permitida: criar/registrar/verificar.

Matching usa UUID exato, cwd normalizado, affinity persistida, ownership e
state. Não usa título, substring, semântica, embeddings ou “última global”.

Reason codes: `EXPLICIT_SESSION_MATCH`, `FOCUSED_SESSION_MATCH`,
`CONVERSATION_AFFINITY_MATCH`, `PROJECT_AFFINITY_MATCH`,
`UNIQUE_PROJECT_SESSION_MATCH`, `NO_REUSABLE_SESSION`, `AMBIGUOUS_SESSION`,
`SESSION_STALE`, `SESSION_BUSY`, `SESSION_BUSY_QUEUED`,
`SESSION_UNAVAILABLE`, `NEW_SESSION_CREATED`,
`SESSION_REGISTRATION_FAILED`, `SESSION_NOT_RECOVERABLE`.

## 8. Baseline

Baseline histórico relevante, duas solicitações reais do usuário:

| Métrica | A, antes |
|---|---:|
| correct session rate | 0/2 = 0% |
| reuse da sessão visualmente solicitada | 0/2 = 0% |
| session resolution success | 0% (resolver inexistente) |
| delegation success | 1/2 = 50% |
| criação desnecessária comprovada | 0; o job percebido como novo usou thread antiga |
| ghost com trabalho | 0 comprovado; havia ponteiro recuperável, mas visibility/affinity eram falhas |

## 9. A/B determinístico

| Caso | A: ponteiro único | B: registry/resolver |
|---|---|---|
| C1 única existente | pode ignorar se ponteiro ausente/diverge | reutiliza única |
| C2 duas, uma focada | foco não chega ao delegate | focada vence |
| C3 duas ambíguas | pode criar/usar ponteiro | erro estruturado; não cria |
| C4 projeto diferente | dependente do state dir | nunca reutiliza cross-project |
| C5 nenhuma | cria e só grava ponteiro | cria, verifica, registra, expõe |
| C6 stale | tentativa tardia/fallback de criação | exclui stale |
| C7 busy | erro do provider durante execução | busy externo recusa; próprio job serializa |
| C8 restart | depende do ponteiro único | provider reconciliation + project binding |
| C9 visibility | sem registry UI | registry + UI thread ID |
| C10 registry failure | caso inexistente | nenhum trabalho enviado |
| C11 concorrência | duas criações possíveis | mutex por project_key; no máximo uma |
| C12 user-created | sem discovery/adoption | provider-native resume; ownership preservado |

## 10. Replay real: sessão existente

Projeto `D:\JARVIS`. IDs provider antes:

```text
019fbbb0-7ba1-7631-835c-229147e9316c
019fb698-69f1-7f62-a105-1b7ec41a67ce
```

TUI compartilhada confirmou thread `019fbbb0...`, 1 cliente. Ordem explícita
via Jarvis produziu:

```text
requested_agent=codex
semantic_pass.used=false
qwen_requests=0
selected_thread_id=019fbbb0-7ba1-7631-835c-229147e9316c
reason=PROJECT_AFFINITY_MATCH
session_reused=true
session_created=false
job_id=4702e9a8-da8e-4fb5-9269-97a476199e5b
turn_id=019ff6b0-759b-7a02-8b2b-e2df7bb2d2ff
response=SESSION-AFFINITY-LIVE-OK
```

IDs depois: exatamente os mesmos dois. Delta de threads: zero.

## 11. Replay real: criação, visibility e segundo reuse

Projeto isolado sem candidata:
`D:\JARVIS\.orchestrator\session-affinity-live-project`.

```text
before=[]
created thread=019ff6b1-a5ec-74b1-98df-bd862b4ba424
job 1=f51b93b4-d8c9-4ad1-8bf7-6d425f17d31f
registered=true recoverable=true visible=true
response 1=NEW-SESSION-FIRST-OK
shared TUI PID=9088, same thread
job 2=aa4721ad-ba5e-4feb-b86b-7d479028e91a
reason 2=CONVERSATION_AFFINITY_MATCH
created 2=false reused 2=true
response 2=NEW-SESSION-SECOND-OK
after=[019ff6b1-a5ec-74b1-98df-bd862b4ba424]
```

Nova instância de `CodexRunner` simulando restart resolveu a mesma thread por
`PROJECT_AFFINITY_MATCH`, com `created=false`, `recoverable=true`,
`visible=true`.

## 12. Ambiguidade, projeto, stale e busy

- Duas candidatas sem binding: `AMBIGUOUS_SESSION`; nenhum terceiro ID.
- Cwd A para request B: candidata excluída; cross-project reuse = 0.
- Stale/unrecoverable: excluída; explicit stale retorna `SESSION_STALE`.
- Busy externo: `SESSION_BUSY`, sem hidden-session workaround.
- Busy Jarvis-owned: fila da mesma thread; `turn/start` serializado pelo mutex
  existente. `steer_codex_job` não é chamado pelo resolver.

## 13. Restart, concorrência e atomicidade

Registry e project binding usam JSON persistente com write temporário +
`os.replace`. Startup/primeiro uso reconcilia `thread/list` do provider,
incluindo fontes `cli`, `vscode`, `appServer`, cwd exato e paginação.

Lock de resolução é por hash do project path, não global. Race fake com duas
delegações simultâneas: duas tarefas concluídas, uma única thread criada.

## 14. Observabilidade e métricas B

Cada resolução/delegação registra request, agent, project, candidatos totais e
reutilizáveis, IDs selecionados, binding source, state, reuse/create/register,
recoverable/visible, active job, job ID, reason code e latência. Conteúdo da
tarefa não entra nesses eventos de resolução.

Três delegações reais medidas após a mudança (reuse existente, criação,
segundo reuse):

| Métrica | B |
|---|---:|
| correct_session_rate | 3/3 = 100% |
| existing_session_reuse_rate | 2/3 = 66,67% |
| unnecessary_new_session_rate | 0% |
| new session when reusable existed | 0 |
| ghost_session_rate | 0% |
| recoverability_rate | 100% |
| visibility_rate | 100% |
| session_resolution_success_rate | 100% |
| delegation_success_rate | 100% |
| wrong_session_rate | 0% |
| cross_project reuse | 0 |
| duplicate creation | 0 |

Resolution latency real: 1.650 ms (reuse), 2.025 ms (criação), 1.966 ms
(conversation reuse), 2.300 ms (restart reconciliation). Decision/policy do
explicit binding: 47,91 ms; Qwen requests = 0; extra inference = 0; tokens de
inferência adicionais = 0. Tempo total inclui o trabalho Codex, não somente
resolução.

## 15. Payload, agent binding e safety

O resolver só acrescenta identidade fora de `task`. Teste envia payload igual
com dois thread IDs e comprova bytes lógicos iguais. O replay registra a mesma
tarefa que o pipeline atual construiria.

Explicit Codex binding continua antes do resolver, pula semantic pass e Qwen,
e expõe somente `delegate_to_codex`. `explicit_agent_binding.py` não mudou.
DeepSeek, web, reference resolver, filesystem routing, approvals, PathPolicy e
tool safety não foram alterados por esta solução.

## 16. Arquivos e linhas principais

- `tern/orchestrator/codex_sessions.py:21`: result, registry, reconciliation e resolver.
- `tern/orchestrator/codex.py:673`: provider discovery/read/adopt/create.
- `tern/orchestrator/codex.py:2775`: contrato delegate com thread/context affinity.
- `tern/orchestrator/codex.py:2893`: transação resolve/create/register.
- `tern/orchestrator/codex.py:3424`: métricas.
- `tern/orchestrator/codex_jobs.py:60`: persistência request/thread/resolution por job.
- `tern/orchestrator/tools.py:763`: schema aceita thread ID exato.
- `tern/orchestrator/tools.py:1132`: forwarding sem alterar task.
- `tern/orchestrator/agent.py:230`: conversation affinity key.
- `tern/orchestrator/agent.py:768`: contexto confiável de foco/conversa.
- `tern/orchestrator/jarvis_ui.py:155`: UI lê binding canônico e mostra thread.
- `tests/test_codex_session_affinity.py:46`: C1–C12, restart e race.
- `tests/test_orchestrator.py:422`: payload preservation.
- `tests/data/codex_session_affinity_diagnostic.jsonl:1`: corpus separado; v2/v3 intactos, sem v4.

## 17. Testes

Resultado total final: `778 passed, 1 skipped, 1 warning`. Casos cobertos:
reuse único, explicit/focused,
conversation/project affinity, ambiguidade, cross-project, stale, busy,
create/register/recover, falha de registro, restart, concorrência,
user ownership, no ghost/duplicate, payload e explicit binding.

## 18. Decisão

Critérios comprovados: existente reutilizada; nova criada, registrada,
recuperável, visível e reutilizada; ambígua não escolhida; thread errada não
selecionada; nenhuma sessão com trabalho ficou ghost.

**APROVADO — CODEX SESSION AFFINITY**
