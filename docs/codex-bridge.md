# Bridge compartilhado Qwen/Codex

## Arquitetura

Antes, `CodexRunner` criava um processo `codex exec --json` por delegacao. Cada
chamada tinha transporte e ciclo de vida proprios; a TUI aberta pelo usuario nao
recebia esses eventos. `codex exec resume` mantinha historico somente quando o
Qwen fornecia manualmente um `session_id`.

Agora, um unico Codex App Server local atende dois clientes:

```text
Codex App Server (ws://127.0.0.1:4500)
|- Codex TUI do usuario
`- CodexSessionManager do Jarvis/Qwen
```

`CodexSessionManager` inicializa o protocolo, valida ou retoma a thread
persistida, inicia turns, le eventos, captura a resposta final, interrompe turns
e reconecta. A ferramenta exposta ao Qwen e pequena:

```text
delegate_to_codex(task, project_path, continue_current_thread=true, wait=true)
```

O retorno inclui `accepted`, `thread_id`, `turn_id`, `status`,
`final_response` e `error`.

Quando o Jarvis e iniciado dentro de uma sessao Codex, ele herda
`CODEX_THREAD_ID`. Essa thread visivel tem prioridade sobre a thread antiga do
bridge: o manager valida `thread/read`, executa `thread/resume` e persiste o
mesmo ID antes do proximo `turn/start`. Se a thread visivel nao puder ser
retomada, a delegacao falha sem criar uma sessao alternativa invisivel.

## Jobs assincronos

Delegacoes curtas usam `wait=true`. O bridge aguarda por
`CODEX_QUICK_WAIT_TIMEOUT_SECONDS` (60 segundos por padrao); se a espera
expirar, o turn continua executando e a ferramenta retorna `running` com
`wait_timed_out=true`. Delegacoes longas usam `wait=false` e retornam assim que
o App Server aceita o unico `turn/start`.

Cada execucao e persistida atomicamente em
`.orchestrator/codex-jobs.json`. O cliente continua consumindo eventos depois
do retorno inicial. Ao receber `turn/completed`, disponibiliza o resultado ao
proximo ciclo do Qwen e o marca como entregue somente depois de processado. A
chave de idempotencia combina `job_id` e `turn_id`.

O hard timeout e independente: `CODEX_TURN_HARD_TIMEOUT_SECONDS=0` desativa
cancelamento automatico por duracao. Um timeout de espera nunca envia
`turn/interrupt`.

Para consultar jobs sem iniciar um turn:

```powershell
python -m tern.orchestrator codex-jobs
python -m tern.orchestrator codex-job-status JOB_ID
python -m tern.orchestrator codex-job-result JOB_ID
```

O Qwen usa `get_codex_job_status` para perguntas de estado. Cancelamento usa
`cancel_codex_job` e direcao humana durante a execucao usa `steer_codex_job`;
ambos operam sobre a thread e o turn registrados, sem criar outro turn.

## Iniciar ambiente compartilhado

```powershell
Set-Location D:\tern
python -m tern.orchestrator codex-shared-start
```

Abra diretamente a TUI na thread persistida, sem copiar o UUID:

```powershell
python -m tern.orchestrator codex-shared-tui
# ou
jarvis codex
```

Se a mesma thread ainda estiver aberta por um `codex` standalone, feche essa
TUI antes de executar `jarvis codex`. O processo standalone mantem o escritor
da thread; o bridge preserva o UUID e falha sem abrir uma sessao alternativa.

Internamente, o comando valida `thread/read` e executa a sintaxe do CLI 0.146.0:

```powershell
codex resume --remote ws://127.0.0.1:4500 --dangerously-bypass-approvals-and-sandbox -C "D:\tern" THREAD_ID
```

Para acompanhar os eventos sem TUI:

```powershell
python -m tern.orchestrator codex-shared-events --follow
```

Para inspecionar a sessao e direcionar ou interromper o turn ativo:

```powershell
python -m tern.orchestrator codex-shared-status
python -m tern.orchestrator codex-steer "Nao altere voice/policy.py."
python -m tern.orchestrator codex-interrupt
```

`codex-steer` nunca inicia um turn. Se a thread estiver ociosa, retorna erro.
O comando valida o turn ativo com `thread/read` e envia `turn/steer` com
`expectedTurnId`.

Para encerrar somente o App Server iniciado e registrado pelo projeto:

```powershell
python -m tern.orchestrator codex-shared-stop
```

## Estado e logs

- `.orchestrator/codex-session.json`: projeto, endpoint, thread e horario.
- `.orchestrator/codex-app-server.json`: PID do servidor gerenciado.
- `.orchestrator/codex-runtime.json`: turn, estado, origem, fila e intervencoes.
- `.orchestrator/codex-bridge.jsonl`: fila, envio, thread, turn, estado e erro.
- `.orchestrator/codex-events.jsonl`: eventos recebidos do App Server.
- `.orchestrator/actions.jsonl`: decisao Qwen, ferramenta, argumentos e retorno.

Campos e valores que parecem credenciais sao removidos dos novos registros.
A origem (`human`, `qwen` ou `system`) fica no estado e nos logs. No protocolo,
`clientUserMessageId` recebe um identificador opaco com a origem. O texto
tecnico enviado ao Codex nao ganha prefixo nem e alterado.

## Fila

Turns sao serializados entre processos por lock do projeto. Mensagens `human`
aguardando recebem prioridade sobre mensagens `qwen`. Um turn Qwen em execucao
nao recebe outro turn Qwen concorrente; direcao usa `turn/steer`, cancelamento
usa `turn/interrupt`. O cancelamento invalida a fila pendente e qualquer
resultado tardio e marcado como descartado. O resultado retornado ao Qwen inclui
`human_interventions`, `state_events` e `result_discarded`.

Turns iniciados diretamente pela TUI sao arbitrados pelo proprio App Server. Se
a TUI ja tiver um turn ativo, nova chamada Qwen e recusada em vez de criar uma
segunda sessao.

## Diagnostico

```powershell
python -m tern.orchestrator codex-bridge-diagnose
```

O diagnostico verifica executavel, versao, readiness, `initialize`, thread,
persistencia, dois turns na mesma thread, eventos, resposta final, reconexao,
cancelamento e uma chamada estruturada real do Qwen.

Use `--skip-qwen` para testar somente transporte e protocolo.

## Compatibilidade confirmada

Implementacao validada com `codex-cli 0.146.0`. Os schemas foram confirmados
com:

```powershell
codex app-server generate-json-schema --experimental --out DIRETORIO
```

Metodos usados: `initialize`, `initialized`, `thread/start`, `thread/read`,
`thread/resume`, `turn/start`, `turn/steer` e `turn/interrupt`. Conclusao vem de
`turn/completed`; resposta final vem de `item/completed` com item
`agentMessage`.

WebSocket do App Server ainda e experimental nessa versao. A TUI remota recebe
eventos emitidos por outros clientes quando esta anexada a mesma thread, mas o
comando remoto sem `resume THREAD_ID` nao garante selecao automatica. O painel
JSONL permanece disponivel como observador deterministico. O App Server nao
expoe contagem total de clientes; `codex-shared-status` mostra somente clientes
TUI locais conhecidos pelo bridge.
