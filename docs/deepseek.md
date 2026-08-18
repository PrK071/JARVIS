# DeepSeek consultivo e TUI

O DeepSeek e opcional. Qwen continua coordenando, Codex continua executando e o
DeepSeek recebe somente mensagens e contexto textual controlado. Ele nao recebe
ferramentas de filesystem, shell, subprocesso ou acesso direto ao Codex.

## Configuracao

```dotenv
DEEPSEEK_ENABLED=true
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
DEEPSEEK_AUTO_ESCALATION=false
DEEPSEEK_REQUEST_TIMEOUT_SECONDS=180
DEEPSEEK_MAX_RETRIES=2
DEEPSEEK_SESSION_MAX_RECENT_TURNS=20
```

O nome do modelo e configuracao, nao constante do codigo. O startup nao faz
health request pago. Sem chave, a TUI abre em modo leitura e o restante do
Jarvis funciona normalmente.

## TUI

Abra de qualquer diretorio depois da instalacao editavel:

```powershell
jarvis deepseek
jarvis deepseek --project llama
```

A interface usa `prompt_toolkit`. Enter envia. Alt+Enter insere uma quebra de
linha no Windows Terminal. Ctrl+C cancela somente a geracao ativa e descarta a
resposta parcial; Ctrl+D encerra quando o estado esta ocioso.

Comandos:

- `/help`
- `/status`
- `/new`
- `/sessions [id]`
- `/project <nome-ou-alias>`
- `/history`
- `/context`
- `/usage`
- `/codex [N]`
- `/clear-context`
- `/model [nome]`
- `/send-codex [confirm]`
- `/exit`

`/codex N` usa somente `review_codex_session`, compacta os turns e cria um anexo
temporario para a proxima consulta. Nao inicia turn Codex. `/send-codex` exige
confirmacao e entrega a recomendacao ao Qwen; somente Qwen pode decidir chamar
`delegate_to_codex`.

## Persistencia e contexto

As sessoes ficam em `D:\JARVIS\.orchestrator\deepseek-sessions.json`, com uma
sessao ativa por projeto resolvido pelo `ProjectRegistry`. Mensagens humanas,
do Qwen e respostas DeepSeek possuem IDs e fontes distintas.

O payload usa: prompt de sistema pequeno, rolling summary local, ultimos N turns,
contexto temporario e mensagem atual. Mensagens originais nunca sao apagadas.
Anexos temporarios nao sao copiados para o historico e sao consumidos somente
apos uma resposta valida. `/context` mostra uma estimativa explicita de tokens;
`/usage` mostra os contadores reais retornados pela API, sem calcular preco.

## API

O cliente envia `POST /chat/completions` pelo protocolo OpenAI-compatible. Para
a TUI usa SSE com `stream=true` e `stream_options.include_usage=true`. Retries
sao limitados a falhas transitorias e nunca reiniciam um stream que ja emitiu
texto. Chave e Authorization sao removidos dos logs.
