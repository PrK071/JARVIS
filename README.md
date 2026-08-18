# JARVIS — assistente local Qwen3.5

Assistente Windows local: Qwen3.5-4B na GPU, arquivos controlados, Codex,
pesquisa web citada e STT faster-whisper. Microsoft Daniel pt-BR é o TTS
principal; Piper permanece como fallback local.

## Uso rápido

```powershell
Set-Location D:\tern
python -m tern.orchestrator start
python -m tern.orchestrator ask "Pesquise notícias recentes sobre IA e cite fontes."
python -m tern.orchestrator voice
python -m tern.orchestrator codex-shared-start
python -m tern.orchestrator codex-shared-tui
jarvis deepseek
```

O Codex usa um App Server local compartilhado. Abra a TUI diretamente na thread
persistida com `python -m tern.orchestrator codex-shared-tui` ou `jarvis codex`;
o comando valida a thread antes de abrir e nao exige copiar o `thread_id`.

O DeepSeek e um consultor opcional e stateless. `jarvis deepseek` abre uma TUI
persistente compartilhada com o Qwen, sem chamar a API no startup. Configure
`DEEPSEEK_API_KEY` e `DEEPSEEK_MODEL` para enviar mensagens; sem chave, historico
e comandos locais continuam disponiveis em modo leitura.

Diagnostico completo da integracao Qwen/Codex:

```powershell
python -m tern.orchestrator codex-bridge-diagnose
python -m tern.orchestrator codex-shared-status
python -m tern.orchestrator codex-steer "Analise somente codex.py."
python -m tern.orchestrator codex-interrupt
python -m tern.orchestrator codex-shared-events --follow
python -m tern.orchestrator codex-jobs
python -m tern.orchestrator codex-job-status JOB_ID
python -m tern.orchestrator codex-job-result JOB_ID
python -m tern.orchestrator projects
python -m tern.orchestrator project-active
python -m tern.orchestrator project-use tern
python -m tern.orchestrator project-find "configuracao da voz"
python -m tern.orchestrator project-refresh
```

Para disponibilizar o assistente pelo nome em qualquer terminal:

```powershell
python -m pip install --editable D:\tern
jarvis
```

Digitar apenas `jarvis` inicia a sessão contínua de voz. Os comandos da CLI
também ficam disponíveis pelo alias, por exemplo `jarvis status`,
`jarvis voice --once` e `jarvis ask "Sua pergunta"`.

Diagnóstico e configuração:

```powershell
python -m tern.orchestrator config
python -m tern.orchestrator search-diagnose "notícias recentes sobre inteligência artificial"
python -m tern.orchestrator voice-devices
python -m tern.orchestrator voice-configure
python -m tern.orchestrator voice-model-info
python -m tern.orchestrator voice-diagnose
python -m tern.orchestrator voice-pronunciation-test
python -m tern.orchestrator voice-playback-diagnose
python -m tern.orchestrator voice-phoneme-diagnose
python -m tern.orchestrator voice-piper-compare
python -m tern.orchestrator voice-compare-models
```

Pesquisa classifica intenção, expande consultas, pontua resultados, valida páginas
abertas e tenta correções. Notícias não aceitam Wikipédia, entretenimento ou
página genérica como fonte principal. TTS inicia pelo primeiro segmento,
sintetiza próximos em fila limitada e Esc cancela reprodução/síntese pendente.
Dispositivos persistem por nome e host API, com ID como fallback.
Piper é o único TTS ativo: totalmente local, sem clonagem e sem custo por uso.
O preset `clear_adult` usa o sample rate nativo do modelo, taxa intuitiva
`VOICE_TTS_RATE=0.94`, defaults acústicos do Piper e normalização falada de
status comuns em inglês. A resposta textual original permanece intacta.

Vozes Piper instaladas podem ser selecionadas por `VOICE_PIPER_VOICE`:
`miro`, `jeff`, `cadu`, `dii` ou `faber`. O comando
`voice-compare-models` gera WAVs equivalentes, calcula CER/WER com o
faster-whisper e reproduz somente candidatos locais com licença identificada.
Nenhum modelo é baixado durante síntese ou conversa.

## Interface visual

A interface HUD do JARVIS está em [`interface/`](interface/README.md). Ela inclui
as versões web e desktop, o avatar Synth-Alpha, o Terminal de Resposta e os
comandos locais do painel.

A versão web tem entrada por voz: o botão de microfone grava a fala e transcreve
com o mesmo faster-whisper do orquestrador, na própria máquina. O texto
reconhecido é enviado automaticamente, então os comandos do painel funcionam
falando. Sem o modelo em `models/voice/`, o botão fica oculto.

Para iniciar a versão web no Windows:

```powershell
Set-Location interface
.\run_web.bat
```

O código da interface pode ser usado, copiado, modificado e incorporado em
projetos próprios conforme a [Licença MIT da interface](interface/LICENSE).

Documentação:

- [Pesquisa web](docs/web-research.md)
- [Voz](docs/voice.md)
- [Bridge compartilhado Qwen/Codex](docs/codex-bridge.md)
- [DeepSeek consultivo e TUI](docs/deepseek.md)
- [Descoberta de projetos](docs/project-discovery.md)

Testes:

```powershell
python -m pytest -q
$env:RUN_VOICE_INTEGRATION_TESTS='true'
python -m pytest -q tests\test_voice_integration.py
Remove-Item Env:RUN_VOICE_INTEGRATION_TESTS

```

## Contribuidores

- [Murilo Roque (@mucamuca)](https://github.com/mucamuca)
- [PrK (@PrK071)](https://github.com/PrK071)
