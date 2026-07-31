# JARVIS — assistente local Qwen3.5

Assistente Windows local: Qwen3.5-4B na GPU, arquivos controlados, Codex,
pesquisa web citada, STT faster-whisper e Piper local. A voz pt-BR é
selecionada por alias; Miro é a seleção provisória e Faber permanece como
fallback.

## Uso rápido

```powershell
Set-Location D:\tern
python -m tern.orchestrator start
python -m tern.orchestrator ask "Pesquise notícias recentes sobre IA e cite fontes."
python -m tern.orchestrator voice
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
faster-whisper e reproduz Miro, Jeff, Cadu, Dii e Faber nessa ordem. Nenhum
modelo é baixado durante síntese ou conversa.

Documentação:

- [Pesquisa web](docs/web-research.md)
- [Voz](docs/voice.md)

Testes:

```powershell
python -m pytest -q
$env:RUN_VOICE_INTEGRATION_TESTS='true'
python -m pytest -q tests\test_voice_integration.py
Remove-Item Env:RUN_VOICE_INTEGRATION_TESTS

```
