# Voz local push-to-talk

## Arquitetura

O modo de voz é uma camada de entrada e saída. Não cria outro agente:

```text
microfone
  -> SoundDeviceAudio (mono, 16 kHz, silêncio/limite/cancelamento)
  -> FasterWhisperSTT (CPU, int8)
  -> Supervisor.run(texto)
  -> Qwen3.5 e ToolRegistry existentes
  -> resposta textual completa
  -> prepare_spoken_text + normalize_for_speech(provider="piper")
  -> segment_for_speech
  -> PiperTTS produtor + fila limitada
  -> SoundDeviceAudio consumidor ordenado (Esc interrompe)
```

`ask` e `voice` usam o mesmo `Supervisor`, `ToolRegistry`, políticas de arquivos,
pesquisa web, Codex, sessões e confirmações destrutivas.

## Dependências e modelos

- `sounddevice`: captura/reprodução via PortAudio.
- `faster-whisper`: STT local, multilíngue, MIT, CTranslate2 CPU/int8.
- `faster-whisper-base`: aproximadamente 148 MB em disco. Bom compromisso inicial
  entre português, memória e latência.
- `faster-whisper-small`: opção manual, aproximadamente 460–500 MB em disco;
  maior precisão esperada, mais RAM e latência que `base`.
- `piper-tts`: síntese local leve, engine GPL-3.0.
- `pt_BR-faber-medium`: voz brasileira, licença MIT, aproximadamente 63 MB,
  22.050 Hz.

Fontes: [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[modelo base](https://huggingface.co/Systran/faster-whisper-base),
[Piper](https://github.com/OHF-Voice/piper1-gpl) e
[pt_BR-faber-medium](https://huggingface.co/rhasspy/piper-voices/tree/main/pt/pt_BR/faber/medium).

STT e Piper rodam em CPU. Qwen permanece na GPU. Nenhum áudio ou texto é
enviado para serviço externo. Não existe clonagem de voz nem custo por uso.

## Instalação no Windows

No PowerShell:

```powershell
Set-Location D:\tern
python -m pip install -e ".[voice]"
New-Item -ItemType Directory -Force D:\tern\models\voice
python -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-base', local_dir=r'D:\tern\models\voice\faster-whisper-base')"
python -m piper.download_voices --data-dir D:\tern\models\voice pt_BR-faber-medium
```

Modelo `small` opcional; não baixado automaticamente:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-small', local_dir=r'D:\tern\models\voice\faster-whisper-small')"
(Get-Content .env) -replace '^VOICE_STT_MODEL=.*$', 'VOICE_STT_MODEL=D:\tern\models\voice\faster-whisper-small' | Set-Content .env
python -m tern.orchestrator voice-model-info
```

Para voltar:

```powershell
(Get-Content .env) -replace '^VOICE_STT_MODEL=.*$', 'VOICE_STT_MODEL=D:\tern\models\voice\faster-whisper-base' | Set-Content .env
```

Arquivos esperados:

```text
D:\tern\models\voice\faster-whisper-base\model.bin
D:\tern\models\voice\pt_BR-faber-medium.onnx
D:\tern\models\voice\pt_BR-faber-medium.onnx.json
```

Primeira carga do STT e TTS demora mais. Depois do download, uso é offline.

## Configuração

`.env` é carregado automaticamente a partir da raiz do projeto. Caminhos abaixo
podem ser omitidos: padrões são derivados da raiz, sem caminhos absolutos no
código.

```dotenv
VOICE_ENABLED=true
VOICE_STT_PROVIDER=faster_whisper
VOICE_STT_MODEL=D:\tern\models\voice\faster-whisper-base
VOICE_STT_DEVICE=cpu
VOICE_STT_COMPUTE_TYPE=int8
VOICE_STT_LANGUAGE=pt
VOICE_STT_THREADS=4
VOICE_STT_TIMEOUT_SECONDS=120

VOICE_TTS_PROVIDER=piper
VOICE_MODE=piper
VOICE_TTS_MODEL=D:\tern\models\voice\pt_BR-faber-medium.onnx
VOICE_TTS_VOICE=pt_BR-faber-medium
VOICE_TTS_DEVICE=cpu
VOICE_TTS_SPEED=0.96
VOICE_TTS_VOLUME=1.0
VOICE_TTS_TIMEOUT_SECONDS=60

VOICE_INPUT_DEVICE=5
VOICE_OUTPUT_DEVICE=3
VOICE_INPUT_DEVICE_NAME=Microfone (High Definition Audio Device) [Windows DirectSound]
VOICE_OUTPUT_DEVICE_NAME=Fones de ouvido (High Definition Audio Device) [Windows DirectSound]
VOICE_SAMPLE_RATE=16000
VOICE_MAX_RECORDING_SECONDS=60
VOICE_SILENCE_TIMEOUT_MS=1200
VOICE_MIN_SPEECH_MS=300
VOICE_SILENCE_THRESHOLD=0.003
VOICE_CONFIRM_TRANSCRIPTION=true

VOICE_MAX_SPOKEN_CHARACTERS=1200
VOICE_READ_CODE=false
VOICE_READ_URLS=false
VOICE_SUMMARIZE_LONG_RESPONSES=true
VOICE_TEMP_DIRECTORY=D:\tern\.orchestrator\voice-temp
VOICE_KEEP_RECORDINGS=false
VOICE_LOG_LEVEL=INFO
VOICE_DEBUG_TRANSCRIPTS=false
VOICE_INTERRUPT_KEY=esc
VOICE_TTS_STREAMING=true
VOICE_TTS_CHUNK_MIN_CHARACTERS=60
VOICE_TTS_CHUNK_MAX_CHARACTERS=220
VOICE_TTS_QUEUE_SIZE=3
VOICE_SENTENCE_PAUSE_MS=140
VOICE_PARAGRAPH_PAUSE_MS=260
VOICE_POST_PROCESSING=false
VOICE_NORMALIZE_LOUDNESS=true
VOICE_LIGHT_COMPRESSION=false
VOICE_LIGHT_EQ=false
```

Resolução: identidade exata `nome [host API]`, nome exato, correspondência
normalizada, ID explícito, dispositivo padrão. Nome duplicado retorna opções;
sistema não troca silenciosamente para dispositivo diferente. IDs ficam como
fallback.

```powershell
python -m tern.orchestrator voice-devices
python -m tern.orchestrator voice-configure
```

`voice-configure` lista dispositivos, grava/reproduz teste e atualiza somente
quatro chaves de dispositivo no `.env`; demais linhas, comentários e segredos são
preservados.

O limiar de silêncio depende do microfone. Calibre
`VOICE_SILENCE_THRESHOLD` acima do ruído ambiente e abaixo da fala normal.

## Uso

Sessão contínua:

```powershell
python -m tern.orchestrator voice
```

Interação única:

```powershell
python -m tern.orchestrator voice --once
python -m tern.orchestrator voice --voice piper
python -m tern.orchestrator voice --no-voice
```

Fluxo:

1. Enter inicia gravação.
2. Enter ou Espaço encerra; silêncio suficiente ou limite também encerram.
3. Esc cancela.
4. Transcrição aparece.
5. `S` envia, `R` regrava, `C` cancela.
6. Resposta textual completa aparece.
7. Esc interrompe somente fala; sessão e texto permanecem.

## Diagnóstico

Sem Qwen:

```powershell
python -m tern.orchestrator voice-diagnose
python -m tern.orchestrator voice-model-info
python -m tern.orchestrator voice-pronunciation-test
```

`voice-diagnose` é leve: não chama Qwen nem STT. Confirma Piper instalado,
modelo Faber, saída resolvida por nome, síntese, reprodução, cancelamento e
limpeza. `voice-pronunciation-test` gera 12 WAVs numa pasta temporária
identificada; use `--play` para ouvi-los em sequência.

## TTS progressivo

Resposta textual completa permanece visível. Texto falado remove código, caminhos
e URLs conforme política; respostas web recebem resumo e aviso de links na tela.
Segmentador preserva abreviações (`Dr.`/`Sr.`), decimais, `Qwen3.5`, URLs,
caminhos e código. Blocos obedecem limites configurados.

Produtor sintetiza segmentos em ordem. Fila limitada fornece pressão de retorno.
Consumidor começa no primeiro bloco enquanto próximos são sintetizados. Esc
define cancelamento compartilhado, interrompe reprodução e síntese cooperativa,
esvazia fila e mantém sessão utilizável. Falha progressiva usa erros estruturados:
`tts_chunk_synthesis_failed`, `tts_stream_cancelled`,
`tts_stream_queue_failed`, `tts_stream_playback_failed`. Desative com
`VOICE_TTS_STREAMING=false` para fallback completo.

Benchmark reproduzível:

```powershell
python scripts\tts_streaming_benchmark.py
python scripts\tts_streaming_benchmark.py --play
python scripts\tts_streaming_benchmark.py --play --cancel-on-segment 3
```

Medição local, 369 caracteres/5 segmentos, Piper real:

- anterior: primeiro áudio após 1,80 s;
- progressivo: primeiro áudio após 0,41 s;
- redução: 77%;
- média de síntese: 0,35 s por segmento;
- fila máxima observada: 2;
- RAM Piper carregado: 117 MiB; sobrecarga progressiva final: 24 MiB;
- VRAM adicional: 0.

Valores variam por CPU, texto e cache.

## Segurança

Confiança do STT nunca autoriza ação perigosa. Pedidos de apagar, sobrescrever,
instalar, remover software, administrar ou mudar sistema exigem confirmação
separada. Ferramenta mostra ação, caminho, impacto e reversibilidade. Execução
só ocorre após digitar exatamente `CONFIRMAR`; confirmação presente no pedido
original não vale.

Confirmação geral de transcrição pode ser desativada, mas pedidos sensíveis
continuam exigindo confirmação. Nomes ambíguos continuam sujeitos às políticas
e esclarecimentos do orquestrador.

## Privacidade e logs

Gravações não são persistidas por padrão. Logs registram tempos, dispositivos,
volume, cancelamentos e erros; não registram áudio, chaves, arquivos locais ou
transcrição completa. Transcrição só entra no log com
`VOICE_DEBUG_TRANSCRIPTS=true`.

Para limpar temporários manualmente:

```powershell
Remove-Item -LiteralPath D:\tern\.orchestrator\voice-temp\* -Force
```

Confira o caminho antes. Para desativar:

```dotenv
VOICE_ENABLED=false
```

## Consumo

Medição local inicial:

- faster-whisper base: cerca de 177 MiB adicionais de RAM no primeiro teste;
- Piper Faber medium: cerca de 147 MiB adicionais de RAM;
- VRAM adicional: zero; STT e TTS foram configurados e executados em CPU.

Valores variam com duração, cache, versão e backend. Uma interação por vez evita
picos simultâneos. Qwen mantém prioridade na GPU.

## Testes

Unitários, sem hardware:

```powershell
python -m pytest -q
```

Integração opcional, com hardware/modelos:

```powershell
$env:RUN_VOICE_INTEGRATION_TESTS='true'
python -m pytest -q tests\test_voice_integration.py
Remove-Item Env:RUN_VOICE_INTEGRATION_TESTS
```

## Limitações desta versão

- push-to-talk; sem palavra de ativação;
- sem detecção automática de barge-in por fala;
- confirmação destrutiva é textual;
- qualidade do STT depende de ganho, ruído e calibração;
- streaming começa após primeiro segmento; ainda não existe streaming token a token;
- uma interação de voz por vez;
- sem biometria, múltiplos usuários ou serviços externos;
- sem clonagem ou imitação de voz;
- qualidade e variedade ficam limitadas à voz Piper Faber.

Próximas etapas úteis: VAD mais robusto, confirmação vocal separada com política
explícita, seleção por identificador estável do sistema e empacotamento dos
modelos.
