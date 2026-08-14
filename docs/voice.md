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
  -> prepare_spoken_text + normalize_for_speech(provider ativo)
  -> WindowsSpeechTTS (Daniel) ou PiperTTS fallback
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
- `pt_BR-faber-medium`: voz brasileira atual e fallback, aproximadamente
  63 MB, 22.050 Hz.
- `pt_BR-cadu-medium` e `pt_BR-jeff-medium`: alternativas brasileiras locais
  de tamanho semelhante.
- Faber, Cadu e Jeff: repositório de modelos MIT; cards declaram dataset CC0.

Fontes: [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[modelo base](https://huggingface.co/Systran/faster-whisper-base),
[Piper](https://github.com/OHF-Voice/piper1-gpl) e
[vozes pt-BR do Piper](https://huggingface.co/rhasspy/piper-voices/tree/main/pt/pt_BR).

STT e Piper rodam em CPU. Qwen permanece na GPU. Nenhum áudio ou texto é
enviado para serviço externo. Não existe clonagem de voz nem custo por uso.

## Instalação no Windows

No PowerShell:

```powershell
Set-Location D:\tern
python -m pip install -e ".[voice]"
New-Item -ItemType Directory -Force D:\tern\models\voice
python -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-base', local_dir=r'D:\tern\models\voice\faster-whisper-base')"
python -m piper.download_voices --data-dir D:\tern\models\voice pt_BR-cadu-medium
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
D:\tern\models\voice\pt_BR-cadu-medium.onnx
D:\tern\models\voice\pt_BR-cadu-medium.onnx.json
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

VOICE_TTS_PROVIDER=windows_sapi
VOICE_MODE=windows_sapi
VOICE_WINDOWS_VOICE_ID=HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices\Tokens\MSTTS_V110_ptBR_DanielM
VOICE_WINDOWS_RATE=1.5
VOICE_WINDOWS_VOLUME=100
VOICE_FALLBACK_PROVIDER=piper
VOICE_PIPER_VOICE=faber
# VOICE_PIPER_MODEL_PATH=
# VOICE_PIPER_CONFIG_PATH=
VOICE_TTS_DEVICE=cpu
VOICE_TTS_RATE=0.94
VOICE_TTS_VOLUME=1.0
VOICE_TTS_TIMEOUT_SECONDS=60
VOICE_STYLE=clear_adult
VOICE_TRANSLATE_COMMON_STATUS_TERMS=true
VOICE_PIPER_USE_MODEL_DEFAULT_NOISE=true

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
VOICE_SENTENCE_PAUSE_MS=160
VOICE_PARAGRAPH_PAUSE_MS=280
VOICE_POST_PROCESSING=false
VOICE_NORMALIZE_LOUDNESS=true
VOICE_LIGHT_COMPRESSION=false
VOICE_LIGHT_EQ=false
```

Microsoft Daniel é selecionado pelo ID OneCore estável. Neste computador ele
está disponível pela API WinRT, não pelo catálogo SAPI 5. O valor
`VOICE_WINDOWS_RATE` usa a escala logarítmica SAPI e é aplicado pela opção
nativa `SpeakingRate` antes da síntese; aceita valores de `-10` a `10`,
inclusive fracionários. Volume padrão: `100`. Piper permanece fallback.

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
python -m tern.orchestrator voice-playback-diagnose
python -m tern.orchestrator voice-phoneme-diagnose
python -m tern.orchestrator voice-piper-compare
python -m tern.orchestrator voice-compare-models
```

`voice-diagnose` é leve: não chama Qwen nem STT. Confirma Piper instalado,
modelo selecionado, saída resolvida por nome, síntese, reprodução, cancelamento e
limpeza. `voice-pronunciation-test` gera 12 WAVs numa pasta temporária
identificada; use `--play` para ouvi-los em sequência.

`voice-playback-diagnose` compara o PCM bruto do Piper com o reconstruído da
fila progressiva, registra sample rates e reproduz pelos caminhos normal e
independente. Os WAVs ficam em
`.orchestrator\piper-playback-diagnose`. `voice-phoneme-diagnose` confirma
o locale `pt-br`, mostra fonemas somente no diagnóstico e salva amostras em
`.orchestrator\piper-phoneme-diagnose`.

`voice-piper-compare` mantém o diagnóstico antigo de Faber, Cadu e Jeff.
`voice-compare-models` usa as mesmas 20 frases normalizadas e os padrões
acústicos de cada modelo. Candidatos ausentes, inválidos ou sem licença
identificada são registrados e ignorados. Também gera uma variante em taxa
0,94, usa o faster-whisper apenas como indicador de CER/WER e grava:

```text
.orchestrator\piper-model-comparison\
  comparison-report.md
  comparison-report.json
  jeff-completo.wav
  cadu-completo.wav
  faber-completo.wav
```

Por padrão, WAVs elegíveis são reproduzidos na ordem preferida e Esc cancela
a reprodução. Ao final, a escolha numérica atualiza somente
`VOICE_PIPER_VOICE` no `.env`. Para gerar sem reproduzir nem selecionar:

```powershell
python -m tern.orchestrator voice-compare-models --no-play --no-select
```

Aliases portáveis:

```dotenv
VOICE_PIPER_VOICE=faber
# Valores: miro, jeff, cadu, dii, faber
# VOICE_PIPER_MODEL_PATH=
# VOICE_PIPER_CONFIG_PATH=
```

A ordem de resolução é: caminho explícito válido, alias, voz padrão e Faber
como fallback de emergência. Somente uma sessão ONNX é carregada. Trocar a voz
exige reiniciar a sessão atual do assistente, liberando o modelo anterior.
Não há download durante conversa nem acesso de rede durante síntese.

Origens testadas: [Faber](https://huggingface.co/rhasspy/piper-voices/tree/main/pt/pt_BR/faber/medium),
[Cadu](https://huggingface.co/rhasspy/piper-voices/tree/main/pt/pt_BR/cadu/medium)
e [Jeff](https://huggingface.co/rhasspy/piper-voices/tree/main/pt/pt_BR/jeff/medium)
no repositório `rhasspy/piper-voices`; [Miro](https://huggingface.co/OpenVoiceOS/pipertts_pt-BR_miro)
e [Dii](https://huggingface.co/OpenVoiceOS/pipertts_pt-BR_dii) no OpenVoiceOS.
Miro e Dii permanecem rejeitados porque seus repositórios/model cards não
identificam licença. Checkpoints `.ckpt` não são usados nem baixados. Modelos
locais permanecem ignorados pelo Git.

## Ritmo, sample rate e pronúncia

O Piper fornece o sample rate pelo JSON do modelo e por cada chunk. O player
abre a saída nessa taxa nativa; se o dispositivo a rejeitar, ocorre uma única
reamostragem real para a taxa padrão do dispositivo. Os bytes nunca são apenas
reinterpretados com outro número.

`VOICE_TTS_RATE` possui semântica intuitiva: `1.0` é o ritmo original, valores
menores falam mais devagar e valores maiores mais rápido. Internamente o Piper
recebe `length_scale = 1 / rate`. `VOICE_TTS_SPEED` continua aceito
temporariamente com aviso de depreciação.

O preset leve `clear_adult` usa taxa 0,94 e pausas moderadas, sem alterar pitch,
formantes ou identidade. `VOICE_PIPER_USE_MODEL_DEFAULT_NOISE=true` mantém
`noise_scale` e `noise_w` definidos pelo próprio modelo.

Termos comuns de status em inglês são traduzidos somente na versão falada:
por exemplo, `working` vira “funcionando”. Código entre crases, comandos,
URLs, caminhos e citações literais são preservados. A resposta exibida nunca é
modificada.

## TTS progressivo

Resposta textual completa permanece visível. Crases são tratadas apenas como
formatação: identificadores, comandos e blocos entre crases permanecem na fala.
Caminhos e URLs seguem a política própria; respostas web recebem resumo e aviso
de links na tela.
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
- Piper medium (Cadu/Faber/Jeff): cerca de 147 MiB adicionais de RAM;
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
- qualidade e variedade ficam limitadas às vozes pt-BR disponíveis no Piper.

Próximas etapas úteis: VAD mais robusto, confirmação vocal separada com política
explícita, seleção por identificador estável do sistema e empacotamento dos
modelos.
