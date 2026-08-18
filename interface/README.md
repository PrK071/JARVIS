# Interface JARVIS — T.R.I.A.D.E.

Interface HUD futurista inspirada em painéis de ficção científica, disponível nas versões web e desktop.

O T.R.I.A.D.E. apresenta animações, telemetria visual, terminal interativo e uma simulação de classificação ternária nos estados `−1`, `0` e `+1`.

> **Uso livre:** qualquer pessoa pode usar, copiar, modificar, adaptar, distribuir ou incorporar esta interface em projetos próprios, inclusive pessoais, acadêmicos ou comerciais, conforme os termos da [Licença MIT](LICENSE).

> As inferências, temperaturas e métricas ternárias são simuladas. A conversa livre pode usar a OpenAI Responses API quando o servidor web é iniciado com `OPENAI_API_KEY`; os comandos do T.R.I.A.D.E continuam funcionando localmente sem chave.

## Recursos

- Interface HUD responsiva para navegador
- Versão desktop desenvolvida em Python com Tkinter
- Núcleo central e forma de onda animados
- Terminal interativo com comandos, conversa livre, memória curta e ações rápidas
- Entrada por voz: o microfone é transcrito localmente, sem enviar áudio para fora
- Conexões de IA: cadastre qualquer API na janela do losango ◈ e converse por ela
- Histórico e distribuição visual das decisões
- Simulação de estados ternários `−1`, `0` e `+1`
- Nenhuma dependência Python externa

## Estrutura

```text
jarvis-ternario/
├── web/              # Versão para navegador
├── hudkit/           # Componentes visuais da versão desktop
├── triade/           # Aplicação e motor ternário simulado
├── main.py           # Ponto de entrada da aplicação desktop
├── web_server.py     # Servidor web e ponte opcional para o modelo
├── run.bat           # Inicializador desktop para Windows
└── run_web.bat       # Inicializador web para Windows
```

As versões web e desktop são independentes. A lógica executada no navegador está em `web/app.js`, enquanto a versão desktop utiliza `triade/engine.py`.

## Executar no navegador

Para ter comandos locais, conversa livre e Terminal de Resposta integrados, defina a chave da API como variável de ambiente e inicie o servidor:

```powershell
$env:OPENAI_API_KEY="sua_chave"
python web_server.py
```

No Windows, também é possível executar `run_web.bat` depois de configurar a variável. Acesse `http://localhost:8000`.

O modelo padrão é `gpt-5.6-terra`. Para selecionar outro modelo disponível na sua conta:

```powershell
$env:OPENAI_MODEL="gpt-5.6-terra"
python web_server.py
```

Nunca coloque a chave dentro de `web/app.js`, `index.html` ou em arquivos versionados. Sem `OPENAI_API_KEY`, os comandos locais continuam disponíveis, mas perguntas livres exibem uma mensagem de configuração no Terminal de Resposta.

Abrir diretamente `web/index.html` mantém a interface visual e os comandos locais, porém não permite conversa livre com o modelo.

Digite `ajuda` ou `comandos` na barra central para ver os comandos disponíveis. O painel “Terminal de Resposta” é o histórico visual da conversa; ele não executa comandos do sistema operacional.

As fontes Orbitron e Rajdhani são carregadas pelo Google Fonts. Sem conexão com a internet, o navegador utilizará fontes alternativas.

## Conexões de IA

Clique no **losango ◈** ao lado do título JARVIS, no cabeçalho, para abrir a
janela **Conexões de IA**. Ela cadastra qualquer API de IA sem editar arquivo
nenhum: preencha nome, formato, endpoint, modelo e chave, salve, e a conversa
livre passa a usar essa conexão.

A coluna **Salvas**, à esquerda da janela, lista tudo que você já cadastrou, com
nome e modelo, marcando em verde a que está em uso. Ela funciona como backup:
basta clicar numa conexão para ativá-la e carregá-la no formulário, sem precisar
buscar a chave de novo. O botão **+ nova conexão** limpa o formulário para
cadastrar outra. A lista fica no arquivo do servidor, então sobrevive a
reinícios.

A janela fecha com o ×, clicando no fundo escurecido ou com `Esc`, e devolve o
foco ao losango.

Três formatos cobrem, na prática, o mercado inteiro:

| Formato | Rota | Serve para |
|---|---|---|
| OpenAI-compatível | `/chat/completions` | OpenAI, DeepSeek, Groq, OpenRouter, Together, Mistral, e servidores locais como Ollama, LM Studio, llama.cpp e vLLM |
| OpenAI Responses | `/responses` | a Responses API da OpenAI |
| Anthropic | `/messages` | Claude |

O endpoint é preenchido com o padrão do formato escolhido, mas aceita qualquer
URL `http://` ou `https://` — é assim que se aponta para um modelo rodando na sua
própria máquina. Colar a URL completa do endpoint também funciona: a rota do
formato é removida e só a raiz é guardada.

Exemplo com a DeepSeek, que é OpenAI-compatível:

| Campo | Valor |
|---|---|
| Formato | OpenAI-compatível (chat/completions) |
| Endpoint | `https://api.deepseek.com` |
| Modelo | `deepseek-chat` (ou o que sua conta oferecer) |
| Chave | a chave `sk-...` do painel da DeepSeek |

É o mesmo endereço que o consultor DeepSeek do orquestrador usa
(`DEEPSEEK_BASE_URL` no `.env.example`), então dá para reaproveitar a mesma chave. No formato OpenAI-compatível a chave é opcional, justamente
porque servidores locais costumam dispensá-la.

### Onde a chave fica

**No servidor, nunca no navegador.** A chave é enviada uma única vez ao salvar e
gravada em `interface/providers.json`, que está no `.gitignore`. As listagens
devolvem apenas um resumo mascarado (`chav••••••7890`), então a chave não aparece
no front-end nem no histórico do navegador. Ao editar uma conexão, deixe o campo
da chave em branco para manter a que já está salva.

O botão **Testar** faz uma chamada real e curta ao provedor, e informa o motivo
quando falha: chave recusada, endpoint inacessível, rota errada, limite atingido.

Quem já usava `OPENAI_API_KEY` no ambiente não precisa migrar: ela continua
aparecendo como uma conexão chamada "Ambiente", e segue sendo usada enquanto
nenhuma outra for cadastrada.

## Falar com o JARVIS

O botão de microfone ao lado do campo de comando grava a sua fala e a transcreve
**na própria máquina**, com o `faster-whisper` já usado pelo orquestrador. O áudio
nunca sai do computador — nenhum serviço de nuvem participa da transcrição.

Como usar, igual a um áudio de WhatsApp:

1. **Segure** o botão do microfone. Ele fica vermelho e mostra o tempo gravado.
2. Fale.
3. **Solte** para enviar. `Esc` cancela sem enviar nada.

Um **toque rápido** (menos de 0,35 s) trava a gravação para falar sem segurar;
nesse modo, um clique encerra e envia. No teclado, `Enter` ou `Espaço` com o botão
focado alternam entre gravar e enviar.

O texto reconhecido entra no campo de comando e é enviado automaticamente, então
comandos locais funcionam por voz: dizer *"mostrar o avatar"* exibe o Synth-Alpha,
*"desativar voz"* silencia as respostas, e assim por diante.

O botão só aparece quando a transcrição local está disponível. Ela depende de:

- `faster-whisper` instalado (já vem em `requirements`), e
- o modelo em `models/voice/faster-whisper-base`.

Faltando qualquer um dos dois, o botão continua visível mas aparece esmaecido, e
explica o motivo ao ser acionado; o servidor também avisa no startup.

> Use o `run_web.bat`, que inicia pelo `.venv` do projeto. Chamar
> `python web_server.py` com o Python do sistema normalmente não funciona: as
> dependências (`faster-whisper`, `PyAV`, o pacote `tern`) estão no venv, e sem
> elas a transcrição fica indisponível. Para trocar o modelo de transcrição por um
mais preciso (ao custo de mais CPU por gravação):

```powershell
$env:VOICE_STT_MODEL="D:\caminho\para\faster-whisper-small"
aster-whisper-small"
python web_server.py
```

As demais variáveis `VOICE_STT_*` do orquestrador (`VOICE_STT_DEVICE`,
`VOICE_STT_COMPUTE_TYPE`, `VOICE_STT_LANGUAGE`, `VOICE_STT_THREADS`) também valem
aqui, porque a interface reaproveita a mesma configuração.

O navegador só libera o microfone em contexto seguro: `http://localhost:8000`
funciona, mas abrir `web/index.html` direto pelo sistema de arquivos não.

## Executar a versão desktop

Requisitos:

- Python 3.10 ou superior
- Tkinter disponível na instalação do Python

No Windows, execute `run.bat`. Pelo terminal, use:

```bash
python main.py
```

## Personalização

- `web/styles.css`: cores, dimensões e aparência da versão web
- `web/app.js`: animações e comportamento da versão web
- `hudkit/theme.py`: tema da versão desktop
- `hudkit/widgets.py`: componentes reutilizáveis da interface desktop
- `triade/engine.py`: lógica da simulação ternária

## Licença e permissão de uso

Este projeto é disponibilizado sob a [Licença MIT](LICENSE).

Você pode aproveitar o projeto inteiro ou apenas partes da interface, alterar cores, componentes e comportamentos, e incorporá-los em trabalhos próprios. A permissão abrange usos pessoais, acadêmicos e comerciais, observados os termos da licença.

A licença se aplica ao código deste repositório. Nomes, marcas e propriedades intelectuais de terceiros permanecem sujeitos aos direitos de seus respectivos titulares.

## Aviso

Este é um projeto independente, inspirado em interfaces HUD de ficção científica, sem vínculo ou afiliação oficial com a Marvel, Disney ou outras detentoras de marcas relacionadas.

## Autor

Desenvolvido por [Murilo Roque (@mucamuca)](https://github.com/mucamuca).
