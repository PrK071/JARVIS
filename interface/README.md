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
