# Interface JARVIS — T.R.I.A.D.E.

Interface HUD futurista inspirada em painéis de ficção científica, disponível nas versões web e desktop.

O T.R.I.A.D.E. apresenta animações, telemetria visual, terminal interativo e uma simulação de classificação ternária nos estados `−1`, `0` e `+1`.

> **Uso livre:** qualquer pessoa pode usar, copiar, modificar, adaptar, distribuir ou incorporar esta interface em projetos próprios, inclusive pessoais, acadêmicos ou comerciais, conforme os termos da [Licença MIT](LICENSE).

> Este projeto é uma demonstração visual. As inferências, temperaturas, métricas e respostas são simuladas localmente; não há inteligência artificial, sensores, banco de dados ou backend conectado.

## Recursos

- Interface HUD responsiva para navegador
- Versão desktop desenvolvida em Python com Tkinter
- Núcleo central e forma de onda animados
- Terminal interativo com comandos e ações rápidas
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
└── run.bat           # Inicializador para Windows
```

As versões web e desktop são independentes. A lógica executada no navegador está em `web/app.js`, enquanto a versão desktop utiliza `triade/engine.py`.

## Executar no navegador

Abra o arquivo `web/index.html` em um navegador moderno.

Opcionalmente, execute um servidor local a partir da pasta do projeto:

```bash
python -m http.server 8000 --directory web
```

Depois acesse `http://localhost:8000`.

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
