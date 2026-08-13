# Pesquisa web do orquestrador

O Qwen3.5 não recebe rede direta. Ele escolhe ferramentas JSON; o processo Python
valida argumentos, URL, DNS, tamanho, timeout e retorno antes de acessar a rede.

## Ferramentas

- `web_search`: pesquisa e retorna título, URL, domínio e snippet. Não abre todos
  os resultados.
- `web_open`: abre uma fonte escolhida e extrai HTML, texto ou PDF. Retorna URL
  final, título, data detectada, hash SHA-256, texto, links e metadados de citação.
- `web_extract`: abre uma fonte e seleciona passagens relevantes para uma consulta.

Snippets não devem ser citados como evidência. O supervisor foi instruído a abrir
fontes, comparar fontes independentes e separar fatos obtidos na web de conhecimento
interno ou inferências.

## Configuração

Valores padrão:

```text
WEB_ENABLED=true
WEB_SEARCH_PROVIDER=bing_rss
WEB_SEARCH_URL=https://www.bing.com/search
WEB_SEARCH_API_KEY=
WEB_TIMEOUT=20
WEB_MAX_DOWNLOAD_BYTES=10485760
WEB_MAX_TEXT_CHARS=65536
WEB_MAX_PDF_PAGES=20
WEB_USER_AGENT=TernLocalResearch/1.0
WEB_ALLOWED_DOMAINS=
WEB_BLOCKED_DOMAINS=
WEB_QUERY_EXPANSION_ENABLED=true
WEB_MAX_QUERY_VARIANTS=4
WEB_CROSS_LANGUAGE_SEARCH=true
WEB_DEFAULT_REGION=BR
WEB_MIN_RESULT_RELEVANCE=0.55
WEB_MIN_SOURCE_RELEVANCE=0.65
WEB_RELEVANCE_TOP_K=8
WEB_MAX_RESEARCH_CORRECTIONS=2
WEB_MAX_TOTAL_SEARCHES=6
WEB_MAX_TOTAL_OPENS=10
```

O processo carrega `D:\tern\.env` automaticamente, sem sobrescrever variáveis já
definidas no ambiente. `python -m tern.orchestrator config` informa caminho,
existência e carregamento do arquivo; chaves nunca são exibidas.

Provedores:

- `bing_rss`: padrão, RSS/XML, sem chave.
- `brave`: JSON oficial; exige `WEB_SEARCH_API_KEY`.
- `duckduckgo_html`: legado; desafios anti-bot são retornados como
  `search_response_invalid`, nunca como zero resultados.

`WEB_ALLOWED_DOMAINS` e `WEB_BLOCKED_DOMAINS` recebem domínios separados por
vírgula. Domínios-filhos seguem a regra do domínio pai. Allowlist vazia permite
domínios públicos; blocklist sempre prevalece.

Exemplo:

```powershell
$env:WEB_ALLOWED_DOMAINS="openai.com,python.org"
$env:WEB_BLOCKED_DOMAINS="ads.example.com"
python -m tern.orchestrator ask "Compare documentação oficial sobre APIs e cite as fontes."
```

Listar configuração e schemas:

```powershell
python -m tern.orchestrator config
python -m tern.orchestrator tools
```

Diagnóstico direto, sem Qwen:

```powershell
python -m tern.orchestrator search-diagnose "OpenAI official documentation" --language en
python -m tern.orchestrator search-diagnose "notícias recentes sobre inteligência artificial" --language pt-BR
```

O diagnóstico retorna provedor efetivo, configuração sanitizada, status HTTP,
`content-type`, duração, bytes e resultados normalizados.

## Intenção, relevância e atualidade

Antes da primeira busca, código determinístico classifica pedido em `news`,
`official_documentation`, `technical_research`, `general_information`,
`product_information`, `troubleshooting`, `academic` ou `current_status`.
Classificação contém tópico, idioma, região, necessidade de atualidade, tipos de
fonte preferidos, significados excluídos e ambiguidade. Termo isolado como
`Artificial` pede esclarecimento.

Consultas de notícias recebem mês/ano atuais, termos explícitos de notícia,
variantes em português/inglês e exclusões de filme/livro/jogo. Variantes são
deduplicadas e limitadas. Para `bing_rss`, intenção de notícia usa endpoint RSS
suportado `/news/search`; outras intenções preservam `/search`.

Cada resultado recebe:

```text
final_score =
  0.40 * topic_score +
  0.25 * intent_score +
  0.20 * freshness_score +
  0.15 * source_quality_score -
  ambiguity_penalty
```

Notícia exige também `topic_score >= 0.65`, tipo `news_article` ou
`official_announcement` e `final_score >= WEB_MIN_RESULT_RELEVANCE`. Wikipédia,
entretenimento, páginas genéricas, resultados sem data e significados excluídos
recebem penalidades ou rejeição. Wikipédia continua permitida para informação
geral.

Após `web_open`, título, corpo principal, data e tipo são validados novamente.
Data RSS pode preencher data ausente no HTML. Fonte rejeitada perde `citation`,
ganha `rejection_reason` e não entra na resposta. Sem fonte válida, resposta é:

```text
Não encontrei fontes suficientemente relevantes para responder com segurança.
```

## Pesquisa corretiva

Resultado inicial inadequado ativa próxima variante. Página aberta irrelevante
também pode ativar correção. Consultas já executadas não repetem. Três limites
independentes impedem loops:

- `WEB_MAX_RESEARCH_CORRECTIONS`;
- `WEB_MAX_TOTAL_SEARCHES`;
- `WEB_MAX_TOTAL_OPENS`.

Logs em `.orchestrator/actions.jsonl` registram classificação, variantes,
pontuações, rejeições, correções, fontes abertas e citações aceitas. Não registram
cadeia de raciocínio.

Regressão obrigatória cobre `Artificial (2026 film)`: pedido de notícia sobre
inteligência artificial rejeita filme/Wikipédia e cita somente matéria
jornalística aceita.

## Controles

- somente HTTP e HTTPS;
- somente portas 80 e 443;
- credenciais em URL bloqueadas;
- localhost, IP privado, loopback, link-local, multicast, reservado e metadata
  cloud bloqueados;
- DNS validado antes de cada download;
- cada redirecionamento e URL final revalidados;
- domínio global permitido/bloqueado aplicado também aos links encontrados;
- downloads, texto, páginas PDF e tempo limitados;
- HTML ativo removido; conteúdo remoto tratado como dado não confiável;
- chamadas e resultados registrados em `.orchestrator/actions.jsonl`;
- limite geral de chamadas e prevenção de repetição continuam ativos.

## Limites conhecidos

- Bing RSS é endpoint público sem SLA de API. Para produção com contrato e chave,
  configure Brave Search API.
- Bing RSS pode ignorar operadores, variar por região ou retornar poucos itens.
  Expansão corretiva e validação protegem qualidade, mas não criam cobertura
  inexistente.
- Páginas que exigem JavaScript, login, CAPTCHA ou interação de navegador não são
  renderizadas.
- PDF escaneado sem camada de texto não recebe OCR.
- Validação DNS reduz SSRF, mas proxy corporativo e DNS local ainda devem possuir
  políticas próprias.
- Sites podem bloquear o User-Agent ou impor limites de frequência.
