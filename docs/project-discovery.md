# Descoberta de projetos

O orquestrador mantem um registro leve em
`.orchestrator/projects.json`. Apenas caminhos, aliases, marcadores, tipo e
datas de uso sao persistidos. As fontes de verdade para permissao continuam
sendo as raizes de `MODEL_ALLOWED_ROOTS`.

A resolucao segue esta ordem: caminho explicito, alias ou nome explicito,
projeto da thread Codex, ultima ferramenta, projeto ativo e diretorio atual
quando permitido. Um caminho externo nunca e transformado em projeto conhecido.

Os indices em `.orchestrator/project-indexes/` guardam somente caminho relativo,
extensao, tamanho e data de modificacao. `.git`, ambientes virtuais, caches,
`node_modules`, builds, modelos, checkpoints, audio, binarios e `.env` sao
ignorados. O indice e revalidado incrementalmente quando uma busca e feita.

Ferramentas do Qwen:

- `resolve_project`: resolve contexto, sem ler ou modificar arquivos.
- `find_project_files`: localiza nomes ou descricoes dentro de um projeto.

Comandos:

```powershell
python -m tern.orchestrator projects
python -m tern.orchestrator project-active
python -m tern.orchestrator project-use tern
python -m tern.orchestrator project-find "configuracao da voz"
python -m tern.orchestrator project-refresh
```

O rastreador de progresso permite chamadas diferentes quando surgem novas
evidencias. Depois de duas chamadas equivalentes sem novos caminhos, entidades
ou estado, exige replanejamento e bloqueia uma terceira execucao equivalente.
