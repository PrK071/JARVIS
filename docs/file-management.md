# Gestão autônoma e segura de arquivos

O módulo `tern.file_management` organiza arquivos, mantém um inventário de
metadados, cria backups incrementais com versões, sincroniza diretórios em um
único sentido e gera relatórios automáticos.

## Segurança por padrão

- Organização, backup e sincronização começam em `dry-run`.
- Alterações exigem `--apply`.
- Somente caminhos sob `allowed_roots` são aceitos.
- Raízes de volume (`C:\`, `/`) são recusadas.
- Origem e destino de backup/sincronização não podem se sobrepor.
- Links simbólicos não são seguidos.
- Arquivos existentes nunca são sobrescritos durante organização; colisões
  recebem sufixos como `arquivo (1).pdf`.
- Sincronização não remove arquivos do destino.
- Sincronização preserva destinos substituídos em `.versions/TIMESTAMP/`.
- Backup preserva o conteúdo substituído em `.versions/TIMESTAMP/`.
- Nenhuma política de exclusão ou retenção destrutiva está habilitada.

O estado interno (`metadata.sqlite3`, journal e notificações) pode ser gravado
mesmo em `dry-run`; os diretórios administrados não são modificados nesse modo.

## Configuração

Copie `file-manager.example.json`, ajuste todos os caminhos e mantenha na
allowlist cada origem e destino. Categorias podem ser personalizadas por
extensão. Arquivos desconhecidos vão para `other`.

Layouts suportados:

- `category`
- `category/year`
- `category/year/month`

`minimum_age_seconds` evita movimentar downloads ainda recentes.
`hash_max_bytes` limita o tamanho dos arquivos cujo SHA-256 será calculado.

## Uso

```powershell
python scripts/manage_files.py --config D:\caminho\file-manager.json validate
python scripts/manage_files.py --config D:\caminho\file-manager.json scan
python scripts/manage_files.py --config D:\caminho\file-manager.json run
python scripts/manage_files.py --config D:\caminho\file-manager.json run --apply
```

Para operação contínua:

```powershell
python scripts/manage_files.py --config D:\caminho\file-manager.json watch --apply
```

O intervalo vem de `interval_seconds` e pode ser substituído por
`--interval SEGUNDOS`. O processo encerra de forma limpa com `Ctrl+C`. Para
inicialização automática, use esse mesmo comando no agendador já adotado pelo
sistema; o módulo não instala tarefas agendadas sem ação explícita do operador.

Também é possível executar etapas isoladas:

```powershell
python scripts/manage_files.py --config D:\caminho\file-manager.json organize
python scripts/manage_files.py --config D:\caminho\file-manager.json backup
python scripts/manage_files.py --config D:\caminho\file-manager.json sync
```

Adicione `--apply` somente depois de revisar a saída do `dry-run`.

## Metadados, notificações e relatórios

O inventário SQLite registra caminho, raiz, tamanho, data de modificação,
extensão, categoria, SHA-256 e estado ausente. Hashes de arquivos inalterados
são reutilizados.

Cada ciclo completo produz:

- relatório JSON detalhado;
- relatório Markdown resumido;
- aliases `latest.json` e `latest.md`;
- evento em `notifications.jsonl`, quando habilitado;
- resumo no console, quando habilitado;
- journal `actions.jsonl` com ações planejadas, aplicadas, ignoradas e falhas.

Esses arquivos ficam nos caminhos `state_dir` e `report_dir` definidos na
configuração e podem ser consumidos pelo JARVIS ou por outro monitor local.

## Política operacional recomendada

1. Validar a configuração.
2. Executar `run` sem `--apply` e revisar `latest.md`.
3. Fazer uma primeira execução manual com `run --apply`.
4. Confirmar backup e versões antes de ativar `watch --apply`.
5. Monitorar `notifications.jsonl` e o crescimento de `.versions`.
