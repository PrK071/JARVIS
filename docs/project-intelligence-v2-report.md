# Project Intelligence V2 — relatório experimental

Data: 2026-08-15

Branch avaliada: `feat/project-intelligence-v2`

Modelo de controle: Qwen local atual. Bonsai, embeddings, vector database e modelos adicionais não foram usados.

Escopo: indexação, candidate generation, context planning e avaliação em dry-run. Nenhuma seleção de agente, delegação, leitura autônoma posterior, mutação ou execução foi ativada.

## Decisão executiva

```text
APROVADO — PROJECT INTELLIGENCE V2

READY FOR AUTOMATIC AGENT SELECTION DRY-RUN
```

No corpus congelado de 13 casos reais, a V2 atingiu 100% de recall de arquivos obrigatórios, 100% de recall relevante, 98,57% de precisão, 1,43% de seleção irrelevante, 100% de provenance, zero leakage entre projetos, zero descarte de evidência hard e zero violação do budget de contexto. O baseline V1 obteve 64,71% de recall obrigatório, 42,50% de recall relevante e 44,74% de seleção irrelevante.

A resposta à pergunta do experimento é: sim, para os sinais cobertos pelo corpus. Dada uma tarefa grounded, o Jarvis agora identifica paths, símbolos, imports, callers, testes, entrypoints, configs, tracebacks e fatos Git deterministicamente e com contexto limitado. A ressalva é o tamanho do corpus e a ausência de casos que precisaram efetivamente do ranking semântico: `semantic_only_recall` é `N/A`, não 100%.

## 1. Arquitetura anterior

A V1 já possuía uma base útil em `project_intelligence.py`:

| Componente | Linha auditada | Comportamento V1 |
|---|---:|---|
| `ExplorationBudget` | 63 | 80 arquivos analisados, 64 KiB/arquivo, 512 KiB total |
| `RepoMapEntry` | 71 | path, tipo, módulo, símbolos top-level, imports, size, mtime, hash |
| `ProjectSnapshot` | 111 | linguagens, arquivos importantes, testes, Git e repo map |
| `ProjectSnapshotBuilder` | 221 | `git ls-files`, AST simples, cache por size+mtime |
| `ProjectFileSelector` | 490 | Qwen escolhia até 20 paths do snapshot compacto |
| `project_understanding_metrics` | `autonomy_eval.py:270` | somente recall global e ruído |

Limitações encontradas:

- apenas os primeiros arquivos dentro do budget eram analisados;
- símbolos tinham apenas nome top-level, sem tipo, qualified name ou linhas;
- métodos, constantes e membros de enum não eram indexados;
- imports relativos não eram resolvidos para paths;
- não existiam reverse imports nem relação produção ↔ teste;
- paths de traceback e referências explícitas não tinham precedência;
- Git aparecia no snapshot, mas não como provenance de relevância;
- cache usava apenas size+mtime e podia ficar stale em alteração de mesmo tamanho/mtime;
- o Qwen recebia a decisão final diretamente e podia omitir evidência explícita;
- não havia candidate generation separada de candidate ranking.

Não havia call graph, parser de traceback, índice persistente de símbolos qualificados ou isolamento do cache por identidade criptográfica do projeto.

## 2. Baseline detalhado

O baseline foi reexecutado no mesmo corpus, com o `ProjectFileSelector`, prompt, schema e Qwen da V1. Uma tentativa inicial saturou o slot por mais de dez minutos e foi encerrada; o lote final usou timeout operacional de 90 s por caso, sem fallback ou mudança semântica. Cinco de 13 respostas ficaram inválidas.

| Sinal | Recall V1 | Suporte |
|---|---:|---:|
| referência explícita de arquivo | 75,00% | 4 |
| referência explícita de símbolo | 75,00% | 4 |
| definição de símbolo | 75,00% | 4 |
| dependência direta | 44,44% | 9 |
| reverse dependency/caller | 0,00% | 4 |
| relação com teste | 100,00% | 2 |
| arquivo de traceback | 0,00% | 1 |
| arquivo modificado | 100,00% | 1 |
| config | 100,00% | 1 |
| entrypoint | 25,00% | 4 |
| somente semântico | N/A | 0 |

O acerto de arquivo modificado/config na V1 veio da seleção semântica do path; a V1 não preservou a causa. `provenance_accuracy=0%`.

Por caso, a V1 perdeu integralmente os casos de símbolo explícito, arquivo explícito, traceback, routing/dependency e escopo web threats. Nos casos válidos, selecionou arquivos sem relação em `DEEPSEEK_DELEGATE`, semantic pass, pytest, entrypoint e Git-modified.

## 3. Error taxonomy

O evaluator registra `case_id`, task de fixture, expected file, candidates observados, classe e estágio.

Baseline:

| Erro | Quantidade |
|---|---:|
| `FALSE_SEMANTIC_MATCH` | 17 |
| `WRONG_MODULE` | 12 |
| `MISSED_CALLER` | 4 |
| `MISSED_DEPENDENCY` | 4 |
| `MISSED_EXPLICIT_SYMBOL` | 1 |
| `MISSED_EXPLICIT_FILE` | 1 |
| `MISSED_TRACEBACK_FILE` | 1 |

Candidato:

| Erro | Quantidade |
|---|---:|
| `SCOPE_EXPANSION` | 1 |

O único extra da V2 foi `project_intelligence_eval_v2.py` ao revisar `project_intelligence_v2.py`; ele é um caller direto real, mas não estava rotulado como supporting. Foi mantido como erro para não ajustar o corpus após observar o resultado.

As classes também previstas pelo evaluator são `MISSED_SYMBOL_DEFINITION`, `MISSED_TEST`, `MISSED_MODIFIED_FILE`, `UNRELATED_NEIGHBOR_FILE`, `WRONG_PROJECT`, `STALE_INDEX` e `CONTEXT_BUDGET_DROP`. As quatro últimas são verificadas principalmente por testes estruturais, porque não ocorreram no A/B real.

## 4. ProjectSnapshot V2

`ProjectSnapshotV2` foi criado em `project_intelligence_v2.py:264` sem substituir o snapshot de produção. Ele contém:

```text
project_path / project_id
languages / directories / files
entry_points / declared_entry_points
config_files / dependency_files / test_roots
git_branch / modified / untracked / diff files
import_graph / reverse_import_graph
test_relationships
known_errors / known_traceback_files
index metrics
```

O snapshot não guarda conteúdo integral. Conteúdo é lido durante indexação AST/hash e descartado; cache persiste metadados, símbolos, imports e hashes.

## 5. Repo map

O builder V2 usa `git ls-files -co --exclude-standard -z` quando existe Git e `os.walk(..., followlinks=False)` como fallback. Diretórios ignorados permanecem os mesmos da V1, incluindo `.git`, ambientes, caches, builds, `.orchestrator` e `models`.

Para cada arquivo são guardados:

```text
path / language
size / modified_ns / sha256
module / package
symbols / imports / referenced_names
is_test / is_config / is_dependency / is_entrypoint
analyzed / parse_error
```

No checkout avaliado: 224 arquivos descobertos, 3.579 símbolos, 360 edges de import e 157 relações de teste. Foram hasheados 23.256.683 bytes e parseados 2.021.493 bytes Python no cold build.

## 6. Symbol Index

A stdlib `ast` extrai:

- classes;
- funções e async functions;
- métodos e async methods;
- constantes de módulo;
- atributos de classe/membros de enum;
- nome, qualified name, kind, arquivo, módulo, linha inicial e final.

Exemplos resolvidos sem Qwen:

```text
CodexSessionResolver.resolve
→ tern/orchestrator/codex_sessions.py

DEEPSEEK_DELEGATE
→ tern/orchestrator/decision_policy.py

test_single_existing_session_reused
→ tests/test_codex_session_affinity.py
```

Nomes duplicados preservam todas as definições; não há desempate arbitrário.

## 7. Dependency relationships

`ImportRecord` preserva módulo, nomes, nível relativo e linha. Após todos os módulos serem conhecidos, imports absolutos e relativos são resolvidos para paths do mesmo projeto. O grafo reverso é derivado deterministamente.

Expansão usa apenas um hop. Raízes hard/strong sempre podem expandir. Raízes obtidas somente por estrutura expandem apenas quando existem no máximo três módulos de produção focados; isso evita que um termo amplo gere vizinhos ilimitados.

Resultado A/B combinado para imports + callers: 30,77% → 100%.

## 8. Test relationships

Uma relação produção ↔ teste é criada quando há:

1. import direto do módulo de produção pelo teste; e/ou
2. referência de símbolo, usada como reforço quando há import;
3. naming convention, apenas como sinal secundário/supporting.

Coincidência de símbolo sem import deixou de criar relação strong. Testes entram na expansão quando `test_execution=TRUE` ou quando a própria raiz é um teste. Accuracy no corpus: 100% em A e B, suporte 2; a diferença é que a V2 preserva provenance.

## 9. Git grounding

O snapshot usa branch, `git status --porcelain=v1` e `git diff --name-only HEAD`. `GIT_MODIFIED_FILE` e `GIT_DIFF_RELATIONSHIP` são sempre supporting: um arquivo modificado não vira candidato sozinho.

No caso Git do corpus, o arquivo explicitamente referenciado e modificado preservou ambas as evidências. O teste de “irrelevant modified file” confirma que outro arquivo modificado não é selecionado globalmente.

## 10. Error/traceback grounding

Paths em formatos Python traceback e `path.py:line` são resolvidos somente se pertencem ao `file_index` do projeto. Paths absolutos fora da raiz são descartados mesmo quando possuem basename igual. `known_traceback_files` fornecidos pelo runtime também entram como hard evidence, mas não são persistidos no cache.

Resultado: traceback recall 0% → 100%, com zero cross-project leakage.

## 11. Evidence Provenance

`RelevantFileEvidence` contém `source`, `strength`, `target`, `relationship` e `evidence_ref`; não usa confidence float.

| Força | Sinais principais |
|---|---|
| `HARD` | arquivo explícito, símbolo explícito, traceback exato/runtime |
| `STRONG` | definição de símbolo, import, reverse import, teste estrutural, entrypoint, diretório explícito |
| `SUPPORTING` | Git, config, estrutura do projeto, naming secundário |
| `SEMANTIC` | seleção opcional do Qwen após geração |

Evidências são agregadas e nunca substituídas. Um arquivo pode preservar simultaneamente explicit symbol, definition, import, test e Git.

Ausência do candidate set significa “sem evidência atual”, não `NOT_RELEVANT`. Não foi criado um enum tri-state de arquivo porque o evaluator e o merge não ganharam benefício adicional com ele nesta etapa.

## 12. Candidate generation

Fluxo V2:

```text
known traceback files
∪ explicit paths/basenames/directories
∪ exact code-shaped symbols
∪ traceback paths/lines
∪ bounded project-structure matches
      ↓
one-hop imports / reverse imports / tests / entrypoints
      ↓
Git evidence aggregation
      ↓
categorical ordering
      ↓
selected relevance set
      ↓
separate context slices
```

Não existe agent name, capability ranking ou execução nessa lógica. Grounded requirements apenas modulam inclusão de testes e preservam mutation scope; não são uma tabela task→file.

## 13. Semantic ranking

`ProjectCandidateRanker` é opcional e só recebe:

```text
task
grounded requirements
candidate paths
module/test flags
evidence categories/relationships
```

Ele não recebe o repositório inteiro nem conteúdo dos arquivos. Só é chamado quando soft candidates excedem slots. Hard candidates são removidos da escolha do modelo e adicionados novamente deterministicamente; portanto o Qwen não pode eliminar arquivo explícito/traceback/símbolo.

No corpus final, todos os conjuntos couberam no budget: semantic ranking calls = 0 e semantic-only recall = N/A. A interface foi testada com resposta sem candidatos e preservou o target hard.

## 14. A/B completo

| Métrica | A — V1 | B — V2 | Variação |
|---|---:|---:|---:|
| required-file recall | 64,71% | 100,00% | +35,29 pp |
| relevant-file recall | 42,50% | 100,00% | +57,50 pp |
| precision | 55,26% | 98,57% | +43,31 pp |
| F1 | 48,05% | 99,28% | +51,23 pp |
| irrelevant selection | 44,74% | 1,43% | -43,31 pp |
| explicit file | 75,00% | 100,00% | +25,00 pp |
| symbol definition | 75,00% | 100,00% | +25,00 pp |
| traceback | 0,00% | 100,00% | +100,00 pp |
| dependency/caller | 30,77% | 100,00% | +69,23 pp |
| provenance | 0,00% | 100,00% | +100,00 pp |
| JSON validity | 61,54% | 100,00% | +38,46 pp |
| cross-project leakage | 0 | 0 | igual |
| context budget violations | 0 | 0 | igual |

## 15. Recall

| Categoria B | Recall | Suporte |
|---|---:|---:|
| required files | 100,00% | 17 |
| relevant + required | 100,00% | 40 |
| explicit file | 100,00% | 4 |
| explicit symbol | 100,00% | 4 |
| symbol definition | 100,00% | 4 |
| dependency | 100,00% | 9 |
| reverse dependency | 100,00% | 4 |
| test relationship | 100,00% | 2 |
| traceback | 100,00% | 1 |
| modified file | 100,00% | 1 |
| config | 100,00% | 1 |
| entrypoint | 100,00% | 4 |
| semantic only | N/A | 0 |

Os suportes pequenos, especialmente traceback/Git/config, limitam confiança estatística. Os gates são satisfeitos no corpus, não afirmados universalmente.

## 16. Precision/noise

A V2 selecionou em média 5,38 arquivos e gerou 5,38 candidatos por caso. O único arquivo fora do conjunto accepted foi um caller real do próprio módulo V2. Ruído global: 1/70 = 1,43%, abaixo do limite de 15%.

O caso estrutural mais amplo, sessões Codex duplicadas, gerou sete arquivos focados em Codex/sessões e selecionou todos; após limitar expansão transitiva de raízes estruturais amplas, não houve vizinhos genéricos de runtime/security/prompt.

## 17. Context efficiency

| Medida média | V1 | V2 |
|---|---:|---:|
| candidate count | 2,92 | 5,38 |
| selected count | 2,92 | 5,38 |
| bytes dos arquivos selecionados | 78.678 | 200.020 |
| bytes de contexto/slices | 22.042 | 34.413 |
| tokens estimados | 5.511 | 8.605 |

Na V1, `context_bytes` é o snapshot metadata enviado ao selector, não o futuro conteúdo dos arquivos; na V2 é o plano de slices de conteúdo. Portanto bytes selecionados e contexto não são comparações equivalentes de consumo final.

Budgets V2:

```text
max_candidates = 32
max_selected_files = 12
max_context_bytes = 48.000
max_context_bytes_per_file = 12.000
max_symbol_lines = 240
```

Hard evidence pode exceder o limite de quantidade de arquivos, mas nunca o limite total de contexto: cada arquivo continua selecionado como relevante e recebe slice somente enquanto houver budget. Resultado: zero hard evidence dropped e máximo estimado abaixo de 12 mil tokens, deixando margem na janela de 16.384 do runtime atual.

## 18. Incremental performance/cache

Medição real, 224 arquivos:

| Medida | Cold | Restart/warm |
|---|---:|---:|
| repo scan | 133,994 ms | 133,008 ms |
| index build | 1.657,192 ms | 524,779 ms |
| arquivos AST/indexados | 224 | 0 |
| arquivos reutilizados | 0 | 224 |
| bytes parseados | 2.021.493 | 0 |
| bytes hasheados | 23.256.683 | 23.256.683 |

O cache persiste versão, project identity, root exata, hashes, símbolos e imports. Arquivo alterado invalida símbolos/imports mesmo com size e mtime artificialmente preservados; arquivo removido desaparece na reconciliação.

Tradeoff deliberado: todos os bytes ainda são hasheados no warm scan para impedir stale silencioso; AST e extração são executados somente nos arquivos alterados. Isso privilegia correctness sobre eliminar completamente I/O sequencial. O warm build ficou 68,3% menor.

## 19. Cross-project isolation

`project_id = sha256(normcase(resolved_root))[:20]`. O cache só é aceito quando versão, ID e project path coincidem. Um teste reutiliza propositalmente o mesmo cache em dois projetos e confirma zero records reaproveitados e zero paths do primeiro projeto.

Tracebacks absolutos fora da raiz e paths que não pertencem ao índice são rejeitados. Métrica A/B: `cross_project_file_leakage=0`.

## 20. Regressões

```text
Project Intelligence V2
Task Requirement Grounding
Explicit Agent Binding
Codex Session Affinity
Autonomy Foundation
```

Resultado direcionado: 113 passed. Nenhuma classe ou fluxo anterior foi alterado; V2 reside em módulos paralelos.

## 21. Safety

- `PathPolicy` pode ser injetada no builder e bloqueia root fora da allowlist;
- symlinks são ignorados e paths resolvidos precisam permanecer sob a raiz;
- cache/index não concede permissão de ferramenta;
- `read_scope` é separado de allowed/forbidden mutation targets;
- ler dependência proibida para mutação continua possível como análise;
- context planning não executa leitura futura nem envia conteúdo automaticamente;
- future worker handoff é apenas estrutura, com `dry_run=true` e `execution_authorized=false`;
- nenhuma mudança em filesystem gates, web threats, confirmation, availability ou sessões;
- nenhuma seleção ou delegação automática.

## 22. Custo

| Latência por caso | V1/Qwen | V2 determinística |
|---|---:|---:|
| p50 | 58.103,973 ms | 24,094 ms |
| p90 | 64.457,438 ms | 27,975 ms |
| p95 | 90.140,349 ms | 101,388 ms |
| total semantic ranking | 781.972,431 ms | 0 ms |

O cold index V2 custa aproximadamente 1,66 s uma vez; restart/warm custa 0,52 s. Candidate generation ocorre depois do índice. A V1 chamou Qwen 13 vezes; a V2 não chamou o modelo neste corpus.

## 23. Arquivos modificados

| Arquivo | Linhas/estrutura principal |
|---|---|
| `tern/orchestrator/project_intelligence_v2.py` | evidence `:41/:864`, budget `:103`, symbols `:112`, indexed file `:171`, snapshot `:264`, builder `:454`, ranker `:1079`, generator `:1146` |
| `tern/orchestrator/project_intelligence_eval_v2.py` | case `:63`, observers `:135/:181`, evaluator `:278` |
| `tests/data/project_intelligence_v2_diagnostic.jsonl` | 13 casos; corpora anteriores intactos |
| `tests/test_project_intelligence_v2.py` | 23 testes V2 |
| `docs/project-intelligence-v2-report.md` | este relatório |

Artefatos A/B e cache ficam em `.orchestrator/` e não são versionados.

## 24. Testes e validação

```text
testes V2: 23 passed
regressões direcionadas: 113 passed
suíte completa: 857 passed, 1 skipped, 1 warning
py_compile: passed
git diff --check: passed
```

O warning é a depreciação preexistente de `VOICE_TTS_SPEED` no teste Piper. O skip também já existia.

## Integração futura, não ativada

`future_worker_handoff()` desenha o payload futuro:

```text
task
grounded requirements
relevant files + evidence
allowed/forbidden mutation targets
dry_run=true
execution_authorized=false
```

O Delegation Payload real não foi alterado. O próximo passo permitido por estes resultados é somente um experimento separado de Automatic Agent Selection em dry-run. Ativação, delegação, mutation, agent loop e replan continuam proibidos até aprovação específica.
