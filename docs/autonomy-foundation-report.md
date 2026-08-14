# Jarvis Autonomy Foundation — relatório de baseline

Data: 2026-08-13

Escopo: fundação determinística, observabilidade e avaliação em dry-run.
Checkout avaliado: `D:\tern`, branch `feat/native-ptbr-f5-tts`.

## 1. Decisões

```text
REJEITADO — BONSAI 27B

APROVADO — AUTONOMY FOUNDATION

NOT READY FOR AUTOMATIC AGENT SELECTION
```

O Bonsai foi rejeitado porque falhou no gate de saída estruturada do runtime atual e foi muito mais lento no hardware real. A fundação determinística está pronta e testada, mas o modelo local atual ainda não tem qualidade suficiente para ativar seleção automática: no corpus live com snapshot obteve 36,67% de acerto dos campos de requisitos e 0% de resolução correta dos casos de candidato único.

Nenhuma seleção automática foi ligada. `AgentSelectionProposal` é somente telemetria, sempre retorna `dry_run=true` e `execution_authorized=false`.

## 2. Compatibility gate e modelo A/B

### Ambiente real

- CPU: Intel Core i5-7400, 4 cores/4 threads.
- RAM: 16 GB.
- GPU: AMD Radeon RX 580, 4 GB.
- Runtime de controle: `llama-server` b10173 Vulkan.
- Controle: `Qwen_Qwen3.5-4B-Q4_K_M.gguf`, 3.013.027.808 bytes.
- Candidato: `Ternary-Bonsai-27B-Q2_g64.gguf`, 7.585.330.240 bytes.
- Configuração mantida: contexto 16.384, um slot, flash attention, KV `q8_0`, temperatura 0, `max_tokens` idêntico por teste, reasoning off, mesma policy, schemas, validators, retries e corpus.

### Gate de compatibilidade

| Contrato | Qwen atual | Bonsai 27B g64 |
|---|---:|---:|
| Windows + Vulkan | passou | passou isoladamente |
| CPU/CUDA disponíveis no runtime | suportado pelo build | suportado pelo build |
| carregamento no hardware | passou | passou somente após parar o Qwen |
| JSON simples | passou | passou |
| `json_schema`/grammar | passou | **falhou** |
| `finish_reason` | passou | passou em chamada não estruturada |
| `max_tokens` | passou | passou em chamada não estruturada |
| `temperature` | passou | passou em chamada não estruturada |
| streaming | passou | passou |
| chamada sem tools | passou | passou |
| contexto usado pelo Jarvis (16K) | passou | carregou |

A variante `Q2_g64` foi escolhida porque roda no llama.cpp mainline recente. A variante group-128 requer o fork Prism; nenhuma migração silenciosa foi feita. O servidor atual conseguiu carregar o candidato com offload automático, mas não junto do Qwen: tentativas com ambos residentes terminaram em falta de memória.

O bloqueio decisivo foi reproduzido em todas as 12 entradas estruturadas:

```text
HTTP 400: Failed to initialize samplers:
Unexpected empty grammar stack after accepting piece: <think>
```

Mesmo com `--reasoning off`, o template do candidato emitiu delimitadores de raciocínio incompatíveis com a gramática do `json_schema`. Corrigir isso exigiria template, payload ou servidor específico para o candidato e quebraria a comparação controlada.

### Medidas

| Medida | Qwen atual | Bonsai 27B g64 |
|---|---:|---:|
| disco do GGUF | 3,01 GB | 7,59 GB |
| working set observado | 3,77 GB | 7,88–7,90 GB |
| VRAM isolada | não exposta pelo runtime | não exposta pelo runtime |
| chamada simples, latência | 609 ms | 8.690 ms |
| prompt throughput simples | 48,467 tok/s | 2,268 tok/s |
| geração simples | 28,942 tok/s | 3,458 tok/s |
| corpus estruturado: JSON válido | 91,67% | 0% |
| first pass válido | 41,67% | 0% |
| válido após retry | 91,67% | 0% |
| retry rate | 58,33% | 100% |
| fallback rate | 0% | 0% |
| requisitos, acerto por campo | 36,67% | 0% |
| constraint accuracy | 55,00% | 0% |
| eligibility precision | 72,22% | 100% sem propostas (não informativo) |
| eligibility recall | 59,09% | 0% |
| single candidate accuracy | 0% | 0% |
| explicit override | 100% | 0% por falha anterior à proposta |
| p50 estruturado | 122,662 s | 87 ms até erro HTTP |
| p90 estruturado | 134,464 s | 107 ms até erro HTTP |
| p95 estruturado | 134,464 s | 107 ms até erro HTTP |
| prompt tokens | 78.956 | 0 aceitos |
| generated tokens | 2.918 | 0 |

Os percentis do Bonsai medem somente a rejeição imediata do sampler, não inferência útil. As alegações do model card não foram usadas como substituto das medições locais.

Após registrar os resultados, o GGUF rejeitado do Bonsai foi removido (7.585.330.240 bytes, recuperável por novo download). O Qwen de controle foi preservado e restaurado como servidor ativo, healthy e compatible em `127.0.0.1:8080`.

## 3. Agent Capability Profiles

Profiles são construídos por `CapabilityProfileBuilder.from_registry`: ferramentas registradas, handlers, managers e fatos de runtime produzem evidências por capability. Availability fica em `AgentRuntimeAvailability`, fora do profile.

| Capability | Local | Codex | DeepSeek |
|---|:---:|:---:|:---:|
| repository read | sim | sim | não |
| repository write | sim | sim | não |
| filesystem read | sim | sim | não |
| filesystem write | sim | sim | não |
| code analysis | sim | sim | sim |
| code edit | sim | sim | não |
| test execution | não | sim | não |
| long-running job | não | sim | não |
| persistent session | não | sim | sim |
| general reasoning | sim | sim | sim |
| code review | sim | sim | sim |
| web access | sim | não demonstrado | não demonstrado |
| mutation capable | sim | sim | não |
| read-only capable | sim | sim | sim |
| semantic interpretation | sim | não é papel do worker | não é papel do worker |
| requirement extraction | sim | não é papel do worker | não é papel do worker |

`sim` significa evidência operacional existente, não preferência. O baseline obteve 100% de cobertura de evidência para capabilities declaradas. Isso não concede permissão: os gates de path, approval, web, projeto e handler continuam posteriores.

### DeepSeek real

O DeepSeek atual recebe texto via `delegate_to_deepseek`, mantém sessão revisável por `review_deepseek_session` e pode fazer reasoning, análise e review somente sobre o conteúdo fornecido. Não possui ferramenta própria de filesystem, repositório, execução de testes, mutação ou web. No runtime auditado está enabled/configured, mas desligá-lo altera apenas availability; o profile permanece igual.

### Codex real

`delegate_to_codex` fundamenta leitura/escrita de projeto, análise/edição, testes, reasoning e review. `get_codex_job_status` fundamenta job longo. `CodexSessionRegistry` fundamenta sessão persistente. Toda proposta futura ainda deve atravessar o `CodexSessionResolver` existente; não foi criado registry paralelo.

### Orquestrador local real

O local possui tools diretas de filesystem e web, além do modelo para interpretação, análise de requisitos, reasoning e síntese. Não foi declarada execução de testes porque não existe handler registrado específico que a fundamente. Seu papel futuro de coordenação não implica capabilities dos workers.

## 4. Project Intelligence e repo map

`ProjectSnapshot` registra path, linguagens, diretórios/arquivos importantes, módulos, entry points, testes, configurações, dependências, branch/status Git, modificados, untracked, mudanças recentes, estado/falhas de teste e `RepoMapEntry`.

Cada entrada contém path, tipo, módulo, símbolos/imports quando obtidos deterministicamente, tamanho, mtime, hash opcional e indicação de análise. Python usa `ast`; nenhuma extração determinística foi delegada ao LLM.

Limites padrão:

```text
max_files_per_analysis = 80
max_bytes_per_file = 65.536
max_total_context_bytes = 524.288
max_repo_entries = 10.000
compact max_entries = 80
compact max_context_bytes = 24.000
```

O cache reutiliza entradas quando tamanho e mtime não mudam; atualização é atômica. Symlinks fora da raiz e diretórios de build, cache, modelos e estado são ignorados. No checkout real: 175 arquivos mapeados; scan cacheado reutilizou 175/175, analisou 0 bytes novos em cerca de 230 ms; snapshot compacto ficou em 23.924 bytes com 72 entradas. Produção e testes são intercalados em proporção 2:1, mantendo módulos centrais e testes dentro do orçamento.

## 5. Task Requirements

`TaskRequirements` representa somente necessidade:

```text
capabilities
mutation_required
read_only_required
target_scope
risk_level
expected_files
forbidden_files
tests_requested
ambiguity_material
```

O schema enviado ao modelo não contém agent, requested_agent, selected_agent, ranking ou preferência. Não há keyword, regex, substring ou tabela de comandos para escolher worker.

## 6. Eligibility e precedência explícita

Para cada agente:

```text
capability_eligible = requirements ⊆ profile.capabilities
permission_eligible = nenhuma capability negada pelo gate
eligible = capability_eligible AND permission_eligible
executable_now = eligible AND availability
```

Reason codes estruturados incluem `CODEX_ELIGIBLE`, `DEEPSEEK_MISSING_REPOSITORY_WRITE`, `LOCAL_MISSING_TEST_EXECUTION`, `*_UNAVAILABLE`, `*_PERMISSION_DENIED`, `NO_ELIGIBLE_AGENT`, `MULTIPLE_ELIGIBLE_AGENTS` e `REQUESTED_AGENT_CANNOT_SATISFY_REQUIREMENTS`.

Se há binding explícito, `selected_agent=requested_agent` e `selection_source=explicit_user`. Incompatibilidade ou indisponibilidade bloqueiam; não existe fallback Codex ↔ DeepSeek. O evaluator preservou 100% dos bindings explícitos no controle determinístico.

## 7. Dry-run e casos multiagente

No corpus de verdade conhecida (12 casos):

- 6 resultaram em múltiplos agentes elegíveis e ficaram sem proposta;
- 2 tiveram candidato único e foram resolvidos deterministicamente;
- 2 usaram agente explícito;
- 1 não teve agente elegível;
- 1 bloqueou o agente explícito incompatível.

Todas as 12 propostas mantiveram `execution_authorized=false`; execução observada: zero. Não foi inventado desempate para os seis casos multiagente.

Métricas determinísticas: task requirements 100%, capability evidence 100%, availability separation 100%, eligibility precision/recall 100%, false/missed eligible 0%, single candidate 100%, explicit override 100%.

## 8. Project Understanding benchmark

Quatro casos somente-leitura avaliaram routing, afinidade Codex, limites DeepSeek e a própria fundação. O Qwen selecionou JSON válido em 2/4, `relevant_file_recall=50%` e `irrelevant_file_selection_rate=34,17%`; latências foram 64,220 s, 82,005 s, 68,367 s e 68,202 s. Nos dois casos válidos, distinguiu arquivos de produção e teste, mas selecionou arquivos extras demais. O Bonsai não pôde participar porque o mesmo `json_schema` falha no gate.

## 9. Verification foundation e provenance

`VerificationResult` compara deterministicamente expected/actual files, tests requested/executed, exit code, forbidden files, scope violation, unexpected mutation e artifact existence. A satisfação qualitativa fica separada e opcional. A ordem futura é Git diff, exit code, tool/job status e artifact antes de julgamento de modelo.

`AgentResultProvenance` preserva agent, session/thread, job/run, task, timestamp, status, artifacts/changes e verification state.

## 10. Estado futuro, budgets e escalada

Estados declarados, ainda sem loop: `OBSERVE`, `PLAN`, `SELECT`, `DELEGATE`, `WAIT_OBSERVE`, `VERIFY`, `REPLAN`, `COMPLETE`, `HUMAN_ESCALATION` e `BLOCKED`.

`AutonomyBudgets` reserva max plan steps, delegations, replans, failures/agent, project reads, runtime e tokens. A arquitetura prevê escalada para ambiguidade destrutiva, permissão ausente, operação irreversível, conflito, falha repetida, budget esgotado, nenhum agente elegível, verificação repetidamente falha e mudança inesperada do projeto.

Não existe `while True`, goal autogerado, retry chain autônoma, swap de agente, fallback, execução em background ou mutação a partir do dry-run.

## 11. Regressões e safety

Os testes cobrem profile truthfulness, availability independente, schema sem agente, candidato único/múltiplo/nenhum, read-only, mutation, test execution, inspeção, informational, DeepSeek unavailable, explicit precedence, no fallback, no execution, snapshot, changed files, incremental cache, budget, seleção fora do snapshot e verificação por fatos.

As suites dedicadas preservaram Explicit Agent Binding e Codex Session Affinity: 64/64 testes passaram. A suite completa passou com `808 passed, 1 skipped, 1 warning` em 21,91 s; o warning é a depreciação já existente de `VOICE_TTS_SPEED`.

Eligibility, selection e planning não concedem permissão. PathPolicy, confirmations, web safety, project restrictions, session isolation e mutation gates existentes não foram removidos nem contornados.

Os experimentos rejeitados não foram usados pela fundação: `compound_plan → ORDERED` não recebeu nova canonicalização, não há length recovery, `cross_field_invariants` permanece vazio por padrão e o prompt genérico de command preservation não é o prompt padrão do interpreter. A única canonicalização ativa continua sendo a allow-list estrutural já aprovada de deduplicação exata de constraints.

## 12. Arquivos desta fundação

- `tern/orchestrator/autonomy_foundation.py`: profiles, requirements, eligibility, proposal, verification, provenance e estados futuros.
- `tern/orchestrator/project_intelligence.py`: snapshot, repo map, AST, cache, budget e selector read-only.
- `tern/orchestrator/autonomy_eval.py`: corpus, métricas e CLI de dry-run.
- `tests/data/autonomy_foundation_diagnostic.jsonl`: corpus diagnóstico separado; corpora históricos intactos.
- `tests/data/project_understanding_diagnostic.jsonl`: corpus de relevância de arquivos separado.
- `tests/test_autonomy_foundation.py`: testes da fundação.
- `docs/autonomy-foundation-report.md`: este relatório.

Arquivos `.orchestrator/autonomy-*.json` são resultados locais ignorados pelo Git e podem ser regenerados. Nenhum corpus histórico foi alterado e nenhuma versão v4 foi criada.

## 13. Próximo gate

Antes de qualquer experimento de seleção automática, melhorar e reavaliar a extração de requisitos do modelo local sem alterar a arquitetura para mascarar baixa qualidade. Critério mínimo: JSON estável no primeiro pass, acurácia de requisitos e candidato único materialmente superiores, e redução da seleção de arquivos irrelevantes. A ativação continua dependendo de aprovação separada.

## Referências do candidato

- Coleção oficial: <https://huggingface.co/collections/prism-ml/bonsai-27b>
- GGUF e requisitos de runtime: <https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf>
- Demo/runtime oficial Prism: <https://github.com/PrismML-Eng/Bonsai-demo/blob/main/README.md>
