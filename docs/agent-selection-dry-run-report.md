# Automatic Agent Selection — dry-run e Selection Provenance

Experimento isolado. Nenhuma execução, delegação, job, sessão ou mutação foi
autorizada. Modelo local: Qwen3.5-4B Q4_K_M (llama.cpp b10437, endpoint
`http://127.0.0.1:8080`). Camadas anteriores não foram reotimizadas.

## Fase 0 — checkpoint

O checkout canônico `C:\Users\User\JARVIS` já estava limpo antes do experimento:

```text
git status --short   -> vazio
git diff --check     -> exit 0
git rev-parse HEAD   == git rev-parse origin/main  (439f379)
```

Não havia alteração genérica pendente do LocalModelRuntime: ela já estava
publicada em `439f379 feat: add model-agnostic local model runtime contract`.
Logo não existia diff para o commit sugerido
`refactor: preserve generic local model runtime infrastructure`; criar um commit
vazio seria ruído. Auditoria complementar: nenhum `.gguf` rastreado ou presente
no repositório, e os únicos artefatos não versionados são ignorados por
`.gitignore` (`.orchestrator/`, `__pycache__/`, `.pytest_cache/`). GGUF, logs
temporários, caches, locks e artefatos Bonsai estão fora do índice.

## 1. Arquitetura

```text
task
 ↓
GroundedTaskRequirements        (task_requirement_grounding, intocado)
 ↓
AgentCapabilityProfile          (autonomy_foundation, intocado)
 ↓
GroundedEligibility             (intocado; UNKNOWN não elimina, não autoriza)
 ↓
AgentRuntimeAvailability        (separada; nunca reescreve eligibility)
 ↓
AgentSelectionProfile + SelectionFactor      (novo)
 ↓
AgentSelectionProposal + provenance          (novo, dry-run)
```

Capability responde `CAN DO?`. Selection factor responde
`AMONG ELIGIBLE AGENTS, WHAT MAKES THIS AGENT A BETTER FIT?`. Capability nunca
vira preferência: `test_capability_alone_is_not_a_preference` prova que Codex
possui `code_review` e `persistent_session` e mesmo assim tem zero fatores de
preferência em tarefa somente-leitura.

Anti-keyword routing estrutural: o seletor semântico **não recebe o texto da
tarefa**. O payload contém apenas dimensões grounded, escopo, risco,
proibições, fatores, capabilities, availability e fatos determinísticos de
projeto. Não existe regex, substring, tabela de nomes ou heurística textual em
nenhum ponto do caminho.

## 2. Precedência

| Ordem | Fonte | Proposta | Chamadas Qwen |
|---|---|---|---:|
| 1 | `EXPLICIT_USER` | `requested_agent`, sem substituição | 0 |
| 2 | `NO_ELIGIBLE_AGENT` | nenhuma | 0 |
| 3 | `SINGLE_ELIGIBLE_AGENT` | único elegível | 0 |
| 4 | `NO_AVAILABLE_ELIGIBLE_AGENT` | nenhuma | 0 |
| 5 | `ONLY_AVAILABLE_ELIGIBLE_AGENT` | único disponível elegível | 0 |
| 6 | `UNRESOLVED` (requisitos ambíguos) | nenhuma | 0 |
| 7 | `DETERMINISTIC_SELECTION` | candidato único após política/justificação | 0 |
| 8 | `SEMANTIC_MULTI_AGENT` | escolha entre candidatos justificados | 1 |
| 9 | `UNRESOLVED` / `INVALID_SELECTION` | nenhuma | ≤1 |

A ordem nunca é invertida. Availability é calculada em conjunto separado
(`available_eligible_agents`) e o que era elegível e ficou indisponível é
preservado em `eligible_but_unavailable`.

## 3. Selection sources e reason codes

Sources: `EXPLICIT_USER`, `SINGLE_ELIGIBLE_AGENT`,
`ONLY_AVAILABLE_ELIGIBLE_AGENT`, `DETERMINISTIC_SELECTION`,
`SEMANTIC_MULTI_AGENT`, `UNRESOLVED`, `NO_ELIGIBLE_AGENT`,
`NO_AVAILABLE_ELIGIBLE_AGENT`, `INVALID_SELECTION`.

Reason codes: `EXPLICIT_AGENT_READY`, `REQUESTED_AGENT_UNAVAILABLE`,
`REQUESTED_AGENT_CANNOT_SATISFY_REQUIREMENTS`,
`REQUESTED_AGENT_EXECUTION_BLOCKED`, `REQUESTED_AGENT_UNKNOWN`,
`SINGLE_ELIGIBLE_AGENT`, `SINGLE_ELIGIBLE_AGENT_UNAVAILABLE`,
`NO_ELIGIBLE_AGENT`, `NO_AVAILABLE_ELIGIBLE_AGENT`, `AMBIGUOUS_REQUIREMENTS`,
`ALL_CANDIDATES_REQUIRE_EXPLICIT_REQUEST`,
`POLICY_RESTRICTED_AUTOMATIC_CANDIDATES`, `UNIQUE_JUSTIFIED_CANDIDATE`,
`NO_JUSTIFIED_CANDIDATE`, `SEMANTIC_SELECTION_UNAVAILABLE`,
`SEMANTIC_UNRESOLVED`, `MODEL_PARSE_FAILURE`,
`PROPOSED_AGENT_OUTSIDE_CANDIDATE_SET`, `UNJUSTIFIED_SEMANTIC_SELECTION`,
`BEST_FACTOR_FIT`, `EQUAL_FIT`, `INSUFFICIENT_BASIS`.

Confiança categórica, sem float arbitrário: `DETERMINISTIC`, `SUPPORTED`,
`AMBIGUOUS`, `UNRESOLVED`. `SUPPORTED` só é emitida quando o modelo declara
`LOW`/`BEST_FACTOR_FIT` **e** o agente proposto tem estritamente mais fatores de
suporte reais que qualquer outro candidato.

## 4. Selection factors realmente usados

Derivados de requisitos:
`MUTATION_REQUIRED`, `READ_ONLY_TASK`, `TEST_EXECUTION_REQUIRED`,
`REPOSITORY_SCOPE_REQUIRED`, `NO_REPOSITORY_ACCESS_REQUIRED`,
`LONG_RUNNING_JOB_REQUIRED`, `AMBIGUOUS_REQUIREMENTS`.

Derivados de agente, cada um com evidência:

| Fator | Fonte real |
|---|---|
| `IMPLEMENTATION_SUPPORT` | capability `code_edit`+`mutation_capable` com evidência de tool registrada |
| `TEST_EXECUTION_SUPPORT` | capability `test_execution` (Codex: `delegate_to_codex`) |
| `LONG_RUNNING_JOB_SUPPORT` | capability `long_running_job` (`get_codex_job_status`) |
| `STRUCTURAL_READ_ONLY_GUARANTEE` | ausência total de capability de escrita; DeepSeek não recebe filesystem/shell (`docs/deepseek.md`) |
| `LOCAL_EXECUTION_NO_REMOTE_SIDE_EFFECT` | runtime: tools in-process, nenhum job/sessão remota criada |
| `EXISTING_REUSABLE_SESSION` | fato do `CodexSessionRegistry` passado como snapshot |
| `PROJECT_AFFINITY` | affinity persistida (Codex) ou sessão de projeto DeepSeek |
| `EXPLICIT_REQUEST_REQUIRED_BY_POLICY` | `DEEPSEEK_AUTO_ESCALATION=false` (`docs/agent-decision-policy.md`), polaridade `EXCLUDE` |

Duas decisões explícitas para não fabricar preferência:

- capability `persistent_session` isolada **não** é fator; só o fato operacional
  de sessão reutilizável existente conta;
- nenhum traço de modelo foi inventado (nada de "DeepSeek raciocina melhor").

Candidato sem nenhum fator de suporte não é proposto automaticamente. Se sobra
exatamente um candidato justificado, a decisão é determinística e o Qwen não é
chamado; se sobram dois ou mais, existe decisão real e o Qwen decide entre eles.

## 5. Corpus

`tests/data/agent_selection_diagnostic.jsonl`, novo e separado: 31 casos.
Nenhum corpus semantic v2/v3, grounding ou Project Intelligence V2 foi alterado.

Categorias: historical_multi_agent (6), read_only_diagnosis (2, uma com sessão),
architecture_review (3, uma com escalation ligada), implementation,
diagnosis_and_proposed_fix, implementation_and_tests, code_review, verification,
root_cause_analysis, explicit_codex, explicit_deepseek,
explicit_unavailable_codex, explicit_unavailable_deepseek,
explicit_ineligible_deepseek, single_eligible_local, no_eligible_agent,
availability (4 combinações), ambiguous_equal_fit, contrastivos.

Pares contrastivos (a diferença aparece nos requirements, não em palavras):
`AS-001/AS-004`, `AS-022/AS-023`, `AS-024/AS-025`, mais o trio
`AS-024` (leitura) → `AS-025` (mutação) → `AS-006` (mutação + testes).

Ground truth por caso: `requested_agent`, `eligible_agents`, `available_agents`,
`acceptable_selected_agents`, `preferred_agent` (somente quando objetivamente
justificável), `selection_sources`, `required_factors`, `forbidden_agents` e
`unresolved_acceptable` para ambiguidade legítima.

## 6. Seis casos históricos multiagente

Conjunto de elegíveis preservado em 6/6 (`eligibility_exact=true`),
provenance correta em 6/6:

| Caso | Elegíveis | Source | Proposto |
|---|---|---|---|
| AS-H01 (AF-001) | local, codex, deepseek | `DETERMINISTIC_SELECTION` | local |
| AS-H02 (AF-002) | local, codex | `DETERMINISTIC_SELECTION` | local |
| AS-H03 (AF-003) | local, codex | `SEMANTIC_MULTI_AGENT` | local |
| AS-H04 (AF-005) | local, codex | `DETERMINISTIC_SELECTION` | local |
| AS-H05 (AF-006) | local, codex, deepseek | `DETERMINISTIC_SELECTION` | local |
| AS-H06 (AF-010) | local, codex, deepseek | `UNRESOLVED` | nenhum |

Nenhuma expectativa de eligibility foi relaxada para facilitar seleção.

## 7. Explicit agent

5 casos explícitos, preservação 100%, zero inferência:

```text
AS-010 codex disponível e elegível        -> EXPLICIT_AGENT_READY
AS-011 deepseek elegível                  -> EXPLICIT_AGENT_READY
AS-012 codex indisponível                 -> REQUESTED_AGENT_UNAVAILABLE
AS-013 deepseek indisponível              -> REQUESTED_AGENT_UNAVAILABLE
AS-014 deepseek incapaz (write + testes)  -> REQUESTED_AGENT_CANNOT_SATISFY_REQUIREMENTS
```

Em AS-012/AS-013/AS-014 `execution_possible=false` e não houve troca silenciosa:
Codex permaneceu fora da proposta em AS-014 mesmo sendo o único elegível.

## 8. Single eligible

3 casos, acurácia 100%, zero chamadas: `AS-006` (mutação + testes → codex),
`AS-008` (verificação com testes → codex), `AS-015` (web → local).
`single_candidate_model_calls = 0`.

## 9. Availability

```text
codex + deepseek disponíveis, local ok    -> decisão normal
local indisponível  (AS-017)              -> ONLY_AVAILABLE_ELIGIBLE_AGENT codex
codex indisponível  (AS-018)              -> ONLY_AVAILABLE_ELIGIBLE_AGENT local
nenhum elegível disponível (AS-019)       -> NO_AVAILABLE_ELIGIBLE_AGENT
somente deepseek disponível (AS-020)      -> ONLY_AVAILABLE_ELIGIBLE_AGENT deepseek
```

`availability_handling_accuracy = 100%`. Eligibility recalculada com todas as
combinações permanece idêntica: `AVAILABILITY_CORRUPTED_ELIGIBILITY = 0`,
`availability_separation_accuracy = 100%`. Em AS-018 o Codex continua registrado
como elegível-mas-indisponível.

## 10. Multi-agent semantic selection

8 casos chegaram ao caminho semântico, sempre com exatamente 1 inferência:

| Caso | Candidatos | Proposto | Reason | Confiança |
|---|---|---|---|---|
| AS-H03 | local, codex | local | EQUAL_FIT | AMBIGUOUS |
| AS-002 | local, codex (sessão) | local | EQUAL_FIT | AMBIGUOUS |
| AS-003 | local, deepseek | local | EQUAL_FIT | AMBIGUOUS |
| AS-004 | local, codex | local | EQUAL_FIT | AMBIGUOUS |
| AS-009 | local, codex (sessão) | codex | BEST_FACTOR_FIT | SUPPORTED |
| AS-021 | local, codex (sessão) | codex | BEST_FACTOR_FIT | SUPPORTED |
| AS-023 | local, codex | local | EQUAL_FIT | AMBIGUOUS |
| AS-025 | local, codex | local | EQUAL_FIT | AMBIGUOUS |

`multi_agent_selection_accuracy = 100%` contra o ground truth aceitável;
`acceptable_selection_rate = 100%`; nenhuma escolha fora do conjunto permitido.

Q1 (sessão como fator): em 3 casos havia sessão Codex reutilizável e affinity.
O modelo escolheu Codex em 2 e Local em 1, ou seja, a existência de sessão não
determinou a escolha sozinha. AS-002 e AS-009 têm fatos quase idênticos e
receberam decisões diferentes (local vs codex) — ambas aceitáveis, mas é
sensibilidade real registrada, não estabilidade semântica.

## 11. Casos ambíguos

`AS-H06` terminou `UNRESOLVED` deterministicamente (ambiguity material +
`mutation_required=UNKNOWN`), sem inferência. `AS-019` terminou
`NO_AVAILABLE_ELIGIBLE_AGENT`. `AS-021` aceitava `UNRESOLVED` e recebeu proposta
justificada (codex com 3 fatores contra 2 do local). Nenhum falso `UNRESOLVED`:
`unresolved_precision = 100%`, `unresolved_recall = 100%`.

## 12. Métricas (corpus completo, Qwen ao vivo, replay 3)

```text
explicit_agent_preservation           100%
single_eligible_selection_accuracy    100%
no_eligible_accuracy                  100%
availability_handling_accuracy        100%
multi_agent_selection_accuracy        100%
acceptable_selection_rate             100%
ineligible_agent_selection_rate         0%
unavailable_agent_selection_rate        0%
unjustified_selection_rate              0%
unresolved_precision                  100%
unresolved_recall                     100%
selection_provenance_accuracy         100%
selection_factor_accuracy             100%
eligibility_exactness                 100%
availability_separation_accuracy      100%
selection_consistency                 100%  (8 casos x 3 execuções)
```

`UNJUSTIFIED_SELECTION` definido como proposta cujo agente não possui nenhum
fator de suporte real (requisito, capability, fato operacional ou política).
O motor rejeita esse caso com `UNJUSTIFIED_SEMANTIC_SELECTION` em vez de trocar
de agente, então a taxa é 0% por construção e verificada no corpus.

Erros observados: apenas `BAD_SELECTION_FACTOR = 2`. Em AS-002 e AS-003 o modelo
propôs `local` mas citou um fator pertencente ao outro candidato
(`PROJECT_AFFINITY` do Codex, `STRUCTURAL_READ_ONLY_GUARANTEE` do DeepSeek). O
motor descartou a citação inválida, manteve os fatores reais calculados
deterministicamente e registrou o erro. Atribuição de fator do modelo: 6/8;
provenance final: 8/8 correta.

Duas definições de métrica foram corrigidas durante a análise, sem tocar no
prompt nem no modelo: citação de fator agora é validada contra fatores do agente
**e** fatores da tarefa, e `OVERCONFIDENT_SELECTION` passou a exigir que o
agente escolhido não tenha vantagem estrita em número de fatores.

## 13. Custo Qwen

```text
chamadas totais                  24  (8 casos semânticos x 3 execuções)
chamadas por caso (1 execução)  0,26
chamadas evitadas                22  (explicit 5, single 3, no eligible 1,
                                     only available 3, determinístico 9,
                                     unresolved determinístico 1)
latência semântica média     13 403 ms
p50                          11 739 ms
p90                          19 280 ms
p95                          19 655 ms
faixa                  11 844 – 22 877 ms
latência determinística p50    0,087 ms
p90                            4,283 ms
p95                            4,530 ms
```

Nenhum caso usou mais de uma inferência de seleção. Não existe verificação
semântica extra nem segunda seleção.

## 14. Safety

```text
execution_authorized      false em 31/31
automatic_tool_calls          0
automatic_delegations         0
filesystem_mutations          0
codex_jobs_created            0
deepseek_jobs_created         0
sessions_resolved             0
```

`AgentSelectionProposal` fixa `execution_authorized=False`, `dry_run=True`,
`jobs_created=0`, `delegations=0`, `filesystem_mutations=0` como campos não
inicializáveis. `CodexSessionResolver.resolve` foi substituído por um guard que
falha se chamado: nenhuma chamada ocorreu (`session_resolved=false`), inclusive
quando Codex foi proposto por fator de sessão existente. Selection não concede
permissão de tool, filesystem, mutação, sessão ou execução.

## 15. Regressão

```text
tests/test_explicit_agent_binding.py
tests/test_codex_session_affinity.py
tests/test_autonomy_foundation.py
tests/test_task_requirement_grounding.py
tests/test_project_intelligence_v2.py
tests/test_local_model_runtime.py
tests/test_local_model_contract_eval.py
tests/test_local_model_reasoning_eval.py
tests/test_local_model_performance_eval.py
tests/test_runtime_startup.py
-> 150 passed
```

Suite completa: `913 passed, 1 skipped`. Zero regressões.

## 16. A/B

Não havia seleção automática real; baseline A é `NO_SELECTION`
(`propose_agent_selection` da fundação).

| Métrica | A: NO_SELECTION | B: Selection Provenance |
|---|---:|---:|
| propostas em casos multiagente | 0 | 17 |
| propostas totais (fora de explicit) | 3 | 23 |
| decisões multiagente resolvidas | 0 | 8 semânticas + 9 determinísticas |
| provenance estruturada | parcial | 100% |
| seleção inelegível/indisponível | 0 | 0 |

## 17. Composição ao vivo (grounding real → seleção)

Sete tarefas passaram pelo pipeline completo com requisitos extraídos ao vivo:

```text
"analise a causa do bug sem modificar nada"          mutation FALSE -> DETERMINISTIC local
"analise a causa do bug e implemente a correcao"     mutation TRUE  -> SEMANTIC codex
"revise a arquitetura"                               mutation FALSE -> DETERMINISTIC local
"revise a arquitetura e aplique as mudancas"         mutation TRUE  -> SEMANTIC local
"descubra por que os testes falham"                  mutation FALSE -> DETERMINISTIC local
"descubra por que os testes falham e corrija"        mutation TRUE  -> SEMANTIC codex
"use DeepSeek para analisar isso"                    requested      -> EXPLICIT_USER deepseek
```

Cada par contrastivo mudou de caminho por causa do requisito de mutação, com
uma chamada de grounding e no máximo uma de seleção, sempre com
`execution_authorized=false`.

## 18. Arquivos

| Arquivo | Linhas | Conteúdo |
|---|---:|---|
| `tern/orchestrator/agent_selection.py` | 1001 | fatores, perfis de seleção, seletor semântico constrained, motor de precedência, proposta com provenance |
| `tern/orchestrator/agent_selection_eval.py` | 639 | corpus loader, métricas, taxonomia de erro, replay, A/B, CLI dry-run |
| `tests/test_agent_selection.py` | 666 | 33 testes |
| `tests/data/agent_selection_diagnostic.jsonl` | 31 | corpus diagnóstico novo |

Nenhum arquivo das camadas aprovadas foi modificado.

## 19. Testes

33 testes cobrindo: explicit Codex, explicit DeepSeek, explicit indisponível
(Codex e DeepSeek), explicit incapaz, single eligible Codex, single eligible
Local, nenhum elegível, dois elegíveis, três elegíveis, somente um disponível,
nenhum disponível, equal-fit ambíguo, agente fora do conjunto permitido
rejeitado sem substituição, falha de parse do modelo, provenance obrigatória,
fatores e evidências, capability que não vira preferência, `UNRESOLVED`,
caminho determinístico com zero chamadas, caminho semântico com uma chamada,
dry-run sem jobs, sem delegação, sem mutação e sem resolução de sessão,
integridade do corpus e execução completa do evaluator.

## Decisões

```text
APROVADO — AUTOMATIC AGENT SELECTION DRY-RUN
```

```text
APROVADO — SELECTION PROVENANCE
```

```text
READY FOR CONTROLLED LIVE AGENT SELECTION
```

Autonomia geral permanece desligada. Ressalva a observar na próxima etapa: a
atribuição de fator citada pelo modelo errou em 2/8 casos e fatos quase
idênticos (AS-002 vs AS-009) produziram escolhas diferentes; ambos os efeitos
são absorvidos hoje pelo recálculo determinístico de fatores e pelo conjunto de
aceitáveis, mas devem ser medidos de novo quando a seleção passar a escolher o
executor de verdade.
