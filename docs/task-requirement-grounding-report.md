# Task Requirement Grounding + Evidence Provenance — relatório

Data: 2026-08-14

Branch avaliada: `feat/task-requirement-grounding`

Modelo: Qwen local atual; Bonsai não foi baixado nem executado.

Escopo: evaluator, grounding e eligibility em dry-run. Nenhuma seleção, delegação ou mutação automática foi ativada.

## Decisão executiva

```text
APROVADO — TASK REQUIREMENT GROUNDING + EVIDENCE PROVENANCE

READY FOR PROJECT INTELLIGENCE V2
```

O candidato passou os gates sugeridos no corpus diagnóstico de 17 casos: macro-F1 de 90,20%, recall crítico de 100%, retenção de constraints explícitas de 100%, preservação de evidência explícita e determinística de 100% e zero autorização indevida de mutação. A precisão e o recall de eligibility foram ambos 100%.

A limitação principal remanescente é sobreconfiança não autorizativa: em 21,30% dos labels cujo valor correto era `UNKNOWN`, o Qwen respondeu `TRUE` ou `FALSE`. Os 36 erros finais são exclusivamente `UNKNOWN_SHOULD_HAVE_BEEN_USED`; não há requisito conhecido perdido nem evidência forte sobrescrita. O corpus é pequeno e a exatidão de inferência semântica tem suporte de apenas um label, portanto os percentuais não devem ser interpretados como intervalo estatístico estreito.

## 1. Arquitetura anterior e auditoria da fase 0

Antes deste experimento, `TaskRequirements` era uma estrutura definitiva em [autonomy_foundation.py](../tern/orchestrator/autonomy_foundation.py):

- `capabilities`: conjunto positivo; ausência equivalia implicitamente a `FALSE`;
- `mutation_required`, `read_only_required` e `ambiguity_material`: booleanos;
- `target_scope`, `risk_level`, `expected_files`, `forbidden_files` e `tests_requested`: metadados definitivos;
- todos os campos eram produzidos pela mesma chamada do Qwen em `TaskRequirementAnalyzer`;
- não havia `UNKNOWN`, `CONFLICT`, evidência por campo nem precedência de fontes;
- `EligibilityEngine` eliminava agentes somente pelas capabilities presentes e pelas permissões; não sabia diferenciar ausência de evidência de evidência negativa.

Mapa auditado:

| Componente | Localização atual | Papel anterior |
|---|---:|---|
| `AgentCapabilityProfile` | `autonomy_foundation.py:57` | capabilities e evidências reais do executor |
| `TaskRequirements` | `autonomy_foundation.py:298` | requisitos booleanos/positivos sem provenance |
| `TaskRequirementAnalyzer` | `autonomy_foundation.py:390` | uma chamada Qwen produzindo todos os campos |
| `EligibilityEngine` | `autonomy_foundation.py:493` | interseção determinística de requisitos, capabilities e permissões |
| `VerificationResult` | `autonomy_foundation.py:637` | fundação de verificação objetiva |
| `ProjectSnapshot` | `project_intelligence.py:111` | estado estruturado do projeto |
| `ProjectSnapshotBuilder` | `project_intelligence.py:221` | snapshot/repo map incremental |
| `diagnostic_baseline` | `autonomy_eval.py:80` | perfis e availability de verdade conhecida |

`AgentCapabilityProfile` já possuía evidência própria e availability separada. `ProjectSnapshot` já fornecia fatos determinísticos. A lacuna era exclusivamente a proveniência de cada requisito inferido.

## 2. Breakdown de erro do baseline

O valor histórico de 36,67% veio do corpus anterior. O breakdown reconstruído desse artefato mostrou os maiores falsos negativos em `repository_read` (6), `repository_write` (6), `code_analysis` (4), `code_edit` (4), `test_execution` (2), `general_reasoning` (3), `mutation_capable` (6) e `mutation_required` (5). Também havia falsos positivos de `filesystem_read` (4), `test_execution` (2) e `read_only_required` (3).

Para o A/B controlado deste experimento, o baseline foi reexecutado no novo corpus com o mesmo Qwen e contrato antigo. Ele produziu 306 divergências de valor ou provenance:

| Classe | Ocorrências |
|---|---:|
| `UNKNOWN_SHOULD_HAVE_BEEN_USED` | 201 |
| `FALSE_NOT_REQUIRED` | 38 |
| `EXPLICIT_EVIDENCE_IGNORED` | 29 |
| `MISSING_REQUIREMENT` | 23 |
| `OVERCONFIDENT_INFERENCE` | 10 |
| `FALSE_REQUIRED` | 4 |
| `PROJECT_FACT_IGNORED` | 1 |

Como o baseline não tinha provenance, mesmo acertos de valor não preservavam a origem esperada.

## 3. Error taxonomy

O evaluator registra, para toda divergência:

```text
case_id
user_request (somente fixture de avaliação)
requirement
expected / actual
expected_source / actual_source
error_class
source_stage
```

As classes implementadas são `FALSE_REQUIRED`, `FALSE_NOT_REQUIRED`, `MISSING_REQUIREMENT`, `OVERCONFIDENT_INFERENCE`, `EXPLICIT_EVIDENCE_IGNORED`, `PROJECT_FACT_IGNORED`, `CONSTRAINT_IGNORED`, `CONTRADICTION` e `UNKNOWN_SHOULD_HAVE_BEEN_USED`.

No candidato final restaram 36 erros, todos `UNKNOWN_SHOULD_HAVE_BEEN_USED`: `code_review` 12, `test_execution` 5, `code_analysis` 4, `web_access` 4, `filesystem_read` 4, `general_reasoning` 4, `read_only_capable` 2 e `filesystem_write` 1. Não houve erro de constraint, project fact, requisito positivo conhecido ou contradição silenciosa.

## 4. Tri-state

Todas as 16 capabilities reais e os campos `mutation_required` e `read_only_required` passaram a ser representados por `TRUE`, `FALSE`, `UNKNOWN` ou `CONFLICT` no caminho novo. Isso não converteu os contratos antigos; `GroundedTaskRequirements` é uma camada paralela e sua adaptação para `TaskRequirements` inclui somente capabilities `TRUE`.

- `TRUE`: existe fato ou inferência suficiente de que a capability é necessária.
- `FALSE`: existe evidência negativa ou implicação determinística de que não faz parte da tarefa.
- `UNKNOWN`: não existe evidência suficiente; nunca é tratado como permissão.
- `CONFLICT`: evidências de mesma força se contradizem; eligibility pode ser calculada, mas execução fica bloqueada.

`long_running_job`, `persistent_session`, `semantic_interpretation` e `task_requirement_extraction` não são enviados à inferência semântica livre. Sem fato estruturado, permanecem `UNKNOWN`. Capabilities de acesso e mutação inferidas como positivas também passam por gate de escopo antes do merge.

## 5. Evidence model

`GroundedRequirement` contém `name`, `value`, `source`, `evidence_refs` e `safety_class`. Não foi adicionado float de confidence.

Fontes:

| Fonte | Uso |
|---|---|
| `EXPLICIT_USER` | pedido positivo inequívoco |
| `EXPLICIT_USER_NEGATION` | proibição/constraint explícita |
| `REQUESTED_AGENT` | binding explícito preservado separadamente |
| `PROJECT_FACT` | fato verificável do `ProjectSnapshot` |
| `RUNTIME_FACT` | fato verificável do runtime, disponível para integrações futuras |
| `TASK_IMPLICATION` | consequência determinística de um fato estruturado |
| `SEMANTIC_INFERENCE` | resolução do Qwen para dimensão ainda aberta |
| `INSUFFICIENT_EVIDENCE` | ausência de suporte suficiente |
| `CONFLICTING_EVIDENCE` | conflito de evidências fortes |

Produção não persiste o prompt integral como evidência. São guardados refs categóricos como `constraint:forbid_web`, `project:target_path_exists`, `model:resolved_dimension` e hash curto do pedido. O texto integral aparece somente nas fixtures e na taxonomia offline.

## 6. Precedência

| Prioridade | Fontes |
|---:|---|
| 5 | `CONFLICTING_EVIDENCE` |
| 4 | `EXPLICIT_USER`, `EXPLICIT_USER_NEGATION`, `REQUESTED_AGENT` |
| 3 | `PROJECT_FACT`, `RUNTIME_FACT` |
| 2 | `TASK_IMPLICATION` |
| 1 | `SEMANTIC_INFERENCE` |
| 0 | `INSUFFICIENT_EVIDENCE` |

Uma fonte mais fraca nunca substitui uma mais forte. A origem aceita também não é rebaixada durante o merge.

## 7. Merge algorithm e grounding pipeline

O novo fluxo é:

1. reutilizar `IntentFrame`, constraints e Explicit Agent Binding já aprovados;
2. coletar fatos explícitos conservadores e refs sem copiar o prompt;
3. incorporar fatos confiáveis do `ProjectSnapshot`;
4. semear requirements conhecidos e marcar o restante como `UNKNOWN`;
5. usar a mesma chamada semântica atual apenas para dimensões semanticamente abertas;
6. rejeitar respostas duplicadas, fora do schema, fora do conjunto unresolved ou incompatíveis com o escopo;
7. mesclar por precedência, validar conflitos e produzir `GroundedTaskRequirements`;
8. calcular eligibility somente com `TRUE`, mantendo autorização e execution safety separadas.

Não foi adicionada segunda inferência: `extra_inference_count = 0`. A extração explícita não seleciona agentes e suas regras conservadoras não são usadas como keyword router.

## 8. Contradictions

Evidências opostas de mesma força geram `CONFLICT`, combinam os refs e recebem `CONFLICTING_EVIDENCE`. O Qwen não pode desfazer esse estado. `GroundedEligibilityEngine` emite reason code estruturado `EXECUTION_BLOCKED_CONFLICT_<DIMENSION>` e marca `execution_safe=false`.

O caso diagnóstico “corrija, mas sem modificar nada” preserva o conflito em `mutation_required`; nenhuma capacidade positiva de escrita é semeada e nenhuma execução é autorizada.

## 9. A/B

| Métrica | A — `TaskRequirements` | B — Grounded | Variação |
|---|---:|---:|---:|
| macro precision | 63,06% | 86,68% | +23,62 pp |
| macro recall | 36,92% | 100,00% | +63,08 pp |
| macro F1 | 35,88% | 90,20% | +54,32 pp |
| accuracy tri-state + fonte | 26,80% | 88,24% | +61,44 pp |
| critical requirement recall | 25,00% | 100,00% | +75,00 pp |
| critical false-positive rate | 41,67% | 3,45% | -38,22 pp |
| explicit evidence preservation | 0,00% | 100,00% | +100,00 pp |
| deterministic fact preservation | 0,00% | 100,00% | +100,00 pp |
| unknown recall | 0,00% | 78,70% | +78,70 pp |
| overconfidence rate | 100,00% | 21,30% | -78,70 pp |
| false mutation authorization | 0 | 0 | igual |

O false positive crítico remanescente é uma inferência de `filesystem_write` no caso que explicitamente solicita correção e testes; a própria mutação já estava explicitamente autorizada. Ele não produziu autorização nova nem agente falsamente elegível.

## 10. Métricas de provenance e constraints

| Métrica B | Resultado |
|---|---:|
| provenance accuracy | 88,24% |
| explicit evidence preservation | 100,00% |
| deterministic fact preservation | 100,00% |
| semantic inference accuracy | 100,00% (suporte = 1) |
| unknown precision | 100,00% |
| unknown recall | 78,70% |
| explicit constraint retention | 100,00% |
| explicit prohibition retention | 100,00% |
| ambiguity accuracy | 94,12% |
| JSON validity | 100,00% |
| retry rate | 0,00% |

## 11. Métricas completas por campo

`P/R/F1` medem a classe `TRUE`; `FP/FN` também se referem a `TRUE`. Campos sem label positivo podem ter recall neutro e F1 zero se houver falso positivo.

| Campo | A P | A R | A F1 | A FP | A FN | B P | B R | B F1 | B FP | B FN | B unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `repository_read` | 0,00% | 0,00% | 0,00% | 1 | 9 | 100,00% | 100,00% | 100,00% | 0 | 0 | 47,06% |
| `repository_write` | 0,00% | 0,00% | 0,00% | 1 | 6 | 100,00% | 100,00% | 100,00% | 0 | 0 | 5,88% |
| `filesystem_read` | 0,00% | 100,00% | 0,00% | 5 | 0 | 0,00% | 100,00% | 0,00% | 4 | 0 | 76,47% |
| `filesystem_write` | 0,00% | 100,00% | 0,00% | 2 | 0 | 0,00% | 100,00% | 0,00% | 1 | 0 | 35,29% |
| `code_analysis` | 80,00% | 36,36% | 50,00% | 1 | 7 | 73,33% | 100,00% | 84,62% | 4 | 0 | 11,76% |
| `code_edit` | 100,00% | 50,00% | 66,67% | 0 | 3 | 100,00% | 100,00% | 100,00% | 0 | 0 | 5,88% |
| `test_execution` | 50,00% | 100,00% | 66,67% | 2 | 0 | 100,00% | 100,00% | 100,00% | 0 | 0 | 52,94% |
| `long_running_job` | 100,00% | 100,00% | 100,00% | 0 | 0 | 100,00% | 100,00% | 100,00% | 0 | 0 | 100,00% |
| `persistent_session` | 100,00% | 100,00% | 100,00% | 0 | 0 | 100,00% | 100,00% | 100,00% | 0 | 0 | 100,00% |
| `general_reasoning` | 100,00% | 0,00% | 0,00% | 0 | 5 | 69,23% | 100,00% | 81,82% | 4 | 0 | 23,53% |
| `code_review` | 0,00% | 0,00% | 0,00% | 1 | 2 | 14,29% | 100,00% | 25,00% | 12 | 0 | 17,65% |
| `web_access` | 100,00% | 50,00% | 66,67% | 0 | 1 | 100,00% | 100,00% | 100,00% | 0 | 0 | 58,82% |
| `mutation_capable` | 100,00% | 0,00% | 0,00% | 0 | 6 | 100,00% | 100,00% | 100,00% | 0 | 0 | 5,88% |
| `read_only_capable` | 60,00% | 90,00% | 72,00% | 6 | 1 | 83,33% | 100,00% | 90,91% | 2 | 0 | 29,41% |
| `semantic_interpretation` | 100,00% | 100,00% | 100,00% | 0 | 0 | 100,00% | 100,00% | 100,00% | 0 | 0 | 100,00% |
| `task_requirement_extraction` | 100,00% | 100,00% | 100,00% | 0 | 0 | 100,00% | 100,00% | 100,00% | 0 | 0 | 100,00% |
| `mutation_required` | 100,00% | 16,67% | 28,57% | 0 | 5 | 100,00% | 100,00% | 100,00% | 0 | 0 | 5,88% |
| `read_only_required` | 66,67% | 100,00% | 80,00% | 5 | 0 | 100,00% | 100,00% | 100,00% | 0 | 0 | 5,88% |

## 12. Eligibility impact

| Métrica | A | B |
|---|---:|---:|
| precision | 84,62% | 100,00% |
| recall | 91,67% | 100,00% |
| false eligible agent | 6 | 0 |
| missed eligible agent | 3 | 0 |
| single candidate correct | 50,00% | 100,00% |
| no candidate correct | 100,00% | 100,00% |
| multi candidate correct | 57,14% | 100,00% |

Política implementada:

- requisito `TRUE`: capability obrigatória;
- requisito `FALSE`: não cria necessidade;
- requisito `UNKNOWN`: não elimina agente;
- `UNKNOWN` ou `CONFLICT` crítico: não concede permissão e pode bloquear `executable_now`;
- eligibility nunca altera permission profile ou availability.

## 13. Casos multiagente

Os seis casos ambíguos históricos (`AF-001`, `AF-002`, `AF-003`, `AF-005`, `AF-006`, `AF-010`) foram reexecutados ao vivo:

```text
continuam genuinamente multiagente: 6
eram multi por requirement incompleto: 0
viraram single-candidate: 0
viraram no-candidate: 0
conjunto exato de elegíveis: 6/6
```

`AF-008` também possui múltiplos agentes capazes, mas tem binding explícito e por isso não faz parte dos seis casos que exigiriam desempate automático; ele também preservou exatamente os três elegíveis. Nenhum ranking ou desempate foi criado.

## 14. Mutation safety

```text
UNKNOWN != permission
```

`mutation_authorized_by_requirements` só retorna verdadeiro para `mutation_required=TRUE`. `UNKNOWN`, `FALSE` e `CONFLICT` não autorizam. Escrita inferida fora de escopo é descartada, conflito bloqueia execution safety e os gates existentes continuam obrigatórios. Resultado do corpus: zero autorização indevida de mutação e zero execução iniciada pelo evaluator.

## 15. Explicit Agent Binding

Binding explícito continua fora da seleção automática e é transportado como `requested_agent`, `REQUESTED_AGENT` e evidence ref. A precedência permanece:

```text
requested_agent > qualquer proposal automática
```

Resultado direcionado: suite passou integralmente; preservação no corpus grounding = 100%. Nenhum fallback Codex/DeepSeek foi implementado.

## 16. Codex Session Affinity

O novo caminho não modifica resolver, registry, persistência ou criação de sessão Codex. A suite completa de affinity passou: reuse, ambiguidade sem chute, bloqueio cross-project, registro/recuperação/visibilidade, restart recovery, concorrência máxima de uma sessão e zero ghost sessions permanecem preservados.

## 17. Custo

| Medida live, 17 casos | A | B | Variação |
|---|---:|---:|---:|
| inference attempts | 28 | 17 | -39,29% |
| extra inference count | 0 | 0 | igual |
| prompt tokens | 24.074 | 33.036 | +37,23% |
| completion tokens | 5.411 | 1.985 | -63,32% |
| retry rate | 64,71% | 0,00% | -64,71 pp |
| latency p50 | 66.045 ms | 19.980 ms | -69,75% |
| latency p90 | 69.202 ms | 24.023 ms | -65,28% |
| latency p95 | 79.038 ms | 24.334 ms | -69,21% |

O prompt cresceu porque carrega fatos/provenance e snapshot compacto; a resposta ficou menor porque o modelo resolve somente dimensões abertas. Os números não isolam warm-up ou variância do servidor e o corpus tem 17 amostras.

## 18. Arquivos modificados

| Arquivo | Conteúdo |
|---|---|
| `tern/orchestrator/task_requirement_grounding.py` | tri-state, evidence, extração, merge, analyzer e eligibility segura |
| `tern/orchestrator/grounding_eval.py` | corpus loader, A/B, métricas por campo, provenance e taxonomia |
| `tests/data/task_requirement_grounding_diagnostic.jsonl` | 17 fixtures independentes; v2/v3 históricos não alterados |
| `tests/test_task_requirement_grounding.py` | 26 testes unitários/integrados |
| `docs/task-requirement-grounding-report.md` | este relatório |

## 19. Validação

Resultados finais:

```text
testes direcionados: 90 passed
suíte completa: 834 passed, 1 skipped, 1 warning
py_compile: passed
git diff --check: passed
```

O corpus histórico não foi reescrito, não foi criada v4 e os artefatos live ficam em `.orchestrator/`, fora do versionamento. A implementação não foi conectada ao fluxo de execução ativo: é exclusivamente dry-run/evaluator.

## 20. Próximo gargalo e futuro Bonsai

Escolha baseada nos resultados:

```text
PROJECT INTELLIGENCE V2
```

O grounding passou os gates, enquanto o baseline anterior de Project Intelligence permanece em 50% de relevant-file recall e 34,17% de seleção irrelevante. Melhorar repo map/seleção de arquivos é agora o próximo gargalo mensurável; não foi implementado neste trabalho.

O Bonsai não foi abandonado e também não foi baixado neste experimento. Um reteste futuro deverá usar, sem relaxamento de schema ou validators, as métricas: requirement macro-F1, critical recall, overconfidence, semantic validity, JSON validity, eligibility accuracy, latency p50/p90/p95, tokens, RAM/VRAM e disco. A troca de runtime/template só pode ser considerada explicitamente e sem reduzir o contrato `json_schema`. O Qwen continua sendo o modelo local de produção e controle até uma aprovação separada.

## Guardrails confirmados

- nenhuma seleção automática;
- nenhuma delegação automática;
- nenhuma mutação automática;
- nenhuma segunda/terceira inferência;
- nenhum ranking de agentes;
- nenhum fallback Codex ↔ DeepSeek;
- nenhum loop autônomo ou replan;
- nenhuma alteração nos gates de tool, filesystem, web, availability ou sessão;
- nenhum experimento rejeitado foi reativado.
