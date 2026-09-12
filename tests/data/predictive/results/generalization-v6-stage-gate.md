# Predictive Generalization & Robustness v6 — stage gate

## Metric audit

O v5 mostrava `root_cause_validity=31.2%` e `top1_validity=72.7%`
porque as métricas usavam populações diferentes. A primeira era `5/16`: todos
os 16 casos positivos, inclusive cinco false abstentions. A segunda era `8/11`:
somente os 11 casos que chegaram a uma recomendação. No v6, toda métrica
registra numerador, denominador e população elegível.

| Métrica | Numerador | Denominador |
|---|---|---|
| root/repair validity | casos positivos estruturalmente válidos | todos os casos avaliáveis que exigem diagnóstico |
| top-1 validity | recomendações estruturalmente válidas | casos positivos recomendados |
| recommendation precision | recomendações válidas | recomendações emitidas para casos positivos |
| recommendation coverage | casos positivos recomendados | todos os casos positivos |
| false abstention | casos positivos com abstention | todos os casos positivos |
| invariance | variantes que preservam a decisão estrutural da base | variantes às quais a expectativa se aplica |
| counterfactual sensitivity | pares que mudam a decisão exigida | pares cuja mudança semântica exige essa decisão |

Invariância não implica correção. O post-processing final adiciona validade
absoluta das variantes ao gate. Sem isso, uma base errada repetida de forma
estável poderia aprovar o benchmark.

## Generalization diagnosis

- Atalhos lexicais: a formulação equivalente do problema falhou em 2/12 casos
  development e 2/5 casos holdout. Renomes, docstrings e distractors também
  expuseram atração lexical: no holdout, somente 7/16 variantes positivas
  selecionaram uma root cause válida.
- Viés de ordem: 0 falhas em 2 casos development e 2 holdout para ordem de
  candidatos, EvidenceAtoms e IDs opacos.
- Cobertura estrutural: nas variantes development houve 5
  `MISSED_STRUCTURAL_NEIGHBOR` e 3 `CAUSAL_SLICE_ERROR`.
- Reasoner indisponível/JSON inválido: 0 casos live. Os erros residuais são de
  seleção, ranking ou perda de cobertura, não outage do modelo.
- Abstention: 0/41 falsos no development canônico e 0/61 nas variantes
  development; no holdout houve 6/16 false abstentions.

## Ablations

| Regra | Estado | Root | Repair pair | Top-1 | Decisão |
|---|---:|---:|---:|---:|---|
| Exigir trigger textual antes de comparar roots de binding | ON | 90.2% | 90.2% | 87.8% | manter como prior fraco |
| Exigir trigger textual antes de comparar roots de binding | OFF | 80.5% | 85.4% | 78.0% | rejeitada; ativava comparação em fluxos não relacionados |
| Palavras boundary/validation restringem schema | ON | não promovido | não promovido | não promovido | somente switch de ablation |
| Palavras boundary/validation restringem schema | OFF | 90.2% | 90.2% | 87.8% | default; texto não elimina reparos estruturais |

Nenhuma heurística nova foi promovida após observar o holdout v6.

## Metamorphic results

| Métrica | Development | Holdout v6 |
|---|---:|---:|
| root cause invariance | 91.4% (64/70) | 90.0% (18/20) |
| repair invariance | 92.9% (65/70) | 95.0% (19/20) |
| target invariance | 95.7% (67/70) | 95.0% (19/20) |
| candidate order invariance | 100% (2/2) | 100% (2/2) |
| evidence order invariance | 100% (2/2) | 100% (2/2) |
| identifier invariance | 100% (2/2) | 100% (2/2) |
| problem wording invariance | 83.3% (10/12) | 60.0% (3/5) |

Validade absoluta das variantes:

| Métrica | Development | Holdout v6 |
|---|---:|---:|
| root cause validity | 83.6% (51/61) | 43.8% (7/16) |
| repair strategy validity | 90.2% (55/61) | 56.2% (9/16) |
| repair target validity | 91.8% (56/61) | 56.2% (9/16) |
| repair pair validity | 90.2% (55/61) | 50.0% (8/16) |
| top-1 validity | 82.0% (50/61) | 80.0% (8/10) |
| recommendation precision | 82.0% (50/61) | 80.0% (8/10) |
| recommendation coverage | 100% (61/61) | 62.5% (10/16) |
| false abstention | 0% (0/61) | 37.5% (6/16) |

## Counterfactual results

| Métrica | Development | Holdout v6 |
|---|---:|---:|
| root sensitivity | 90.0% (9/10) | 80.0% (4/5) |
| repair sensitivity | 90.0% (9/10) | 80.0% (4/5) |
| target sensitivity | 90.0% (9/10) | 80.0% (4/5) |

Quatro pares development tinham uma expectativa antiga inválida: causas
diferentes ainda exigiam a mesma família `CORRECT_IMPORT` ou
`CORRECT_RETURN_VALUE`. A readjudicação mudou somente a expectativa estrutural,
sem rerun do engine. A falha real restante é `CV6D-008`, que mantém
`RETURN_CONTRACT` quando o contrafactual demonstra `ARGUMENT_BINDING`.

## Development v6

O corpus canônico continua forte: root 90.2% (37/41), repair pair 90.2%
(37/41), top-1 e recommendation precision 87.8% (36/41), coverage 100%
(41/41), false abstention 0%. Grounding permaneceu em 100% de referências
válidas e 0% de claims unsupported. Latência média canônica: 27.6 s; robustez:
29.0 s/request.

O development v6 não passa quando variantes entram na população de qualidade:
root 83.6%, top-1 82.0%, recommendation precision 82.0% e wording invariance
83.3%, abaixo dos gates de 85%, 85%, 85% e 90%.

## Holdout v6

Split congelado com 20 variantes, cinco pares contrafactuais e SHA-256
`cba2c3ed22e3f94c2e6b24e7b77057058a3b15b3aa59c286b252d1fac72db0e8`.
A execução live completa ocorreu uma vez, com 24.3 s/request. Um lançamento
anterior abortou durante materialização de `FILE_RENAME` antes de gerar
artefato; nenhuma previsão desse lançamento foi inspecionada. A correção foi
somente no harness (fallback determinístico de path) e todas as 20 variantes
foram validadas offline antes da execução completa.

O holdout não passa: root validity 43.8%, repair pair 50.0%, coverage 62.5%,
false abstention 37.5% e wording invariance 60.0%. Há seis
`FALSE_ABSTENTION`, três `ROOT_CAUSE_SELECTION_ERROR`, dois `RANKING_ERROR`,
dois `FAILED_TO_ABSTAIN` e um `REPAIR_TARGET_GRANULARITY_ERROR`.

## Coverage funnel

Development canônico, todos os 57 casos: 53 recuperaram contexto suficiente,
53 produziram roots, 53 selecionaram root, 53 produziram candidato e 49
recomendaram. As oito abstentions incluem quatro perdas em retrieval e quatro
em causal slice; entre os 41 casos positivos, nenhuma foi falsa.

Variantes development: 61 positivos, 61 recomendações, 0 false abstentions.
Variantes holdout: 16 positivos, 10 recomendações e 6 false abstentions. A
queda principal acontece antes da recomendação, após perturbações superficiais
em docstrings, wording, símbolos ou distractors.

## Stability and latency

Uma amostra de duas paráfrases executada três vezes obteve 100% de estabilidade
estruturada e de decisão (24 chamadas; 22.2 s/request). O resultado demonstra
estabilidade de runtime, mas não corrige a sensibilidade semântica ao wording.
Todas as execuções ficaram abaixo do gate de 45 s/request.

## Safety and runtime readiness

| Invariante | Valor |
|---|---:|
| unsupported claims | 0 |
| evidence reference validity | 100% |
| forbidden recommendations | 0 |
| filesystem mutations | 0 |
| tool dispatches | 0 |
| execution authorized | 0 |
| authority grants | 0 |
| destructive actions | 0 |

`predictive-sandbox-check` retornou provider `unavailable`: Docker executable
unavailable. Confinamento de escrita, rede, ambiente, processos, recursos e
cleanup permanecem todos não verificados. O provider falha fechado; nenhum
código de projeto foi executado no host.

## Tests

- Predictive: 117 passed, 1235 deselected.
- Suíte completa: 1352 passed, 1 skipped, 1 warning em 93.85 s.
- `git diff --check`: sem erros; somente avisos locais de conversão LF/CRLF.

## Final gate

`PREDICTIVE_QUALITY_READY = false`

`SIMULATION_RUNTIME_READY = false`

Os bloqueadores são generalização absoluta de root cause, coverage/false
abstention no holdout e invariância à formulação do problema. Candidate
Simulation não foi implementada.
