# JARVIS v15 — Behavioral Contracts & Investigable Hypotheses

## CONTRACT REPRESENTATION AUDIT

Baseline: `76a15a4c8aeba3385a8d001903037defd0cbf99f`. A auditoria anterior ao hardening esta em `v15-contract-audit-before.md`; a matriz congelada em `v15-contract-families-before.json`.

O v14 representava chamadas e efeitos, mas nao demonstrava que executaram ou violaram o comportamento esperado. Retornos/condicoes forneciam anchors ao slice; CALLABLE e BEHAVIORAL_EFFECT nao eram, por si, origens de defeitos. O servico encerrava antes do Qwen quando nao havia roots. `expected_behavior` nao transportava automaticamente a expectativa do relato. Timers anteriores nao preservavam as fases de workers encerrados.

A matriz inicial tem 12 familias controladas; 8/12 produzem algum candidato legado. Isso nao significa 8 diagnosticos corretos: candidatos existentes podem ser manifestacoes genericas. Nao houve justificativa para alterar ranking, budgets de recovery ou grounding.

`BehavioralContractContext` separa CONTRACT_EXISTS, CONTRACT_APPLIES e violacao sustentada. As origens tipadas incluem relato, anotacao, assertion, guard, contrato explicito e desconhecido. A implementacao inicial materializa expectativa explicitamente marcada `Expected:` no relato, anotacoes primitivas e relacoes AST limitadas; nao extrai genericamente contratos de testes, guards ou documentacao textual.

Somente um retorno literal, incondicional e unico, em callable simples com anotacao primitiva nao sombreada, pode demonstrar incompatibilidade estatica. Decorators, async, retornos condicionais e valores dinamicos permanecem incertos. Anotacao existente nao e violacao. Incompatibilidade estatica nao demonstra execucao nem causalidade universal.

## EFFECT-TO-OBSERVATION MODEL

O grafo v14 foi reutilizado sem novo indexador. Links preservam IDs de entrada, efeito, relacoes de implementacao e atoms reais. A ligacao ao relato permanece INFERRED_FROM_OBSERVATION, com obrigacao VERIFY_EXECUTION_AND_OBSERVATION; nao e marcada como responsabilidade causal demonstrada.

Duas relacoes limitadas sao AST_DEMONSTRATED: retorno de helper efetivamente consumido pelo retorno da API e condicao AST que governa um efeito. Chamar um helper e descartar seu retorno nao conta como consumo. Coexistencia no arquivo, comentario e docstring nao provam relacao. Retorno, estado e yield podem apontar locais investigaveis, mas a existencia do efeito nao prova defeito.

Na matriz final, 10/12 familias possuem locais investigaveis e 1/12 possui violacao estatica do subconjunto suportado. Esses numeros medem capacidade de representacao local, nao accuracy externa ou execucao demonstrada. `v15-contract-families-final.json` preserva os fatos por familia.

## INVESTIGABLE HYPOTHESIS ARCHITECTURE

Nova entidade `InvestigableCausalHypothesis`, distinta de Hypothesis diagnostica, RootCauseCandidate e SolutionCandidate. Armazena entidade real, local, atoms exatos, relacoes confirmadas/inferidas, requisitos adicionais e limitacoes. Status: INVESTIGABLE, INSUFFICIENT_EVIDENCE, CONTRADICTED e SUPPORTED_FOR_ROOT_CANDIDACY; o modelo nao escolhe arbitrariamente esses status.

O caminho separado opera somente quando nao existem roots, mas existe uma API/implementacao ancorada com evidencias. No maximo seis alternativas e tres selecionadas; uma chamada Qwen, schema compacto, IDs e reason codes fechados. Python mantem todos os requisitos faltantes; o modelo nao pode apaga-los, inventar evidencia, pontuar, recomendar reparo ou criar patch. Sem anchor: abstention completa, zero chamadas.

O DecisionReport adiciona campos versionados de contratos e investigacao, mantendo campos anteriores. A validacao impede misturar investigacao com diagnosticos/candidatos/recomendacao. CLI explicita investigacao consultiva e requisitos de evidencia. `execution_authorized=false`, `requires_approval=true`.

Limite importante: roots legados nao vazios continuam no caminho diagnostico existente, mesmo quando seu significado esta errado. O v15 nao reescreve esse seletor; portanto a nova investigacao nao resolve automaticamente falsas causas ja geradas.

## HYPOTHESIS PROMOTION VALIDATION

O validador deterministico exige entidade existente, evidencia compativel, return site identificado, relacao completa demonstrada no slice e ausencia de contradicao. A implementacao inicial valida somente o subconjunto estatico de RETURN_CONTRACT e a correspondencia com um candidato realmente representado. Nao fabrica root nem realiza promocao automatica. Confianca do Qwen nao participa.

Expectativa de usuario, anotacao sem mismatch, reachability e efeito existente nao bastam. Os testes positivos e negativos verificam a aprovacao; em todas as execucoes live houve zero promocoes automaticas. A taxa externa de promocao incorreta e 0/0, undefined, nao uma estimativa de seguranca perfeita.

Correcao independente pos-seal: `bool` satisfaz `int`/`float` no subconjunto numerico suportado. Dois controles negativos detectaram uma falsa incompatibilidade. Foi corrigida depois do holdout, sem reutilizar falhas ou repetir sua execucao. Os hashes do runtime avaliado e atual sao preservados separadamente; a qualificacao externa do runtime atual foi explicitamente invalidada. Nenhum threshold foi alterado.

## CLASS API RESOLUTION

Reutilizada a resolucao v14 de metodo conhecido para `self._helper`, retorno encaminhado e assignments simples quando ancorados. Atribuicao de atributo e evidenciada, mas nao prova estado defeituoso. Dispatch dinamico, heranca ambigua e receiver por instancia estaticamente conhecida ainda podem ficar UNRESOLVED_DISPATCH; nao foi introduzida uma resolucao baseada apenas no nome citado no problema. Negativos cobrem receiver ambiguo e instancia nao resolvida. Essa limitacao permanece aberta.

## DEVELOPMENT BEFORE/AFTER

Seis defeitos independentes novos, tres repositorios: Flask, Sphinx e pytest. Uma formulacao original por defeito; development=3 e holdout=3. Corpus congelado antes do hardening, usando commits pre-correcao e adjudicacao upstream, sem fornecer gold diff ao reasoner e sem executar projetos externos. Apenas snapshots Python foram analisados, o que limita cobertura de assets/configuracoes nao Python.

| Metrica / populacao | v14 congelado, dev | v15 avaliado, dev | Holdout one-shot |
|---|---:|---:|---:|
| Casos concluidos / tentados | 2/3 | 2/3 | 2/3 |
| Arquivo correto / concluidos | 1/2 | 1/2 | 0/2 |
| Simbolo correto no ledger / concluidos | 1/2 | 1/2 | 0/2 |
| Candidato correto / concluidos | 1/2 | 1/2 | 0/2 |
| Root correto / concluidos | 0/2 | 0/2 | 0/2 |
| Selecao correta / casos com candidato correto | 0/1 | 0/1 | 0/0, undefined |
| Repair-pair correto / concluidos | 0/2 | 0/2 | 0/2 |
| Recomendacao correta / recomendados | 0/1 | 0/1 | 0/1 |
| Coverage / concluidos | 1/2 | 1/2 | 1/2 |
| False abstention / positivos concluidos | 1/2 | 1/2 | 1/2 |
| Hipotese investigavel / concluidos | N/A | 0/2 | 0/2 |
| Contrato identificado / concluidos instrumentados | N/A | 1/2 | 0/2 |
| Vínculo efeito-observacao demonstrado / concluidos | N/A | 0/2 | 0/2 |
| SCC irrelevante selecionado / concluidos | 0/2 | 0/2 | 0/2 |

Delta diagnostico externo: zero melhoria medida nas populacoes concluidas. O contrato disponivel em Config.from_file nao demonstra violacao em seu corpo complexo e nao valida uma causa. A amostra e proposital e pequena; intervalos Wilson estao nos JSONs. Para root 0/2, IC95% [0; 65,8%]; nao permite conclusao ampla. Timeouts nao sao diagnósticos errados nem abstentions.

Development: Flask4992 tinha candidato no local adjudicado, mas selecao/reparo errados; Sphinx8506 perdeu arquivo/candidato e absteve; Sphinx7975 foi censurado na construcao do slice. Nenhuma dessas falhas justifica scoring novo.

## EXTERNAL HOLDOUT

Corpus seal SHA-256: `8d53d42008dfd87c7d9604cede84cc667456f31052d35ad632021525cb9376fe`.
Freeze: `2026-10-11T00:24:37.053868+00:00`; runtime freeze: `2026-10-11T00:42:22.596476+00:00`. Casos, commits, arquivos, hashes e thresholds estao em `external_v15/manifest.json` e nas fichas de adjudicacao. Referencias upstream constam do README do corpus.

One-shot: pytest8906 e Sphinx8801 concluidos, Sphinx8627 censurado em 240s. Os dois concluidos nao tinham arquivo/candidato correto; um recomendou incorretamente e outro absteve. Resultado root=0/2, repair=0/2, completion=2/3. Sphinx8627 nao entra no denominador diagnostico.

Nao houve rerun nem tuning baseado no holdout. `v15-external-holdout-one-shot.json` permanece intacto. A correcao numerica posterior invalida a qualificacao do runtime atual, independentemente de a avaliacao anterior ja reprovar. Gates finais em `v15-stage-gate-final.json` mantem essa restricao explicitamente.

## CONFIRMED ROOT VS INVESTIGABLE HYPOTHESIS

Live controlado, separado do benchmark externo: estado e gerador produziram locais investigaveis corretos, 2/2, sem roots, reparos ou recomendacoes; controle sem anchor absteve sem Qwen, 1/1. Total: 3/3 resultados esperados, duas chamadas. Isso demonstra funcionamento do caminho de investigacao, nao melhora de diagnostico externo.

| Populacao | Hipoteses localizadas corretamente | Roots confirmados corretos | Recomendacoes |
|---|---:|---:|---:|
| Estado/gerador controlados | 2/2 | Nenhum emitido | 0 |
| Externo dev concluido | 0/0, undefined | 0/2 | 1, incorreta |
| Externo holdout concluido | 0/0, undefined | 0/2 | 1, incorreta |

Nos externos, hypothesis_location_precision e hypothesis_evidence_validity sao undefined porque nenhuma hipotese foi emitida. Investigacao util em controles, sem progresso externo demonstrado, nao promove INVESTIGATION_CAPABILITY_READY.

## TIMEOUT DECOMPOSITION

Checkpoints START/END/ERROR por fase, gravados e fsynced no evaluator, sao independentes do retorno final. O runtime apenas emite observacoes, sem receber writer. Journals preservam memoria amostrada e a ultima fase de cada worker. Limites mantidos: 240s de parede, 60s por etapa estatica cooperativa e 1024MB de working-set monitorado; este ultimo nao e um hard allocation cap.

| Populacao | Mediana de parede / concluidos | p95 / concluidos | Timeouts / tentados |
|---|---:|---:|---:|
| Development v15 | 42,02s / 2 | 42,04s / 2 | 1/3 |
| Holdout v15 | 65,44s / 2 | 101,48s / 2 | 1/3 |

Sphinx7975: causal_slice consumiu 88,28s; ultimo checkpoint END em 157,69s, seguido da rejeicao pelo limite de etapa. Sphinx8627: qwen_analysis iniciou em 124,74s; requisicao Qwen iniciou em 170,59s e estava pendente no encerramento em 240s. Preparacao deterministica previa ao request: 45,86s. Nao e correto atribuir esses 45,86s a geracao do modelo.

Requests exatos concluidos: development 21,30s / 1, holdout 87,59s / 1; o segundo request do holdout e censurado e nao integra a media. Fila, TTFT e geracao nao sao distinguiveis pelo backend atual. `qwen_analysis` inclui preparacao/comparacao Python e nao deve ser chamado de tempo puro do modelo.

Os seis journals externos v15 registram tres requests iniciados, dois terminados e um censurado; o JSON historico `qwen_request_count` conta apenas terminados. O audit final acrescenta essa distincao sem reescrever o historico. Baseline congelado teve uma chamada concluida; controles tiveram duas, media 9,42s / 2. Nenhum caso teve segunda chamada de critica/judge.

Tempo/memoria em index, retrieval, symbol resolution, implementation graph, ledger, slice, recovery, contratos, Qwen, policy e ranking estao nos journals e no audit do gate. O preflight diagnostico tambem e contado no tempo total do worker; nao constitui segundo diagnostico do modelo.

## SYNTHETIC REGRESSION

Runtime atual, apos correcao numerica: retrieval/root/relation=172/172; identidade de slots=102/102; SCC=10/10. `v15-synthetic-structural-current.json` registra a suite/hash reproduzivel. A versao anterior tambem foi preservada.

Testes unitarios mantem recovery v7, relevancia SCC v12, transporte v13, grafo v14, identidade, targets, ranking, negativos, metamorfismos e contrafactuais. Scoring geral, policy, recovery budgets e representacao causal legado nao foram alterados.

O gate sintetico se refere ao escopo deterministico anteriormente validado e suas regressoes. Nao houve nova execucao live exaustiva das populacoes v10 neste v15; resultados live historicos nao sao apresentados como experimentos atuais.

## SAFETY AND TESTS

Auditoria independente dos outputs completos: referencias validas 5/5 baseline, 5/5 development e 3/3 holdout; unsupported grounding claims 0/4 em cada populacao. Grounding mede compatibilidade com fatos tipados, nao prova causalidade do diagnostico errado. SCC irrelevante=0/2 em cada populacao externa concluida. Forbidden recommendations=0.

Por populacao externa e nos controles: filesystem_mutations do projeto analisado=0; tool_dispatches=0; authority_grants=0; execution_authorized=0; destructive_actions=0. Writers de fixtures/relatorios do evaluator nao sao mutacoes de projeto externo. Nenhum codigo, teste, installer ou patch de terceiros foi executado/importado no host. Nenhuma promocao ou aplicacao automatica, nenhuma Simulation. Sandbox v11 permanece fail-closed, Docker nao foi instalado nem configurado.

Testes novos cobrem tipos primitivos validos/invalidos e shadowing, promocoes/contradicoes, IDs invalidos, schema fechado, requisitos que o modelo nao pode remover, efeitos sem causalidade, receiver ambiguo, uma chamada/zero chamadas, serializacao/CLI, timings, censura, hashes dos journals e backup fora de Temp.

Suite completa final: 1628 passed, 3 skipped, 1 aviso preexistente VOICE_TTS_SPEED (164,39s). Suite preditiva final: 387 passed, 2 skipped (94,20s), incluindo os tres novos controles do audit de journals. Integracoes opt-in indisponiveis sao skipped, nao declaradas verificadas.

## GIT AND ARTIFACT PRESERVATION

Commits: `53b94fa` avaliacao; `6bcf6ab` infraestrutura de contratos/hipoteses; `e322958` correcao numerica independente. O commit final acrescenta apenas relatorios compactos, gates e verificacao operacional. Remote HEAD consta da entrega apos push. Autoria configurada preservada; nenhum rewrite, `_arquivo/`, snapshot ou segredo adicionado.

Armazenamento permanente: `C:/Users/User/AppData/Local/JARVIS/evaluation-artifacts/`. V15: nove traces gzip completos e seis journals, copias byte-for-byte verificadas com SHA-256 em `v15-artifact-preservation-final.json`; originais retidos. V13: 32 traces revalidados; v14: 14 traces revalidados. Historico v12: 22 traces e ZIP preservados; ZIP SHA-256 `4c4acc4f4f038283a7e388367402b7022b40474cffafff0c319308e6c49f4b05`. Indices compactos permanecem no Git. Backup local no mesmo host nao e disaster recovery off-machine.

## FINAL GATES

`PREDICTIVE_SYNTHETIC_READY = true`

`INVESTIGATION_CAPABILITY_READY = false`

`PREDICTIVE_EXTERNAL_COVERAGE_READY = false`

`PREDICTIVE_EXTERNAL_DECISION_READY = false`

`SIMULATION_RUNTIME_READY = false`

Conclusao: caminho consultivo de investigacao e contratos limitados implementado e testado; diagnostico externo nao melhorou, completude operacional falhou e a qualificacao externa da correcao pos-seal foi invalidada. O proximo trabalho deve investigar arquivos/anchors ausentes, significado de roots existentes e custo do slice/comparacao; nao aumentar confianca, scoring ou autoridade. Candidate Simulation nao foi executada.
