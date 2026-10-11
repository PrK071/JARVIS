# JARVIS v16 — Evidence Sufficiency e estabilização operacional

## CURRENT CAUSAL FUNNEL

Baseline: `4c259064f8761b0295ff846e45fa63ca21c61690`. Sua qualificação externa já estava invalidada pela correção de subtipagem numérica posterior ao holdout v15. Não requalificamos esse runtime usando os holdouts históricos.

O novo corpus contém **12 defeitos independentes** de Requests, pytest e Sphinx: quatro development e oito holdout. Formulações A/B não são defeitos adicionais. A matriz elegível foi congelada antes das mudanças decisórias: A/B para todos, C somente para um development, D indisponível.

Funil observado no holdout, modo A, população de oito defeitos concluídos:

| Etapa | Numerador / denominador | Interpretação |
|---|---:|---|
| Relato disponível | 8/8 | Não mede completude da observação |
| Arquivo adjudicado recuperado | 6/8 | Duas perdas de retrieval permanecem |
| API relatada resolvida | 5/8 | Resolução, não recall de um oracle independente de API |
| Implementação alcançável | 5/8 | Alcance estático, não execução demonstrada |
| Efeitos representados | 5/8 | Existência de operação, não responsabilidade causal |
| Algum contrato existente / aplicável | 1/8 / 1/8 | Não comprova violação |
| Violação sustentada | 0/8 | Gargalo decisivo |
| Roots estruturais presentes | 3/8 | Não necessariamente corretos |
| Root correto representado como candidato | 0/8 | Proxy de localização adjudicada |
| Root aprovado pela triagem | 0/8 | Não encaminhar roots sem suporte para reparo |
| Investigação emitida | 5/8 | Sugestão consultiva, não diagnóstico |
| Recomendação de reparo | 0/8 | Cobertura diagnóstica zero |

`v16-evidence-study-audit.json` contém os registros por caso; traces completos preservam entidades, relações, evidências, custos e requisitos adicionais. `affected_api_recall`, `root_support_validity` e eficácia das solicitações são **undefined**, pois não há adjudicação independente suficiente dessas populações. O sinal `observation_identified` do runtime apenas distingue o tipo de sintoma; não comprova que inputs, outputs e condições estejam completos.

## UNSUPPORTED ROOT ANALYSIS

Na matriz A/B do holdout, foram preservadas 72 avaliações de roots: **70 `ROOT_EXISTS_BUT_CAUSAL_BRIDGE_MISSING` e dois `ROOT_EXISTS_BUT_CONTRACT_UNVERIFIED`**. São avaliações correlacionadas de candidatos, não 72 defeitos. Nenhum root foi aprovado.

Antes, presença de um root genérico podia impedir a investigação mesmo sem ponte entre efeito e comportamento. Escores estruturais não resolvem essa lacuna. Nenhuma classe foi penalizada por nome ou palavra-chave; aplicamos as validações existentes de evidência, caminho, relevância de SCC e promoção de contrato. Não há novos tipos de causa, pesos ou budgets.

## INVESTIGATION ROUTING AUDIT

O serviço agora separa ausência de roots de **`ROOT_PRESENT_BUT_UNSUPPORTED`**. Roots originais permanecem no relatório, com IDs intactos; apenas roots aprovados podem entrar na geração de reparos. Quando não há suporte, locais reais e ancorados podem seguir o protocolo investigativo v15, com no máximo uma chamada Qwen. Sem local confiável, permanece abstention com solicitação específica de dados.

Investigações não produzem candidatos elegíveis, patches ou recomendações. O diagnóstico confirmado e a investigação continuam incompatíveis. A ambiguidade informada pelo seletor preserva os candidatos e solicita informação para distinguir alternativas, sem nova chamada.

## EVIDENCE SUFFICIENCY MODEL

`predictive/sufficiency.py` adiciona avaliação categórica, sem score ou confiança do modelo. As categorias incluem diagnóstico, investigação, necessidade de resolução de API, contrato, observação runtime, ambiguidade e insuficiência. O relatório expõe uma subversão explícita `evidence_sufficiency_schema_version=1`; interfaces anteriores permanecem.

Requisitos: evidências existentes no ledger, entidades reais e caminho determinístico. SCC reutiliza relevância v12; comportamentos reutilizam a promoção v15; exceções exigem frame relatado correspondente e lineage estrutural. **Um frame e um caminho estático não provam todas as condições de execução nem responsabilidade universal.** Essa limitação permanece visível e impede alegar triagem externa validada.

Existência, aplicabilidade e violação de contrato continuam distintas. Reachability nunca passa a significar execução observada. Não há oracle no runtime.

## EVIDENCE ACQUISITION CONTRACT

`DiagnosticEvidenceRequest` registra fato ausente, por que distinguiria alternativas, símbolos relacionados, tipos de evidência aceitáveis e IDs/decisões que pode desambiguar. Exemplos implementados:

- `AFFECTED_API_INVOCATION`: chamada reproduzível ou traceback real para resolver a entrada.
- `EFFECT_TO_OBSERVATION`: input/output, estado inicial/final ou trace existente para ligar efeito ao comportamento.
- `APPLICABLE_BEHAVIORAL_CONTRACT`: expectativa/assertion preexistente para distinguir comportamento permitido de violação.
- `DISTINGUISH_CAUSAL_ALTERNATIVES`: dados que diferenciem roots fundamentados ainda ambíguos.

Todas têm `execution_requested=false`. Não executamos terceiros para adquiri-las. **A utilidade externa dessas solicitações ainda não foi adjudicada: `required_evidence_precision=undefined (0/0)`.** Os testes provam forma e bloqueios, não eficácia universal nem informação necessária em todos os casos.

## VALUE-OF-INFORMATION STUDY

| Modo | Informação permitida | Development elegível | Holdout elegível |
|---|---|---:|---:|
| A | Relato upstream original | 4 | 8 |
| B | Mesmo relato estruturado, sem novo conhecimento | 4 | 8 |
| C | Teste no commit anterior citado pelo próprio relato | 1 | 0 |
| D | Logs/traces adicionais preexistentes | 0 | 0 |

Gold patch, explicação corretiva e root adjudicado nunca entram no reasoner. Dados externos são entradas não confiáveis, não instruções de execução. A ausência de D não autoriza fabricar logs.

Em development, B criou novos metadados de contrato em três dos quatro pares, mas o formatter usa `Expected: UNKNOWN` quando não há expectativa normalizada. O parser existente preserva isso como expectativa relatada não verificada. **IDs novos nessa situação não são ganho causal.** A auditoria registra o artefato sem alterar engine ou formatter após o holdout.

O único C alterou contexto de API/símbolos/efeitos e passou de timeout baseline para investigação concluída, mas continuou com **0/1 candidato correto e 0/1 diagnóstico correto**. Não permite estimar valor geral de evidência runtime. `EVIDENCE_INFORMATION_GAIN` registra apenas transições observáveis; nunca as interpreta automaticamente como melhoria de verdade.

Intervenções oracle isoladas em quatro fixtures próprios: sem identidade de API, quatro casos pedem resolução. API conhecida permite um contrato de retorno literal incondicional aprovado; retorno condicional, subtipagem bool correta e expectativa apenas relatada continuam investigáveis sem promoção. **Zero chamadas Qwen e zero execução externa.** Inserir candidato não permite ignorar contrato não verificado. Esses controles não são accuracy externa, nem estudo completo de oracle em projetos reais.

## OPERATIONAL BOTTLENECKS

Profiling determinístico histórico Sphinx7975, mesmo snapshot, sem Qwen:

| Fase | Antes | Depois |
|---|---:|---:|
| Índice | 14,551 s | 14,566 s |
| Causal slice | 102,792 s | 47,246 s |
| Comparações pairwise | 28,657 s | 2,766 s |

Redundâncias demonstradas: ordenação de edges por root, varredura global por par determinístico, recomputação de assinaturas por comparação. A correção reutiliza lookups **locais ao snapshot**, sem cache global, novos pesos ou mudança de semântica. Mesmo conjunto de 87.742 nós, 101.711 arestas e 12 roots; slice, evidências, IDs e comparações possuem o mesmo SHA-256 antes/depois: `236b16f8063032988850fe10319e013835495ebe0b3d2a47d9e152b527c6167f`.

O harness novo observa o fluxo real, removendo o preflight duplicado, sem repetir index/context/slice só para métricas. Não confundir esse ganho de medição com ganho do modelo. Orçamento continua 240 s por input; memória monitorada a cada segundo, limite de 1.024 MiB, não confinamento de alocação nem sandbox.

O C baseline terminou a requisição Qwen aos 216,40 s e foi censurado depois, aos 240 s. Preparação pré-modelo levou 92,33 s; HTTP levou 78,51 s. Portanto não era apenas timeout do backend. Depois, esse input concluiu em 80,31 s. Checkpoints e memória sobrevivem à interrupção; respostas brutas passam a ser preservadas antes da validação. A resposta bruta daquele timeout baseline não foi recuperada, mas seu último checkpoint foi preservado.

Wall time A: development mediana 42,29→14,70 s, p95 90,52→28,59 s; holdout mediana 30,21 s, p95 42,93 s. B holdout: mediana 29,76 s, p95 60,82 s. As mudanças de rota também alteram prompts; não atribuir todo ganho a caching. Tempos HTTP incluem fila+geração, que o backend não separou: média por requisição 55,24 s baseline, 19,68 s development v16 e 24,96 s holdout. Foram sete requisições em cada matriz development e dez no holdout. `v16-operational-journals.json` mantém checkpoints e picos amostrados; maior working set amostrado no holdout 451.006.464 bytes (430,1 MiB), não pico absoluto garantido.

## BEFORE/AFTER DEVELOPMENT

Métricas principais A, população de quatro defeitos independentes; não misturar A/B:

| Métrica | Baseline | v16 |
|---|---:|---:|
| Completion | 4/4 | 4/4 |
| Arquivo correto | 4/4 | 4/4 |
| Candidato correto | 0/4 | 0/4 |
| Root correto | 0/4 | 0/4 |
| Repair-pair correto | 0/4 | 0/4 |
| Recomendações / população | 2/4 | 0/4 |
| Recommendation precision | 0/2 | undefined (0/0) |
| False abstention diagnóstica | 2/4 | 4/4 |
| Investigação emitida | 1/4 | 3/4 |
| Localização exata de hipóteses | 0/2 | 0/6 |

Matriz operacional completa: baseline **8/9** inputs, v16 **9/9**. Investigar não conta como diagnóstico. Eliminamos duas recomendações incorretas, **não melhoramos root recall**. Localização de hipótese usa proxy exato do root adjudicado; pode penalizar investigação legítima na API pública. Não prova que toda investigação seja inútil. Não há controles negativos externos suficientes para estimar correct-abstention rate.

## NEW SEALED HOLDOUT

Seal de corpus: `898bb85b4fb7a29dc689e31e7ee0e64d3e55b620acc000e2f2443a793c36b5e4`, congelado em `2026-10-11T01:43:58.326903+00:00`. Runtime/evaluator congelados em `2026-10-11T02:11:25.423723+00:00`, com hashes em `v16-runtime-freeze.json`. Thresholds e elegibilidade registrados antes da execução. Não houve tuning posterior.

Oito novos defeitos, três repositórios; A/B totalizam **16/16 entradas concluídas**, sem timeout determinístico ou Qwen. No A: candidatos corretos **0/8**, root **0/8**, repair-pair **0/8**, investigação **5/8**, precisão de localização **0/9**, recomendações **0/8**, recommendation precision **undefined (0/0)**, false abstention diagnóstica **8/8**. B também não gerou candidato correto. Dez requisições Qwen no total A/B; nunca duas no mesmo input.

Root 0/8 tem intervalo Wilson 95% aproximadamente [0%; 32,4%]; completion 8/8 [67,6%; 100%]. A/B são correlacionados; não estreitar intervalos fingindo 16 defeitos. Amostra intencional pequena, projeções Python limitadas, casos históricos inclusive Python 2 e dois oracles em nível de classe: não qualifica produção nem toda biblioteca atual. API oracle não foi selado separadamente; não alegar API recall independente.

## SYNTHETIC REGRESSION

Regressão estrutural gerada v10: retrieval/root candidate/relação **172/172**, binding identity **102/102**, SCC **10/10**. Comparações pairwise preservadas, incluindo ordenação invertida. Suíte funcional inclui recovery v7, SCC relevance v12, transporte v13, grafo v14 e contratos v15.

Readiness sintética é preservada **no escopo anteriormente validado**, apoiada nessa regressão e nos testes. Não repetimos o holdout v10 inteiro live e não transformamos esse gate em generalização externa.

## SAFETY

Em todos os 16 inputs do holdout: mutações dos snapshots externos **0**, dispatches **0**, authority grants **0**, execution_authorized **0**, ações destrutivas **0**, recomendações proibidas **0**. Nenhum import/execução/teste/installer de terceiros, patch, simulação ou instalação Docker. Escritas autorizadas de implementação e artefatos diagnósticos não são mutações do projeto externo.

Referências válidas: baseline **300/300**, development v16 **380/380**, holdout A/B **540/540**. Claims confirmados unsupported: baseline **0/16**; development e holdout v16 **0/0**, taxa **undefined**, pois não houve diagnósticos. Não apresentar ausência de claims como evidência de accuracy. Hipóteses consultivas continuam rotuladas e com IDs validados. Sandbox v11 permanece inalterado e fail-closed; sem nova verificação real de isolamento.

## TESTS

Suíte Predictive: **398 passed, 2 skipped**, sem Qwen obrigatório. Suíte completa final registrada separadamente em `v16-tests.json`. Skips de integrações permanecem explícitos; warning existente de configuração TTS não é falha causal.

Testes novos cobrem root presente sem suporte, caminho inválido, contrato estático verificável, investigação de uma chamada sem reparo, solicitação específica sem anchor, seal e denominadores de timeout, formatter sem oracle e equivalência de comparações cacheadas. Não afrouxamos candidate policy nem regras de grounding; fixtures antigos sem traceback real foram explicitados como exceção ou expectativa investigativa conforme o contrato novo.

## GIT AND ARTIFACT PRESERVATION

Checkout de entrega: `D:/JARVIS-v16-work`, branch main, baseline preservada separadamente. Histórico anterior não foi reescrito. O clone de entrega possui seus próprios objetos Git, verificados por `git fsck --connectivity-only`; não depende do clone em Temp. `_arquivo/`, credenciais, workspaces, cache e traces brutos ficam fora dos commits.

Traces completos v16: `D:/JARVIS-evaluation-v16/artifacts/v16`, **112 arquivos com hashes**, inventariados em `v16-artifact-preservation.json`; relatórios compactos e seals ficam no Git. Snapshots/cache em `D:/JARVIS-evaluation-v16/external-v16`. Nenhum trace histórico temporário foi limpo.

Históricos permanentes em `C:/Users/User/AppData/Local/JARVIS/evaluation-artifacts`: ZIP v12 revalidado por hash integral `4c4acc4f4f038283a7e388367402b7022b40474cffafff0c319308e6c49f4b05` (contém os 22 traces preservados anteriormente); v13 **32**, v14 **14**, v15 **9** gzip revalidados individualmente. Não alegar revalidação individual dos 22 v12 quando a verificação realizada foi do ZIP completo. Todas essas cópias estão no mesmo host; **backup off-host continua recomendado**.

Commits e remote HEAD são informados na entrega após validação e push. Mensagens curtas usam `caveman-commit`; a skill não concede autoridade nem faz staging/push por conta própria.

## FINAL GATES

```text
PREDICTIVE_SYNTHETIC_READY = true
DIAGNOSTIC_TRIAGE_READY = false
OPERATIONAL_EVALUATION_READY = true
INVESTIGATION_CAPABILITY_READY = false
PREDICTIVE_EXTERNAL_COVERAGE_READY = false
PREDICTIVE_EXTERNAL_DECISION_READY = false
SIMULATION_RUNTIME_READY = false
```

Operação passou **no corpus e orçamento selados**, não em qualquer repositório. Triagem e investigação externas não passam: localização não demonstrada e eficácia da evidência requisitada undefined. Mais classes/testes não promovem esses gates.

**`STATIC_EVIDENCE_INSUFFICIENT_FOR_RELIABLE_DIAGNOSIS`**, no escopo deste motor e informação estudados. Não é prova de impossibilidade teórica da análise estática; perdas de retrieval e modelagem também permanecem. A amostra não demonstra que toda falha necessariamente exige runtime. Contudo, não há evidência para continuar adicionando regras especulativas ou ajustar ranking.

Próximo marco recomendado: **Secure Sandbox Validation → Controlled Runtime Evidence → Causal Reassessment**. Validar isolamento antes de adquirir execução controlada, manter diagnóstico consultivo e sem aplicação automática. Candidate Simulation não foi implementada nesta tarefa.
