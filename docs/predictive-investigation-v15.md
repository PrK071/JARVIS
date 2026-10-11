# Investigação consultiva v15

`jarvis predict` e `jarvis predict --json` continuam compatíveis. Quando há uma
API ancorada e fatos verificáveis, mas nenhum candidato causal sustentado, o
serviço pode usar **uma chamada estruturada** para selecionar locais investigáveis.
Não há chamada adicional de crítica/julgamento. Sem âncora, não há chamada.

O JSON mantém os campos públicos anteriores e acrescenta:

- `behavioral_contract_schema_version=1` e `behavioral_contract_contexts`;
- `investigation_schema_version=1` e `investigable_hypotheses`;
- tempos de `behavioral_contracts` e `policy` em `phase_seconds`.

Hipóteses investigáveis ficam separadas de `hypotheses`, `root_cause_candidates`
e `candidates`. Um relatório investigativo é insuficiente para recomendar reparo:
`recommended_candidate_id=null`, `execution_authorized=false`,
`requires_approval=true`. O modelo escolhe IDs e requisitos de evidência, não
escreve livremente causas, scores ou reparos. Python preserva todos os requisitos
faltantes, mesmo quando o modelo menciona apenas alguns.

## Limites de contrato

Existência, aplicabilidade e violação são campos distintos. A implementação inicial
reconhece expectativas explicitamente rotuladas `Expected:` como relatos, e um
subconjunto conservador de anotações primitivas. Somente uma incompatibilidade
literal incondicional, sem decorators/async/rebinding conhecido, sustenta a
verificação estática de violação. Isso **não prova execução ou responsabilidade
pelo sintoma**. Contratos complexos, asserts arbitrários, estados dinâmicos e
semântica de bibliotecas permanecem desconhecidos. As categorias de fonte para
testes/guards/contratos explícitos existem no modelo, mas não são automaticamente
inferidas de texto ou docstrings.

As relações de retorno encaminhado e condição governando efeito possuem
proveniência AST. Os vínculos ao relato continuam inferidos e condicionais.
Alcancabilidade não equivale a execução nem a defeito. `self._helper` permanece
resolvido somente no subconjunto estático v14; instanciação conhecida ainda não
introduz automaticamente um tipo de receiver. Dispatch ambíguo continua explícito.

O validador de promoção aceita apenas o subtipo estático implementado e um
RootCauseCandidate **já representado**, com caminho determinístico, entidade e
evidências compatíveis. Não cria paths, candidatos ou recomendações a partir de
confiança do modelo. A pipeline investigativa não promove automaticamente nada.

## Checkpoints e avaliação

O evaluator registra START/END/ERROR com working set amostrado, em JSONL fora do
snapshot. Um worker encerrado mantém o último checkpoint completo. Timeout é
`NOT_EVALUATED_TIMEOUT`, não erro causal ou abstention. Os limites de 240 s/caso,
60 s/etapa estática e 1024 MB amostrados permanecem; a amostragem não é um limite
absoluto de alocação. Fila e geração do modelo não são separadas quando o backend
não fornece essa informação.

```powershell
python -m scripts.predictive_external_eval --split development --mode live --corpus tests/data/predictive/external_v15 --snapshots .orchestrator/external-v15/snapshots --case-timeout 240 --output caminho-novo.json
```

Live é opt-in. Nenhum projeto externo é importado ou executado. Métricas de
localização de hipóteses não são accuracy de diagnóstico. Populações vazias são
undefined; casos censurados aparecem separadamente e impedem readiness.
Simulation, patches, concessão de autoridade e mudança de scoring não fazem
parte desta etapa. O sandbox permanece fail-closed.
