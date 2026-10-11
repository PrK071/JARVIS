# Auditoria v16 antes de alterar o runtime

Baseline `4c259064f8761b0295ff846e45fa63ca21c61690`, com qualificacao externa explicitamente invalidada pela correcao numerica pos-holdout v15. Resultados v12-v15 permanecem historicos.

## Funil e suporte

O servico chama investigacao somente se `root_causes` estiver vazio. Presenca nao implica suporte: `contract_demonstrated` pode indicar apenas anotacao/operacao downstream; o perfil de responsabilidade tambem pode assumir violacao de NULL_FLOW/TYPE_FLOW sem validar o comportamento relatado. Roots com score estrutural elevado e failure anchor sintetico podem, portanto, evitar a investigacao. A validacao v15 de promocao existe separadamente, mas nao governa esses roots legados.

Os dois externos concluidos do development v15 produziram: um candidato localizado com selecao errada, e um caso sem arquivo/candidato correto. No holdout, nenhum dos dois concluidos tinha candidato correto. Nao existe evidencia para ajustar ranking. Falta distinguir entidade existente, ponte causal existente e violacao de contrato comprovada.

## Profiling deterministico historico

`v16-static-profile-before.json`, Sphinx7975, sem Qwen ou execucao de projeto: 87.742 nodes, 101.711 edges, 12 roots. Slice 102,79s sob cProfile; comparacao pairwise 28,66s. Tempos incluem overhead do profiler e nao sao diretamente comparaveis aos 88,28s do worker v15.

Hotspots: `_responsibility_profile` 46,85s cumulativos; ordenacao repetida de todas as edges; 58,4 milhoes de iteracoes para testar pares deterministas. Pairwise calcula 132 assinaturas para 12 roots e reconstroi `_incoming` 44 vezes (19,74s cumulativos). Sao redundancias de lookup, nao necessidade de maior contexto ou scoring diferente.

O evaluator externo tambem reconstroi indice/contexto/ledger/slice no preflight e no servico. Essa duplicacao e observabilidade, nao segunda chamada Qwen; torna o custo de parede maior. Qualquer remocao precisa preservar os fatos iniciais usados nas metricas.

## Hipoteses de correcao e prova exigida

1. Lookups/ordenacao por snapshot: hash identico de slice, roots e comparacoes antes/depois, testes de isolamento e sem cache global persistente.
2. Triagem categorica de suporte: controles onde root existe mas contrato/ponte faltam, controles com roots demonstrados, negativos/ambiguos; nenhum oracle no runtime.
3. Requests de evidencia por relacao faltante: nao solicitar inputs/configuracao indiscriminadamente se nao distinguem alternativas reais.

Nenhum budget de recovery, score, politica, grounding, modelo ou sandbox deve mudar. Novo holdout precisa ser selado antes de qualquer mudanca decisoria. Completion=100% e requisito operacional, nao prova de diagnostico correto. Evidence enrichment precisa de proveniencia pre-correcao; gold patch so pertence ao oracle.
