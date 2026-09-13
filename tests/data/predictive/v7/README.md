# Structural Recovery v7

O conjunto v7 mede recuperacao causal limitada, nao busca lexical ampla. Os 15
casos novos de development cobrem retorno em wrappers, forwarding posicional e
keyword, origem de atributo, SCC de imports e abstention verdadeiro. O
`holdout_v7` contem 20 casos novos em dominios diferentes e foi congelado antes
da execucao live.

O hash selado cobre os casos, adjudicacoes estruturais e todos os arquivos dos
fixtures referenciados. Recovery pode usar somente relacoes verificadas no
`ProjectSnapshotV2`, com uma tentativa, profundidade dois, quatro arquivos,
doze simbolos e trinta novos atoms como limites padrao.

Metas e definicoes:

- `recovery_success_rate`: tentativa inicialmente insuficiente que termina com
  suficiencia causal e root cause estruturalmente valido;
- `recovery_precision`: mudancas de contexto que terminam com root cause valido
  divididas por todas as mudancas de contexto;
- `false_abstention_before`: casos positivos sem suficiencia causal inicial;
- `false_abstention_after`: casos positivos que ainda abstiveram no relatorio;
- `recovery_noise_rate`: expansoes que nao terminam em root cause valido.

`holdout_v7` e one-shot. Depois de observado, nao deve orientar tuning nesta
iteracao.
