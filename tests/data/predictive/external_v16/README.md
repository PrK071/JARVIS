# Estudo externo v16

`cases.json` e `manifest.json` contêm adjudicação e seal, não são entradas completas do reasoner. Há 12 defeitos independentes (quatro development, oito holdout), não 25 defeitos correspondentes às formulações elegíveis.

A é relato original; B contém o mesmo conhecimento reorganizado; C é um teste preexistente citado no relato (um caso); D está indisponível. O diff corretivo pertence somente ao evaluator. Projetos externos são lidos como dados e jamais importados ou executados pelo estudo.

Os hashes dos snapshots são verificados antes e depois; traces grandes ficam fora de Temp e do Git. Consulte `../results/v16-artifact-preservation.json` e `../results/v16-report.md`. O corpus é intencional e pequeno, não qualificação de produção. Fontes upstream e commits estão em cada caso; os arquivos de terceiros conservam suas licenças nos projetos de origem.

O holdout v16 já foi aberto uma única vez. Não repetir para tuning ou promoção. Novos estudos devem usar outro corpus/seal e outro destino de artefatos.

Para um estudo development opt-in, com snapshots já preparados e Qwen local:

```powershell
python scripts/predictive_v16_evidence_study.py --split development --live `
  --snapshots D:/JARVIS-evaluation-v16/external-v16/snapshots `
  --archive D:/JARVIS-evaluation-v16/artifacts/new-development-run `
  --output D:/JARVIS-evaluation-v16/artifacts/new-development-summary.json
```

O runtime padrão é o checkout corrente e deve ser código JARVIS confiável. `--runtime` serve exclusivamente para comparar uma baseline JARVIS explicitamente confiável, nunca um projeto externo. Os destinos devem ser novos e externos aos snapshots. Orçamento selado: 240 segundos/input, uma chamada Qwen no máximo, memória monitorada de 1.024 MiB.

`record_predictive_v16.py` valida o freeze, os resultados e a preservação. Não usar seu `--freeze` para substituir o freeze histórico. `audit_diagnostic_study.py` somente analisa traces existentes e não altera entradas, resultados ou thresholds.
