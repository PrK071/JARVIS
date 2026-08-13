# Recuperação do controle de versão

Data: 2026-07-30

## Problema e decisão

`D:\tern\.git` existia, mas estava completamente vazio: não havia `HEAD`,
`config`, índice, objetos, referências, logs, locks ou informação de remote.
Sem commits ou remote comprovável, não foi possível recuperar histórico real.

Foi usado o cenário de novo repositório local:

- backup externo: `D:\tern-git-recovery-20260730-160616`;
- metadado vazio preservado também em
  `D:\tern\.git-corrupt-20260730-160616`;
- branch principal: `main`;
- commit baseline: `Initial stable baseline: Qwen, Codex, web search and Piper`;
- tag local: `stable-piper-baseline-20260730`;
- remote: nenhum;
- push: não realizado.

## Arquivos deliberadamente ignorados

O `.gitignore` mantém fora do baseline:

- `.env` e variantes locais;
- ambientes virtuais, caches Python e cobertura;
- `.orchestrator`, logs, temporários e bancos locais;
- áudio gerado;
- pesos `GGUF`, `safetensors`, ONNX, PyTorch e arquivos binários de modelos;
- runtimes executáveis e arquivos compactados;
- diretórios locais de modelos legados;
- instalações locais de skills;
- `.git-corrupt-*`.

Metadados pequenos e configurações de modelos não secretos permanecem
versionáveis; os pesos ficam ignorados.

## Verificação

```powershell
Set-Location D:\tern
git status
git branch -vv
git log --oneline --decorate -n 5
git tag --list
git remote -v
git fsck --full
python -m pytest -q
python -m tern.orchestrator status
python -m tern.orchestrator voice-diagnose
```

A identidade Git existente foi preservada. Para alterá-la somente neste
repositório:

```powershell
git config user.name "Seu Nome"
git config user.email "voce@example.com"
```
