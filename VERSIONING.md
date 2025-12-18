# Guia simples de versionamento (Quantmaster)

Este projeto publica no PyPI via GitHub Actions quando uma **tag** `v*` é criada/pushada.

## Regra de versão (simples)

Use **SemVer**: `MAJOR.MINOR.PATCH`

- **PATCH**: correções/ajustes internos (sem quebrar API)
- **MINOR**: nova feature compatível
- **MAJOR**: quebra de compatibilidade

## Checklist rápido de release

### 1) Atualize a versão

Edite `pyproject.toml`:

- `version = "X.Y.Z"`

### 2) Rode testes

```bash
python -m pytest -q
```

### 3) Commit do bump

```bash
git status -sb

git add pyproject.toml
# (adicione outros arquivos que façam parte do release, se houver)

git commit -m "Bump version to X.Y.Z"
```

### 4) Crie a tag (importante: prefixo `v`)

O workflow de publish roda em tags `v*`, então use `vX.Y.Z`:

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
```

### 5) Push do commit e da tag

```bash
git push origin main
git push origin vX.Y.Z
```

### 6) Verifique o publish

- GitHub Actions: workflow **Publish to PyPI** deve rodar automaticamente
- PyPI: confirme que a versão `X.Y.Z` apareceu

## Dicas / problemas comuns

- Se você criar a tag sem `v` (ex.: `0.1.2`), **não dispara** o publish.
- Se já existir a tag (mesmo nome), o Git vai recusar; nesse caso escolha outra versão.
- Para testar build local (opcional):

```bash
python -m pip install --upgrade build
python -m build
```
