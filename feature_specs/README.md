# Feature Specs

Use esta pasta para especificar novas features antes de codificar.

## Template mínimo de spec

```md
# <feature_name>

## Hipótese
- Qual ineficiência/efeito financeiro a feature tenta capturar?

## Fórmula
- Definição matemática explícita.

## Entradas
- Colunas necessárias (ex.: close, volume).
- Tipo de input (`DataFrame`/`Series`).

## Parâmetros
- Nome, tipo, default e restrições.

## Saída
- Tipo (`Series` ou `DataFrame`), nome esperado e index alignment.

## Regras de borda
- Tratamento de NaN, zeros, divisões por zero e janela insuficiente.

## Plano de teste
- Testes de shape/index, no-lookahead, nomes e validação de parâmetros.
```

