# relative_jump_contribution

## Hipótese

Em janelas curtas, aumentos na parcela de variância explicada por jumps indicam estresse de mercado, pior qualidade de liquidez e maior prêmio de risco exigido.
Como feature de regime/risk-state, valores altos tendem a sinalizar deterioração de previsibilidade de sinais suaves e aumento de risco de cauda.

## Fórmula

Para retornos intradiários \(r_{t,i}\) no dia \(t\):

- Realized Variance:  
  \(RV_t = \sum_i r_{t,i}^2\)
- Bipower Variation:  
  \(BV_t = \mu_1^{-2} \sum_{i=2}^{n_t} |r_{t,i}||r_{t,i-1}|\), com \(\mu_1=\sqrt{2/\pi}\)
- Jump Variation truncada em zero:  
  \(JV_t = \max(RV_t - BV_t, 0)\)
- Relative Jump Contribution:  
  \(RJC_t = JV_t / RV_t\), quando \(RV_t > 0\), caso contrário `NaN`.

Versão suavizada para uso em modelagem:

\[
RJC^{(w)}_t = \text{rolling\_mean}_w(RJC_t)
\]

## Entradas

- `data`: `pd.DataFrame` com preços intradiários ou retornos intradiários.
- Modo preço:
  - coluna obrigatória: `price_col` (default: `close`)
  - retornos calculados internamente via log-retorno.
- Modo retorno:
  - coluna obrigatória: `return_col`
- Coluna de agrupamento diário:
  - índice `DatetimeIndex` ou coluna temporal explícita.

## Parâmetros

- `window: int = 20`
  - janela de suavização diária do `RJC_t`.
  - restrição: `window >= 1`.
- `price_col: str = "close"` (quando usando preços).
- `return_col: str | None = None` (quando já houver retornos).
- `min_intraday_obs: int = 10`
  - mínimo de observações intradiárias para computar um dia válido.
- `eps: float = 1e-12`
  - proteção numérica para divisões por quase-zero.

## Saída

- `pd.Series` indexada por dia (ou alinhada ao índice diário final da agregação).
- Nome esperado:
  - `relative_jump_contribution_<window>`.
- Faixa esperada: \([0, 1]\) para dias válidos.

## Regras de borda

- Dias com menos de `min_intraday_obs`: `NaN`.
- `RV_t <= eps`: `NaN`.
- `BV_t > RV_t`: usar `JV_t = 0` (truncamento).
- Valores não numéricos: coercion para `NaN`.
- Se usar preços não positivos para log-retorno: observações inválidas descartadas no cálculo de retorno.

## Prova de causalidade temporal (no-lookahead)

- Para calcular `RJC_t`, usar somente ticks/barras intradiárias do próprio dia `t` até seu fechamento.
- Para `RJC^{(w)}_t`, usar somente `RJC_{t-w+1}, ..., RJC_t`.
- Não usar `shift(-k)`, janela central (`center=True`) ou qualquer agregação que consuma dias futuros.

## Política no-repaint

- Uma vez fechado o dia `t`, `RJC_t` não pode ser recalculado com dados de `t+1+`.
- Em backfill incremental, anexar novos dias não altera valores históricos já emitidos (exceto se o usuário alterar explicitamente dados históricos brutos).

## Plano de teste

1. **Contrato básico**
   - shape/index alinhado com série diária.
   - `name` correto.
   - valores em \([0, 1]\) para válidos.
2. **No-lookahead**
   - usar `assert_no_lookahead` adaptado ao nível diário.
3. **No-repaint**
   - calcular com conjunto A, anexar dias futuros (conjunto B), recalcular e verificar igualdade dos dias de A.
4. **Bordas**
   - dias com baixa contagem intradiária.
   - `RV_t` quase zero.
   - dados com NaN e preços não positivos.
5. **Sanidade econômica**
   - cenário sintético com jumps explícitos deve elevar `RJC`.
   - cenário contínuo suave deve gerar `RJC` baixo.

## Nota de priorização do discovery

Score total: **20/25**.

- poder preditivo esperado: 4/5
- implementabilidade: 4/5
- execução/custo: 4/5
- robustez por regime: 4/5
- risco de leakage: 4/5

