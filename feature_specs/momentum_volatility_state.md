# momentum_volatility_state

## Hipótese

Momentum tende a degradar em regimes de volatilidade em aceleração e tende a funcionar melhor em regimes de volatilidade estável/declinante.
A feature combina sinal de momentum com estado de volatilidade para capturar esse efeito de regime.

## Fórmula

Com preços \(P_t\):

- Momentum log-retorno:
  \[
  M_t = \log(P_t) - \log(P_{t-k})
  \]
- Volatilidade de curto prazo:
  \[
  \sigma_t = \text{std}\big(\Delta \log(P)\big)_{w\_vol}
  \]
- Estado de volatilidade (z-score backward-only):
  \[
  Z^\sigma_t = \frac{\sigma_t - \mu^\sigma_t}{s^\sigma_t}, \quad
  \mu^\sigma_t = \text{mean}(\sigma)_{w\_state}, \quad
  s^\sigma_t = \text{std}(\sigma)_{w\_state}
  \]
- Weight de regime:
  \[
  W_t = \frac{1}{1 + \max(Z^\sigma_t, 0)}
  \]
- Feature final:
  \[
  F_t = M_t \cdot W_t
  \]

Intuição: quando volatilidade está acima do normal recente (\(Z^\sigma_t>0\)), o peso reduz a exposição do momentum.

## Entradas

- `data: pd.DataFrame | pd.Series`
- Se `DataFrame`, requer `price_col` (default: `close`)

## Parâmetros

- `mom_window: int = 20`
- `vol_window: int = 20`
- `state_window: int = 60`
- `price_col: str = "close"`
- `eps: float = 1e-12`

Restrições:
- `mom_window >= 1`
- `vol_window >= 2`
- `state_window >= 5`

## Saída

- `pd.Series` alinhada ao índice de entrada
- `name`: `momentum_volatility_state_<mom_window>_<vol_window>_<state_window>`

## Regras de borda

- Janela insuficiente: `NaN`
- Preços não positivos para log: ponto inválido vira `NaN`
- Desvio padrão muito baixo (`<= eps`): z-score de vol vira `0` localmente
- Divisões protegidas com `eps`

## Prova de causalidade temporal (no-lookahead)

- \(M_t\) usa apenas \(P_t\) e \(P_{t-k}\), ambos observáveis em \(t\).
- \(\sigma_t\), \(\mu^\sigma_t\), \(s^\sigma_t\) usam rolling backward-only até \(t\).
- Não usa `shift(-k)`, `center=True` ou qualquer janela forward.

## Política no-repaint

- O valor em \(t\) depende só do histórico até \(t\).
- Adicionar dados futuros não altera valores já emitidos para timestamps passados.

## Plano de teste

1. Contrato:
   - tipo `Series`, índice preservado, `name` correta
2. Sanidade:
   - quando vol acelera, magnitude de \(F_t\) reduz vs momentum cru em média local
3. No-lookahead:
   - `assert_no_lookahead` em janela intermediária
4. No-repaint:
   - recomputar após append de dados e comparar histórico antigo
5. Bordas:
   - preços constantes/baixíssima vol
   - NaNs no input

## Nota de priorização do discovery

Score total: **21/25**

- poder preditivo esperado: 4/5
- implementabilidade: 5/5
- execução/custo: 4/5
- robustez por regime: 5/5
- risco de leakage: 3/5

## Risco principal

Se calibrada agressivamente, pode suprimir demais o momentum em períodos de transição de regime.
Mitigar com grid de parâmetros pequeno e validação walk-forward.

