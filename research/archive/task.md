# Tasks: Implementação de Features (Sem Lookahead Bias)

Este arquivo é um checklist para implementar as features propostas no `review.md` **sem lookahead bias**.

Regra de “done” para **cada feature** (seguir `new_feature.md`):
- Implementar função no módulo correto em `src/quantmaster/features/`
- Adicionar testes em `tests/`
- Exportar em `src/quantmaster/features/__init__.py`
- Criar docs em `docs/features/<categoria>/<feature>.md`
- Atualizar `mkdocs.yml` (nav)

---

## Etapa 0 — Infra / Estrutura (pré-requisito)

- [x] Criar módulos que ainda não existem (se necessário):
  - [x] `src/quantmaster/features/microstructure.py`
  - [x] `src/quantmaster/features/regime.py`
  - [x] `src/quantmaster/features/risk.py`
  - [x] `src/quantmaster/features/returns.py`
  - [x] `src/quantmaster/features/trend.py`

- [x] Criar arquivos de teste (se necessário):
  - [x] `tests/test_microstructure.py`
  - [x] `tests/test_regime.py`
  - [x] `tests/test_risk.py`
  - [x] `tests/test_returns.py`
  - [x] `tests/test_trend.py`

- [x] Adicionar **template de teste anti-lookahead** (reutilizável) para features rolling:
  - [x] `tests/helpers.py::assert_no_lookahead`

---

# Etapa 1 — Volatility (range-based / OHLC)

**Módulo alvo:** `src/quantmaster/features/volatility.py`

- [x] `parkinson_volatility`
- [x] `garman_klass_volatility`
- [x] `rogers_satchell_volatility`
- [x] `intraday_range`
- [x] `volatility_ratio`

---

# Etapa 2 — Microstructure & Liquidity

**Módulo alvo:** `src/quantmaster/features/microstructure.py`

- [x] `amihud_illiquidity`
- [x] `roll_spread`
- [x] `corwin_schultz_spread`
- [x] `relative_spread_proxy`

---

# Etapa 3 — Jumps / Realized Measures / Decomposições

**Módulo alvo:** `src/quantmaster/features/volatility.py` (ou `statistical.py` se preferir agrupar realized measures)

- [x] `bipower_variation`
- [x] `jump_variation`
- [x] `realized_semivariance`
- [x] `signed_jump_variation`
- [x] `minrv`
- [x] `medrv` (implementar versão **sem lookahead**, ex: mediana backward-only)
- [x] `realized_quarticity`
- [x] `harq_adjustment`
- [x] `shar_components`

---

# Etapa 4 — Rough Volatility

**Módulo alvo:** `src/quantmaster/features/volatility.py`

- [x] `realized_roughness`
- [x] `log_volatility_increment`

---

# Etapa 5 — Entropy & Complexity

**Módulo alvo:** `src/quantmaster/features/statistical.py` (ou criar `entropy.py` se preferir)

- [x] `sample_entropy`
- [x] `permutation_entropy`
- [x] `approximate_entropy`
- [x] `cross_sample_entropy`

---

# Etapa 6 — Regime / Structural Breaks / Market Efficiency

**Módulo alvo:** `src/quantmaster/features/regime.py`

- [x] `cusum_statistic` (garantir μ/σ **backward-only**, e adicionar teste anti-lookahead)
- [x] `variance_ratio`
- [x] `market_efficiency_index`
- [x] `runs_test_statistic`

---

# Etapa 7 — Trend & Momentum

**Módulo alvo:**
- Trend: `src/quantmaster/features/trend.py`
- Momentum: `src/quantmaster/features/momentum.py`

- [x] `trend_intensity` (trend.py)
- [x] `time_series_momentum` (momentum.py ou trend.py)
- [x] `trend_strength_indicator` (trend.py)
- [x] `price_acceleration` (trend.py)
- [x] `kaufman_efficiency_ratio` (trend.py)
- [x] `chande_momentum_oscillator` (momentum.py)
- [x] `trend_strength_autocorr` (regime.py ou trend.py)

---

# Etapa 8 — Returns (decomposição intraday/overnight)

**Módulo alvo:** `src/quantmaster/features/returns.py`

- [x] `overnight_gap`
- [x] `intraday_return`

---

# Etapa 9 — Risk / Tail Risk

**Módulo alvo:** `src/quantmaster/features/risk.py`

- [x] `value_at_risk_historical`
- [x] `expected_shortfall`
- [x] `tail_risk_measure`
- [x] `max_drawdown_duration`

---

# Etapa 10 — ML-Ready (fatores, betas)

**Módulo alvo:** `src/quantmaster/features/statistical.py` (ou criar `factors.py` se preferir)

- [x] `rolling_beta`
- [x] `downside_beta`
- [x] `information_discreteness`

---

# Etapa 11 — Higher Moments

**Módulo alvo:** `src/quantmaster/features/statistical.py` (ou `risk.py` se preferir)

- [x] `realized_skewness`
- [x] `realized_kurtosis`

---

# Etapa 12 — Volume & Price Interaction

**Módulo alvo:** `src/quantmaster/features/volume.py`

- [x] `price_volume_correlation`
- [x] `volume_volatility_ratio`
- [x] `close_location_value`
- [x] `tick_imbalance_proxy`
- [x] `volume_weighted_close_location`
- [x] `order_flow_imbalance`

---

# Etapa 13 — Autocorrelation & Memory

**Módulo alvo:** `src/quantmaster/features/statistical.py`

- [x] `return_autocorrelation`
- [x] `absolute_return_autocorrelation`
- [x] `generalized_hurst_exponent`
- [x] `fractal_dimension_mincover`
- [x] `mean_reversion_half_life`

---

# Etapa 14 — Cointegration / Relative Value

**Módulo alvo:** `src/quantmaster/features/statistical.py` (ou criar `relative_value.py`)

- [x] `spread_zscore`

---

# Etapa 15 — Path Signatures (2020+)

**Módulo alvo:** `src/quantmaster/features/statistical.py` (ou criar `signatures.py`)

- [x] `path_signature_features` (definir dependência opcional + fallback/erro claro se ausente)

---

# Etapa 16 — Candlestick / Price Action (MQL5-derived)

**Módulo alvo:** `src/quantmaster/features/trend.py` (ou criar `price_action.py`)

- [x] `bar_range_position`
- [x] `body_to_range_ratio`
- [x] `upper_shadow_ratio`
- [x] `lower_shadow_ratio`

---

## Excluídas (Lookahead Bias CRÍTICO — não implementar na forma original)

- `leverage_effect_measure` (usa `RV_{t+horizon}`)
- `mean_reversion_strength` (usa `Δz_{t+1}`)
