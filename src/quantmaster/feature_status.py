from __future__ import annotations

from collections.abc import Iterable

# Single source of truth for public feature lifecycle status.
# Allowed values: approved, experimental, parked, rejected.
FEATURE_STATUS: dict[str, str] = {
    "absolute_return_autocorrelation": "approved",
    "amihud_illiquidity": "approved",
    "bar_range_position": "approved",
    "bipower_variation": "approved",
    "chande_momentum_oscillator": "approved",
    "close_location_value": "approved",
    "corwin_schultz_spread": "approved",
    "cusum_statistic": "approved",
    "downside_beta": "approved",
    "expected_shortfall": "approved",
    "fracdiff": "approved",
    "fractal_dimension_mincover": "approved",
    "garman_klass_volatility": "approved",
    "generalized_hurst_exponent": "approved",
    "har_rv": "approved",
    "har_rv_forecast": "approved",
    "harq_adjustment": "approved",
    "hurst_dfa": "approved",
    "information_discreteness": "approved",
    "intraday_range": "approved",
    "intraday_return": "approved",
    "jump_variation": "approved",
    "kaufman_efficiency_ratio": "approved",
    "ljung_box_stat": "approved",
    "log_volatility_increment": "approved",
    "lower_shadow_ratio": "approved",
    "market_efficiency_index": "approved",
    "max_drawdown_duration": "approved",
    "mean_reversion_half_life": "approved",
    "medrv": "approved",
    "minrv": "approved",
    "order_flow_imbalance": "approved",
    "order_flow_imbalance_range": "approved",
    "ornstein_uhlenbeck": "approved",
    "overnight_gap": "approved",
    "parkinson_volatility": "approved",
    "path_signature_features": "approved",
    "permutation_entropy": "approved",
    "price_acceleration": "approved",
    "price_volume_correlation": "approved",
    "realized_kurtosis": "approved",
    "realized_quarticity": "approved",
    "realized_roughness": "approved",
    "realized_semivariance": "approved",
    "realized_skewness": "approved",
    "realized_variance": "approved",
    "relative_spread_proxy": "approved",
    "relative_jump_contribution": "experimental",
    "return_autocorrelation": "approved",
    "rogers_satchell_volatility": "approved",
    "roll_spread": "approved",
    "rolling_beta": "approved",
    "runs_test_statistic": "approved",
    "rsi": "approved",
    "rvol": "approved",
    "shannon_entropy": "approved",
    "shar_components": "approved",
    "signed_jump_variation": "approved",
    "spread_zscore": "approved",
    "tail_risk_measure": "approved",
    "tick_imbalance_proxy": "approved",
    "time_series_momentum": "approved",
    "trend_intensity": "approved",
    "trend_strength_autocorr": "approved",
    "trend_strength_indicator": "approved",
    "upper_shadow_ratio": "approved",
    "value_at_risk_historical": "approved",
    "variance_ratio": "approved",
    "volatility_clustering": "approved",
    "volatility_ratio": "approved",
    "volume_volatility_ratio": "approved",
    "volume_weighted_close_location": "approved",
    "vpin_proxy": "approved",
    "vwap_deviation": "approved",
    "yang_zhang_volatility": "approved",
}

_VALID_STATUSES: set[str] = {"approved", "experimental", "parked", "rejected"}


def get_feature_status(name: str) -> str:
    status = FEATURE_STATUS.get(name, "approved")
    return status if status in _VALID_STATUSES else "approved"


def normalize_status_filter(include_statuses: Iterable[str] | None) -> set[str] | None:
    if include_statuses is None:
        return {"approved"}
    normalized = {str(s).strip().lower() for s in include_statuses}
    valid = {s for s in normalized if s in _VALID_STATUSES}
    return valid if valid else set()

