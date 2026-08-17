"""Configuration defaults for cross-sectional stock forecast experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


UNIVERSE_KEYS = ("sp500", "eem", "efa")
MARKET_LABELS = {
    "sp500": "US",
    "eem": "EM",
    "efa": "Developed",
}


@dataclass(frozen=True)
class FeatureConfig:
    momentum_horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 40, 60, 120, 250)
    relative_horizons: tuple[int, ...] = (5, 10, 20, 60, 120, 250)
    sector_relative_horizons: tuple[int, ...] = (20, 60, 120)
    ewmac_pairs: tuple[tuple[int, int], ...] = ((4, 16), (8, 32), (16, 64), (32, 128), (64, 256))
    vol_windows: tuple[int, ...] = (10, 20, 60, 120)
    breakout_windows: tuple[int, ...] = (20, 60, 120, 250)
    drawdown_windows: tuple[int, ...] = (20, 60, 120, 250)
    target_horizon: int = 20
    target_vol_window: int = 60
    target_clip: float = 3.0
    min_history_days: int = 260
    add_cross_sectional_ranks: bool = True
    add_sector_ranks: bool = False
    include_market_id: bool = False
    include_sector_id: bool = False
    include_volume_features: bool = True
    include_ohlc_features: bool = False


@dataclass(frozen=True)
class ModelConfig:
    ridge_alpha: float = 10.0
    lightgbm_estimators: int = 250
    lightgbm_learning_rate: float = 0.03
    lightgbm_num_leaves: int = 31
    mlp_learning_rate: float = 0.001
    mlp_epochs: int = 20
    mlp_batch_size: int = 8192
    mlp_patience: int = 3
    mlp_dropout: float = 0.10
    seeds: tuple[int, ...] = (11, 23, 37, 41, 53)


@dataclass(frozen=True)
class ExperimentConfig:
    start: str = "2016-01-01"
    end: str = "2026-08-07"
    final_holdout_start: str = "2024-01-01"
    purge_days: int = 20
    universes: tuple[str, ...] = UNIVERSE_KEYS
    target_type: str = "market"
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
