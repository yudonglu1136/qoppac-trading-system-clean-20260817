"""Build pooled stock-day datasets for cross-sectional model training."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .features import build_feature_frames
from .targets import build_target_frame, label_end_dates
from .universe import UniverseData, annual_membership_counts, load_universe


def active_index(active: pd.DataFrame) -> pd.MultiIndex:
    stacked = active.stack()
    stacked = stacked[stacked]
    stacked.index = stacked.index.set_names(["Date", "Stock"])
    return stacked.index


def frames_to_dataset(data: UniverseData, config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = build_feature_frames(data, config.feature)
    target = build_target_frame(data, config.feature, config.target_type)
    base_index = active_index(data.active.loc[config.start : config.end])

    columns = {}
    for name, frame in features.frames.items():
        series = frame.stack().reindex(base_index)
        columns[name] = series
    panel = pd.DataFrame(columns, index=base_index)
    panel["target"] = target.stack().reindex(base_index)
    panel = panel.dropna(subset=["target"])

    end_date_by_date = label_end_dates(data.price.index, config.feature.target_horizon)
    dates = panel.index.get_level_values("Date")
    stocks = panel.index.get_level_values("Stock")
    panel.insert(0, "Date", dates)
    panel.insert(1, "Stock", stocks)
    panel.insert(2, "Market", data.market)
    panel.insert(3, "MarketKey", data.key)
    sector_series = data.sector.stack().reindex(panel.set_index(["Date", "Stock"]).index)
    panel.insert(4, "Sector", sector_series.to_numpy())
    panel["label_end_date"] = end_date_by_date.reindex(dates).to_numpy()
    panel = panel.reset_index(drop=True)
    return panel, features.metadata


def build_global_dataset(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universes = [load_universe(key, config.start, config.end) for key in config.universes]
    frames = []
    metadata = []
    for data in universes:
        print(f"Building features for {data.market} ({data.key})", flush=True)
        frame, feature_metadata = frames_to_dataset(data, config)
        print(f"Built {len(frame):,} stock-day rows for {data.market}", flush=True)
        frames.append(frame)
        feature_metadata = feature_metadata.copy()
        feature_metadata.insert(0, "market", data.market)
        metadata.append(feature_metadata)
    dataset = pd.concat(frames, ignore_index=True)
    feature_dictionary = pd.concat(metadata, ignore_index=True).drop_duplicates(subset=["feature"])
    membership_counts = annual_membership_counts(universes)
    return dataset, feature_dictionary, membership_counts


def save_dataset_artifacts(
    dataset: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    membership_counts: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out_dir / "stock_day_dataset.parquet", index=False)
    feature_dictionary.to_csv(out_dir / "feature_dictionary.csv", index=False)
    membership_counts.to_csv(out_dir / "membership_counts.csv", index=False)
