"""Chronological walk-forward splits with target-window purging."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FoldSpec:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    purge_days: int


def development_folds(purge_days: int = 20) -> list[FoldSpec]:
    return [
        FoldSpec("dev_2020", pd.Timestamp("2016-01-01"), pd.Timestamp("2019-12-31"), pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"), purge_days),
        FoldSpec("dev_2021", pd.Timestamp("2016-01-01"), pd.Timestamp("2020-12-31"), pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"), purge_days),
        FoldSpec("dev_2022", pd.Timestamp("2016-01-01"), pd.Timestamp("2021-12-31"), pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"), purge_days),
        FoldSpec("dev_2023", pd.Timestamp("2016-01-01"), pd.Timestamp("2022-12-31"), pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"), purge_days),
    ]


def final_holdout_folds(purge_days: int = 20) -> list[FoldSpec]:
    return [
        FoldSpec("holdout_2024", pd.Timestamp("2016-01-01"), pd.Timestamp("2023-12-31"), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"), purge_days),
        FoldSpec("holdout_2025", pd.Timestamp("2016-01-01"), pd.Timestamp("2024-12-31"), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"), purge_days),
        FoldSpec("holdout_2026", pd.Timestamp("2016-01-01"), pd.Timestamp("2025-12-31"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31"), purge_days),
    ]


def purged_train_mask(frame: pd.DataFrame, fold: FoldSpec) -> pd.Series:
    dates = pd.to_datetime(frame["Date"])
    label_end = pd.to_datetime(frame["label_end_date"])
    return dates.between(fold.train_start, fold.train_end) & (label_end < fold.validation_start)


def validation_mask(frame: pd.DataFrame, fold: FoldSpec) -> pd.Series:
    dates = pd.to_datetime(frame["Date"])
    return dates.between(fold.validation_start, fold.validation_end)


def fold_summary(frame: pd.DataFrame, fold: FoldSpec) -> dict[str, int | str]:
    train = frame.loc[purged_train_mask(frame, fold)]
    val = frame.loc[validation_mask(frame, fold)]
    purge_start = train["Date"].max() if not train.empty else pd.NaT
    return {
        "fold": fold.name,
        "train_start": str(fold.train_start.date()),
        "train_end_after_purge": "" if pd.isna(purge_start) else str(pd.Timestamp(purge_start).date()),
        "validation_start": str(fold.validation_start.date()),
        "validation_end": str(fold.validation_end.date()),
        "purge_days": fold.purge_days,
        "train_observations": int(len(train)),
        "validation_observations": int(len(val)),
        "train_unique_stocks": int(train["Stock"].nunique()),
        "validation_unique_stocks": int(val["Stock"].nunique()),
        "markets": ",".join(sorted(val["Market"].dropna().unique())),
    }

