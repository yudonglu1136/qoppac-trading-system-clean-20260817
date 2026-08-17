"""Diagnostics for cross-sectional forecasts and predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .forecast_conversion import raw_alpha_to_forecast


def with_diagnostic_model(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["diagnostic_model"] = out["model"].astype(str)
    if "seed" in out.columns:
        seed = pd.to_numeric(out["seed"], errors="coerce")
        mask = out["model"].eq("mlp") & seed.notna() & seed.ne(-1)
        out.loc[mask, "diagnostic_model"] = "mlp_seed" + seed.loc[mask].astype(int).astype(str)
    return out


def daily_rank_ic(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    predictions = with_diagnostic_model(predictions)
    for (model, market, date), frame in predictions.groupby(["diagnostic_model", "Market", "Date"], dropna=False):
        if len(frame) < 10:
            continue
        pearson = frame["raw_alpha"].corr(frame["target"], method="pearson")
        spearman = frame["raw_alpha"].corr(frame["target"], method="spearman")
        rows.append(
            {
                "model": model,
                "market": market,
                "date": date,
                "n": len(frame),
                "pearson_ic": pearson,
                "rank_ic": spearman,
            }
        )
    return pd.DataFrame(rows)


def ic_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    daily = daily_rank_ic(predictions)
    rows = []
    groups = [("All", daily)]
    groups.extend((market, frame) for market, frame in daily.groupby("market"))
    for market, frame in groups:
        for model, model_frame in frame.groupby("model"):
            rank_ic = model_frame["rank_ic"].dropna()
            pearson_ic = model_frame["pearson_ic"].dropna()
            rows.append(
                {
                    "model": model,
                    "market": market,
                    "mean_daily_pearson_ic": pearson_ic.mean(),
                    "mean_daily_rank_ic": rank_ic.mean(),
                    "rank_ic_std": rank_ic.std(),
                    "rank_ic_ir": rank_ic.mean() / rank_ic.std() if rank_ic.std() else np.nan,
                    "positive_rank_ic_days": (rank_ic > 0.0).mean(),
                    "days": len(rank_ic),
                }
            )
    return pd.DataFrame(rows)


def bucket_test(predictions: pd.DataFrame, buckets: int = 5) -> pd.DataFrame:
    rows = []
    frame = with_diagnostic_model(predictions)
    frame["bucket"] = frame.groupby(["diagnostic_model", "MarketKey", "Date"], dropna=False)["raw_alpha"].transform(
        lambda values: pd.qcut(values.rank(method="first"), buckets, labels=False, duplicates="drop") + 1
    )
    for (model, market, bucket), group in frame.groupby(["diagnostic_model", "Market", "bucket"], dropna=False):
        rows.append(
            {
                "model": model,
                "market": market,
                "bucket": int(bucket),
                "avg_future_target": group["target"].mean(),
                "observations": len(group),
            }
        )
    return pd.DataFrame(rows)


def forecast_distribution(
    predictions: pd.DataFrame,
    *,
    forecast_mode: str = "dense_rank",
    selection_frac: float = 0.10,
    selected_forecast: float = 10.0,
    score_smoothing_span: int = 1,
    selection_interval: str = "daily",
    exit_frac: float | None = None,
) -> pd.DataFrame:
    frame = with_diagnostic_model(
        raw_alpha_to_forecast(
            predictions,
            forecast_mode=forecast_mode,
            selection_frac=selection_frac,
            selected_forecast=selected_forecast,
            score_smoothing_span=score_smoothing_span,
            selection_interval=selection_interval,
            exit_frac=exit_frac,
        )
    )
    rows = []
    for (model, market), group in frame.groupby(["diagnostic_model", "Market"], dropna=False):
        rows.append(
            {
                "model": model,
                "market": market,
                "forecast_mean": group["forecast"].mean(),
                "forecast_median": group["forecast"].median(),
                "forecast_std": group["forecast"].std(),
                "positive_forecast_share": (group["forecast"] > 0.0).mean(),
                "non_null_forecast_share": group["forecast"].notna().mean(),
            }
        )
    return pd.DataFrame(rows)


def forecast_persistence(
    predictions: pd.DataFrame,
    *,
    forecast_mode: str = "dense_rank",
    selection_frac: float = 0.10,
    selected_forecast: float = 10.0,
    score_smoothing_span: int = 1,
    selection_interval: str = "daily",
    exit_frac: float | None = None,
) -> pd.DataFrame:
    frame = with_diagnostic_model(
        raw_alpha_to_forecast(
            predictions,
            forecast_mode=forecast_mode,
            selection_frac=selection_frac,
            selected_forecast=selected_forecast,
            score_smoothing_span=score_smoothing_span,
            selection_interval=selection_interval,
            exit_frac=exit_frac,
        )
    )
    rows = []
    for (model, market), group in frame.groupby(["diagnostic_model", "Market"], dropna=False):
        if group.duplicated(["Date", "Stock"]).any():
            group = group.groupby(["Date", "Stock"], as_index=False, dropna=False)["forecast"].mean()
        matrix = group.pivot(index="Date", columns="Stock", values="forecast").sort_index()
        active = matrix.notna()

        def active_retention(lag: int) -> float:
            previous = active.shift(lag)
            previous_count = previous.sum(axis=1).replace(0, np.nan)
            retained = (active & previous).sum(axis=1)
            return (retained / previous_count).mean()

        def active_jaccard(lag: int) -> float:
            previous = active.shift(lag)
            union = (active | previous).sum(axis=1).replace(0, np.nan)
            intersection = (active & previous).sum(axis=1)
            return (intersection / union).mean()

        corr_1 = matrix.corrwith(matrix.shift(1), axis=1).mean()
        corr_5 = matrix.corrwith(matrix.shift(5), axis=1).mean()
        corr_20 = matrix.corrwith(matrix.shift(20), axis=1).mean()
        rows.append(
            {
                "model": model,
                "market": market,
                "forecast_autocorr_1d": corr_1,
                "forecast_autocorr_5d": corr_5,
                "forecast_autocorr_20d": corr_20,
                "active_retention_1d": active_retention(1),
                "active_retention_5d": active_retention(5),
                "active_retention_20d": active_retention(20),
                "active_jaccard_1d": active_jaccard(1),
                "active_jaccard_5d": active_jaccard(5),
                "active_jaccard_20d": active_jaccard(20),
            }
        )
    return pd.DataFrame(rows)


def forecast_correlation(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> pd.DataFrame:
    rows = []
    merged = left.merge(
        right,
        on=["Date", "Stock", "MarketKey", "Market"],
        suffixes=(f"_{left_name}", f"_{right_name}"),
    )
    for market, frame in merged.groupby("Market"):
        rows.append(
            {
                "market": market,
                "left": left_name,
                "right": right_name,
                "forecast_corr": frame[f"forecast_{left_name}"].corr(frame[f"forecast_{right_name}"]),
            }
        )
    return pd.DataFrame(rows)
