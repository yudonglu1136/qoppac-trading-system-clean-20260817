"""Convert raw model alpha scores into Rob-style forecast scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

FORECAST_CAP = 20.0
FORECAST_MODES = ("dense_rank", "top_long", "top_bottom")
SELECTION_INTERVALS = ("daily", "weekly", "monthly")


def forecast_group_columns(frame: pd.DataFrame) -> list[str]:
    group_cols = ["MarketKey", "Date"]
    if "model" in frame.columns:
        group_cols.insert(0, "model")
    if "seed" in frame.columns:
        insert_at = 1 if "model" in frame.columns else 0
        group_cols.insert(insert_at, "seed")
    return group_cols


def validate_forecast_selection(
    forecast_mode: str,
    selection_frac: float,
    selected_forecast: float,
    *,
    score_smoothing_span: int = 1,
    selection_interval: str = "daily",
    exit_frac: float | None = None,
) -> None:
    if forecast_mode not in FORECAST_MODES:
        raise ValueError(f"forecast_mode must be one of {FORECAST_MODES}; got {forecast_mode!r}")
    if not 0.0 < selection_frac <= 1.0:
        raise ValueError(f"selection_frac must be in (0, 1]; got {selection_frac}")
    if forecast_mode == "top_bottom" and selection_frac > 0.5:
        raise ValueError("top_bottom selection_frac is per side and must be <= 0.5")
    if not 0.0 < selected_forecast <= FORECAST_CAP:
        raise ValueError(f"selected_forecast must be in (0, {FORECAST_CAP}]; got {selected_forecast}")
    if score_smoothing_span < 1:
        raise ValueError(f"score_smoothing_span must be >= 1; got {score_smoothing_span}")
    if selection_interval not in SELECTION_INTERVALS:
        raise ValueError(f"selection_interval must be one of {SELECTION_INTERVALS}; got {selection_interval!r}")
    if exit_frac is not None:
        if not selection_frac <= exit_frac <= 1.0:
            raise ValueError(f"exit_frac must be in [{selection_frac}, 1]; got {exit_frac}")
        if forecast_mode == "top_bottom" and exit_frac > 0.5:
            raise ValueError("top_bottom exit_frac is per side and must be <= 0.5")


def score_group_columns(frame: pd.DataFrame) -> list[str]:
    group_cols = [column for column in forecast_group_columns(frame) if column != "Date"]
    return group_cols + ["Stock"]


def add_smoothed_score(frame: pd.DataFrame, score_smoothing_span: int) -> pd.DataFrame:
    out = frame.copy()
    out["score"] = out["raw_alpha"].astype(float)
    if score_smoothing_span == 1:
        return out
    order = out.sort_values(score_group_columns(out) + ["Date"]).index
    smoothed = out.loc[order].groupby(score_group_columns(out), sort=False, dropna=False)["raw_alpha"].transform(
        lambda values: values.ewm(span=score_smoothing_span, adjust=False, min_periods=1).mean()
    )
    out.loc[order, "score"] = smoothed.to_numpy(dtype=float)
    return out


def is_rebalance_date(dates: pd.Series, selection_interval: str) -> pd.Series:
    if selection_interval == "daily":
        return pd.Series(True, index=dates.index)
    periods = dates.dt.to_period("W-FRI" if selection_interval == "weekly" else "M")
    return periods.ne(periods.shift(1)).fillna(True)


def select_top_sticky(
    ranked: pd.DataFrame,
    previous: set[str],
    *,
    selection_count: int,
    exit_count: int,
) -> set[str]:
    stock_to_rank = ranked.set_index("Stock")["rank_order"].to_dict()
    keep = {
        stock
        for stock in previous
        if stock in stock_to_rank and int(stock_to_rank[stock]) <= exit_count
    }
    if len(keep) > selection_count:
        return set(
            ranked[ranked["Stock"].isin(keep)]
            .sort_values("rank_order")
            .head(selection_count)["Stock"]
            .astype(str)
        )
    additions = [
        stock
        for stock in ranked.sort_values("rank_order")["Stock"].astype(str)
        if stock not in keep
    ]
    selected = set(keep)
    selected.update(additions[: max(0, selection_count - len(selected))])
    return selected


def sparse_forecast_for_group(
    group: pd.DataFrame,
    *,
    forecast_mode: str,
    selection_frac: float,
    selected_forecast: float,
    selection_interval: str,
    exit_frac: float | None,
) -> pd.Series:
    out = pd.Series(np.nan, index=group.index, dtype=float)
    group = group.sort_values(["Date", "Stock"])
    rebalance_by_date = is_rebalance_date(group[["Date"]].drop_duplicates()["Date"], selection_interval)
    rebalance_dates = set(group[["Date"]].drop_duplicates().loc[rebalance_by_date.index[rebalance_by_date], "Date"])
    long_holdings: set[str] = set()
    short_holdings: set[str] = set()
    exit_selection_frac = selection_frac if exit_frac is None else exit_frac

    for date, daily in group.groupby("Date", sort=True):
        valid = daily.dropna(subset=["score"]).copy()
        if valid.empty:
            long_holdings = set()
            short_holdings = set()
            continue
        selection_count = max(1, int(np.ceil(len(valid) * selection_frac)))
        exit_count = max(selection_count, int(np.ceil(len(valid) * exit_selection_frac)))
        if date in rebalance_dates:
            long_ranked = valid[["Stock", "score"]].copy()
            long_ranked["rank_order"] = long_ranked["score"].rank(method="first", ascending=False).astype(int)
            long_holdings = select_top_sticky(
                long_ranked,
                long_holdings,
                selection_count=selection_count,
                exit_count=exit_count,
            )
            if forecast_mode == "top_bottom":
                short_ranked = valid[["Stock", "score"]].copy()
                short_ranked["rank_order"] = short_ranked["score"].rank(method="first", ascending=True).astype(int)
                short_holdings = select_top_sticky(
                    short_ranked,
                    short_holdings,
                    selection_count=selection_count,
                    exit_count=exit_count,
                )

        available = set(valid["Stock"].astype(str))
        long_holdings &= available
        short_holdings &= available
        out.loc[daily[daily["Stock"].astype(str).isin(long_holdings)].index] = selected_forecast
        if forecast_mode == "top_bottom":
            out.loc[daily[daily["Stock"].astype(str).isin(short_holdings)].index] = -selected_forecast
            out.loc[daily[daily["Stock"].astype(str).isin(long_holdings)].index] = selected_forecast
    return out


def raw_alpha_to_forecast(
    predictions: pd.DataFrame,
    *,
    forecast_mode: str = "dense_rank",
    selection_frac: float = 0.10,
    selected_forecast: float = 10.0,
    score_smoothing_span: int = 1,
    selection_interval: str = "daily",
    exit_frac: float | None = None,
) -> pd.DataFrame:
    required = {"Date", "Stock", "MarketKey", "raw_alpha"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    validate_forecast_selection(
        forecast_mode,
        selection_frac,
        selected_forecast,
        score_smoothing_span=score_smoothing_span,
        selection_interval=selection_interval,
        exit_frac=exit_frac,
    )

    frame = add_smoothed_score(predictions, score_smoothing_span)
    group_cols = forecast_group_columns(frame)
    group = frame.groupby(group_cols, dropna=False)["score"]
    ranks = group.rank(method="average")
    counts = group.transform("count")
    frame["rank"] = (ranks - 1.0) / (counts - 1.0).replace(0.0, pd.NA)
    frame.loc[counts <= 1, "rank"] = 0.5
    dense_forecast = (FORECAST_CAP * (2.0 * frame["rank"] - 1.0)).clip(-FORECAST_CAP, FORECAST_CAP)
    if forecast_mode == "dense_rank":
        frame["forecast"] = dense_forecast
        return frame

    state_group_cols = [column for column in group_cols if column != "Date"]
    forecasts = []
    for _key, group_frame in frame.groupby(state_group_cols, sort=False, dropna=False):
        forecasts.append(
            sparse_forecast_for_group(
                group_frame,
                forecast_mode=forecast_mode,
                selection_frac=selection_frac,
                selected_forecast=selected_forecast,
                selection_interval=selection_interval,
                exit_frac=exit_frac,
            )
        )
    frame["forecast"] = pd.concat(forecasts).sort_index()
    return frame


def forecast_matrix(
    predictions: pd.DataFrame,
    market_key: str,
    *,
    forecast_mode: str = "dense_rank",
    selection_frac: float = 0.10,
    selected_forecast: float = 10.0,
    score_smoothing_span: int = 1,
    selection_interval: str = "daily",
    exit_frac: float | None = None,
) -> pd.DataFrame:
    frame = raw_alpha_to_forecast(
        predictions,
        forecast_mode=forecast_mode,
        selection_frac=selection_frac,
        selected_forecast=selected_forecast,
        score_smoothing_span=score_smoothing_span,
        selection_interval=selection_interval,
        exit_frac=exit_frac,
    )
    frame = frame[frame["MarketKey"].eq(market_key)]
    if frame.duplicated(["Date", "Stock"]).any():
        frame = frame.groupby(["Date", "Stock"], as_index=False, dropna=False)["forecast"].mean()
    return frame.pivot(index="Date", columns="Stock", values="forecast").sort_index()
