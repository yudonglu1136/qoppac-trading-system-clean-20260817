#!/usr/bin/env python3
"""Run pooled cross-sectional ML forecasts through the existing Rob stock system."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_rob_style_stock_backtest as rob_stock  # noqa: E402

from cross_sectional_nn.backtest import (  # noqa: E402
    combined_forecast,
    forecast_long_from_matrix,
    model_forecast_matrix,
    original_rob_forecast_matrix,
    run_forecast_backtest,
    save_forecast_matrix,
)
from cross_sectional_nn.config import ExperimentConfig, FeatureConfig, ModelConfig  # noqa: E402
from cross_sectional_nn.dataset import build_global_dataset, save_dataset_artifacts  # noqa: E402
from cross_sectional_nn.diagnostics import (  # noqa: E402
    bucket_test,
    forecast_distribution,
    forecast_persistence,
    ic_summary,
)
from cross_sectional_nn.forecast_conversion import FORECAST_MODES, SELECTION_INTERVALS, raw_alpha_to_forecast  # noqa: E402
from cross_sectional_nn.models import (  # noqa: E402
    TRAINING_OBJECTIVES,
    add_cross_sectional_training_targets,
    feature_columns,
    maybe_sample_train,
    maybe_sample_train_groups,
    train_lightgbm,
    train_lightgbm_ranker,
    train_mlp,
    train_ridge,
    training_target_column,
    training_target_kind,
)
from cross_sectional_nn.walk_forward import (  # noqa: E402
    development_folds,
    final_holdout_folds,
    fold_summary,
    purged_train_mask,
    validation_mask,
)


DEFAULT_OUT = ROOT / "backtests" / "cross_sectional_nn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-08-07")
    parser.add_argument("--universes", nargs="+", default=["sp500", "eem", "efa"], choices=sorted(rob_stock.SUPPORTED_UNIVERSES))
    parser.add_argument("--target-type", choices=["market", "sector"], default="market")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge", "lightgbm", "mlp"],
        choices=["ridge", "lightgbm", "lightgbm_ranker", "mlp"],
    )
    parser.add_argument("--training-objective", choices=TRAINING_OBJECTIVES, default="regression")
    parser.add_argument("--rank-bins", type=int, default=10)
    parser.add_argument("--fold-set", choices=["development", "holdout", "both"], default="development")
    parser.add_argument("--folds", nargs="*", help="Optional fold names to run, e.g. dev_2023 holdout_2024.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37, 41, 53])
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--vol-target", type=float, default=0.10)
    parser.add_argument("--weight-mode", default="sector_equal", choices=sorted(rob_stock.WEIGHT_MODES))
    parser.add_argument("--idm-method", choices=["fixed", "rob_estimated"], default="rob_estimated")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--use-existing-predictions", type=Path, default=None)
    parser.add_argument("--no-sector-relative-features", action="store_true")
    parser.add_argument("--no-volume-features", action="store_true")
    parser.add_argument("--enable-ohlc-features", action="store_true")
    parser.add_argument("--forecast-mode", choices=FORECAST_MODES, default="dense_rank")
    parser.add_argument("--selection-frac", type=float, default=0.10)
    parser.add_argument("--selected-forecast", type=float, default=10.0)
    parser.add_argument("--score-smoothing-span", type=int, default=1)
    parser.add_argument("--selection-interval", choices=SELECTION_INTERVALS, default="daily")
    parser.add_argument("--exit-frac", type=float, default=None)
    return parser.parse_args()


def filter_requested_predictions(predictions: pd.DataFrame, requested_models: list[str]) -> pd.DataFrame:
    allowed = set(requested_models)
    if "mlp" in allowed:
        allowed.add("mlp_ensemble")
    filtered = predictions[predictions["model"].isin(allowed)].copy()
    if filtered.empty:
        raise ValueError(f"No predictions match requested models: {sorted(requested_models)}")
    return filtered


def load_existing_run_context(path: Path) -> tuple[list[str], list[dict], list[dict]]:
    source_dir = path.parent
    features: list[str] = []
    fold_rows: list[dict] = []
    model_rows: list[dict] = []
    metadata_path = source_dir / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        features = list(metadata.get("config", {}).get("features", []))
    fold_path = source_dir / "fold_summary.csv"
    if fold_path.exists():
        fold_rows = pd.read_csv(fold_path).to_dict("records")
    model_path = source_dir / "model_fit_summary.csv"
    if model_path.exists():
        model_rows = pd.read_csv(model_path).to_dict("records")
    return features, fold_rows, model_rows


def experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    feature_config = FeatureConfig(
        sector_relative_horizons=() if args.no_sector_relative_features else FeatureConfig().sector_relative_horizons,
        include_volume_features=not args.no_volume_features,
        include_ohlc_features=args.enable_ohlc_features,
    )
    return ExperimentConfig(
        start=args.start,
        end=args.end,
        universes=tuple(args.universes),
        target_type=args.target_type,
        feature=feature_config,
        model=ModelConfig(seeds=tuple(args.seeds)),
    )


def load_or_build_dataset(config: ExperimentConfig, out_dir: Path, rebuild: bool) -> tuple[pd.DataFrame, list[str]]:
    dataset_path = out_dir / "stock_day_dataset.parquet"
    feature_path = out_dir / "feature_dictionary.csv"
    if dataset_path.exists() and feature_path.exists() and not rebuild:
        dataset = pd.read_parquet(dataset_path)
    else:
        dataset, feature_dictionary, membership_counts = build_global_dataset(config)
        save_dataset_artifacts(dataset, feature_dictionary, membership_counts, out_dir)
    dataset["Date"] = pd.to_datetime(dataset["Date"])
    dataset["label_end_date"] = pd.to_datetime(dataset["label_end_date"])
    features = feature_columns(dataset)
    return dataset, features


def selected_folds(args: argparse.Namespace):
    folds = []
    if args.fold_set in {"development", "both"}:
        folds.extend(development_folds())
    if args.fold_set in {"holdout", "both"}:
        folds.extend(final_holdout_folds())
    if args.folds:
        wanted = set(args.folds)
        folds = [fold for fold in folds if fold.name in wanted]
    return folds


def ensemble_predictions(predictions: pd.DataFrame, model_name: str) -> pd.DataFrame:
    model_frame = predictions[predictions["model"].eq(model_name)]
    if model_frame.empty:
        return pd.DataFrame()
    group_cols = ["Date", "Stock", "Market", "MarketKey", "Sector", "target"]
    out = model_frame.groupby(group_cols, as_index=False, dropna=False)["raw_alpha"].mean()
    out["model"] = f"{model_name}_ensemble"
    out["seed"] = -1
    return out


def train_fold_models(
    dataset: pd.DataFrame,
    features: list[str],
    fold,
    args: argparse.Namespace,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, list[dict]]:
    train = dataset.loc[purged_train_mask(dataset, fold)].copy()
    validation = dataset.loc[validation_mask(dataset, fold)].copy()
    train = train.dropna(subset=["target"])
    validation = validation.dropna(subset=["target"])
    if train.empty or validation.empty:
        return pd.DataFrame(), []
    train = add_cross_sectional_training_targets(train, selection_frac=args.selection_frac, rank_bins=args.rank_bins)
    validation = add_cross_sectional_training_targets(validation, selection_frac=args.selection_frac, rank_bins=args.rank_bins)

    model_dir = args.out_dir / "models" / fold.name
    predictions = []
    metadata = []
    train_sample = maybe_sample_train(train, args.max_train_rows, seed=101)
    train_group_sample = maybe_sample_train_groups(train, args.max_train_rows, seed=101)
    target_column = training_target_column(args.training_objective)
    target_kind = training_target_kind(args.training_objective)

    if "ridge" in args.models:
        result = train_ridge(
            train_sample,
            validation,
            features,
            config.model,
            model_dir / "ridge",
            target_column=target_column,
            training_objective=args.training_objective,
            selection_frac=args.selection_frac,
        )
        result.metadata["fold"] = fold.name
        predictions.append(result.predictions)
        metadata.append(result.metadata)

    if "lightgbm" in args.models:
        seed = config.model.seeds[0]
        result = train_lightgbm(
            train_sample,
            validation,
            features,
            config.model,
            seed,
            model_dir / "lightgbm",
            target_column=target_column,
            target_kind=target_kind,
            training_objective=args.training_objective,
            selection_frac=args.selection_frac,
        )
        result.metadata["fold"] = fold.name
        predictions.append(result.predictions)
        metadata.append(result.metadata)

    if "lightgbm_ranker" in args.models:
        seed = config.model.seeds[0]
        result = train_lightgbm_ranker(
            train_group_sample,
            validation,
            features,
            config.model,
            seed,
            model_dir / "lightgbm_ranker",
            selection_frac=args.selection_frac,
            rank_bins=args.rank_bins,
        )
        result.metadata["fold"] = fold.name
        predictions.append(result.predictions)
        metadata.append(result.metadata)

    if "mlp" in args.models:
        seed_predictions = []
        for seed in config.model.seeds:
            result = train_mlp(
                train_sample,
                validation,
                features,
                config.model,
                seed,
                model_dir / "mlp",
                target_column=target_column,
                target_kind=target_kind,
                training_objective=args.training_objective,
                selection_frac=args.selection_frac,
            )
            result.metadata["fold"] = fold.name
            seed_predictions.append(result.predictions)
            metadata.append(result.metadata)
        mlp_seed_frame = pd.concat(seed_predictions, ignore_index=True)
        predictions.append(mlp_seed_frame)
        predictions.append(ensemble_predictions(mlp_seed_frame, "mlp"))

    return pd.concat(predictions, ignore_index=True), metadata


def run_prediction_diagnostics(predictions: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    converted = raw_alpha_to_forecast(
        predictions,
        forecast_mode=args.forecast_mode,
        selection_frac=args.selection_frac,
        selected_forecast=args.selected_forecast,
        score_smoothing_span=args.score_smoothing_span,
        selection_interval=args.selection_interval,
        exit_frac=args.exit_frac,
    )
    converted.to_parquet(out_dir / "oos_predictions_with_forecast.parquet", index=False)
    converted.head(50_000).to_csv(out_dir / "oos_predictions_sample.csv", index=False)
    ic_summary(predictions).to_csv(out_dir / "ic_summary.csv", index=False)
    bucket_test(predictions, buckets=5).to_csv(out_dir / "quintile_test.csv", index=False)
    bucket_test(predictions, buckets=10).to_csv(out_dir / "decile_test.csv", index=False)
    forecast_distribution(
        predictions,
        forecast_mode=args.forecast_mode,
        selection_frac=args.selection_frac,
        selected_forecast=args.selected_forecast,
        score_smoothing_span=args.score_smoothing_span,
        selection_interval=args.selection_interval,
        exit_frac=args.exit_frac,
    ).to_csv(out_dir / "forecast_distribution.csv", index=False)
    forecast_persistence(
        predictions,
        forecast_mode=args.forecast_mode,
        selection_frac=args.selection_frac,
        selected_forecast=args.selected_forecast,
        score_smoothing_span=args.score_smoothing_span,
        selection_interval=args.selection_interval,
        exit_frac=args.exit_frac,
    ).to_csv(out_dir / "forecast_persistence.csv", index=False)
    for model in sorted(predictions["model"].unique()):
        save_forecast_matrix(
            predictions,
            out_dir,
            model,
            forecast_mode=args.forecast_mode,
            selection_frac=args.selection_frac,
            selected_forecast=args.selected_forecast,
            score_smoothing_span=args.score_smoothing_span,
            selection_interval=args.selection_interval,
            exit_frac=args.exit_frac,
        )


def run_rob_forecast_correlations(predictions: pd.DataFrame, args: argparse.Namespace, start: str, end: str) -> pd.DataFrame:
    converted = raw_alpha_to_forecast(
        predictions,
        forecast_mode=args.forecast_mode,
        selection_frac=args.selection_frac,
        selected_forecast=args.selected_forecast,
        score_smoothing_span=args.score_smoothing_span,
        selection_interval=args.selection_interval,
        exit_frac=args.exit_frac,
    )
    rows = []
    for key in args.universes:
        rob_matrix = original_rob_forecast_matrix(key, start, end)
        rob_long = forecast_long_from_matrix(rob_matrix, key, rob_stock.SUPPORTED_UNIVERSES[key], "original_rob")
        rob_long = rob_long.rename(columns={"forecast": "rob_forecast"})
        for model, model_frame in converted[converted["MarketKey"].eq(key)].groupby("model"):
            merged = model_frame.merge(rob_long[["Date", "Stock", "rob_forecast"]], on=["Date", "Stock"], how="inner")
            rows.append(
                {
                    "market": key,
                    "model": model,
                    "corr_to_original_rob": merged["forecast"].corr(merged["rob_forecast"]),
                    "observations": len(merged),
                }
            )
    return pd.DataFrame(rows)


def run_backtests(predictions: pd.DataFrame, args: argparse.Namespace, start: str, end: str) -> pd.DataFrame:
    out_dir = args.out_dir / "portfolio"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    model_names = sorted(name for name in predictions["model"].unique() if not name == "mlp")
    # Use the ensemble as the neural production forecast; individual seeds remain diagnostics.
    if "mlp_ensemble" in model_names:
        model_names = [name for name in model_names if not name.startswith("mlp")] + ["mlp_ensemble"]
    elif "mlp" in predictions["model"].unique():
        model_names = sorted(set(model_names + ["mlp"]))
    neural_combo_model = "mlp_ensemble" if "mlp_ensemble" in model_names else "mlp" if "mlp" in model_names else None

    for key in args.universes:
        rob_forecast = original_rob_forecast_matrix(key, start, end)
        _daily, stats = run_forecast_backtest(
            key,
            rob_forecast,
            "original_rob",
            start,
            end,
            out_dir,
            vol_target=args.vol_target,
            weight_mode=args.weight_mode,
            idm_method=args.idm_method,
        )
        rows.append(stats)

        neural_for_combo = None
        for model in model_names:
            forecast = model_forecast_matrix(
                predictions,
                key,
                model,
                forecast_mode=args.forecast_mode,
                selection_frac=args.selection_frac,
                selected_forecast=args.selected_forecast,
                score_smoothing_span=args.score_smoothing_span,
                selection_interval=args.selection_interval,
                exit_frac=args.exit_frac,
            )
            if forecast.empty:
                continue
            _daily, stats = run_forecast_backtest(
                key,
                forecast,
                model,
                start,
                end,
                out_dir,
                vol_target=args.vol_target,
                weight_mode=args.weight_mode,
                idm_method=args.idm_method,
            )
            rows.append(stats)
            if model == neural_combo_model:
                neural_for_combo = forecast

        if neural_for_combo is not None:
            combo = combined_forecast(rob_forecast, neural_for_combo, neural_weight=0.20).where(neural_for_combo.notna())
            _daily, stats = run_forecast_backtest(
                key,
                combo,
                "rob80_mlp20",
                start,
                end,
                out_dir,
                vol_target=args.vol_target,
                weight_mode=args.weight_mode,
                idm_method=args.idm_method,
            )
            rows.append(stats)

    stats_frame = pd.DataFrame(rows)
    stats_frame.to_csv(out_dir / "portfolio_stats.csv", index=False)
    return stats_frame


def write_summary(
    out_dir: Path,
    args: argparse.Namespace,
    config: ExperimentConfig,
    features: list[str],
    fold_rows: list[dict],
    model_rows: list[dict],
    portfolio_stats: pd.DataFrame | None,
) -> None:
    lines = [
        "# Cross-Sectional Neural Forecast",
        "",
        "## Integration Boundary",
        "",
        "- Neural/Ridge/LightGBM modules only generate forecasts.",
        "- Existing Rob volatility estimation, volatility target, IDM, instrument weights, sizing, buffers, costs, PnL, and performance calculations are reused unchanged.",
        "- Raw model alpha is converted into Rob-style forecast scores before entering the unchanged Rob risk system.",
        "- Optional forecast stabilizers smooth raw scores, rebalance less frequently, and keep existing names until they leave the exit buffer.",
        "",
        "## Data Availability",
        "",
        "- Available local fields: adjusted close, benchmark adjusted close, yfinance adjusted OHLCV, FX conversion for EEM/EFA close-price panels, point-in-time membership snapshots, sector labels.",
        "- Missing locally: shares outstanding and point-in-time market cap. Those feature families are skipped rather than fabricated.",
        "- SPY 2016-2020 uses annual S&P 500 membership with equal-weight fallback before cached SPY holdings begin; historical sector labels are parsed from same-date Wikipedia revisions.",
        "- EEM/EFA sector labels and weights come from archived iShares holdings snapshots; OHLCV ticker coverage is limited by free Yahoo symbol support.",
        "",
        "## Run Settings",
        "",
        f"- Universes: {', '.join(args.universes)}.",
        f"- Target: {config.feature.target_horizon} trading-day `{args.target_type}` excess return divided by trailing stock volatility and clipped to +/-{config.feature.target_clip}.",
        f"- Models requested: {', '.join(args.models)}.",
        f"- Training objective: {args.training_objective}.",
        f"- Rank bins: {args.rank_bins}.",
        f"- Fold set: {args.fold_set}.",
        f"- Max train rows per fold: {args.max_train_rows if args.max_train_rows else 'none'}.",
        f"- Feature count: {len(features)}.",
        f"- Sector-relative price features enabled: {not args.no_sector_relative_features}.",
        f"- Volume/liquidity features enabled: {not args.no_volume_features}.",
        f"- OHLC/intraday features enabled: {args.enable_ohlc_features}.",
        f"- Forecast conversion mode: {args.forecast_mode}.",
        f"- Selection fraction: {args.selection_frac:.1%}.",
        f"- Exit buffer fraction: {args.exit_frac:.1%}." if args.exit_frac is not None else "- Exit buffer fraction: none.",
        f"- Selection interval: {args.selection_interval}.",
        f"- Score smoothing span: {args.score_smoothing_span}.",
        f"- Selected forecast level: {args.selected_forecast:.1f}.",
        "",
        "## Folds",
        "",
        "| Fold | Train Start | Train End After Purge | Validation Start | Validation End | Train Obs | Validation Obs | Markets |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in fold_rows:
        lines.append(
            f"| {row['fold']} | {row['train_start']} | {row['train_end_after_purge']} | {row['validation_start']} | {row['validation_end']} | {row['train_observations']} | {row['validation_observations']} | {row['markets']} |"
        )

    if model_rows:
        lines += [
            "",
            "## Model Fits",
            "",
            "| Fold | Model | Seed | Objective | Train Rows | Validation Rows | Alpha Pearson | Fit Pearson | Top 10% AUC | Precision@Selection | Params/Epoch |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in model_rows:
            extra = ""
            if row.get("model") == "mlp":
                extra = f"{row.get('parameter_count')} params / epoch {row.get('early_stopping_epoch')}"
            alpha_pearson = row.get("alpha_pearson", np.nan)
            top_decile_auc = row.get("top_decile_auc", np.nan)
            precision_at_selection = row.get("precision_at_selection", np.nan)
            lines.append(
                f"| {row.get('fold')} | {row.get('model')} | {row.get('seed', '')} | {row.get('training_objective', args.training_objective)} | {row.get('train_rows')} | {row.get('validation_rows')} | {alpha_pearson:.4f} | {row.get('pearson', np.nan):.4f} | {top_decile_auc:.4f} | {precision_at_selection:.2%} | {extra} |"
            )

    ic_path = out_dir / "diagnostics" / "ic_summary.csv"
    if ic_path.exists():
        ic = pd.read_csv(ic_path)
        lines += [
            "",
            "## IC Summary",
            "",
            "| Model | Market | Mean Daily Rank IC | IC IR | Positive IC Days | Days |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for row in ic.itertuples(index=False):
            lines.append(
                f"| {row.model} | {row.market} | {row.mean_daily_rank_ic:.4f} | {row.rank_ic_ir:.2f} | {row.positive_rank_ic_days:.1%} | {row.days} |"
            )

    if portfolio_stats is not None and not portfolio_stats.empty:
        lines += [
            "",
            "## Portfolio Stats",
            "",
            "| Market | Strategy | CAGR | Vol | Sharpe | MDD | Avg Gross | Ann Cost | Cost Mult |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in portfolio_stats.itertuples(index=False):
            lines.append(
                f"| {row.market} | {row.strategy} | {row.cagr:.2%} | {row.vol:.2%} | {row.sharpe:.2f} | {row.mdd:.2%} | {row.avg_gross_exposure:.2%} | {row.avg_cost_annual:.2%} | {row.cost_multiplier:.1f} |"
            )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = experiment_config(args)

    if args.use_existing_predictions is not None:
        predictions = pd.read_parquet(args.use_existing_predictions)
        predictions["Date"] = pd.to_datetime(predictions["Date"])
        predictions = filter_requested_predictions(predictions, args.models)
        predictions.to_parquet(args.out_dir / "oos_predictions_raw.parquet", index=False)
        features, fold_rows, model_rows = load_existing_run_context(args.use_existing_predictions)
        run_prediction_diagnostics(predictions, args.out_dir / "diagnostics", args)

        portfolio_stats = None
        if not args.skip_backtest:
            validation_start = predictions["Date"].min().date().isoformat()
            validation_end = predictions["Date"].max().date().isoformat()
            corr = run_rob_forecast_correlations(predictions, args, validation_start, validation_end)
            corr.to_csv(args.out_dir / "diagnostics" / "corr_to_original_rob.csv", index=False)
            portfolio_stats = run_backtests(predictions, args, validation_start, validation_end)

        metadata = {
            "config": {
                "start": config.start,
                "end": config.end,
                "target_type": config.target_type,
                "universes": list(config.universes),
                "features": features,
            },
            "args": vars(args) | {
                "out_dir": str(args.out_dir),
                "use_existing_predictions": str(args.use_existing_predictions),
            },
        }
        (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        write_summary(args.out_dir, args, config, features, fold_rows, model_rows, portfolio_stats)
        print(args.out_dir / "summary.md")
        return

    dataset, features = load_or_build_dataset(config, args.out_dir, args.rebuild_dataset)
    folds = selected_folds(args)
    fold_rows = [fold_summary(dataset, fold) for fold in folds]
    pd.DataFrame(fold_rows).to_csv(args.out_dir / "fold_summary.csv", index=False)

    all_predictions = []
    model_rows = []
    for fold in folds:
        predictions, metadata = train_fold_models(dataset, features, fold, args, config)
        if not predictions.empty:
            predictions["fold"] = fold.name
            all_predictions.append(predictions)
        model_rows.extend(metadata)

    if not all_predictions:
        raise RuntimeError("No predictions were generated")

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_parquet(args.out_dir / "oos_predictions_raw.parquet", index=False)
    pd.DataFrame(model_rows).to_csv(args.out_dir / "model_fit_summary.csv", index=False)
    run_prediction_diagnostics(predictions, args.out_dir / "diagnostics", args)

    portfolio_stats = None
    if not args.skip_backtest:
        validation_start = min(fold.validation_start for fold in folds).date().isoformat()
        validation_end = max(fold.validation_end for fold in folds).date().isoformat()
        corr = run_rob_forecast_correlations(predictions, args, validation_start, validation_end)
        corr.to_csv(args.out_dir / "diagnostics" / "corr_to_original_rob.csv", index=False)
        portfolio_stats = run_backtests(predictions, args, validation_start, validation_end)

    metadata = {
        "config": {
            "start": config.start,
            "end": config.end,
            "target_type": config.target_type,
            "universes": list(config.universes),
            "features": features,
        },
        "args": vars(args) | {"out_dir": str(args.out_dir)},
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_summary(args.out_dir, args, config, features, fold_rows, model_rows, portfolio_stats)
    print(args.out_dir / "summary.md")


if __name__ == "__main__":
    main()
