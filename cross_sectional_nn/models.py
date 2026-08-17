"""Baseline and neural models for cross-sectional alpha prediction."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline

from .config import ModelConfig
from .normalization import standard_preprocessor

TRAINING_OBJECTIVES = ("regression", "top_decile", "rank_quantile")
RANK_GROUP_COLUMNS = ["MarketKey", "Date"]

NON_FEATURE_COLUMNS = {
    "Date",
    "Stock",
    "Market",
    "MarketKey",
    "Sector",
    "target",
    "target_rank_pct",
    "target_rank_grade",
    "target_top_decile",
    "label_end_date",
}


@dataclass
class ModelResult:
    model_name: str
    seed: int | None
    predictions: pd.DataFrame
    metadata: dict[str, float | int | str]


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])]


def maybe_sample_train(frame: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame
    # Deterministic row sampling inside the already chronological training window.
    return frame.sample(n=max_rows, random_state=seed).sort_values(["Date", "MarketKey", "Stock"])


def maybe_sample_train_groups(frame: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame
    group_sizes = frame.groupby(RANK_GROUP_COLUMNS, dropna=False).size().reset_index(name="rows")
    rng = np.random.default_rng(seed)
    shuffled = group_sizes.iloc[rng.permutation(len(group_sizes))].reset_index(drop=True)
    selected = []
    total_rows = 0
    for row in shuffled.itertuples(index=False):
        selected.append((row.MarketKey, row.Date))
        total_rows += int(row.rows)
        if total_rows >= max_rows:
            break
    selected_frame = pd.DataFrame(selected, columns=RANK_GROUP_COLUMNS)
    sampled = frame.merge(selected_frame, on=RANK_GROUP_COLUMNS, how="inner")
    return sampled.sort_values(["Date", "MarketKey", "Stock"])


def add_cross_sectional_training_targets(
    frame: pd.DataFrame,
    *,
    selection_frac: float = 0.10,
    rank_bins: int = 10,
) -> pd.DataFrame:
    if not 0.0 < selection_frac <= 1.0:
        raise ValueError(f"selection_frac must be in (0, 1]; got {selection_frac}")
    if rank_bins < 2:
        raise ValueError(f"rank_bins must be at least 2; got {rank_bins}")
    out = frame.copy()
    group = out.groupby(RANK_GROUP_COLUMNS, dropna=False)["target"]
    counts = group.transform("count")
    ascending_rank = group.rank(method="first", ascending=True)
    pct = (ascending_rank - 1.0) / (counts - 1.0).replace(0.0, np.nan)
    out["target_rank_pct"] = pct.fillna(0.5).astype(float)

    top_count = pd.Series(np.ceil(counts.astype(float) * selection_frac), index=out.index).clip(lower=1.0)
    top_order = group.rank(method="first", ascending=False)
    out["target_top_decile"] = (top_order <= top_count).astype(int)
    grade = np.floor(out["target_rank_pct"].to_numpy(dtype=float) * rank_bins)
    out["target_rank_grade"] = np.clip(grade, 0, rank_bins - 1).astype(int)
    return out


def training_target_column(training_objective: str) -> str:
    if training_objective == "regression":
        return "target"
    if training_objective == "top_decile":
        return "target_top_decile"
    if training_objective == "rank_quantile":
        return "target_rank_pct"
    raise ValueError(f"training_objective must be one of {TRAINING_OBJECTIVES}; got {training_objective!r}")


def training_target_kind(training_objective: str) -> str:
    return "binary" if training_objective == "top_decile" else "regression"


def prediction_frame(validation: pd.DataFrame, raw_alpha: np.ndarray, model_name: str, seed: int | None) -> pd.DataFrame:
    columns = ["Date", "Stock", "Market", "MarketKey", "Sector", "target"]
    out = validation[columns].copy()
    out["raw_alpha"] = raw_alpha.astype(float)
    out["model"] = model_name
    out["seed"] = -1 if seed is None else seed
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    corr = pd.Series(y_pred).corr(pd.Series(y_true))
    return {"rmse": rmse, "mae": mae, "pearson": float(corr) if pd.notna(corr) else np.nan}


def top_decile_metrics(validation: pd.DataFrame, pred: np.ndarray, selection_frac: float) -> dict[str, float]:
    if "target_top_decile" not in validation.columns:
        return {}
    y_true = validation["target_top_decile"].to_numpy(dtype=int)
    if y_true.min() == y_true.max():
        auc = np.nan
    else:
        auc = float(roc_auc_score(y_true, pred))
    rows = []
    scored = validation[["Date", "MarketKey", "target_top_decile"]].copy()
    scored["prediction"] = pred
    for (_market, _date), group in scored.groupby(RANK_GROUP_COLUMNS, dropna=False):
        if group.empty:
            continue
        selected_count = max(1, int(np.ceil(len(group) * selection_frac)))
        selected = group.nlargest(selected_count, "prediction")
        rows.append(float(selected["target_top_decile"].mean()))
    precision = float(np.mean(rows)) if rows else np.nan
    base_rate = float(y_true.mean())
    return {"top_decile_auc": auc, "precision_at_selection": precision, "top_decile_base_rate": base_rate}


def rank_groups(frame: pd.DataFrame) -> list[int]:
    return frame.groupby(RANK_GROUP_COLUMNS, sort=False, dropna=False).size().astype(int).tolist()


def sort_for_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(RANK_GROUP_COLUMNS + ["Stock"]).reset_index(drop=True)


def train_ridge(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    config: ModelConfig,
    out_dir: Path,
    *,
    target_column: str = "target",
    training_objective: str = "regression",
    selection_frac: float = 0.10,
) -> ModelResult:
    pipeline = Pipeline(
        [
            ("preprocess", standard_preprocessor()),
            ("model", Ridge(alpha=config.ridge_alpha)),
        ]
    )
    x_train = train[features]
    y_train = train[target_column].to_numpy(dtype=float)
    x_val = validation[features]
    y_val = validation[target_column].to_numpy(dtype=float)
    pipeline.fit(x_train, y_train)
    pred = pipeline.predict(x_val)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump(pipeline, out_dir / "ridge.joblib")
    metadata = {
        "model": "ridge",
        "ridge_alpha": config.ridge_alpha,
        "training_objective": training_objective,
        "target_column": target_column,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "alpha_pearson": float(pd.Series(pred).corr(pd.Series(validation["target"].to_numpy(dtype=float)))),
        **regression_metrics(y_val, pred),
        **top_decile_metrics(validation, pred, selection_frac),
    }
    return ModelResult("ridge", None, prediction_frame(validation, pred, "ridge", None), metadata)


def train_lightgbm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    config: ModelConfig,
    seed: int,
    out_dir: Path,
    *,
    target_column: str = "target",
    target_kind: str = "regression",
    training_objective: str = "regression",
    selection_frac: float = 0.10,
) -> ModelResult:
    if target_kind == "binary":
        positives = max(1, int(train[target_column].sum()))
        negatives = max(1, len(train) - positives)
        model = LGBMClassifier(
            objective="binary",
            n_estimators=config.lightgbm_estimators,
            learning_rate=config.lightgbm_learning_rate,
            num_leaves=config.lightgbm_num_leaves,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=negatives / positives,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        model = LGBMRegressor(
            objective="huber",
            n_estimators=config.lightgbm_estimators,
            learning_rate=config.lightgbm_learning_rate,
            num_leaves=config.lightgbm_num_leaves,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    x_train = train[features]
    y_train = train[target_column].to_numpy(dtype=float)
    x_val = validation[features]
    y_val = validation[target_column].to_numpy(dtype=float)
    eval_metric = "auc" if target_kind == "binary" else "l2"
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric=eval_metric)
    pred = model.predict_proba(x_val)[:, 1] if target_kind == "binary" else model.predict(x_val)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump(model, out_dir / f"lightgbm_seed{seed}.joblib")
    metadata = {
        "model": "lightgbm",
        "seed": seed,
        "training_objective": training_objective,
        "target_column": target_column,
        "n_estimators": config.lightgbm_estimators,
        "learning_rate": config.lightgbm_learning_rate,
        "num_leaves": config.lightgbm_num_leaves,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "alpha_pearson": float(pd.Series(pred).corr(pd.Series(validation["target"].to_numpy(dtype=float)))),
        **regression_metrics(y_val, pred),
        **top_decile_metrics(validation, pred, selection_frac),
    }
    return ModelResult("lightgbm", seed, prediction_frame(validation, pred, "lightgbm", seed), metadata)


def train_lightgbm_ranker(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    config: ModelConfig,
    seed: int,
    out_dir: Path,
    *,
    selection_frac: float = 0.10,
    rank_bins: int = 10,
) -> ModelResult:
    train_sorted = sort_for_ranking(train)
    validation_sorted = sort_for_ranking(validation)
    train_group = rank_groups(train_sorted)
    validation_group = rank_groups(validation_sorted)
    median_group = int(np.median(train_group)) if train_group else 10
    top_n = max(1, int(np.ceil(median_group * selection_frac)))
    eval_at = tuple(sorted({max(1, top_n // 2), top_n, min(median_group, top_n * 2)}))
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=list(range(rank_bins)),
        n_estimators=config.lightgbm_estimators,
        learning_rate=config.lightgbm_learning_rate,
        num_leaves=config.lightgbm_num_leaves,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    x_train = train_sorted[features]
    y_train = train_sorted["target_rank_grade"].to_numpy(dtype=int)
    x_val = validation_sorted[features]
    y_val = validation_sorted["target_rank_grade"].to_numpy(dtype=int)
    model.fit(
        x_train,
        y_train,
        group=train_group,
        eval_set=[(x_val, y_val)],
        eval_group=[validation_group],
        eval_at=eval_at,
    )
    pred = model.predict(x_val)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump(model, out_dir / f"lightgbm_ranker_seed{seed}.joblib")
    metadata = {
        "model": "lightgbm_ranker",
        "seed": seed,
        "training_objective": "lambdarank",
        "target_column": "target_rank_grade",
        "rank_bins": rank_bins,
        "eval_at": ",".join(str(value) for value in eval_at),
        "n_estimators": config.lightgbm_estimators,
        "learning_rate": config.lightgbm_learning_rate,
        "num_leaves": config.lightgbm_num_leaves,
        "train_rows": len(train_sorted),
        "validation_rows": len(validation_sorted),
        "alpha_pearson": float(pd.Series(pred).corr(pd.Series(validation_sorted["target"].to_numpy(dtype=float)))),
        **regression_metrics(y_val.astype(float), pred),
        **top_decile_metrics(validation_sorted, pred, selection_frac),
    }
    return ModelResult(
        "lightgbm_ranker",
        seed,
        prediction_frame(validation_sorted, pred, "lightgbm_ranker", seed),
        metadata,
    )


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:
        return False


def train_mlp(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    config: ModelConfig,
    seed: int,
    out_dir: Path,
    *,
    target_column: str = "target",
    target_kind: str = "regression",
    training_objective: str = "regression",
    selection_frac: float = 0.10,
) -> ModelResult:
    import tensorflow as tf
    from keras import callbacks, layers, models, optimizers

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    preprocessor = standard_preprocessor()
    x_train = preprocessor.fit_transform(train[features]).astype("float32")
    y_train = train[target_column].to_numpy(dtype="float32")
    x_val = preprocessor.transform(validation[features]).astype("float32")
    y_val = validation[target_column].to_numpy(dtype="float32")

    output_activation = "sigmoid" if target_kind == "binary" else None
    model = models.Sequential(
        [
            layers.Input(shape=(len(features),)),
            layers.Dense(128),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(config.mlp_dropout),
            layers.Dense(64, activation="relu"),
            layers.Dropout(config.mlp_dropout),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation=output_activation),
        ]
    )
    loss = tf.keras.losses.BinaryCrossentropy() if target_kind == "binary" else tf.keras.losses.Huber()
    model.compile(optimizer=optimizers.Adam(learning_rate=config.mlp_learning_rate), loss=loss)
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=config.mlp_patience, restore_best_weights=True)
    class_weight = None
    if target_kind == "binary":
        positives = max(1, int(y_train.sum()))
        negatives = max(1, len(y_train) - positives)
        class_weight = {0: 1.0, 1: negatives / positives}
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.mlp_epochs,
        batch_size=config.mlp_batch_size,
        verbose=0,
        callbacks=[early_stop],
        class_weight=class_weight,
    )
    pred = model.predict(x_val, batch_size=config.mlp_batch_size, verbose=0).reshape(-1)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / f"mlp_seed{seed}.keras")
    dump(preprocessor, out_dir / f"mlp_seed{seed}_preprocessor.joblib")
    metadata = {
        "model": "mlp",
        "seed": seed,
        "training_objective": training_objective,
        "target_column": target_column,
        "parameter_count": int(model.count_params()),
        "epochs_requested": config.mlp_epochs,
        "early_stopping_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "learning_rate": config.mlp_learning_rate,
        "batch_size": config.mlp_batch_size,
        "dropout": config.mlp_dropout,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "final_val_loss": float(min(history.history["val_loss"])),
        "alpha_pearson": float(pd.Series(pred).corr(pd.Series(validation["target"].to_numpy(dtype=float)))),
        **regression_metrics(y_val, pred),
        **top_decile_metrics(validation, pred, selection_frac),
    }
    (out_dir / f"mlp_seed{seed}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return ModelResult("mlp", seed, prediction_frame(validation, pred, "mlp", seed), metadata)
