from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import featurize_dataframe


def build_feature_matrix(df: pd.DataFrame, sequence_col: str = "sequence", id_col: str = "id") -> tuple[pd.DataFrame, list[str]]:
    feats = featurize_dataframe(df, sequence_col=sequence_col, id_col=id_col)
    feature_cols = [c for c in feats.columns if c != id_col]
    return feats, feature_cols


def make_classifier(model_type: str = "hgb") -> Pipeline:
    if model_type == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        scale = StandardScaler()
    elif model_type == "rf":
        clf = RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced_subsample", n_jobs=-1)
        scale = "passthrough"
    else:
        clf = HistGradientBoostingClassifier(random_state=42, learning_rate=0.07, max_iter=300)
        scale = "passthrough"

    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", scale),
            ("model", clf),
        ]
    )


def make_regressor(model_type: str = "hgb") -> Pipeline:
    if model_type == "ridge":
        reg = Ridge(alpha=1.0)
        scale = StandardScaler()
    else:
        reg = HistGradientBoostingRegressor(random_state=42, learning_rate=0.07, max_iter=300)
        scale = "passthrough"
    return Pipeline(steps=[("impute", SimpleImputer(strategy="median")), ("scale", scale), ("model", reg)])


def train_model(
    data_path: str | Path,
    target: str,
    model_out: str | Path,
    task: str = "classification",
    model_type: str = "hgb",
    test_size: float = 0.2,
) -> dict[str, float | str]:
    df = pd.read_csv(data_path)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Available columns: {list(df.columns)}")

    df = df.dropna(subset=[target, "sequence"]).copy()
    feats, feature_cols = build_feature_matrix(df)
    y = df.loc[feats.index, target].astype(float)

    valid = y.notna()
    feats = feats.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    X = feats[feature_cols]

    if task == "classification":
        y = y.astype(int)
        stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        model = make_classifier(model_type)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics: dict[str, float | str] = {
            "task": task,
            "target": target,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        }
        if hasattr(model, "predict_proba") and y_test.nunique() == 2:
            prob = model.predict_proba(X_test)[:, 1]
            metrics["auroc"] = float(roc_auc_score(y_test, prob))
            metrics["auprc"] = float(average_precision_score(y_test, prob))
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        model = make_regressor(model_type)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = {
            "task": task,
            "target": target,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "r2": float(r2_score(y_test, pred)),
            "rmse": float(mean_squared_error(y_test, pred, squared=False)),
            "mae": float(mean_absolute_error(y_test, pred)),
        }

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "target": target,
        "task": task,
        "model_type": model_type,
    }
    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_out)
    return metrics


def evaluate_model(data_path: str | Path, model_path: str | Path, target: str) -> dict[str, float | str]:
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    task = artifact.get("task", "classification")

    df = pd.read_csv(data_path).dropna(subset=[target, "sequence"]).copy()
    feats, _ = build_feature_matrix(df)
    X = feats[feature_cols]
    y = df.loc[feats.index, target].astype(float)

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    pred = model.predict(X)
    metrics: dict[str, float | str] = {"task": task, "target": target, "n": int(len(X))}
    if task == "classification":
        y_int = y.astype(int)
        metrics["accuracy"] = float(accuracy_score(y_int, pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_int, pred))
        if hasattr(model, "predict_proba") and y_int.nunique() == 2:
            prob = model.predict_proba(X)[:, 1]
            metrics["auroc"] = float(roc_auc_score(y_int, prob))
            metrics["auprc"] = float(average_precision_score(y_int, prob))
    else:
        metrics["r2"] = float(r2_score(y, pred))
        metrics["rmse"] = float(mean_squared_error(y, pred, squared=False))
        metrics["mae"] = float(mean_absolute_error(y, pred))
    return metrics


def predict_fasta(fasta_path: str | Path, model_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    from .fasta import read_fasta, records_to_dataframe

    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    task = artifact.get("task", "classification")
    target = artifact.get("target", "prediction")

    df = records_to_dataframe(read_fasta(fasta_path))
    feats, _ = build_feature_matrix(df)
    X = feats[feature_cols]

    out = df.copy()
    out[f"{target}_pred"] = model.predict(X)
    if task == "classification" and hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.shape[1] == 2:
            out[f"{target}_probability"] = probs[:, 1]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out
