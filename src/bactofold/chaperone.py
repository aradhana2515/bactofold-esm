from __future__ import annotations

import pandas as pd


def add_chaperone_delta_columns(
    df: pd.DataFrame,
    baseline_col: str,
    tf_col: str,
    groe_col: str,
    kje_col: str,
) -> pd.DataFrame:
    """Add delta-solubility columns for each chaperone system."""
    out = df.copy()
    out["delta_TF"] = out[tf_col] - out[baseline_col]
    out["delta_GroE"] = out[groe_col] - out[baseline_col]
    out["delta_KJE"] = out[kje_col] - out[baseline_col]
    return out


def add_chaperone_rescue_labels(
    df: pd.DataFrame,
    threshold: float = 20.0,
) -> pd.DataFrame:
    """Create binary rescue labels using a percentage-point improvement threshold."""
    out = df.copy()

    out["rescued_by_TF"] = out["delta_TF"] >= threshold
    out["rescued_by_GroE"] = out["delta_GroE"] >= threshold
    out["rescued_by_KJE"] = out["delta_KJE"] >= threshold

    out["rescued_by_any"] = (
        out[["rescued_by_TF", "rescued_by_GroE", "rescued_by_KJE"]]
        .any(axis=1)
    )

    delta_cols = ["delta_TF", "delta_GroE", "delta_KJE"]
    name_map = {
        "delta_TF": "TF",
        "delta_GroE": "GroE",
        "delta_KJE": "KJE",
    }

    out["best_chaperone"] = out[delta_cols].idxmax(axis=1).map(name_map)

    return out
