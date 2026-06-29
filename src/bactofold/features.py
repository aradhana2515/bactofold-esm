from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"

# Average residue masses after peptide-bond water loss approximation.
AA_MASS = {
    "A": 89.09, "C": 121.16, "D": 133.10, "E": 147.13, "F": 165.19,
    "G": 75.07, "H": 155.16, "I": 131.17, "K": 146.19, "L": 131.17,
    "M": 149.21, "N": 132.12, "P": 115.13, "Q": 146.15, "R": 174.20,
    "S": 105.09, "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19,
}

# Kyte-Doolittle hydropathy.
HYDROPATHY = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9,
    "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3,
    "P": -1.6, "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5,
    "K": -3.9, "R": -4.5,
}

POSITIVE = set("KRH")
NEGATIVE = set("DE")
CHARGED = POSITIVE | NEGATIVE
POLAR = set("STNQCYW")
AROMATIC = set("FWY")
ALIPHATIC = set("AILV")
TINY = set("AGSC")
PROLINE = set("P")
CYSTEINE = set("C")


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def shannon_entropy(seq: str) -> float:
    n = len(seq)
    if n == 0:
        return 0.0
    counts = Counter(seq)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def simple_pseudo_pI(seq: str) -> float:
    """A deliberately lightweight pI proxy, not a replacement for full Bjellqvist pI.

    We keep this dependency-light for the baseline. It is good enough as a rough feature.
    Replace with Bio.SeqUtils.IsoelectricPoint for production if desired.
    """
    n = len(seq)
    if n == 0:
        return 7.0
    frac_pos = sum(seq.count(a) for a in POSITIVE) / n
    frac_neg = sum(seq.count(a) for a in NEGATIVE) / n
    return float(np.clip(7.0 + 12.0 * (frac_pos - frac_neg), 3.0, 11.5))


def sequence_features(seq: str) -> dict[str, float]:
    seq = "".join([aa for aa in seq.upper() if aa in AA])
    n = len(seq)
    counts = Counter(seq)

    if n == 0:
        raise ValueError("Empty or invalid protein sequence")

    feats: dict[str, float] = {
        "length": float(n),
        "mw_da": float(sum(AA_MASS[a] for a in seq) - 18.015 * max(n - 1, 0)),
        "mean_hydropathy": float(np.mean([HYDROPATHY[a] for a in seq])),
        "entropy": shannon_entropy(seq),
        "pseudo_pI": simple_pseudo_pI(seq),
    }

    groups = {
        "frac_positive": POSITIVE,
        "frac_negative": NEGATIVE,
        "frac_charged": CHARGED,
        "frac_polar": POLAR,
        "frac_aromatic": AROMATIC,
        "frac_aliphatic": ALIPHATIC,
        "frac_tiny": TINY,
        "frac_proline": PROLINE,
        "frac_cysteine": CYSTEINE,
    }
    for name, group in groups.items():
        feats[name] = _safe_div(sum(counts[a] for a in group), n)

    for aa in AA:
        feats[f"aa_{aa}"] = _safe_div(counts[aa], n)

    # Crude aggregation-associated descriptors.
    feats["hydrophobic_minus_charged"] = feats["frac_aliphatic"] + feats["frac_aromatic"] - feats["frac_charged"]
    feats["cys_per_100aa"] = 100.0 * feats["frac_cysteine"]
    feats["pro_per_100aa"] = 100.0 * feats["frac_proline"]
    feats["charge_balance"] = feats["frac_positive"] - feats["frac_negative"]

    return feats


def featurize_dataframe(df: pd.DataFrame, sequence_col: str = "sequence", id_col: str = "id") -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        seq = str(row[sequence_col])
        if not seq or seq == "nan":
            continue
        try:
            feats = sequence_features(seq)
        except ValueError:
            continue
        feats[id_col] = row.get(id_col, None)
        rows.append(feats)
    return pd.DataFrame(rows)
