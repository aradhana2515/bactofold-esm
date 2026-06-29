from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


SEQUENCE_CANDIDATES = [
    "sequence", "Sequence", "aa_sequence", "protein_sequence", "Protein sequence",
    "Amino acid sequence", "amino_acid_sequence", "seq",
]
ID_CANDIDATES = ["id", "ID", "Entry", "protein_id", "JW_ID", "B number", "b_number", "gene", "Gene name K-12"]
SOLUBILITY_CANDIDATES = ["Solubility (%)", "Solubility", "solubility", "solubility_percent", "sol"]


def _normalize_colname(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    exact = {str(c): c for c in df.columns}
    for cand in candidates:
        if cand in exact:
            return exact[cand]
    norm = {_normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_colname(cand)
        if key in norm:
            return norm[key]
    return None


def _read_member_from_zip(zip_path: Path, member: str) -> pd.DataFrame:
    suffix = Path(member).suffix.lower()
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as handle:
            if suffix in {".xlsx", ".xls"}:
                return pd.read_excel(handle)
            if suffix in {".csv"}:
                return pd.read_csv(handle)
            if suffix in {".tsv", ".txt"}:
                # sep=None lets pandas sniff comma/tab/space in Python engine.
                return pd.read_csv(handle, sep=None, engine="python")
    raise ValueError(f"Unsupported file type inside zip: {member}")


def read_first_table_from_zip(zip_path: str | Path) -> pd.DataFrame:
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
    table_members = [m for m in members if Path(m).suffix.lower() in {".csv", ".tsv", ".txt", ".xlsx", ".xls"}]
    if not table_members:
        raise ValueError(f"No readable table files found in {zip_path}")

    errors = []
    for member in table_members:
        try:
            df = _read_member_from_zip(zip_path, member)
            if len(df) > 0 and len(df.columns) > 1:
                df.attrs["source_member"] = member
                return df
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{member}: {exc}")
    raise ValueError("Could not read a table from zip. Errors: " + " | ".join(errors))


def add_binary_labels_from_esol_columns(df: pd.DataFrame, solubility_threshold: float = 30.0, rescue_delta: float = 15.0) -> pd.DataFrame:
    """Create usable classification labels from eSOL-like columns.

    - soluble_binary: 1 if solubility percent >= solubility_threshold.
    - aggregation_prone_binary: 1 if baseline chaperone-free solubility < threshold.
    - tf/groe/kje_rescue_binary: 1 if chaperone solubility - baseline >= rescue_delta.
    """
    out = df.copy()

    sol_col = _find_column(out, SOLUBILITY_CANDIDATES)
    if sol_col is not None:
        out["solubility_percent"] = pd.to_numeric(out[sol_col], errors="coerce")
        out["soluble_binary"] = (out["solubility_percent"] >= solubility_threshold).astype("Int64")

    # Chaperone labels if available.
    normalized = {_normalize_colname(c): c for c in out.columns}
    minus_col = normalized.get("minus_sol") or normalized.get("minus_sol_percent") or normalized.get("minus_sol_")
    tf_col = normalized.get("tf_sol") or normalized.get("tf_sol_percent") or normalized.get("tf_sol_")
    groe_col = normalized.get("groe_sol") or normalized.get("groe_sol_percent") or normalized.get("groe_sol_")
    kje_col = normalized.get("kje_sol") or normalized.get("kje_sol_percent") or normalized.get("kje_sol_")

    if minus_col:
        out["minus_sol_percent"] = pd.to_numeric(out[minus_col], errors="coerce")
        out["aggregation_prone_binary"] = (out["minus_sol_percent"] < solubility_threshold).astype("Int64")

    for label_name, col in [("tf", tf_col), ("groe", groe_col), ("kje", kje_col)]:
        if minus_col and col:
            out[f"{label_name}_sol_percent"] = pd.to_numeric(out[col], errors="coerce")
            out[f"{label_name}_rescue_delta"] = out[f"{label_name}_sol_percent"] - out["minus_sol_percent"]
            out[f"{label_name}_rescue_binary"] = (out[f"{label_name}_rescue_delta"] >= rescue_delta).astype("Int64")

    rescue_cols = [c for c in ["tf_rescue_binary", "groe_rescue_binary", "kje_rescue_binary"] if c in out.columns]
    if rescue_cols:
        vals = out[rescue_cols].astype("float")
        out["any_chaperone_rescue_binary"] = (vals.max(axis=1) == 1).astype("Int64")

    return out


def prepare_public_table(
    input_path: str | Path,
    output_path: str | Path,
    sequence_col: str | None = None,
    id_col: str | None = None,
    target_col: str | None = None,
    solubility_threshold: float = 30.0,
    rescue_delta: float = 15.0,
) -> pd.DataFrame:
    """Prepare a public solubility table from csv/tsv/xlsx or a zip containing one."""
    input_path = Path(input_path)
    if input_path.suffix.lower() == ".zip":
        df = read_first_table_from_zip(input_path)
    elif input_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path, sep=None, engine="python")

    seq_col = sequence_col or _find_column(df, SEQUENCE_CANDIDATES)
    found_id_col = id_col or _find_column(df, ID_CANDIDATES)

    if found_id_col is None:
        df["id"] = [f"protein_{i}" for i in range(len(df))]
        found_id_col = "id"

    if seq_col is not None:
        df = df.rename(columns={seq_col: "sequence", found_id_col: "id"})
    else:
        # Keep the table and write an actionable error file for the user.
        raise ValueError(
            "Could not find a protein sequence column. Add a column called 'sequence', "
            "or pass --sequence-col. If using eSOL and the archive lacks sequences, merge sequences from UniProt/NCBI first using B number/JW_ID."
        )

    if target_col and target_col in df.columns:
        df["soluble_binary"] = pd.to_numeric(df[target_col], errors="coerce").astype("Int64")
    else:
        df = add_binary_labels_from_esol_columns(
            df, solubility_threshold=solubility_threshold, rescue_delta=rescue_delta
        )

    # Clean sequence and remove empty sequences.
    df["sequence"] = df["sequence"].astype(str).str.upper().str.replace(r"[^ACDEFGHIKLMNPQRSTVWY]", "", regex=True)
    df = df[df["sequence"].str.len() > 0].copy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
