"""Optional protein language model embeddings.

This file is intentionally a stub so the baseline repo installs quickly.

Recommended next steps:
1. Add ESM-2 mean-pooled embeddings using `fair-esm` or Hugging Face transformers.
2. Save embeddings as parquet/csv with columns: id, emb_0, emb_1, ...
3. Concatenate embeddings with sequence features before training.

The simple feature baseline should work before this file is touched.
"""

from __future__ import annotations


def not_implemented_message() -> str:
    return (
        "Embeddings are optional. Start with sequence features; then add ESM-2 or ProtT5 "
        "mean-pooled embeddings once the public-data pipeline is stable."
    )
