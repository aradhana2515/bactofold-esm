from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FastaRecord:
    identifier: str
    sequence: str
    description: str = ""


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(seq: str) -> str:
    """Uppercase and keep standard amino-acid letters only."""
    seq = seq.upper().replace("*", "")
    return "".join([aa for aa in seq if aa in VALID_AA])


def read_fasta(path: str | Path) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    identifier: str | None = None
    description = ""
    chunks: list[str] = []

    with Path(path).open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    records.append(
                        FastaRecord(identifier=identifier, sequence=clean_sequence("".join(chunks)), description=description)
                    )
                header = line[1:].strip()
                parts = header.split(maxsplit=1)
                identifier = parts[0]
                description = parts[1] if len(parts) > 1 else ""
                chunks = []
            else:
                chunks.append(line)

    if identifier is not None:
        records.append(FastaRecord(identifier=identifier, sequence=clean_sequence("".join(chunks)), description=description))

    if not records:
        raise ValueError(f"No FASTA records found in {path}")

    return records


def records_to_dataframe(records: Iterable[FastaRecord]):
    import pandas as pd

    return pd.DataFrame(
        [{"id": r.identifier, "description": r.description, "sequence": r.sequence} for r in records]
    )
