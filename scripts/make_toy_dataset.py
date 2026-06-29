"""Create a tiny synthetic dataset to smoke-test the pipeline.

This is NOT a scientific dataset. It only verifies that the code runs.
"""

from pathlib import Path
import pandas as pd

rows = [
    {"id": "toy_1", "sequence": "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPDWQNY", "soluble_binary": 1},
    {"id": "toy_2", "sequence": "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDW", "soluble_binary": 1},
    {"id": "toy_3", "sequence": "MFFFFFFFFFFFFFFLLLLLLLLLLLLVVVVVVVVVVAAAA", "soluble_binary": 0},
    {"id": "toy_4", "sequence": "MLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL", "soluble_binary": 0},
    {"id": "toy_5", "sequence": "MADEEKLPPGWEKRMSRSSGRVYYFNHITNASQWERPSGNSS", "soluble_binary": 1},
    {"id": "toy_6", "sequence": "MVVVVVVVVVVVVLLLLLLLLLLLFFFFFFFGGGG", "soluble_binary": 0},
]

out = Path("data/processed/toy_solubility.csv")
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(out)
