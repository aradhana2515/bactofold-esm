# BactoFold-ESM

**BactoFold-ESM** is a public-data-only starter project for predicting bacterial / *E. coli* protein solubility, aggregation risk, and chaperone-rescue potential from protein sequence.

This repo is designed as a clean GitHub project you can build without using unpublished lab data.

## Core idea

Most “bacterial protein folding” labels are not direct folding-pathway measurements. They are practical proxies:

- soluble vs insoluble expression in *E. coli*
- quantitative solubility percentage
- aggregation-prone vs not aggregation-prone
- rescue by bacterial chaperones such as Trigger Factor, GroEL/ES, or DnaK/DnaJ/GrpE

The first milestone is intentionally simple:

> Train a sequence-feature baseline on public solubility data, then add protein language model embeddings as an optional second stage.

## Public datasets to use

### 1. eSOL

eSOL contains solubility and yield measurements for *E. coli* proteins synthesized in a chaperone-free PURE system. It also includes chaperone-effect measurements for aggregation-prone proteins.

Download URL:

```text
https://dbarchive.biosciencedbc.jp/data/esol/LATEST/esol.zip
```

### 2. PLM_Sol / UESolDS

PLM_Sol is a 2024 protein solubility prediction project based on ProtT5 embeddings and the updated *E. coli* solubility dataset, UESolDS. You can use it as the larger benchmark after the eSOL baseline is working.

GitHub:

```text
https://github.com/Violet969/PLM_Sol
```

### 3. NetSolP-1.0

NetSolP predicts solubility and purification usability for proteins expressed in *E. coli*. Use it as an external comparator.

GitHub:

```text
https://github.com/teevee112/NetSolP-1.0
```

## Recommended project roadmap

### Milestone 1: eSOL baseline

Goal: train a classical ML model using only interpretable sequence features.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Put eSOL zip here:
# data/raw/esol.zip

bactofold prepare-esol --input data/raw/esol.zip --output data/processed/esol_processed.csv
bactofold train --data data/processed/esol_processed.csv --target soluble_binary --model-out models/esol_baseline.joblib
bactofold evaluate --data data/processed/esol_processed.csv --model models/esol_baseline.joblib --target soluble_binary
```

### Milestone 2: predict from FASTA

```bash
bactofold predict --fasta examples/example.fasta --model models/esol_baseline.joblib --output reports/example_predictions.csv
```

### Milestone 3: chaperone-rescue task

Train labels such as:

- `tf_rescue_binary`
- `groe_rescue_binary`
- `kje_rescue_binary`
- `any_chaperone_rescue_binary`

```bash
bactofold train --data data/processed/esol_processed.csv --target any_chaperone_rescue_binary --model-out models/chaperone_rescue.joblib
```

### Milestone 4: protein language model embeddings

Add ESM-2 or ProtT5 embeddings and concatenate them with the interpretable features in `src/bactofold/embeddings.py`.

For a lightweight first pass, keep the feature baseline. Then add embeddings only after the full dataset pipeline is stable.

## Why this repo is useful

Generic solubility prediction is crowded. The more interesting biological angle is:

> Can protein sequence features predict which aggregation-prone bacterial proteins are rescued by specific chaperone systems?

That turns the project from “another solubility classifier” into a bacterial folding/chaperone-rescue benchmark.

## Repository structure

```text
bactofold-esm/
├── data/
│   ├── raw/
│   └── processed/
├── examples/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── scripts/
├── src/
│   └── bactofold/
├── tests/
├── README.md
└── pyproject.toml
```

## Citation notes

Please cite eSOL, PLM_Sol, and NetSolP if you use their data or compare against their models.
