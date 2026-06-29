# BactoFold-ESM project plan

## Phase 0 — Make the repo run

- [ ] Create virtual environment
- [ ] Install package with `pip install -e .[dev]`
- [ ] Run `pytest -q`
- [ ] Create toy dataset with `python scripts/make_toy_dataset.py`
- [ ] Train toy model

Commands:

```bash
python scripts/make_toy_dataset.py
bactofold train --data data/processed/toy_solubility.csv --target soluble_binary --model-out models/toy.joblib
bactofold predict --fasta examples/example.fasta --model models/toy.joblib --output reports/toy_predictions.csv
```

## Phase 1 — eSOL baseline

- [ ] Download eSOL
- [ ] Inspect columns
- [ ] Confirm whether sequence column is present
- [ ] If sequence column is absent, merge sequences externally using gene IDs/B numbers
- [ ] Train binary soluble/insoluble classifier
- [ ] Train quantitative solubility regressor if `% solubility` exists

## Phase 2 — chaperone rescue

- [ ] Create labels using `rescue_delta >= 15%`
- [ ] Train TF, GroE, KJE, and any-chaperone classifiers
- [ ] Compare features enriched in chaperone-rescuable proteins

## Phase 3 — PLM embeddings

- [ ] Add ESM-2 embeddings
- [ ] Compare feature-only vs ESM-only vs feature+ESM
- [ ] Evaluate calibration and cross-dataset generalization

## Phase 4 — GitHub polish

- [ ] Add README figures
- [ ] Add results table
- [ ] Add example FASTA prediction
- [ ] Add limitations section
- [ ] Add citation section
