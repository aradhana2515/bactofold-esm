from __future__ import annotations

import argparse
import json
from pathlib import Path

from .datasets import prepare_public_table
from .modeling import evaluate_model, predict_fasta, train_model


def main() -> None:
    parser = argparse.ArgumentParser(prog="bactofold", description="Bacterial protein solubility/foldability toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare-esol", help="Prepare eSOL or another public solubility table")
    p_prepare.add_argument("--input", required=True, help="Input csv/tsv/xlsx/zip")
    p_prepare.add_argument("--output", required=True, help="Output processed csv")
    p_prepare.add_argument("--sequence-col", default=None, help="Protein sequence column name")
    p_prepare.add_argument("--id-col", default=None, help="Protein ID column name")
    p_prepare.add_argument("--target-col", default=None, help="Existing binary target column, if any")
    p_prepare.add_argument("--solubility-threshold", type=float, default=30.0)
    p_prepare.add_argument("--rescue-delta", type=float, default=15.0)

    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--target", required=True)
    p_train.add_argument("--model-out", required=True)
    p_train.add_argument("--task", choices=["classification", "regression"], default="classification")
    p_train.add_argument("--model-type", choices=["hgb", "logreg", "rf", "ridge"], default="hgb")

    p_eval = sub.add_parser("evaluate", help="Evaluate a saved model")
    p_eval.add_argument("--data", required=True)
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--target", required=True)

    p_predict = sub.add_parser("predict", help="Predict from FASTA")
    p_predict.add_argument("--fasta", required=True)
    p_predict.add_argument("--model", required=True)
    p_predict.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "prepare-esol":
        df = prepare_public_table(
            input_path=args.input,
            output_path=args.output,
            sequence_col=args.sequence_col,
            id_col=args.id_col,
            target_col=args.target_col,
            solubility_threshold=args.solubility_threshold,
            rescue_delta=args.rescue_delta,
        )
        print(json.dumps({"rows": len(df), "columns": list(df.columns), "output": args.output}, indent=2))

    elif args.command == "train":
        metrics = train_model(
            data_path=args.data,
            target=args.target,
            model_out=args.model_out,
            task=args.task,
            model_type=args.model_type,
        )
        print(json.dumps(metrics, indent=2))

    elif args.command == "evaluate":
        metrics = evaluate_model(data_path=args.data, model_path=args.model, target=args.target)
        print(json.dumps(metrics, indent=2))

    elif args.command == "predict":
        out = predict_fasta(fasta_path=args.fasta, model_path=args.model, output_path=args.output)
        print(json.dumps({"rows": len(out), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
