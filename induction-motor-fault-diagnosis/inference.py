"""
Simple inference helper for the saved model produced by `train_fault_diagnosis.py`.
Usage:
    python inference.py --model models/best_model.joblib --input-csv sample_features.csv

The script expects a features CSV with the same column names produced by the training script.
"""
import argparse
import sys
from pathlib import Path
import joblib
import pandas as pd


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    obj = joblib.load(path)
    # Some saved files store the model directly or wrapped in a dict
    if isinstance(obj, dict) and 'model' in obj:
        return obj['model']
    return obj


def main(argv):
    p = argparse.ArgumentParser(description="Run inference with a saved model")
    p.add_argument("--model", type=Path, required=True, help="Path to model joblib file")
    p.add_argument("--input-csv", type=Path, required=True, help="CSV file with features")
    args = p.parse_args(argv)

    model = load_model(args.model)
    df = pd.read_csv(args.input_csv)

    if 'label' in df.columns:
        X = df.drop(columns=['label'])
    else:
        X = df

    preds = model.predict(X)
    out = df.copy()
    out['predicted_label'] = preds
    print(out[['predicted_label']].value_counts())
    print('\nSample predictions:')
    print(out.head())


if __name__ == '__main__':
    main(sys.argv[1:])
