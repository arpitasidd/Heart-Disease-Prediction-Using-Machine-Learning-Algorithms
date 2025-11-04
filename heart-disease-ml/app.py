from __future__ import annotations
import argparse, os, json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.heart_disease.data import load_heart_dataframe
from src.heart_disease.features import build_preprocess
from src.heart_disease.models import get_model_specs
from src.heart_disease.evaluate import evaluate_and_save

DEFAULT_MODELS = ("knn","rf","dt","lr")

def cmd_train(args):
    df = load_heart_dataframe(data_dir=args.data_dir)
    pre, num_cols, cat_cols = build_preprocess(df)
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.outputs_dir, exist_ok=True)

    best = None
    best_name = None
    best_auc = -1.0

    model_specs = get_model_specs(tuple(args.models))

    for name, spec in model_specs.items():
        pipe = Pipeline([("pre", pre), ("model", spec.estimator)])
        pipe.fit(X_train, y_train)

        # Try predict_proba else decision_function else 0/1
        try:
            y_prob = pipe.predict_proba(X_test)[:,1]
        except Exception:
            try:
                y_prob = pipe.decision_function(X_test)
            except Exception:
                y_prob = pipe.predict(X_test)

        y_pred = pipe.predict(X_test)
        metrics = evaluate_and_save(name, y_test, y_prob, y_pred, out_dir=args.outputs_dir)

        # Save model
        model_path = os.path.join(args.models_dir, f"model_{name}.joblib")
        joblib.dump(pipe, model_path)

        auc = metrics.get("roc_auc", -1.0) or -1.0
        if auc > best_auc:
            best_auc = auc
            best = pipe
            best_name = name

    # Save best model alias
    if best is not None:
        best_path = os.path.join(args.models_dir, "best_model.joblib")
        joblib.dump(best, best_path)
        with open(os.path.join(args.outputs_dir, "best_model.txt"), "w") as f:
            f.write(best_name or "unknown")
        print(f"[OK] Trained. Best model: {best_name} (AUC={best_auc:.4f})")
    else:
        print("[WARN] No model trained.")

def cmd_evaluate(args):
    model_path = os.path.join(args.models_dir, "best_model.joblib")
    if not os.path.exists(model_path):
        raise SystemExit("No best_model.joblib found. Run 'train' first.")

    df = load_heart_dataframe(data_dir=args.data_dir)
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    pipe = joblib.load(model_path)
    try:
        y_prob = pipe.predict_proba(X)[:,1]
    except Exception:
        try:
            y_prob = pipe.decision_function(X)
        except Exception:
            y_prob = pipe.predict(X)
    y_pred = pipe.predict(X)
    metrics = evaluate_and_save("best_model", y, y_prob, y_pred, out_dir=args.outputs_dir)
    print(json.dumps(metrics, indent=2))

def cmd_predict(args):
    if not os.path.exists(args.input):
        raise SystemExit(f"Input file not found: {args.input}")
    model_path = os.path.join(args.models_dir, "best_model.joblib")
    if not os.path.exists(model_path):
        raise SystemExit("No best_model.joblib found. Run 'train' first.")

    pipe = joblib.load(model_path)
    df = pd.read_csv(args.input)
    preds = pipe.predict(df)
    try:
        probs = pipe.predict_proba(df)[:,1]
    except Exception:
        try:
            probs = pipe.decision_function(df)
        except Exception:
            probs = preds

    out_df = df.copy()
    out_df["prediction"] = preds
    out_df["probability"] = probs

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"[OK] Wrote predictions -> {args.output}")

def main():
    p = argparse.ArgumentParser(prog="heart-ml", description="Heart Disease Prediction CLI")
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("train", help="Train and evaluate models")
    pt.add_argument("--data-dir", default="data")
    pt.add_argument("--models-dir", default="models")
    pt.add_argument("--outputs-dir", default="outputs")
    pt.add_argument("--test-size", type=float, default=0.2)
    pt.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help="Subset of: knn rf dt lr")
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("evaluate", help="Evaluate the saved best model on all available data")
    pe.add_argument("--data-dir", default="data")
    pe.add_argument("--models-dir", default="models")
    pe.add_argument("--outputs-dir", default="outputs")
    pe.set_defaults(func=cmd_evaluate)

    pp = sub.add_parser("predict", help="Predict on a CSV with the same schema as training X")
    pp.add_argument("--input", required=True)
    pp.add_argument("--output", required=True)
    pp.add_argument("--models-dir", default="models")
    pp.set_defaults(func=cmd_predict)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
