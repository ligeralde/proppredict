import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from scipy.stats import sem, t
import subprocess
import pickle

# === CONFIG (defaults, can be overridden via CLI) ===
default_config = {
    "base_dir": "coadd_collins_cv_train_output",
    "dataset_path": "your_dataset.csv",
    "external_folds": 5,
    "internal_folds": 5,
    "smiles_col": "SMILES",
    "target_col": "ACTIVITY",
    "seed": 42
}


def mean_ci(scores, confidence=0.95):
    mean = np.mean(scores)
    stderr = sem(scores)
    ci_range = t.ppf((1 + confidence) / 2., len(scores) - 1) * stderr
    return mean, (mean - ci_range, mean + ci_range)


def run_internal_cv(train_val_df, ext_dir, ext_fold_idx, config):
    skf_internal = StratifiedKFold(n_splits=config["internal_folds"], shuffle=True, random_state=ext_fold_idx)
    best_score = -np.inf
    best_model_dir = None

    for int_fold_idx, (int_train_idx, int_val_idx) in enumerate(skf_internal.split(train_val_df, train_val_df[config["target_col"]])):
        int_dir = os.path.join(ext_dir, f"int_{int_fold_idx}")
        os.makedirs(int_dir, exist_ok=True)

        int_train_df = train_val_df.iloc[int_train_idx].reset_index(drop=True)
        int_val_df = train_val_df.iloc[int_val_idx].reset_index(drop=True)

        train_path = os.path.join(int_dir, "train.csv")
        val_path = os.path.join(int_dir, "val.csv")
        int_train_df.to_csv(train_path, index=False)
        int_val_df.to_csv(val_path, index=False)

        model_dir = os.path.join(int_dir, "model")
        subprocess.run([
        "chemprop_train",
        "--data_path", train_path,
            "--separate_val_path", val_path,
            "--dataset_type", "classification",
            "--smiles_column", config["smiles_col"],
            "--target_columns", config["target_col"],
            "--save_dir", model_dir,
            "--num_folds", "1",
            "--metric", "auc"
        ], check=True)

        try:
            scores = pd.read_json(os.path.join(model_dir, "val_scores.json"), typ='series')
            if scores['auc'] > best_score:
                best_score = scores['auc']
                best_model_dir = model_dir
        except Exception as e:
            print(f"❌ Failed to read AUC for int_{int_fold_idx}: {e}")

    if best_model_dir is None:
        raise RuntimeError(f"❌ No valid internal models found for {ext_dir} — check internal CV logs.")

    return best_model_dir


def refit_final_model(train_val_df, test_df, ext_dir, config):
    final_dir = os.path.join(ext_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    final_train_csv = os.path.join(final_dir, "train.csv")
    final_test_csv = os.path.join(final_dir, "test.csv")
    final_model_dir = os.path.join(final_dir, "model")

    train_val_df.to_csv(final_train_csv, index=False)
    test_df.to_csv(final_test_csv, index=False)

    subprocess.run([
        "chemprop_train",
        "--data_path", final_train_csv,
        "--separate_test_path", final_test_csv,
        "--save_preds",
        "--dataset_type", "classification",
        "--smiles_column", config["smiles_col"],
        "--target_columns", config["target_col"],
        "--save_dir", final_model_dir,
        "--num_folds", "1",
        "--metric", "auc"
    ], check=True)

    return os.path.join(final_model_dir, "test_preds.csv")


def evaluate_predictions(preds_csv, config):
    test_preds = pd.read_csv(preds_csv)
    y_true = test_preds[config["target_col"]]
    y_score = test_preds["prediction"]
    y_pred = (y_score >= 0.5).astype(int)

    return {
        "auc_roc": roc_auc_score(y_true, y_score),
        "auc_pr": average_precision_score(y_true, y_score),
        "f1": f1_score(y_true, y_pred)
    }


def run_nested_cv(config):
    df = pd.read_csv(config["dataset_path"])
    with open("external_cv_splits.pkl", "rb") as f:
        splits = pickle.load(f)

    external_metrics = []

    for ext_fold_idx, (train_val_idx, test_idx) in enumerate(splits):
        ext_dir = os.path.join(config["base_dir"], f"ext_{ext_fold_idx}")
        preds_csv = os.path.join(ext_dir, "final", "model", "test_preds.csv")

        if os.path.exists(preds_csv):
            print(f"✅ ext_{ext_fold_idx} already complete — skipping")
            continue

        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        print(f"\n🔁 Running internal CV for ext_{ext_fold_idx}...")
        best_model_dir = run_internal_cv(train_val_df, ext_dir, ext_fold_idx, config)

        print(f"🔄 Re-training best model for ext_{ext_fold_idx}...")
        preds_csv = refit_final_model(train_val_df, test_df, ext_dir, config)

        print(f"📊 Evaluating predictions for ext_{ext_fold_idx}...")
        metrics = evaluate_predictions(preds_csv, config)
        print(f"  ✅ Test Metrics: {metrics}")
        external_metrics.append(metrics)

    if external_metrics:
        external_df = pd.DataFrame(external_metrics)
        print("\n📊 Final Summary Across External Folds:")
        for metric in external_df.columns:
            mean, (ci_low, ci_high) = mean_ci(external_df[metric])
            print(f"{metric.upper()}: {mean:.3f} (95% CI: {ci_low:.3f}–{ci_high:.3f})")
    else:
        print("\n⚠️ No folds were successfully evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run nested CV with Chemprop")
    parser.add_argument("--base_dir", type=str, default=default_config["base_dir"])
    parser.add_argument("--dataset_path", type=str, default=default_config["dataset_path"])
    parser.add_argument("--external_folds", type=int, default=default_config["external_folds"])
    parser.add_argument("--internal_folds", type=int, default=default_config["internal_folds"])
    parser.add_argument("--smiles_col", type=str, default=default_config["smiles_col"])
    parser.add_argument("--target_col", type=str, default=default_config["target_col"])
    parser.add_argument("--seed", type=int, default=default_config["seed"])

    args = parser.parse_args()
    config = vars(args)

    run_nested_cv(config)
