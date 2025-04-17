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
    "metric": "auc",
    "num_epochs": 30,
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

def safe_run_chemprop_train(train_path, val_path, test_path, save_dir, config):
    command = [
        "chemprop_train",
        "--data_path", train_path,
        "--separate_val_path", val_path,
        "--separate_test_path", test_path,
        "--save_preds",
        "--dataset_type", "classification",
        "--smiles_column", config["smiles_col"],
        "--target_columns", config["target_col"],
        "--save_dir", save_dir,
        "--num_folds", "1",
        "--metric", config["metric"],
        "--epochs", str(config["num_epochs"])
    ]
    subprocess.run(command, check=True)

def run_internal_cv(train_val_df, ext_dir, ext_fold_idx, config):
    skf_internal = StratifiedKFold(n_splits=config["internal_folds"], shuffle=True, random_state=ext_fold_idx)
    best_score = -np.inf
    best_model_dir = None

    for int_fold_idx, (int_train_idx, int_val_idx) in enumerate(skf_internal.split(train_val_df, train_val_df[config["target_col"]])):
        model_dir = os.path.join(ext_dir, f"int_{int_fold_idx}", "model")
        val_scores_path = os.path.join(model_dir, "val_scores.json")
        if os.path.exists(val_scores_path):
            print(f"⏭️  int_{int_fold_idx} already trained — skipping to val score loading")
            try:
                scores = pd.read_json(val_scores_path, typ='series')
                if scores[config["metric"]] > best_score:
                    best_score = scores[config["metric"]]
                    best_model_dir = model_dir
            except Exception as e:
                print(f"❌ Failed to read existing score for int_{int_fold_idx}: {e}")
            continue
        int_dir = os.path.join(ext_dir, f"int_{int_fold_idx}")
        os.makedirs(int_dir, exist_ok=True)

        int_train_df = train_val_df.iloc[int_train_idx].reset_index(drop=True)
        int_val_df = train_val_df.iloc[int_val_idx].reset_index(drop=True)

        train_path = os.path.join(int_dir, "train.csv")
        val_path = os.path.join(int_dir, "val.csv")
        int_train_df.to_csv(train_path, index=False)
        int_val_df.to_csv(val_path, index=False)

        model_dir = os.path.join(int_dir, "model")
        
        safe_run_chemprop_train(train_path,
                                val_path,
                                # test_path, #TODO: rewrite to incorporate test_path 
                                model_dir,
                                config) 

        
        if not os.path.exists(val_scores_path):
            print(f"⚠️ val_scores.json not found for int_{int_fold_idx} — trying manual prediction")

            val_preds_path = os.path.join(model_dir, "val_preds.csv")

            subprocess.run([
                "chemprop_predict",
                "--test_path", val_path,
                "--checkpoint_path", os.path.join(model_dir, "fold_0", "model_0", "model.pt"),
                "--preds_path", val_preds_path,
                "--smiles_column", config["smiles_col"]
            ], check=True)

            val_df = pd.read_csv(val_path)
            preds = pd.read_csv(val_preds_path)

            # if "prediction" in preds.columns:
            y_true = val_df[config["target_col"]]
            # y_score = preds.get("prediction", preds[config["target_col"]])
            y_score = pd.to_numeric(preds.get("prediction", preds[config["target_col"]]), errors="coerce")
            if y_score.isnull().any():
                print("⚠️ Warning: Some predicted scores could not be converted to floats.")
                y_score = y_score.fillna(0.0)  # or drop rows

            try:
       
                
                # if config["metric"] == "auc":
                roc_auc = roc_auc_score(y_true, y_score)
                # elif config["metric"] == "f1":
                y_pred = (y_score >= 0.5).astype(int)
                f1 = f1_score(y_true, y_pred)
                # elif config["metric"] == "auc_pr":
                auc_pr = average_precision_score(y_true, y_score)
                # else:
                    # raise ValueError(f"Unsupported metric: {config['metric']}")
                with open(val_scores_path, "w") as f:
                    f.write(f'{{"auc": {roc_auc:.4f}, "f1": {f1:.4f}, "auc_pr": {auc_pr:.4f}}}')
                print(f"✅ Manually computed and saved val scores: AUC={roc_auc:.4f}, F1={f1:.4f}, PR={auc_pr:.4f}")


            except Exception as e:
                print(f"❌ Failed to compute score manually: {e}")
                continue
            # else:
            #     print("❌ Prediction column missing from manual prediction output")
            #     continue

        try:
            scores = pd.read_json(val_scores_path, typ='series')
            if scores[config["metric"]] > best_score:
                best_score = scores[config["metric"]]
                best_model_dir = model_dir
        except Exception as e:
            print(f"❌ Failed to read AUC for int_{int_fold_idx}: {e}")

    if best_model_dir is None:
        raise RuntimeError(f"❌ No valid internal models found for {ext_dir} — check internal CV logs.")

    return best_model_dir


def refit_final_model(train_val_df, test_df, ext_dir, config, best_model_dir):
    final_dir = os.path.join(ext_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    # Combine the train and val splits from the best internal model
    best_int_dir = os.path.dirname(best_model_dir)
    best_train_df = pd.read_csv(os.path.join(best_int_dir, "train.csv"))
    best_val_df = pd.read_csv(os.path.join(best_int_dir, "val.csv"))
    combined_train_df = pd.concat([best_train_df, best_val_df], ignore_index=True)

    final_train_csv = os.path.join(final_dir, "train.csv")
    final_test_csv = os.path.join(final_dir, "test.csv")
    final_model_dir = os.path.join(final_dir, "model")

    combined_train_df.to_csv(final_train_csv, index=False)
    test_df.to_csv(final_test_csv, index=False)

    # Use checkpoint_path for warm-starting from best internal model
    # subprocess.run([
    #     "chemprop_train",
    #     "--data_path", final_train_csv,
    #     "--separate_val_path", final_test_csv,  # treat test set as validation
    #     "--save_preds",
    #     "--dataset_type", "classification",
    #     "--smiles_column", config["smiles_col"],
    #     "--target_columns", config["target_col"],
    #     "--save_dir", final_model_dir,
    #     "--num_folds", "1",
    #     "--metric", config["metric"],
    #     "--checkpoint_path", os.path.join(best_model_dir, "fold_0", "model_0", "model.pt")
    # ], check=True)
    safe_run_chemprop_train(
                            # train_path,
                            # val_path,
                            # test_path, #TODO: rewrite to incorporate test_path, update var names
                            # model_dir,
                            config) 
    

    return os.path.join(final_model_dir, "val_preds.csv")



def evaluate_predictions(preds_csv, config):
    test_preds = pd.read_csv(preds_csv)
    y_true = test_preds[config["target_col"]]
    # y_score = test_preds[config["target_col"]]
    if "prediction" not in test_preds.columns and config["target_col"] not in test_preds.columns:
        raise ValueError("❌ No prediction column found in test predictions.")
    # y_score = test_preds.get("prediction", test_preds[config["target_col"]])
    y_score = pd.to_numeric(test_preds.get("prediction", test_preds[config["target_col"]]), errors="coerce")
    if y_score.isnull().any():
        print("⚠️ Warning: Some predicted scores could not be converted to floats.")
        y_score = y_score.fillna(0.0)  # or drop rows
    y_pred = (y_score >= 0.5).astype(int)

    if config["metric"] == "auc":
        metric_val = roc_auc_score(y_true, y_score)
    elif config["metric"] == "f1":
        metric_val = f1_score(y_true, y_pred)
    elif config["metric"] == "auc_pr":
        metric_val = average_precision_score(y_true, y_score)
    else:
        raise ValueError(f"Unsupported metric: {config['metric']}")

    return {config["metric"]: metric_val}


def run_kfold_cv(config):
    df = pd.read_csv(config["dataset_path"])
    skf = StratifiedKFold(n_splits=config["external_folds"], shuffle=True, random_state=config["seed"])

    use_holdout = config.get("holdout_test_path") is not None

    if use_holdout:
        print("🧪 Using a held-out test set for evaluation across all folds.")
        test_df = pd.read_csv(config["holdout_test_path"])
    else:
        print("⚠️ No held-out test set provided — each fold will be evaluated on its own validation split.")

    fold_metrics = []
    all_preds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df[config["target_col"]])):
        print(f"🔁 Fold {fold_idx + 1}/{config['external_folds']}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        fold_dir = os.path.join(config["base_dir"], f"kfold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        train_path = os.path.join(fold_dir, "train.csv")
        val_path = os.path.join(fold_dir, "val.csv")

        # 🔧 Save both train and val CSVs
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        if use_holdout:
            test_path = os.path.join(fold_dir, "test.csv")
            test_df.to_csv(test_path, index=False)
        else:
            test_path = val_path  # reuse val.csv as test
            print(f"🔁 Using val.csv as both validation and test set for fold {fold_idx}")

        model_dir = os.path.join(fold_dir, "model")

        # Train the model using all 3 datasets
        safe_run_chemprop_train(train_path, val_path, test_path, model_dir, config)

        redundant_preds = os.path.join(model_dir, "fold_0", "test_preds.csv")
        if os.path.exists(redundant_preds):
            os.remove(redundant_preds)

        # Read predictions on the test set
        preds_path = os.path.join(model_dir, "test_preds.csv")
        preds = pd.read_csv(preds_path)

        # Fix column name if needed
        if "smiles" in preds.columns:
            preds.rename(columns={"smiles": "SMILES"}, inplace=True)

        # Unwrap lists in SMILES column if necessary
        if preds["SMILES"].apply(lambda x: isinstance(x, list)).any():
            preds["SMILES"] = preds["SMILES"].apply(lambda x: x[0] if isinstance(x, list) else x)


        y_true = test_df[config["target_col"]] if use_holdout else val_df[config["target_col"]]
        y_score = preds.get("prediction", preds[config["target_col"]])
        y_pred = (y_score >= 0.5).astype(int)

        metrics = {
            "auc_roc": roc_auc_score(y_true, y_score),
            "auc_pr": average_precision_score(y_true, y_score),
            "f1": f1_score(y_true, y_pred)
        }
        # Optional: attach SMILES and prediction details for curve plotting
        fold_preds_df = pd.DataFrame({
            "fold": fold_idx,
            "SMILES": preds.get("SMILES", None),
            "y_true": y_true,
            "y_score": y_score
        })
        all_preds.append(fold_preds_df)


        fold_metrics.append(metrics)
        print(f"✅ Fold {fold_idx} metrics: {metrics}")
        

    metrics_df = pd.DataFrame(fold_metrics)

    # Save per-sample predictions for plotting
    preds_df = pd.concat(all_preds, ignore_index=True)
    preds_df.to_csv(os.path.join(config["base_dir"], "kfold_predictions.csv"), index=False)



# Compute mean and 95% CI for each metric
    # === Compute summary metrics with confidence intervals ===
    summary_rows = []
    for metric in metrics_df.columns:
        values = metrics_df[metric]
        mean = values.mean()
        stderr = sem(values)
        ci = t.ppf((1 + 0.95) / 2., len(values) - 1) * stderr
        ci_low = mean - ci
        ci_high = mean + ci

        summary_rows.append({
            "Metric": metric.upper(),
            "Mean": round(mean, 4),
            "CI lower": round(ci_low, 4),
            "CI upper": round(ci_high, 4)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(config["base_dir"], "kfold_metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n💾 Saved summary metrics with CIs to {summary_path}")


    print("\n📊 Final K-Fold CV Summary:")
    print(summary_df.to_string(index=False))



def run_nested_cv(config):
    df = pd.read_csv(config["dataset_path"])
    skf = StratifiedKFold(n_splits=config["external_folds"], shuffle=True, random_state=config["seed"])
    splits = list(skf.split(df, df[config["target_col"]]))

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
        preds_csv = refit_final_model(train_val_df, test_df, ext_dir, config, best_model_dir)

        print(f"📊 Evaluating predictions for ext_{ext_fold_idx}...")
        metrics = evaluate_predictions(preds_csv, config)
        print(f"  ✅ Test Metrics: {metrics}")
        external_metrics.append(metrics)

    if external_metrics:
        external_df = pd.DataFrame(external_metrics)
        external_df.to_csv(os.path.join(config["base_dir"], "external_metrics.csv"), index=False)
        print(f"💾 Saved detailed test metrics to {os.path.join(config['base_dir'], 'external_metrics.csv')}")
        print("\n📊 Final Summary Across External Folds:")
        for metric in external_df.columns:
            mean, (ci_low, ci_high) = mean_ci(external_df[metric])
            print(f"{metric.upper()}: {mean:.3f} (95% CI: {ci_low:.3f}–{ci_high:.3f})")
    else:
        print("\n⚠️ No folds were successfully evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run nested CV with Chemprop")
    parser.add_argument("--metric", type=str, default=default_config["metric"], choices=["auc", "f1", "auc_pr"], help="Metric to optimize and evaluate")
    parser.add_argument("--num_epochs", type=int, default=default_config["num_epochs"])
    parser.add_argument("--base_dir", type=str, default=default_config["base_dir"])
    parser.add_argument("--dataset_path", type=str, default=default_config["dataset_path"])
    parser.add_argument("--external_folds", type=int, default=default_config["external_folds"])
    parser.add_argument("--internal_folds", type=int, default=default_config["internal_folds"])
    parser.add_argument("--smiles_col", type=str, default=default_config["smiles_col"])
    parser.add_argument("--target_col", type=str, default=default_config["target_col"])
    parser.add_argument("--seed", type=int, default=default_config["seed"])
    parser.add_argument("--holdout_test_path", type=str, default=None,
                    help="Optional path to a held-out test set. If provided, all folds will evaluate on this.")
    parser.add_argument("--cv_mode", type=str, default="nested", choices=["nested", "kfold"],
    help="Choose between 'nested' or 'kfold' cross-validation strategies"
)


    args = parser.parse_args()
    config = vars(args)

    if config["cv_mode"] == "kfold":
        run_kfold_cv(config)
    else:
        run_nested_cv(config)

