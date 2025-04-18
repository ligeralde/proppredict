import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from scipy.stats import sem, t
import subprocess
# import pickle
import ast
import threading
import time
import shutil
from multiprocessing import Process, cpu_count


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
    "seed": 42,
    "use_rdkit_features": False,
    "scale_rdkit_features": False,
    "hyperparams_path": None,
    "num_gpus": 2
}


def patch_config_for_rdkit(config):
    """
    Updates the Chemprop training and prediction configuration to include
    RDKit features and normalization if specified in the config.
    """
    base_args = []

    if config.get("use_rdkit_features", False):
        base_args.extend([
            "--features_generator", "rdkit_2d_normalized",
            "--no_features_scaling" if not config.get("scale_rdkit_features", True) else ""
        ])

    return [arg for arg in base_args if arg]


def mean_ci(scores, confidence=0.95):
    mean = np.mean(scores)
    stderr = sem(scores)
    ci_range = t.ppf((1 + confidence) / 2., len(scores) - 1) * stderr
    return mean, (mean - ci_range, mean + ci_range)


def monitor_gpu(gpu_id, stop_event):
    while not stop_event.is_set():
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                text=True
            )
            lines = output.strip().split("\n")
            for line in lines:
                index, name, util, mem = [x.strip() for x in line.split(",")]
                if int(index) == gpu_id:
                    print(f"🖥️  [GPU {index}] {name} — Utilization: {util}% — Memory: {mem} MiB")
        except Exception as e:
            print(f"⚠️ GPU monitor failed: {e}")
        time.sleep(10)


def safe_run_chemprop_train(train_path, val_path, test_path, save_dir, config, gpu_id=0):
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
        "--epochs", str(config["num_epochs"]),
    ] + patch_config_for_rdkit(config)

    if config.get("hyperparams_path") and config["hyperparams_path"] != "default":
        command.extend(["--config_path", config["hyperparams_path"]])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"🚀 Starting Chemprop on GPU {gpu_id} with command:\n{' '.join(command)}\n")

    # Start GPU monitoring in a background thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_gpu, args=(gpu_id, stop_event))
    monitor_thread.start()

    try:
        subprocess.run(command, check=True, env=env)
    finally:
        stop_event.set()
        monitor_thread.join()
        print(f"✅ Finished training on GPU {gpu_id}\n")



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
                "--smiles_column", config["smiles_col"] #TODO: add rdkit config, hyperparams
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




def unwrap_smiles_column(preds_csv):
    try:
        df = pd.read_csv(preds_csv)

        # Normalize the SMILES column name
        if "smiles" in df.columns:
            df.rename(columns={"smiles": "SMILES"}, inplace=True)

        # Fix list-like SMILES values: if they are strings that look like lists or actual lists
        def clean_smiles(val):
            if isinstance(val, list):
                return val[0]
            if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        return parsed[0]
                except Exception:
                    return val  # fallback
            return val

        if "SMILES" in df.columns:
            df["SMILES"] = df["SMILES"].apply(clean_smiles)
            df.to_csv(preds_csv, index=False)
            print(f"🛠️ Fixed SMILES format in: {preds_csv}")
        else:
            print(f"⚠️ No SMILES column found in: {preds_csv}")

    except Exception as e:
        print(f"❌ Failed to fix SMILES in {preds_csv}: {e}")



def run_fold_parallel(fold_idx, train_df, val_df, test_df, config, gpu_id):
    fold_dir = os.path.join(config["base_dir"], f"kfold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    train_path = os.path.join(fold_dir, "train.csv")
    val_path = os.path.join(fold_dir, "val.csv")
    model_dir = os.path.join(fold_dir, "model")
    preds_path = os.path.join(model_dir, "test_preds.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    if not os.path.exists(preds_path):
        # Use val_path as test_path if same
        test_path = val_path if test_df.equals(val_df) else os.path.join(fold_dir, "test.csv")
        if not test_df.equals(val_df):
            test_df.to_csv(test_path, index=False)

        safe_run_chemprop_train(train_path, val_path, test_path, model_dir, config, gpu_id=gpu_id)

    unwrap_smiles_column(preds_path)



def run_kfold_cv(config):
    df = pd.read_csv(config["dataset_path"])
    skf = StratifiedKFold(n_splits=config["external_folds"], shuffle=True, random_state=config["seed"])

    use_holdout = config.get("holdout_test_path") is not None
    if use_holdout:
        print("🧑‍🔬 Using held-out test set.")
        test_df_full = pd.read_csv(config["holdout_test_path"])
    else:
        print("⚠️ No held-out test set — validation used as test.")

    all_preds = []
    fold_metrics = []
    max_concurrent = config["num_gpus"]
    active_processes = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df[config["target_col"]])):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        test_df = test_df_full.copy() if use_holdout else val_df.copy()

        gpu_id = fold_idx % config["num_gpus"]

        p = Process(target=run_fold_parallel, args=(fold_idx, train_df, val_df, test_df, config, gpu_id))
        p.start()
        active_processes.append(p)

        if len(active_processes) == max_concurrent:
            for proc in active_processes:
                proc.join()
            active_processes = []

    for proc in active_processes:
        proc.join()

    print("✅ All folds finished. Starting evaluation...")

    for fold_idx in range(config["external_folds"]):
        fold_dir = os.path.join(config["base_dir"], f"kfold_{fold_idx}")
        val_path = os.path.join(fold_dir, "val.csv")
        test_path = val_path if not use_holdout else os.path.join(fold_dir, "test.csv")
        preds_path = os.path.join(fold_dir, "model", "test_preds.csv")

        test_df = pd.read_csv(test_path)
        preds = pd.read_csv(preds_path)
        if "smiles" in preds.columns:
            preds.rename(columns={"smiles": "SMILES"}, inplace=True)
        preds["SMILES"] = preds["SMILES"].apply(lambda x: x[0] if isinstance(x, list) else x)

        matched_test_df = test_df[test_df[config["smiles_col"]].isin(preds["SMILES"])].reset_index(drop=True)
        matched_preds = preds[preds["SMILES"].isin(matched_test_df[config["smiles_col"]])].reset_index(drop=True)

        assert len(matched_test_df) == len(matched_preds), f"Mismatch in fold {fold_idx}"

        y_true = matched_test_df[config["target_col"]]
        y_score = matched_preds.get("prediction", matched_preds[config["target_col"]])
        y_pred = (y_score >= 0.5).astype(int)

        metrics = {
            "auc_roc": roc_auc_score(y_true, y_score),
            "auc_pr": average_precision_score(y_true, y_score),
            "f1": f1_score(y_true, y_pred)
        }

        fold_preds_df = pd.DataFrame({
            "fold": fold_idx,
            "SMILES": matched_preds["SMILES"],
            "y_true": y_true,
            "y_score": y_score
        })
        all_preds.append(fold_preds_df)
        fold_metrics.append(metrics)

        print(f"✅ Fold {fold_idx} metrics: {metrics}")

    preds_df = pd.concat(all_preds, ignore_index=True)
    preds_df.to_csv(os.path.join(config["base_dir"], "kfold_predictions.csv"), index=False)

    metrics_df = pd.DataFrame(fold_metrics)
    summary_rows = []
    for metric in metrics_df.columns:
        values = metrics_df[metric]
        mean = values.mean()
        stderr = values.sem()
        ci = 1.96 * stderr
        summary_rows.append({
            "Metric": metric.upper(),
            "Mean": round(mean, 4),
            "CI lower": round(mean - ci, 4),
            "CI upper": round(mean + ci, 4)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(config["base_dir"], "kfold_metrics_summary.csv"), index=False)

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
    help="Choose between 'nested' or 'kfold' cross-validation strategies",)
    parser.add_argument("--use_rdkit_features", action="store_true", help="Use RDKit 2D normalized features"),
    parser.add_argument("--scale_rdkit_features", action="store_true", default=True, help="Normalize RDKit features (default: True)"),
    parser.add_argument("--config_path", type=str, default=default_config["hyperparams_path"])
    parser.add_argument("--num_gpus", type=int, default=default_config["num_gpus"], help="Number of GPUs to use in parallel")



    args = parser.parse_args()
    config = vars(args)

    if config["cv_mode"] == "kfold":
        run_kfold_cv(config)
    else:
        run_nested_cv(config)

