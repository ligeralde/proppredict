import os
import re
import json
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from chemprop.train import make_predictions
from chemprop.args import PredictArgs

def evaluate_kfold_inference_on_test(
    base_path,
    test_csv_path,
    smiles_col="standardized_SMILES",
    target_col="activity",
    output_csv_path="kfold_preds_on_heldout.csv"
):
    test_df_full = pd.read_csv(test_csv_path)
    all_preds = []

    kfold_dirs = sorted(
        [d for d in os.listdir(base_path) if re.match(r"kfold_\d+", d)],
        key=lambda x: int(x.split("_")[1])
    )

    for kfold in kfold_dirs:
        fold_idx = int(kfold.split("_")[1])
        fold_root = os.path.join(base_path, kfold, "model", "fold_0", "model_0")
        if not os.path.exists(fold_root):
            print(f"❌ Missing model folder in {fold_root}, skipping.")
            continue

        args_json_path = os.path.join(fold_root, "args.json")
        temp_preds_path = os.path.join(base_path, kfold, "heldout_preds.csv")

        checkpoint_paths = [
            os.path.join(fold_root, f)
            for f in os.listdir(fold_root)
            if f.endswith(".pt")
        ]
        if not checkpoint_paths:
            print(f"⚠️ No .pt checkpoint found in {fold_root}, skipping fold {fold_idx}.")
            continue

        args_dict = {
            "test_path": test_csv_path,  # will be overridden below
            "preds_path": temp_preds_path,
            "checkpoint_paths": checkpoint_paths,
            "smiles_column": smiles_col,
            "use_compound_names": False
        }

        if os.path.exists(args_json_path):
            with open(args_json_path) as f:
                train_args = json.load(f)
            if "features_generator" in train_args:
                args_dict["features_generator"] = train_args["features_generator"]
            if "features_scaling" in train_args:
                args_dict["features_scaling"] = train_args["features_scaling"]
            if "no_features" in train_args:
                args_dict["no_features"] = train_args["no_features"]

        args = PredictArgs().from_dict(args_dict)

        # === Clean test set and save to /tmp
        clean_test_path = f"/tmp/chemprop_temp_fold{fold_idx}.csv"
        clean_test_df = test_df_full[[smiles_col, target_col]].dropna(subset=[smiles_col])
        clean_test_df.to_csv(clean_test_path, index=False)
        args.test_path = clean_test_path

        make_predictions(args=args)

        if not os.path.exists(temp_preds_path):
            print(f"❌ Fold {fold_idx}: Prediction file not generated. Possibly invalid SMILES.")
            continue

        preds = pd.read_csv(temp_preds_path)
        if "smiles" in preds.columns:
            preds.rename(columns={"smiles": smiles_col}, inplace=True)
        preds[smiles_col] = preds[smiles_col].apply(lambda x: x[0] if isinstance(x, list) else x)

        matched_test_df = test_df_full[test_df_full[smiles_col].isin(preds[smiles_col])].reset_index(drop=True)
        matched_preds = preds[preds[smiles_col].isin(matched_test_df[smiles_col])].reset_index(drop=True)

        if len(matched_test_df) == 0:
            print(f"⚠️ Fold {fold_idx}: No predictions matched test set.")
            continue

        # === Detect prediction column
        pred_col = "prediction" if "prediction" in matched_preds.columns else target_col

        # === Convert predictions to numeric and filter
        y_score = pd.to_numeric(matched_preds[pred_col], errors="coerce")
        valid_mask = y_score.notna()

        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            print(f"⚠️ Fold {fold_idx}: Dropped {n_invalid} invalid predictions (non-numeric).")

        matched_test_df = matched_test_df[valid_mask].reset_index(drop=True)
        matched_preds = matched_preds[valid_mask].reset_index(drop=True)
        y_score = y_score[valid_mask].reset_index(drop=True)

        y_true = matched_test_df[target_col]

        fold_preds_df = pd.DataFrame({
            "fold": fold_idx,
            "SMILES": matched_preds[smiles_col],
            "y_true": y_true,
            "y_score": y_score
        })
        all_preds.append(fold_preds_df)

        auc_roc = roc_auc_score(y_true, y_score)
        auc_pr = average_precision_score(y_true, y_score)
        y_pred = (y_score >= 0.5).astype(int)
        f1 = f1_score(y_true, y_pred)

        print(f"✅ Fold {fold_idx}: AUC ROC = {auc_roc:.3f}, AUC PR = {auc_pr:.3f}, F1 = {f1:.3f}")

    if not all_preds:
        raise RuntimeError("No predictions were made. Check SMILES validity and model paths.")

    preds_df = pd.concat(all_preds, ignore_index=True)
    preds_df.to_csv(output_csv_path, index=False)
    print(f"\n💾 Saved combined predictions to: {output_csv_path}")
