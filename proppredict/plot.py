import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score
from scipy.stats import sem, t


def mean_ci(values, confidence=0.95):
    mean = np.mean(values)
    stderr = sem(values)
    ci_range = t.ppf((1 + confidence) / 2., len(values) - 1) * stderr
    return mean, (mean - ci_range, mean + ci_range)


def plot_kfold_curves(preds_csv, target_col="y_true", score_col="y_score", fold_col="fold"):
    df = pd.read_csv(preds_csv)
    unique_folds = df[fold_col].unique()

    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    all_tprs = []
    all_precisions = []
    all_roc_aucs = []
    all_pr_aucs = []
    f1_scores = []

    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))

    best_f1 = 0
    best_f1_coords = (0, 0)

    for fold in unique_folds:
        fold_df = df[df[fold_col] == fold]
        y_true = fold_df[target_col]
        y_score = fold_df[score_col]

        # === ROC ===
        fpr, tpr, _ = roc_curve(y_true, y_score)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        all_tprs.append(interp_tpr)
        roc_auc = auc(fpr, tpr)
        all_roc_aucs.append(roc_auc)
        ax_roc.plot(fpr, tpr, alpha=0.2, label=f"Fold {fold} (AUC = {roc_auc:.2f})")

        # === PR + F1 ===
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        all_precisions.append(interp_precision)
        pr_auc = auc(recall, precision)
        all_pr_aucs.append(pr_auc)

        # F1 curve
        f1_curve = 2 * precision * recall / (precision + recall + 1e-8)
        f1_max = np.max(f1_curve)
        f1_scores.append((recall, f1_curve))

        ax_pr.plot(recall, precision, alpha=0.2,
                   label=f"Fold {fold} (AUC = {pr_auc:.2f}, F1 = {f1_max:.2f})")

        # Track best F1 point across folds
        best_idx = np.argmax(f1_curve)
        if f1_curve[best_idx] > best_f1:
            best_f1 = f1_curve[best_idx]
            best_f1_coords = (recall[best_idx], precision[best_idx])

    # === Mean ROC ===
    mean_tpr = np.mean(all_tprs, axis=0)
    mean_roc_auc, (roc_ci_low, roc_ci_high) = mean_ci(all_roc_aucs)
    ax_roc.plot(mean_fpr, mean_tpr, color="black", lw=2,
                label=f"Mean ROC (AUC = {mean_roc_auc:.2f} ± {roc_ci_high - mean_roc_auc:.2f})")
    ax_roc.plot([0, 1], [0, 1], "--", color="gray")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.spines['top'].set_visible(False)
    ax_roc.spines['right'].set_visible(False)
    ax_roc.legend()

    # === Mean PR ===
    mean_precision = np.mean(all_precisions, axis=0)
    mean_pr_auc, (pr_ci_low, pr_ci_high) = mean_ci(all_pr_aucs)
    ax_pr.plot(mean_recall, mean_precision, color="black", lw=2,
               label=f"Mean PR (AUC = {mean_pr_auc:.2f} ± {pr_ci_high - mean_pr_auc:.2f})")

    # Best F1 annotation
    ax_pr.scatter(*best_f1_coords, color="red", zorder=5, label=f"Best F1 = {best_f1:.2f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.spines['top'].set_visible(False)
    ax_pr.spines['right'].set_visible(False)
    ax_pr.legend()

    return fig_roc, fig_pr
