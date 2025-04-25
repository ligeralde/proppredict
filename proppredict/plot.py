import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score
from scipy.stats import sem, t
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE


def plot_categorical_histogram(df, column, top_n=None, title=None):
    """
    Plots a bar chart for categorical string data in a specified column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): Name of the column containing categorical string data.
        top_n (int, optional): If specified, only the top_n most frequent categories are shown.
        title (str, optional): Title for the plot.
    """
    # Count the occurrences of each category
    value_counts = df[column].value_counts()

    # Optionally select only the top N categories
    if top_n is not None:
        value_counts = value_counts.head(top_n)

    # Plot
    plt.figure(figsize=(10, 6))
    value_counts.plot(kind='bar')
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title(title or f"Histogram of '{column}'")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Example usage:
# df = pd.read_csv("your_file.csv")
# plot_categorical_histogram(df, column='your_column_name', top_n=10)


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
    f1_points = []
    pr_colors = []

    best_f1 = 0
    best_f1_coords = (0, 0)

    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))

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

        # === PR ===
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        all_precisions.append(interp_precision)
        pr_auc = auc(recall, precision)
        all_pr_aucs.append(pr_auc)

        # F1 score at threshold 0.5
        y_pred_0_5 = (y_score >= 0.5).astype(int)
        fold_f1 = f1_score(y_true, y_pred_0_5)
        f1_scores.append(fold_f1)

        # Precision and recall at threshold 0.5
        tp = ((y_true == 1) & (y_pred_0_5 == 1)).sum()
        fp = ((y_true == 0) & (y_pred_0_5 == 1)).sum()
        fn = ((y_true == 1) & (y_pred_0_5 == 0)).sum()
        precision_0_5 = tp / (tp + fp + 1e-8)
        recall_0_5 = tp / (tp + fn + 1e-8)
        f1_points.append((recall_0_5, precision_0_5))

        # Compute F1 scores across all thresholds
        f1_curve = 2 * precision * recall / (precision + recall + 1e-8)
        max_f1_idx = np.argmax(f1_curve)
        if f1_curve[max_f1_idx] > best_f1:
            best_f1 = f1_curve[max_f1_idx]
            best_f1_coords = (recall[max_f1_idx], precision[max_f1_idx])

        # Plot PR curve and capture the color
        pr_line, = ax_pr.plot(recall, precision, alpha=0.2,
                              label=rf"Fold {fold} (AUC = {pr_auc:.2f}, F1_{50} = {fold_f1:.2f})")
        pr_colors.append(pr_line.get_color())

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

    # === Mean PR and F1 ===
    mean_precision = np.mean(all_precisions, axis=0)
    mean_pr_auc, (pr_ci_low, pr_ci_high) = mean_ci(all_pr_aucs)
    mean_f1, (f1_ci_low, f1_ci_high) = mean_ci(f1_scores)
    ax_pr.plot(mean_recall, mean_precision, color="black", lw=2,
               label=f"Mean PR (AUC = {mean_pr_auc:.2f} ± {pr_ci_high - mean_pr_auc:.2f}, F1 = {mean_f1:.2f} ± {f1_ci_high - mean_f1:.2f})")

    # Plot F1 scatter points with corresponding PR curve colors
    for (recall_pt, precision_pt), color in zip(f1_points, pr_colors):
        ax_pr.scatter(recall_pt, precision_pt, s=30, color=color, alpha=0.8)

    # Plot best F1 point across all thresholds and folds
    ax_pr.scatter(*best_f1_coords, color="red", marker="*", s=100, label=f"Best F1 = {best_f1:.2f}")

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.spines['top'].set_visible(False)
    ax_pr.spines['right'].set_visible(False)
    ax_pr.legend()

    return fig_roc, fig_pr





def smiles_to_morgan_fp(smiles, radius=2, n_bits=2048):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

def merge_and_deduplicate(dfs, smiles_column):
    """
    Merges DataFrames, deduplicates, and tracks original source.
    Adds a column 'source_df' indicating origin.
    """
    labeled_dfs = []
    for i, df in enumerate(dfs):
        df_copy = df.copy()
        df_copy["source_df"] = f"df_{i}"
        labeled_dfs.append(df_copy)
    
    combined_df = pd.concat(labeled_dfs, ignore_index=True)
    deduped_df = combined_df.drop_duplicates(subset=smiles_column)
    return deduped_df

def compute_fingerprints(df, smiles_column):
    fps = []
    valid_indices = []
    for idx, smi in enumerate(df[smiles_column]):
        fp = smiles_to_morgan_fp(smi)
        if fp is not None:
            fps.append(fp)
            valid_indices.append(idx)
    return np.array(fps), valid_indices

def tsne_plot(fps, labels, title="t-SNE colored by source DataFrame"):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(fps)

    plt.figure(figsize=(10, 8))
    unique_labels = sorted(set(labels))
    for label in unique_labels:
        mask = np.array(labels) == label
        plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], label=label, alpha=0.7)
    
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Source DF")
    plt.tight_layout()
    plt.show()
