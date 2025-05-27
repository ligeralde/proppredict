import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score
from scipy.stats import sem, t
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt,ceil



def plot_aggregate_topk_enrichment(
    df,
    smiles_col="SMILES",
    y_true_col="y_true",
    y_pred_col="y_score",
    top_k_true=50,
    top_k_pred=50,
    direction="top",  # "top" or "bottom"
    title="Aggregate Top-K Enrichment",
    n_random_trials=0,
    random_seed=42
):

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    assert direction in ["top", "bottom"]

    # Aggregate all predictions per SMILES by average prediction
    df = df.copy()
    avg_scores = df.groupby(smiles_col)[y_pred_col].mean()
    true_values = df.groupby(smiles_col)[y_true_col].mean()

    # If antibiotic column exists, propagate it
    has_antibiotic_col = "antibiotic" in df.columns
    if has_antibiotic_col:
        antibiotic_flags = df.groupby(smiles_col)["antibiotic"].max()
    else:
        antibiotic_flags = pd.Series(0, index=avg_scores.index)

    # Compute average rank for tie-breaking in consistent sorting
    df["rank"] = df.groupby("fold")[y_pred_col].rank(ascending=(direction == "bottom"))
    avg_ranks = df.groupby(smiles_col)["rank"].mean()

    # Build a new DataFrame for plotting
    agg_df = pd.DataFrame({
        "avg_rank": avg_ranks,
        "avg_score": avg_scores,
        "y_true": true_values,
        "antibiotic": antibiotic_flags
    }).reset_index()

    # Determine top-K true and predicted compounds
    if direction == "top":
        top_true_idx = agg_df.nlargest(top_k_true, "y_true").index
        top_pred_idx = agg_df.nlargest(top_k_pred, "avg_score").index
        top_true_ranks = agg_df.loc[top_true_idx]
    else:
        top_true_idx = agg_df.nsmallest(top_k_true, "y_true").index
        top_pred_idx = agg_df.nsmallest(top_k_pred, "avg_score").index
        top_true_ranks = agg_df.loc[top_true_idx]

    true_idx_set = set(top_true_idx)
    pred_idx_set = set(top_pred_idx)

    sorted_pred_idx = agg_df.sort_values("avg_score", ascending=(direction == "bottom")).index
    k_needed = 0
    covered_true = set()
    for idx in sorted_pred_idx:
        k_needed += 1
        if idx in true_idx_set:
            covered_true.add(idx)
        if covered_true == true_idx_set:
            break
    else:
        k_needed = None

    tp_idx = true_idx_set & pred_idx_set
    fp_idx = pred_idx_set - true_idx_set
    fn_idx = true_idx_set - pred_idx_set

    n_tp = len(tp_idx)
    n_fp = len(fp_idx)
    n_fn = len(fn_idx)
    recall = n_tp / max(len(true_idx_set), 1)
    precision = n_tp / max(len(pred_idx_set), 1)

    z_score = None
    p_value = None
    rng = np.random.default_rng(random_seed)
    if n_random_trials > 0:
        rand_recalls = []
        all_indices = agg_df.index.tolist()
        for _ in range(n_random_trials):
            sampled = set(rng.choice(all_indices, size=top_k_pred, replace=False))
            r = len(sampled & true_idx_set) / max(len(true_idx_set), 1)
            rand_recalls.append(r)
        mean_rand = np.mean(rand_recalls)
        std_rand = np.std(rand_recalls)
        z_score = (recall - mean_rand) / (std_rand + 1e-8)
        p_value = (np.sum(np.array(rand_recalls) >= recall) + 1) / (n_random_trials + 1)

    fig, (ax_scatter, ax_cum, ax_bar) = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 2, 1]})

    # --- Scatter plot: Predicted vs True ---
    ax_scatter.scatter(agg_df["y_true"], agg_df["avg_score"], alpha=0.15, edgecolors='none', color='gray', label='All', zorder=0)
    ax_scatter.scatter(agg_df.loc[list(fn_idx), "y_true"], agg_df.loc[list(fn_idx), "avg_score"],
                       color='purple', edgecolors='k', label='Missed (FN)', zorder=1)
    ax_scatter.scatter(agg_df.loc[list(fp_idx), "y_true"], agg_df.loc[list(fp_idx), "avg_score"],
                       color='red', edgecolors='k', label='False + (FP)', zorder=2)
    ax_scatter.scatter(agg_df.loc[list(tp_idx), "y_true"], agg_df.loc[list(tp_idx), "avg_score"],
                       color='green', edgecolors='k', label='Captured (TP)', zorder=3)

    if has_antibiotic_col:
        antibiotic_df = agg_df[agg_df["antibiotic"] == 1]
        ax_scatter.scatter(
            antibiotic_df["y_true"], antibiotic_df["avg_score"],
            facecolors='none', edgecolors='black', marker='*', s=150,
            label='Antibiotic', zorder=4
        )

    ax_scatter.set_xlabel("True")
    ax_scatter.set_ylabel("Predicted (avg)")
    ax_scatter.set_title(title)
    ax_scatter.grid(True)
    ax_scatter.legend(loc='lower right', frameon=True)

    # --- Annotation box ---
    annotation = (
        f"TP: {n_tp}  FP: {n_fp}  FN: {n_fn}\n"
        f"Recall: {recall:.1%}  Precision: {precision:.1%}"
    )
    if z_score is not None:
        annotation += f"\nZ (recall vs rand): {z_score:.2f}  p: {p_value:.3f}"
    annotation += f"\nk to get all top-k true: {k_needed}"

    ax_scatter.text(
        0.95, 0.95, annotation,
        transform=ax_scatter.transAxes,
        ha='right', va='top',
        fontsize=9,
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4')
    )

    # --- Cumulative capture plot ---

    sorted_preds = agg_df.sort_values("avg_score", ascending=(direction == "bottom"))
    hit_vector = [1 if idx in true_idx_set else 0 for idx in sorted_preds.index]
    cumulative_hits = np.cumsum(hit_vector)
    x = np.arange(1, len(cumulative_hits) + 1)

    ax_cum.plot(x[:top_k_pred], cumulative_hits[:top_k_pred], color="blue", lw=2, label="Cumulative TP")
    ax_cum.set_xlim(1, top_k_pred)
    ax_cum.set_ylim(0, top_k_true + 1)
    ax_cum.set_title("Cumulative TP")
    ax_cum.set_ylabel("# True Top-K Captured")
    ax_cum.set_xlabel("Top-K Predicted Rank")
    ax_cum.grid(True)

    if has_antibiotic_col:
        antibiotic_in_preds = sorted_preds.iloc[:top_k_pred][sorted_preds["antibiotic"] == 1]
        ranks_abx = antibiotic_in_preds.index
        hits_abx = cumulative_hits[ranks_abx]
        ax_cum.scatter(ranks_abx + 1, hits_abx, facecolors='none', edgecolors='black', marker='*', s=100, label='Antibiotic')

    ax_cum.legend()


    # --- Rank capture bar plot ---
    capture_flags = [1 if idx in pred_idx_set else 0 for idx in top_true_ranks.index]
    y_ranks = np.arange(1, top_k_true + 1)
    ax_bar.vlines(y_ranks, 0, capture_flags, color='green', linewidth=2)
    ax_bar.set_xlim(0, top_k_true + 1)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.set_ylabel("Captured")
    ax_bar.set_xlabel("Top-K True Rank")
    ax_bar.set_yticks([0, 1])
    xticks = sorted(set([1] + list(range(10, top_k_true + 1, 10))))
    ax_bar.set_xticks(xticks)
    ax_bar.set_xticklabels([str(t) for t in xticks])
    ax_bar.grid(True, axis='x', linestyle='--', alpha=0.3)

    if has_antibiotic_col:
        antibiotic_in_top_true = top_true_ranks[top_true_ranks["antibiotic"] == 1]
        abx_positions = [i + 1 for i, idx in enumerate(top_true_ranks.index) if idx in antibiotic_in_top_true.index]
        abx_captured = [capture_flags[i] for i, idx in enumerate(top_true_ranks.index) if idx in antibiotic_in_top_true.index]
        ax_bar.scatter(abx_positions, abx_captured, facecolors='none', edgecolors='black', marker='*', s=100)

    plt.tight_layout()
    plt.show()

    return agg_df




def plot_pred_vs_true_topk_enrichment_by_fold(
    df,
    fold_col="fold",
    y_true_col="y_true",
    y_pred_col="y_score",
    top_k_true=50,
    top_k_pred=50,
    direction="top",  # "top" or "bottom"
    title="Top-K Enrichment per Fold",
    n_random_trials=0,
    random_seed=42
):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    assert direction in ["top", "bottom"]

    folds = sorted(df[fold_col].unique())
    n_folds = len(folds)

    summary = []
    rng = np.random.default_rng(random_seed)

    for fold in folds:
        fig, (ax_scatter, ax_cum, ax_bar) = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 2, 1]})

        fold_df = df[df[fold_col] == fold].copy()
        y_true = fold_df[y_true_col].values
        y_pred = fold_df[y_pred_col].values

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        if direction == "top":
            top_true_idx = fold_df[y_true_col].nlargest(top_k_true).index
            top_pred_idx = fold_df[y_pred_col].nlargest(top_k_pred).index
            top_true_ranks = fold_df[y_true_col].nlargest(top_k_true)
        else:
            top_true_idx = fold_df[y_true_col].nsmallest(top_k_true).index
            top_pred_idx = fold_df[y_pred_col].nsmallest(top_k_pred).index
            top_true_ranks = fold_df[y_true_col].nsmallest(top_k_true)

        true_idx_set = set(top_true_idx)
        pred_idx_set = set(top_pred_idx)

        sorted_pred_idx = fold_df[y_pred_col].sort_values(ascending=(direction == "bottom")).index
        k_needed = 0
        covered_true = set()
        for idx in sorted_pred_idx:
            k_needed += 1
            if idx in true_idx_set:
                covered_true.add(idx)
            if covered_true == true_idx_set:
                break
        else:
            k_needed = None

        tp_idx = true_idx_set & pred_idx_set
        fp_idx = pred_idx_set - true_idx_set
        fn_idx = true_idx_set - pred_idx_set

        n_tp = len(tp_idx)
        n_fp = len(fp_idx)
        n_fn = len(fn_idx)
        recall = n_tp / max(len(true_idx_set), 1)
        precision = n_tp / max(len(pred_idx_set), 1)

        top_k_true_captures = [int(idx in pred_idx_set) for idx in top_true_ranks.index]

        z_score = None
        p_value = None
        if n_random_trials > 0:
            rand_recalls = []
            all_indices = fold_df.index.tolist()
            for _ in range(n_random_trials):
                sampled = set(rng.choice(all_indices, size=top_k_pred, replace=False))
                r = len(sampled & true_idx_set) / max(len(true_idx_set), 1)
                rand_recalls.append(r)
            mean_rand = np.mean(rand_recalls)
            std_rand = np.std(rand_recalls)
            z_score = (recall - mean_rand) / (std_rand + 1e-8)
            p_value = (np.sum(np.array(rand_recalls) >= recall) + 1) / (n_random_trials + 1)

        ax_scatter.scatter(y_true, y_pred, alpha=0.15, edgecolors='none', color='gray', label='All', zorder=0)
        ax_scatter.scatter(fold_df.loc[list(fn_idx), y_true_col], fold_df.loc[list(fn_idx), y_pred_col],
                           color='purple', edgecolors='k', label='Missed (FN)', zorder=1)
        ax_scatter.scatter(fold_df.loc[list(fp_idx), y_true_col], fold_df.loc[list(fp_idx), y_pred_col],
                           color='red', edgecolors='k', label='False + (FP)', zorder=2)
        ax_scatter.scatter(fold_df.loc[list(tp_idx), y_true_col], fold_df.loc[list(tp_idx), y_pred_col],
                           color='green', edgecolors='k', label='Captured (TP)', zorder=3)
        ax_scatter.set_xlabel("True")
        ax_scatter.set_ylabel("Predicted")
        ax_scatter.set_title(f"Fold {fold}")
        ax_scatter.grid(True)
        ax_scatter.legend(loc='lower right', frameon=True)

        annotation = (
            f"$R^2$: {r2:.2f}  RMSE: {rmse:.2f}  MAE: {mae:.2f}\n"
            f"TP: {n_tp}  FP: {n_fp}  FN: {n_fn}\n"
            f"Recall: {recall:.1%}  Precision: {precision:.1%}"
        )
        if z_score is not None:
            annotation += f"\nZ (recall vs rand): {z_score:.2f}  p: {p_value:.3f}"
        annotation += f"\nk to get all top-k true: {k_needed}"

        ax_scatter.text(
            0.95, 0.95, annotation,
            transform=ax_scatter.transAxes,
            ha='right', va='top',
            fontsize=9,
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4')
        )

        sorted_preds = fold_df[y_pred_col].sort_values(ascending=(direction == "bottom"))
        hit_vector = [1 if idx in true_idx_set else 0 for idx in sorted_preds.index[:top_k_pred]]
        cumulative_hits = np.cumsum(hit_vector)
        x = np.arange(1, top_k_pred + 1)
        ax_cum.plot(x, cumulative_hits, color="blue", lw=2)
        ax_cum.set_xlim(1, top_k_pred)
        ax_cum.set_ylim(0, top_k_true + 1)
        ax_cum.set_title("Cumulative TP")
        ax_cum.set_ylabel("# True Top-K Captured")
        ax_cum.set_xlabel("Top-K Predicted Rank")
        ax_cum.grid(True)

        capture_flags = [1 if idx in pred_idx_set else 0 for idx in top_true_ranks.index]
        y_ranks = np.arange(1, top_k_true + 1)
        ax_bar.vlines(y_ranks, 0, capture_flags, color='green', linewidth=2)
        ax_bar.set_xlim(0, top_k_true + 1)
        ax_bar.set_ylim(0, 1.1)
        ax_bar.set_ylabel("Captured")
        ax_bar.set_xlabel("Top-K True Rank")
        ax_bar.set_yticks([0, 1])

        # Always include 1 and then multiples of 10 for x-axis ticks
        xticks = sorted(set([1] + list(range(10, top_k_true + 1, 10))))
        ax_bar.set_xticks(xticks)
        ax_bar.set_xticklabels([str(t) for t in xticks])

        ax_bar.grid(True, axis='x', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.show()

        summary.append({
            "fold": fold,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "top_k_true": len(true_idx_set),
            "top_k_pred": len(pred_idx_set),
            "captured": n_tp,
            "false_positives": n_fp,
            "missed_positives": n_fn,
            "recall": recall,
            "precision": precision,
            "z_score_vs_random_recall": z_score,
            "empirical_p_vs_random_recall": p_value,
            "k_pred_to_capture_all_true_topk": k_needed,
            "top_k_true_captures": top_k_true_captures
        })

    return pd.DataFrame(summary)









def plot_categorical_histogram(df, column, top_n=None, title=None,logy=False):
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
    value_counts.plot(kind='bar',logy=logy)
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
