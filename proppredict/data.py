import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import Descriptors
import re
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
from matplotlib.lines import Line2D
from itertools import combinations

from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Create the generator once (outside the function if possible)
morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)

def smiles_to_fp(smiles):
    try:
        if not isinstance(smiles, str) or not smiles.strip():
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return morgan_gen.GetFingerprint(mol)
    except:
        return None
    return None


def get_fingerprints(df, smiles_col):
    fps = []
    valid_idx = []
    for idx, smi in enumerate(df[smiles_col]):
        fp = smiles_to_fp(smi)
        if fp:
            fps.append(fp)
            valid_idx.append(idx)
    return np.array([np.array(fp) for fp in fps]), valid_idx

def merge_dfs_with_labels(dfs, smiles_col, label_col=None, df_names=None):
    labeled_dfs = []
    for i, df in enumerate(dfs):
        name = df_names[i] if df_names and i < len(df_names) else f'df{i+1}'
        df = df.copy()
        df['__source_df__'] = name
        if label_col and label_col not in df.columns:
            df[label_col] = np.nan
        labeled_dfs.append(df)
    merged = pd.concat(labeled_dfs, ignore_index=True)
    return merged


def reduce_and_plot(df, fps, method='tsne', color_by='__source_df__', label_col=None, n_components=2, perplexity=30):
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=perplexity)
    else:
        raise ValueError("method must be 'pca', 'tsne', or 'umap'")
    
    reduced = reducer.fit_transform(fps)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if color_by == '__source_df__':
        groups = df[color_by].unique()
        for g in groups:
            mask = df[color_by] == g
            ax.scatter(reduced[mask, 0], reduced[mask, 1], label=g, alpha=0.3,s=2)
        ax.legend(title='DataFrame Source')
    elif color_by and color_by in df.columns:
        labels = df[color_by]
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', alpha=0.3,s=2)
        fig.colorbar(scatter, ax=ax, label=color_by)
    else:
        ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.3,s=2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title(f"{method.upper()} projection of Morgan Fingerprints")
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    plt.tight_layout()
    return fig, ax



def find_within_df_duplicates(
    df,
    smiles_col="SMILES",
    raw_col="raw_activity",
    error_tolerance=None,
    random_state=42
):
    """
    Fast detection of within-DF duplicates with optional tolerance-based retention.

    Returns:
        cleaned_df, duplicates_df (with 'retained_in_cleaned' column)
    """
    rng = np.random.default_rng(seed=random_state)

    # Identify all duplicates
    dup_mask = df[smiles_col].duplicated(keep=False)
    dups = df[dup_mask].copy()
    dups["source_df"] = "input_df"
    dups["retained_in_cleaned"] = False

    # Non-duplicate rows go straight into cleaned_df
    cleaned_df = df[~dup_mask].copy()

    if error_tolerance is not None:
        # Compute stddev per SMILES group
        std_per_group = (
            dups[[smiles_col, raw_col]]
            .dropna()
            .groupby(smiles_col)[raw_col]
            .std()
            .reset_index(name="std_dev")
        )

        # SMILES that qualify for retention
        eligible_smiles = std_per_group[std_per_group["std_dev"] <= error_tolerance][smiles_col]

        # Filter rows from those SMILES
        eligible_rows = dups[dups[smiles_col].isin(eligible_smiles)]

        # Randomly select one row per group (vectorized)
        retained = (
            eligible_rows.groupby(smiles_col, group_keys=False)
            .apply(lambda x: x.sample(n=1, random_state=random_state))
        )

        retained["retained_in_cleaned"] = True

        # Add retained to cleaned_df in one go
        cleaned_df = pd.concat([cleaned_df, retained], ignore_index=True)

        # Update flag in dups
        dups.loc[retained.index, "retained_in_cleaned"] = True

        return cleaned_df.reset_index(drop=True), dups.reset_index(drop=True), std_per_group

    return cleaned_df.reset_index(drop=True), dups.reset_index(drop=True)



def find_duplicates(
    dfs,
    smiles_col="SMILES",
    raw_col="raw_activity",
    bin_col="activity",
    df_names=None
):
    """
    Detects SMILES duplicated across or within dfs.
    Removes these SMILES from cleaned dfs and summarizes all instances in tall format.

    Args:
        dfs (list of pd.DataFrame): Input dataframes.
        smiles_col (str): Name of SMILES column.
        raw_col (str): Name of raw activity column.
        bin_col (str): Name of binarized activity column.
        df_names (list of str, optional): Names for each dataframe. If None, uses 'df0', 'df1', etc.

    Returns:
        cleaned_dfs (list of pd.DataFrame): List of input dfs with duplicated SMILES removed.
        duplicates_summary (pd.DataFrame): Tall-format table of duplicate SMILES and their activities.
    """
    if df_names is None:
        df_names = [f"df{i}" for i in range(len(dfs))]
    if len(df_names) != len(dfs):
        raise ValueError("Length of df_names must match number of dfs.")

    # Step 1: Collect all SMILES and source info
    smiles_all = []
    source_info = []
    raw_vals = []
    bin_vals = []

    for idx, df in enumerate(dfs):
        for i, row in df.iterrows():
            smiles_all.append(row[smiles_col])
            source_info.append(df_names[idx])
            raw_vals.append(row.get(raw_col, pd.NA))
            bin_vals.append(row.get(bin_col, pd.NA))

    # Step 2: Identify duplicated SMILES (across and within dfs)
    smiles_series = pd.Series(smiles_all)
    duplicated_smiles = smiles_series[smiles_series.duplicated(keep=False)]

    # Step 3: Build the tall-format duplicates summary
    data = []
    for smi, src_df, raw, bin_ in zip(smiles_all, source_info, raw_vals, bin_vals):
        if smi in duplicated_smiles.values:
            data.append({
                smiles_col: smi,
                "source_df": src_df,
                raw_col: raw,
                bin_col: bin_
            })

    duplicates_summary = pd.DataFrame(data)

    # Step 4: Remove duplicated SMILES from the original dfs
    duplicated_set = set(duplicated_smiles)
    cleaned_dfs = []
    for df in dfs:
        cleaned_df = df[~df[smiles_col].isin(duplicated_set)].reset_index(drop=True)
        cleaned_dfs.append(cleaned_df)

    return cleaned_dfs + [duplicates_summary]



def run_deduped_sanity_checks(
    original_dfs,
    cleaned_dfs,
    duplicates_summary,
    smiles_col="SMILES"
):
    """
    Performs sanity checks:
    - Total number of rows is conserved.
    - No duplicate SMILES remain in cleaned dfs.
    - All SMILES in the duplicates summary were truly duplicated originally.

    Args:
        original_dfs (list of pd.DataFrame): Original input dfs (before cleaning).
        cleaned_dfs (list of pd.DataFrame): Cleaned dfs (after removing duplicates).
        duplicates_summary (pd.DataFrame): Summary df (tall format).
        smiles_col (str): Column name for SMILES (default: 'SMILES').
    """

    # 1. Check total rows are conserved
    total_original = sum(len(df) for df in original_dfs)
    total_cleaned = sum(len(df) for df in cleaned_dfs)
    total_duplicates = len(duplicates_summary)
    assert total_original == (total_cleaned + total_duplicates), (
        f"Row count mismatch! "
        f"Original: {total_original}, Cleaned + Duplicates: {total_cleaned + total_duplicates}"
    )

    # 2. Check no duplicates remain in cleaned dfs
    all_cleaned_smiles = pd.concat([df[[smiles_col]] for df in cleaned_dfs])
    duplicated_in_cleaned = all_cleaned_smiles.duplicated(subset=smiles_col).sum()
    assert duplicated_in_cleaned == 0, f"There are still {duplicated_in_cleaned} duplicate SMILES in cleaned dfs!"

    # 3. Check all SMILES in summary were truly duplicated
    all_original_smiles = pd.concat([df[[smiles_col]] for df in original_dfs])
    duplicated_in_original = all_original_smiles[all_original_smiles.duplicated(keep=False)][smiles_col].unique()
    summary_smiles = set(duplicates_summary[smiles_col])

    missing_from_original = summary_smiles.difference(set(duplicated_in_original))
    assert not missing_from_original, f"Some SMILES in summary were not duplicated in original dfs: {missing_from_original}"

    print("✅ Sanity checks passed: all rows accounted for, no leftover duplicates, and summary matches actual duplicates.")



def drop_missing_in_column(input_csv_path, output_csv_path, column_name):
    """
    Drops rows where the specified column has missing (NaN) values and saves the cleaned CSV.

    Parameters:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the cleaned CSV file.
        column_name (str): Name of the column to check for missing values.
    """
    # Load CSV
    df = pd.read_csv(input_csv_path)

    # Drop rows where the specified column is NaN
    df_cleaned = df.dropna(subset=[column_name])

    # Save cleaned DataFrame
    df_cleaned.to_csv(output_csv_path, index=False)

# Example usage:
# drop_missing_in_column("raw_data.csv", "cleaned_data.csv", "SMILES")


# Example usage:
# drop_missing_entries("raw_data.csv", "cleaned_data.csv")


def add_binary_indicator(input_csv, output_csv, column_to_check, threshold=0.2,
                         new_column_name="binary_indicator", direction="below"):
    """
    Adds a binary column to indicate whether the value in a specified column is
    above or below a threshold.

    Parameters:
    - input_csv (str): Path to the input CSV file.
    - output_csv (str): Path to save the updated CSV.
    - column_to_check (str): Name of the column to evaluate.
    - threshold (float): Threshold to compare against (default is 0.2).
    - new_column_name (str): Name of the new binary column.
    - direction (str): 'below' or 'above' to indicate comparison direction.
    """
    df = pd.read_csv(input_csv)

    if direction == "below":
        df[new_column_name] = (df[column_to_check] < threshold).astype(int)
    elif direction == "above":
        df[new_column_name] = (df[column_to_check] > threshold).astype(int)
    else:
        raise ValueError("direction must be either 'above' or 'below'")

    df.to_csv(output_csv, index=False)



def mark_presence(df1, df2, col1, col2, new_col_name='presence'):
    """
    Adds a column to df1 with 1 if the value in col1 appears in df2[col2], 0 otherwise.

    Args:
        df1 (pd.DataFrame): The dataframe to annotate.
        df2 (pd.DataFrame): The reference dataframe.
        col1 (str): Column name in df1 to check.
        col2 (str): Column name in df2 to compare against.
        new_col_name (str): Name of the new column to be added to df1.

    Returns:
        pd.DataFrame: df1 with an additional column indicating presence.
    """
    reference_values = set(df2[col2].dropna())
    df1[new_col_name] = df1[col1].isin(reference_values).astype(int)
    return df1


def filter_csv(input_path, filters=None, output_path=None):
    """
    Reads a CSV file, filters by column values, and optionally saves the result to another CSV.

    Parameters:
        input_path (str): Path to the input CSV file.
        filters (dict, optional): Dictionary of filters where keys are column names and values are the values to filter by.
                                  If a value is a list, set, or tuple, rows matching any of the values are kept.
        output_path (str, optional): Path to save the filtered DataFrame as a CSV. If None, no file is saved.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    df = pd.read_csv(input_path)

    if filters:
        for col, val in filters.items():
            if isinstance(val, (list, set, tuple)):
                df = df[df[col].isin(val)]
            else:
                df = df[df[col] == val]

    if output_path:
        df.to_csv(output_path, index=False)

    return df

def compute_mol_weight(smiles):
    """
    Computes the molecular weight of a molecule given a SMILES string.

    Parameters:
        smiles (str): SMILES representation of the molecule.

    Returns:
        float or None: Molecular weight, or None if SMILES is invalid.
    """
    if pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return Descriptors.MolWt(mol) if mol else None
    except:
        return None

def add_molecular_weight_column(df, smiles_col='SMILES',mol_weight_col='mol_weight'):
    """
    Adds a 'mol_weight' column to the DataFrame using SMILES strings.

    Parameters:
        df (pd.DataFrame): DataFrame with a column of SMILES strings.
        smiles_col (str): Name of the column containing SMILES strings.

    Returns:
        pd.DataFrame: DataFrame with added 'mol_weight' column.
    """
    df[mol_weight_col] = df[smiles_col].apply(compute_mol_weight)
    return df


def drop_unnamed_columns(filepath):
    """
    Reads a CSV file, drops all 'Unnamed' columns, and saves it back to the same file.

    Parameters:
        filepath (str): Path to the CSV file.
    """
    df = pd.read_csv(filepath)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.to_csv(filepath, index=False)


def parse_concentration(conc_str):
    """
    Parses a concentration string like '32 µg/mL' or '1 mg/mL' and returns (value, unit).
    """
    match = re.match(r'([\d\.]+)\s*([^\s]+)', str(conc_str))
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        return value, unit
    return None, None

def convert_to_molar(conc_str=None, mol_weight=None, value=None, unit=None):
    """
    Converts a concentration (value and unit) and molecular weight to molar concentration (mol/L).
    Accepts either:
      - a single concentration string (e.g., '32 µg/mL')
      - or separate value and unit columns (e.g., 32 and 'ug.mL-1')
    """
    if value is None or unit is None:
        value, unit = parse_concentration(conc_str)
    if value is None or mol_weight is None or pd.isna(mol_weight):
        return None

    # Normalize the unit string
    unit = str(unit).lower()
    unit = unit.replace('μ', 'u').replace('µ', 'u')  # fix micro symbol
    unit = unit.replace('.ml-1', '/ml')              # convert 'ug.mL-1' -> 'ug/ml'

    # Convert to grams per liter
    if unit == 'ug/ml':
        grams_per_liter = value * 1e-6 * 1000
    elif unit == 'mg/ml':
        grams_per_liter = value * 1e-3 * 1000
    elif unit == 'g/ml':
        grams_per_liter = value * 1000
    elif unit == 'nM':
        return value * 1e-9  # molarity in mol/L
    else:
        return None  # unsupported unit

    molarity = grams_per_liter / mol_weight
    return molarity


def add_molar_concentration_column(
    df,
    mol_weight_col='mol_weight',
    conc_col='concentration',
    value_col=None,
    unit_col=None,
    out_col = 'molar_concentration'
):
    """
    Adds a column 'molar_concentration' based on molecular weight and either:
      - a combined concentration string column (e.g., '32 µg/mL'), or
      - separate value and unit columns.
    """
    if value_col and unit_col:
        df[out_col] = df.apply(
            lambda row: convert_to_molar(
                mol_weight=row[mol_weight_col],
                value=row[value_col],
                unit=row[unit_col]
            ), axis=1
        )
    else:
        df[out_col] = df.apply(
            lambda row: convert_to_molar(
                conc_str=row[conc_col],
                mol_weight=row[mol_weight_col]
            ), axis=1
        )
    return df



def standardize(smiles):
    """
    Standardize a SMILES string using RDKit with error handling.
    Removes explicit hydrogens to prevent kekulization issues.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Remove explicit hydrogens
        mol = Chem.RemoveHs(mol)

        # Cleanup molecule
        clean_mol = rdMolStandardize.Cleanup(mol)
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
        uncharger = rdMolStandardize.Uncharger()
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

        # Canonicalize tautomers
        te = rdMolStandardize.TautomerEnumerator()
        canonical_mol = te.Canonicalize(uncharged_parent_clean_mol)

        if canonical_mol:
            return Chem.MolToSmiles(canonical_mol, canonical=True)
        else:
            return None

    except (RuntimeError, ValueError, TypeError) as e:
        print(f"Error standardizing SMILES '{smiles}': {e}")
        return None  # Return None on error

def standardize_csv_smiles(input_path, output_path,smiles_col='SMILES'):
    df = pd.read_csv(input_path)
    df["standardized_SMILES"] = df[smiles_col].apply(standardize)
    df.to_csv(output_path,index=False)
    




# import os
# import pandas as pd
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem.MolStandardize import rdMolStandardize

# def standardize(smiles):
#     """
#     Standardize a SMILES string using RDKit with error handling.
#     Removes explicit hydrogens to prevent kekulization issues.
#     """
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             raise ValueError(f"Invalid SMILES: {smiles}")

#         # **Remove explicit hydrogens early to prevent kekulization errors**
#         mol = Chem.RemoveHs(mol)

#         # Cleanup: remove Hs, disconnect metals, normalize, reionize
#         clean_mol = rdMolStandardize.Cleanup(mol)

#         # Get the parent fragment
#         parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

#         # Neutralize molecule
#         uncharger = rdMolStandardize.Uncharger()
#         uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

#         # Canonicalize tautomers
#         te = rdMolStandardize.TautomerEnumerator()
#         canonical_mol = te.Canonicalize(uncharged_parent_clean_mol)

#         # return Chem.MolToSmiles(canonical_mol, canonical=True), True if canonical_mol else 'smiles', False
#         if canonical_mol:
#             return Chem.MolToSmiles(canonical_mol, canonical=True)
#         else:
#             return None

#     except (RuntimeError, ValueError, TypeError) as e:
#         print(f"Error standardizing SMILES '{smiles}': {e}")
#         return e

# n_sources = len(os.listdir('sources'))
# for i, vendor in enumerate(sorted(os.listdir('sources'))):
#     if not vendor.startswith('.'):
#         print(f'working on vendor {vendor}, {i+1} of {n_sources}')
#         n_files = len(os.listdir(f'sources/{vendor}/CSV'))
#         for j, file in enumerate(sorted(os.listdir(f'sources/{vendor}/CSV'))):
#             if not file.startswith('.'):
#                 print(f'working on file {file}, {j+1} of {n_files})')
#                 in_df = pd.read_csv(f'sources/{vendor}/CSV/{file}')
#                 # in_df = in_df['SMILES'].apply(standardize)
#                 in_df['standardized_SMILES'] = in_df['SMILES'].map(standardize)
#                 in_df['vendor'] = [vendor]*len(in_df)
#                 in_df.to_csv(f'outfiles/standardized_{file}')

# in_df = pd.read_csv('unprocessed.csv')
# in_df['error'] = in_df['SMILES'].map(standardize)
# in_df.to_csv('unprocessed_errors.csv')   
