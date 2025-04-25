import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import Descriptors
import re


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

def add_molecular_weight_column(df, smiles_col='SMILES'):
    """
    Adds a 'mol_weight' column to the DataFrame using SMILES strings.

    Parameters:
        df (pd.DataFrame): DataFrame with a column of SMILES strings.
        smiles_col (str): Name of the column containing SMILES strings.

    Returns:
        pd.DataFrame: DataFrame with added 'mol_weight' column.
    """
    df['mol_weight'] = df[smiles_col].apply(compute_mol_weight)
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
    else:
        return None  # unsupported unit

    molarity = grams_per_liter / mol_weight
    return molarity


def add_molar_concentration_column(
    df,
    mol_weight_col='mol_weight',
    conc_col='concentration',
    value_col=None,
    unit_col=None
):
    """
    Adds a column 'molar_concentration' based on molecular weight and either:
      - a combined concentration string column (e.g., '32 µg/mL'), or
      - separate value and unit columns.
    """
    if value_col and unit_col:
        df['molar_concentration'] = df.apply(
            lambda row: convert_to_molar(
                mol_weight=row[mol_weight_col],
                value=row[value_col],
                unit=row[unit_col]
            ), axis=1
        )
    else:
        df['molar_concentration'] = df.apply(
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
