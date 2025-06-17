#!/usr/bin/env python3
"""
clean_data.py

Load raw insurance data, clean types, handle missing values, derive key columns, and save cleaned DataFrame.
"""

import argparse
import os
import pandas as pd
import numpy as np

def load_raw(path):
    """Load the raw pipe-separated file into a DataFrame."""
    df = pd.read_csv(path, sep='|', low_memory=False)
    return df

def clean_column_names(df):
    """Strip whitespace from column names."""
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    return df

def parse_dates(df):
    """Convert TransactionMonth to datetime; create YearMonth period."""
    df = df.copy()
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
        df['YearMonth'] = df['TransactionMonth'].dt.to_period('M')
    return df

def convert_numeric(df, numeric_cols):
    """Convert specified columns to numeric, coercing errors to NaN."""
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def convert_categoricals(df, categorical_cols):
    """Convert specified columns to category dtype."""
    df = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({'': np.nan, 'nan': np.nan})
            df[col] = df[col].astype('category')
    return df

def derive_metrics(df):
    """Derive HasClaim, LossRatio, Margin."""
    df = df.copy()
    if 'TotalClaims' in df.columns:
        df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce').fillna(0)
    else:
        df['TotalClaims'] = 0.0
    if 'TotalPremium' in df.columns:
        df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce').fillna(0)
    else:
        df['TotalPremium'] = 0.0

    df['HasClaim'] = df['TotalClaims'] > 0
    df['LossRatio'] = np.where(df['TotalPremium'] > 0,
                               df['TotalClaims'] / df['TotalPremium'],
                               np.nan)
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    return df

def handle_missing(df):
    """Handle missing values; leave NaNs for downstream processing."""
    df = df.copy()
    return df

def save_clean(df, output_path):
    """Save cleaned DataFrame to parquet."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Clean raw insurance data")
    parser.add_argument('--input', required=True, help="Path to raw data file (MachineLearningRating_v3.txt)")
    parser.add_argument('--output', required=True, help="Path to save cleaned parquet file (e.g., data/processed/df_clean.parquet)")
    args = parser.parse_args()

    df = load_raw(args.input)
    print(f"Loaded raw data with shape: {df.shape}")

    df = clean_column_names(df)
    df = parse_dates(df)

    numeric_cols = [
        'Cylinders', 'cubiccapacity', 'kilowatts',
        'CustomValueEstimate', 'TotalPremium', 'TotalClaims'
    ]
    df = convert_numeric(df, numeric_cols)

    categorical_cols = [
        'IsVATRegistered', 'Citizenship', 'LegalType', 'Title',
        'Language', 'Bank', 'AccountType', 'MaritalStatus',
        'Gender', 'Country', 'Province', 'PostalCode',
        'MainCrestaZone', 'SubCrestaZone', 'ItemType', 'VehicleType',
        'make', 'Model', 'bodytype', 'TermFrequency', 'ExcessSelected',
        'CoverCategory', 'CoverType', 'CoverGroup', 'Section', 'Product',
        'StatutoryClass', 'StatutoryRiskType'
    ]
    df = convert_categoricals(df, categorical_cols)

    df = derive_metrics(df)
    df = handle_missing(df)

    save_clean(df, args.output)

if __name__ == "__main__":
    main()
