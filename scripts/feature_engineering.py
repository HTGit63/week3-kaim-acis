#!/usr/bin/env python3
"""
feature_engineering.py

Build modeling features from cleaned data.
"""

import argparse
import os
import pandas as pd
import numpy as np

def add_vehicle_age(df):
    """
    Compute vehicle age from VehicleIntroDate and TransactionMonth.
    """
    df = df.copy()
    if 'VehicleIntroDate' in df.columns:
        # Parse VehicleIntroDate to datetime if not already
        df['VehicleIntroDate'] = pd.to_datetime(df['VehicleIntroDate'], errors='coerce')
        # Ensure TransactionMonth exists and is datetime
        if 'TransactionMonth' in df.columns:
            df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
            df['VehicleAge'] = df['TransactionMonth'].dt.year - df['VehicleIntroDate'].dt.year
        else:
            df['VehicleAge'] = np.nan
        # Fill missing ages with median
        median_age = df['VehicleAge'].median()
        df['VehicleAge'] = df['VehicleAge'].fillna(median_age)
    else:
        df['VehicleAge'] = np.nan
    return df

def encode_categoricals(df, categorical_cols):
    """
    One-hot encode specified categorical columns.
    """
    df = df.copy()
    # Before encoding, fill NaN with 'Unknown'
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('Unknown')
    # Use pandas.get_dummies
    df_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)
    return df_encoded

def build_features(df, output_path):
    """
    Build features DataFrame and save to parquet.
    """
    df = df.copy()
    # Ensure HasClaim exists
    if 'HasClaim' not in df.columns:
        df['HasClaim'] = df['TotalClaims'] > 0
    # Add vehicle age
    df = add_vehicle_age(df)
    # Log transform of CustomValueEstimate
    if 'CustomValueEstimate' in df.columns:
        df['CustomValueEstimate'] = pd.to_numeric(df['CustomValueEstimate'], errors='coerce').fillna(0)
        df['log_CustomValueEstimate'] = np.log1p(df['CustomValueEstimate'])
    else:
        df['log_CustomValueEstimate'] = 0.0

    # Numeric features
    numeric_cols = []
    if 'VehicleAge' in df.columns:
        numeric_cols.append('VehicleAge')
    if 'kilowatts' in df.columns:
        df['kilowatts'] = pd.to_numeric(df['kilowatts'], errors='coerce').fillna(df['kilowatts'].median())
        numeric_cols.append('kilowatts')
    if 'Cylinders' in df.columns:
        df['Cylinders'] = pd.to_numeric(df['Cylinders'], errors='coerce').fillna(df['Cylinders'].median())
        numeric_cols.append('Cylinders')
    numeric_cols.append('log_CustomValueEstimate')

    df_numeric = df[numeric_cols].copy()

    # Categorical features to encode
    categorical_cols = []
    for col in ['Province', 'VehicleType', 'CoverCategory', 'TermFrequency']:
        if col in df.columns:
            categorical_cols.append(col)
    df_cat = encode_categoricals(df, categorical_cols)

    # Combine
    df_features = pd.concat([df_numeric.reset_index(drop=True), df_cat.reset_index(drop=True)], axis=1)
    # Add target columns
    df_features['HasClaim'] = df['HasClaim'].astype(int).values
    df_features['TotalClaims'] = df['TotalClaims'].values

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_features.to_parquet(output_path, index=False)
    print(f"Saved features to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Feature engineering for insurance modeling")
    parser.add_argument('--input', required=True, help="Path to cleaned data parquet")
    parser.add_argument('--output', required=True, help="Path to save features parquet")
    args = parser.parse_args()
    df = pd.read_parquet(args.input)
    build_features(df, args.output)

if __name__ == "__main__":
    main()
