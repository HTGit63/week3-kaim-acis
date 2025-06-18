#!/usr/bin/env python3
"""
interpretability.py

Use SHAP to explain the classification model for claim occurrence.
Saves summary and dependence plots of top features.

This version aligns input features to the model's expected features (via feature_names_in_),
samples for faster SHAP, and logs progress.
"""

import argparse
import os
import joblib
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import time

def align_features(df, feature_names):
    """
    Given a DataFrame df and a list of feature_names (model expects),
    return a DataFrame with exactly these columns, in this order.
    - If a column from feature_names is missing in df, fill with 0.
    - If df has extra columns, drop them.
    """
    # Initialize new DataFrame with zeros for missing columns
    # Use dtype float for numeric features; if categorical features encoded as 0/1, float is fine
    aligned = pd.DataFrame(0, index=df.index, columns=feature_names)
    # For columns present in both df and feature_names, copy values
    common = [col for col in feature_names if col in df.columns]
    if common:
        aligned[common] = df[common]
    # If none common, aligned remains zeros; downstream SHAP may be less meaningful, but we avoid errors
    missing = [col for col in feature_names if col not in df.columns]
    if missing:
        print(f"[WARNING] These expected features were missing in input and filled with 0: {missing[:5]}{'...' if len(missing)>5 else ''}")
    # Drop extra columns from df (not in feature_names) is implicit, since we only copied common columns
    return aligned

def explain_model(clf, X_orig, output_dir, sample_size=1000):
    os.makedirs(output_dir, exist_ok=True)

    # Determine model’s expected feature names
    if hasattr(clf, 'feature_names_in_'):
        feature_names = list(clf.feature_names_in_)
        print(f"[INFO] Model expects {len(feature_names)} features.")
        # Align original X_orig to these features
        X_aligned = align_features(X_orig, feature_names)
    else:
        # Cannot retrieve feature_names_in_; fall back to X_orig columns
        print("[WARNING] Model does not have attribute feature_names_in_. Proceeding with X.columns, but SHAP-feature alignment may fail.")
        feature_names = list(X_orig.columns)
        X_aligned = X_orig.copy()

    # Sample for SHAP
    n_rows = len(X_aligned)
    if n_rows > sample_size:
        X_sample = X_aligned.sample(n=sample_size, random_state=42)
        print(f"[INFO] Sampling {sample_size} rows from {n_rows} for SHAP.")
    else:
        X_sample = X_aligned
        print(f"[INFO] Using full dataset of {n_rows} rows for SHAP.")

    # Start timing SHAP
    start_time = time.time()

    # Compute SHAP values
    print("[INFO] Computing SHAP values with TreeExplainer...")
    explainer = shap.TreeExplainer(clf)
    try:
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        print(f"[ERROR] SHAP computation failed: {e}")
        return

    # Handle classifier multi-output
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    duration = time.time() - start_time
    print(f"[INFO] SHAP values computed in {duration:.2f} seconds. Shape: {shap_vals.shape}")

    # After alignment, shap_vals.shape[1] should equal len(feature_names)
    if shap_vals.ndim != 2:
        print(f"[ERROR] Unexpected SHAP values dimension: {shap_vals.ndim}D array.")
        return
    n_feat_shap = shap_vals.shape[1]
    if n_feat_shap != len(feature_names):
        print(f"[WARNING] SHAP returned {n_feat_shap} features but expected {len(feature_names)}.")
        # We proceed by truncating or padding feature_names to match shap_vals shape:
        if n_feat_shap < len(feature_names):
            # Fewer SHAP columns: truncate feature_names
            feature_names_used = feature_names[:n_feat_shap]
            print(f"[INFO] Truncating feature names list to first {n_feat_shap}.")
        else:
            # More SHAP columns: create generic names for extra
            extra = n_feat_shap - len(feature_names)
            feature_names_used = feature_names + [f"extra_feat_{i}" for i in range(extra)]
            print(f"[INFO] Extending feature names with {extra} generic names.")
    else:
        feature_names_used = feature_names

    # Prepare DataFrame for plotting: only if shapes match
    if shap_vals.shape[1] == X_sample.shape[1]:
        plot_data = X_sample
    else:
        plot_data = None
        print("[WARNING] Input DataFrame columns do not match SHAP feature dimension; summary plot will use only feature_names.")

    # SHAP summary plot
    print("[INFO] Generating SHAP summary plot...")
    plt.figure()
    try:
        shap.summary_plot(shap_vals, plot_data, feature_names=feature_names_used, show=False)
    except Exception as e:
        print(f"[WARNING] summary_plot raised exception: {e}. Trying without plot_data...")
        try:
            shap.summary_plot(shap_vals, feature_names=feature_names_used, show=False)
        except Exception as e2:
            print(f"[ERROR] summary_plot failed again: {e2}. Skipping summary plot.")
            plt.close()
            summary_path = None
        else:
            plt.tight_layout()
            summary_path = os.path.join(output_dir, "shap_summary.png")
            plt.savefig(summary_path)
            plt.close()
    else:
        plt.tight_layout()
        summary_path = os.path.join(output_dir, "shap_summary.png")
        plt.savefig(summary_path)
        plt.close()

    if summary_path:
        print(f"[SUCCESS] Saved SHAP summary plot to {summary_path}")

    # Identify top feature by mean absolute SHAP value
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx = int(np.argmax(mean_abs))
    if top_idx < len(feature_names_used):
        top_feature = feature_names_used[top_idx]
    else:
        top_feature = f"feature_{top_idx}"
    print(f"[INFO] Top feature by mean|SHAP|: '{top_feature}' (index {top_idx})")

    # SHAP dependence plot for top feature
    print(f"[INFO] Generating dependence plot for '{top_feature}'...")
    plt.figure()
    try:
        shap.dependence_plot(top_feature, shap_vals, plot_data, feature_names=feature_names_used, show=False)
    except Exception as e:
        print(f"[WARNING] dependence_plot raised exception: {e}. Skipping dependence plot.")
        plt.close()
        dep_path = None
    else:
        plt.tight_layout()
        dep_path = os.path.join(output_dir, f"shap_dependence_{top_feature}.png")
        plt.savefig(dep_path)
        plt.close()
    if dep_path:
        print(f"[SUCCESS] Saved SHAP dependence plot to {dep_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate SHAP interpretability plots for classification model")
    parser.add_argument('--model_path', required=True, help="Path to trained classifier model (.pkl)")
    parser.add_argument('--input', required=True, help="Path to features parquet file (same features used in training)")
    parser.add_argument('--output_dir', required=True, help="Directory to save SHAP plots")
    parser.add_argument('--sample_size', type=int, default=1000, help="Number of rows to use for SHAP (default: 1000)")
    args = parser.parse_args()

    print("[INFO] Loading model...")
    clf = joblib.load(args.model_path)

    print("[INFO] Loading features DataFrame...")
    df = pd.read_parquet(args.input)

    # Drop target columns to isolate features
    if 'HasClaim' in df.columns or 'TotalClaims' in df.columns:
        drop_cols = [c for c in ['HasClaim', 'TotalClaims'] if c in df.columns]
        X = df.drop(columns=drop_cols, errors='ignore')
    else:
        X = df

    explain_model(clf, X, args.output_dir, sample_size=args.sample_size)

if __name__ == "__main__":
    main()
