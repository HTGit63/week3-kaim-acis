"""
train_model.py

Train classification model for claim occurrence and regression model for claim severity.
Save models and evaluation metrics; compute expected cost on test set.
"""

import argparse
import os
import json
import joblib
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
  accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
  mean_squared_error, r2_score
)

def train_classification(X_train, y_train, X_test, y_test, params):
  rf_params = params.get('random_forest', {})
  clf = RandomForestClassifier(
      random_state=params.get('random_state', 42),
      **rf_params
  )
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  y_prob = clf.predict_proba(X_test)[:, 1]
  metrics = {
      'accuracy': accuracy_score(y_test, y_pred),
      'precision': precision_score(y_test, y_pred, zero_division=0),
      'recall': recall_score(y_test, y_pred, zero_division=0),
      'f1': f1_score(y_test, y_pred, zero_division=0),
      'roc_auc': roc_auc_score(y_test, y_prob)
  }
  return clf, metrics

def train_regression(X_train, y_train, X_test, y_test, params):
  rf_params = params.get('random_forest', {})
  regr = RandomForestRegressor(
      random_state=params.get('random_state', 42),
      **rf_params
  )
  regr.fit(X_train, y_train)
  y_pred = regr.predict(X_test)
  metrics = {
      'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
      'r2': r2_score(y_test, y_pred)
  }
  return regr, metrics

def main():
  parser = argparse.ArgumentParser(description="Train insurance claim models")
  parser.add_argument('--input', required=True, help="Path to features parquet")
  parser.add_argument('--output_dir', required=True, help="Directory to save models and metrics")
  parser.add_argument('--params', default="params.yaml", help="YAML file with model params")
  args = parser.parse_args()

  # Load parameters
  with open(args.params) as f:
      full_params = yaml.safe_load(f)
  model_params = full_params.get('model', {})
  test_size = model_params.get('test_size', 0.2)
  random_state = model_params.get('random_state', 42)

  # Load features
  df = pd.read_parquet(args.input)
  # Prepare classification data
  if 'HasClaim' not in df.columns:
      raise ValueError("Features DataFrame must contain 'HasClaim' column")
  X = df.drop(columns=['HasClaim', 'TotalClaims'], errors='ignore')
  y_cls = df['HasClaim']
  # Split with stratify to maintain class balance
  X_train, X_test, y_train, y_test = train_test_split(
      X, y_cls, test_size=test_size, random_state=random_state, stratify=y_cls
  )
  os.makedirs(args.output_dir, exist_ok=True)

  # Classification training
  clf, cls_metrics = train_classification(X_train, y_train, X_test, y_test, model_params)
  joblib.dump(clf, os.path.join(args.output_dir, "clf_model.pkl"))
  with open(os.path.join(args.output_dir, "classification_metrics.json"), 'w') as f:
      json.dump(cls_metrics, f, indent=2)
  print(f"Saved classification model and metrics to {args.output_dir}")

  # Regression training on positive claims
  df_pos = df[df['HasClaim'] == 1]
  if len(df_pos) > 50:  # ensure enough samples
      X_reg = df_pos.drop(columns=['HasClaim', 'TotalClaims'], errors='ignore')
      y_reg = df_pos['TotalClaims']
      X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
          X_reg, y_reg, test_size=test_size, random_state=random_state
      )
      regr, regr_metrics = train_regression(X_train_r, y_train_r, X_test_r, y_test_r, model_params)
      joblib.dump(regr, os.path.join(args.output_dir, "regr_model.pkl"))
      with open(os.path.join(args.output_dir, "regression_metrics.json"), 'w') as f:
          json.dump(regr_metrics, f, indent=2)
      print(f"Saved regression model and metrics to {args.output_dir}")
  else:
      print("Not enough positive claim samples for regression training.")

  # Expected cost evaluation (if both models exist)
  clf_path = os.path.join(args.output_dir, "clf_model.pkl")
  regr_path = os.path.join(args.output_dir, "regr_model.pkl")
  if os.path.exists(clf_path) and os.path.exists(regr_path):
      clf = joblib.load(clf_path)
      regr = joblib.load(regr_path)
      # Use full X_test from classification stage
      prob = clf.predict_proba(X_test)[:, 1]
      # Predict severity for all X_test rows (for policies predicted to have claim or to compute expectation)
      # In practice, severity model expects features for positive cases; here we predict for all and multiply by prob
      sev_pred = regr.predict(X_test)
      expected_cost = prob * sev_pred
      df_expected = pd.DataFrame({
          'prob_claim': prob,
          'pred_severity': sev_pred,
          'expected_cost': expected_cost,
          # actual_totalclaims might be missing for some: align indices
          'actual_totalclaims': y_test.values
      })
      df_expected.to_csv(os.path.join(args.output_dir, "expected_cost_evaluation.csv"), index=False)
      print(f"Saved expected cost evaluation to {args.output_dir}/expected_cost_evaluation.csv")
  else:
      print("Skipping expected cost evaluation: model files not found.")

if __name__ == "__main__":
  main()
