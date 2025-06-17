#!/usr/bin/env python3
"""
stat_tests.py

Perform statistical tests on cleaned insurance data:
- Claim frequency differences (two-proportion z-test)
- Claim severity differences (two-sample t-test) on positive claims
- Margin differences (t-test) by group
Saves CSV of results and markdown summary.
"""

import argparse
import os
import pandas as pd
import numpy as np
from scipy import stats
import yaml

def test_proportion(df, group_col, value_col, groupA, groupB, alpha=0.05):
  dfA = df[df[group_col] == groupA]
  dfB = df[df[group_col] == groupB]
  nA = len(dfA)
  nB = len(dfB)
  # If too small, skip
  if nA == 0 or nB == 0:
      return None
  pA = dfA[value_col].mean()
  pB = dfB[value_col].mean()
  # pooled proportion
  p_pool = (dfA[value_col].sum() + dfB[value_col].sum()) / (nA + nB)
  se = np.sqrt(p_pool * (1 - p_pool) * (1/nA + 1/nB))
  if se == 0:
      return None
  z = (pA - pB) / se
  p_value = 2 * (1 - stats.norm.cdf(abs(z)))
  return {
      'test': 'frequency',
      'group_col': group_col,
      'groupA': groupA,
      'groupB': groupB,
      'nA': nA,
      'nB': nB,
      'metricA': pA,
      'metricB': pB,
      'statistic': z,
      'p_value': p_value,
      'reject_null': p_value < alpha
  }

def test_mean(df, group_col, value_col, groupA, groupB, alpha=0.05):
  dfA = df[df[group_col] == groupA][value_col].dropna()
  dfB = df[df[group_col] == groupB][value_col].dropna()
  if len(dfA) < 2 or len(dfB) < 2:
      return None
  # Welch's t-test
  try:
      tstat, p_value = stats.ttest_ind(dfA, dfB, equal_var=False, nan_policy='omit')
  except Exception:
      return None
  return {
      'test': f'mean_{value_col}',
      'group_col': group_col,
      'groupA': groupA,
      'groupB': groupB,
      'nA': len(dfA),
      'nB': len(dfB),
      'metricA': dfA.mean(),
      'metricB': dfB.mean(),
      'statistic': tstat,
      'p_value': p_value,
      'reject_null': p_value < alpha
  }

def run_tests(df, output_dir, alpha=0.05):
  os.makedirs(output_dir, exist_ok=True)
  results = []
  # Load group choices from params.yaml if exists
  # For simplicity, compare top 2 provinces and top 2 postal codes
  # 1. Province tests
  if 'Province' in df.columns:
      counts = df['Province'].value_counts().dropna()
      if len(counts) >= 2:
          top_provs = counts.index[:2].tolist()
          res = test_proportion(df, 'Province', 'HasClaim', top_provs[0], top_provs[1], alpha)
          if res: results.append(res)
          # severity on positive claims
          df_pos = df[df['HasClaim']]
          res2 = test_mean(df_pos, 'Province', 'TotalClaims', top_provs[0], top_provs[1], alpha)
          if res2: results.append(res2)
  # 2. PostalCode tests
  if 'PostalCode' in df.columns:
      counts = df['PostalCode'].value_counts().dropna()
      if len(counts) >= 2:
          top_zips = counts.index[:2].tolist()
          res = test_proportion(df, 'PostalCode', 'HasClaim', top_zips[0], top_zips[1], alpha)
          if res: results.append(res)
          df_pos = df[df['HasClaim']]
          res2 = test_mean(df_pos, 'PostalCode', 'TotalClaims', top_zips[0], top_zips[1], alpha)
          if res2: results.append(res2)
          # margin test on full data
          res3 = test_mean(df, 'PostalCode', 'Margin', top_zips[0], top_zips[1], alpha)
          if res3: results.append(res3)
  # 3. Gender tests (exclude 'Not specified' if possible)
  if 'Gender' in df.columns:
      # Find categories with sufficient size
      counts = df['Gender'].value_counts().dropna()
      valid = [g for g, cnt in counts.items() if g != 'Not specified' and cnt > 1000]
      if len(valid) >= 2:
          res = test_proportion(df, 'Gender', 'HasClaim', valid[0], valid[1], alpha)
          if res: results.append(res)
          df_pos = df[df['HasClaim']]
          res2 = test_mean(df_pos, 'Gender', 'TotalClaims', valid[0], valid[1], alpha)
          if res2: results.append(res2)
  # 4. Additional tests could be added via params.yaml
  # Convert to DataFrame and save
  if not results:
      print("No tests were run; check data and thresholds.")
  df_res = pd.DataFrame(results)
  # Save CSV
  out_csv = os.path.join(output_dir, "stat_tests_results.csv")
  df_res.to_csv(out_csv, index=False)
  # Save markdown summary
  summary = os.path.join(output_dir, "README.md")
  with open(summary, 'w') as f:
      f.write("# Statistical Test Results\n\n")
      for _, row in df_res.iterrows():
          metric = row['metricA']
          metricB = row['metricB']
          f.write(f"- **{row['group_col']}**: {row['groupA']} vs {row['groupB']}, "
                  f"stat={row['statistic']:.4f}, p={row['p_value']:.4f}, "
                  f"reject_null={row['reject_null']}\n")
  print(f"Saved test results to {out_csv} and summary to {summary}")

def main():
  parser = argparse.ArgumentParser(description="Run statistical tests on cleaned insurance data")
  parser.add_argument('--input', required=True, help="Path to cleaned data parquet file")
  parser.add_argument('--output_dir', required=True, help="Directory to save test results")
  parser.add_argument('--alpha', type=float, default=0.05, help="Significance level")
  args = parser.parse_args()

  df = pd.read_parquet(args.input)
  if 'HasClaim' not in df.columns:
      df['HasClaim'] = df['TotalClaims'] > 0
  run_tests(df, args.output_dir, alpha=args.alpha)

if __name__ == "__main__":
  main()
