## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/HTGit63/week3-kaim-acis.git
cd week3-kaim-acis
```

### 2. Set Up Python Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Retrieve DVC Data
```bash
dvc pull
```

### 4. Run Full Pipeline
```bash
dvc repro
```

### 5. Inspect Results
- **EDA Figures:** Located in `results/eda/`
- **Statistical Test Outputs:**  
    - CSV: `results/stat_tests/stat_tests_results.csv`  
    - Additional details: `results/stat_tests/README.md`
- **Modeling Metrics:**  
    - Classification metrics: `results/models/classification_metrics.json`  
    - Regression metrics: `regression_metrics.json`  
    - Cost evaluation: `expected_cost_evaluation.csv`
- **Interpretability Plots:** e.g., `results/interpretability/shap_summary.png`

### 6. Open Jupyter Notebook
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### 7. Run Individual Stages Manually
- **Data Cleaning:**
```bash
python scripts/clean_data.py --input data/raw/MachineLearningRating_v3.txt --output data/processed/df_clean.parquet
```

- **Statistical Testing:**
```bash
python scripts/stat_tests.py --input data/processed/df_clean.parquet --output_dir results/stat_tests
```

- **Feature Engineering:**
```bash
python scripts/feature_engineering.py --input data/processed/df_clean.parquet --output data/processed/features.parquet
```

- **Model Training:**
```bash
python scripts/train_model.py --input data/processed/features.parquet --output_dir results/models --params params.yaml
```

- **Model Interpretability:**
```bash
python scripts/interpretability.py --model_path results/models/clf_model.pkl --input data/processed/features.parquet --output_dir results/interpretability
```

Remember to commit these changes once everything is confirmed.