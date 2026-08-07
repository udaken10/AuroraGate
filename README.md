# AuroraGate Expense Categorization Challenge 🏆

This repository contains the solution pipeline, model architectures, and ensembling strategies for the **AuroraGate Expense Categorization Challenge**.

## 📌 Features & Architecture Overview

The solution combines multi-scale text feature extraction, domain domain rules, Bayesian target encoding, and ensembling across 7 distinct model families:

1. **Preprocessing & Merchant Extraction**: NFKC normalization, US state code regex removal, store unit number extraction, and merchant root identification.
2. **80+ Domain Merchant Rules**: High-precision rule mapping for expense categories (Food & Dining, Transportation, Subscriptions, Groceries, Health & Fitness, Bills & Utilities, Shopping, Entertainment, Travel).
3. **Amount, Cents & Date Features**: Log amount, exact dollar indicators, 99/95/50 cents flags, amount deciles, payday indicators, and sine/cosine day-of-week cyclical features.
4. **Leak-Free Bayesian Target Encoding**: Out-of-fold Stratified 5-Fold target encoding on `merchant_root` with prior smoothing (`smooth_weight=10.0`).
5. **Transformers NLP & Multi-Scale Text Features**:
   - **SentenceTransformers (`all-MiniLM-L6-v2`) 384-dimensional dense semantic embeddings**.
   - Sublinear Word (1-3gram) and Char (3-5gram) TF-IDF vectors.
   - TruncatedSVD (100-dim LSA) dimensionality reduction.
6. **7 Model Families Suite**:
   - **LightGBM**
   - **XGBoost**
   - **CatBoost**
   - **RandomForest Classifier**
   - **Calibrated Linear Classifier** (Logistic Regression / LinearSVC)
   - **MLP Neural Network**
   - **PyTorch Deep & Wide NN**
7. **Scipy SLSQP Constrained Weight Optimization**: Non-negative SLSQP optimization solving for weights $(w_1, \dots, w_7)$ that directly maximize Out-of-Fold Macro F1 score.

## 📂 Repository Structure

- `trust_cv_master_ensemble.ipynb`: Main Trust-CV Jupyter Notebook implementing the complete 5-Fold CV training, Transformers NLP embeddings, SLSQP weight optimization, and post-processing.
- `master_ensemble_pipeline.ipynb`: Full multi-model ensemble notebook.
- `master_ensemble_pipeline.py`: Standalone Python training script.
- `requirements.txt`: Python package dependencies.
- `.gitignore`: Confidential data (`*.csv`, `*.zip`), virtual environments (`.venv/`), and secrets exclusion rules.

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/udaken10/AuroraGate.git
   cd AuroraGate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the master pipeline:
   ```bash
   python master_ensemble_pipeline.py
   ```
