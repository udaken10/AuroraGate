# 🏆 AuroraGate Expense Categorization Challenge Solution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete pipeline, feature engineering framework, multi-model suite, and Trust-CV ensemble for the **AuroraGate Expense Categorization Challenge**.

---

## 📌 Features & Pipeline Architecture

The solution integrates advanced text processing, domain-specific merchant rules, leak-free target encoding, multi-scale text embeddings, and 7 model families into a unified Stratified 5-Fold Cross Validation pipeline:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Transaction Data                              │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Text Preprocessing & Regex Cleaning                  │
│       - NFKC Normalization & Lowercasing                               │
│       - US State Code Regex Stripping (AL, AK, CA, TX, NY, etc.)       │
│       - Merchant Root (`merchant_root`) Identification                 │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         Feature Extraction Engine                      │
│ 1. 80+ Domain Merchant Rules (Food, Transit, Subscriptions, etc.)      │
│ 2. Cents & Amount Patterns (cents==0, 99c, 95c, 50c, mod 5, mod 10)    │
│ 3. Payday & Date Cyclical Features (dow_sin, dow_cos, payday flags)    │
│ 4. OOF Bayesian Target Encoding (Stratified 5-Fold, smooth_weight=10)  │
│ 5. Word (1-3gram) & Char (3-5gram) Sublinear TF-IDF + TruncatedSVD LSA │
│ 6. SentenceTransformers (`all-MiniLM-L6-v2`) 384-dim Dense Embeddings │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Stratified 5-Fold Cross-Validation Suite                 │
│  [1] LightGBM Classifier      [5] Calibrated Linear Classifier         │
│  [2] XGBoost Classifier       [6] Scikit-Learn MLP Neural Network      │
│  [3] CatBoost Classifier      [7] PyTorch Deep & Wide Transformer NN   │
│  [4] RandomForest Classifier                                           │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                Scipy SLSQP Constrained Weight Optimization              │
│       Solves: argmax_w MacroF1(y, sum w_i * P_i) subject to sum w_i=1  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│              High-Precision Domain Post-Processing Rules               │
│                  submission_trust_cv_ensemble.csv                      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🛡️ Trust-CV Methodology

1. **Strict Out-of-Fold (OOF) Evaluation**:
   Model weights and ensemble performance are evaluated purely on unseen validation folds during 5-Fold Stratified CV to eliminate leaderboard overfitting.
2. **Leak-Free Bayesian Target Encoding**:
   Merchant target encoding is computed strictly out-of-fold with global prior smoothing ($w_{\text{smooth}} = 10.0$).
3. **Model Diversity**:
   Combines Tree-based GBDTs, Random Forests, Calibrated Linear Models, Multi-Layer Perceptrons, and PyTorch Deep & Wide Neural Networks.

---

## 📂 Repository Contents

| File / Directory | Description |
|---|---|
| 📄 [`trust_cv_master_ensemble.ipynb`](./trust_cv_master_ensemble.ipynb) | Dedicated Trust-CV Jupyter Notebook with 12 structured sections. |
| 📄 [`master_ensemble_pipeline.ipynb`](./master_ensemble_pipeline.ipynb) | Master Jupyter Notebook for full multi-model ensemble. |
| 📄 [`aurora_ensemble_pipeline.ipynb`](./aurora_ensemble_pipeline.ipynb) | Complete pipeline notebook. |
| 🐍 [`master_ensemble_pipeline.py`](./master_ensemble_pipeline.py) | Standalone Python training script. |
| 📋 [`requirements.txt`](./requirements.txt) | Python dependencies specification. |
| 🔒 [`.gitignore`](./.gitignore) | Exclusion rules for confidential datasets (`*.csv`, `*.zip`), virtual environments (`.venv/`), and logs. |

---

## 🚀 Quick Start Guide

### 1. Repository Setup

```bash
git clone https://github.com/udaken10/AuroraGate.git
cd AuroraGate
```

### 2. Environment & Dependency Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run Pipeline Script

```bash
python master_ensemble_pipeline.py
```

---

## 🔒 Confidentiality & Data Exclusion

The following sensitive / competition data files are strictly excluded via `.gitignore`:
- Confidential Datasets: `train.csv`, `test.csv`, `sample_submission.csv`
- Submissions & Output Logs: `submission*.csv`, `catboost_info/`
- Local Environments: `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`
