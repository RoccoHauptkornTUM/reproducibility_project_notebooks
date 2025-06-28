# Reproducibility Platforms – Evaluation Project

This repository contains three realistic programs designed to evaluate the reproducibility of workflows on the platforms **Binder** and **CodeOcean**

---

## Contents

### 1. Business Analytics
**File:** `business_analytics.py`  
Performs customer and product analysis on sales data:
- Cleans and aggregates transactional data
- Performs clustering and customer segmentation (via KMeans)
- Outputs visualizations and `.csv` files
- Generates a `.txt` report with key metrics

### 2. Machine Learning
**File:** `ml_pipeline.py`  
A classification model predicting student placement:
- Loads dataset from Kaggle via `kagglehub`
- Full ML pipeline: preprocessing, training, evaluation
- Models: Logistic Regression, Random Forest
- Saves trained models and plots
- Text summary of results

### 3. Scientific Computing
**File:** `scientific_pi_simulation.py`  
Monte Carlo simulation to estimate $\pi$:
- Evaluates convergence with increasing sample sizes
- Statistical error analysis from repeated runs
- Saves plots and a summary `.txt` report

---