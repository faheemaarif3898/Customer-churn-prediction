# 📊 Telco Customer Churn Prediction  
**XGBoost + Probability Calibration**

---

## Overview
This project builds a **customer churn prediction system** using the Telco Customer Churn dataset.  
The focus is not only on predictive performance, but on **decision quality**, especially identifying customers at high risk of churning with stable probability estimates.

The solution emphasizes:
- Behavioral feature engineering  
- PR-AUC–optimized modeling  
- Probability calibration for reliable thresholds  
- A clean, leakage-safe sklearn pipeline  

---

## Dataset
**Source:** Kaggle – Telco Customer Churn  

Each record represents a customer with:
- subscribed services  
- billing and payment behavior  
- tenure and usage charges  
- churn label (Yes / No)  

The dataset is **imbalanced**, making accuracy an unsuitable primary metric.

---

## Problem Framing
Churn prediction is treated as a **ranking and decision problem**, not just binary classification.

Key challenges:
- Class imbalance  
- High cost of false negatives (missed churners)  
- Unstable probability outputs from tree-based models  

To address these:
- The model is optimized for **PR-AUC**
- Probabilities are **explicitly calibrated**
- Threshold selection is performed **after calibration**

---

## Feature Engineering
Rather than relying on raw categorical variables, the project introduces **behaviorally meaningful features**:

### Service Engagement
- `num_services`: total number of subscribed services  

### Billing Behavior
- `avg_monthly_charge`: stabilized charge-to-tenure ratio  
- `is_auto_payment`: automatic vs manual payment  

### Customer Lifecycle
- `tenure_bin`: lifecycle phases (0–6, 6–12, 12–24, 24+ months)  

### Context Flags
- `has_internet`  
- `PaperlessBilling`  
- `SeniorCitizen`  
- `Dependents`  

Redundant service-level columns are dropped after aggregation to reduce noise and split competition.

---

## Preprocessing & Pipeline
All preprocessing is handled through a **sklearn Pipeline** to ensure reproducibility and prevent data leakage.

Pipeline components:
- `ColumnTransformer`
  - Standard scaling for selected numeric features  
  - One-hot encoding for categorical features  
  - Pass-through for binary features  
- `XGBClassifier` as the core estimator  

The train/validation split is performed **before** encoding and scaling.

---

## Model Choice
**XGBoost** is used due to:
- Strong performance on tabular datasets  
- Robust handling of non-linear interactions  
- Compatibility with PR-AUC optimization  

Class imbalance is addressed using `scale_pos_weight`.

---

## Probability Calibration
Models optimized for PR-AUC often produce **poorly calibrated probabilities**.

To address this, **Platt scaling (sigmoid calibration)** is applied using `CalibratedClassifierCV`.

Benefits:
- Probabilities align with true churn rates  
- Thresholds become stable and transferable  
- Business decisions become defensible  

> Calibration is performed on validation data only.

---

## Evaluation Strategy
Instead of accuracy, evaluation focuses on:
- Recall for churners  
- Precision–recall trade-off  
- Confusion matrix at a calibrated threshold  

A threshold of **0.19** is selected to prioritize churn recall while maintaining reasonable precision.

---

## Results (Validation Set)
- High recall for churners  
- Stable probability distribution  
- No degenerate predictions (e.g., predicting all customers as churners)

The model is suitable for **retention-focused decision-making**.

---

## Repository Structure
├── churn_pipeline.ipynb   # Final notebook (clean and reproducible)
├── README.md             # Project documentation

---

## Key Takeaways
- Feature engineering often matters more than aggressive hyperparameter tuning  
- Calibration is essential for real-world decision systems  
- Pipelines prevent silent leakage and instability  
- Threshold selection should follow calibration, not precede it  

---

## Future Improvements
- Wrap feature engineering into a custom sklearn Transformer  
- Add SHAP-based explainability  
- Introduce cost-sensitive threshold optimization  

---

## Author
Faheem  
B.Tech Computer Science & Engineering


# Deployment Notes

Model trained using scikit-learn X.Y.Z and XGBoost A.B.C
Deployment environment strictly matches training versions
Calibration applied offline; inference uses fixed threshold (0.19)
Docker used to ensure environment reproducibility
Known pitfalls: sklearn and xgboost pickling incompatibilities
