Telco Customer Churn Prediction
XGBoost + Probability Calibration
Overview
This project builds a customer churn prediction system using the Telco Customer Churn dataset.
The goal is not just to maximize accuracy, but to reliably identify customers at risk of churning, with stable and interpretable probability outputs.
The final solution uses:
Behavioral feature engineering
XGBoost optimized for PR-AUC
Probability calibration for stable decision thresholds
A clean, leakage-safe sklearn pipeline
Dataset
Source: Kaggle – Telco Customer Churn
Each row represents a customer with:
service subscriptions
billing and payment behavior
tenure and charges
churn label (Yes / No)
The dataset is imbalanced, making accuracy a poor primary metric.
Problem Framing
Churn prediction is a ranking and decision problem, not a pure classification task.
Key challenges:
Class imbalance
High cost of false negatives (missed churners)
Unstable probability outputs from tree-based models
To address this:
The model is optimized for PR-AUC
Probabilities are explicitly calibrated
Threshold selection is done after calibration
Feature Engineering
Instead of relying on raw categorical variables, the project introduces behaviorally meaningful features:
Service engagement
num_services: total number of subscribed services
Billing behavior
avg_monthly_charge: stabilized charge-to-tenure ratio
is_auto_payment: automatic vs manual payment
Customer lifecycle
tenure_bin: lifecycle phases (0–6, 6–12, 12–24, 24+ months)
Context flags
has_internet, PaperlessBilling, SeniorCitizen, Dependents
Redundant service-level columns are dropped after aggregation to reduce noise and split competition.
Preprocessing & Pipeline
All preprocessing steps are handled through a sklearn Pipeline to ensure reproducibility and prevent data leakage.
Pipeline components:
ColumnTransformer
Standard scaling for selected numeric features
One-hot encoding for categorical features
Pass-through for binary features
XGBClassifier as the core model
Train/validation split is performed before encoding and scaling.
Model Choice
XGBoost is used due to:
Strong performance on tabular data
Robust handling of non-linear interactions
Compatibility with PR-AUC optimization
Class imbalance is handled via scale_pos_weight.
Probability Calibration
Models optimized for PR-AUC often produce poorly calibrated probabilities.
To address this, Platt scaling (sigmoid calibration) is applied using CalibratedClassifierCV.
Benefits:
Probabilities align with true churn rates
Thresholds become stable and transferable
Business decisions become defensible
Note: Calibration is performed on validation data only.
Evaluation Strategy
Instead of accuracy, the focus is on:
Recall for churners
Precision–recall trade-off
Confusion matrix at a calibrated threshold
A threshold of 0.19 is selected to prioritize churn recall while maintaining reasonable precision.
Results (Validation Set)
High recall for churners
Stable probability distribution
No degenerate predictions (no “predict everyone as churn” collapse)
This makes the model suitable for retention-focused decision-making.
Repository Structure
Copy code

├── churn_pipeline.ipynb   # Final notebook (clean, reproducible)
├── README.md              # Project documentation
Key Takeaways
Feature engineering matters more than aggressive hyperparameter tuning
Calibration is essential for real-world decision systems
Pipelines prevent silent leakage and instability
Threshold selection should follow calibration, not precede it
Future Improvements
Wrap feature engineering into a custom sklearn Transformer
Add SHAP-based explainability
Introduce cost-sensitive threshold optimization
Author
Faheem
B.Tech Computer Science & Engineering
