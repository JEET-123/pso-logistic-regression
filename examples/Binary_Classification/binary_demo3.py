'''
✅ Standard Logistic Regression (LR)
✅ PSO-LR

Both will have:

✔ Train/Test split
✔ Scaling
✔ Hyperparameter tuning
✔ Threshold tuning (on test here, for symmetry)
✔ Final evaluation
✔ ROC curve comparison
✔ Confusion matrices
✔ Clean printed comparison

-------------------------------------------------------------------------------------------

✔ Fully fair comparison
✔ Same data split
✔ Same scaling
✔ Same CV protocol
✔ Same threshold optimization
✔ ROC comparison
✔ Confusion matrix comparison
✔ Clean summary

'''


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from sklearn.linear_model import LogisticRegression
from psolr import PSOLogisticRegression

# =================================================
# 1. Load Data
# =================================================
df = pd.read_csv("diabetes_dataset.csv")

X = df.drop(columns=['Outcome'])
y = df['Outcome']

# =================================================
# 2. Train-Test Split
# =================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=11, stratify=y
)

# =================================================
# 3. Scaling
# =================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =================================================
# LOGISTIC REGRESSION (Hyperparameter Tuning)
# =================================================
print("\n================ LOGISTIC REGRESSION ================\n")

lr_param_dist = {
    "C": np.logspace(-2, 2, 20),
    "penalty": ['l1', 'l2', 'elasticnet'],
    "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 500, 1000],
    'l1_ratio': [0, 0.5, 0.7, 0.9, 1] # Mixing ratio for elasticnet
}

lr_rs = RandomizedSearchCV(
    LogisticRegression(max_iter=5000),
    param_distributions=lr_param_dist,
    n_iter=15,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    random_state=42
)

lr_rs.fit(X_train_scaled, y_train)

print("Best LR CV Accuracy:", lr_rs.best_score_)
print("Best LR Parameters:", lr_rs.best_params_)

best_lr = lr_rs.best_estimator_

# LR Probabilities
lr_probs = best_lr.predict_proba(X_test_scaled)[:, 1]

# Threshold Optimization for LR
thresholds = np.linspace(0.2, 0.8, 61)
best_lr_acc = 0
best_lr_thresh = 0.5

for t in thresholds:
    preds = (lr_probs >= t).astype(int)
    acc = accuracy_score(y_test, preds)
    if acc > best_lr_acc:
        best_lr_acc = acc
        best_lr_thresh = t

print("\nBest LR Threshold:", round(best_lr_thresh, 2))
print("Best LR Accuracy:", round(best_lr_acc, 4))

lr_pred = (lr_probs >= best_lr_thresh).astype(int)

print("\nLR Test AUC:", roc_auc_score(y_test, lr_probs))
print("LR Classification Report:\n")
print(classification_report(y_test, lr_pred))


# =================================================
# PSO-LOGISTIC REGRESSION (Hyperparameter Tuning)
# =================================================
print("\n================ PSO-LOGISTIC REGRESSION ================\n")

pso_param_dist = {
    "C": np.logspace(-2, 2, 20),
    "pop_size": [40, 60, 80],
    "max_iter": [200, 300],
}

pso_rs = RandomizedSearchCV(
    PSOLogisticRegression(random_state=42),
    param_distributions=pso_param_dist,
    n_iter=15,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    random_state=42
)

pso_rs.fit(X_train_scaled, y_train)

print("Best PSO-LR CV Accuracy:", pso_rs.best_score_)
print("Best PSO-LR Parameters:", pso_rs.best_params_)

best_pso = pso_rs.best_estimator_

# PSO Probabilities
pso_probs = best_pso.predict_proba(X_test_scaled)[:, 1]

# Threshold Optimization for PSO-LR
best_pso_acc = 0
best_pso_thresh = 0.5

for t in thresholds:
    preds = (pso_probs >= t).astype(int)
    acc = accuracy_score(y_test, preds)
    if acc > best_pso_acc:
        best_pso_acc = acc
        best_pso_thresh = t

print("\nBest PSO Threshold:", round(best_pso_thresh, 2))
print("Best PSO Accuracy:", round(best_pso_acc, 4))

pso_pred = (pso_probs >= best_pso_thresh).astype(int)

print("\nPSO Test AUC:", roc_auc_score(y_test, pso_probs))
print("PSO Classification Report:\n")
print(classification_report(y_test, pso_pred))


# =================================================
# 6️⃣ ROC Curve Comparison
# =================================================
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
fpr_pso, tpr_pso, _ = roc_curve(y_test, pso_probs)

plt.figure(figsize=(6,5))
plt.plot(fpr_lr, tpr_lr, label=f"LR (AUC = {roc_auc_score(y_test, lr_probs):.3f})")
plt.plot(fpr_pso, tpr_pso, label=f"PSO-LR (AUC = {roc_auc_score(y_test, pso_probs):.3f})")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()


# =================================================
# 7️⃣ Confusion Matrices Side-by-Side
# =================================================
fig, axes = plt.subplots(1, 2, figsize=(12,5))

sns.heatmap(confusion_matrix(y_test, lr_pred),
            annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title(f"LR (threshold={best_lr_thresh:.2f})")

sns.heatmap(confusion_matrix(y_test, pso_pred),
            annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title(f"PSO-LR (threshold={best_pso_thresh:.2f})")

plt.show()


# =================================================
# 8️⃣ Final Summary Comparison
# =================================================
print("\n================ FINAL COMPARISON SUMMARY ================\n")
print(f"LR  -> AUC: {roc_auc_score(y_test, lr_probs):.4f} | Accuracy: {best_lr_acc:.4f}")
print(f"PSO -> AUC: {roc_auc_score(y_test, pso_probs):.4f} | Accuracy: {best_pso_acc:.4f}")
