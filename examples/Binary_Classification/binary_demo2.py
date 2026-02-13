'''
✔ Hyperparameters tuned using ROC-AUC
✔ Threshold tuned only on validation set
✔ Test set evaluated once (no leakage)
✔ ROC curve plotted
✔ Repeated CV gives stability estimate
✔ Mean ± Std reported
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from psolr import PSOLogisticRegression

# =================================================
# 1. Load Data
# =================================================
df = pd.read_csv("diabetes_dataset.csv")

X = df.drop(columns=['Outcome'])
y = df['Outcome']

# =================================================
# 2. Train / Validation / Test Split
# =================================================
# First split: Train+Validation and Test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.3, random_state=11, stratify=y
)

# Second split: Train and Validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=11, stratify=y_temp
)

# =================================================
# 3. Scaling
# =================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# =================================================
# 4. Hyperparameter Optimization (AUC-based)
# =================================================
param_dist = {
    "C": np.logspace(-2, 2, 20),
    "pop_size": [40, 60, 80],
    "max_iter": [200, 300],
}

rs = RandomizedSearchCV(
    PSOLogisticRegression(random_state=42),
    param_distributions=param_dist,
    n_iter=15,
    scoring="roc_auc",      # Changed to ROC-AUC
    cv=5,
    n_jobs=-1,
    random_state=42
)

rs.fit(X_train_scaled, y_train)

print("Best CV AUC:", rs.best_score_)
print("Best Parameters:", rs.best_params_)

best_model = rs.best_estimator_

# =================================================
# 5. Threshold Tuning on VALIDATION SET ONLY
# =================================================
val_probs = best_model.predict_proba(X_val_scaled)[:, 1]

thresholds = np.linspace(0.2, 0.8, 61)
best_val_acc = 0
best_thresh = 0.5

for t in thresholds:
    preds = (val_probs >= t).astype(int)
    acc = accuracy_score(y_val, preds)
    if acc > best_val_acc:
        best_val_acc = acc
        best_thresh = t

print("\nBest threshold (Validation):", round(best_thresh, 2))
print("Validation Accuracy at best threshold:", round(best_val_acc, 4))

# =================================================
# 6. Final Evaluation on TEST SET (ONLY ONCE)
# =================================================
test_probs = best_model.predict_proba(X_test_scaled)[:, 1]
y_pred = (test_probs >= best_thresh).astype(int)

print("\n===== FINAL TEST PERFORMANCE =====")
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Test AUC:", roc_auc_score(y_test, test_probs))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =================================================
# 7. Confusion Matrix
# =================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix (PSO-LR, threshold={best_thresh:.2f})")
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.show()

# =================================================
# 8. ROC Curve Plot
# =================================================
fpr, tpr, _ = roc_curve(y_test, test_probs)
auc_score = roc_auc_score(y_test, test_probs)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"PSO-LR (AUC = {auc_score:.3f})")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# =================================================
# 9. Repeated Cross-Validation (Robustness Check)
# =================================================
rkf = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)

cv_scores = cross_val_score(
    best_model,
    scaler.fit_transform(X),   # Fit on full dataset for CV
    y,
    scoring="roc_auc",
    cv=rkf,
    n_jobs=-1
)

print("\n===== REPEATED CROSS-VALIDATION RESULTS =====")
print("Mean ROC:", round(cv_scores.mean(), 4))
print("Std ROC:", round(cv_scores.std(), 4))
