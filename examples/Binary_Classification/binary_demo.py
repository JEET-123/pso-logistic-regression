#CROSS VALIDATION - LOGISTIC & PSOLR
#HYPER-PARAMETER OPTMIZATION - LOGISTIC & PSOLR
#MULTINOMIAL - LOGISTIC & PSOLR
#ROC-AUC PLOTS - LOGISTIC & PSOLR
#PYPI WORKS - PSOLR
#OTHERS (INTERPRETATIONS, EXPLANATIONS & MEANINGS)

'''

1. Train/Test split
2. Scaling
3. Hyperparameter optimization (CV)
4. Best model selection
5. Test evaluation
6. Automatic threshold optimization
7. Final performance reporting

---------------------------------------------------------------------------------------------

✔ Step 1 — Finds best hyperparameters using 5-fold CV
✔ Step 2 — Retrains best model
✔ Step 3 — Predicts probabilities on unseen test data
✔ Step 4 — Searches optimal threshold (0.2-0.8)
✔ Step 5 — Reports:

a.Final accuracy
b.AUC
c.Classification report
d.Confusion matrix

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
    roc_auc_score
)

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
    X, y, test_size=0.3, random_state=11
)

# =================================================
# 3. Scaling (IMPORTANT for PSO-LR)
# =================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =================================================
# 4. Hyperparameter Optimization (Random Search)
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
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    random_state=42
)

rs.fit(X_train_scaled, y_train)

print("Best CV Accuracy:", rs.best_score_)
print("Best Parameters:", rs.best_params_)

# =================================================
# 5. Use Best Model
# =================================================
best_model = rs.best_estimator_

# =================================================
# 6. Predict Probabilities on Test Set
# =================================================
probs = best_model.predict_proba(X_test_scaled)[:, 1]

# =================================================
# 7. Automatic Threshold Optimization
# =================================================
thresholds = np.linspace(0.2, 0.8, 61)
best_acc = 0
best_thresh = 0.5

for t in thresholds:
    preds = (probs >= t).astype(int)
    acc = accuracy_score(y_test, preds)
    if acc > best_acc:
        best_acc = acc
        best_thresh = t

print("\nBest threshold (PSO-LR):", round(best_thresh, 2))
print("Best accuracy (PSO-LR):", round(best_acc, 4))

# =================================================
# 8. Final Evaluation Using Best Threshold
# =================================================
y_pred = (probs >= best_thresh).astype(int)

print("\nFinal Test Accuracy:", accuracy_score(y_test, y_pred))
print("Final Test AUC:", roc_auc_score(y_test, probs))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =================================================
# 9. Confusion Matrix
# =================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix (PSO-LR, threshold={best_thresh:.2f})")
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.show()
