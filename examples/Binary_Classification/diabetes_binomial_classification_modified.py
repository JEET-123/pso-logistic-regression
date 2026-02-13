#Logistic Regression & PSOLR Accuracy (Best Threshold wrt PSOLR)
#Logistic Regression AUC & PSOLR AUC (Best Threshold wrt PSOLR)
#Finding best threshold for both Logistic Regression & PSOLR 
#Classification Report for PSOLR (Best Threshold wrt PSOLR)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# -------------------------------------------------
# Load data
# -------------------------------------------------
df = pd.read_csv("diabetes_dataset.csv")
df.head(10)

df.isna().sum()
df.describe()

# -------------------------------------------------
# Features & target
# -------------------------------------------------
X = df.drop(columns=['Outcome'])
y = df['Outcome']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=11
)

# -------------------------------------------------
# Scaling (IMPORTANT for PSO-LR)
# -------------------------------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------
# 🔁 PSO-LR instead of Logistic Regression
# -------------------------------------------------
from psolr import PSOLogisticRegression

model = PSOLogisticRegression(
    pop_size=40,
    max_iter=200,
    inertia=0.7,
    c1=1.5,
    c2=1.5,
    C=1.0,
    penalty="l2",
    random_state=42
)

model.fit(X_train_scaled, y_train)

# -------------------------------------------------
# Evaluation
# -------------------------------------------------
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Get predicted probabilities
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Apply custom threshold
threshold = 0.54
y_pred = (y_prob >= threshold).astype(int)

print("Accuracy of PSO-LR model:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("==================================================================================")

# -------------------------------------------------
# Confusion matrix
# -------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (PSO-LR)")
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.show()

# -----------------------------------------------------------------------------------------

# Logistic Regression baseline
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='liblinear', max_iter=250)
lr.fit(X_train_scaled, y_train)

# Get predicted probabilities
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

# Apply custom threshold
threshold = 0.54
y_pred_lr = (y_prob_lr >= threshold).astype(int)

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, y_pred_lr))

print("PSO-LR Accuracy:",
      accuracy_score(y_test, y_pred))

print("==================================================================================")

#-----------------------------------------------------------------------------------------

print("LR AUC:",
      roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:,1]))

print("PSO-LR AUC:",
      roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:,1]))


print("==================================================================================")

#-----------------------------------------------------------------------------------------

probs = lr.predict_proba(X_test_scaled)[:, 1]

thresholds = np.linspace(0.2, 0.8, 61)
best_acc = 0
best_thresh = 0.5

for t in thresholds:
    preds = (probs >= t).astype(int)
    acc = accuracy_score(y_test, preds)
    if acc > best_acc:
        best_acc = acc
        best_thresh = t

print(f"Best threshold (LR): {best_thresh:.2f}")
print(f"Best accuracy (LR): {best_acc:.4f}")

print("==================================================================================")

#-----------------------------------------------------------------------------------------

probs = model.predict_proba(X_test_scaled)[:, 1]

thresholds = np.linspace(0.2, 0.8, 61)
best_acc = 0
best_thresh = 0.5

for t in thresholds:
    preds = (probs >= t).astype(int)
    acc = accuracy_score(y_test, preds)
    if acc > best_acc:
        best_acc = acc
        best_thresh = t

print(f"Best threshold (PSOLR): {best_thresh:.2f}")
print(f"Best accuracy (PSOLR): {best_acc:.4f}")

print("==================================================================================")