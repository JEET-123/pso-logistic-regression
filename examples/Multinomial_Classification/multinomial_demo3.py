import urllib.request
import pandas as pd
import requests
import io
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from numpy import mean, std

from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from psolr import PSOLogisticRegression


# ==========================================================
# 1️⃣ Load Dataset
# ==========================================================
link = "https://stats.idre.ucla.edu/stat/data/hsb2.csv"

s = requests.get(link).content
hsb2 = pd.read_csv(io.StringIO(s.decode('utf-8')))

# Convert categorical variables
hsb2["race"] = hsb2["race"].astype('category')
hsb2["female"] = hsb2["female"].astype('category')
hsb2["ses"] = hsb2["ses"].astype('category')

# Create dummies
race = pd.get_dummies(hsb2['race'], drop_first=True, prefix='race')
ses = pd.get_dummies(hsb2['ses'], drop_first=True, prefix='ses')

hsb3 = hsb2.drop(['race','ses'], axis=1)
hsb3 = pd.concat([hsb3, race, ses], axis=1)

# Features and target
y = hsb3['prog']
X = hsb3.drop(['prog','id'], axis=1)

# ==========================================================
# 2️⃣ Scaling (IMPORTANT)
# ==========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Binarize target for ROC-AUC (multiclass)
classes = np.unique(y)
y_bin = label_binarize(y, classes=classes)


# ==========================================================
# 3️⃣ Multinomial Logistic Regression (Baseline)
# ==========================================================
print("\n================ MULTINOMIAL LOGISTIC REGRESSION ================\n")

lr_model = LogisticRegression(
    solver='lbfgs',
    penalty='l2',
    C=1.0,
    max_iter=100000
)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Accuracy
lr_acc = cross_val_score(
    lr_model, X_scaled, y,
    scoring='accuracy',
    cv=cv, n_jobs=-1
)

# Multiclass ROC-AUC (OvR)
lr_auc = cross_val_score(
    lr_model, X_scaled, y,
    scoring='roc_auc_ovr',
    cv=cv, n_jobs=-1
)

print('LR Mean Accuracy: %.3f (%.3f)' % (mean(lr_acc), std(lr_acc)))
print('LR Mean ROC-AUC (OvR): %.3f (%.3f)' % (mean(lr_auc), std(lr_auc)))

# Fit full model
lr_model.fit(X_scaled, y)


# ==========================================================
# 4️⃣ Multinomial PSO-Logistic Regression
# ==========================================================
print("\n================ PSO MULTINOMIAL LOGISTIC REGRESSION ================\n")

pso_model = PSOLogisticRegression(
    multi_class="multinomial",
    pop_size=60,
    max_iter=300,
    C=1.0,
    random_state=42
)

# Accuracy
pso_acc = cross_val_score(
    pso_model, X_scaled, y,
    scoring='accuracy',
    cv=cv, n_jobs=-1
)

# Multiclass ROC-AUC
pso_auc = cross_val_score(
    pso_model, X_scaled, y,
    scoring='roc_auc_ovr',
    cv=cv, n_jobs=-1
)

print('PSO-LR Mean Accuracy: %.3f (%.3f)' % (mean(pso_acc), std(pso_acc)))
print('PSO-LR Mean ROC-AUC (OvR): %.3f (%.3f)' % (mean(pso_auc), std(pso_auc)))

# Fit full model
pso_model.fit(X_scaled, y)


# ==========================================================
# 5️⃣ Compare Predictions (Example Row)
# ==========================================================
row = X_scaled[0:1, :]

lr_probs = lr_model.predict_proba(row)
pso_probs = pso_model.predict_proba(row)

print("\nPredicted Probabilities (LR):", lr_probs[0])
print("Predicted Class (LR):", lr_model.predict(row)[0])

print("\nPredicted Probabilities (PSO-LR):", pso_probs[0])
print("Predicted Class (PSO-LR):", pso_model.predict(row)[0])


# ==========================================================
# 6️⃣ Visual Comparison (Accuracy Distribution)
# ==========================================================
plt.figure(figsize=(6,5))
sns.kdeplot(lr_acc, label="LR Accuracy", fill=True)
sns.kdeplot(pso_acc, label="PSO-LR Accuracy", fill=True)
plt.title("Cross-Validation Accuracy Distribution")
plt.legend()
plt.show()
