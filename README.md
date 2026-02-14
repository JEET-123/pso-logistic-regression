# PSO-LR: Metaheuristic-Optimized Logistic Regression (MOLR)

**PSO-LR** is an open-source, sklearn-compatible implementation of  
**Metaheuristic-Optimized Logistic Regression (MOLR)**, where model parameters
are estimated using **Particle Swarm Optimization (PSO)** instead of
gradient-based solvers.

This framework preserves the classical interpretability of logistic regression
while improving robustness, global convergence behavior, and stability in
noisy or ill-conditioned datasets.

---

## Motivation

Traditional logistic regression relies on gradient-based optimization
(LBFGS, Newton-CG, SGD). These methods can struggle when data exhibits:

- Multicollinearity
- Noisy or sparse signals
- Poor conditioning
- Strong regularization requirements

PSO-LR replaces gradient descent with a **population-based global optimizer**,
making it suitable for:

- Marketing analytics
- Choice modeling
- Behavioral modeling
- Econometrics
- High-noise real-world datasets

---

## PSOLR Quick Start Example

```python
from psolr import PSOLogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=10)

model = PSOLogisticRegression(
    pop_size=50,
    max_iter=200,
    random_state=42
)

model.fit(X, y)
y_pred = model.predict(X)

---

## Key Features

- Binary & Multinomial Logistic Regression
- Particle Swarm Optimization (PSO)
- Early stopping for efficiency
- L1 / L2 regularization
- Hard coefficient constraints
- Fully sklearn-compatible API
- Interpretable coefficients & odds ratios
- Domain-independent design

---

## Mathematical Formulation

The logistic model remains unchanged:

\[
P(y=1|x) = \sigma(x^\top \beta)
\]

The regularized negative log-likelihood is optimized using PSO:

\[
\min_{\beta} \; -\mathcal{L}(\beta) + \lambda \Omega(\beta)
\]

PSO searches the parameter space globally without requiring gradients,
ensuring robust convergence.

---

## Installation

```bash
pip install psolr

## Citation

If you use PSO-LR in your research, please cite:

Dutta, K. (2026). PSO-LR: Particle Swarm Optimization based Logistic Regression. GitHub. https://github.com/JEET-123/pso-logistic-regression

DOI: 10.5281/zenodo.xxxxx

