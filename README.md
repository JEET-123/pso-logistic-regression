# PSO-LR: Metaheuristic-Optimized Logistic Regression (MOLR)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18639831.svg)](https://doi.org/10.5281/zenodo.18639831)
[![PyPI version](https://img.shields.io/pypi/v/psolr.svg)](https://pypi.org/project/psolr/)
[![Python](https://img.shields.io/pypi/pyversions/psolr.svg)](https://pypi.org/project/psolr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

**PSO-LR** is an open-source, sklearn-compatible implementation of **Metaheuristic-Optimized Logistic Regression (MOLR)**.

Unlike traditional logistic regression, which relies on gradient-based solvers  
(LBFGS, Newton-CG, SGD), PSO-LR estimates model parameters using  
**Particle Swarm Optimization (PSO)** — a population-based global optimization algorithm.

This approach preserves the classical interpretability of logistic regression  
while improving robustness, global convergence behavior, and stability in  
noisy or ill-conditioned datasets.

---

## Motivation

Traditional logistic regression can struggle when:

- Multicollinearity is present  
- The optimization landscape is poorly conditioned  
- Data contains strong noise  
- Strong regularization is required  

PSO-LR replaces gradient descent with a global, population-based search mechanism,  
making it suitable for:

- Marketing analytics  
- Choice modeling  
- Behavioral modeling  
- Econometrics  
- Healthcare analytics  
- High-noise real-world datasets  

---

## Key Features

- Binary Logistic Regression  
- Multinomial Logistic Regression  
- Particle Swarm Optimization (PSO)  
- Early stopping for efficiency  
- L1 / L2 regularization  
- Hard coefficient constraints  
- Fully sklearn-compatible API  
- Interpretable coefficients & odds ratios  
- Domain-independent design  

---

## Quick Start Example

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
y_prob = model.predict_proba(X)
```

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


This removes reliance on gradients and enables global exploration.

---

## Installation

```bash
pip install psolr
```

Upgrade:

```bash
pip install --upgrade psolr
```

---

## Example: Hyperparameter Optimization

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    "C": [0.1, 1, 10],
    "pop_size": [40, 60, 80],
    "max_iter": [200, 300]
}

search = RandomizedSearchCV(
    PSOLogisticRegression(random_state=42),
    param_distributions=param_dist,
    scoring="roc_auc",
    cv=5
)

search.fit(X, y)
```

---

## Use Cases

PSO-LR is particularly beneficial when:

- Interpretability is critical  
- Data is noisy or unstable  
- Traditional solvers fail to converge reliably  
- Hard constraints on coefficients are required  
- Global optimization is preferred  

---

## Citation

If you use PSO-LR in your research, please cite:

Dutta, K. (2026). *PSO-LR: Particle Swarm Optimization based Logistic Regression* (Version 0.1.1). Zenodo. https://doi.org/10.5281/zenodo.18639831

BibTeX:

```bibtex
@software{dutta2026psolr,
  author = {Dutta, Koustav},
  title = {PSO-LR: Particle Swarm Optimization based Logistic Regression},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18639831}
}
```

---

## License

MIT License.

---

## Author

**Koustav Dutta**  
PhD Researcher — Machine Learning, Artificial Intelligence & Neuralcomputing 

---

## Contributing

Contributions and improvements are welcome.  
Please open an issue or submit a pull request.
