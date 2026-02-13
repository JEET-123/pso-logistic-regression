"""
Metaheuristic-Optimized Logistic Regression (MOLR / PSO-LR)
=========================================================

Author: Koustav Dutta (PhD Research)
License: MIT

Description:
------------
A fully sklearn-compatible Logistic Regression model where
parameter estimation is performed using Particle Swarm Optimization (PSO)
instead of gradient-based solvers.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer


class PSOLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Metaheuristic-Optimized Logistic Regression (MOLR / PSO-LR)
    """

    def __init__(
        self,
        penalty="l2",
        C=1.0,
        pop_size=40,
        max_iter=200,
        inertia=0.7,
        c1=1.5,
        c2=1.5,
        tol=1e-4,
        patience=10,
        bounds=None,
        random_state=None,
        multi_class="auto"
    ):
        self.penalty = penalty
        self.C = C
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.tol = tol
        self.patience = patience
        self.bounds = bounds
        self.random_state = random_state
        self.multi_class = multi_class

    # ------------------------------------------------------------------
    # Utility Functions
    # ------------------------------------------------------------------

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def _apply_bounds(self, particles):
        if self.bounds is None:
            return particles
        low, high = self.bounds
        return np.clip(particles, low, high)

    # ------------------------------------------------------------------
    # Loss Functions
    # ------------------------------------------------------------------

    def _binary_loss(self, beta, X, y):
        logits = X @ beta
        p = self._sigmoid(logits)
        eps = 1e-9

        loss = -np.mean(
            y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)
        )

        if self.penalty == "l2":
            loss += 0.5 * np.sum(beta ** 2) / self.C
        elif self.penalty == "l1":
            loss += np.sum(np.abs(beta)) / self.C

        return loss

    def _multinomial_loss(self, theta, X, Y):
        B = theta.reshape(self.n_classes_, self.n_features_)
        logits = X @ B.T
        P = self._softmax(logits)
        eps = 1e-9

        loss = -np.mean(np.sum(Y * np.log(P + eps), axis=1))
        loss += 0.5 * np.sum(B ** 2) / self.C

        return loss

    # ------------------------------------------------------------------
    # PSO Optimizer with Early Stopping
    # ------------------------------------------------------------------

    def _pso_optimize(self, loss_fn, dim):
        rng = np.random.default_rng(self.random_state)

        particles = rng.normal(0, 0.1, size=(self.pop_size, dim))
        velocities = rng.normal(0, 0.01, size=(self.pop_size, dim))

        particles = self._apply_bounds(particles)

        personal_best = particles.copy()
        personal_scores = np.array([loss_fn(p) for p in particles])

        global_best = personal_best[np.argmin(personal_scores)]
        global_score = np.min(personal_scores)

        no_improve = 0

        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                r1, r2 = rng.random(), rng.random()

                velocities[i] = (
                    self.inertia * velocities[i]
                    + self.c1 * r1 * (personal_best[i] - particles[i])
                    + self.c2 * r2 * (global_best - particles[i])
                )

                particles[i] += velocities[i]
                particles[i] = self._apply_bounds(particles[i])

                score = loss_fn(particles[i])

                if score < personal_scores[i]:
                    personal_best[i] = particles[i]
                    personal_scores[i] = score

            new_global_score = np.min(personal_scores)

            if abs(global_score - new_global_score) < self.tol:
                no_improve += 1
                if no_improve >= self.patience:
                    break
            else:
                no_improve = 0

            global_best = personal_best[np.argmin(personal_scores)]
            global_score = new_global_score

        return global_best

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.n_features_ = X.shape[1]

        if self.multi_class == "auto":
            self.multi_class_ = "multinomial" if len(np.unique(y)) > 2 else "binary"
        else:
            self.multi_class_ = self.multi_class

        if self.multi_class_ == "binary":
            self.classes_ = np.array([0, 1])
            loss_fn = lambda b: self._binary_loss(b, X, y)
            self.coef_ = self._pso_optimize(loss_fn, self.n_features_)

        else:
            lb = LabelBinarizer()
            Y = lb.fit_transform(y)
            self.classes_ = lb.classes_
            self.n_classes_ = Y.shape[1]

            dim = self.n_classes_ * self.n_features_
            loss_fn = lambda t: self._multinomial_loss(t, X, Y)

            theta = self._pso_optimize(loss_fn, dim)
            self.coef_ = theta.reshape(self.n_classes_, self.n_features_)

        return self

    def predict_proba(self, X):
        X = np.asarray(X)

        if self.multi_class_ == "binary":
            p = self._sigmoid(X @ self.coef_)
            return np.vstack([1 - p, p]).T

        logits = X @ self.coef_.T
        return self._softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
