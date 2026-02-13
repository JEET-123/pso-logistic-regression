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
