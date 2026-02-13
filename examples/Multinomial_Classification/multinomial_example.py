from psolr import PSOLogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    n_features=8,
    n_classes=3,
    n_informative=6
)

model = PSOLogisticRegression(multi_class="multinomial")
model.fit(X, y)
