import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_blobs, load_iris
from modular_Output import InnerModel

X, y = make_moons(n_samples=200, noise=2.0, shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

model = InnerModel(x=X_train, y=y_train, learning_rate=0.001, n_inner_neurons=20)
model.fit(100)
