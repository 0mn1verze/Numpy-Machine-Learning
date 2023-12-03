from sklearn.datasets import load_iris
import numpy as np

from nn.utilities import train_test_split, standardise, onehot, load_model

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.3, seed=42)

X_train, mu, std = standardise(X_train)

X_test, _, _ = standardise(X_test, mu, std)

nn = load_model("./models/iris_cpu.mdl")

nn.loss_plot()

nn.accuracy_plot()

nn.cm_plot(X_test, y_test)

nn.accuracy(X_test, y_test)