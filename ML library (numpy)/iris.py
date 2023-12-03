from sklearn.datasets import load_iris
import numpy as np

from nn import NN
from nn.utilities import train_test_split, standardise, onehot, save_model
from nn.layers import Dense
from nn.activations import ReLU, SoftMax
from nn.cost import CCE
from nn.optimisers import GD
from nn.initialisers import RandNorm

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.3, seed=42)

X_train, mu, std = standardise(X_train)

X_test, _, _ = standardise(X_test, mu, std)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_ratio=0.2, seed=42)

y_train = onehot(y_train)

y_val = onehot(y_val)

y_test = onehot(y_test)

lr, epochs, batch_size = 0.1, 100, 2

in_dim = X_train.shape[1]
out_dim = y_train.shape[1]

model = [
      Dense(out_shape=6,
            seed=42,
            initialiser=RandNorm()),
      ReLU(),
      Dense(out_shape=out_dim,
            seed=42,
            initialiser=RandNorm()),
      SoftMax()
]

nn = NN(model)

nn.Input(in_dim)

nn.compile(cost_func=CCE(), optimiser=GD)

nn.summary()

nn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, X_val=X_val, y_val=y_val, lr_0=lr)

save_model(nn, "./models/iris_cpu.mdl")

