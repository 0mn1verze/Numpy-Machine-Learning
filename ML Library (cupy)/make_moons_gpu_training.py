from sklearn.datasets import make_moons
import numpy as np
import cupy as cp

from nn import NN
from nn.utilities import train_test_split, onehot, save_model
from nn.layers import Dense, Dropout, BatchNorm
from nn.activations import ReLU, SoftMax, Sigmoid, Tanh
from nn.cost import CCE
from nn.optimisers import GD
from nn.initialisers import RandNorm

# number of samples in the data set
N_SAMPLES = 1000
# ratio between training and test sets
TEST_RATIO = 0.2

VAL_RATIO = 0.2

X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)

X, y = cp.array(X), cp.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=TEST_RATIO, seed=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_ratio=VAL_RATIO, seed=42)

y_train = onehot(y_train)

y_val = onehot(y_val)

lr, epochs, batch_size = 0.05, 10, 32
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
      SoftMax(),
]

nn = NN(model)

nn.Input(in_dim)

nn.compile(cost_func=CCE(), optimiser=GD)

nn.summary()

nn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, lr_0=lr, X_val=X_val, y_val=y_val)

save_model(nn, "./models/make_moons_gpu.mdl")

