from keras.datasets import mnist

import cupy as cp

from nn import NN
from nn.utilities import train_test_split, onehot, save_model, load_model
from nn.layers import Dense, Conv2D, Flatten, Pool2D, Dropout, BatchNorm
from nn.activations import ReLU, SoftMax, Tanh
from nn.cost import CCE
from nn.optimisers import Adam
from nn.decay import StepDecay

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, y_train, X_test, y_test = cp.array(X_train).astype(cp.float32), cp.array(y_train), cp.array(X_test).astype(cp.float32), cp.array(y_test)

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

SAMPLES = 10000

X_train = X_train[:SAMPLES, :]/255
X_test = X_test[:SAMPLES, :]/255

y_train = y_train[:SAMPLES]
y_test = y_test[:SAMPLES]

# train validation split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_ratio=0.2, seed=42)

y_train = onehot(y_train)

y_val = onehot(y_val)

lr, epochs, batch_size = 0.1, 30, 200
in_shape = X_train.shape[1:]
out_shape = y_train.shape[1]

model = [
    Conv2D(32, (5, 5)),
    ReLU(),
    Pool2D(),
    Flatten(),
    Dropout(0.75),
    Dense(100),
    ReLU(),
    Dense(out_shape),
    SoftMax()
]

nn = NN(model)

nn.Input(in_shape)

nn.compile(cost_func=CCE(), optimiser=Adam)

nn.summary()

nn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, X_val=X_val, y_val=y_val, lr_decay=StepDecay(), lr_0=lr, F=0.75, D=5000)

save_model(nn, "./models/mnist_conv_gpu.mdl")