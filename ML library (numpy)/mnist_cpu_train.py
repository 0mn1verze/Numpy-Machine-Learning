from keras.datasets import mnist

from nn import NN
from nn.utilities import train_test_split, onehot, save_model, load_model
from nn.layers import Dense
from nn.activations import ReLU, SoftMax, Tanh
from nn.cost import CCE
from nn.optimisers import Adam
from nn.initialisers import Xavier, He, RandNorm

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]**2)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]**2)
X_train = X_train/255
X_test = X_test/255

# train validation split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_ratio=0.2, seed=42)

y_train = onehot(y_train)

y_val = onehot(y_val)

lr, epochs, batch_size = 0.05, 10, 200
in_dim = X_train.shape[1]
out_dim = y_train.shape[1]

model = [
    Dense(240, seed=42),
    ReLU(),
    Dense(200, seed=42),
    ReLU(),
    Dense(out_dim, seed=42),
    SoftMax()
]

nn = NN(model)

nn.Input(in_dim)

nn.compile(cost_func=CCE(), optimiser=Adam)

nn.summary()

nn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, lr_0=lr, X_val=X_val, y_val=y_val)

save_model(nn, "./models/mnist_cpu.mdl")