from keras.datasets import mnist
import cupy as cp

from nn.utilities import load_model, onehot

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, y_train, X_test, y_test = cp.array(X_train).astype(cp.float32), cp.array(y_train), cp.array(X_test).astype(cp.float32), cp.array(y_test)

X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

nn = load_model("./models/mnist_conv_gpu.mdl")

nn.loss_plot()
nn.accuracy_plot()

nn.cm_plot(X_test, y_test, 256)
nn.accuracy(X_test, y_test, 256)
