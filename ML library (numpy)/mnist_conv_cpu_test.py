from keras.datasets import mnist
from nn.utilities import load_model, onehot

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

nn = load_model("./models/mnist_conv_cpu.mdl")

nn.loss_plot()
nn.accuracy_plot()

nn.cm_plot(X_test, y_test, 256)
nn.accuracy(X_test, y_test, 256)
