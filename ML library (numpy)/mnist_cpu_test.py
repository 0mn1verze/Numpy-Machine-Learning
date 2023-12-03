from keras.datasets import mnist
from nn.utilities import load_model

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]**2)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]**2)
X_train = X_train/255
X_test = X_test/255

nn = load_model("./models/mnist_cpu.mdl")

nn.accuracy(X_test, y_test, 256)


