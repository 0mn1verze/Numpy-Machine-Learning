from sklearn.datasets import make_moons

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import cupy as cp

from nn.utilities import train_test_split, onehot, load_model

sns.set_style("whitegrid")

# the function making up the graph of a dataset
def make_plot(X, y, plot_name, file_name=None, XX=None, YY=None, preds=None, dark=False):
    if (dark):
        plt.style.use('dark_background')
    else:
        sns.set_style("whitegrid")
    plt.figure(figsize=(16,12))
    axes = plt.gca()
    axes.set(xlabel="$X_1$", ylabel="$X_2$")
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    if(XX is not None and YY is not None and preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha = 1, cmap=cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], cmap="Greys", vmin=0, vmax=.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors='black')
    if(file_name):
        plt.savefig(file_name)
        plt.close()

# boundary of the graph
GRID_X_START = -1.5
GRID_X_END = 2.5
GRID_Y_START = -1.0
GRID_Y_END = 2
# output directory (the folder must be created on the drive)

grid = np.mgrid[GRID_X_START:GRID_X_END:100j,GRID_X_START:GRID_Y_END:100j]
grid_2d = grid.reshape(2, -1).T
XX, YY = grid

def numpy_plot(nn, X_test, y_test):
    file_path = 'model.png'
    prediction_probs = nn.predict(cp.array(grid_2d), 100).get()
    prediction_probs = prediction_probs.reshape(prediction_probs.shape[0], 1)
    make_plot(X_test, y_test, "Machine Learning", file_name=file_path, XX=XX, YY=YY, preds=prediction_probs, dark=True)

nn = load_model("./models/make_moons_gpu.mdl")

# number of samples in the data set
N_SAMPLES = 1000
# ratio between training and test sets
TEST_RATIO = 0.2

VAL_RATIO = 0.2

X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)

X, y = cp.array(X), cp.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=TEST_RATIO, seed=42)

X_test, y_test = X_test.get(), y_test.get()

nn.accuracy_plot()
nn.loss_plot()

numpy_plot(nn, X_test, y_test)
