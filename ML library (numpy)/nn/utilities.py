import pickle
import numpy as np

def train_test_split(X, y, test_ratio=0.2, seed=None):
    m = X.shape[0]
    if seed:
        np.random.seed(seed)
    train_ratio = 1 - test_ratio
    indicies = np.random.permutation(m)
    train_idx, test_idx = indicies[:int(train_ratio*m)], indicies[:int(test_ratio*m)]
    X_train, X_test = X[train_idx,:], X[test_idx, :]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test

def onehot(Y):
    y = np.unique(Y)
    m = Y.shape[0]
    cat = y.shape[0]

    Y_onehot = np.zeros((m, cat))
    Y_onehot[np.arange(m), Y] = 1
    return Y_onehot

def standardise(X, mu=None, std=None):
    if mu is None:
        mu = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
    Z = (X - mu) / std
    return Z, mu, std

def save_model(model, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)

def load_model(file_name):
    with open(file_name, 'rb') as file:
        nn = pickle.load(file)
    return nn
    

