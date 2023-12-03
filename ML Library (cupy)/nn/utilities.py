import pickle
import cupy as cp

def train_test_split(X, y, test_ratio=0.2, seed=None):
    m = X.shape[0]
    if seed:
        cp.random.seed(seed)
    train_ratio = 1 - test_ratio
    indicies = cp.random.permutation(m)
    train_idx, test_idx = indicies[:int(train_ratio*m)], indicies[:int(test_ratio*m)]
    X_train, X_test = X[train_idx,:], X[test_idx, :]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test

def onehot(Y):
    y = cp.unique(Y)
    m = Y.shape[0]
    cat = y.shape[0]

    Y_onehot = cp.zeros((m, cat))
    Y_onehot[cp.arange(m), Y] = 1
    return Y_onehot

def standardise(X, mu=None, std=None):
    if mu is None:
        mu = cp.mean(X, axis=0)
    if std is None:
        std = cp.std(X, axis=0)
    Z = (X - mu) / std
    return Z, mu, std

def save_model(model, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)

def load_model(file_name):
    with open(file_name, 'rb') as file:
        nn = pickle.load(file)
    return nn
    

