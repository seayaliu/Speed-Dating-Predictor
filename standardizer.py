import numpy as np

def std_scale(X):
    X_c = X - np.mean(X, axis=0)
    X_cs = X_c / np.std(X, axis=0, ddof=1)
    return X_cs

def evd_dot(X):
    X_dot = X.T @ X
    return X_dot