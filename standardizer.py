import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def std_scale(X):
    X_c = X - np.mean(X, axis=0)
    X_cs = X_c / np.std(X, axis=0, ddof=1)
    return X_cs

def evd_dot(X):
    X_dot = X.T @ X
    return X_dot

def create_distribution_plot(data, xlabel, axes, x, y):
    sns.histplot(data, bins=30, ax=axes[x, y], color='blue')
    axes[x, y].set_xlabel(xlabel, fontsize=3)
    axes[x, y].set_ylabel("Frequency", fontsize=3)
    # axes[x, y].xticks(rotation=0, fontsize=5)
    # axes[x, y].yticks(rotation=0, fontsize=5)
    axes[x, y].tick_params(axis='both', labelsize=3)

