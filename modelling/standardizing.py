import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fitter import Fitter
import pandas as pd
import standardizing as std

def distribution(df):
    fig, axes = plt.subplots(9, 9) 
    idx = 0
    for i in range(0, 9):
        for j in range(0, 9):                
            col_name = df.columns[idx]
            col_data = df.iloc[:, idx] 
            std.create_distribution_plot(col_data, col_name, axes, i, j)
            j+=1
            idx +=1
            if idx > 75:
                break
    plt.tight_layout()
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, hspace=0.05, wspace=0.15)
    plt.show()

def create_distribution_plot(data, xlabel, axes, x, y):
    sns.histplot(data, bins=30, ax=axes[x, y], color='blue')
    axes[x, y].set_xlabel(xlabel, fontsize=3)
    if y==0:
        axes[x, y].set_ylabel("Frequency", fontsize=3)
    else:
        axes[x, y].set_ylabel("", fontsize=3)
    axes[x, y].tick_params(axis='both', labelsize=3)

def binomial_set(df):
    binoms = []
    for col in df.columns:
        binom = set(df[col].unique()).issubset({0, 1})
        if binom == True:
            binoms.append(col)
    return binoms

def normal_scale(X, binoms):
    X_df = pd.DataFrame(X)
    non_binoms = [col for col in X_df.columns if col not in binoms]

    X_c = X_df.copy()
    X_c[non_binoms] = (X_c[non_binoms] - X_c[non_binoms].mean(axis=0)) / X_c[non_binoms].std(axis=0, ddof=1)
    return X_c

def evd_dot(X):
    X_dot = X.T @ X
    return X_dot

def poisson_scale(X):
    return np.sqrt(X)


    

