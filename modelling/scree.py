import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# plot scree plot for cumulative explained variance
def scree_plot(A, r2, part):
    title = 'Scree Plot (Explained Variance in Y)' + part
    vals = range(1, len(r2) + 1)
    plt.plot(vals, r2, color='#791523')
    plt.title(title)
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance in Y')
    plt.xticks(vals, fontsize=5)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# plot two lines on a scre plot for cumulative explained variance
def scree_plot2(A, r21, r22, name1, name2):
    title = 'Scree Plot (Explained Variance in Y)'
    vals = range(1, len(r21) + 1)
    plt.plot(vals, r21, color='#791523')
    plt.plot(vals, r22, color='#495965')
    plt.title(title)
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance in Y')
    plt.legend([name1, name2])
    plt.xticks(vals, fontsize=5)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
