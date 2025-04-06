import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def scree_plot(A, r2):
    vals = range(1, len(r2) + 1)
    plt.plot(vals, r2, color='blue')
    plt.title('Scree Plot (Explained Variance in Y)')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance in Y')
    plt.xticks(vals)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
