import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

def spe(df, n, t, p, X):
    X_hat = np.dot(t, p.T)
    residuals = X - X_hat
    SPE = np.sum((X - X_hat) ** 2, axis=1)

    mean = np.mean(SPE)
    var = np.std(SPE) ** 2
    doff = 2 * (mean ** 2) / var

    spe_95 = (var/(2*mean)) * chi2.ppf(0.95, doff)
    spe_99 = (var/(2*mean)) * chi2.ppf(0.99, doff)

    plt.plot(range(1, n + 1), SPE, marker='.', linestyle='-', label="SPE")
    plt.axhline(y=spe_95, color='r', linestyle='--', label="95% Confidence Interval")
    plt.axhline(y=spe_99, color='b', linestyle='-.', label="99% Confidence Interval")
    plt.xlabel("Observation Number")
    plt.ylabel("SPE Value")
    plt.title("SPE Value vs Observation Number")
    plt.legend()
    plt.show()

    greater_95 = np.where(SPE > spe_95)[0] + 1
    greater_99 = np.where(SPE > spe_99)[0] + 1
    # print(f"Observations exceeding 95% CI: {greater_95}")
    # print(f"Observations exceeding 99% CI: {greater_99}")