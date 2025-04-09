import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f, chi2

def spe(df, n, t, p, X):
    X_hat = np.dot(t, p.T)
    residuals = X - X_hat
    SPE = np.sum((X - X_hat) ** 2, axis=1)

    mean = np.mean(SPE)
    var = np.std(SPE) ** 2
    doff = 2 * (mean ** 2) / var

    spe_95 = (var/(2*mean)) * chi2.ppf(0.95, doff)
    spe_99 = (var/(2*mean)) * chi2.ppf(0.99, doff)

    plt.plot(range(1, n + 1), SPE, linewidth=0.5, marker='.', linestyle='-', label="SPE", color='#791523')
    plt.axhline(y=spe_95, color='#FDBF57', linestyle='--', label="95% Confidence Interval")
    plt.axhline(y=spe_99, color='#495965', linestyle='-.', label="99% Confidence Interval")
    plt.xlabel("Observation Number")
    plt.ylabel("SPE Value")
    plt.title("SPE Value vs Observation Number")
    plt.legend()
    plt.show()

    greater_95 = np.where(SPE > spe_95)[0] + 1
    greater_99 = np.where(SPE > spe_99)[0] + 1
    print("Number of observations exceeding 95% CI:", len(greater_95))
    print("Number of observations exceeding 99% CI:", len(greater_99))

def hotellings_t2(t):
    n, A = t.shape

    # confidence intervals at 95% and 99% 
    CI_95 = 0.95
    CI_99 = 0.99

    # calculating 95% and 99% confidence thresholds using f-distribution
    F_95 = f.ppf(CI_95, A, n - A) * (A * (n - 1)) / (n - A)
    F_99 = f.ppf(CI_99, A, n - A) * (A * (n - 1)) / (n - A)

    # hotellings t2 formula for each observation, sums squared normalized scores across all principal components
    T2 = np.sum((t / np.std(t, axis=0, ddof=1))**2, axis=1)

    # plotting line plot
    plt.plot(range(1, n + 1), T2, linewidth=0.5, marker='.', linestyle='-', label="Hotelling's T2", color='#791523')
    plt.axhline(y=F_95, color='r', linestyle='--', label="95% Confidence Interval")
    plt.axhline(y=F_99, color='b', linestyle='-.', label="99% Confidence Interval")
    plt.xlabel("Observation Number")
    plt.ylabel("Hotelling's T2")
    plt.title("Hotelling's T2 vs Observation Number")
    plt.legend()
    plt.show()

    # show obervation nums which exceed 95 and 99 CI's 
    greater_95 = np.where(T2 > F_95)[0] + 1
    greater_99 = np.where(T2 > F_99)[0] + 1
    # print(f"Observations exceeding 95% CI: {greater_95}")
    # print(f"Observations exceeding 99% CI: {greater_99}")