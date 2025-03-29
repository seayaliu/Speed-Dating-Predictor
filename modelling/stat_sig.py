
t, p, R2 = nipalspca(X, 4)
n, A = X.shape


# calculate residuals and SPE
#before going into the function, STANDARD DATA RAAHHHH
X =((df - df.mean()) / df.std(ddof=1))
X_hat = np.dot(t, p.T)
residuals = X - X_hat
SPE = np.sum((X - X_hat) ** 2, axis=1)

# calculate confidence interval lines
mean = np.mean(SPE)
var = np.std(SPE) ** 2
doff = 2 * (mean ** 2) / var

spe_95 = (var/(2*mean)) * chi2.ppf(0.95, doff)
spe_99 = (var/(2*mean)) * chi2.ppf(0.99, doff)
