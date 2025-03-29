import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def nipalspca(x, A):
    x = x.copy()

    t = np.zeros((x.shape[0], A))  #scores 
    p = np.zeros((x.shape[1], A))  #loadings
    explained_variance = np.zeros(A)  #R^2

    #getting the total variance for the data so the R^2 can be calculated later skater based on the total variance
    total_variance = np.sum(np.var(x, axis=0, ddof=1))  

    #goes through for number of components that want to be calculated 
    for i in range(A):
        t_pca = x[:, 0]

        #set a covergence limit so that it runs for a good number of times
        for _ in range(500):  
            p_pca = np.dot(x.T, t_pca) / np.dot(t_pca, t_pca)  #based off of slides for a bigger dataset, get loadings
            p_pca = p_pca / np.linalg.norm(p_pca)  #normalize loadings as the next step in the NIPALS steps
            t_pca_new = np.dot(x, p_pca)  #And then get the scores from the loadings
            
            #THE STEP AFTER IS CHECKING FOR CONVERGENCE
            if np.allclose(t_pca, t_pca_new, atol=1e-8):  #from slides: they said to use up until e10-8 so that's what we're doing
                break
            t_pca = t_pca_new #store right after as the slides say LOL

        #and now we deflate the data to remove any of the explained variance that we just calculated for the column
        x = x - np.outer(t_pca, p_pca)

        #STORE THE LOADINGS AND SCORES PLEASE
        t[:, i] = t_pca
        p[:, i] = p_pca

        #Calculating the explained variance for this component now...this stuff is completely based on the EIGENVALUE of the covariance matrix 
        explained_variance[i] = np.sum(np.var(np.outer(t_pca, p_pca), axis=0, ddof=1))  #variance explained by each component
        
    #this gets the variance of a component over the total variance
    R2 = explained_variance / total_variance
    
    #convert R2 to cumulative so explained variance overall all of the columns for the component
    R2 = np.cumsum(R2)

    return t, p, R2

def evd_pca(X, A):
    X = X.copy()
    
    eig_vals, eig_vecs = np.linalg.eig(X)
    
    # Sorting eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[sorted_indices]
    eig_vecs = eig_vecs[:, sorted_indices]
    
    p = eig_vecs[:, :A]  # Loadings
    t = X @ p  # Score vectors
    
    variance = np.sum(eig_vals)
    explained_var = eig_vals[:A]
    r2 = explained_var / variance
    
    return t, p, r2
