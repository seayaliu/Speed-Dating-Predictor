import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import standardizing as std
import statsmodels.multivariate.pca as pca

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

def nipalspcaNaN(x, A=2):
    x = x.copy()

    t = np.zeros((x.shape[0], A))  #scores 
    p = np.zeros((x.shape[1], A))  #loadings
    explained_variance = np.zeros(A)  #R^2

    col_means = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(col_means, inds[1])
    x -= np.nanmean(x, axis=0)

    #getting the total variance for the data so the R^2 can be calculated later skater based on the total variance
    total_variance = np.nansum(np.nanvar(x, axis=0, ddof=1))  

    #goes through for number of components that want to be calculated 
    for i in range(A):
        t_pca = x[:, np.nanargmax(np.nanvar(x, axis=0))].copy()
        t_pca[np.isnan(t_pca)] = 0
        t_pca = t_pca.reshape(-1, 1)

        #set a covergence limit so that it runs for a good number of times
        for _ in range(500):  
            p_pca = (np.nansum(x * t_pca, axis=0) / np.nansum(t_pca ** 2)).reshape(-1, 1)  #based off of slides for a bigger dataset, get loadings
            p_pca = p_pca / np.linalg.norm(p_pca)  #normalize loadings as the next step in the NIPALS steps
    
            t_pca_new = np.nansum(x * p_pca.T, axis=1).reshape(-1, 1) / np.nansum((p_pca.T)**2)  #And then get the scores from the loadings

            #THE STEP AFTER IS CHECKING FOR CONVERGENCE
            if np.linalg.norm(t_pca - t_pca_new) < 1e-6:  #from slides: they said to use up until e10-8 so that's what we're doing
                break
            t_pca = t_pca_new #store right after as the slides say LOL

        #and now we deflate the data to remove any of the explained variance that we just calculated for the column
        x = x - (t @ p.T)

        #STORE THE LOADINGS AND SCORES PLEASE
        t[:, i] = t_pca.ravel()
        p[:, i] = p_pca.ravel()

        #Calculating the explained variance for this component now...this stuff is completely based on the EIGENVALUE of the covariance matrix 
        explained_variance[i] = np.nansum(np.nanvar(np.outer(t_pca, p_pca), axis=0, ddof=1))  #variance explained by each component
        
    #this gets the variance of a component over the total variance
    R2 = explained_variance / total_variance
    
    #convert R2 to cumulative so explained variance overall all of the columns for the component
    R2 = np.nancumsum(R2)

    return t, p, R2