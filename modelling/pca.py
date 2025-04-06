import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import standardizing as std
import statsmodels.multivariate.pca as pca
from sklearn.model_selection import KFold

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


def get_q2_please(x_true, x_pred):
    total_var = np.sum((x_true - np.mean(x_true, axis=0)) ** 2)
    press = np.sum((x_true - x_pred) ** 2)
    return 1 - (press / total_var)


def kfold_pca(X, groups):
    r2_components = []
    q2_components = []
    for n_components in range(1, 41):
        print("go")
        r2_groups = []
        q2_groups = []
        for train_i, test_i in groups.split(X):
            
            #splitting dataset to get the train and test separately 
            X_train, X_test = X[train_i], X[test_i]

            #NIPALS PCA IS SO BACK
            t_train, p_train, R2_train = nipalspca(X_train, n_components)

            #gets r^2 value here for the training data for each iteration
            r2_groups.append(R2_train[-1])  

            #for each iteration, projects the testing data on to the trained one
            t_test = np.dot(X_test, p_train) #equations taken from previous lecture slides
            X_again = np.dot(t_test, p_train.T)

            #gets Q^2 values from the function described above
            q2 = get_q2_please(X_test, X_again)
            q2_groups.append(q2)

        #for each of the components, get the average R^2 and Q^2 values across the 4 groups
        r2_components.append(np.mean(r2_groups))
        q2_components.append(np.mean(q2_groups))

    return r2_components, q2_components



