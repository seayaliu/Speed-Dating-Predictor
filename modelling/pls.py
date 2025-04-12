import numpy as np
from sklearn.cross_decomposition import PLSRegression

def sklearnpls(X, Y, A):
    pls = PLSRegression(A)
    pls.fit(X, Y)
    # y_pred = pls.predict(X)
    SSY_total = np.sum(np.square(Y - np.mean(Y)))  # Total variance in Y
    explained_variance = [np.sum(np.square(pls.y_scores_[:, i])) / SSY_total for i in range(A)]

    return pls.x_scores_, pls.y_scores_, pls.x_weights_, pls.y_weights_, pls.x_loadings_, np.array(explained_variance)

def nipalspls(X, Y, A): # A = number of components, use A = 3 for assignment

    x_c = X - np.mean(X, axis=0)
    x_cs = x_c / np.std(X, axis=0)
    y_c = Y - np.mean(Y, axis=0)
    y_cs = y_c / np.std(Y, axis=0)

    t = np.zeros((x_cs.shape[0], A))  #scores 
    w = np.zeros((x_cs.shape[1], A))  #loadings
    u = np.zeros((y_cs.shape[0], A))  #scores 
    c = np.zeros((y_cs.shape[1], A ))  #loadings
    p = np.zeros((x_cs.shape[1], A))
    r2 = np.zeros(A) 

    explained_variance = np.var(y_cs, axis=0)

    #goes through for number of components that want to be calculated 
    for i in range(A):

        u_pls = y_cs[:, 0] # selecting the first column of y_cs as our initial estimate for u

        for j in range(300):  
            w_pls = np.dot(x_cs.T, u_pls) / np.dot(u_pls.T, u_pls) 
            w_pls = w_pls / np.linalg.norm(w_pls)

            t_pls = np.dot(x_cs, w_pls) / np.dot(w_pls.T, w_pls) 

            c_pls = np.dot(y_cs.T, t_pls) / np.dot(t_pls.T, t_pls)
           
            u_pls_new = np.dot(y_cs, c_pls) / np.dot(c_pls.T, c_pls)
     
            #THE STEP AFTER IS CHECKING FOR CONVERGENCE
            if np.linalg.norm(u_pls - u_pls_new) < 1e-10:
                break
            u_pls = u_pls_new #store right after as the slides say LOL

        # calculate loadings for x_cs space
        p_i = np.dot(x_cs.T, t_pls) / np.dot(t_pls.T, t_pls) 
        p[:, i] = p_i

        # deflate data
        temp = y_cs
        x_cs = x_cs - np.outer(t_pls, p_i)
        y_cs = y_cs - np.outer(t_pls, c_pls)

        num = np.var(temp, axis=0)
        r2_pls = 1 - num/explained_variance
        r2[i] = np.sum(r2_pls)

        t[:, i] = t_pls
        u[:, i] = u_pls
        w[:, i] = w_pls
        c[:, i] = c_pls

    w_star = np.dot(w , np.linalg.pinv(p.T @ w))


    # should return scores (t, u), loadings (w*, c, p), and the R2 for each component
    return  t, u, w_star, c, p, r2

def nipalspls_NaN(X, Y, A): # A = number of components, use A = 3 for assignment

    x_c = X - np.nanmean(X, axis=0)
    x_cs = x_c / np.nanstd(X, axis=0)
    y_c = Y - np.nanmean(Y, axis=0)
    y_cs = y_c / np.nanstd(Y, axis=0)

    t = np.zeros((x_cs.shape[0], A))  #scores 
    w = np.zeros((x_cs.shape[1], A))  #loadings
    u = np.zeros((y_cs.shape[0], A))  #scores 
    c = np.zeros((y_cs.shape[1], A ))  #loadings
    p = np.zeros((x_cs.shape[1], A))
    r2 = np.zeros(A) 

    explained_variance = np.nanvar(y_cs, axis=0)

    #goes through for number of components that want to be calculated 
    for i in range(A):

        u_pls = np.nan_to_num(y_cs[:, 0])  # selecting the first column of y_cs as our initial estimate for u

        for j in range(300):  
            w_pls = np.nansum(x_cs.T * u_pls, axis=1) / np.nansum(u_pls**2)
            w_pls = w_pls / np.linalg.norm(w_pls)  # Normalize

            t_pls = np.nansum(x_cs * w_pls, axis=1) / np.nansum(w_pls**2)

            c_pls = np.nansum(y_cs.T * t_pls, axis=1) / np.nansum(t_pls**2)

            u_pls_new = np.nansum(y_cs * c_pls, axis=1) / np.nansum(c_pls**2)
     
            #THE STEP AFTER IS CHECKING FOR CONVERGENCE
            if np.linalg.norm(u_pls - u_pls_new) < 1e-10:
                break
            u_pls = u_pls_new #store right after as the slides say LOL

        # calculate loadings for x_cs space
        p_i = np.nansum(x_cs.T * t_pls, axis=1) / np.nansum(t_pls**2)
        p[:, i] = p_i

        # deflate data
        x_cs = x_cs - np.outer(t_pls, p_i)
        y_cs = y_cs - np.outer(t_pls, c_pls)

        num = np.nanvar(y_cs, axis=0)
        r2_pls = num/explained_variance
        r2[i] = np.nansum(r2_pls)

        t[:, i] = t_pls
        u[:, i] = u_pls
        w[:, i] = w_pls
        c[:, i] = c_pls

    w_star = np.dot(w , np.linalg.pinv(p.T @ w))


    # should return scores (t, u), loadings (w*, c, p), and the R2 for each component
    return  t, u, w_star, c, p, r2