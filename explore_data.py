import pca
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import standardizer as std
# import prince

def main():
    sdg_imputed = "speeddating_grouped_imputed.csv"
    df = pd.read_csv(sdg_imputed)
    
    X = std.std_scale(df)

    A = 3
    # t1, p1, r21 = pca.nipalspca(X.values, A)


    # pca = prince.PCA(n_components=3)
    # pca.fit(X)

    X_dot = std.evd_dot(X)
    t1, p1, r21 = pca.evd_pca(X_dot.values, A)
    # t2, p2, r22 = pca.nipalspca(X.values, A)


    #MOVING ON TO: Loadings Bar Plot
    variable_names = X.columns 
    pc1_loadings = p1[:, 0]  #loadings for PC1
    pc2_loadings = p1[:, 1]  #loadings for PC2
    pc1_scores = t1[:, 0]
    pc2_scores = t1[:, 1]

    #PLOTTING FOR LOADINGS FOR PC1
    plt.bar(variable_names, pc1_loadings, color='purple', alpha=0.8,)
    plt.title("PC1 Loadings")
    plt.ylabel("Magnitude of Loadings", fontsize=12)
    plt.xlabel("Variables", fontsize=12)
    plt.xticks(rotation=90, fontsize=5)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.show()

    #PLOTTING FOR LOADINGS FOR PC2
    plt.bar(variable_names, pc2_loadings, color='pink', alpha=0.8)
    plt.title("PC2 Loadings")
    plt.ylabel("Magnitude of Loadings", fontsize=12)
    plt.xlabel("Variables",fontsize=12)
    plt.xticks(rotation=90, fontsize=5)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.show()

    #PLOTTING FOR SCORES FOR PC1
    plt.scatter(pc1_scores, pc2_scores, color='purple', alpha = 0.08)
    plt.title("Score Plot")
    plt.ylabel("T2", fontsize=12)
    plt.xlabel("T1",fontsize=12)
    plt.xticks(rotation=90, fontsize=5)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.show()

if __name__=="__main__":
    main()