import pca
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import standardizer as std
import seaborn as sns
# import prince

def distribution(df):
    fig, axes = plt.subplots(9, 9) 
    
    idx = 0
    for i in range(0, 9):
        for j in range(0, 9):                
            col_name = df.columns[idx]
            col_data = df.iloc[:, idx] 
            std.create_distribution_plot(col_data, col_name, axes, i, j)
            j+=1
            idx +=1
            if idx > 75:
                break

    plt.tight_layout()
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, hspace=0.05, wspace=0.15)
    plt.show()


def plot_loadings(variable_names, p):
    pc1_loadings = p[:, 0]  #loadings for PC1
    pc2_loadings = p[:, 1]  #loadings for PC2

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

def plot_scores(t):
    pc1_scores = t[:, 0]
    pc2_scores = t[:, 1]

    #PLOTTING FOR SCORES FOR PC1
    plt.scatter(pc1_scores, pc2_scores, color='purple', alpha = 0.08)
    plt.title("Score Plot")
    plt.ylabel("T2", fontsize=12)
    plt.xlabel("T1",fontsize=12)
    plt.xticks(rotation=90, fontsize=5)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.show()

def main():
    sdg_imputed = "speeddating_grouped_imputed.csv"
    sdte_imputed = "speeddating_target_encoded_imputed.csv"
    sdg_NaN = "speeddating_grouped_NaN.csv"
    sdte_imputed = "speeddating_target_encoded_NaN.csv"

    df_gi = pd.read_csv(sdg_imputed)
    df_tei = pd.read_csv(sdte_imputed)
    df_gn = pd.read_csv(sdg_NaN)
    df_ten = pd.read_csv(sdte_imputed)

    dfs = [df_gi, df_tei, df_gn, df_ten]
    distribution(df_tei)

    # dstr_age = df_gi["age"]
    # std.create_distribution_plot(dstr_age)

    
    # X = std.std_scale(df)

    A = 3
    # t1, p1, r21 = pca.nipalspca(X.values, A)


    # pca = prince.PCA(n_components=3)
    # pca.fit(X)

    # X_dot = std.evd_dot(X)
    # t1, p1, r21 = pca.evd_pca(X_dot.values, A)
    # t2, p2, r22 = pca.nipalspca(X.values, A)

    # variable_names = X.columns 
    # plot_loadings(variable_names, p1)
    # plot_scores(t1)






if __name__=="__main__":
    main()