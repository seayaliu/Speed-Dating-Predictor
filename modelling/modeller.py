import pca
import pls
import standardizing as std
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns
import stat_sig as sig
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


def plot_loadings(variable_names, p, method, data_name):
    pc1_loadings = p[:, 0]  #loadings for PC1
    pc2_loadings = p[:, 1]  #loadings for PC2

    #PLOTTING FOR LOADINGS FOR PC1
    title1 = "PC1 Loadings - " + method + " " + data_name
    plt.bar(variable_names, pc1_loadings, color='purple', alpha=0.8,)
    plt.title(title1, fontsize=12)
    plt.ylabel("Magnitude of Loadings", fontsize=12)
    plt.xlabel("Variables", fontsize=12)
    plt.xticks(rotation=90, fontsize=5)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.show()

    #PLOTTING FOR LOADINGS FOR PC2
    title2 = "PC2 Loadings - " + method + " " + data_name
    plt.bar(variable_names, pc2_loadings, color='pink', alpha=0.8)
    plt.title(title2, fontsize=12)
    plt.ylabel("Magnitude of Loadings", fontsize=12)
    plt.xlabel("Variables",fontsize=12)
    plt.xticks(rotation=90, fontsize=5)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.show()

def plot_scores(t, method, data_name):
    pc1_scores = t[:, 0]
    pc2_scores = t[:, 1]

    #PLOTTING FOR SCORES
    titlet = "Score Plot - " + method + " " + data_name
    plt.scatter(pc1_scores, pc2_scores, color='purple', alpha = 0.08)
    plt.title(titlet, fontsize=12)
    plt.ylabel("T2", fontsize=12)
    plt.xlabel("T1",fontsize=12)
    plt.xticks(rotation=90, fontsize=5)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.show()

def main():
    sdg_imputed = "../data/cleaned/speeddating_grouped_imputed.csv"
    sdte_imputed = "../data/cleaned/speeddating_target_encoded_imputed.csv"
    sdg_NaN = "../data/cleaned/speeddating_grouped_NaN.csv"
    sdte_NaN = "../data/cleaned/speeddating_target_encoded_NaN.csv"

    df_gi = pd.read_csv(sdg_imputed)
    df_tei = pd.read_csv(sdte_imputed)
    df_gn = pd.read_csv(sdg_NaN)
    df_ten = pd.read_csv(sdte_NaN)

    dfs = [df_gi, df_tei]
    df_names = ["SDImp_FGroups", "SDImp_FTEncodings"]
    # dfs = [df_gi, df_tei, df_gn, df_ten]
    # df_names = ["SDImp_FGroups", "SDImp_FTEncodings", "SDNaN_FTGroups", "SDNaN_FTEncodings"]

    for idx, df in enumerate(dfs):
        distribution(df)
        binoms = std.binomial_set(df)
        X = std.normal_scale(df, binoms)

        variable_names = X.columns 
        variable_names2 = list(variable_names)
        variable_names2.remove("decision")
        variable_names2.remove("decision_o")
        variable_names2.remove("match")

        # nipals pca
        t1, p1, r21 = pca.nipalspca(X.values, 3)
        n1, A1 = X.shape
        plot_loadings(variable_names, p1, "NIPALS PCA", df_names[idx])
        plot_scores(t1, "NIPALS PCA", df_names[idx])
        # sig.spe(df, n1, t1, p1, X) # spe

        # evd pca
        if idx < 2:
            t2, p2, r22 = pca.evd_pca(X, 3)
            t2_np = t2.to_numpy()
            plot_loadings(variable_names, p2, "EVD PCA", df_names[idx])
            plot_scores(t2_np, "EVD PCA", df_names[idx])
            # sig.spe(df, n1, t2_np, p2, X)

        # pls
            Xpls = df.iloc[:, list(range(0, 58)) + list(range(61, df.shape[1]))].to_numpy()
            binoms = std.binomial_set(df)
            Xpls = std.normal_scale(Xpls, binoms)
            Ypls = df.iloc[:, [58]].to_numpy()
            t, u, w_star, c, p, r2 = pls.nipalspls(Xpls, Ypls, 3)

            n_pls, A2 = X.shape            
            plot_loadings(variable_names2, p, "NIPALS PLS", df_names[idx])
            plot_scores(t, "NIPALS PLS", df_names[idx])
            # sig.spe(df, n_pls, t, p, Xpls)

if __name__=="__main__":
    main()