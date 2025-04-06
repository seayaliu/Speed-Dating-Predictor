import pca
import pls
import standardizing as std
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns
import stat_sig as sig
import scree
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
    pclast_loadings = p[:, -1]

    #PLOTTING FOR LOADINGS FOR PC1
    title1 = "PC1 Loadings - " + method + " " + data_name
    plt.bar(variable_names, pc1_loadings, color='purple', alpha=0.8,)
    plt.title(title1, fontsize=12)
    plt.ylabel("Magnitude of Loadings", fontsize=12)
    plt.xlabel("Variables", fontsize=12)
    plt.xticks(rotation=90, fontsize=5)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    #PLOTTING FOR LOADINGS FOR PC2
    title2 = "PC2 Loadings - " + method + " " + data_name
    plt.bar(variable_names, pc2_loadings, color='pink', alpha=0.8)
    plt.title(title2, fontsize=12)
    plt.ylabel("Magnitude of Loadings", fontsize=12)
    plt.xlabel("Variables",fontsize=12)
    plt.xticks(rotation=90, fontsize=5)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    #PLOTTING FOR LOADINGS FOR PC20
    title2 = "Last PC Loadings - " + method + " " + data_name
    plt.bar(variable_names, pclast_loadings, color='blue', alpha=0.8)
    plt.title(title2, fontsize=12)
    plt.ylabel("Magnitude of Loadings", fontsize=12)
    plt.xlabel("Variables",fontsize=12)
    plt.xticks(rotation=90, fontsize=5)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
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

def run_nipalspca(df, df_name, A):
    binoms = std.binomial_set(df)
    X = std.normal_scale(df, binoms)
    X = X.drop(columns=["decision", "decision_o", "match"])

    # nipals pca
    variable_names = X.columns 
    t1, p1, r21 = pca.nipalspca(X.values, A)
    n1, A = X.shape
    plot_loadings(variable_names, p1, "NIPALS PCA", df_name)
    # plot_scores(t1, "NIPALS PCA", df_name)
    scree.scree_plot(A, r21)
    # sig.spe(df, n1, t1, p1, X)
    # sig.hotellings_t2(t1)
    return t1, p1, r21

def run_nipalspca_NaN(df, df_name, A):
    X = df.drop(columns=["decision", "decision_o", "match"])

    # nipals pca
    variable_names = X.columns 
    t1, p1, r21 = pca.nipalspcaNaN(X.values, A)
    n1, A = X.shape
    plot_loadings(variable_names, p1, "NIPALS PCA NaN", df_name)
    # plot_scores(t1, "NIPALS PCA", df_name)
    scree.scree_plot(A, r21)
    # sig.spe(df, n1, t1, p1, X)
    # sig.hotellings_t2(t1)
    return t1, p1, r21

def save_results(t):
    df = pd.DataFrame(t, columns=[f'PC{i+1}' for i in range(t.shape[1])])
    df.to_csv('pca_scores.csv', index=False)

def main():
    sdg_imputed = "../data/cleaned/speeddating_grouped_imputed.csv"
    sdg_NaN = "../data/cleaned/speeddating_grouped_NaN.csv"

    df_gi = pd.read_csv(sdg_imputed)
    df_gn = pd.read_csv(sdg_NaN)

    numComponents = 100
    numComponents1 = 20
    numComponents2 = 45


    # distribution(df_gi)
    t1, p1, r211 = run_nipalspca(df_gi, "SDImp_FGroups", numComponents1)
    save_results(t1)
    # run_sk_pca(df_gi, "SDImp_FGroups", numComponents)
    # run_nipalspca_NaN(df_gn, "SDNaN_FGroups", numComponents1)
    

if __name__=="__main__":
    main()