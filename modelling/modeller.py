import pca
import standardizing as std
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import stat_sig as sig
import scree
import numpy as np
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
    pclast_loadings = p[:, -1]

    #PLOTTING FOR LOADINGS FOR PC1
    title1 = "PC1 Loadings"
    plt.bar(variable_names, pc1_loadings, color='purple', alpha=0.8,)
    plt.title(title1, fontsize=12)
    plt.ylabel("Magnitude of Loadings", fontsize=12)
    plt.xlabel("Variables", fontsize=12)
    plt.xticks(rotation=90, fontsize=5)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    plot_top_loadings(pc1_loadings, "PC1", variable_names, 30)

    #PLOTTING FOR LOADINGS FOR PC2
    title2 = "PC2 Loadings"
    plt.bar(variable_names, pc2_loadings, color='pink', alpha=0.8)
    plt.title(title2, fontsize=12)
    plt.ylabel("Magnitude of Loadings", fontsize=12)
    plt.xlabel("Variables",fontsize=12)
    plt.xticks(rotation=90, fontsize=5)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    plot_top_loadings(pc2_loadings, "PC2", variable_names, 30)

    #PLOTTING FOR LOADINGS FOR PC20
    title2 = "Last PC Loadings"
    plt.bar(variable_names, pclast_loadings, color='blue', alpha=0.8)
    plt.title(title2, fontsize=12)
    plt.ylabel("Magnitude of Loadings", fontsize=12)
    plt.xlabel("Variables",fontsize=12)
    plt.xticks(rotation=90, fontsize=5)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    plot_top_loadings(pclast_loadings, "Last PC", variable_names, 30)


def plot_scores(t):
    pc1_scores = t[:, 0]
    pc2_scores = t[:, 1]

    #PLOTTING FOR SCORES
    titlet = "Score Plot"
    plt.scatter(pc1_scores, pc2_scores, color='purple', alpha = 0.08)
    plt.title(titlet, fontsize=12)
    plt.ylabel("T2", fontsize=12)
    plt.xlabel("T1",fontsize=12)
    plt.xticks(rotation=90, fontsize=5)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.show()

def plot_top_loadings(p, pc_label, variable_names, num):
    top_idxs = np.argsort(np.abs(p))[-num:][::-1]
    top_varis = [variable_names[i] for i in top_idxs]
    top_p = p[top_idxs]
    top_p = abs(top_p)

    title = "Top " + str(num) + " Loadings for " + pc_label 
    plt.bar(top_varis, top_p, alpha=0.8)
    plt.title(title, fontsize=12)
    plt.ylabel("Magnitude of Loadings", fontsize=12)
    plt.xlabel("Variables", fontsize=12)
    plt.xticks(rotation=90, fontsize=10)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def run_nipalspca(df, df_name, A):
    binoms = std.binomial_set(df)
    X = std.normal_scale(df, binoms)
    X = X.drop(columns=["decision", "decision_o", "match"])

    # nipals pca
    variable_names = X.columns 
    t1, p1, r21 = pca.nipalspca(X.values, A)
    n1, A = X.shape
    plot_loadings(variable_names, p1)
    plot_scores(t1)
    scree.scree_plot(A, r21)
    sig.spe(df, n1, t1, p1, X)
    sig.hotellings_t2(t1)
    return t1, p1, r21

def save_results(t):
    df = pd.DataFrame(t, columns=[f'PC{i+1}' for i in range(t.shape[1])])
    df.to_csv('pca_scores.csv', index=False)

def main():
    sdg_imputed = "../data/cleaned/speeddating_grouped_imputed.csv"
    sdg_NaN = "../data/cleaned/speeddating_grouped_NaN.csv"

    df_gi = pd.read_csv(sdg_imputed)
    # df_gn = pd.read_csv(sdg_NaN)

    # numComponents = 100
    numComponents1 = 20
    numComponents2 = 45

    t1, p1, r21 = run_nipalspca(df_gi, "SDImp_FGroups", numComponents1)
    save_results(t1)

    t2, p2, r22 = run_nipalspca(df_gi, "SDImp_FGroups", numComponents2)
    save_results(t2)
    

if __name__=="__main__":
    main()