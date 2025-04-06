import pca
import standardizing as std
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import stat_sig as sig
import scree
import numpy as np
from sklearn.model_selection import KFold
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


def plot_loadings(variable_names, p, name):
    pc1_loadings = p[:, 0]  #loadings for PC1
    pc2_loadings = p[:, 1]  #loadings for PC2
    pclast_loadings = p[:, -1]

    #PLOTTING FOR LOADINGS FOR PC1
    title1 = "PC1 Loadings" + name
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
    title2 = "PC2 Loadings" + name
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
    title2 = "Last PC Loadings" + name
    plt.bar(variable_names, pclast_loadings, color='blue', alpha=0.8)
    plt.title(title2, fontsize=12)
    plt.ylabel("Magnitude of Loadings", fontsize=12)
    plt.xlabel("Variables",fontsize=12)
    plt.xticks(rotation=90, fontsize=5)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    plot_top_loadings(pclast_loadings, "Last PC", variable_names, 30)


def plot_scores(t, name):
    pc1_scores = t[:, 0]
    pc2_scores = t[:, 1]

    #PLOTTING FOR SCORES
    titlet = "Score Plot" + name
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

def run_nipalspca(df, A, title):
    binoms = std.binomial_set(df)
    X = std.normal_scale(df, binoms)
    # cross_validate(X.to_numpy())
    X = X.drop(columns=["decision", "decision_o", "match"])

    # nipals pca
    variable_names = X.columns 
    t1, p1, r21 = pca.nipalspca(X.values, A)
    n1, A = X.shape
    plot_loadings(variable_names, p1, title)
    plot_scores(t1, title)
    print(title, "Fit -", n1, "Observations")
    scree.scree_plot(A, r21)
    sig.spe(df, n1, t1, p1, X)
    return t1, p1, r21

def save_results(t, file_name):
    df = pd.DataFrame(t, columns=[f'PC{i+1}' for i in range(t.shape[1])])
    df.to_csv(file_name, index=False)

def cross_validate(X):
    groups = KFold(n_splits=10, shuffle=True, random_state=52)

    r2_components, q2_components = pca.kfold_pca(X, groups)

    for i, (r2, q2) in enumerate(zip(r2_components, q2_components), start=1):
        print(f"Components: {i}, R²: {r2:.4f}, Q²: {q2:.4f}")


def main():
    sdg_imputed = "../data/cleaned/speeddating_grouped_imputed.csv"
    sdg_5050 = "../data/cleaned/speeddating_grouped_imputed_balanced5050.csv"
    sdg_4060 = "../data/cleaned/speeddating_grouped_imputed_balanced4060.csv"

    df_gi = pd.read_csv(sdg_imputed)
    df_5050 =pd.read_csv(sdg_5050)
    df_4060 = pd.read_csv(sdg_4060)

    numComponents = 18

    t1, p1, r21 = run_nipalspca(df_gi, numComponents, " - NIPALS PCA, 18 PCs")
    save_results(t1, "scores_A18.csv")

    t50, p50, r250 = run_nipalspca(df_5050, numComponents, " - NIPALS PCA, 18 PCs, Undersampled Data (50% Match 1)")
    save_results(t50, "scores_A18_5050balance.csv")

    t40, p40, r240 = run_nipalspca(df_4060, numComponents, "- NIPALS PCA, 18 PCs, Undersampled Data (40% Match 1)")
    save_results(t40, "scores_A18_4060balance.csv")
    

if __name__=="__main__":
    main()