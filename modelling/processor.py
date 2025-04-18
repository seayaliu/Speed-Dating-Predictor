import pca
import standardizing as std
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import stat_sig as sig
import scree
import numpy as np
from sklearn.model_selection import KFold
import pls

# plot first 3 principal components' loadings plots
def plot_loadings(variable_names, p, name):
    pc1_loadings = p[:, 0]      # loadings for PC1
    pc2_loadings = p[:, 1]      # loadings for PC2
    pc3_loadings = p[:, 2]      # loadings for PC3

    # plotting loadings for PC1
    title1 = "PC1 Loadings" + name
    plt.bar(variable_names, pc1_loadings, color='#791523', alpha=0.8)
    plt.title(title1, fontsize=12)
    plt.ylabel("Magnitude of Loadings", fontsize=12)
    plt.xlabel("Variables", fontsize=12)
    plt.xticks(fontsize=5, rotation=90)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # plotting the top 30 loadings for PC1
    plot_top_loadings(pc1_loadings, "PC1", variable_names, 30)

    # plotting loadings for PC2
    title2 = "PC2 Loadings" + name
    plt.bar(variable_names, pc2_loadings, color='#791523', alpha=0.8)
    plt.title(title2, fontsize=12)
    plt.xlabel("Magnitude of Loadings", fontsize=12)
    plt.ylabel("Variables",fontsize=12)
    plt.xticks(fontsize=5, rotation=90)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # plotting the top 30 loadings for PC2
    plot_top_loadings(pc2_loadings, "PC2", variable_names, 30)

    # plotting loadings for PC3
    title3 = "PC3 Loadings" + name
    plt.bar(variable_names, pc3_loadings, color='#791523', alpha=0.8)
    plt.title(title3, fontsize=12)
    plt.xlabel("Magnitude of Loadings", fontsize=12)
    plt.ylabel("Variables",fontsize=12)
    plt.xticks(fontsize=5, rotation=90)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # plotting the top 30 loadings for PC3
    plot_top_loadings(pc3_loadings, "PC3", variable_names, 30)


# scores plot
def plot_scores(t, name):
    pc1_scores = t[:, 0]    # scores for PC1
    pc2_scores = t[:, 1]    # scores for PC2

    # plotting scores
    titlet = "Score Plot" + name
    plt.scatter(pc1_scores, pc2_scores, color='#791523', alpha = 0.08)
    plt.title(titlet, fontsize=12)
    plt.ylabel("T2", fontsize=12)
    plt.xlabel("T1",fontsize=12)
    plt.xticks(fontsize=5)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.show()

# plotting top loadings 
def plot_top_loadings(p, pc_label, variable_names, num):
    # get top loadings from vectors
    top_idxs = np.argsort(np.abs(p))[-num:][::-1]
    top_varis = [variable_names[i] for i in top_idxs]
    top_p = p[top_idxs]
    top_p = top_p

    # sorted loadings plot
    title = "Top " + str(num) + " Loadings for " + pc_label 
    plt.barh(np.flip(top_varis), np.flip(top_p), color = '#791523', alpha=0.8)
    plt.title(title, fontsize=12)
    plt.xlabel("Magnitude of Loadings", fontsize=12)
    plt.ylabel("Variables", fontsize=12)
    plt.yticks(fontsize=8)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# run nipals pca 
def run_nipalspca(df, A, title):
    binoms = std.binomial_set(df)
    X = std.normal_scale(df, binoms)
    # # cross validate nipals here to determine # of PCs (commented out for time)
    # cross_validate(X.to_numpy())
    X = X.drop(columns=["decision", "decision_o", "match"])

    # nipals pca
    variable_names = X.columns 
    t1, p1, r21 = pca.nipalspca(X.values, A)
    n1, A = X.shape
    plot_loadings(variable_names, p1, title)    # plot loadings
    plot_scores(t1, title)    # plot scores
    print(title, "Fit -", n1, "Observations")    # scree plot result header
    scree.scree_plot(A, r21, title)   # plot scree plot
    sig.spe(df, n1, t1, p1, X)   # plot SPE
    return t1, p1, r21

# run nipals pls for scree plot
def test_pls(df, A, title):
    binoms = std.binomial_set(df)
    X = std.normal_scale(df, binoms)
    Y = X[['match']].copy()
    X = X.drop(columns=["decision", "decision_o", "match"])

    variable_names = X.columns 
    t, u, w_star, c, p, r2 = pls.nipalspls(X.to_numpy(), Y.to_numpy(), A)
    n1, A = X.shape
    scree.scree_plot(A, r2, title)
    return r2

# run nipals pca for scree plot
def test_pca(df, A, title):
    binoms = std.binomial_set(df)
    X = std.normal_scale(df, binoms)
    Y = X[['match']].copy()
    X = X.drop(columns=["decision", "decision_o", "match"])

    t1, p1, r21 = pca.nipalspca(X.values, A)
    n1, A = X.shape
    scree.scree_plot(A, r21, title)
    return r21

# method to save results to csv
def save_results(t, file_name):
    df = pd.DataFrame(t, columns=[f'PC{i+1}' for i in range(t.shape[1])])
    df.to_csv(file_name, index=False)

# method to cross validate using kFold & find PC #
def cross_validate(X):
    groups = KFold(n_splits=10, shuffle=True, random_state=52)

    r2_components, q2_components = pca.kfold_pca(X, groups)

    for i, (r2, q2) in enumerate(zip(r2_components, q2_components), start=1):
        print(f"Components: {i}, R²: {r2:.4f}, Q²: {q2:.4f}")


def main():
    # extract files
    sdg_imputed = "../data/cleaned/speeddating_grouped_imputed.csv"
    sdg_5050 = "../data/cleaned/speeddating_grouped_imputed_balanced5050.csv"
    sdg_4060 = "../data/cleaned/speeddating_grouped_imputed_balanced4060.csv"

    df_gi = pd.read_csv(sdg_imputed)
    df_5050 =pd.read_csv(sdg_5050)
    df_4060 = pd.read_csv(sdg_4060)

    numComponents = 18

    # # scree plots to show pls's poor principal component variance coverage
    # r2pca = test_pca(df_gi, 60, " - NIPALS PCA")
    # r2pls = test_pls(df_gi, 60, " - NIPALS PLS")
    # scree.scree_plot2(60, r2pca, r2pls, "PCA", "PLS")

    # nipals pca for 18 components
    t1, p1, r21 = run_nipalspca(df_gi, numComponents, " - NIPALS PCA, 18 PCs")
    save_results(t1, "scores_A18.csv")

    # nipals pca for 18 compoents, undersampled (50% match=0)
    t50, p50, r250 = run_nipalspca(df_5050, numComponents, " - NIPALS PCA, 18 PCs, Undersampled (50% Match=0)")
    save_results(t50, "scores_A18_5050balance.csv")


if __name__=="__main__":
    main()