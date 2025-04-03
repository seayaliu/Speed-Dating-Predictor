import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np



def scree_plot(A, r2):
    # df = pd.read_csv("../data/cleaned/speeddating_grouped_imputed.csv")
    # scaler = StandardScaler()
    # scaled_df=df.copy()
    # scaled_df=pd.DataFrame(scaler.fit_transform(scaled_df), columns=scaled_df.columns)
    # pca = PCA(n_components=18)
    # pca_fit = pca.fit(scaled_df)

    vals = range(1, len(r2) + 1)
    plt.plot(vals, r2 * 100, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot (Explained Variance in Y)')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance in Y (%)')
    plt.xticks(vals)
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.tight_layout()
    plt.show()


# def scree_plot2():
#     plt.plot(np.cumsum(pca.explained_variance_ratio_))
#     plt.title('Scree Plot')
#     plt.xlabel('Number of Components')
#     plt.ylabel('Cumulative Explained Variance')
#     plt.show()