import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def plot_pca(pca: PCA):
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    _ = axes[0].set_xticks(range(1, pca.n_components_ + 1))
    _ = axes[0].set_xticklabels(['PC{}'.format(i) for i in range(1, pca.n_components_ + 1)])
    _ = axes[0].bar(x=range(1, pca.n_components_ + 1), height=pca.explained_variance_ratio_,
                    label='verklaarde variantie',
                    alpha=0.7,
                    edgecolor='black')
    _ = axes[0].grid(linestyle='--', axis='y')
    _ = axes[0].legend()

    _ = axes[1].set_xticks(range(1, pca.n_components_ + 1))
    _ = axes[1].set_xticklabels(['PC{}'.format(i) for i in range(1, pca.n_components_ + 1)])
    _ = axes[1].bar(x=range(1, pca.n_components_ + 1), height=pca.explained_variance_ratio_.cumsum(),
                    label='cumulatief verklaarde\nvariantie', alpha=0.7,
                    edgecolor='black')
    _ = axes[1].grid(linestyle='--', axis='y')
    _ = axes[1].axhline(0.9, color='red', linestyle='--')
    _ = axes[1].legend()


def pca_eigenvector(df: pd.DataFrame):
    X = df.dropna()
    model = make_pipeline(StandardScaler(), PCA())
    model.fit(X)
    pca: PCA = model['pca']
    header1 = X.columns
    header2 = ['PC{}'.format(i) for i in range(1, pca.n_components_ + 1)]

    return pd.DataFrame(pca.components_, columns=header1, index=header2)

def pca_biplot(df: pd.DataFrame):
    X = df.dropna()
    model = make_pipeline(StandardScaler(), PCA())
    X_PCA = model.fit_transform(X)  # kan ook in één stap met fit_transform()
    pd.DataFrame(np.hstack([X_PCA, y.to_numpy().reshape(-1, 1)]),
                 columns=['PC1', 'PC2', 'class'],
                 index=biopsy.dropna().index).head()
