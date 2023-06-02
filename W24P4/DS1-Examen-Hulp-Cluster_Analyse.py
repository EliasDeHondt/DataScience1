import pandas as pd
import numpy as np

# Cluster Analyse

"""
 * @author Elias De Hondt
 * @see https://eliasdh.com
 * @version 1.0v
"""


# Index:

# Euclidische Afstanden
# Seuclidean Afstanden
# Cityblock Afstanden
# K-Means

def euclidische_afstanden(series: pd.Series):
    """
    De euclidische afstand is de rechte lijnafstand tussen twee punten in een n-dimensionale ruimte,
    gemeten als de wortel van de som van de kwadraten van de verschillen tussen de coördinaten van de punten.
    (Of simpelweg de afstand tussen twee punten in een vlak (x,y) of ruimte (x,y,z) in 2D of 3D)

    Voorbeeld:
    ----------
    >>> euclidischeAfstanden = euclidische_afstanden(simpsons) # Output is een pandas DataFrame.
    >>> display(euclidischeAfstanden)
    """
    from scipy.spatial.distance import cdist

    euclidischeAfstanden = pd.DataFrame(cdist(series, series, metric='euclidean'),
                                        columns=series.index,
                                        index=series.index)

    # Voor tussen twee specifieke punten te vergelijken.
    # x = cdist(studentpunten.iloc[0], studentpunten.iloc[1], metric='euclidean')
    return euclidischeAfstanden


def seuclidean_afstanden(series: pd.Series):
    """
    De semi-euclidische afstand is een aangepaste afstandsmaat die de euclidische afstand
    aanpast door rekening te houden met de schaal van elke dimensie of functie.

    Voorbeeld:
    ----------
    >>> seuclideanAfstanden = seuclidean_afstanden(simpsons) # Output is een pandas DataFrame.
    >>> display(seuclideanAfstanden)
    """
    from scipy.spatial.distance import cdist

    seuclideanAfstanden = pd.DataFrame(cdist(series, series, metric='seuclidean'),
                                       columns=series.index,
                                       index=series.index)

    # Voor tussen twee specifieke punten te vergelijken.
    # x = cdist(studentpunten.iloc[0], studentpunten.iloc[1], metric='seuclidean')
    return seuclideanAfstanden


def cityblock_afstanden(series: pd.Series):
    """
    De cityblock afstand, ook wel bekend als de Manhattan afstand, is een afstandsmaat
    die de som van absolute verschillen tussen de coördinaten van twee punten in een n-dimensionale ruimte meet.

    Voorbeeld:
    ----------
    >>> cityblockAfstanden = cityblock_afstanden(simpsons) # Output is een pandas DataFrame.
    >>> display(cityblockAfstanden)
    """
    from scipy.spatial.distance import cdist

    cityblockAfstanden = pd.DataFrame(cdist(series, series, metric='cityblock'),
                                      columns=series.index,
                                      index=series.index)

    # Voor tussen twee specifieke punten te vergelijken.
    # x = cdist(studentpunten.iloc[0], studentpunten.iloc[1], metric='seuclidean')
    return cityblockAfstanden


def kMeans(df: pd.DataFrame):
    """
    Het K-means algoritme wordt gebruikt om een gegeven dataset te clusteren in k clusters.
    Hier wordt het algoritme toegepast op een pandas DataFrame df.

    Voorbeeld:
    ----------
    >>> model_simpsons_KMeans = kMeans(simpsons)
    >>> display(model_simpsons_KMeans)
    """
    # K-means algoritme
    from sklearn.cluster import KMeans
    # n_clusters = aantal clusters, n_init = aantal keer dat het algoritme wordt uitgevoerd, max_iter = aantal iteraties
    model = KMeans(n_clusters=2, n_init='auto', max_iter=100)
    model.fit(df)
    # De kolommen df.x en df.y zijn hier statis gemaakt.
    model_simpsons_KMeans = pd.DataFrame(zip(df.x, df.y, model.labels_), columns=["x", "y", "cluster"])

    return model_simpsons_KMeans


def Single_Linkage(df: pd.DataFrame):
    from scipy.cluster.hierarchy import linkage

    distances_single = linkage(df, method='single')  # Single linkage (kleinste afstand eerst)
    return distances_single


def Complete_Linkage(df: pd.DataFrame):
    from scipy.cluster.hierarchy import linkage

    distances_Complete = linkage(df, method='complete')  # Complete linkage (grootste afstand eerst)
    return distances_Complete


def kMeans(df: pd.DataFrame):
    """


    Voorbeeld:
    ----------
    >>>
    >>>
    """
    from matplotlib import pyplot as plt
    import matplotlib
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import cdist

    afstanden_euclidische = pd.DataFrame(cdist(df, df), columns=df.index, index=df.index)

    colors = list(matplotlib.colors.cnames.keys())[0:100:2]  # Kleuren voor de dendrogram

    fig, ax = plt.subplots(figsize=(12, 5))
    distances_single = linkage(df, method='single')  # Single linkage (kleinste afstand eerst)
    # distances_Complete = linkage(df, method='complete')  # Complete linkage (grootste afstand eerst)
    _ = ax.set_title('Dendrogram met single linkage')
    _ = ax.set_xlabel('punt')
    _ = ax.set_ylabel('Euclidische afstand')
    _ = ax.grid(linestyle='--', axis='y')

    dgram = dendrogram(distances_single, labels=afstanden_euclidische.index.values, link_color_func=lambda x: colors[x], ax=ax)

    return
