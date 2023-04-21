import math

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def median(data: pd.Series):
    """
    Berekent de mediaan van gegevens op ordinale, interval of ratioschaal.
    :param data: inputgegevens als Pandas Series
    :return: de mediaan
    """
    d = pd.Series(data.dropna())
    n = len(d)
    middle = math.floor(n / 2)
    return d.sort_values().reset_index(drop=True)[middle]


def uitschieters(data: pd.Series, mode='normaal', output='index'):
    """
    In mode 'normaal' en output 'index' wordt van elk element van de input aangegeven of het uitschieters is (True)
    of niet (False). Deze gegevens kunnen gebruikt worden om te indexeren. In mode 'extreem' wordt er gewerkt met extreme uitschieters.
    Met de output gelijk aan 'grenzen' kunnen de grenzen van de uitschieters (normaal of extreem) bepaald worden.
    :param data: inputgegevens als Pandas Series
    :param mode: 'normaal' of 'extreem'
    :param output: 'index' of 'grenzen'
    :return: Pandas Series of grenzen
    """

    # bereken eerst Q1, Q3 and IQR
    Q1, Q3 = data.quantile([0.25, 0.75])
    IQR = Q3 - Q1

    # kan ook in één regel met zgn. walrus operator (:=), maar dit is niet per se nodig
    # IQR = (Q3 := data.quantile(0.75)) - (Q1 := data.quantile(0.25))

    # bereken de grenzen voor de uitschieters
    grenzen = Q1 - 3 * IQR, Q1 - 1.5 * IQR, Q3 + 1.5 * IQR, Q3 + 3 * IQR

    if output == 'grenzen' and mode == 'normaal':
        return grenzen[1], grenzen[2]

    if output == 'grenzen' and mode == 'extreem':
        return grenzen[0], grenzen[3]

    if mode == 'extreem':
        return ~data.between(grenzen[0], grenzen[3])

    return ~data.between(grenzen[1], grenzen[2])


def signif(x, digits=6):
    """
    Zet een getal om naar een aantal significante cijfers
    :param x: het getal
    :param digits: het aantal significante cijfers
    :return: het getal met digits significante cijfers
    """
    if x == 0 or not math.isfinite(x):
        return x
    digits -= math.ceil(math.log10(abs(x)))
    return round(x, digits)


def mean_median_mode(series: pd.Series):
    """
     Geeft gemiddelde, modus en mediaan van een pandas.series object input is een rij van float  of int
    :param series:
    """
    print(f"Het gemiddelde is:{series.mean()}")
    print(f"De mediaan is:{series.median()}")
    print(f"De modus is:{series.value_counts().idxmax()}")


# %% general regression function
class GeneralRegression:
    def __init__(self, degree=1, exp=False, log=False):
        self.degree = degree
        self.exp = exp
        self.log = log
        self.model = None
        self.X = None
        self.y = None

    def fit(self, x: np.array, y: np.array):
        self.X = x.reshape(-1, 1)

        if self.exp:
            self.y = np.log(y)

        else:
            self.y = y

        if self.log:
            self.X = np.log(self.X)

        self.model = make_pipeline(PolynomialFeatures(degree=self.degree), LinearRegression())
        self.model.fit(self.X, self.y)

    def predict(self, x: np.array):
        X = x.reshape(-1, 1)

        if self.exp:
            return np.exp(self.model.predict(X))

        if self.log:
            return self.model.predict(np.log(X))

        return self.model.predict(X)

    @property
    def r2_score(self):
        return self.model.score(self.X, self.y)

    @property
    def coef(self):
        return self.model.steps[1][1].coef_

    @property
    def intercept(self):
        return self.model.steps[1][1].intercept_


def plot_dendrogram(dataset, hor=True):
    colors = list(matplotlib.colors.cnames.keys())
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    distances = linkage(dataset, method='single')
    ax.set_title("Dendrogram")
    # ax.set_xlabel('point')
    ax.set_ylabel('Euclidian distance')
    ax.grid(linestyle='--', axis='y')
    if (hor):
        orient = 'right'
    else:
        orient = 'top'
    dgram = dendrogram(distances, labels=dataset.index.values,
                       link_color_func=lambda x: colors[x % len(colors)],
                       leaf_font_size=15., ax=ax, orientation=orient)
    plt.show()
