import pandas as pd
import numpy as np


"""
 * @author Elias De Hondt
 * @see https://eliasdh.com
 * @version 1.0v
"""

# Correlatiecoefficient
# Rangcorrelatiecoefficient (Kendall)
# Rangcorrelatiecoefficient (Spearman)
# Lineare regressie
# Lineare regressie (Coef)
# Lineare regressie (Intercept)
# Standaardschattingsfout
# Kwadraat van correlatiecoëfficiënt

# Niet-lineaire regressie (kwadratisch)
# Niet-lineaire regressie (kubisch)
# Niet-lineaire regressie (logaritmisch)
# Niet-lineaire regressie (exponentieel)


def correlatiecoefficient(df: pd.DataFrame) -> float:
    from scipy.stats import pearsonr

    corr = pearsonr(df.iloc[:, 0], df.iloc[:, 1])
    return float(corr)


def rangcorrelatiecoefficientKendall(df: pd.DataFrame) -> float:
    from IPython import InteractiveShell

    InteractiveShell.ast_node_interactivity = 'all'
    corr = df.corr(method='kendall').iloc[1:, 0]
    return float(corr)


def rangcorrelatiecoefficientSpearman(df: pd.DataFrame) -> float:
    from IPython import InteractiveShell

    InteractiveShell.ast_node_interactivity = 'all'
    corr = df.corr(method='spearman').iloc[1:, 0]
    return float(corr)


def lineareRegressie(df: pd.DataFrame):
    from sklearn import linear_model

    X = df.iloc[:, 0].to_numpy().reshape(-1, 1)
    y = df.iloc[:, 1].to_numpy()
    regr = linear_model.LinearRegression()
    model = regr.fit(X, y)
    return model


def coef(df: pd.DataFrame) -> float:
    from sklearn import linear_model

    X = df.iloc[:, 0].to_numpy().reshape(-1, 1)
    y = df.iloc[:, 1].to_numpy()
    regr = linear_model.LinearRegression()
    model = regr.fit(X, y)
    coef = model.coef_
    return float(coef)


def intercept(df: pd.DataFrame) -> float:
    from sklearn import linear_model

    X = df.iloc[:, 0].to_numpy().reshape(-1, 1)
    y = df.iloc[:, 1].to_numpy()
    regr = linear_model.LinearRegression()
    model = regr.fit(X, y)
    intercept = model.intercept_
    return float(intercept)


def standaardschattingsfout(df: pd.DataFrame) -> float:
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    # X = linkedin[['connecties']]
    # y = linkedin.loon
    X = df.iloc[:, 0] # Onafhankelijke (x)
    y = df.iloc[:, 1] # Afhankelijke variabele (y)
    model.fit(X, y)
    y_hat = model.predict(X) # voorspellingen maken
    standaardschattingsfout = mean_squared_error(y, y_hat, squared=False)
    return float(standaardschattingsfout)


def kwadraatVanCorrelatiecoefficient(df: pd.DataFrame) -> float: # Verklaarde variantie en R²
    from sklearn.metrics import r2_score
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    # X = linkedin[['connecties']]
    # y = linkedin.loon
    X = df.iloc[:, 0] # Onafhankelijke (x)
    y = df.iloc[:, 1] # Afhankelijke variabele (y)
    model.fit(X, y)
    y_hat = model.predict(X) # voorspellingen maken

    kwadraatVanCorrelatiecoefficient = r2_score(y, y_hat)  # kan je altijd gebruiken
    return float(kwadraatVanCorrelatiecoefficient)

#=====================================================================================================#
#====================================== Niet-lineaire regressie ======================================#
#=====================================================================================================#

def kwadratisch(df: pd.DataFrame) -> float:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    X = df.iloc[:, 0] # Onafhankelijke (x)
    y = df.iloc[:, 1] # Afhankelijke variabele (y)
    model.fit(X, y)
    kwadratisch = model.score(X, y)

    return float(kwadratisch)


def kubisch(df: pd.DataFrame) -> float:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    model = make_pipeline(PolynomialFeatures(3), LinearRegression())
    X = df.iloc[:, 0] # Onafhankelijke (x)
    y = df.iloc[:, 1] # Afhankelijke variabele (y)
    model.fit(X, y)
    kubisch = model.score(X, y)

    return float(kubisch)


def logaritmisch(df: pd.DataFrame) -> float:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    model = LinearRegression()
    X = df.iloc[:, 0] # Onafhankelijke (x)
    y = df.iloc[:, 1] # Afhankelijke variabele (y)
    # X moet logaritmisch gemaakt worden
    model.fit(np.log(X), y)

    # voorspelling doen op logaritmisch gemaakte X
    y_hat = model.predict(np.log(X))
    logaritmisch = r2_score(y, y_hat)

    return float(logaritmisch)


def exponentieel(df: pd.DataFrame) -> float:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    model = LinearRegression()
    X = df.iloc[:, 0] # Onafhankelijke (x)
    y = df.iloc[:, 1] # Afhankelijke variabele (y)
    # y moet logaritmisch gemaakt worden
    model.fit(X, np.log(y))

    # voorspelling nog exponentieel maken
    y_hat = np.exp(model.predict(X))
    exponentieel = r2_score(y, y_hat)

    return float(exponentieel)