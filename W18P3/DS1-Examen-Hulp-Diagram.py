import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
 * @author Elias De Hondt
 * @see https://eliasdh.com
 * @version 1.2v
"""

def lijnDiagram(figSizeX: int, figSizeY: int, xLabel: str, yLabel: str, label: str, title: str, x: pd.Series, y: pd.Series):
    """
    Maakt een lijndiagram van de gegeven x en y waarden met de gegeven labels en titel.

    Parameters:
    -----------
    figSizeX : int
        Breedte van de figuur in inches.
    figSizeY : int
        Hoogte van de figuur in inches.
    xLabel : str
        Label voor de x-as.
    yLabel : str
        Label voor de y-as.
    label : str
        Legenda label voor de plot.
    title : str
        Titel van de plot.
    x : pd.Series
        De x-waarden voor de plot als pandas series.
    y : pd.Series
        De y-waarden voor de plot als pandas series.

    Returns:
    --------
    None

    Voorbeeld:
    ----------
    >>> import pandas as pd
    >>> x = pd.Series([1, 2, 3, 4, 5])
    >>> y = pd.Series([10, 20, 30, 40, 50])
    >>> lijnDiagram(8, 6, "x-as", "y-as", "plot", "Titel", x, y)
    """
    fig, ax = plt.subplots(figsize=(figSizeX, figSizeY))
    ax.plot(x, y, label=label)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title(title)
    ax.grid(linestyle='--')
    _ = ax.legend()


def taartDiagram(s: pd.Series, title: str):
    """
    Maakt een taartdiagram van de gegeven pandas series met de gegeven titel.

    Parameters:
    -----------
    s : pd.Series
        Pandas Series met numerieke waarden.
    title : str
        Titel van de plot.

    Returns:
    --------
    None

    Voorbeeld:
    ----------
    >>> import pandas as pd
    >>> s = pd.Series([10, 20, 30, 40])
    >>> taartDiagram(s, "Titel")
    """
    x = s.values
    lab = s.keys()
    explode = np.full(len(lab), 0.025)
    fig, ax = plt.subplots()
    ax.pie(x, labels=lab, explode=explode, shadow=True)
    _ = ax.set_title(title)


def staafDiagram(s: pd.Series, title: str, xLabel: str, yLabel: str):
    """
    Maakt een staafdiagram van de gegeven pandas series met de gegeven titel en labels.

    Parameters:
    -----------
    s : pd.Series
        Pandas Series met numerieke waarden.
    title : str
        Titel van de plot.
    xLabel : str
        Label voor de x-as.
    yLabel : str
        Label voor de y-as.

    Returns:
    --------
    None

    Voorbeeld:
    ----------
    >>> import pandas as pd
    >>> s = pd.Series([10, 20, 30, 40])
    >>> staafDiagram(s, "Titel", "x-as", "y-as")
    """
    x = s.values
    lab = s.keys()
    _, ax = plt.subplots()
    ax.bar(lab, x)
    ax.set_title(title)
    ax.set_xlabel(xLabel)
    _ = ax.set_ylabel(yLabel)


def histogram(s: pd.Series, cutpoints: list, title: str, xLabel: str, yLabel: str):
    """
    Maakt een histogram van de gegeven pandas series met de gegeven titel, labels en cutpoints.

    Parameters:
    -----------
    s : pd.Series
        Pandas Series met numerieke waarden.
    cutpoints : list
        Lijst met grenswaarden om de bins van het histogram te bepalen.
    title : str
        Titel van de plot.
    xLabel : str
        Label voor de x-as.
    yLabel : str
        Label voor de y-as.

    Returns:
    --------
    None

    Voorbeeld:
    ----------
    >>> import pandas as pd
    >>> s = pd.Series([10, 20, 30, 40])
    >>> cutpoints = [0, 15, 25, 35, 50]
    >>> histogram(s, cutpoints, "Titel", "x-as", "y-as")
    """
    klassen = pd.cut(s, bins=cutpoints)
    lab = s.keys()
    x = klassen.values
    _, ax = plt.subplots()
    ax.bar(lab, x)
    ax.set_title(title)
    ax.set_xlabel(xLabel)
    _ = ax.set_ylabel(yLabel)