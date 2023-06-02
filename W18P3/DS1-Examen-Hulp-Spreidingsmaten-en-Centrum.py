import numpy as np
import pandas as pd

"""
 * @author Elias De Hondt
 * @see https://eliasdh.com
 * @version 1.2v
"""

# Median
# Median Categorical
# Mode
# Gemiddelde Rekenkundig
# Gemiddelde Meetkundig
# gemiddelde Gewogen
# Gemiddelde Harmonisch
# Bereik
# Q1, Q2, Q3, IQR
# Outliers
# Extreme Outliers
# Gemiddelde Absolute Afwijking
# Variantie
# Standaardafwijking
# Extreme Outliers
# Z scores

def median(series: pd.Series) -> float:
    """
    Bereken de mediaan van een gegeven pandas Series-object.

    Parameters:
    -----------
    series : pd.Series
        Een pandas Series-object waarvan de mediaan wordt berekend.

    Returns:
    --------
    float
        De mediaan van de gegeven pandas Series-object.

    Raises:
    -------
    TypeError
        Als de gegeven parameter geen pandas Series-object is.
    """
    median = series.median()
    # print("Median ->", round(median(series),2))
    return float(median)


def median_categorical(series: pd.Series):
    """
    Bereken de mediaan van een gegeven pandas Series-object.

    Parameters:
    -----------
    series : pd.Series
        A pandas Series object containing categorical data.

    Returns:
    -----------
    medianCategorical
        The median value of the non-null categorical data in the Series.
    """
    import math

    d = series.dropna()
    n = len(d)
    middle = math.floor(n / 2)
    medianCategorical = d.sort_values().reset_index(drop=True)[middle]
    return medianCategorical


def modeN(series: pd.Series) -> float:
    """
    Bereken de modus van een gegeven pandas Series-object.

    Parameters:
    -----------
    series : pd.Series
        Een pandas Series-object waarvan de modus wordt berekend.

    Returns:
    --------
    float
        De modus van de gegeven pandas Series-object.

    Raises:
    -------
    TypeError
        Als de gegeven parameter geen pandas Series-object is.
    """
    mode = series.mode()
    # print("ModeN ->", round(modeN(series),2))
    return mode


def modeK(series: pd.Series) -> float:
    """
    Bereken de kleinste modus van een gegeven pandas Series-object.

    Parameters:
    -----------
    series : pd.Series
        Een pandas Series-object waarvan de kleinste modus wordt berekend.

    Returns:
    --------
    float
        De kleinste modus van de gegeven pandas Series-object.

    Raises:
    -------
    TypeError
        Als de gegeven parameter geen pandas Series-object is.
    """
    mode = series.value_counts().sort_index().head(1)
    # print("ModeK ->", round(modeK(series),2))
    return mode


def gemiddeldeRekenkundig(series: pd.Series) -> float:
    """
    Bereken het rekenkundig gemiddelde van een gegeven pandas Series-object.

    Parameters:
    -----------
    series : pd.Series
        Een pandas Series-object waarvan het rekenkundig gemiddelde wordt berekend.

    Returns:
    --------
    float
        Het rekenkundig gemiddelde van de gegeven pandas Series-object.

    Raises:
    -------
    TypeError
        Als de gegeven parameter geen pandas Series-object is.
    """
    gemiddelde = series.mean()
    # print("Gemiddelde Rekenkundig ->", round(gemiddeldeRekenkundig(series),2))
    return float(gemiddelde)


def gemiddeldeMeetkundig(series: pd.Series) -> float:
    """
    Bereken het meetkundig gemiddelde van een gegeven pandas Series-object.

    Parameters:
    -----------
    series : pd.Series
        Een pandas Series-object waarvan het meetkundig gemiddelde wordt berekend.

    Returns:
    --------
    float
        Het meetkundig gemiddelde van de gegeven pandas Series-object.

    Raises:
    -------
    TypeError
        Als de gegeven parameter geen pandas Series-object is.
    """
    x = series / 100 + 1
    gemiddelde = (np.exp(np.mean(np.log(x))) - 1) * 100
    # print("Gemiddelde Meetkundig ->", round(gemiddeldeMeetkundig(series),2))
    return float(gemiddelde)


def gemiddeldeGewogen(series: pd.Series, weights: pd.Series) -> float:
    """
    Bereken het gewogen gemiddelde van de gegeven reeks en gewichten.

    Parameters:
    -----------
    series : pandas.Series
        Een pandas Series object met numerieke waarden waarvan het gewogen gemiddelde moet worden berekend.
    weights : pandas.Series
        Een pandas Series object met numerieke waarden die de gewichten zijn die worden toegepast op de overeenkomstige elementen in de series.

    Returns:
    --------
    float
        Het gewogen gemiddelde van de gegeven reeks en gewichten.

    Raises:
    -------
    TypeError
        Als series of weights niet van het type pandas.Series zijn.
    ValueError
        Als series of weights niet dezelfde lengte hebben.
    """
    gemiddelde = sum(series * weights) / sum(weights)
    # print("Gemiddelde Gewogen ->", round(gemiddeldeGewogen(series, weights),2))
    return float(gemiddelde)


def gemiddeldeHarmonisch(series: pd.Series) -> float:
    """
    Bereken het harmonisch gemiddelde van een gegeven serie.

    Parameters:
    -----------
    series : pandas.Series
        Een serie met numerieke waarden.

    Returns:
    --------
    float
        Het berekende harmonisch gemiddelde van de serie.

    Raises:
    -------
    ZeroDivisionError
        Als de serie minstens één nulwaarde bevat.
    """
    gemiddelde = 1 / np.mean(1 / series)
    # print("Gemiddelde Harmonisch ->", round(gemiddeldeHarmonisch(series),2))
    return float(gemiddelde)


def bereik(series: pd.Series) -> float:
    """
    Bereken het bereik van een gegeven Pandas Series.

    Het bereik wordt gedefinieerd als het verschil tussen de maximale
    en minimale waarde van de serie.

    Args:
        series (pd.Series): De Pandas Series waarvan het bereik moet worden berekend.

    Returns:
        float: Het berekende bereik van de serie.

    """
    bereik = max(series) - min(series)
    # print("Bereik ->", round(bereik(series),2))
    return float(bereik)


def Q1(series: pd.Series) -> float:
    """
    Bereken het eerste kwartiel van een gegeven Pandas Series.

    Het eerste kwartiel, Q1, is het getal dat op de 25e percentiel
    van de gegeven serie ligt.

    Args:
        series (pd.Series): De Pandas Series waarvan het eerste kwartiel moet worden berekend.

    Returns:
        float: Het berekende eerste kwartiel van de serie.

    """
    Q1 = series.quantile(0.25)
    # print("Q1 ->", round(Q1(series),2))
    return float(Q1)


def Q2(series: pd.Series) -> float:
    """
    Bereken het 2e kwartiel (Q2) van een gegeven Pandas Series-object.

    Parameters:
    -----------
    series : pd.Series
        Een pandas Series-object dat numerieke waarden bevat.

    Returns:
    --------
    float
        Het 2e kwartiel (Q2) van de gegeven Series als een floating point getal.

    Raises:
    -------
    TypeError
        Als het gegeven object geen Pandas Series-object is.

    """
    Q2 = series.quantile(0.50)
    # print("Q2 ->", round(Q2(series),2))
    return float(Q2)


def Q3(series: pd.Series) -> float:
    """
    Bereken het 3e kwartiel (Q3) van een gegeven Pandas Series-object.

    Parameters:
    -----------
    series : pd.Series
        Een pandas Series-object dat numerieke waarden bevat.

    Returns:
    --------
    float
        Het 3e kwartiel (Q3) van de gegeven Series als een floating point getal.

    Raises:
    -------
    TypeError
        Als het gegeven object geen Pandas Series-object is.

    """
    Q3 = series.quantile(0.75)
    # print("Q3 ->", round(Q3(series),2))
    return float(Q3)


def IQR(series: pd.Series) -> float:
    """
    Bereken de interkwartielafstand (IQR) voor een gegeven pandas series.

    Parameters:
    -----------
    series : pd.Series
        De pandas series waarvoor de IQR berekend moet worden.

    Returns:
    --------
    float
        De berekende IQR waarde.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    # print("IQR ->", round(IQR(series),2))
    return float(IQR)


def outliers(series: pd.Series) -> float:
    """
    Deze functie berekent de outliers van een gegeven pandas Series object.
    Het berekent de Q1 en Q3 quartiles en IQR interquartile range van de gegeven series.
    Outliers worden gedefinieerd als punten die onder Q1 - 1.5*IQR of boven Q3 + 1.5*IQR vallen.

    Parameters:
    -----------
    series : pd.Series
        Een pandas Series object waarvan de outliers moeten worden bepaald.

    Returns:
    --------
    pd.Series
        Een pandas Series object met de outliers van de gegeven series.
    """
    outliers = series[(series < (Q1(series) - 1.5 * IQR(series))) | (series > (Q3(series) + 1.5 * IQR(series)))]
    # print("Uutliers ->", outliers(series))
    return outliers.values


def extreme_outliers(series: pd.Series) -> float:
    """
    Deze functie berekent de extreme outliers van een gegeven pandas Series object.
    Het berekent de Q1 en Q3 quartiles en IQR interquartile range van de gegeven series.
    Extreme outliers worden gedefinieerd als punten die onder Q1 - 3*IQR of boven Q3 + 3*IQR vallen.

    Parameters:
    -----------
    series : pd.Series
        Een pandas Series object waarvan de extreme outliers moeten worden bepaald.

    Returns:
    --------
    pd.Series
        Een pandas Series object met de extreme outliers van de gegeven series.
    """
    extreme_outliers = series[(series < (Q1(series) - 3 * IQR(series))) | (series > (Q3(series) + 3 * IQR(series)))]
    # print("Extreme_outliers ->", extreme_outliers(series))
    return extreme_outliers.values


def gemiddeldeAbsoluteAfwijking(series: pd.Series) -> float:
    """
    Deze functie berekent de gemiddelde absolute afwijking van een gegeven pandas Series object.
    De gemiddelde absolute afwijking wordt berekend als de gemiddelde afstand van elk punt in de serie tot de gemiddelde waarde van de serie.
    De afwijking van een punt wordt als significant beschouwd als het meer dan één keer de gemiddelde absolute afwijking van de serie afwijkt van de gemiddelde waarde van de serie.

    Parameters:
    -----------
    series : pd.Series
        Een pandas Series object waarvan de gemiddelde absolute afwijking moet worden berekend.

    Returns:
    --------
    pd.Series
        Een pandas Series object met de punten die significant afwijken van de gemiddelde waarde van de gegeven series.
    """
    mean = series.mean()
    mad = mad = (series - series.mean()).abs().mean()
    gemiddeldeAbsoluteAfwijking = series[(series < (mean - mad)) | (series  > (mean + mad))]
    # print("Gemiddelde Absolute Afwijking ->", gemiddeldeAbsoluteAfwijking(series))
    return gemiddeldeAbsoluteAfwijking.values


def variantie(series: pd.Series) -> float:
    """
    Bereken de variantie voor een gegeven pandas series.

    Parameters:
    -----------
    series : pd.Series
        De pandas series waarvoor de variantie berekend moet worden.

    Returns:
    --------
    float
        De berekende variantie waarde.
    """
    variantie = series.var()
    # print("Variantie ->", round(variantie(series),2))
    return float(variantie)


def standaardafwijking(series: pd.Series) -> float:
    """
    Bereken de standaardafwijking voor een gegeven pandas series.

    Parameters:
    -----------
    series : pd.Series
        De pandas series waarvoor de standaardafwijking berekend moet worden.

    Returns:
    --------
    float
        De berekende standaardafwijking waarde.
    """
    standaardafwijking = series.std()
    # print("Standaardafwijking ->", round(standaardafwijking(series),2))
    return float(standaardafwijking)


def z_scores(series: pd.Series):
    """
    Bereken de Z-score van elke waarde in een pandas Series
    """
    # from scipy import stats
    # zscores = stats.zscore(laptops_small.diskspace)

    import scipy.stats as stats

    mean = series.mean()
    std = series.std()
    z_scores = (series - mean) / std
    # print("Z scores ->\n", z_scores(series))
    return z_scores


def printAll(series: pd.Series, weights: pd.Series):
    """
    Print de resultaten van verschillende statistische berekeningen van een pandas Series.

    Parameters:
        series (pd.Series): De pandas Series waarop de statistische berekeningen worden uitgevoerd.
        weights (pd.Series): De gewichten die worden gebruikt bij het berekenen van het gewogen gemiddelde.

    Returns:
        None

    Prints:
        - Median: Het middelste getal in de Series.
        - Median_categorical: Het middelste categorical in de Series. (Niet in lijst!)
        - ModeN: Het meest voorkomende getal in de Series.
        - ModeK: Het getal met het hoogste aantal voorkomens in de Series.
        - Gemiddelde Rekenkundig: Het rekenkundig gemiddelde van de Series.
        - Gemiddelde Meetkundig: Het meetkundig gemiddelde van de Series.
        - Gemiddelde Gewogen: Het gewogen gemiddelde van de Series.
        - Gemiddelde Harmonisch: Het harmonisch gemiddelde van de Series.
        - Bereik: Het verschil tussen het grootste en het kleinste getal in de Series.
        - Q1: Het eerste kwartiel van de Series.
        - Q2: Het tweede kwartiel (of de mediaan) van de Series.
        - Q3: Het derde kwartiel van de Series.
        - IQR: Het interkwartielbereik van de Series.
        - Uutliers: De outliers van de Series (getallen die meer dan 1,5 keer de IQR afwijken van het eerste of derde kwartiel).
        - Extreme_outliers: De extreme outliers van de Series (getallen die meer dan 3 keer de IQR afwijken van het eerste of derde kwartiel).
        - Gemiddelde Absolute Afwijking: De gemiddelde absolute afwijking van de Series.
        - Variantie: De variantie van de Series.
        - Standaardafwijking: De standaardafwijking van de Series.
        - z_scores: De z_scores van de Series.
    """
    print("Median ->", round(median(series),2))
    print("ModeN ->", round(modeN(series),2))
    print("ModeK ->", round(modeK(series),2))
    print("Gemiddelde Rekenkundig ->", round(gemiddeldeRekenkundig(series),2))
    print("Gemiddelde Meetkundig ->", round(gemiddeldeMeetkundig(series),2))
    print("Gemiddelde Gewogen ->", round(gemiddeldeGewogen(series, weights),2))
    print("Gemiddelde Harmonisch ->", round(gemiddeldeHarmonisch(series),2))
    print("Bereik ->", round(bereik(series),2))
    print("Q1 ->", round(Q1(series),2))
    print("Q2 ->", round(Q2(series),2))
    print("Q3 ->", round(Q3(series),2))
    print("IQR ->", round(IQR(series),2))
    print("Uutliers ->", outliers(series))
    print("Extreme_outliers ->", extreme_outliers(series))
    print("Gemiddelde Absolute Afwijking ->", gemiddeldeAbsoluteAfwijking(series))
    print("Variantie ->", round(variantie(series),2))
    print("Standaardafwijking ->", round(standaardafwijking(series),2))
    print("Z scores ->\n", z_scores(series))


# TEST (Kan verwijderd worden indien je de functies wilt aanspreken in een ander bestand)
weights = pd.Series([6, 11, 5, 3, 3, 7, 4, 7, 7, 3, 4])
series = pd.Series([18, 15, 12, 10, 18, 13, 17, 15, 13, 12, 12])
printAll(series, weights)

my_series = series
# Controleer het meetniveau van de serie
if my_series.dtype == "int64" or my_series.dtype == "float64":
    print("De serie heeft een ratio meetniveau")
elif my_series.dtype == "object":
    if my_series.str.isnumeric().all():
        print("De serie heeft een ratio meetniveau")
    else:
        print("De serie heeft een nominaal of ordinaal meetniveau")
else:
    print("Het meetniveau van de serie kan niet worden bepaald")