import numpy as np
import pandas as pd

"""
 * @author Elias De Hondt
 * @see https://eliasdh.com
 * @version 1.2v
"""

# Absolute Frequenties (Klassen)
# Relatieve Frequenties
# Cumulatieve Frequenties
# Cumulatieve Percentages

def maakOrdinale(series: pd.Series, category: pd.Series) -> Series:
    # category = ["i3", "i5", "i7"] # VB data
    ordinale = pd.Categorical(series, ordered=True, categories=category)
    return ordinale


def absoluteFrequenties(series: pd.Series) -> Series:
    aFreq = series.value_counts(dropna=False) # Om de NaN toch mee te tellen (Standaard is dit True).
    return aFreq


def ascending(series: pd.Series)  -> Series:
    ascending = series.value_counts(ascending=True) # Sorteren omdraaien (Standaard is dit False).
    return ascending


def sorterenIndex(series: pd.Series) -> Series:
    sorterenIndex = series.value_counts().sort_index() # Sorteren op de inhoud van de index.
    return sorterenIndex


def klassen(series: pd.Series) -> Series:
    range = range(0, int(max(series) - min(series)), 100)
    klassen = pd.cut(series, bins=range)
    return klassen


def klassenAbsoluteFrequenties(series: pd.Series) -> Series: # (series = Ordinale)!
    range = range(0, int(max(series) - min(series)), 100)
    klassen = pd.cut(series, bins=range)
    aFreq = klassen.value_counts().sort_index()
    return aFreq


def intervallenSturges(series: pd.Series) -> Series:
    import math

    series = series.dropna()
    n = len(series)
    sturges = math.ceil(1 + math.log2(n))
    return sturges


def intervallenScott(series: pd.Series) -> Series:
    import statistics as stat

    b = 3.5 * stat.stdev(series) / (n ** (1 / 3))
    scott = math.ceil((max(series) - min(series)) / b)
    return scott


def intervallenExcel(series: pd.Series) -> Series:
    import math

    series = series.dropna()
    n = len(series)
    excel = math.ceil(math.sqrt(n))
    return excel


def intervallenAuto(series: pd.Series) -> Series:
    auto = series.value_counts(bins=11).sort_index()
    return auto


def relatieveFrequenties(series: pd.Series) -> Series:
    """
    Relatieve frequenties worden vaak gebruikt om de verdeling van gegevens in een dataset te analyseren en te communiceren.
    Ze kunnen bijvoorbeeld worden gebruikt om te bepalen welke waarden het vaakst voorkomen in een dataset, om te
    vergelijken hoe vaak verschillende waarden voorkomen, of om veranderingen in de verdeling van een dataset in de tijd te volgen.
    """
    # rFreq = (series.value_counts(normalize=True)*100).round(1) in %
    rFreq = series.value_counts(normalize=True)
    return rFreq


def cumulatieveFrequenties(series: pd.Series) -> Series: # (series = Ordinale)!
    """
    In een cumulatieve frequentieverdeling worden de frequenties opgeteld van de kleinste tot
    de grootste waarde in de dataset. Dit geeft een cumulatieve frequentie of cumulatieve teller voor elk datapunt.
    Deze cumulatieve frequenties kunnen vervolgens worden weergegeven in een grafiek, waarbij de cumulatieve
    frequenties worden uitgezet tegen de corresponderende datapunten.
    """
    cFreq = series.value_counts().sort_index().cumsum()
    return cFreq


def cumulatievePercentages(series: pd.Series) -> Series: # (series = Ordinale)!
    """
    Cumulatieve percentages kunnen worden gebruikt om te bepalen hoeveel procent van de waarnemingen in een
    dataset lager of hoger zijn dan een bepaalde waarde, of om te bepalen welk percentage van de dataset een bepaalde waarde of categorie omvat.
    """
    cPerc = (series.value_counts(normalize=True).sort_index().cumsum() * 100).round(1)
    return cPerc