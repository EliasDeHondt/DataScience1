import pandas as pd
import numpy as np

# Beslissingsbomen

"""
 * @author Elias De Hondt
 * @see https://eliasdh.com
 * @version 1.0v
"""

# Index:

# Entropy
# Information Gain


def entropy(series: pd.Series):
    """
    Entropie is een maat voor de wanorde of onvoorspelbaarheid van een systeem. 1 is maximale, 0 is minimale.
    Dus als de entropie hoger is dan 1 is er iets mis met de data.

    Voorbeeld:
    ----------
    >>> E_ouder = entropy(simpsons.geslacht)
    >>> print('Entropy, E(S) =', E_ouder)
    """
    vc = series.value_counts(normalize=True, sort=False)
    return -(vc * np.log2(vc)).sum()


def information_gain(parent_table: pd.DataFrame, attribute: str, target: str):
    """
    Informatiegroei is een maat die de vermindering van onzekerheid of willekeur in een dataset kwantificeert wanneer
    een specifieke eigenschap of kenmerk wordt gebruikt voor classificatie of besluitvorming.

    De maximale waarde van informatieaanwinst (information gain) is 1. Het geeft aan dat het gebruik van een bepaalde
    eigenschap of kenmerk de maximale vermindering van onzekerheid in de dataset heeft veroorzaakt, waardoor het meest
    informatief is voor classificatie of besluitvorming.

    Voorbeeld:
    ----------
    >>> IG_haarlengte = information_gain(simpsons, 'haarlengte', 'geslacht')
    >>> IG_leeftijd = information_gain(simpsons, 'leeftijd', 'geslacht')
    >>> IG_gewicht = information_gain(simpsons, 'gewicht', 'geslacht')
    >>> print(IG_haarlengte, IG_leeftijd, IG_gewicht)
    """
    # Bepaal entropie van parent table.
    entropy_parent = entropy(parent_table[target])
    child_entropies = []
    child_weights = []

    # Bereken entropies of child tables.
    for (label, fraction) in parent_table[attribute].value_counts().items():
        child_df = parent_table[parent_table[attribute] == label]
        child_entropies.append(entropy(child_df[target]))
        child_weights.append(int(fraction))

    # Calculate the difference between parent entropy and weighted child entropies.
    return entropy_parent - np.average(child_entropies, weights=child_weights)

def printAll(df: pd.DataFrame):
    print("Entropy ->", entropy(df['loon']))
    print("Information Gain ->", information_gain(df, 'loon', 'connecties'))


# Hier staat wat code om de functies te testen.
linkedin = pd.read_csv('data/testDataVoorHulpfuncties.csv', sep=',')
printAll(linkedin)