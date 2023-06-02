import pandas as pd
import numpy as np

# Principale-componenten analyse (PCA)

"""
 * @author Elias De Hondt
 * @see https://eliasdh.com
 * @version 1.0v
"""

# Index:

# Extra informatie.


# Voor dit onderdeel van de cursus zijn er niet echt goede mogelijkheden voor functies te
# schrijven wegens de gelimiteerd cursusmateriaal.


# Extra informatie.
"""
    Principale Componenten Analyse (PCA) is een statistische methode die wordt gebruikt om de belangrijkste 
    onderliggende patronen of variabelen in een dataset te identificeren en deze te vereenvoudigen tot een 
    kleiner aantal lineair onafhankelijke componenten.
    
    -> [1]
    df.index = df['kolom1'] # Zet kolom1 zijn data (str) als index krijgen.
    
    -> [2]
    df = df.drop(columns=['kolom1', 'kolom2', 'kolom3']) # Verwijder van nuteloze kolommen.
    
    -> [3]
    dfCorr = df.corr() # Correlatie matrix maken van de dataframe.
    
    -> [4]
    model = make_pipeline(StandardScaler(), PCA()) # Maak een model aan met de StandardScaler en PCA. (pipeline)
    model.fit(X) # Fit het model met de data.
    
    
    from pca import pca
    model = pca(normalize=True)
    out = model.fit_transform(goblet, verbose=False)
    display(out['loadings'])
    display(out['topfeat'])
    display(out['explained_var'])
    
    Het genereren van een plot hiervan:
    # Plot en biplot maken is alsvolgt:
    model.plot(figsize=(10,8)) # Is een plot van de data in2D
    model.biplot(figsize=(10,8)) # Is een biplot van de data in2D
    # model.biplot(n_feat=5) # 5 is het aantal features dat je wil zien
"""