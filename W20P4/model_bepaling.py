# voer de decompositieanalyse uit op de tijdreeksgegevens
sd_model = seasonal_decompose(pretpark['aantal_bezoekers'], model='multiplicative', period=12)

# verkrijg de standaarddeviaties van de seizoens- en restcomponenten
seasonal_std = np.std(sd_model.seasonal)
residual_std = np.std(sd_model.resid)

# bereken de verhouding van de standaarddeviaties
std_ratio = seasonal_std / residual_std

# bepaal welk model geschikter is op basis van de verhouding van de standaarddeviaties
if std_ratio < 1:
    print("Het additieve model is geschikter.")
else:
    print("Het multiplicatieve model is geschikter.")