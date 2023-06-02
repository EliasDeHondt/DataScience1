import pandas as pd
import numpy as np

# Forecasting

"""
 * @author Elias De Hondt
 * @see https://eliasdh.com
 * @version 1.0v
"""

# Index:

# Bereken Gewichten
# Voorspelling (Naive)
# Voorspelling (Average)
# Voorspelling (Moving Average)
# Voorspelling (Linear Combination)
# Predictor
# Predict
# Find Period
# Seizoen Component
# Plot Trends

# Forecast Errors (NAE)
# Forecast Errors (RMSE)
# Forecast Errors (MAPE)
# Forecast Errors (Forecast Errors)

def bereken_gewichten(y: np.array, m: int):
    n = y.size
    if n < 2 * m:
        return np.nan
    M = y[-(m + 1):-1]
    for i in range(1, m):
        M = np.vstack([M, y[-(m + i + 1):-(i + 1)]])

    v = np.flip(y[-m:])
    return np.linalg.solve(M, v)


def naive(y: np.array):
    if y.size > 0:
        return y[-1]
    return np.nan


def average(y: np.array):
    if y.size < 1:
        return np.nan

    return y.mean()


def moving_average(y: np.array, m=4):
    if y.size < m:
        return np.nan

    return np.mean(y[-m:])


def linear_combination(y: np.array, m=4) -> np.ndarray:
    n = y.size
    # check op minstens 2*m gegevens
    if n < 2 * m:
        return np.nan
    # bereken de gewichten
    a = bereken_gewichten(y, m)
    # bereken de voorspelde waarde en geef de voorspelde waarde terug
    return np.sum(y[-m:] * a)


def predictor(y: np.array, f, *argv):
    i = 0
    while True:
        if i <= y.size:
            yield f(y[:i], *argv)
        else:
            y = np.append(y, f(y, *argv))
            yield f(y, *argv)
        i += 1


def predict(y: np.array, start, end, f, *argv):
    generator = predictor(y, f, *argv)
    predictions = np.array([next(generator) for _ in range(end)])
    predictions[:start] = np.nan
    return predictions


def find_period(series: np.array, maxlags=10, top_n=1) -> int:
    from statsmodels.tsa.stattools import acf

    acf_vals = acf(series, nlags=maxlags)
    period = (np.argsort(-1 * acf_vals)[1:top_n + 1])
    return period


def seizoen_component(series: np.array) -> np.array:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    lags, acfs, _, _ = ax.acorr(series, maxlags=15)
    autocorrelatie = pd.DataFrame({'lags': lags, 'acf': acfs}).sort_values(by='acf', ascending=False)
    return autocorrelatie


def plot_trends(y1: np.array, y2=None, sub_title=None, label1='gegeven', label2='voorspeld', color='C0', ax=None):
    import matplotlib.pyplot as plt
    if y2 is not None:
        n = max(y1.size, y2.size)
    else:
        n = y1.size

    if not ax:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    if sub_title:
        fig.suptitle(sub_title, y=1.02)

    ax.set_title('Opbrengsten voorbije 5 jaar')
    ax.set_xlabel('kwartaal')
    ax.set_ylabel('opbrengst (€)')
    ax2 = ax.secondary_xaxis('top')
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(['Q{}'.format(j % 4 + 1) for j in range(n)])

    ax.set_xticks(range(n))
    ax.plot(y1, label=label1, color=color, marker='o')
    if y2 is not None:
        ax.plot(y2, label=label2, color='C1', marker='^')
    for i in range(0, n, 4):
        ax.axvline(i, color='gray', linewidth=0.5)

    ax.legend()


# ============================================================================= #
# ============================== Forecast Errors ============================== #
# ==============================================================================#

def forecast_error_mae(x: np.array, f: np.array):
    e = x - f
    mae = np.nanmean(np.abs(e))
    return mae


def forecast_error_rmse(x: np.array, f: np.array):
    e = x - f
    rmse = np.sqrt(np.nanmean(e ** 2))
    return rmse


def forecast_error_mape(x: np.array, f: np.array):
    e = x - f
    mape = np.nanmean(np.abs(e / x))
    return mape


def forecast_errors(x: np.array, f: np.array, method: str):
    mae = forecast_error_mae(x, f)
    rmse = forecast_error_mae(x, f)
    mape = forecast_error_mae(x, f)
    errors = pd.DataFrame({'MAE': [mae],
                           'RMSE': [rmse],
                           'MAPE': [mape]}, index=[method])
    return errors

"""
x = opbrengsten
f1 = predict(opbrengsten, 0, 20, naive)
f2 = predict(opbrengsten, 0, 20, average)
f3 = predict(opbrengsten, 0, 20, moving_average)
f4 = predict(opbrengsten, 0, 20, linear_combination)

pd.concat([forecast_errors(x, f1, 'naive'),
           forecast_errors(x, f2, 'average'),
           forecast_errors(x, f3, 'moving average'),
           forecast_errors(x, f4, 'linear combination')
           ])
"""

def printAll(df: pd.DataFrame):
    print("Voorspelling (Naive) ->", naive(df['loon']))
    print("Voorspelling (Average) ->", average(df['loon']))
    print("Voorspelling (Moving Average) ->", moving_average(df['loon']), m=4)
    print("Voorspelling (Linear Combination) ->", linear_combination(df['loon']))
    print("Find Period ->", find_period(df['loon']))
    # print("seizoen Component ->", seizoen_component(df['loon']))


# Hier staat wat code om de functies te testen.
linkedin = pd.read_csv('data/testDataVoorHulpfuncties.csv', sep=',')
printAll(linkedin)
