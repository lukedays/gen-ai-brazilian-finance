from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import curve_fit


def fetch_usdbrl_data(start_date="2020-01-01", end_date=None):
    """
    Fetches historical USDBRL exchange rate data from Yahoo Finance.

    Parameters:
        start_date (str): The start date for fetching data in 'YYYY-MM-DD' format.
        end_date (str): The end date for fetching data in 'YYYY-MM-DD' format.
                        If None, defaults to today's date.

    Returns:
        pandas.DataFrame: DataFrame containing the historical exchange rates.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    # Yahoo Finance ticker for USDBRL
    ticker = "BRL=X"

    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        raise ValueError(
            "No data fetched. Please check the ticker symbol and date range."
        )

    # Use the 'Close' price for analysis
    usdbrl = data["Close"].dropna()
    print(f"Fetched {len(usdbrl)} data points.")
    return usdbrl


def exponential_model(x, a, b):
    """
    Exponential model y = a * exp(b * x)

    Parameters:
        x (float or array-like): Independent variable.
        a (float): Parameter a.
        b (float): Parameter b.

    Returns:
        float or array-like: Computed exponential values.
    """
    return a * np.exp(b * x)


def fit_exponential_model(dates, rates):
    """
    Fits an exponential model to the provided data.

    Parameters:
        dates (array-like): Numerical representation of dates (e.g., days since start).
        rates (array-like): Exchange rates corresponding to the dates.

    Returns:
        popt (tuple): Optimized parameters (a, b) for the exponential model.
        pcov (2D array): Covariance of popt.
    """
    # Initial guess for a and b
    initial_guess = (rates.iloc[0], 0.001)

    print("Fitting exponential model to data...")
    popt, pcov = curve_fit(
        exponential_model, dates, rates, p0=initial_guess, maxfev=10000
    )
    print(f"Fitted parameters: a = {popt[0]:.4f}, b = {popt[1]:.6f}")
    return popt, pcov


def plot_data_and_model(dates, rates, popt):
    """
    Plots the historical exchange rates and the fitted exponential model.

    Parameters:
        dates (array-like): Original date values.
        rates (array-like): Historical exchange rates.
        popt (tuple): Parameters of the fitted exponential model.
    """
    plt.figure(figsize=(12, 6))
    plt.scatter(dates, rates, label="Historical USDBRL Rates", color="blue", s=10)

    # Generate values for the fitted model
    dates_fit = np.linspace(dates.min(), dates.max(), 1000)
    rates_fit = exponential_model(dates_fit, *popt)
    plt.plot(dates_fit, rates_fit, label="Fitted Exponential Model", color="red")

    plt.title("USDBRL Exchange Rate and Exponential Fit")
    plt.xlabel("Days Since Start Date")
    plt.ylabel("Exchange Rate (BRL per USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # Parameters
    START_DATE = "2020-01-01"  # You can change the start date as needed
    END_DATE = None  # None will set it to today's date

    # Step 1: Fetch data
    usdbrl_series = fetch_usdbrl_data(start_date=START_DATE, end_date=END_DATE)

    # Step 2: Prepare data for modeling
    # Convert dates to numerical values (days since start)
    dates = (usdbrl_series.index - usdbrl_series.index[0]).days
    rates = usdbrl_series.values

    # Step 3: Fit the exponential model
    popt, pcov = fit_exponential_model(dates, rates)

    # Step 4: Plot the results
    plot_data_and_model(dates, rates, popt)


if __name__ == "__main__":
    main()
