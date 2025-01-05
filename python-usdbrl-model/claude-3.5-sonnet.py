from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score


def fetch_exchange_rate_data(start_date, end_date):
    """
    Fetch USD/BRL exchange rate data from Yahoo Finance
    """
    # Get USDBRL data
    usdbrl = yf.download("BRL=X", start=start_date, end=end_date)

    # Reset index to make date a column
    df = usdbrl.reset_index()

    # Keep only date and closing price
    df = df[["Date", "Close"]]

    # Rename columns
    df.columns = ["date", "rate"]

    return df


def exponential_func(x, a, b, c):
    """
    Exponential function of the form: a * exp(b * x) + c
    """
    return a * np.exp(b * x) + c


def fit_exponential_model(df):
    """
    Fit exponential model to the exchange rate data
    """
    # Create numerical x values (days since start)
    df["days"] = (df["date"] - df["date"].min()).dt.days

    # Fit exponential model
    popt, pcov = curve_fit(
        exponential_func,
        df["days"],
        df["rate"],
        p0=[1, 0.001, 0],  # Initial parameter guesses
        maxfev=10000,
    )

    # Generate predicted values
    df["predicted_rate"] = exponential_func(df["days"], *popt)

    # Calculate R-squared and RMSE
    r2 = r2_score(df["rate"], df["predicted_rate"])
    rmse = np.sqrt(mean_squared_error(df["rate"], df["predicted_rate"]))

    return df, popt, r2, rmse


def plot_results(df, title="USD/BRL Exchange Rate - Exponential Model"):
    """
    Plot actual vs predicted exchange rates
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["rate"], "b.", label="Actual Rate", alpha=0.5)
    plt.plot(df["date"], df["predicted_rate"], "r-", label="Exponential Model")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Exchange Rate (USD/BRL)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    # Set date range (5 years of data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5 * 365)

    # Fetch data
    print("Fetching exchange rate data...")
    df = fetch_exchange_rate_data(start_date, end_date)

    # Fit model
    print("Fitting exponential model...")
    df, params, r2, rmse = fit_exponential_model(df)

    # Print results
    print("\nModel Results:")
    print(f"R-squared: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("\nModel Parameters:")
    print(f"a: {params[0]:.4f}")
    print(f"b: {params[1]:.6f}")
    print(f"c: {params[2]:.4f}")

    # Plot results
    plot_results(df)


if __name__ == "__main__":
    main()
