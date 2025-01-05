from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.optimize import curve_fit


def fetch_usdbrl_data(start_date, end_date):
    """Fetches USD/BRL exchange rate data from Alpha Vantage."""

    api_key = "XWVVW3WOLDZ9WNZD"  # Replace with your actual API key
    if api_key == "YOUR_ALPHA_VANTAGE_API_KEY":
        raise ValueError(
            "Please set up your Alpha Vantage API Key. Get one at https://www.alphavantage.co/support/#api-key"
        )

    url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=USD&to_symbol=BRL&outputsize=full&apikey={api_key}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        data = response.json()
        if "Time Series FX (Daily)" not in data:
            if "Error Message" in data:
                raise ValueError(f"Alpha Vantage API Error: {data['Error Message']}")
            else:
                raise ValueError("Unexpected data format from Alpha Vantage API")

        df = pd.DataFrame.from_dict(data["Time Series FX (Daily)"], orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        df = df["4. close"]
        df = df.loc[start_date:end_date]
        df = df.sort_index()
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except ValueError as e:
        print(e)
        return None


def exponential_model(x, a, b):
    """Defines the exponential model."""
    return a * np.exp(b * x)


def fit_exponential_model(data):
    """Fits the exponential model to the data."""
    x = np.arange(len(data))
    y = data.values

    try:
        popt, pcov = curve_fit(
            exponential_model, x, y, p0=[1, 0.01], maxfev=5000
        )  # Increased max function evaluations
        return popt, pcov
    except RuntimeError as e:
        print(f"Error during curve fitting: {e}")
        return None, None


# Example usage:
start_date = "2020-01-01"
end_date = "2024-01-01"

usdbrl_data = fetch_usdbrl_data(start_date, end_date)

if usdbrl_data is not None:
    popt, pcov = fit_exponential_model(usdbrl_data)

    if popt is not None:
        a, b = popt
        print(f"Fitted parameters: a = {a:.4f}, b = {b:.6f}")

        # Plot the data and the fitted model
        x = np.arange(len(usdbrl_data))
        plt.figure(figsize=(12, 6))
        plt.plot(usdbrl_data.index, usdbrl_data.values, label="Actual USD/BRL")
        plt.plot(
            usdbrl_data.index,
            exponential_model(x, a, b),
            label=f"Exponential Fit: y = {a:.2f} * exp({b:.4f}x)",
            color="red",
        )
        plt.xlabel("Date")
        plt.ylabel("USD/BRL Exchange Rate")
        plt.title("USD/BRL Exchange Rate and Exponential Fit")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Forecast (example: next 30 days)
        forecast_days = 30
        forecast_x = np.arange(len(usdbrl_data), len(usdbrl_data) + forecast_days)
        forecast = exponential_model(forecast_x, a, b)
        forecast_dates = pd.date_range(
            usdbrl_data.index[-1] + pd.Timedelta(days=1), periods=forecast_days
        )
        forecast_df = pd.DataFrame(forecast, index=forecast_dates, columns=["Forecast"])

        plt.figure(figsize=(12, 6))
        plt.plot(usdbrl_data.index, usdbrl_data.values, label="Actual USD/BRL")
        plt.plot(
            forecast_df.index,
            forecast_df["Forecast"],
            label="Forecast",
            linestyle="--",
            color="green",
        )
        plt.xlabel("Date")
        plt.ylabel("USD/BRL Exchange Rate")
        plt.title("USD/BRL Exchange Rate and Forecast")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Could not fit the exponential model.")
