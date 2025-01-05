from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.optimize import curve_fit


# Step 1: Fetch USDBRL exchange rate data from an API
def fetch_usdbrl_data():
    # Example API: Alpha Vantage (replace with your API key)
    api_key = "XWVVW3WOLDZ9WNZD"
    url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=USD&to_symbol=BRL&apikey={api_key}"
    response = requests.get(url)
    data = response.json()

    # Extract dates and exchange rates
    time_series = data.get("Time Series FX (Daily)", {})
    dates = []
    rates = []
    for date, values in time_series.items():
        dates.append(datetime.strptime(date, "%Y-%m-%d"))
        rates.append(float(values["4. close"]))  # Use closing rate

    # Sort by date
    sorted_indices = np.argsort(dates)
    dates = np.array(dates)[sorted_indices]
    rates = np.array(rates)[sorted_indices]

    return dates, rates


# Step 2: Define the exponential model
def exponential_model(x, a, b, c):
    return a * np.exp(b * x) + c


# Step 3: Fit the model to the data
def fit_exponential_model(dates, rates):
    # Convert dates to numerical values (days since the first date)
    x_data = np.array([(date - dates[0]).days for date in dates])
    y_data = rates

    # Initial guess for parameters (a, b, c)
    p0 = (1, 0.01, 1)

    # Fit the model
    popt, pcov = curve_fit(exponential_model, x_data, y_data, p0=p0)
    return popt, x_data, y_data


# Step 4: Visualize the results
def plot_results(dates, rates, popt, x_data):
    plt.figure(figsize=(10, 6))
    plt.scatter(dates, rates, label="Actual USDBRL Rates", color="blue")

    # Generate fitted curve
    fitted_rates = exponential_model(x_data, *popt)
    plt.plot(dates, fitted_rates, label="Fitted Exponential Model", color="red")

    plt.title("USDBRL Exchange Rate with Exponential Fit")
    plt.xlabel("Date")
    plt.ylabel("Exchange Rate (USD/BRL)")
    plt.legend()
    plt.grid(True)
    plt.show()


# Main function
def main():
    # Fetch data
    dates, rates = fetch_usdbrl_data()

    # Fit the model
    popt, x_data, y_data = fit_exponential_model(dates, rates)

    # Print fitted parameters
    print(f"Fitted parameters: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}")

    # Plot results
    plot_results(dates, rates, popt, x_data)


if __name__ == "__main__":
    main()
