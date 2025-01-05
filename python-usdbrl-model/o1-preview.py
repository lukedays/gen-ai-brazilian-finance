# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from scipy.optimize import curve_fit

# Step 1: Fetch historical USDBRL exchange rate data
symbol = "BRL=X"  # Yahoo Finance ticker symbol for USD/BRL exchange rate
data = yf.download(symbol, period="1y")  # Download data for the past 1 year

# Check if data is fetched successfully
if data.empty:
    raise ValueError(
        "No data fetched. Please check the ticker symbol or your internet connection."
    )

# Step 2: Prepare the data for fitting
dates = data.index
times = np.arange(len(dates))  # Numeric time variable for fitting
rates = data["Close"].values  # Closing exchange rates


# Step 3: Define the exponential model function
def exponential_model(x, a, b, c):
    return a * np.exp(b * x) + c


# Initial guess for the parameters [a, b, c]
initial_guess = [rates[0], 0, 0]

# Step 4: Fit the exponential model to the data
try:
    popt, pcov = curve_fit(exponential_model, times, rates, p0=initial_guess)
except RuntimeError:
    raise RuntimeError(
        "Optimal parameters not found: Number of calls to function has reached maxfev."
    )

# Extract the optimal parameters
a_opt, b_opt, c_opt = popt
print(f"Optimal parameters:\na = {a_opt}\nb = {b_opt}\nc = {c_opt}")

# Step 5: Plot the original data and the fitted model
plt.figure(figsize=(12, 6))
plt.plot(dates, rates, "b.", label="Actual Exchange Rate")
plt.plot(dates, exponential_model(times, *popt), "r-", label="Fitted Exponential Model")
plt.xlabel("Date")
plt.ylabel("Exchange Rate (BRL per USD)")
plt.title("USDBRL Exchange Rate and Exponential Fit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
