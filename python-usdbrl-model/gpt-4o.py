import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.optimize import curve_fit

# Step 1: Fetch the USDBRL rate data from a web source
url = "https://api.exchangerate.host/timeseries?start_date=2022-01-01&end_date=2023-01-01&base=USD&symbols=BRL"
response = requests.get(url)
data = response.json()

# Step 2: Parse the data and store it in a pandas DataFrame
rates = data["rates"]
dates = list(rates.keys())
values = [rates[date]["BRL"] for date in dates]

df = pd.DataFrame({"Date": pd.to_datetime(dates), "Rate": values})
df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)


# Step 3: Fit an exponential model to the data
def exponential_model(x, a, b, c):
    return a * np.exp(b * x) + c


# Convert dates to ordinal for fitting
x_data = df["Date"].map(pd.Timestamp.toordinal).values
y_data = df["Rate"].values

# Initial guess for the parameters
initial_guess = [1, 0.0001, 1]

# Fit the model
params, covariance = curve_fit(exponential_model, x_data, y_data, p0=initial_guess)

# Step 4: Plot the data and the fitted model
plt.figure(figsize=(10, 6))
plt.plot(df["Date"], df["Rate"], label="Actual Data")
plt.plot(
    df["Date"], exponential_model(x_data, *params), label="Fitted Model", linestyle="--"
)
plt.xlabel("Date")
plt.ylabel("USDBRL Rate")
plt.title("USDBRL Rate with Exponential Fit")
plt.legend()
plt.show()
