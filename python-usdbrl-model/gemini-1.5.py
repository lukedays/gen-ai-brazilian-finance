import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# Define the exponential function
def exponential_func(x, a, b):
    return a * np.exp(b * x)


# Get historical USDBRL rates (replace with your data source)
# Example using hypothetical data:
dates = ["2023-01-01", "2023-01-02", "2023-01-03", ...]
rates = [5.10, 5.15, 5.20, ...]

# Convert dates to numerical values (e.g., days since a starting date)
x_data = np.arange(len(rates))

# Fit the exponential curve
popt, pcov = curve_fit(exponential_func, x_data, rates)
a, b = popt

# Generate predicted values
y_pred = exponential_func(x_data, a, b)

# Plot the results
plt.plot(x_data, rates, label="Actual")
plt.plot(x_data, y_pred, label="Predicted")
plt.xlabel("Time")
plt.ylabel("USDBRL Rate")
plt.legend()
plt.show()
