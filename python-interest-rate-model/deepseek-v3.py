import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup


# Step 1: Retrieve the current SELIC rate from the web
def get_current_selic_rate():
    url = "https://tradingeconomics.com/brazil/interest-rate"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract the SELIC rate from the webpage
    selic_rate = (
        soup.find("div", {"class": "form-group"})
        .find("span", {"class": "text-success"})
        .text.strip()
    )
    return float(selic_rate.replace("%", ""))


# Step 2: Define the CIR model parameters and simulate the interest rate path
def cir_model_simulation(r0, kappa, theta, sigma, T=1, dt=0.01, n_simulations=100):
    """
    Simulate the CIR interest rate model.

    Parameters:
    - r0: Initial interest rate (current SELIC rate)
    - kappa: Speed of mean reversion
    - theta: Long-term mean interest rate
    - sigma: Volatility of the interest rate
    - T: Time horizon (in years)
    - dt: Time step (in years)
    - n_simulations: Number of simulations

    Returns:
    - t: Time array
    - r: Simulated interest rate paths
    """
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps + 1)
    r = np.zeros((n_steps + 1, n_simulations))
    r[0, :] = r0

    for i in range(n_steps):
        dW = np.random.normal(0, np.sqrt(dt), n_simulations)
        r[i + 1, :] = (
            r[i, :] + kappa * (theta - r[i, :]) * dt + sigma * np.sqrt(r[i, :]) * dW
        )

    return t, r


# Step 3: Main execution
if __name__ == "__main__":
    # Retrieve the current SELIC rate
    current_selic_rate = get_current_selic_rate()
    print(f"Current SELIC Rate: {current_selic_rate}%")

    # Define CIR model parameters (example values)
    kappa = 0.5  # Speed of mean reversion
    theta = 12.0  # Long-term mean interest rate
    sigma = 0.1  # Volatility

    # Simulate the CIR model
    t, r = cir_model_simulation(
        current_selic_rate, kappa, theta, sigma, T=1, dt=0.01, n_simulations=10
    )

    # Plot the results
    plt.figure(figsize=(10, 6))
    for i in range(r.shape[1]):
        plt.plot(t, r[:, i], lw=1)
    plt.title("CIR Model Simulation for SELIC Rate")
    plt.xlabel("Time (Years)")
    plt.ylabel("Interest Rate (%)")
    plt.grid(True)
    plt.show()
