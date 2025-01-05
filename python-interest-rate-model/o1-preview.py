import matplotlib.pyplot as plt
import numpy as np
import requests


def get_current_selic_rate():
    # URL to get the latest SELIC rate from Banco Central do Brasil API
    url = (
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/1?formato=json"
    )
    response = requests.get(url)
    data = response.json()

    # Extract the SELIC rate and convert to float
    selic_rate = float(data[0]["valor"].replace(",", "."))
    return selic_rate


def simulate_CIR(r0, a, b, sigma, T, N):
    dt = T / N  # Time step
    rates = np.zeros(N + 1)
    times = np.linspace(0, T, N + 1)
    rates[0] = r0

    for t in range(1, N + 1):
        r_t = rates[t - 1]
        sqrt_rt = np.sqrt(max(r_t, 0))
        # Generate a random shock from normal distribution
        epsilon = np.random.normal(0, 1)
        dr = a * (b - max(r_t, 0)) * dt + sigma * sqrt_rt * epsilon * np.sqrt(dt)
        rates[t] = r_t + dr
        rates[t] = max(rates[t], 0)  # Ensure the rate stays non-negative

    return times, rates


def main():
    # Get the current SELIC rate
    current_selic_rate = get_current_selic_rate()
    print(f"Current SELIC rate: {current_selic_rate}%")

    # Parameters for the CIR model
    r0 = current_selic_rate  # Initial rate
    a = 0.1  # Speed of mean reversion
    b = 8.0  # Long-term mean rate (%)
    sigma = 0.02  # Volatility parameter
    T = 1.0  # Time horizon in years
    N = 252  # Number of time steps (e.g., trading days in a year)

    # Simulate the CIR model
    times, rates = simulate_CIR(r0, a, b, sigma, T, N)

    # Plot the simulated SELIC rates over time
    plt.figure(figsize=(10, 6))
    plt.plot(times, rates, label="Simulated SELIC Rate")
    plt.xlabel("Time (years)")
    plt.ylabel("Interest Rate (%)")
    plt.title("CIR Model Simulation of SELIC Rate")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
