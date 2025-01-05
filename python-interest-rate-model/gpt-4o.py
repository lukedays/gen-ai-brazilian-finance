import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup


# Step 1: Scrape the current SELIC rate
def get_selic_rate():
    url = "https://www.bcb.gov.br"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    selic_rate = soup.find("div", {"class": "selic-rate"}).text.strip()
    return float(selic_rate.replace(",", "."))


# Step 2: Implement the CIR model
def cir_model(r0, kappa, theta, sigma, T, dt):
    num_steps = int(T / dt)
    rates = np.zeros(num_steps)
    rates[0] = r0
    for t in range(1, num_steps):
        dr = (
            kappa * (theta - rates[t - 1]) * dt
            + sigma * np.sqrt(rates[t - 1]) * np.random.normal()
        )
        rates[t] = rates[t - 1] + dr
    return rates


# Step 3: Simulate the interest rate path
def simulate_selic_rate():
    r0 = get_selic_rate()
    kappa = 0.5  # Speed of reversion
    theta = r0  # Long-term mean level
    sigma = 0.02  # Volatility
    T = 1.0  # Time horizon in years
    dt = 1 / 252  # Daily steps

    rates = cir_model(r0, kappa, theta, sigma, T, dt)

    plt.plot(rates)
    plt.title("Simulated SELIC Rate using CIR Model")
    plt.xlabel("Time (days)")
    plt.ylabel("SELIC Rate")
    plt.show()


if __name__ == "__main__":
    simulate_selic_rate()
