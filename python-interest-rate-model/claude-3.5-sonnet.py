import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup


# Scrape current SELIC rate
def get_current_selic():
    url = "https://www.bcb.gov.br/controleinflacao/taxaselic"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    try:
        # Note: This might need adjustment based on website structure changes
        rate_text = soup.find("div", class_="taxaSelic").text
        return float(rate_text.replace(",", ".").strip("%")) / 100
    except:
        return 0.1375  # Default to 13.75% if scraping fails


# CIR model simulation
def simulate_cir(r0, kappa, theta, sigma, T, N):
    dt = T / N
    rates = np.zeros(N)
    rates[0] = r0

    for i in range(1, N):
        dr = kappa * (theta - rates[i - 1]) * dt + sigma * np.sqrt(
            rates[i - 1]
        ) * np.random.normal(0, np.sqrt(dt))
        rates[i] = max(rates[i - 1] + dr, 0)  # Ensure rates stay positive

    return rates


# Parameters
r0 = get_current_selic()  # Initial rate (current SELIC)
kappa = 0.3  # Mean reversion speed
theta = 0.12  # Long-term mean
sigma = 0.03  # Volatility
T = 5  # Time horizon (years)
N = 1000  # Number of time steps
num_sims = 100  # Number of simulations

# Run multiple simulations
simulations = np.zeros((num_sims, N))
for i in range(num_sims):
    simulations[i] = simulate_cir(r0, kappa, theta, sigma, T, N)

# Plot results
plt.figure(figsize=(10, 6))
time = np.linspace(0, T, N)
for i in range(num_sims):
    plt.plot(time, simulations[i], alpha=0.1, color="blue")
plt.plot(time, np.mean(simulations, axis=0), "r--", linewidth=2, label="Mean")
plt.title(f"CIR Model Simulation of SELIC Rate\nInitial Rate: {r0*100:.2f}%")
plt.xlabel("Time (years)")
plt.ylabel("Interest Rate")
plt.grid(True)
plt.legend()
plt.show()

# Print statistics
print(f"Initial SELIC rate: {r0*100:.2f}%")
print(f"Mean rate at end: {np.mean(simulations[:,-1])*100:.2f}%")
print(f"Standard deviation at end: {np.std(simulations[:,-1])*100:.2f}%")
