import numpy as np
import requests
from bs4 import BeautifulSoup
from scipy.stats import norm


def get_selic_rate():
    """Scrapes the current SELIC rate from the Central Bank of Brazil website."""
    try:
        url = "https://www.bcb.gov.br/dados/serie/bcdata.shtm?id=11"
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find("table", {"class": "tabela-padrao"})
        last_row = table.find_all("tr")[-1]
        rate_str = last_row.find_all("td")[1].text.replace(",", ".")
        return float(rate_str)
    except (requests.exceptions.RequestException, ValueError, IndexError) as e:
        print(f"Error fetching SELIC rate: {e}")
        return None


def cir_process(r0, k, theta, sigma, dt, T, N):
    """Simulates the CIR interest rate process.

    Args:
      r0: Initial interest rate.
      k: Mean reversion speed.
      theta: Long-term mean of the interest rate.
      sigma: Volatility of the interest rate.
      dt: Time step.
      T: Time horizon.
      N: Number of simulations.

    Returns:
      A NumPy array containing the simulated interest rate paths.
    """
    timesteps = int(T / dt)
    rates = np.zeros((N, timesteps + 1))
    rates[:, 0] = r0

    for t in range(timesteps):
        dr1 = k * (theta - rates[:, t]) * dt
        dr2 = sigma * np.sqrt(rates[:, t]) * np.random.normal(size=N)
        rates[:, t + 1] = rates[:, t] + dr1 + dr2

    return rates


if __name__ == "__main__":
    # Get current SELIC rate
    selic_rate = get_selic_rate()
    if selic_rate is None:
        print("Could not fetch SELIC rate. Exiting.")
        exit()

    # Model parameters (example values)
    k = 0.05  # Mean reversion speed
    theta = 0.10  # Long-term mean
    sigma = 0.02  # Volatility
    dt = 0.01  # Time step (monthly)
    T = 1  # Time horizon (in years)
    N = 1000  # Number of simulations

    # Simulate interest rate paths
    simulated_rates = cir_process(selic_rate, k, theta, sigma, dt, T, N)

    # Further analysis and visualization can be performed here
    # (e.g., plotting the simulated paths, calculating statistics)
    print(f"Simulated SELIC rate paths for {N} simulations.")
