import datetime

import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup


def get_current_selic_rate():
    """
    Scrapes the current SELIC rate from the Banco Central do Brasil website.
    Note: The structure of the website might change over time, so this function
    may need adjustments if the scraping fails.
    """
    try:
        # URL where the SELIC rate is published
        url = "https://www.bcb.gov.br/controleinflacao/historicoselic"

        # Send a GET request to the website
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status

        # Parse the HTML content
        soup = BeautifulSoup(response.content, "html.parser")

        # The SELIC rate is usually in a table. Find the latest rate.
        # This selector may need to be updated based on the actual website structure.
        table = soup.find("table", {"class": "tabela-tecnologia"})
        if table is None:
            raise ValueError("Could not find the SELIC rate table on the page.")

        # Extract table rows
        rows = table.find_all("tr")

        # Assuming the first data row contains the latest rate
        latest_row = rows[1]
        cols = latest_row.find_all("td")

        # The rate is usually in the second column
        selic_rate_str = cols[1].get_text(strip=True).replace(",", ".")
        selic_rate = float(selic_rate_str)

        print(f"Current SELIC Rate: {selic_rate}%")

        return selic_rate

    except Exception as e:
        print(f"An error occurred while scraping the SELIC rate: {e}")
        # Fallback rate if scraping fails
        print("Using default SELIC rate of 13.75%")
        return 13.75  # Default SELIC rate


def simulate_cir_model(r0, a, b, sigma, T=1.0, dt=1 / 252, paths=1000):
    """
    Simulates the CIR interest rate model.

    Parameters:
    - r0: initial interest rate
    - a: speed of mean reversion
    - b: long-term mean rate
    - sigma: volatility
    - T: total time in years
    - dt: time step size in years
    - paths: number of simulation paths

    Returns:
    - time_grid: array of time points
    - rates: simulated interest rates (paths x time steps)
    """
    n_steps = int(T / dt)
    time_grid = np.linspace(0, T, n_steps + 1)
    rates = np.zeros((paths, n_steps + 1))
    rates[:, 0] = r0

    for t in range(1, n_steps + 1):
        drift = a * (b - rates[:, t - 1]) * dt
        diffusion = (
            sigma
            * np.sqrt(rates[:, t - 1])
            * np.random.normal(0, np.sqrt(dt), size=paths)
        )
        rates[:, t] = rates[:, t - 1] + drift + diffusion
        # Ensure that rates stay positive
        rates[:, t] = np.maximum(rates[:, t], 0)

    return time_grid, rates


def main():
    # Step 1: Get current SELIC rate
    r0 = get_current_selic_rate()

    # Step 2: Define CIR model parameters
    # These parameters may need to be adjusted based on historical data
    a = 0.15  # Speed of mean reversion
    b = 13.5  # Long-term mean rate (%)
    sigma = 0.4  # Volatility

    # Step 3: Simulate the CIR model
    T = 1.0  # 1 year
    dt = 1 / 252  # Daily steps
    paths = 10  # Number of simulation paths (use higher for more realistic results)

    print("Simulating CIR model...")
    time_grid, rates = simulate_cir_model(r0, a, b, sigma, T, dt, paths)

    # Step 4: Plot the results
    plt.figure(figsize=(12, 6))
    for i in range(paths):
        plt.plot(time_grid, rates[i], lw=1, alpha=0.6)
    plt.title("CIR Model Simulation for SELIC Rate")
    plt.xlabel("Time (Years)")
    plt.ylabel("Interest Rate (%)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
