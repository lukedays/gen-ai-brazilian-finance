import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy.optimize import minimize


def scrape_selic():
    """Scrapes the current SELIC target rate from the BCB website."""
    try:
        url = "https://www.bcb.gov.br/controleinflacao/taxaselic"  # Updated URL
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.content, "html.parser")
        # Updated selector to target the current Selic rate. Inspect the website to get the correct selector.
        selic_element = soup.find(
            "span", {"class": "txt-highlight"}
        )  # This is a broad search, might need refinement
        if selic_element:
            selic_text = selic_element.text.replace("%", "").replace(",", ".")
            selic_rate = float(selic_text) / 100  # Convert to decimal
            return selic_rate
        else:
            print(
                "Could not find the SELIC rate on the page. Check the website's HTML structure."
            )
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching SELIC rate: {e}")
        return None
    except (ValueError, TypeError) as e:
        print(f"Error parsing SELIC rate: {e}. Check the website's HTML structure.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def cir_model(r0, kappa, theta, sigma, T, n=1000):
    """Simulates the Cox-Ingersoll-Ross (CIR) model."""
    dt = T / n
    r = np.zeros(n + 1)
    r[0] = r0

    for i in range(n):
        dW = np.random.normal(0, np.sqrt(dt))
        r[i + 1] = (
            r[i]
            + kappa * (theta - r[i]) * dt
            + sigma * np.sqrt(np.maximum(0, r[i])) * dW
        )  # Max to prevent negative rates
    return r


def calibrate_cir(r_data, initial_guess=[0.1, 0.1, 0.1]):  # Calibration function
    """Calibrates the CIR model parameters using historical data."""

    def loss_function(params):
        kappa, theta, sigma = params
        r_simulated = cir_model(
            r_data[0], kappa, theta, sigma, len(r_data) - 1, len(r_data) - 1
        )  # Simulate for the same period as data
        return np.sqrt(np.mean((r_simulated - r_data) ** 2))  # RMSE as loss function

    bounds = [(0, 1), (0, 1), (0, 1)]  # Bounds for the parameters
    result = minimize(
        loss_function, initial_guess, bounds=bounds, method="L-BFGS-B"
    )  # Optimization
    return result.x


if __name__ == "__main__":
    current_selic = scrape_selic()

    if current_selic is not None:
        print(f"Current SELIC rate: {current_selic:.4%}")

        # Example usage (replace with your historical data if available):
        # For this part, you'll need a CSV file with historical SELIC data.
        try:
            historical_selic = pd.read_csv(
                "selic_historico.csv", index_col="Date", parse_dates=True
            )
            historical_selic = historical_selic["Value"].values / 100

            # Calibrate the CIR model
            calibrated_params = calibrate_cir(historical_selic)
            kappa, theta, sigma = calibrated_params

            print(
                f"Calibrated parameters: kappa={kappa:.4f}, theta={theta:.4f}, sigma={sigma:.4f}"
            )

            # Simulate future SELIC rates
            T = 1  # Time horizon (e.g., 1 year)
            simulated_rates = cir_model(current_selic, kappa, theta, sigma, T)

            # Plotting (optional)
            import matplotlib.pyplot as plt

            plt.plot(simulated_rates)
            plt.xlabel("Time Steps")
            plt.ylabel("SELIC Rate")
            plt.title("Simulated SELIC Rates using CIR Model")
            plt.show()

        except FileNotFoundError:
            print("selic_historico.csv file not found. Please provide historical data.")

        except Exception as e:
            print(
                f"An error occurred during historical data handling or simulation: {e}"
            )
    else:
        print("Could not retrieve current SELIC rate. Simulation aborted.")
