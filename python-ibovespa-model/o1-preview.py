# Import necessary libraries
# Suppress warnings
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# Define the list of 10 Ibovespa stock tickers
tickers = [
    "PETR4.SA",
    "VALE3.SA",
    "ITUB4.SA",
    "BBDC4.SA",
    "ABEV3.SA",
    "BBAS3.SA",
    "MGLU3.SA",
    "BRKM5.SA",
    "SUZB3.SA",
]

# Fetch historical data for the past 3 years
end_date = datetime.now()
start_date = end_date - timedelta(days=3 * 365)

data = yf.download(
    tickers, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")
)["Close"]

# Drop any columns with missing data
data = data.dropna()

# Calculate daily returns
returns = data.pct_change().dropna()

# Prepare data for LSTM
prediction_days = 60  # Use the past 60 days to predict the next day


# Function to prepare the data for LSTM
def prepare_lstm_data(stock_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data.values.reshape(-1, 1))

    X_train = []
    y_train = []

    for i in range(prediction_days, len(scaled_data)):
        X_train.append(scaled_data[i - prediction_days : i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    # Reshape data for LSTM input
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return X_train, y_train, scaler


# Dictionary to store predicted returns
predicted_returns = {}

# Train an LSTM model for each stock and predict the next day's return
for ticker in tickers:
    print(f"Training LSTM model for {ticker}...")
    stock_data = data[ticker]

    # Prepare the data
    X_train, y_train, scaler = prepare_lstm_data(stock_data)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Prediction of the next day

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    # Prepare the test input (most recent data)
    last_days = stock_data[-prediction_days:].values
    last_days_scaled = scaler.transform(last_days.reshape(-1, 1))
    X_test = []
    X_test.append(last_days_scaled[:, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Predict the next day's price
    predicted_price_scaled = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)

    # Calculate predicted return
    last_price = stock_data.values[-1]
    predicted_return = (predicted_price[0, 0] - last_price) / last_price
    predicted_returns[ticker] = predicted_return

# Display predicted returns
print("\nPredicted Returns:")
for ticker, ret in predicted_returns.items():
    print(f"{ticker}: {ret:.4f}")

# Convert predicted_returns to a DataFrame
predicted_returns_df = pd.DataFrame.from_dict(
    predicted_returns, orient="index", columns=["Expected Return"]
)

# Calculate the covariance matrix of returns
cov_matrix = returns.cov().to_numpy()

# Expected returns array
expected_returns = np.array([predicted_returns[ticker] for ticker in tickers])


# Mean-Variance Optimization
def portfolio_optimization(expected_returns, cov_matrix):
    num_assets = len(expected_returns)

    # Objective function (to minimize)
    def objective(weights):
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_return = np.dot(weights, expected_returns)
        # We want to maximize return and minimize variance
        # So we can minimize -return / variance
        return -portfolio_return / np.sqrt(portfolio_variance)

    # Constraints: sum of weights = 1
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    # Bounds: weights between 0 and 1
    bounds = tuple((0, 1) for _ in range(num_assets))
    # Initial guess
    init_guess = num_assets * [
        1.0 / num_assets,
    ]

    # Optimization
    result = minimize(
        objective, init_guess, method="SLSQP", bounds=bounds, constraints=constraints
    )

    return result.x


# Optimize the portfolio
optimal_weights = portfolio_optimization(expected_returns, cov_matrix)

# Display the optimal portfolio weights
print("\nOptimal Portfolio Allocation:")
for i, ticker in enumerate(tickers):
    print(f"{ticker}: {optimal_weights[i]:.4f}")

# Create a DataFrame for the portfolio allocation
portfolio_allocation = pd.DataFrame({"Ticker": tickers, "Allocation": optimal_weights})

# Plot the portfolio allocation
plt.figure(figsize=(10, 6))
plt.pie(
    portfolio_allocation["Allocation"],
    labels=portfolio_allocation["Ticker"],
    autopct="%1.1f%%",
    startangle=140,
)
plt.title("Optimal Portfolio Allocation")
plt.axis("equal")
plt.show()
