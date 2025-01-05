# Import required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Step 1: Fetch Ibovespa Stock Data
tickers = [
    "PETR4.SA",
    "VALE3.SA",
    "ITUB4.SA",
    "BBDC4.SA",
    "BBAS3.SA",
    "B3SA3.SA",
    "WEGE3.SA",
    "SUZB3.SA",
    "RENT3.SA",
    "MGLU3.SA",
]

# Fetch historical data for the last 5 years
data = yf.download(tickers, start="2020-01-01", end="2025-01-05")["Close"]

# Drop rows with missing values
data.dropna(inplace=True)

# Step 2: Preprocess Data
# Calculate daily returns
returns = data.pct_change().dropna()

# Split data into features (X) and target (y)
X = returns.shift(1).dropna()  # Lagged returns as features
y = returns.iloc[1:]  # Current returns as target

# Step 3: Train a Machine Learning Model
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Predict returns
predicted_returns = model.predict(X_test)

# Step 4: Optimize Portfolio Weights
# Calculate covariance matrix
cov_matrix = returns.cov()


# Define objective function (maximize Sharpe ratio)
def objective(weights, returns, cov_matrix, risk_free_rate=0.02):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix, weights))
    ) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio  # Minimize negative Sharpe ratio


# Constraints: weights sum to 1
constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

# Bounds: weights between 0 and 1
bounds = tuple((0, 1) for _ in range(len(tickers)))

# Initial guess: equal weights
initial_guess = [1 / len(tickers)] * len(tickers)

# Optimize
optimal_results = minimize(
    objective,
    initial_guess,
    args=(returns, cov_matrix),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)

# Optimal weights
optimal_weights = optimal_results.x
print("Optimal Weights:", dict(zip(tickers, optimal_weights)))

# Step 5: Evaluate Portfolio Performance
# Portfolio metrics
portfolio_return = np.sum(returns.mean() * optimal_weights) * 252
portfolio_volatility = np.sqrt(
    np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
) * np.sqrt(252)
sharpe_ratio = (portfolio_return - 0.02) / portfolio_volatility

print(f"Expected Annual Return: {portfolio_return:.2%}")
print(f"Annual Volatility: {portfolio_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Step 6: Visualize Results
# Cumulative returns
cumulative_returns = (1 + returns).cumprod()
portfolio_cumulative_returns = (cumulative_returns * optimal_weights).sum(axis=1)

# Plot portfolio performance
plt.figure(figsize=(12, 6))
plt.plot(portfolio_cumulative_returns, label="Portfolio Cumulative Returns")
plt.title("Portfolio Performance")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.show()

# Asset allocation
plt.figure(figsize=(8, 8))
plt.pie(optimal_weights, labels=tickers, autopct="%1.1f%%", startangle=140)
plt.title("Optimal Asset Allocation")
plt.show()
