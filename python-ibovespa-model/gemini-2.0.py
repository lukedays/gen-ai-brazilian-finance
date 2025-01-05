import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from sklearn.cluster import KMeans

# Define the list of Ibovespa tickers (adjust as needed)
tickers = [
    "PETR4.SA",
    "VALE3.SA",
    "ITUB4.SA",
    "BBDC4.SA",
    "BBAS3.SA",
    "ABEV3.SA",
    "SUZB3.SA",
    "RAIL3.SA",
    "ELET3.SA",
    "WEGE3.SA",
]

# Download historical data
data = yf.download(tickers, period="5y")["Close"]

# Calculate daily returns
returns = data.pct_change().dropna()


# Feature Engineering for KMeans (using technical indicators)
def get_features(df):
    features = pd.DataFrame()
    features["Volatility"] = df.rolling(window=20).std()
    features["Momentum"] = df.pct_change(periods=5)  # 5-day return
    features["RSI"] = relative_strength_index(df, n=14)
    features["SMA_20"] = df.rolling(window=20).mean()
    features["SMA_50"] = df.rolling(window=50).mean()
    features["SMA_diff"] = features["SMA_20"] - features["SMA_50"]
    return features.dropna()


def relative_strength_index(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[: n + 1]
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]  # cause the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.0
        else:
            upval = 0.0
            downval = -delta

        up = (up * (n - 1) + upval) / n
        down = (down * (n - 1) + downval) / n

        rs = up / down
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


features = get_features(data)

# KMeans Clustering
n_clusters = 3  # Number of clusters (can be optimized)
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(features)


# Portfolio Optimization (using cluster means as expected returns)
def portfolio_volatility(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252  # annualized
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_std_dev


def neg_sharpe_ratio(weights, returns, risk_free_rate=0):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return -sharpe_ratio  # We minimize the negative Sharpe ratio


best_weights = []
for cluster in range(n_clusters):
    cluster_returns = returns[cluster_labels == cluster]
    if len(cluster_returns) == 0:
        continue  # Avoid empty clusters

    num_assets = len(cluster_returns.columns)
    args = (cluster_returns,)
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_weights = np.array([1 / num_assets] * num_assets)
    result = minimize(
        neg_sharpe_ratio,
        initial_weights,
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights.append(result.x)

# Print optimal weights for each cluster
for i, weights in enumerate(best_weights):
    print(f"Optimal weights for cluster {i+1}:")
    for j, ticker in enumerate(returns.columns[cluster_labels == i]):
        print(f"{ticker}: {weights[j]:.4f}")
    print("-" * 20)

# Example of how to get the portfolio return and volatility
all_weights = np.concatenate(best_weights)
all_returns = returns
port_vol = portfolio_volatility(all_weights, all_returns)
port_ret = np.sum(all_returns.mean() * all_weights) * 252
print(f"Portfolio Return: {port_ret}")
print(f"Portfolio Volatility: {port_vol}")
