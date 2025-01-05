import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.cluster import KMeans

# List of 10 Ibovespa stocks
ibovespa_stocks = [
    "PETR4.SA",
    "VALE3.SA",
    "ITUB4.SA",
    "BBDC4.SA",
    "ABEV3.SA",
    "BBAS3.SA",
    "B3SA3.SA",
    "WEGE3.SA",
    "RENT3.SA",
    "MGLU3.SA",
]

# Fetch stock data
data = yf.download(ibovespa_stocks, start="2020-01-01", end="2023-01-01")["Close"]

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Number of clusters
num_clusters = 3

# KMeans clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(returns.T)

# Get cluster labels
labels = kmeans.labels_

# Create a DataFrame with stock symbols and their cluster labels
clustered_stocks = pd.DataFrame({"Stock": ibovespa_stocks, "Cluster": labels})

# Print clustered stocks
print(clustered_stocks)

# Plot clusters
for i in range(num_clusters):
    cluster = clustered_stocks[clustered_stocks["Cluster"] == i]
    plt.scatter(cluster["Stock"], [i] * len(cluster), label=f"Cluster {i}")

plt.legend()
plt.xlabel("Stock")
plt.ylabel("Cluster")
plt.title("KMeans Clustering of Ibovespa Stocks")
plt.show()

# Select stocks from each cluster for diversification
selected_stocks = []
for i in range(num_clusters):
    cluster = clustered_stocks[clustered_stocks["Cluster"] == i]
    selected_stocks.append(cluster["Stock"].iloc[0])

print("Selected stocks for portfolio optimization:", selected_stocks)

# Calculate portfolio weights
weights = np.random.random(len(selected_stocks))
weights /= np.sum(weights)

# Calculate expected portfolio return and risk
portfolio_return = np.dot(weights, mean_returns[selected_stocks])
portfolio_risk = np.sqrt(
    np.dot(weights.T, np.dot(cov_matrix[selected_stocks], weights))
)

print("Expected Portfolio Return:", portfolio_return)
print("Expected Portfolio Risk:", portfolio_risk)
