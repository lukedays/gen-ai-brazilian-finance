import datetime

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# 1. Select 10 Ibovespa Stocks
# For demonstration, we'll use the following 10 popular Ibovespa stocks
tickers = [
    "PETR4.SA",  # Petrobras
    "VALE3.SA",  # Vale
    "ITUB4.SA",  # Itaú Unibanco
    "B3SA3.SA",  # B3
    "ABEV3.SA",  # Ambev
    "BBAS3.SA",  # Banco do Brasil
    "WEGE3.SA",  # Weg
    "MGLU3.SA",  # Magazine Luiza
    "GGBR4.SA",  # Gerdau
    "RENT3.SA",  # Localiza
]

# 2. Fetch Historical Data
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=5 * 365)  # Last 5 years

data = yf.download(tickers, start=start_date, end=end_date)["Close"]

# Drop any columns with missing data
data.dropna(axis=1, inplace=True)

# 3. Preprocess Data
# Calculate daily returns
returns = data.pct_change().dropna()

# We'll use the past 60 days to predict the next day's return
look_back = 60


# Function to create dataset
def create_dataset(dataset, look_back=look_back):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i : (i + look_back)].values)
        y.append(dataset[i + look_back].values)
    return np.array(X), np.array(y)


X, y = create_dataset(returns, look_back)

# Split into training and testing sets (80% training, 20% testing)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Scale the data
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(
    X_train.shape
)
X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(
    X_test.shape
)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# 4. Build and Train the LSTM Model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, len(tickers)), return_sequences=True))
model.add(LSTM(50))
model.add(Dense(len(tickers)))

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(
    X_train_scaled, y_train_scaled, epochs=20, batch_size=32, validation_split=0.1
)

# Predict future returns
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Assume the last predicted returns as the expected returns for optimization
expected_returns = y_pred[-1]

# 5. Portfolio Optimization
# Calculate the covariance matrix from the training data
cov_matrix = returns.iloc[:split].cov().values

# Define optimization variables
weights = cp.Variable(len(tickers))

# Define the expected portfolio return
portfolio_return = expected_returns @ weights

# Define the portfolio risk
portfolio_risk = cp.quad_form(weights, cov_matrix)

# Define the optimization problem (maximize return for a given risk or maximize Sharpe ratio)
# Here, we'll maximize the Sharpe Ratio assuming risk-free rate of 0%
risk_free_rate = 0.0
objective = cp.Maximize((portfolio_return - risk_free_rate) / cp.sqrt(portfolio_risk))

# Constraints: weights sum to 1, weights >= 0 (no short selling)
constraints = [cp.sum(weights) == 1, weights >= 0]

# Define and solve the problem
prob = cp.Problem(objective, constraints)
prob.solve()

# Extract the optimized weights
optimized_weights = weights.value

# Display the results
portfolio = pd.DataFrame({"Ticker": tickers, "Weight": optimized_weights})

print("Optimized Portfolio Weights:")
print(portfolio)

# Plot the weights
portfolio.plot(kind="bar", x="Ticker", y="Weight", legend=False)
plt.title("Optimized Portfolio Weights")
plt.ylabel("Weight")
plt.show()
