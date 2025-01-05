from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from torch.distributions import Normal


# Define the Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs


# Define the Critic network
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class PortfolioOptimizer:
    def __init__(self):
        # Top 10 Ibovespa stocks by market cap (as of knowledge cutoff)
        self.tickers = [
            "VALE3.SA",
            "ITUB4.SA",
            "PETR4.SA",
            "BBDC4.SA",
            "B3SA3.SA",
            "ABEV3.SA",
            "PETR3.SA",
            "BBAS3.SA",
            "WEGE3.SA",
            "RENT3.SA",
        ]

        self.lookback = 30  # Days of historical data to consider
        self.episode_length = 252  # Trading days in a year

        # Initialize networks
        self.state_dim = len(self.tickers) * 5  # 5 features per stock
        self.action_dim = len(self.tickers)

        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

    def get_data(self):
        """Fetch historical data for the selected stocks"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.episode_length + self.lookback)

        data = {}
        for ticker in self.tickers:
            try:
                stock = yf.download(ticker, start=start_date, end=end_date)
                data[ticker] = stock
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")

        return data

    def preprocess_data(self, data):
        """Extract features from the raw data"""
        features = []
        for ticker in self.tickers:
            df = data[ticker]

            # Calculate technical indicators
            df["Returns"] = df["Close"].pct_change()
            df["SMA_20"] = df["Close"].rolling(window=20).mean()
            df["Volatility"] = df["Returns"].rolling(window=20).std()
            df["RSI"] = self.calculate_rsi(df["Close"])

            features.append(df[["Returns", "SMA_20", "Volatility", "RSI", "Close"]])

        return features

    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def get_state(self, features, t):
        """Create state vector from features at time t"""
        state = []
        for feature_set in features:
            state.extend(feature_set.iloc[t].values)
        return torch.FloatTensor(state)

    def train(self, episodes=100):
        """Train the portfolio optimizer"""
        best_reward = float("-inf")
        rewards_history = []

        for episode in range(episodes):
            data = self.get_data()
            features = self.preprocess_data(data)

            episode_rewards = []
            portfolio_value = 1.0
            portfolio_history = [portfolio_value]

            for t in range(self.lookback, len(features[0]) - 1):
                state = self.get_state(features, t)

                # Get action (portfolio weights) from actor
                action_probs = self.actor(state)
                action = action_probs.detach().numpy()

                # Calculate reward (portfolio returns)
                next_returns = [
                    feature_set["Returns"].iloc[t + 1] for feature_set in features
                ]
                portfolio_return = np.sum(action * next_returns)
                portfolio_value *= 1 + portfolio_return
                portfolio_history.append(portfolio_value)

                # Calculate reward (Sharpe Ratio component)
                reward = portfolio_return - 0.02 * np.std(
                    action
                )  # Include penalty for high concentration
                episode_rewards.append(reward)

                # Get next state
                next_state = self.get_state(features, t + 1)

                # Update networks
                value = self.critic(state)
                next_value = self.critic(next_state)

                # Compute advantage
                advantage = reward + 0.99 * next_value.detach() - value

                # Update critic
                critic_loss = advantage.pow(2).mean()
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # Update actor
                actor_loss = -(torch.log(action_probs) * advantage.detach()).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

            episode_reward = np.mean(episode_rewards)
            rewards_history.append(episode_reward)

            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(self.actor.state_dict(), "best_actor.pth")

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{episodes}")
                print(f"Average Reward: {episode_reward:.4f}")
                print(f"Final Portfolio Value: {portfolio_value:.2f}")
                print("Current Portfolio Weights:")
                for ticker, weight in zip(self.tickers, action):
                    print(f"{ticker}: {weight:.4f}")
                print("-" * 50)

        return rewards_history, portfolio_history

    def get_optimal_portfolio(self):
        """Get the optimal portfolio weights using the trained actor"""
        data = self.get_data()
        features = self.preprocess_data(data)
        state = self.get_state(features, -1)  # Get most recent state

        with torch.no_grad():
            weights = self.actor(state).numpy()

        return dict(zip(self.tickers, weights))


# Usage example
optimizer = PortfolioOptimizer()
rewards_history, portfolio_history = optimizer.train(episodes=100)

# Get optimal portfolio weights
optimal_portfolio = optimizer.get_optimal_portfolio()
print("\nFinal Optimal Portfolio Weights:")
for ticker, weight in optimal_portfolio.items():
    print(f"{ticker}: {weight:.4f}")

# Plot training results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(rewards_history)
plt.title("Training Rewards")
plt.xlabel("Episode")
plt.ylabel("Average Reward")

plt.subplot(1, 2, 2)
plt.plot(portfolio_history)
plt.title("Portfolio Value Evolution")
plt.xlabel("Trading Day")
plt.ylabel("Portfolio Value")
plt.tight_layout()
plt.show()
