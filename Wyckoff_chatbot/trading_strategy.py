import math
import random
import numpy as np
import pandas as pd
from collections import deque

# Global variables
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 1000
q_table = None  # Will be initialized during q_learning call

# Define the stock trading environment
class StockTradingEnv:
    def __init__(self, stock_prices):
        self.stock_prices = stock_prices
        self.current_step = 0
        self.max_steps = len(stock_prices)
        self.position = 0  # Position: 1 for holding stock, 0 for not holding
        self.cash = 50000  # Initial cash
        self.stock_owned = 100  # Number of stocks owned
        self.total_reward = self.cash
        self.portfolio_value = self.cash  # Starting portfolio value

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.cash = 50000 - 100 * self.stock_prices[0]  # take an arbitrary action of Buy
        self.stock_owned = 100
        self.total_reward = 0
        self.portfolio_value = self.cash
        return self._get_state()

    def _get_state(self):
        return np.array([self.stock_prices[self.current_step], self.position])

    def step(self, action):
        current_price = self.stock_prices[self.current_step]     
        reward = self.portfolio_value

        # Action: 0 = hold, 1 = buy, 2 = sell
        if action == 1 and self.position == 0:  # do not own the stock, and Buy
            self.stock_owned = self.cash // current_price
            self.cash = self.portfolio_value - self.stock_owned * current_price
            reward = self.cash + self.stock_owned * current_price
            self.position = 1  # Now holding stock

        elif action == 2 and self.position == 1:  # own the stock, and Sell
            self.cash = self.cash + self.stock_owned * current_price
            reward = self.cash  # Profit (compared to initial cash)
            self.stock_owned = 0
            self.position = 0  # No longer holding stock

        # Portfolio value = cash + value of stock owned
        self.portfolio_value = self.cash + self.stock_owned * current_price   
        
        # Move to the next step (next time step in stock price)
        self.current_step += 1
        done = self.current_step >= self.max_steps - 1
        next_state = self._get_state()

        return next_state, reward, done

    def get_possible_actions(self):
        return [0, 1, 2]  # Hold, Buy, Sell

# Discretize stock price into bins for Q-table
def discretize(price, stock_prices=None, n_bins=100):
    if stock_prices is None:
        # Use a default approach if stock_prices not provided
        return min(n_bins - 1, max(0, int(price // 10)))
    else:
        min_price, max_price = min(stock_prices), max(stock_prices)
        bin_width = (max_price - min_price) / n_bins
        return min(n_bins - 1, max(0, int((price - min_price) // bin_width)))

# Q-learning function
def q_learning(env):
    global q_table  # Use global q_table to make it accessible outside
    
    # Initialize the Q-table: states (price, position) x actions (hold, buy, sell)
    q_table = np.zeros((100, 2, 3))  # Discretize price into 100 bins, 2 positions (holding or not), 3 actions
    
    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            price_bin = discretize(state[0], env.stock_prices)
            position = int(state[1])

            # Epsilon-greedy policy
            if random.uniform(0, 1) < epsilon:
                action = random.choice(env.get_possible_actions())
            else:
                action = np.argmax(q_table[price_bin, position])

            # Take action
            next_state, reward, done = env.step(action)
            next_price_bin = discretize(next_state[0], env.stock_prices)
            next_position = int(next_state[1])

            # Update Q-value
            best_next_action = np.argmax(q_table[next_price_bin, next_position])
            q_table[price_bin, position, action] = q_table[price_bin, position, action] + alpha * (
                reward + gamma * q_table[next_price_bin, next_position, best_next_action] - q_table[price_bin, position, action])

            state = next_state