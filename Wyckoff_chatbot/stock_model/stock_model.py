import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from collections import deque
import random

class StockTradingModel:
    def __init__(self):
        self.q_table = np.zeros((100, 2, 3))  # As in your notebook code
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.episodes = 1000
        self.trained = False
        
    def discretize(self, price, stock_prices, n_bins=100):
        min_price, max_price = min(stock_prices), max(stock_prices)
        bin_width = (max_price - min_price) / n_bins
        return min(n_bins - 1, max(0, int((price - min_price) // bin_width)))
    
    def train(self, stock_prices):
        if self.trained:
            return
            
        env = StockTradingEnv(stock_prices)
        
        for episode in range(self.episodes):
            state = env.reset()
            done = False
            
            while not done:
                price_bin = self.discretize(state[0], stock_prices)
                position = int(state[1])
                
                # Epsilon-greedy policy
                if random.uniform(0, 1) < self.epsilon:
                    action = random.choice(env.get_possible_actions())
                else:
                    action = np.argmax(self.q_table[price_bin, position])
                
                # Take action
                next_state, reward, done = env.step(action)
                next_price_bin = self.discretize(next_state[0], stock_prices)
                next_position = int(next_state[1])
                
                # Update Q-value
                best_next_action = np.argmax(self.q_table[next_price_bin, next_position])
                self.q_table[price_bin, position, action] = self.q_table[price_bin, position, action] + self.alpha * (
                    reward + self.gamma * self.q_table[next_price_bin, next_position, best_next_action] - 
                    self.q_table[price_bin, position, action])
                
                state = next_state
        
        self.trained = True
    
    def analyze(self, stock_data):
        stock_prices = stock_data['Close'].values
        self.train(stock_prices)
        
        # Now test the model using the trained Q-table
        env = StockTradingEnv(stock_prices)
        state = env.reset()
        done = False
        actions = []
        portfolio_values = []
        
        while not done:
            price_bin = self.discretize(state[0], stock_prices)
            position = int(state[1])
            action = np.argmax(self.q_table[price_bin, position])
            state, reward, done = env.step(action)
            
            actions.append(action)
            portfolio_values.append(env.portfolio_value)
        
        # Calculate metrics
        initial_value = 50000  # Your initial portfolio value
        final_value = portfolio_values[-1]
        total_return = ((final_value - initial_value) / initial_value) * 100
        
        # Count trades
        buy_count = actions.count(1)
        sell_count = actions.count(2)
        hold_count = actions.count(0)
        
        # Create analysis results
        analysis = {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'hold_count': hold_count,
            'portfolio_values': portfolio_values,
            'actions': actions,
            'stock_prices': stock_prices.tolist()
        }
        
        # Format the response nicely
        response = f"""
        Stock Analysis Results:
        
        Initial Portfolio Value: ${initial_value:,.2f}
        Final Portfolio Value: ${final_value:,.2f}
        Total Return: {total_return:.2f}%
        
        Trading Activity:
        - Buy actions: {buy_count}
        - Sell actions: {sell_count}
        - Hold actions: {hold_count}
        
        The model suggests {'a positive outlook' if total_return > 0 else 'a negative outlook'} 
        for this stock based on historical performance.
        """
        
        return response

# Stock Trading Environment class (same as your notebook)
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
