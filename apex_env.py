import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class ApexTradingEnv(gym.Env):
    """
    Custom Trading Environment for Stable Baselines3 (Deep Reinforcement Learning)
    The AI acts as a player in this 'video game'.
    Actions: 0 (Hold), 1 (Buy/Long), 2 (Sell/Short)
    """
    def __init__(self, df, initial_balance=10000):
        super(ApexTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        
        # Action Space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: The AI looks at the last 20 candles of data (Prices + Volatility)
        # Assuming the dataframe has 5 features: Open, High, Low, Close, Volume
        self.window_size = 20
        self.num_features = len(self.df.columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, self.num_features), dtype=np.float32
        )
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.position = 0 # 0=flat, 1=long, -1=short
        self.entry_price = 0
        self.done = False
        return self._next_observation(), {}
        
    def _next_observation(self):
        # The AI "sees" the last 20 candles
        obs = self.df.iloc[self.current_step - self.window_size : self.current_step].values
        return obs.astype(np.float32)
        
    def step(self, action):
        self.current_step += 1
        current_price = self.df.iloc[self.current_step]['Close']
        reward = 0
        
        # AI Chooses to BUY (Go Long)
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price
            
        # AI Chooses to SELL (Go Short)
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = current_price
            
        # AI Closes Position
        elif (action == 2 and self.position == 1) or (action == 1 and self.position == -1):
            profit = (current_price - self.entry_price) if self.position == 1 else (self.entry_price - current_price)
            reward = profit # The AI is rewarded instantly for making profit, punished for loss
            self.balance += reward
            self.position = 0
            self.entry_price = 0
            
        # Penalty for holding a losing trade (forces AI to cut losses early)
        if self.position != 0:
            unrealized = (current_price - self.entry_price) if self.position == 1 else (self.entry_price - current_price)
            if unrealized < 0:
                reward -= abs(unrealized) * 0.1 # Tiny negative shock to encourage tight Stop Losses
                
        # Game Over if we run out of data or blow the account
        if self.current_step >= len(self.df) - 1 or self.balance <= 0:
            self.done = True
            
        return self._next_observation(), reward, self.done, False, {}
