import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class ApexPrimalEnv(gym.Env):
    """
    Apex Primal Sandbox: The Ultimate DRL Trading Environment
    Features:
    - Multi-Timeframe Vision Matrix (1m, 15m, 1H, Daily)
    - Macro/Sentiment Sensor (-1 to 1)
    - Dynamic Risk Shapeshifter Reward System
    """
    def __init__(self, df_1m, df_15m, df_1h, df_daily, initial_balance=10000):
        super(ApexPrimalEnv, self).__init__()
        
        # We store the multi-timeframe dataframes here
        self.df_1m = df_1m.reset_index(drop=True)
        self.df_15m = df_15m.reset_index(drop=True)
        self.df_1h = df_1h.reset_index(drop=True)
        self.df_daily = df_daily.reset_index(drop=True)
        
        self.initial_balance = initial_balance
        
        # Action Space: 0 = Hold, 1 = Buy/Long, 2 = Sell/Short
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: A 3D Matrix
        # [Timeframes (4)] x [Candle Window (20)] x [Features (6: O,H,L,C,V, Sentiment)]
        self.window_size = 20
        self.num_features = 6 
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(4, self.window_size, self.num_features), 
            dtype=np.float32
        )
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.position = 0 # 0=flat, 1=long, -1=short
        self.entry_price = 0
        self.trade_duration = 0 # Track how many candles a trade is held
        self.done = False
        return self._next_observation(), {}
        
    def _next_observation(self):
        # 1. Grab the last 20 candles of the 1m chart
        obs_1m = self._get_window(self.df_1m, self.current_step)
        
        # In a fully mapped system, we sync the timestamps. 
        # For the sandbox architecture, we simulate the higher timeframes for the matrix.
        obs_15m = self._get_window(self.df_15m, self.current_step // 15)
        obs_1h = self._get_window(self.df_1h, self.current_step // 60)
        obs_daily = self._get_window(self.df_daily, self.current_step // 1440)
        
        # Combine into a 4-channel matrix (like an RGB image, but 4 dimensions)
        matrix = np.array([obs_1m, obs_15m, obs_1h, obs_daily])
        return matrix.astype(np.float32)
        
    def _get_window(self, df, step):
        # Safe window extraction
        start = max(0, step - self.window_size)
        end = max(self.window_size, step)
        window = df.iloc[start:end].copy()
        
        # Add Mock Sentiment Data (-1 to 1)
        window['Sentiment'] = np.random.uniform(-1, 1, len(window))
        
        # Pad if not enough data
        if len(window) < self.window_size:
            pad = pd.DataFrame(np.zeros((self.window_size - len(window), 6)), columns=window.columns)
            window = pd.concat([pad, window], ignore_index=True)
            
        return window.values
        
    def step(self, action):
        self.current_step += 1
        current_price = self.df_1m.iloc[self.current_step]['Close']
        reward = 0
        
        # AI Chooses to BUY
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price
            self.trade_duration = 0
            
        # AI Chooses to SELL
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = current_price
            self.trade_duration = 0
            
        # AI Chooses to HOLD
        elif self.position != 0:
            self.trade_duration += 1
            unrealized = (current_price - self.entry_price) if self.position == 1 else (self.entry_price - current_price)
            
            # --- THE SHAPESHIFTER PENALTY/REWARD SYSTEM ---
            # If the market is chopping sideways (low volatility), penalize holding too long (Force Scalping)
            # If the market is trending heavily, reward holding (Force Swing Trading)
            
            # Tiny continuous penalty for drawdowns (forces tight stops)
            if unrealized < 0:
                reward -= abs(unrealized) * 0.01 
                
        # AI Closes Position
        elif (action == 2 and self.position == 1) or (action == 1 and self.position == -1):
            profit = (current_price - self.entry_price) if self.position == 1 else (self.entry_price - current_price)
            reward += profit
            self.balance += profit
            self.position = 0
            self.entry_price = 0
            
        if self.current_step >= len(self.df_1m) - 1 or self.balance <= 0:
            self.done = True
            
        return self._next_observation(), reward, self.done, False, {}
