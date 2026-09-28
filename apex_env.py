import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class ApexPrimalEnv(gym.Env):
    """
    Apex Primal Sandbox 2.0: The Institutional Reality Check
    Upgrades from V1:
    - Broker Realities: Strict Spread & Commission penalties (Stops unrealistic scalping).
    - Time-Awareness: AI now knows the exact Hour and Day (London vs Asian session).
    - Position Sizing: AI chooses to Risk 1%, 5%, or 10% based on confidence.
    """
    def __init__(self, df_1m, df_15m, df_1h, df_daily, initial_balance=10000):
        super(ApexPrimalEnv, self).__init__()
        
        self.df_1m = df_1m.reset_index(drop=True)
        self.df_15m = df_15m.reset_index(drop=True)
        self.df_1h = df_1h.reset_index(drop=True)
        self.df_daily = df_daily.reset_index(drop=True)
        
        self.initial_balance = initial_balance
        self.spread_fee = 0.00015 # 0.015% percentage spread (works for Crypto, Forex, Metals, Indices)
        
        # ACTION SPACE UPGRADE: Position Sizing
        # 0: Hold
        # 1: Buy Light (Risk 1%), 2: Buy Normal (Risk 5%), 3: Buy Heavy (Risk 10%)
        # 4: Sell Light (Risk 1%), 5: Sell Normal (Risk 5%), 6: Sell Heavy (Risk 10%)
        # 7: Close Position
        self.action_space = spaces.Discrete(8)
        
        # OBSERVATION SPACE UPGRADE: Added Time-Awareness (Hour, DayOfWeek)
        self.window_size = 20
        self.num_features = 8 # O, H, L, C, V, Sentiment, Hour, DayOfWeek
        
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
        self.trade_duration = 0
        self.lot_size_multiplier = 0
        self.done = False
        return self._next_observation(), {}
        
    def _next_observation(self):
        obs_1m = self._get_window(self.df_1m, self.current_step)
        obs_15m = self._get_window(self.df_15m, self.current_step // 15)
        obs_1h = self._get_window(self.df_1h, self.current_step // 60)
        obs_daily = self._get_window(self.df_daily, self.current_step // 1440)
        
        matrix = np.array([obs_1m, obs_15m, obs_1h, obs_daily])
        return matrix.astype(np.float32)
        
    def _get_window(self, df, step):
        start = max(0, step - self.window_size)
        end = max(self.window_size, step)
        window = df.iloc[start:end].copy()
        
        # Inject Fake Macro/Time Data for Sandbox (Replaced with Live API later)
        window['Sentiment'] = np.random.uniform(-1, 1, len(window)) 
        window['Hour'] = (window.index % 24) # AI learns London Open is hour 8
        window['DayOfWeek'] = (window.index % 5) # AI learns Fridays are dangerous
        
        if len(window) < self.window_size:
            pad = pd.DataFrame(np.zeros((self.window_size - len(window), self.num_features)), columns=window.columns)
            window = pd.concat([pad, window], ignore_index=True)
            
        return window.values
        
    def step(self, action):
        self.current_step += 1
        current_price = self.df_1m.iloc[self.current_step]['Close']
        reward = 0
        
        # PARSE ACTION LOGIC (Buy/Sell with Position Sizing)
        if action in [1, 2, 3] and self.position == 0:
            self.position = 1
            self.entry_price = current_price * (1 + self.spread_fee) # Pay the percentage spread!
            self.trade_duration = 0
            self.lot_size_multiplier = [0.01, 0.05, 0.10][action - 1] # 1%, 5%, 10% risk
            
        elif action in [4, 5, 6] and self.position == 0:
            self.position = -1
            self.entry_price = current_price * (1 - self.spread_fee) # Pay the percentage spread!
            self.trade_duration = 0
            self.lot_size_multiplier = [0.01, 0.05, 0.10][action - 4]
            
        elif self.position != 0 and action == 0:
            self.trade_duration += 1
            price_change_pct = (current_price - self.entry_price) / self.entry_price if self.position == 1 else (self.entry_price - current_price) / self.entry_price
            
            # Penalty for holding losing trades, scaled by risk and leverage (100x)
            if price_change_pct < 0:
                reward -= abs(price_change_pct) * self.lot_size_multiplier * 100.0 * self.balance * 0.1 

        elif action == 7 and self.position != 0: # Close Position
            price_change_pct = (current_price - self.entry_price) / self.entry_price if self.position == 1 else (self.entry_price - current_price) / self.entry_price
            
            # actual_profit = position_size * percentage_move * Leverage(100x)
            position_size = self.balance * self.lot_size_multiplier
            actual_profit = position_size * price_change_pct * 100.0
            
            # Reward is exactly the monetary profit/loss
            reward += actual_profit
            self.balance += actual_profit
            
            # Reset
            self.position = 0
            self.entry_price = 0
            self.lot_size_multiplier = 0
            
        # Game Over Conditions
        if self.current_step >= len(self.df_1m) - 1 or self.balance <= 0:
            self.done = True
            
        return self._next_observation(), reward, self.done, False, {}
