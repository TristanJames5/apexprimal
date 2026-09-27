import yfinance as yf
from stable_baselines3 import PPO
from apex_env import ApexTradingEnv
from notifier import send_discord_alert # We will implement this to track progress

print("========================================")
print(" 🚀 PROJECT APEX - COLAB TRAINING INIT")
print("========================================")

# 1. Download 10 years of Bitcoin data as the training ground
print("Downloading 10 years of market data for the AI to study...")
btc = yf.download("BTC-USD", period="max")
btc = btc[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# 2. Build the Virtual Sandbox
print("Building the virtual trading gymnasium...")
env = ApexTradingEnv(df=btc)

# 3. Create the Deep Reinforcement Learning Agent (PPO)
print("Initializing Proximal Policy Optimization (PPO) Neural Network...")
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

# 4. Train the AI (This takes hours on Colab)
print("WARNING: Starting 1 Million Simulated Trades...")
# In Colab, we will change this to 10,000,000 steps.
model.learn(total_timesteps=10000)

print("Training Complete! Saving Brain...")
model.save("apex_god_model")
print("Brain saved as apex_god_model.zip")
