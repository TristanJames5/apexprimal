# 🦍 Apex Primal (Deep Reinforcement Learning Engine)

Welcome to the Apex Primal Sandbox. This repository contains the architecture for a fully autonomous Deep Reinforcement Learning (DRL) trading bot, built to train in the cloud.

## 🔗 How to Connect Discord (Live Training Updates)

To watch the AI learn in real-time, you must connect it to your Discord server via a Webhook.

**Step 1: Get your Webhook URL**
1. Open your Discord Server.
2. Right-click the text channel where you want the AI to send updates (e.g., `#ai-updates`).
3. Click **Edit Channel** ➔ **Integrations** ➔ **Webhooks**.
4. Click **New Webhook**. Name it "Apex Primal".
5. Click **Copy Webhook URL**.

**Step 2: Plug it into the Code**
You have two options:
* **Option A (Easy):** Open `discord_notifier.py` and replace `"YOUR_DISCORD_WEBHOOK_URL_HERE"` on Line 9 with the URL you just copied.
* **Option B (Secure for Kaggle):** When you upload this to Kaggle, create a Kaggle Secret called `DISCORD_WEBHOOK_URL` and paste the URL there.

---

## ☁️ How to Train on Kaggle (Offline Background Training)

1. Go to [Kaggle.com](https://www.kaggle.com/) and create a free account.
2. Click **Create** ➔ **New Notebook**.
3. In the top menu, click **File** ➔ **Import from GitHub**.
4. Paste your repository URL: `https://github.com/TristanJames5/apexprimal`
5. Open `colab_train.py` inside the Kaggle notebook.
6. Click **Save Version** (Top Right) ➔ Select **Save & Run All (Commit)**.

You can now completely shut down your laptop and go to sleep. Kaggle's servers will run the 10 million simulated trades in the background, and your Discord will light up with progress reports!
