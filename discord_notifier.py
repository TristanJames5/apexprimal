import requests
import json
import os

def send_training_update(step, total_steps, current_reward, win_rate):
    """
    Sends a rich embed message to Discord to update you on Kaggle/Colab training progress.
    """
    # In Colab/Kaggle, you will paste your actual webhook URL here or in the environment variables
    WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1553775394887303240/QY3e3vfF1qzSCtKdvmpyU69LVUy_yajqcR1uSBjb2kfeYeYoltlvpBxe1iF9aAEQfHLP")
    
    if WEBHOOK_URL == "YOUR_DISCORD_WEBHOOK_URL_HERE":
        print("Discord Webhook not set up yet. Skipping notification.")
        return

    progress = (step / total_steps) * 100
    
    embed = {
        "title": "🧠 Project Apex: Training Update",
        "description": f"The AI is actively learning in the Sandbox. \n**Progress:** {progress:.2f}%",
        "color": 3447003, # Blue color
        "fields": [
            {
                "name": "Current Step",
                "value": f"{step:,} / {total_steps:,}",
                "inline": True
            },
            {
                "name": "Avg Reward (Score)",
                "value": f"${current_reward:,.2f}",
                "inline": True
            },
            {
                "name": "Est. Win Rate",
                "value": f"{win_rate:.1f}%",
                "inline": True
            }
        ],
        "footer": {
            "text": "Apex Primal Engine • Deep Reinforcement Learning"
        }
    }

    payload = {
        "embeds": [embed]
    }
    
    try:
        response = requests.post(WEBHOOK_URL, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
        if response.status_code != 204:
            print(f"Failed to send Discord message: {response.status_code}")
    except Exception as e:
        print(f"Error sending Discord update: {e}")
