# 🚀 Lunar Lander – Reinforcement Learning Agent (Gymnasium)

This project implements a **Deep Q-Network (DQN)** agent to solve the classic [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) environment from **Gymnasium**.  
The goal is to train an agent that learns to control the lander to safely reach the landing pad using reinforcement learning.

Video from repository: https://youtu.be/GvOSRJf_hhw
---

## 🧠 Overview

- **Environment:** `LunarLander-v3` (Gymnasium)
- **Algorithm:** Deep Q-Network (DQN)
- **Frameworks:** TensorFlow, NumPy, Matplotlib
- **Render Mode:** `rgb_array` for video previews

The agent observes the lander’s state (position, velocity, angle, leg contact, etc.) and learns to perform one of four possible discrete actions to maximize total reward.

---

## 📁 Project Structure 
```bash
lunar-lander/
│
├── models/ # Trained models (ready to use)
│ └── model_max.h5
│
├── videos/
│ ├── model_1/
│ │ ├── eval-episode-1.mp4
│ │ └── eval-episode-2.mp4
│ │ └── eval-episode-3.mp4
│
├── utils.py # Helper functions (get_action, replay buffer, etc.)
└── README.md
```
---

## 🎥 Preview

Here’s an example of the trained agent in action 👇  

<video src="videos/model_1/eval-episode-2.mp4" controls width="480"></video>

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/lunar-lander.git
cd lunar-lander
python -m venv venv
venv\Scripts\activate     # on Windows
# source venv/bin/activate  # on macOS/Linux
pip install -r requirements.txt
load model_max with tensorflow
use show_ladning func to display landing in notebook
```
## 🪳 Bugs
Model was trained on numpy 1.26.4, but when installing "gymnasium[other]" needed to record Lunar Lander it automatically
changes numpy to 2.24 which causes tensorflow to throw an error
