# 🤖 AI-Based Autonomous Navigation System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-RL_Environment-orange.svg)
![Stable_Baselines3](https://img.shields.io/badge/Stable_Baselines3-PPO-brightgreen.svg)
![Pygame](https://img.shields.io/badge/Pygame-Simulation-yellow.svg)

An industry-grade simulation of an Autonomous Mobile Robot (AMR) navigation system. This project demonstrates how an AI agent learns to navigate dynamic environments, avoid obstacles, and reach targets using **Deep Reinforcement Learning (Proximal Policy Optimization - PPO)**. 

To demonstrate versatility, this project includes **two distinct simulation environments**:
1. **Pro-Level Continuous Environment:** Features LiDAR sensor simulation and dynamic path planning.
2. **Discrete Grid Environment:** A classic A*-style grid world where the agent learns coordinate-based navigation.

---

## 🎥 Project Demos

### 1. Continuous Environment (with Simulated LiDAR)

https://github.com/user-attachments/assets/74ed8af6-1eb8-4817-9ea0-24c1931667b9




> The robot uses 5 ray-casted sensor beams to detect walls and safely navigate to the goal while displaying real-time telemetry.

### 2. Discrete Grid Environment


https://github.com/user-attachments/assets/8ff849cd-b457-48a8-bca9-fcda73b73676


> The agent learns to navigate a 15x15 grid, intelligently avoiding static black obstacles to reach the target block.

---

## 🚀 Key Features & Architecture
* **🧠 Reinforcement Learning Brain:** Powered by the **PPO** algorithm via `Stable-Baselines3`.
* **📡 Simulated LiDAR Sensors:** (In Pro version) The robot utilizes sensor beams to detect walls in real-time.
* **🌐 Custom Gym Environments:** Built fully custom `gymnasium` environments from scratch to bridge Pygame physics with the AI learning model.
* **📊 Telemetry Dashboard:** Real-time HUD displaying distance to goal, current reward, and sensor status.
* **🛤️ Dynamic Path Planning:** The AI adapts dynamically without pre-programmed paths, mimicking real-world autonomous behavior.

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** Stable-Baselines3 (PPO Algorithm)
* **Environment Design:** Gymnasium
* **Physics & Visualization:** Pygame-CE
* **Math Operations:** NumPy, Math

---

## ⚙️ How to Run Locally

### Step 1: Setup Environment
```bash
# Clone the repository
git clone https://github.com/dalimkumar452-sudo/AI-Autonomous-Navigation-System.git
cd AI-Autonomous-Navigation-System

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # On Windows
source venv/bin/activate # On Mac/Linux

# Install dependencies
pip install -r requirements.txt
Step 2: Choose Your Simulation
Option A: Run the Pro-Level Continuous Environment
Bash
# Train the model (~2 mins)
python train_agent.py

# Watch the simulation
python run_pro_agent.py
Option B: Run the Discrete Grid Environment
Bash
# Train the grid model (~1 min)
python train_grid.py

# Watch the simulation
python run_grid.py
🧠 Learning Strategy (Reward Function)
The PPO agent learns through a carefully balanced reward system:

Goal Reached: +100 points.

Wall Collision (Screen Bounds / Static Obstacles): -50 points (Termination).

Time Penalty: Negative reward per step (Encourages finding the shortest path).

Proximity Warning: Penalty applied if LiDAR sensors detect obstacles closer than safety thresholds.

👨‍💻 Author
Dalim Kumar * GitHub: https://github.com/dalimkumar452-sudo

( https://www.linkedin.com/in/dalim-kumar-612038402 )

This project was built as a proof-of-work for implementing Deep Reinforcement Learning in robotic navigation systems.
