from stable_baselines3 import PPO
from grid_env import GridNavEnv
import os

def main():
    os.makedirs("models", exist_ok=True)
    env = GridNavEnv()

    print("Initializing Grid-Based RL Agent (PPO)...")
    print("Training started! Please wait (~1 minute)...")
    
    # MLP Policy is perfect for grid coordinate observations
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0005)

    # Train for 80,000 steps to ensure it learns to navigate around the black blocks
    model.learn(total_timesteps=80000)

    model.save("models/ppo_grid_model")
    print("\n✅ Grid Training Complete & Model Saved!")
    
    env.close()

if __name__ == "__main__":
    main()