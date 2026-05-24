from stable_baselines3 import PPO
from pro_env import ProNavEnv
import time

def main():
    # 1. Load the Environment
    env = ProNavEnv()
    
    # 2. Load the Trained AI Model
    print("Loading Trained Pro AI Model...")
    try:
        model = PPO.load("models/ppo_pro_navigation")
    except FileNotFoundError:
        print("Error: Pro Model not found! Please run 'python train_agent.py' first.")
        return

    print("Running Simulation. Observe LiDAR beams and wall avoidance!")
    
    obs, info = env.reset()
    
    for _ in range(3000):
        # AI prediction
        action, _states = model.predict(obs, deterministic=True)
        
        # Take action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Show pro graphics
        env.render()
        
        if terminated or truncated:
            print("Round Ended. Resetting environment...")
            time.sleep(1)
            obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()