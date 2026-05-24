from stable_baselines3 import PPO
from grid_env import GridNavEnv
import time

def main():
    env = GridNavEnv()
    
    print("Loading Trained Grid AI Model...")
    try:
        model = PPO.load("models/ppo_grid_model")
    except FileNotFoundError:
        print("Error: Model not found! Run 'python train_grid.py' first.")
        return

    print("Running Simulation. Watch the Grid Pygame window!")
    
    obs, info = env.reset()
    
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        
        if terminated or truncated:
            print("Round Ended. Resetting...")
            time.sleep(1)
            obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()