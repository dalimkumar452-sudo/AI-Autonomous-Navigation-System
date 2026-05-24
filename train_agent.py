from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from pro_env import ProNavEnv
import os

def main():
    # 1. Create directory to save the trained model
    os.makedirs("models", exist_ok=True)

    # 2. Initialize the Custom Pro Environment
    env = ProNavEnv()

    print("Verifying Environment integrity...")
    check_env(env, warn=True)

    print("Initializing Advanced RL Agent (PPO)...")
    print("This requires more training time due to obstacles. Wait ~2 minutes...")
    
    # 3. Define smarter PPO Model with Custom Policy network architecture
    # PPO learns faster than DQN for continuous navigation tasks.
    policy_kwargs = dict(net_arch=[dict(pi=[128, 64], vf=[128, 64])]) # Pro level network setting
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, policy_kwargs=policy_kwargs)

    # 4. Train the model for 100,000 steps (essential for obstacle learning)
    model.learn(total_timesteps=100000)

    # 5. Save the brain
    model.save("models/ppo_pro_navigation")
    print("\n✅ Advanced Training Complete & Pro Model Saved Successfully in 'models' folder!")
    
    env.close()

if __name__ == "__main__":
    main()