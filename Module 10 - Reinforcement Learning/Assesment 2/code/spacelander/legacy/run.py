import os
import glob
import gymnasium as gym
import torch
from stable_baselines3 import PPO
import numpy as np
import time

def find_latest_model(models_dir="models"):
    """Find the most recent .pth model file in the models directory."""
    try:
        # Get all .pth files in the models directory
        model_files = glob.glob(os.path.join(models_dir, "*.pth"))
        if not model_files:
            print(f"No .pth model files found in {models_dir} directory")
            return None
            
        # Sort by modification time (most recent last)
        latest_model = max(model_files, key=os.path.getmtime)
        print(f"Found latest model: {os.path.basename(latest_model)}")
        return latest_model
    except Exception as e:
        print(f"Error finding latest model: {e}")
        return None

def load_model(model_path):
    """Load a pre-trained model from a .pth file."""
    try:
        # Create environment to get observation and action space dimensions
        env = gym.make("LunarLander-v3")
        
        # Create a policy with the same architecture as during training
        policy = PPO("MlpPolicy", env, verbose=0).policy
        
        # Load the state dictionary from the .pth file
        state_dict = torch.load(model_path)
        policy.load_state_dict(state_dict)
        
        print(f"Successfully loaded model from {model_path}")
        return policy
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def run_model(policy, episodes=5, render=True):
    """Run the loaded model in the environment for a specified number of episodes."""
    # Create environment with rendering if specified
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    
    total_rewards = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        print(f"\nStarting Episode {episode + 1}/{episodes}...")
        
        while not done:
            # Get action from policy
            with torch.no_grad():
                action, _ = policy.predict(obs)
            
            # Take step in environment
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            # Add a small delay for better visualization
            if render:
                time.sleep(0.01)
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1} finished with reward: {episode_reward:.2f}")
    
    env.close()
    
    print(f"\nSummary of {episodes} episodes:")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Min Reward: {min(total_rewards):.2f}")
    print(f"Max Reward: {max(total_rewards):.2f}")

def main():
    print("Spacelander Model Runner")
    print("=======================")
    
    # Find the latest model file
    latest_model = find_latest_model()
    if not latest_model:
        print("Could not find a valid model file. Exiting.")
        return
    
    # Load the model
    policy = load_model(latest_model)
    if not policy:
        print("Failed to load model. Exiting.")
        return
    
    # Ask user for configuration
    try:
        episodes = int(input("Enter number of episodes to run (default: 5): ") or 5)
        render_input = input("Enable rendering? (Y/n): ").lower() or 'y'
        render = render_input.startswith('y')
    except ValueError:
        print("Invalid input, using default values")
        episodes = 5
        render = True
    
    # Run the model
    print(f"\nRunning model for {episodes} episodes with rendering {'enabled' if render else 'disabled'}")
    run_model(policy, episodes, render)

if __name__ == "__main__":
    main() 