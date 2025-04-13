import gymnasium as gym
import argparse
import numpy as np
import torch
import os
from dqn import DQN
from duel_dqn import DuelingDQN

ENV_NAME = "LunarLander-v3"
NUM_EPISODES = 5
RENDER = True
device = torch.device("cpu")

def evaluate_model(model_path, arch='dqn'):
    env = gym.make(ENV_NAME, render_mode="human" if RENDER else None)
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    
    model = DuelingDQN(state_dim, 128, num_actions).to(device) if arch == 'duel' else DQN(state_dim, 128, num_actions).to(device)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    total_rewards = []
    
    for episode in range(NUM_EPISODES):
        state = env.reset()[0]
        done = False
        total_reward = 0
        
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            state = next_state
            total_reward += reward
        
        total_rewards.append(total_reward)
        print(f"Episode {episode+1}/{NUM_EPISODES}, Reward: {total_reward:.2f}")
    
    env.close()
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nEvaluation Results:")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min Reward: {np.min(total_rewards):.2f}")
    print(f"Max Reward: {np.max(total_rewards):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--arch", type=str, choices=["dqn", "duel"], default="dqn")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--no-render", action="store_true")
    
    args = parser.parse_args()
    
    NUM_EPISODES = args.episodes
    RENDER = not args.no_render
    evaluate_model(args.model_path, args.arch)