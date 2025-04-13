import gymnasium as gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import pandas as pd
import time

# Configuration parameters
ENV_NAME = "LunarLander-v3"
HIDDEN_DIM = 128
REPLAY_BUFFER_CAPACITY = 10000
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 1000
MIN_REPLAY_SIZE = 1000
NUM_EPISODES = 3000
MAX_TIMESTEPS = 1000
RENDER_MODE = None
MODEL_SAVE_PATH = "lunar_lander_double_dqn_mac.pt"
RESULTS_SAVE_PATH = "double_dqn_results.csv"

# Device configuration
device = torch.device("cpu")

# Environment
env = gym.make(ENV_NAME, render_mode=RENDER_MODE)

# Neural Network
class DQNNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, x):
        return self.net(x.float())

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(args)
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*samples))
    
    def __len__(self):
        return len(self.buffer)

# Double DQN Agent
class DoubleDQNAgent:
    def __init__(self, state_dim, num_actions, hidden_dim=HIDDEN_DIM, capacity=REPLAY_BUFFER_CAPACITY, 
                 batch_size=BATCH_SIZE, lr=LEARNING_RATE, gamma=GAMMA, epsilon_start=EPSILON_START, 
                 epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY, target_update_freq=TARGET_UPDATE_FREQ, 
                 min_replay_size=MIN_REPLAY_SIZE):
        self.online_net = DQNNetwork(state_dim, hidden_dim, num_actions).to(device)
        self.target_net = DQNNetwork(state_dim, hidden_dim, num_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.min_replay_size = min_replay_size
        self.steps_done = 0
        self.num_actions = num_actions
        
        # Only use GradScaler with CUDA, not with CPU or MPS
        self.use_mixed_precision = device.type == "cuda"
        if self.use_mixed_precision:
            from torch.amp import autocast, GradScaler
            self.scaler = GradScaler()
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.online_net(state).argmax().item()
    
    def train_step(self):
        if len(self.replay_buffer) < self.min_replay_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states_t = torch.tensor(states, device=device, dtype=torch.float32)
        actions_t = torch.tensor(actions, device=device, dtype=torch.long).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32).unsqueeze(-1)
        next_states_t = torch.tensor(next_states, device=device, dtype=torch.float32)
        dones_t = torch.tensor(dones, device=device, dtype=torch.float32).unsqueeze(-1)
        
        if self.use_mixed_precision:
            from torch.amp import autocast
            # Mixed precision training (CUDA only)
            with torch.no_grad():
                with autocast():
                    next_actions = self.online_net(next_states_t).argmax(dim=1, keepdim=True)
                    targets = rewards_t + self.gamma * (1 - dones_t) * \
                          self.target_net(next_states_t).gather(1, next_actions)
            
            with autocast():
                q_values = self.online_net(states_t).gather(1, actions_t)
                loss = F.mse_loss(q_values, targets)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training path for CPU and MPS
            with torch.no_grad():
                next_actions = self.online_net(next_states_t).argmax(dim=1, keepdim=True)
                targets = rewards_t + self.gamma * (1 - dones_t) * \
                      self.target_net(next_states_t).gather(1, next_actions)
            
            q_values = self.online_net(states_t).gather(1, actions_t)
            loss = F.mse_loss(q_values, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

# Training Loop
def run_training_double_dqn(pretrained=None, num_episodes=NUM_EPISODES, max_timesteps=MAX_TIMESTEPS):
    obs, info = env.reset()
    state_dim = len(obs)
    num_actions = env.action_space.n
    agent = DoubleDQNAgent(state_dim, num_actions)
    
    if pretrained:
        agent.online_net.load_state_dict(torch.load(pretrained, map_location=device))
        agent.target_net.load_state_dict(agent.online_net.state_dict())
    
    # Warm-up
    obs, info = env.reset()
    for _ in range(agent.min_replay_size):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.replay_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            obs, info = env.reset()
    
    # Training
    episode_rewards = []
    episode_steps = []
    elapsed_times = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_step = 0
        start_time = time.time()
        
        for t in range(max_timesteps):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            agent.train_step()
            obs = next_obs
            episode_reward += reward
            episode_step += 1
            if done:
                break
        
        elapsed_time = time.time() - start_time
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)
        elapsed_times.append(elapsed_time)
        
        print(f"Episode {episode+1}, Reward: {episode_reward:.2f}, Time: {elapsed_time:.2f}s")
    
    env.close()
    return pd.DataFrame({
        'episode': np.arange(1, num_episodes + 1),
        'reward': episode_rewards,
        'time_seconds': elapsed_times,
        'steps': episode_steps
    }), agent

if __name__ == "__main__":
    print("Starting Double DQN training...")
    data, agent = run_training_double_dqn()
    print(data.describe())
    
    # Save model
    torch.save(agent.online_net.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # Save results to CSV
    data.to_csv(RESULTS_SAVE_PATH, index=False)
    print(f"Results saved to {RESULTS_SAVE_PATH}")