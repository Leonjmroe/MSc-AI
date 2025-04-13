import gymnasium as gym
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from collections import deque

# Configuration parameters
ENV_NAME = "LunarLander-v3"
HIDDEN_DIM = 256
REPLAY_BUFFER_CAPACITY = 100000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 500000
TARGET_UPDATE_FREQ = 1000
MIN_REPLAY_SIZE = 10000
NUM_EPISODES = 6000
MAX_TIMESTEPS = 1000
RENDER_MODE = None
MODEL_SAVE_PATH = "lunar_lander_dueling_dqn_mac.pt"
RESULTS_SAVE_PATH = "dueling_dqn_results.csv"

# Device selection with macOS support
# device = torch.device("mps" if torch.backends.mps.is_available() else 
#                      "cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

print(f"Using device: {device}")

# Environment Setup
env = gym.make(ENV_NAME, render_mode=RENDER_MODE)
preprocess_state = lambda x: x  # No preprocessing needed

# Dueling DQN Network
class DuelingDQNNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions):
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, x):
        x = x.float()
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), actions, rewards, np.array(next_states), dones
    
    def __len__(self):
        return len(self.buffer)

# Dueling DQN Agent
class DuelingDQNAgent:
    def __init__(self, state_dim, num_actions, hidden_dim=HIDDEN_DIM, batch_size=BATCH_SIZE, gamma=GAMMA, lr=LEARNING_RATE,
                 replay_buffer_size=REPLAY_BUFFER_CAPACITY, min_replay_size=MIN_REPLAY_SIZE, eps_start=EPSILON_START, 
                 eps_end=EPSILON_END, eps_decay=EPSILON_DECAY, target_update_freq=TARGET_UPDATE_FREQ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0
        self.target_update_freq = target_update_freq
        
        self.online_net = DuelingDQNNetwork(state_dim, hidden_dim, num_actions).to(device)
        self.target_net = DuelingDQNNetwork(state_dim, hidden_dim, num_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        
        # Only use GradScaler with CUDA, not with CPU or MPS
        self.use_mixed_precision = device.type == "cuda"
        if self.use_mixed_precision:
            from torch.amp import GradScaler
            self.scaler = GradScaler()
            
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
    
    def select_action(self, state):
        self.steps_done += 1
        eps = max(self.eps_end, self.eps * (1 - self.steps_done / self.eps_decay))
        if random.random() < eps:
            return random.randrange(self.num_actions)
        # Convert state to numpy array first to avoid warning
        state_np = np.array(state, dtype=np.float32)
        state_t = torch.tensor(state_np, device=device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.online_net(state_t).argmax().item()
    
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
                    next_q_values = self.target_net(next_states_t)
                    targets = rewards_t + self.gamma * (1 - dones_t) * next_q_values.max(dim=1, keepdim=True)[0]
            
            with autocast():
                q_values = self.online_net(states_t)
                action_q_values = torch.gather(q_values, dim=1, index=actions_t)
                loss = nn.functional.mse_loss(action_q_values, targets)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training path for CPU and MPS
            with torch.no_grad():
                next_q_values = self.target_net(next_states_t)
                targets = rewards_t + self.gamma * (1 - dones_t) * next_q_values.max(dim=1, keepdim=True)[0]
            
            q_values = self.online_net(states_t)
            action_q_values = torch.gather(q_values, dim=1, index=actions_t)
            loss = nn.functional.mse_loss(action_q_values, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

# Training Loop
def run_training_dueling_dqn(pretrained=None, num_episodes=NUM_EPISODES, max_timesteps=MAX_TIMESTEPS):
    obs, info = env.reset()
    state_dim = len(preprocess_state(obs))
    num_actions = env.action_space.n
    agent = DuelingDQNAgent(state_dim, num_actions)
    
    if pretrained:
        state_dict = torch.load(pretrained, map_location=device)
        agent.online_net.load_state_dict(state_dict)
        agent.target_net.load_state_dict(state_dict)
    
    # Warm-up
    obs, info = env.reset()
    obs = preprocess_state(obs)
    for _ in range(agent.min_replay_size):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_obs = preprocess_state(next_obs)
        agent.replay_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            obs, info = env.reset()
            obs = preprocess_state(obs)
    
    # Training
    episode_rewards, episode_steps, episode_times = [], [], []
    for episode in range(num_episodes):
        obs, info = env.reset()
        obs = preprocess_state(obs)
        episode_reward, episode_step = 0, 0
        start_time = time.time()
        
        for _ in range(max_timesteps):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_obs = preprocess_state(next_obs)
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            agent.train_step()
            obs = next_obs
            episode_reward += reward
            episode_step += 1
            if done:
                break
        
        episode_time = time.time() - start_time
        episode_steps.append(episode_step)
        episode_rewards.append(episode_reward)
        episode_times.append(episode_time)
        print(f"Episode {episode+1}, Reward: {episode_reward:.2f}, Time: {episode_time:.2f}s")
    
    env.close()
    return pd.DataFrame({
        'episode': np.arange(1, num_episodes + 1),
        'reward': episode_rewards,
        'steps': episode_steps,
        'time': episode_times
    }), agent

# Add main execution block
if __name__ == "__main__":
    print("Starting Dueling DQN training...")
    data, agent = run_training_dueling_dqn(num_episodes=NUM_EPISODES)
    print(data.describe())
    
    # Save model and results
    torch.save(agent.online_net.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # Save results to CSV
    data.to_csv(RESULTS_SAVE_PATH, index=False)
    print(f"Results saved to {RESULTS_SAVE_PATH}")