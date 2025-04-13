import gymnasium as gym
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt

# Config
ENV_NAME = "LunarLander-v3"
HIDDEN_DIM = 128
BUFFER_SIZE = 10000
BATCH_SIZE = 64
LR = 0.0001
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_FREQ = 1000
MIN_BUFFER = 1000
EPISODES = 3000
MAX_STEPS = 1000
SAVE_PATH = "duel_dqn.pt"
CSV_PATH = "duel_dqn_results.csv"
PLOT_PATH = "duel_dqn_results.png"

device = torch.device("cpu")
print(f"Device: {device}")

# Environment
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

# Network
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, hidden_dim, actions):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, actions)
        )
    
    def forward(self, x):
        x = x.float()
        f = self.feature(x)
        v, a = self.value(f), self.advantage(f)
        return v + (a - a.mean(dim=1, keepdim=True))

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        return map(np.array, zip(*random.sample(self.buffer, n)))
    
    def __len__(self):
        return len(self.buffer)

# Agent
class DuelingDQNAgent:
    def __init__(self, state_dim, actions, hidden_dim=HIDDEN_DIM, batch_size=BATCH_SIZE, gamma=GAMMA, lr=LR,
                 buffer_size=BUFFER_SIZE, min_buffer=MIN_BUFFER, eps_start=EPS_START,
                 eps_end=EPS_END, eps_decay=EPS_DECAY, target_freq=TARGET_FREQ):
        self.actions = actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer = min_buffer
        self.eps = eps_start # Added self.eps
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps = 0
        self.target_freq = target_freq

        self.online_net = DuelingDQN(state_dim, hidden_dim, actions).to(device)
        self.target_net = DuelingDQN(state_dim, hidden_dim, actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def act(self, state):
        self.steps += 1
        # Removed old epsilon calculation
        if random.random() < self.eps: # Use self.eps
            return random.randrange(self.actions)
        state_t = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.online_net(state_t).argmax().item()

    def train(self):
        if len(self.buffer) < self.min_buffer:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states_t = torch.tensor(states, device=device, dtype=torch.float32)
        actions_t = torch.tensor(actions, device=device, dtype=torch.long).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32).unsqueeze(-1)
        next_states_t = torch.tensor(next_states, device=device, dtype=torch.float32)
        dones_t = torch.tensor(dones, device=device, dtype=torch.float32).unsqueeze(-1)

        with torch.no_grad():
            targets = rewards_t + self.gamma * (1 - dones_t) * self.target_net(next_states_t).max(dim=1, keepdim=True)[0]

        q_values = self.online_net(states_t)
        action_q = torch.gather(q_values, dim=1, index=actions_t)
        loss = F.mse_loss(action_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Added epsilon decay update
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

        if self.steps % self.target_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())


# Training
def train_duel_dqn(episodes=EPISODES, max_steps=MAX_STEPS, pretrained=None):
    agent = DuelingDQNAgent(state_dim, num_actions)
    
    if pretrained:
        state_dict = torch.load(pretrained, map_location=device)
        agent.online_net.load_state_dict(state_dict)
        agent.target_net.load_state_dict(state_dict)
    
    # Warm-up
    obs = env.reset()[0]
    for _ in range(agent.min_buffer):
        action = env.action_space.sample()
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        agent.buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs if not done else env.reset()[0]
    
    # Main loop
    data = []
    for ep in range(episodes):
        obs = env.reset()[0]
        reward, steps, start = 0, 0, time.time()
        
        for _ in range(max_steps):
            action = agent.act(obs)
            next_obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.buffer.push(obs, action, r, next_obs, done)
            agent.train()
            obs, reward, steps = next_obs, reward + r, steps + 1
            if done:
                break
        
        data.append([ep + 1, reward, time.time() - start, steps])
        print(f"Episode {ep+1}, Reward: {reward:.2f}, Time: {data[-1][2]:.2f}s")
    
    env.close()
    return pd.DataFrame(data, columns=['episode', 'reward', 'time', 'steps']), agent

if __name__ == "__main__":
    print("Starting Dueling DQN training...")
    data, agent = train_duel_dqn()

    torch.save(agent.online_net.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")
    
    data.to_csv(CSV_PATH, index=False)
    print(f"Data saved to {CSV_PATH}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(data['episode'], data['reward'])
    plt.title('Dueling DQN Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(PLOT_PATH)
    print(f"Plot saved to {PLOT_PATH}")