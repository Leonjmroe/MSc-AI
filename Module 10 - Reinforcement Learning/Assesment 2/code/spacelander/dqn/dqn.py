import gymnasium as gym
import random, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import os

# Configuration parameters
ENV_NAME = "LunarLander-v3"
HIDDEN_DIM = 256
REPLAY_BUFFER_CAPACITY = 100000
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 500000
TARGET_UPDATE_FREQ = 1000
MIN_REPLAY_SIZE = 10000
EPISODES = 3000
MAX_TIMESTEPS = 1000
RENDER_MODE = None
MODEL_SAVE_PATH = "lunar_lander_dqn_mac.pt"
RESULTS_SAVE_PATH = "training_results.csv"
RESULTS_PLOT_PATH = "training_results.png"

device = torch.device("cpu")

# Environment
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

# Neural Network
class DQNNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x.float())

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, transition):
        self.buffer.append(transition)
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*samples))
    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, num_actions, hidden_dim=HIDDEN_DIM, batch_size=BATCH_SIZE, gamma=GAMMA, lr=LEARNING_RATE, 
                 buffer_size=REPLAY_BUFFER_CAPACITY, min_replay=MIN_REPLAY_SIZE, eps_start=EPSILON_START, 
                 eps_end=EPSILON_END, eps_decay=EPSILON_DECAY, target_update=TARGET_UPDATE_FREQ):
        self.num_actions = num_actions
        self.gamma, self.batch_size, self.min_replay = gamma, batch_size, min_replay
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.target_update = target_update
        self.online_net = DQNNetwork(state_dim, hidden_dim, num_actions).to(device)
        self.target_net = DQNNetwork(state_dim, hidden_dim, num_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.steps = 0
        # MPS doesn't support mixed precision and GradScaler
        self.use_mixed_precision = device.type == "cuda"
        if self.use_mixed_precision:
            from torch.amp import autocast, GradScaler
            self.scaler = GradScaler()

    def calc_epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * max(0, (self.eps_decay - self.steps)) / self.eps_decay

    def select_action(self, state):
        self.steps += 1
        if random.random() < self.calc_epsilon():
            return random.randrange(self.num_actions)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            return self.online_net(state).argmax().item()

    def train_step(self):
        if len(self.buffer) < self.min_replay:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.bool, device=device)
        actions = actions.unsqueeze(-1)
        rewards = rewards.unsqueeze(-1)
        dones = dones.float().unsqueeze(-1)

        if self.use_mixed_precision:
            from torch.amp import autocast
            with torch.no_grad(), autocast(device_type=device.type):
                targets = rewards + self.gamma * (1 - dones) * self.target_net(next_states).max(dim=1, keepdim=True)[0]
            
            with autocast(device_type=device.type):
                q_values = self.online_net(states).gather(1, actions)
                loss = F.mse_loss(q_values, targets)
                
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training path for CPU and MPS
            with torch.no_grad():
                targets = rewards + self.gamma * (1 - dones) * self.target_net(next_states).max(dim=1, keepdim=True)[0]
            
            q_values = self.online_net(states).gather(1, actions)
            loss = F.mse_loss(q_values, targets)
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

# Training Loop
def train(pretrained=None, episodes=EPISODES, max_steps=MAX_TIMESTEPS):
    agent = DQNAgent(state_dim, num_actions)
    
    if pretrained:
        agent.online_net.load_state_dict(torch.load(pretrained, map_location=device))
        agent.target_net.load_state_dict(agent.online_net.state_dict())

    # Warm-up
    obs, _ = env.reset()
    for _ in range(agent.min_replay):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.buffer.push((obs, action, reward, next_obs, done))
        obs = next_obs if not done else env.reset()[0]

    # Training
    data = []
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward, ep_steps, start = 0, 0, time.time()
        for _ in range(max_steps):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.buffer.push((obs, action, reward, next_obs, done))
            agent.train_step()
            obs, ep_reward, ep_steps = next_obs, ep_reward + reward, ep_steps + 1
            if done:
                break
        data.append([ep + 1, ep_reward, time.time() - start, ep_steps])
        print(f"Episode {ep+1}, Reward: {ep_reward}")

    env.close()
    return pd.DataFrame(data, columns=['episode', 'reward', 'time_seconds', 'steps']), agent

# Run
if __name__ == "__main__":
    try:
        print("Starting training...")
        data, agent = train()  
        
        # Save the model
        torch.save(agent.online_net.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")
        
        # Save training data to CSV
        data.to_csv(RESULTS_SAVE_PATH, index=False)
        print(f"Training data saved to {RESULTS_SAVE_PATH}")
        
        # Plot and save results
        plt.figure(figsize=(10, 6))
        plt.plot(data['episode'], data['reward'])
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress on Mac with MPS')
        plt.savefig(RESULTS_PLOT_PATH)
        plt.show()
        print("Training completed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()