import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import ale_py

from collections import deque
import matplotlib.pyplot as plt

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



ENV_NAME = "ALE/MsPacman-v5"  # Adjust if your local environment differs

env = gym.make(ENV_NAME, render_mode=None)  # or "human" if you want to see it
print("Action Space:", env.action_space)
print("Observation Space:", env.observation_space)

def preprocess_state(state):
    """
    Convert state to grayscale, resize, or otherwise transform if needed.
    Currently returns the original frame. 
    Implement your own transformations here if desired.
    """
    return state

# ---
# SECTION 2: NEURAL NETWORK & REPLAY BUFFER
# ---

class DQNNetwork(nn.Module):
    """
    A simple CNN for Ms. Pac-Man (raw frames).
    Adjust architecture as needed for performance.
    """
    def __init__(self, input_shape, num_actions):
        super(DQNNetwork, self).__init__()
        c, h, w = input_shape  # e.g. (3, 210, 160)

        self.conv1 = nn.Conv2d(c, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)

        # Dummy forward pass to calculate conv_out_size
        dummy = torch.zeros(1, c, h, w)
        dummy = self.conv1(dummy)
        dummy = self.conv2(dummy)
        conv_out_size = dummy.numel()

        self.fc1 = nn.Linear(conv_out_size, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Ensure contiguity before flattening:
        x = x.contiguous()
        x = x.view(x.size(0), -1)  # or x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    """A simple FIFO experience replay memory for transitions."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return np.array(states), actions, rewards, np.array(next_states), dones
    
    def __len__(self):
        return len(self.buffer)

# ---
# SECTION 3: DQN AGENT
# ---

class DQNAgent:
    def __init__(self, state_shape, num_actions,
                 batch_size=32,
                 gamma=0.99,
                 lr=1e-4,
                 replay_buffer_size=100000,
                 min_replay_size=10000,
                 eps_start=1.0,
                 eps_end=0.01,
                 eps_decay=1e6,
                 target_update_freq=1000):
        
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update_freq = target_update_freq
        
        # Initialize networks
        self.online_net = DQNNetwork(state_shape, num_actions).to(device)
        self.target_net = DQNNetwork(state_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        
        self.steps_done = 0  # for epsilon decay

        # Create a GradScaler for AMP
        self.scaler = GradScaler()  # <-- THIS is the key line
    
    def calc_epsilon(self):
        """
        Decay epsilon linearly from eps_start to eps_end over eps_decay steps.
        """
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
            max(0, (self.eps_decay - self.steps_done)) / self.eps_decay
        return epsilon
    
    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        self.steps_done += 1
        epsilon = self.calc_epsilon()
        
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                # Permute to (batch_size=1, channels, height, width)
                state_t = torch.tensor(state, device=device).unsqueeze(0).permute(0, 3, 1, 2)
                q_values = self.online_net(state_t)
                action = q_values.argmax(dim=1).item()
            return action
    
    def train_step(self):
        """Sample a batch from replay buffer and perform one training step with AMP."""
        if len(self.replay_buffer) < self.min_replay_size:
            return
        
        # sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # move to GPU, shape [batch, channels, height, width]
        states_t = torch.tensor(states, device=device).permute(0, 3, 1, 2)
        actions_t = torch.tensor(actions, device=device, dtype=torch.long).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32).unsqueeze(-1)
        next_states_t = torch.tensor(next_states, device=device).permute(0, 3, 1, 2)
        dones_t = torch.tensor(dones, device=device, dtype=torch.float32).unsqueeze(-1)

        # compute target Q-values
        with torch.no_grad():
            with autocast("cuda"):  # for half-precision
                next_q_values = self.target_net(next_states_t)
                max_next_q_values = next_q_values.max(dim=1, keepdim=True)[0]
                targets = rewards_t + self.gamma * (1 - dones_t) * max_next_q_values

        # forward pass in mixed precision
        with autocast("cuda"):
            q_values = self.online_net(states_t)
            action_q_values = torch.gather(q_values, dim=1, index=actions_t)
            loss = F.mse_loss(action_q_values, targets)

        # backprop with GradScaler
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # update target net periodically
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

# ---
# SECTION 4: TRAINING LOOP
# ---

def run_training(pretrained = None, num_episodes=1000, max_timesteps=10000):
    """
    Main training loop using the new Gym API.
    """
    # 1) Reset once to inspect shape
    obs, info = env.reset()
    obs = preprocess_state(obs)
    
    # Some Atari frames are shape (210, 160, 3). Let’s confirm:
    print("Initial observation shape:", obs.shape)
    c = obs.shape[2]    # channels
    h, w = obs.shape[0], obs.shape[1]
    state_shape = (c, h, w)
    
    num_actions = env.action_space.n
    
    # 2) Create DQN Agent
    agent = DQNAgent(state_shape, num_actions)

    # 3) If we have a pretrained path, load the weights into agent's networks
    if pretrained is not None:
        # For example, 'pretrained' might be "mspacman_dqn.pth"
        print(f"Loading pretrained model from {pretrained} ...")
        state_dict = torch.load(pretrained)
        
        # You likely want to load into the agent's ONLINE network
        agent.online_net.load_state_dict(state_dict)
        # Then copy those weights to the TARGET network
        agent.target_net.load_state_dict(agent.online_net.state_dict())
        print("Pretrained weights loaded successfully!")
    
    # 3) Fill replay buffer with random actions (warm-up phase)
    print("Filling replay buffer with random transitions...")
    obs, info = env.reset()
    obs = preprocess_state(obs)
    
    for _ in range(agent.min_replay_size):
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        next_obs = preprocess_state(next_obs)
        
        # Combine done and truncated to match old-style "done"
        done_ = done or truncated
        agent.replay_buffer.push(obs, action, reward, next_obs, done_)
        
        obs = next_obs
        if done_:
            obs, info = env.reset()
            obs = preprocess_state(obs)
    
    # 4) Training episodes
    episode_rewards = []
    
    for episode in range(num_episodes):
        print(episode + 1, end="")
        obs, info = env.reset()
        obs = preprocess_state(obs)
        episode_reward = 0
        
        for t in range(max_timesteps):
            action = agent.select_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            next_obs = preprocess_state(next_obs)
            done_ = done or truncated
            
            agent.replay_buffer.push(obs, action, reward, next_obs, done_)
            agent.train_step()
            
            obs = next_obs
            episode_reward += reward
            
            if done_:
                break
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1}, Reward: {episode_reward}")
    
    env.close()
    return episode_rewards, agent


