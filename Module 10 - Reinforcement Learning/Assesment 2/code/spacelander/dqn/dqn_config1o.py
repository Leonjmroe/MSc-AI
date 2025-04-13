import gym, random, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.amp import autocast, GradScaler
from collections import deque
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment
env = gym.make("LunarLander-v2")
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
    def __init__(self, state_dim, num_actions, hidden_dim=256, batch_size=32, gamma=0.99, lr=1e-4, 
                 buffer_size=100000, min_replay=10000, eps_start=1.0, eps_end=0.01, eps_decay=500000, target_update=1000):
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
        self.steps, self.scaler = 0, GradScaler()

    def calc_epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * max(0, (self.eps_decay - self.steps)) / self.eps_decay

    def select_action(self, state):
        self.steps += 1
        if random.random() < self.calc_epsilon():
            return random.randrange(self.num_actions)
        with torch.no_grad():
            state = torch.tensor(state, device=device).unsqueeze(0)
            return self.online_net(state).argmax().item()

    def train_step(self):
        if len(self.buffer) < self.min_replay:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.tensor(states, device=device)
        actions = torch.tensor(actions, device=device)
        rewards = torch.tensor(rewards, device=device)
        next_states = torch.tensor(next_states, device=device)
        dones = torch.tensor(dones, dtype=torch.bool, device=device)
        actions, rewards, dones = actions.long().unsqueeze(-1), rewards.float().unsqueeze(-1), dones.float().unsqueeze(-1)

        with torch.no_grad(), autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            targets = rewards + self.gamma * (1 - dones) * self.target_net(next_states).max(dim=1, keepdim=True)[0]

        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            q_values = self.online_net(states).gather(1, actions)
            loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

# Training Loop
def train(pretrained=None, episodes=1000, max_steps=1000):
    agent = DQNAgent(state_dim, num_actions)
    
    if pretrained:
        agent.online_net.load_state_dict(torch.load(pretrained, map_location=device))
        agent.target_net.load_state_dict(agent.online_net.state_dict())

    # Warm-up
    obs, _ = env.reset()
    for _ in range(agent.min_replay):
        action = env.action_space.sample()
        next_obs, reward, done, truncated, _ = env.step(action)
        agent.buffer.push((obs, action, reward, next_obs, done or truncated))
        obs = next_obs if not (done or truncated) else env.reset()[0]

    # Training
    data = []
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward, ep_steps, start = 0, 0, time.time()
        for _ in range(max_steps):
            action = agent.select_action(obs)
            next_obs, reward, done, truncated, _ = env.step(action)
            agent.buffer.push((obs, action, reward, next_obs, done or truncated))
            agent.train_step()
            obs, ep_reward, ep_steps = next_obs, ep_reward + reward, ep_steps + 1
            if done or truncated:
                break
        data.append([ep + 1, ep_reward, time.time() - start, ep_steps])
        print(f"Episode {ep+1}, Reward: {ep_reward}")

    env.close()
    return pd.DataFrame(data, columns=['episode', 'reward', 'time_seconds', 'steps']), agent

# Run
if __name__ == "__main__":
    data, agent = train()
    plt.plot(data['episode'], data['reward'])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()