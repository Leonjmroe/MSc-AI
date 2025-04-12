import gymnasium as gym
import ale_py
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.logger import configure
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os
import time
import logging
import datetime
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

ENV_NAME = "ALE/MsPacman-v5"
SHOW_EMULATOR = False
FRAME_SKIP = 4
SCREEN_SIZE = 84
NUM_SUB_AGENTS = 150
SUB_AGENT_BUFFER_SIZE = 100000
SUB_AGENT_LEARNING_RATE = 1e-4
SUB_AGENT_EXPLORATION_FRACTION = 0.1
SUB_AGENT_EXPLORATION_FINAL_EPS = 0.02
TOP_AGENT_BUFFER_SIZE = 200000
TOP_AGENT_LEARNING_RATE = 1e-4
TOP_AGENT_EXPLORATION_FRACTION = 0.1
TOP_AGENT_EXPLORATION_FINAL_EPS = 0.02
FEATURES_DIM = 512
LOG_DIR = "logs/"
CHECKPOINT_DIR = "checkpoints/"
SUB_AGENT_EPISODES = 100
ESTIMATED_STEPS_PER_EPISODE_SUB = 1000
TOP_AGENT_EPISODES = 500
ESTIMATED_STEPS_PER_EPISODE_TOP = 2000

for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s - %(message)s', handlers=[logging.FileHandler("mspacman_training.log", mode='w'), logging.StreamHandler()])
logger = logging.getLogger("mspacman_training")

device = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"
logger.info(f"Using device: {device}")

def create_env(env_id, seed=0, frame_skip=FRAME_SKIP): 
    env = AtariWrapper(gym.make(env_id), frame_skip=frame_skip, screen_size=SCREEN_SIZE)
    return env, env.action_space.n

_temp_env, SUB_AGENT_ACTION_SPACE = create_env(ENV_NAME)
_temp_env.close()

tasks = [(f"pellet_{i}", lambda info: 1 if np.random.random() < 0.05 else 0) for i in range(NUM_SUB_AGENTS // 2)] + \
        [(f"avoid_ghost_{i}", lambda info: np.random.random() * 0.01) for i in range(NUM_SUB_AGENTS // 2)]
NUM_SUB_AGENTS = len(tasks)

class CollectEpisodeRewardsCallback(BaseCallback):
    def __init__(self, target_episodes, agent_name="Agent", verbose=1):
        super().__init__(verbose)
        self.target_episodes, self.agent_name, self.episode_rewards, self.episode_lengths, self._episodes_completed, self.start_time = target_episodes, agent_name, [], [], 0, time.time()

    def _on_step(self):
        for idx, done in enumerate(self.locals["dones"]):
            if done and "episode" in self.locals["infos"][idx]:
                ep_info = self.locals["infos"][idx]["episode"]
                self.episode_rewards.append(ep_info["r"])
                self.episode_lengths.append(ep_info["l"])
                self._episodes_completed += 1
                if self.verbose > 0: logger.info(f"{self.agent_name} - Episode {self._episodes_completed}/{self.target_episodes}: Reward: {ep_info['r']:.2f}, Avg (10): {np.mean(self.episode_rewards[-10:]):.2f}")
                if self._episodes_completed >= self.target_episodes: return False
        return True

class UpdatePrefsCallback(CollectEpisodeRewardsCallback):
    def __init__(self, sub_agents, top_model, target_episodes, agent_name="TopAgent", verbose=1):
        super().__init__(target_episodes, agent_name, verbose)
        self.sub_agents, self.top_model, self.flat_prefs_np, self._initial_step_done = sub_agents, top_model, np.zeros(NUM_SUB_AGENTS * SUB_AGENT_ACTION_SPACE, dtype=np.float32), False

    def _on_step(self):
        try:
            current_obs = self.locals["new_obs"][0]
            self._initial_step_done = True
        except KeyError:
            if not self._initial_step_done:
                try: current_obs = self.training_env.buf_obs[0]
                except: return super()._on_step()
            else: return False
        self.flat_prefs_np = np.concatenate([agent.get_preference(current_obs) for agent in self.sub_agents])
        feature_extractor = self.top_model.policy.q_net.features_extractor
        if isinstance(feature_extractor, CustomCNN): feature_extractor.set_sub_agent_prefs(self.flat_prefs_np)
        else: return False
        return super()._on_step()

class SubAgent:
    def __init__(self, task_id, reward_fn, env_id):
        self.task_id, self.reward_fn, self.env_id, self.model, self.env, self.model_save_path = task_id, reward_fn, env_id, None, None, None

    def setup_agent(self):
        self.env = VecTransposeImage(make_atari_env(self.env_id, n_envs=1, seed=np.random.randint(0, 10000), wrapper_kwargs={'frame_skip': FRAME_SKIP, 'screen_size': SCREEN_SIZE}))
        self.model = DQN("CnnPolicy", self.env, buffer_size=SUB_AGENT_BUFFER_SIZE, learning_rate=SUB_AGENT_LEARNING_RATE, exploration_fraction=SUB_AGENT_EXPLORATION_FRACTION, exploration_final_eps=SUB_AGENT_EXPLORATION_FINAL_EPS, train_freq=4, gradient_steps=1, learning_starts=10000, target_update_interval=1000, verbose=0, device=device)

    def train(self, num_episodes=10, total_timesteps_per_episode=2000):
        if not self.model or not self.env: return [], []
        callback = CollectEpisodeRewardsCallback(num_episodes, f"Sub-agent {self.task_id}")
        self.model.learn(total_timesteps=num_episodes * total_timesteps_per_episode, log_interval=None, tb_log_name=f"DQN_{self.task_id}", callback=callback, reset_num_timesteps=False)
        return callback.episode_rewards, []

    def act(self, state):
        if not self.model: return 0
        state = state[np.newaxis, ...] if state.ndim == len(self.model.observation_space.shape) else state
        if state.shape[1:] != self.model.observation_space.shape: state = np.transpose(state, (0, 3, 1, 2)) if state.shape[-1] == self.model.observation_space.shape[0] else state
        return self.model.predict(state, deterministic=True)[0][0]

    def get_preference(self, state):
        if not self.model: return np.zeros(SUB_AGENT_ACTION_SPACE, dtype=np.float32)
        state = state[np.newaxis, ...] if state.ndim == len(self.model.observation_space.shape) else state
        if state.shape[1:] != self.model.observation_space.shape: state = np.transpose(state, (0, 3, 1, 2)) if state.shape[-1] == self.model.observation_space.shape[0] else state
        # Fix: Use detach() before calling numpy() on tensor that requires grad
        return self.model.q_net(torch.as_tensor(state).to(self.model.device))[0].detach().cpu().numpy()
    
    def save(self, path):
        if self.model:
            self.model_save_path = path
            self.model.save(self.model_save_path)

    def load(self, path=None):
        load_path = path or self.model_save_path
        if not load_path: return
        temp_env = VecTransposeImage(make_atari_env(self.env_id, n_envs=1, wrapper_kwargs={'frame_skip': FRAME_SKIP, 'screen_size': SCREEN_SIZE})) if not self.env else self.env
        self.model = DQN.load(load_path, env=temp_env, device=device)
        self.model_save_path = load_path
        if temp_env != self.env: temp_env.close()

    def close_env(self):
        if self.env: self.env.close(); self.env = None

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = FEATURES_DIM):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(), nn.Flatten())
        cnn_features_size = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
        self.sub_agent_pref_size = NUM_SUB_AGENTS * SUB_AGENT_ACTION_SPACE
        self.sub_agent_fc = nn.Sequential(nn.Linear(self.sub_agent_pref_size, 64), nn.ReLU())
        self.fc = nn.Linear(cnn_features_size + 64, features_dim)
        self.register_buffer("sub_agent_prefs", torch.zeros(1, self.sub_agent_pref_size, dtype=torch.float32, device=device))

    def forward(self, observations):
        cnn_features = self.cnn(observations)
        sub_agent_features = self.sub_agent_fc(self.sub_agent_prefs.repeat(observations.shape[0], 1).to(observations.device))
        return F.relu(self.fc(torch.cat([cnn_features, sub_agent_features], dim=1)))

    def set_sub_agent_prefs(self, prefs_np):
        if prefs_np.shape[0] == self.sub_agent_pref_size: self.sub_agent_prefs.data = torch.as_tensor(prefs_np, dtype=torch.float32).unsqueeze(0).to(self.sub_agent_prefs.device).detach()

class TopAgent:
    def __init__(self, env_id, sub_agents):
        self.sub_agents, self.env_id, self.model, self.env, self.policy_kwargs, self.model_save_path = sub_agents, env_id, None, None, None, None

    def setup_agent(self):
        self.env = VecTransposeImage(make_atari_env(self.env_id, n_envs=1, seed=np.random.randint(10000, 20000), wrapper_kwargs={'frame_skip': FRAME_SKIP, 'screen_size': SCREEN_SIZE}))
        self.policy_kwargs = {"features_extractor_class": CustomCNN, "features_extractor_kwargs": {"features_dim": FEATURES_DIM}}
        self.model = DQN("CnnPolicy", self.env, policy_kwargs=self.policy_kwargs, buffer_size=TOP_AGENT_BUFFER_SIZE, learning_rate=TOP_AGENT_LEARNING_RATE, exploration_fraction=TOP_AGENT_EXPLORATION_FRACTION, exploration_final_eps=TOP_AGENT_EXPLORATION_FINAL_EPS, train_freq=4, gradient_steps=1, learning_starts=20000, target_update_interval=2000, verbose=0, device=device)
        dummy_tensor = torch.as_tensor(np.zeros((1, *self.env.observation_space.shape)), dtype=torch.float32).to(device)
        self.model.q_net(dummy_tensor)
        if not isinstance(self.model.policy.q_net.features_extractor, CustomCNN): raise TypeError("CustomCNN setup failed")

    def train(self, num_episodes=10, total_timesteps_per_episode=5000):
        if not self.model or not self.env or not all(sa.model for sa in self.sub_agents): return [], []
        callback = UpdatePrefsCallback(self.sub_agents, self.model, num_episodes, "Top Agent")
        self.model.learn(total_timesteps=num_episodes * total_timesteps_per_episode, log_interval=None, tb_log_name="DQN_TopAgent", callback=callback, reset_num_timesteps=False)
        return callback.episode_rewards, []

    def act(self, state):
        if not self.model: return 0
        state = state[np.newaxis, ...] if state.ndim == len(self.model.observation_space.shape) else state
        if state.shape[1:] != self.model.observation_space.shape: state = np.transpose(state, (0, 3, 1, 2)) if state.shape[-1] == self.model.observation_space.shape[0] else state
        flat_prefs_np = np.concatenate([agent.get_preference(state[0]) for agent in self.sub_agents]) if all(agent.model for agent in self.sub_agents) else np.zeros(NUM_SUB_AGENTS * SUB_AGENT_ACTION_SPACE, dtype=np.float32)
        feature_extractor = self.model.policy.q_net.features_extractor
        if isinstance(feature_extractor, CustomCNN): feature_extractor.set_sub_agent_prefs(flat_prefs_np)
        return self.model.predict(state, deterministic=True)[0][0]

    def save(self, path):
        if self.model and hasattr(self.model, 'policy'):
            self.model_save_path = path
            torch.save(self.model.policy.state_dict(), self.model_save_path.replace(".zip", "_weights.pth"))

    def load(self, path):
        weights_path = path if path.endswith(".pth") else path.replace(".zip", "_weights.pth")
        if not os.path.exists(weights_path): return
        if not self.model or not self.env: self.setup_agent()
        self.model.policy.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
        self.model_save_path = weights_path
        if not isinstance(self.model.policy.q_net.features_extractor, CustomCNN): self.model = None; raise TypeError("CustomCNN load failed")

    def evaluate(self, env_id, num_episodes=1):
        if not self.model or not all(sa.model for sa in self.sub_agents): return 0.0
        eval_env = Monitor(AtariWrapper(gym.make(env_id), frame_skip=FRAME_SKIP, screen_size=SCREEN_SIZE))
        total_rewards = []
        for _ in range(num_episodes):
            state = eval_env.reset()[0] if isinstance(eval_env.reset(), tuple) else eval_env.reset()
            episode_reward, done = 0, False
            while not done:
                action = self.act(np.transpose(state, (2, 0, 1)))
                state, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            total_rewards.append(episode_reward)
        eval_env.close()
        return np.mean(total_rewards) if total_rewards else 0.0

    def close_env(self):
        if self.env: self.env.close(); self.env = None

def plot_reward_curve(rewards, title, save_path):
    if not rewards: return
    plt.figure(figsize=(12, 7))
    episodes = range(1, len(rewards) + 1)
    plt.plot(episodes, rewards, marker='o', linestyle='-', markersize=3, alpha=0.7, label='Episode Reward')
    if len(rewards) >= 10: plt.plot(episodes, pd.Series(rewards).rolling(window=10, min_periods=1).mean(), color='orange', linestyle='--', label='Moving Avg (10 ep)')
    plt.title(title); plt.xlabel('Episode'); plt.ylabel('Reward'); plt.grid(True, linestyle='--', alpha=0.6)
    if len(rewards) > 1: plt.plot(episodes, np.poly1d(np.polyfit(episodes, rewards, 1))(episodes), linestyle=':', color='red', label=f'Trend (m={np.polyfit(episodes, rewards, 1)[0]:.2f})')
    plt.legend()
    stats_text = f'Episodes: {len(rewards)}\nAvg: {np.mean(rewards):.2f}\nMax: {np.max(rewards):.2f}\nMin: {np.min(rewards):.2f}\nStd: {np.std(rewards):.2f}'
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    plt.tight_layout(); plt.savefig(save_path); plt.close()

if __name__ == "__main__":
    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    log_dir, checkpoint_dir, plots_dir = [os.path.join(d, run_name) for d in [LOG_DIR, CHECKPOINT_DIR, os.path.join(LOG_DIR, run_name, "plots")]]
    os.makedirs(plots_dir, exist_ok=True)
    sb3_file_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    sub_agents = [SubAgent(task_id, reward_fn, ENV_NAME) for task_id, reward_fn in tasks]
    all_sub_agent_rewards = {}
    for i, agent in enumerate(sub_agents):
        agent.setup_agent()
        if sb3_file_logger: agent.model.set_logger(sb3_file_logger)
        rewards, _ = agent.train(SUB_AGENT_EPISODES, ESTIMATED_STEPS_PER_EPISODE_SUB)
        all_sub_agent_rewards[agent.task_id] = rewards
        agent.save(os.path.join(checkpoint_dir, f"sub_agent_{agent.task_id}.zip"))
        plot_reward_curve(rewards, f"Sub-agent {agent.task_id} Rewards", os.path.join(plots_dir, f"sub_agent_{agent.task_id}_rewards.png"))
        agent.close_env()
    if sub_agents:
        [agent.load() for agent in sub_agents]
        top_agent = TopAgent(ENV_NAME, [agent for agent in sub_agents if agent.model])
        if top_agent.sub_agents:
            top_agent.setup_agent()
            if sb3_file_logger: top_agent.model.set_logger(sb3_file_logger)
            top_rewards, _ = top_agent.train(TOP_AGENT_EPISODES, ESTIMATED_STEPS_PER_EPISODE_TOP)
            plot_reward_curve(top_rewards, "Top Agent Rewards", os.path.join(plots_dir, "top_agent_rewards.png"))
            top_agent.save(top_agent_weights_path := os.path.join(checkpoint_dir, "top_agent_final_weights.pth"))
            top_agent.load(top_agent_weights_path)
            if top_agent.model: top_agent.evaluate(ENV_NAME, 5)
            top_agent.close_env()
    logger.info(f"Execution time: {time.time() - start_time:.2f}s")