import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import time
import os
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any, Union
from pathlib import Path
import platform
import json
import shutil

@dataclass
class Config:
    env_name: str = "ALE/MsPacman-v5"
    mode: str = "train"
    test_episodes: int = 10
    play_episodes: int = 3
    auto_play_after_training: bool = True
    rendering: bool = False
    max_episodes: int = 5000
    total_timesteps: int = 1_000_000
    save_model: bool = True
    seed: int = 42
    num_agents: int = 5
    aggregate_agent_results: bool = True
    share_replay_buffer: bool = False
    ensemble_policy: bool = False
    use_competitive_training: bool = True
    eval_frequency: int = 10
    continuation_ratio: float = 0.4
    performance_window: int = 10
    use_knowledge_distillation: bool = True
    distillation_temperature: float = 2.0
    distillation_frequency: int = 1000
    distillation_alpha: float = 0.7
    learning_rate: float = 5e-5
    batch_size: int = 64
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 100_000
    target_update_freq: int = 500
    target_update_method: str = "hard"
    memory_size: int = 100_000
    num_frames: int = 4
    gradient_clip_norm: float = 0.1
    soft_update_tau: float = 0.001
    reward_clip: float = 1.0
    weight_decay: float = 1e-4
    huber_delta: float = 0.5
    use_per: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100_000
    use_reward_shaping: bool = True
    no_movement_penalty: float = -0.01
    survival_reward: float = 0.001
    pellet_reward_multiplier: float = 1.1
    power_pellet_bonus: float = 2.0
    ghost_eaten_bonus: float = 5.0
    life_lost_penalty: float = -1.0
    use_mixed_precision: bool = True
    prefetch_factor: int = 2
    pin_memory: bool = True
    non_blocking: bool = True
    log_level: str = "INFO"
    report_freq: int = 5000
    save_freq: int = 50000
    plot_freq: int = 50
    reward_level_threshold: int = 10000
    model_dir: Path = field(default_factory=lambda: Path("dqn_models"))
    log_dir: Path = field(default_factory=lambda: Path("dqn_logs"))

    def __post_init__(self):
        if self.mode == "fast_train":
            self._apply_fast_train_settings()
    
    def _apply_fast_train_settings(self):
        self.max_episodes = 10
        self.total_timesteps = 50_000
        self.report_freq = 1000
        self.save_freq = 5000
        self.epsilon_decay_steps = 10_000
        self.target_update_freq = 500
        self.memory_size = 10_000
        self.per_beta_frames = 10_000
        self.plot_freq = 10
        self.num_agents = 2

def setup_logger(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('DQNAgentLogger')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"dqn_train_{timestamp}.log"
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

def estimate_game_level(score: float, reward_level_threshold: int) -> int:
    return int(score // reward_level_threshold) + 1

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_env(config: Config, seed: int, render_mode: Optional[str] = None) -> gym.Env:
    env = gym.make(config.env_name, render_mode=render_mode, full_action_space=False)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.AtariPreprocessing(
        env, 
        frame_skip=1, 
        screen_size=84,
        grayscale_obs=True, 
        scale_obs=False,
        noop_max=30
    )
    if config.use_reward_shaping:
        env = RewardShapingWrapper(env, config)
    env = gym.wrappers.FrameStackObservation(env, config.num_frames)
    env.action_space.seed(seed)
    return env

class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, config: Config):
        super().__init__(env)
        self.config = config
        self.last_raw_score = 0
        self.last_lives = None
        self.current_raw_score = 0

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self.last_raw_score = 0
        self.current_raw_score = 0
        self.last_lives = info.get('ale.lives', 3)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_lives = info.get('ale.lives', self.last_lives)
        self.current_raw_score = info.get('episode', {}).get('r', self.current_raw_score)
        if 'ale.score' in info:
            self.current_raw_score = info['ale.score']
        shaped_reward = float(reward)
        shaped_reward += self._apply_survival_reward()
        shaped_reward += self._apply_score_change_reward()
        shaped_reward += self._apply_life_lost_penalty(current_lives)
        self.last_raw_score = self.current_raw_score
        self.last_lives = current_lives
        return obs, shaped_reward, terminated, truncated, info
    
    def _apply_survival_reward(self) -> float:
        return self.config.survival_reward
    
    def _apply_score_change_reward(self) -> float:
        score_diff = self.current_raw_score - self.last_raw_score
        if score_diff <= 0:
            return 0.0
        if score_diff > 50:
            return self.config.ghost_eaten_bonus
        else:
            return (score_diff * self.config.pellet_reward_multiplier) / 10.0
    
    def _apply_life_lost_penalty(self, current_lives: int) -> float:
        if current_lives < self.last_lives:
            return self.config.life_lost_penalty
        return 0.0

class QNetwork(nn.Module):
    def __init__(self, num_actions: int, num_frames: int = 4):
        super().__init__()
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.q_min = -10.0
        self.q_max = 10.0
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        feature_size = self._calculate_feature_size()
        self.value_stream = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        self._initialize_weights()

    def _calculate_feature_size(self) -> int:
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.num_frames, 84, 84)
            return self.feature_extractor(dummy_input).shape[1]

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -0.01, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().contiguous() / 255.0
        features = self.feature_extractor(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        q_values = torch.clamp(q_values, self.q_min, self.q_max)
        return q_values

class PrioritizedReplayBuffer:
    def __init__(
        self, 
        capacity: int, 
        alpha: float = 0.6, 
        beta_start: float = 0.4, 
        beta_frames: int = 100_000,
        pin_memory: bool = True, 
        non_blocking: bool = True, 
        prefetch_factor: int = 2
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        self.pin_memory = pin_memory
        self.non_blocking = non_blocking
        self.prefetch_factor = prefetch_factor
        self.using_mps = False
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        experience = (
            np.array(state, dtype=np.uint8),
            action,
            reward,
            np.array(next_state, dtype=np.uint8),
            done
        )
        priority = self.max_priority
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def _get_beta(self) -> float:
        progress = min(1.0, self.frame / self.beta_frames)
        return self.beta_start + progress * (1.0 - self.beta_start)

    def _prepare_sampling_probs(self, buffer_size: int) -> np.ndarray:
        priorities = self.priorities[:buffer_size]
        probs = priorities ** self.alpha
        probs_sum = probs.sum()
        if probs_sum < 1e-9 or np.isnan(probs).any():
            return np.ones(buffer_size) / buffer_size
        return probs / probs_sum

    def sample(self, batch_size: int, device: torch.device):
        self.using_mps = (device.type == 'mps')
        buffer_size = len(self.buffer)
        if buffer_size < batch_size:
            raise ValueError("Not enough samples in buffer to form a batch.")
        probs = self._prepare_sampling_probs(buffer_size)
        indices = np.random.choice(buffer_size, batch_size, p=probs, replace=False)
        weights = (buffer_size * probs[indices]) ** (-self._get_beta())
        weights /= weights.max()
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        if self.pin_memory and not self.using_mps:
            weights_tensor = weights_tensor.pin_memory()
        weights_tensor = weights_tensor.to(device, non_blocking=self.non_blocking).unsqueeze(1)
        batch = self._create_batch_from_indices(indices, device)
        self.frame += 1
        return (*batch, indices, weights_tensor)

    def _create_batch_from_indices(self, indices, device):
        experiences = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = np.stack(states)
        next_states = np.stack(next_states)
        states_tensor = torch.from_numpy(states)
        next_states_tensor = torch.from_numpy(next_states)
        if self.pin_memory and not self.using_mps:
            states_tensor = states_tensor.pin_memory()
            next_states_tensor = next_states_tensor.pin_memory()
        states_tensor = states_tensor.to(device, non_blocking=self.non_blocking)
        actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(
            device, non_blocking=self.non_blocking)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(
            device, non_blocking=self.non_blocking)
        next_states_tensor = next_states_tensor.to(device, non_blocking=self.non_blocking)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(
            device, non_blocking=self.non_blocking)
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        priorities = np.abs(td_errors) + 1e-6
        np.clip(priorities, 1e-6, None, out=priorities)
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority.item()
        self.max_priority = max(self.max_priority, np.max(priorities).item())

    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    def __init__(self, env: gym.Env, config: Config, device: torch.device, logger: logging.Logger):
        self.env = env
        self.config = config
        self.device = device
        self.logger = logger
        self.num_actions = env.action_space.n
        self.total_steps = 0
        self.current_epsilon = config.epsilon_start
        self.recent_losses = deque(maxlen=100)
        self._setup_networks()
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scaler = self._setup_mixed_precision()
        self.memory = self._setup_replay_buffer()
        
    def _setup_networks(self):
        self.policy_net = QNetwork(self.num_actions, self.config.num_frames).to(self.device)
        self.target_net = QNetwork(self.num_actions, self.config.num_frames).to(self.device)
        self.update_target_network()
        self.target_net.eval()
        
    def _setup_mixed_precision(self):
        if self.config.use_mixed_precision and self.device.type == 'cuda':
            return torch.cuda.amp.GradScaler()
        return None
        
    def _setup_replay_buffer(self):
        if not self.config.use_per:
            raise NotImplementedError("Standard replay buffer not implemented. Set use_per=True.")
        using_mps = (self.device.type == 'mps')
        actual_pin_memory = self.config.pin_memory and not using_mps
        return PrioritizedReplayBuffer(
            capacity=self.config.memory_size,
            alpha=self.config.per_alpha,
            beta_start=self.config.per_beta_start,
            beta_frames=self.config.per_beta_frames,
            pin_memory=actual_pin_memory,
            non_blocking=self.config.non_blocking,
            prefetch_factor=self.config.prefetch_factor
        )
        
    def _update_epsilon(self):
        fraction = min(1.0, self.total_steps / self.config.epsilon_decay_steps)
        self.current_epsilon = self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )
        self.current_epsilon = max(self.config.epsilon_end, self.current_epsilon)

    def select_action(self, state: np.ndarray, test_mode: bool = False) -> int:
        if test_mode:
            epsilon = 0.01
        else:
            self._update_epsilon()
            epsilon = self.current_epsilon
        if random.random() < epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            state_tensor = torch.from_numpy(np.array(state, dtype=np.uint8))
            if self.config.pin_memory and not state_tensor.is_cuda and self.device.type != 'mps':
                state_tensor = state_tensor.pin_memory()
            state_tensor = state_tensor.unsqueeze(0).to(
                self.device, non_blocking=self.config.non_blocking
            )
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def get_avg_loss(self) -> float:
        if not self.recent_losses:
            return 0.0
        valid_losses = [
            loss for loss in self.recent_losses 
            if not (np.isnan(loss) or np.isinf(loss) or loss > 1e10)
        ]
        if not valid_losses:
            return 0.0
        return np.mean(valid_losses)

    def train_step(self) -> Optional[float]:
        if len(self.memory) < self.config.batch_size:
            return None
        batch = self.memory.sample(self.config.batch_size, self.device)
        states, actions, rewards, next_states, dones, indices, weights = batch
        if self.config.reward_clip > 0:
            rewards = torch.clamp(rewards, -self.config.reward_clip, self.config.reward_clip)
        if self.scaler:
            loss, q_values = self._train_step_amp(states, actions, rewards, next_states, dones, weights, indices)
        else:
            loss, q_values = self._train_step_standard(states, actions, rewards, next_states, dones, weights, indices)
        if self.total_steps % self.config.target_update_freq == 0:
            self.update_target_network()
        if loss is not None:
            loss_value = loss.item()
            if self._is_valid_loss(loss_value):
                return loss_value
        return None
        
    def _train_step_amp(self, states, actions, rewards, next_states, dones, weights, indices):
        with torch.cuda.amp.autocast():
            loss, current_q_values, target_q_values = self._compute_loss(
                states, actions, rewards, next_states, dones, weights, return_q_values=True
            )
        if not self._is_valid_loss_tensor(loss):
            return None, (None, None)
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.gradient_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self._record_loss(loss.item())
        self._update_priorities(indices, current_q_values, target_q_values)
        return loss, (current_q_values, target_q_values)
    
    def _train_step_standard(self, states, actions, rewards, next_states, dones, weights, indices):
        loss, current_q_values, target_q_values = self._compute_loss(
            states, actions, rewards, next_states, dones, weights, return_q_values=True
        )
        if not self._is_valid_loss_tensor(loss):
            return None, (None, None)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        self._record_loss(loss.item())
        self._update_priorities(indices, current_q_values, target_q_values)
        return loss, (current_q_values, target_q_values)
    
    def _compute_loss(self, states, actions, rewards, next_states, dones, weights, return_q_values=False):
        current_q_values = self.policy_net(states).gather(1, actions)
        if torch.isnan(current_q_values).any() or torch.isinf(current_q_values).any():
            return (None, None, None) if return_q_values else None
        with torch.no_grad():
            next_q_policy = self.policy_net(next_states)
            if not self._tensor_is_valid(next_q_policy):
                return (None, None, None) if return_q_values else None
            next_actions = next_q_policy.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_states)
            if not self._tensor_is_valid(next_q_target):
                return (None, None, None) if return_q_values else None
            next_q_values = next_q_target.gather(1, next_actions)
            if not self._tensor_is_valid(next_q_values):
                return (None, None, None) if return_q_values else None
            target_q_values = rewards + (self.config.gamma * next_q_values * (1 - dones))
            if not self._tensor_is_valid(target_q_values):
                return (None, None, None) if return_q_values else None
        delta = self.config.huber_delta
        diff = target_q_values - current_q_values
        elementwise_loss = torch.where(
            diff.abs() <= delta,
            0.5 * diff.pow(2),
            delta * (diff.abs() - 0.5 * delta)
        )
        loss = (weights * elementwise_loss).mean()
        if return_q_values:
            return loss, current_q_values, target_q_values
        else:
            return loss
    
    def _tensor_is_valid(self, tensor):
        return not (torch.isnan(tensor).any() or torch.isinf(tensor).any())
    
    def _is_valid_loss_tensor(self, loss):
        return loss is not None and self._tensor_is_valid(loss)
    
    def _is_valid_loss(self, loss_value):
        return not (np.isnan(loss_value) or np.isinf(loss_value) or loss_value > 1e10)
    
    def _record_loss(self, loss_value):
        if self._is_valid_loss(loss_value):
            self.recent_losses.append(loss_value)
    
    def _update_priorities(self, indices, current_q_values, target_q_values):
        if not self.config.use_per or current_q_values is None or target_q_values is None:
            return
        with torch.no_grad():
            td_errors = (target_q_values - current_q_values).abs()
            if not self._tensor_is_valid(td_errors):
                return
            td_errors = torch.clamp(td_errors, 0.0, 10.0)
            td_errors_np = td_errors.squeeze().cpu().numpy()
            self.memory.update_priorities(indices, td_errors_np)

    def update_target_network(self):
        if self.config.target_update_method == "soft":
            self._soft_update_target()
        else:
            self._hard_update_target()
            
    def _soft_update_target(self):
        tau = self.config.soft_update_tau
        target_state_dict = self.target_net.state_dict()
        policy_state_dict = self.policy_net.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[key] * tau + target_state_dict[key] * (1 - tau)
        self.target_net.load_state_dict(target_state_dict)
        
    def _hard_update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), filepath)

    def load_model(self, filepath: Path) -> bool:
        if not filepath.exists():
            return False
        state_dict = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.train()
        self.target_net.eval()
        return True

    def increment_step(self):
        self.total_steps += 1

def train_agent(agent, env, config, logger):
    state, info = env.reset(seed=config.seed + agent.total_steps)
    stats = {
        'all_rewards': [],
        'episode_rewards': [],
        'moving_avg_rewards': [],
        'losses': [],
        'episode_lengths': [],
        'episode_times': [],
        'estimated_levels_reached': []
    }
    num_episodes = 0
    global_start_time = time.time()
    while agent.total_steps < config.total_timesteps and num_episodes < config.max_episodes:
        episode_stats = run_single_episode(
            agent, env, state, config, logger, num_episodes, global_start_time, stats
        )
        for key, value in episode_stats.items():
            if key in stats:
                stats[key].append(value)
        state = episode_stats['next_state']
        num_episodes += 1
        if num_episodes % config.plot_freq == 0:
            plot_training_progress(
                config.model_dir, config.env_name, num_episodes, agent.total_steps,
                stats['all_rewards'], stats['moving_avg_rewards'], 
                stats['estimated_levels_reached'], stats['episode_lengths'],
                agent.current_epsilon, config.epsilon_decay_steps
            )
        if agent.total_steps >= config.total_timesteps or num_episodes >= config.max_episodes:
            break
    finalize_training(agent, config, logger, stats, global_start_time, num_episodes)
    
def run_single_episode(agent, env, state, config, logger, num_episodes, global_start_time, stats):
    current_episode_reward = 0.0
    episode_step = 0
    highest_score_this_episode = 0.0
    max_estimated_level_this_episode = 1
    episode_start_time = time.time()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        current_episode_reward += reward
        highest_score_this_episode = max(highest_score_this_episode, current_episode_reward)
        current_est_level = estimate_game_level(highest_score_this_episode, config.reward_level_threshold)
        max_estimated_level_this_episode = max(max_estimated_level_this_episode, current_est_level)
        agent.memory.add(state, action, reward, next_state, done)
        state = next_state
        agent.increment_step()
        episode_step += 1
        loss = agent.train_step()
        if loss is not None:
            stats['losses'].append(loss)
        if config.save_model and agent.total_steps % config.save_freq == 0:
            save_path = config.model_dir / f"{config.env_name.replace('/', '_')}_step_{agent.total_steps}.pth"
            agent.save_model(save_path)
    episode_time = time.time() - episode_start_time
    final_ep_reward = info.get("episode", {}).get("r", current_episode_reward)
    all_rewards = stats['all_rewards'] + [final_ep_reward]
    moving_avg = np.mean(all_rewards[-100:])
    logger.info(f"Episode {num_episodes + 1} finished with score: {final_ep_reward:.2f}")
    next_state, info = env.reset() if not done else (None, None)
    return {
        'all_rewards': final_ep_reward,
        'episode_rewards': final_ep_reward,
        'moving_avg_rewards': moving_avg,
        'episode_lengths': episode_step,
        'episode_times': episode_time,
        'estimated_levels_reached': max_estimated_level_this_episode,
        'next_state': next_state
    }

def finalize_training(agent, config, logger, stats, global_start_time, num_episodes):
    total_training_time = time.time() - global_start_time
    if config.save_model:
        final_save_path = config.model_dir / f"{config.env_name.replace('/', '_')}_final.pth"
        agent.save_model(final_save_path)
    plot_training_progress(
        config.model_dir, config.env_name, num_episodes, agent.total_steps,
        stats['all_rewards'], stats['moving_avg_rewards'], 
        stats['estimated_levels_reached'], stats['episode_lengths'],
        agent.current_epsilon, config.epsilon_decay_steps, is_final=True
    )

def test_agent(agent, config, logger):
    test_env = create_env(config, seed=config.seed + 1000)
    test_stats = {
        'rewards': [],
        'lengths': [],
        'max_levels': []
    }
    for episode in range(config.test_episodes):
        episode_stats = run_test_episode(agent, test_env, episode, config, logger)
        for key, value in episode_stats.items():
            test_stats[key].append(value)
    test_env.close()
    log_test_summary(test_stats, logger)

def run_test_episode(agent, env, episode, config, logger):
    state, _ = env.reset()
    total_reward = 0.0
    steps = 0
    max_level_ep = 1
    done = False
    episode_start_time = time.time()
    while not done:
        action = agent.select_action(state, test_mode=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        raw_score = info.get('episode', {}).get('r', total_reward)
        max_level_ep = max(max_level_ep, estimate_game_level(raw_score, config.reward_level_threshold))
    ep_time = time.time() - episode_start_time
    logger.info(f"Test Episode {episode + 1} finished with score: {total_reward:.2f}")
    return total_reward, steps, max_level_ep

def play_ensemble_agents(agents: List[DQNAgent], config: Config, logger: logging.Logger):
    play_env = create_env(config, seed=config.seed + 2000, render_mode="human")
    for episode in range(config.play_episodes):
        state, info = play_env.reset()
        total_reward = 0.0
        steps = 0
        max_level_ep = 1
        done = False
        while not done:
            actions = [agent.select_action(state, test_mode=True) for agent in agents]
            action_counts = np.bincount(actions, minlength=play_env.action_space.n)
            action = np.argmax(action_counts).item()
            confidence = action_counts[action] / len(agents)
            state, reward, terminated, truncated, info = play_env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            raw_score = info.get('episode', {}).get('r', total_reward)
            max_level_ep = max(max_level_ep, estimate_game_level(raw_score, config.reward_level_threshold))
        logger.info(f"Ensemble Play Episode {episode + 1} finished with score: {total_reward:.2f}")
    play_env.close()

def evaluate_agent_performance(rewards: List[float], window: int) -> float:
    if not rewards:
        return 0.0
    recent_rewards = rewards[-window:] if len(rewards) >= window else rewards
    recent_avg = np.mean(recent_rewards)
    peak_score = np.max(rewards) if rewards else 0
    exceptional_threshold = np.mean(rewards) * 1.5 if rewards else 0
    exceptional_count = sum(1 for r in rewards if r > exceptional_threshold)
    exceptional_bonus = exceptional_count * 0.2
    combined_score = (0.5 * recent_avg) + (0.3 * peak_score) + (0.2 * exceptional_bonus * np.mean(rewards))
    return combined_score

def rank_agents(agents: List[DQNAgent], 
                all_rewards: List[List[float]], 
                performance_window: int) -> List[Tuple[int, float]]:
    agent_scores = []
    for i, agent_rewards in enumerate(all_rewards):
        score = evaluate_agent_performance(agent_rewards, performance_window)
        agent_scores.append((i, score))
    return sorted(agent_scores, key=lambda x: x[1], reverse=True)

def select_elite_agents(agent_rankings: List[Tuple[int, float]], 
                      continuation_ratio: float) -> List[int]:
    num_to_keep = max(1, int(len(agent_rankings) * continuation_ratio))
    return [idx for idx, _ in agent_rankings[:num_to_keep]]

def identify_exceptional_agent(agent_rankings: List[Tuple[int, float]], 
                              threshold_ratio: float) -> Optional[int]:
    if len(agent_rankings) < 2:
        return agent_rankings[0][0] if agent_rankings else None
    top_score = agent_rankings[0][1]
    runner_up_score = agent_rankings[1][1] if len(agent_rankings) > 1 else 0
    if runner_up_score == 0 or (top_score / runner_up_score) > (1 + threshold_ratio):
        return agent_rankings[0][0]
    return None

def knowledge_distillation_update(student_agent: DQNAgent, 
                                 teacher_agent: DQNAgent, 
                                 states: torch.Tensor,
                                 temperature: float,
                                 alpha: float) -> Optional[float]:
    if states.shape[0] == 0:
        return None
    student_agent.policy_net.train()
    teacher_agent.policy_net.eval()
    with torch.no_grad():
        teacher_q_values = teacher_agent.policy_net(states)
        soft_targets = F.softmax(teacher_q_values / temperature, dim=1)
    student_q_values = student_agent.policy_net(states)
    soft_student = F.log_softmax(student_q_values / temperature, dim=1)
    distillation_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean') * (temperature ** 2)
    student_agent.optimizer.zero_grad()
    distillation_loss.backward()
    torch.nn.utils.clip_grad_norm_(student_agent.policy_net.parameters(), 
                                   student_agent.config.gradient_clip_norm)
    student_agent.optimizer.step()
    return distillation_loss.item()

def copy_weights_from_to(source_agent: DQNAgent, target_agent: DQNAgent, tau: float):
    target_dict = target_agent.policy_net.state_dict()
    source_dict = source_agent.policy_net.state_dict()
    for key in source_dict:
        target_dict[key] = source_dict[key] * tau + target_dict[key] * (1 - tau)
    target_agent.policy_net.load_state_dict(target_dict)
    if tau == 1.0:
        target_agent.target_net.load_state_dict(target_dict)

def collect_distillation_batch(agents: List[DQNAgent], 
                              env: gym.Env, 
                              batch_size: int, 
                              device: torch.device,
                              test_mode: bool = True) -> torch.Tensor:
    states = []
    state, _ = env.reset()
    for _ in range(batch_size):
        states.append(state)
        action = random.choice([agent.select_action(state, test_mode) for agent in agents])
        next_state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            next_state, _ = env.reset()
        state = next_state
    states_batch = np.stack(states)
    states_tensor = torch.from_numpy(states_batch).to(device)
    return states_tensor

def setup_device(config, logger):
    if torch.backends.mps.is_available() and platform.system() == "Darwin":
        device = torch.device("mps")
        config.pin_memory = False
        if config.use_mixed_precision:
            config.use_mixed_precision = False
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def create_dummy_env(config, logger):
    try:
        env = create_env(config, seed=config.seed)
        num_actions = env.action_space.n
        obs_shape = env.observation_space.shape
        env.close()
        return env
    except Exception as e:
        raise

def run_single_agent_mode(config, device, dummy_env, logger):
    agent = DQNAgent(dummy_env, config, device, logger)
    load_latest_checkpoint(agent, config, logger)
    if config.mode in ["train", "fast_train"]:
        render_mode = "human" if config.rendering else None
        env = create_env(config, seed=config.seed, render_mode=render_mode)
        train_agent(agent, env, config, logger)
        env.close()
        if config.auto_play_after_training and config.save_model:
            if load_latest_checkpoint(agent, config, logger):
                play_agent(agent, config, logger)
    elif config.mode == "test":
        test_agent(agent, config, logger)
    elif config.mode == "play":
        play_agent(agent, config, logger)

def run_multi_agent_mode(config, device, dummy_env, logger):
    agents = []
    for i in range(config.num_agents):
        agent_seed = config.seed + i * 100
        agent_model_dir = config.model_dir / f"agent_{i}"
        agent_model_dir.mkdir(parents=True, exist_ok=True)
        agent_config = Config()
        for key, value in vars(config).items():
            setattr(agent_config, key, value)
        agent_config.seed = agent_seed
        agent_config.model_dir = agent_model_dir
        env = create_env(config, seed=agent_seed)
        agent = DQNAgent(env, agent_config, device, logger)
        env.close()
        model_files = list(agent_model_dir.glob("*.pth"))
        if model_files:
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            agent.load_model(latest_model)
        agents.append(agent)
    if config.mode in ["train", "fast_train"]:
        agent_rewards = [[] for _ in range(config.num_agents)]
        for episode in range(config.max_episodes):
            for i, agent in enumerate(agents):
                reward = train_agent_episode(agent, config, logger)
                agent_rewards[i].append(reward)
                logger.info(f"Agent {i+1} Episode {episode+1} finished with score: {reward:.2f}")
                if episode % 10 == 0:
                    save_path = agent.config.model_dir / f"{config.env_name.replace('/', '_')}_episode_{episode+1}.pth"
                    agent.save_model(save_path)
            if config.use_competitive_training and (episode + 1) % config.eval_frequency == 0:
                if all(len(rewards) >= config.performance_window for rewards in agent_rewards):
                    agent_rankings = rank_agents(agents, agent_rewards, config.performance_window)
                    elite_indices = select_elite_agents(agent_rankings, config.continuation_ratio)
                    if config.use_knowledge_distillation:
                        for i in range(config.num_agents):
                            if i not in elite_indices:
                                teacher_idx = random.choice(elite_indices)
                                teacher_agent = agents[teacher_idx]
                                student_agent = agents[i]
                                distill_env = create_env(config, seed=config.seed + 500)
                                states_tensor = collect_distillation_batch(
                                    [teacher_agent], distill_env, batch_size=64, device=device
                                )
                                distill_env.close()
                                knowledge_distillation_update(
                                    student_agent=student_agent,
                                    teacher_agent=teacher_agent,
                                    states=states_tensor,
                                    temperature=config.distillation_temperature,
                                    alpha=config.distillation_alpha
                                )
        for i, agent in enumerate(agents):
            final_save_path = agent.config.model_dir / f"{config.env_name.replace('/', '_')}_final.pth"
            agent.save_model(final_save_path)
        if config.auto_play_after_training:
            mean_performances = [np.mean(rewards[-config.performance_window:]) 
                                if len(rewards) >= config.performance_window 
                                else np.mean(rewards) if rewards 
                                else 0 
                                for rewards in agent_rewards]
            best_agent_idx = np.argmax(mean_performances)
            if config.ensemble_policy:
                play_ensemble_agent(agents, config, logger)
            else:
                play_agent(agents[best_agent_idx], config, logger)
    elif config.mode == "test":
        test_rewards = []
        episode_lengths = []
        max_levels = []
        for i, agent in enumerate(agents):
            test_env = create_env(config, seed=config.seed + 1000 + i)
            agent_rewards = []
            agent_lengths = []
            agent_levels = []
            for episode in range(config.test_episodes):
                reward, steps, max_level = run_test_episode(agent, test_env, episode, config, logger)
                agent_rewards.append(reward)
                agent_lengths.append(steps)
                agent_levels.append(max_level)
            test_env.close()
            test_rewards.extend(agent_rewards)
            episode_lengths.extend(agent_lengths)
            max_levels.extend(agent_levels)
    elif config.mode == "play":
        if config.ensemble_policy:
            play_ensemble_agent(agents, config, logger)
        else:
            for i, agent in enumerate(agents):
                play_agent(agent, config, logger)

def train_agent_episode(agent, config, logger):
    env = create_env(config, seed=agent.config.seed + agent.total_steps)
    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.memory.add(state, action, reward, next_state, done)
        loss = agent.train_step()
        state = next_state
        episode_reward += reward
        episode_steps += 1
        agent.increment_step()
    env.close()
    return episode_reward

def play_ensemble_agent(agents, config, logger):
    env = create_env(config, seed=config.seed, render_mode="human")
    for episode in range(config.play_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        while not done:
            q_values_list = []
            for agent in agents:
                with torch.no_grad():
                    state_tensor = torch.from_numpy(np.array(state, dtype=np.uint8)).unsqueeze(0).to(agent.device)
                    q_values = agent.policy_net(state_tensor)
                    q_values_list.append(q_values.cpu().numpy())
            ensemble_q_values = np.mean(q_values_list, axis=0)
            action = np.argmax(ensemble_q_values)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
            steps += 1
        logger.info(f"Episode {episode+1} finished with score: {episode_reward:.2f}")
    env.close()

if __name__ == "__main__":
    cfg = Config()
    logger = setup_logger(cfg.log_dir, cfg.log_level)
    set_seeds(cfg.seed)
    device = setup_device(cfg, logger)
    dummy_env = create_dummy_env(cfg, logger)
    try:
        if cfg.num_agents <= 1:
            run_single_agent_mode(cfg, device, dummy_env, logger)
        else:
            run_multi_agent_mode(cfg, device, dummy_env, logger)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        pass