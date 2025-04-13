from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
import numpy as np
import torch
import os
import importlib.util
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import argparse

@dataclass
class TrainingConfig:
    env_id: str
    timesteps: int
    n_envs: int
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    device: str = "auto"
    norm_obs: bool = True
    norm_reward: bool = False
    use_lr_schedule: bool = False
    vec_normalize_filename: str = "vecnormalize.pkl"
    callback_window: int = 100
    policy_kwargs: Optional[Dict] = None
    target_episodes: Optional[int] = None

class TrainingProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, n_envs: int, window: int = 100, target_episodes: Optional[int] = None):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.window = window
        self.target_episodes = target_episodes
        self.episode_rewards: List[float] = []
        self.episode_times: List[float] = []
        self.episode_lengths: List[int] = []
        self.moving_avg_window = window
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        if self.locals["dones"].any():
            for idx, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][idx].get("episode", {})
                    self.episode_rewards.append(info.get("r", 0.0))
                    self.episode_lengths.append(info.get("l", 0))
                    self.episode_times.append(info.get("t", 0.0) / 1000)
                    self.episode_count += 1
                    progress_percent = (self.episode_count / self.target_episodes) * 100
                    print(f"Episode {self.episode_count}: Reward = {self.episode_rewards[-1]:.2f}, Steps = {self.episode_lengths[-1]}, Time = {self.episode_times[-1]:.2f}s, Progress = {progress_percent:.2f}%")
          
        if self.target_episodes is not None and self.episode_count >= self.target_episodes:
            print(f"\nReached target number of episodes: {self.target_episodes}. Stopping training.")
            return False 
            
        return True 
        
    def get_summary(self):
        return {
            'total_episodes': len(self.episode_rewards),
            'total_steps': self.num_timesteps,
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'best_reward': max(self.episode_rewards) if self.episode_rewards else 0.0,
            'avg_steps_per_episode': np.mean(self.episode_lengths) if self.episode_lengths else 0.0
        }

def create_plots(rewards: List[float], times: List[float], paths: Tuple[str, str], window: int) -> None:
    reward_path, time_path = paths
    episodes = list(range(1, len(rewards) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label="Reward", alpha=0.6)
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(episodes[window - 1:], moving_avg, label=f"{window}-Ep MA", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(reward_path, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, times, label="Time (s)", alpha=0.6)
    plt.xlabel("Episode")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid(True)
    plt.savefig(time_path, bbox_inches="tight")
    plt.close()

def linear_schedule(initial_value: float) -> callable:
    def func(progress: float) -> float:
        return progress * initial_value
    return func

def train_lunar_lander(config: TrainingConfig, dirs: Dict[str, str]) -> Tuple[str, str, List[float]]:
    start_time = time.time()
    vec_env_cls = SubprocVecEnv if config.n_envs > 1 else DummyVecEnv
    env = make_vec_env(config.env_id, n_envs=config.n_envs, vec_env_cls=vec_env_cls, seed=int(time.time() * 1000) % (2**32 - 1))
    env = VecNormalize(env, norm_obs=True, norm_reward=False, gamma=config.gamma)

    lr = linear_schedule(config.learning_rate) if config.use_lr_schedule else config.learning_rate
    logger = configure(dirs["logs"], ["csv", "tensorboard"])
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        device=config.device,
        policy_kwargs=config.policy_kwargs or {},
        seed=int(time.time() * 1000 + 1) % (2**32 - 1),
    )
    model.set_logger(logger)

    callback = TrainingProgressCallback(
        config.timesteps, 
        config.n_envs, 
        window=config.callback_window,
        target_episodes=config.target_episodes
    )
    model.learn(total_timesteps=config.timesteps, callback=callback)
    rewards = callback.episode_rewards
    times = callback.episode_times
    lengths = callback.episode_lengths

    model_path = os.path.join(dirs["zip_models"], f"lunar_lander_ppo_{model.num_timesteps}steps.zip")
    stats_path = os.path.join(dirs["zip_models"], config.vec_normalize_filename)
    torch_path = os.path.join(dirs["pth_models"], f"lunar_lander_policy_{model.num_timesteps}steps.pth")
    model.save(model_path)
    env.save(stats_path)
    torch.save(model.policy.state_dict(), torch_path)

    if rewards:
        reward_path = os.path.join(dirs["plots"], f"reward_curve_{model.num_timesteps}steps.png")
        time_path = os.path.join(dirs["plots"], f"episode_times_{model.num_timesteps}steps.png")
        create_plots(rewards, times, (reward_path, time_path), callback.window)
        csv_path = os.path.join(dirs["plots"], f"episode_data_{model.num_timesteps}steps.csv")
        with open(csv_path, "w") as f:
            f.write("episode,reward,time,steps\n")
            for i, (r, t, s) in enumerate(zip(rewards, times, lengths)):
                f.write(f"{i+1},{r:.4f},{t:.4f},{s}\n")

    summary = callback.get_summary()
    print('\nTRAINING SUMMARY')
    print(f'Time: {time.time() - start_time:.2f}m | Steps: {summary['total_steps']}/{config.timesteps}')
    print(f'Episodes: {summary['total_episodes']} | Avg Reward: {summary['avg_reward']:.2f}')
    if len(rewards) >= callback.moving_avg_window:
        print(f'Last {callback.moving_avg_window} Avg: {np.mean(rewards[-callback.moving_avg_window:]):.2f}')
    print(f'Best: {summary['best_reward']:.2f} | Avg Steps: {summary['avg_steps_per_episode']:.1f}')


    env.close()
    return model_path, stats_path, rewards

def load_config(config_file: str, config_dir: str) -> TrainingConfig:
    if not config_file.endswith('.py'):
        config_file += '.py'
    config_path = os.path.join(config_dir, config_file)
    if not os.path.exists(config_path):
        raise FileNotFoundError
    module_name = os.path.basename(config_file).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if not spec or not spec.loader:
        raise ImportError
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'TrainingConfig'):
        raise AttributeError
    return module.TrainingConfig()

def main():
    parser = argparse.ArgumentParser(description='Train PPO on LunarLander-v3')
    parser.add_argument('config', type=str, help='Path to config file (relative to configs directory, .py extension optional)')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base_dir, 'configs')
    config = load_config(args.config, config_dir)

    run_dir = os.path.join(base_dir, 'training_runs', f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    dirs = {
        'logs': os.path.join(run_dir, 'logs'),
        'zip_models': os.path.join(run_dir, 'models', 'zip'),
        'pth_models': os.path.join(run_dir, 'models', 'pth'),
        'plots': os.path.join(run_dir, 'plots')
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    train_lunar_lander(config, dirs)

if __name__ == '__main__':
    main()