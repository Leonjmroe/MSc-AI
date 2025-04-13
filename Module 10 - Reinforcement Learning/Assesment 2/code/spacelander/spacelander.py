from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
import numpy as np
import torch
import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

@dataclass
class TrainingConfig:
    env_id: str = "LunarLander-v3"
    algorithm: str = "PPO"
    device: str = "cuda" # cuda / mps / cpu
    n_envs: int = 8
    timesteps: int = 1000000
    learning_rate: float = 0.003
    use_lr_schedule: bool = True
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy_kwargs: Dict = None
    vec_normalize_filename: str = "vec_normalize.pkl"

class TrainingProgressCallback(BaseCallback):
    """Callback for tracking and reporting training progress per episode."""
    def __init__(self, total_timesteps: int, n_envs: int, moving_avg_window: int = 100):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.moving_avg_window = moving_avg_window
        self.episode_rewards: List[float] = []
        self.episode_times: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_count: int = 0

    def _on_step(self) -> bool:
        if self.locals["dones"].any():
            for idx, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][idx].get("episode", {})
                    reward = info.get("r", 0.0)
                    length = info.get("l", 0)
                    time_taken = info.get("t", 0.0) / 1000  
                    self.episode_count += 1
                    self.episode_rewards.append(reward)
                    self.episode_lengths.append(length)
                    self.episode_times.append(time_taken)
                    progress_percent = (self.num_timesteps / self.total_timesteps) * 100
                    print(f"Episode {self.episode_count}: Reward = {reward:.2f}, Steps = {length}, Time = {time_taken:.2f}s, Progress = {progress_percent:.2f}%")
        return True

    def get_summary(self) -> Dict[str, float]:
        return {
            "total_steps": self.num_timesteps,
            "total_episodes": self.episode_count,
            "avg_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "best_reward": max(self.episode_rewards) if self.episode_rewards else 0.0,
            "avg_steps_per_episode": np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
        }

def create_plots(rewards: List[float], episode_times: List[float], paths: Tuple[str, str], moving_avg_window: int) -> None:
    reward_path, time_path = paths
    episodes = list(range(1, len(rewards) + 1))

    # Reward plot
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label="Episode Reward", alpha=0.6)
    if len(rewards) >= moving_avg_window:
        moving_avg = np.convolve(rewards, np.ones(moving_avg_window) / moving_avg_window, mode="valid")
        plt.plot(episodes[moving_avg_window - 1:], moving_avg, label=f"{moving_avg_window}-Episode MA", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(reward_path, bbox_inches="tight")
    plt.close()

    # Episode time plot
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, episode_times, label="Episode Time (s)", alpha=0.6)
    plt.xlabel("Episode")
    plt.ylabel("Time (seconds)")
    plt.title("Episode Duration Over Training")
    plt.legend()
    plt.grid(True)
    plt.savefig(time_path, bbox_inches="tight")
    plt.close()

def linear_schedule(initial_value: float) -> callable:
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def train_lunar_lander(config: TrainingConfig, directories: Dict[str, str]) -> Tuple[Optional[str], Optional[str], List[float]]:
    """Train PPO agent on LunarLander-v3, saving model and VecNormalize stats."""
    # Setup environment
    vec_env_cls = SubprocVecEnv if config.n_envs > 1 else DummyVecEnv
    env = make_vec_env(config.env_id, n_envs=config.n_envs, vec_env_cls=vec_env_cls, seed=int(time.time() * 1000) % (2**32 - 1))
    env = VecNormalize(env, norm_obs=True, norm_reward=False, gamma=config.gamma)

    # Initialise model
    lr = linear_schedule(config.learning_rate) if config.use_lr_schedule else config.learning_rate
    logger = configure(directories["logs"], ["csv", "tensorboard"])
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

    # Train
    callback = TrainingProgressCallback(config.timesteps, config.n_envs)
    start_time = time.time()
    model.learn(total_timesteps=config.timesteps, callback=callback)
    training_time = time.time() - start_time
    rewards = callback.episode_rewards
    episode_times = callback.episode_times
    episode_lengths = callback.episode_lengths

    # Save model and stats
    model_path = os.path.join(directories["zip_models"], f"lunar_lander_ppo_{model.num_timesteps}steps.zip")
    stats_path = os.path.join(directories["zip_models"], config.vec_normalize_filename)
    torch_path = os.path.join(directories["pth_models"], f"lunar_lander_policy_{model.num_timesteps}steps.pth")
    model.save(model_path)
    if isinstance(model.get_env(), VecNormalize):
        model.get_env().save(stats_path)
    torch.save(model.policy.state_dict(), torch_path)

    # Save plot data
    if rewards:
        reward_path = os.path.join(directories["plots"], f"reward_curve_{model.num_timesteps}steps.png")
        time_path = os.path.join(directories["plots"], f"episode_times_{model.num_timesteps}steps.png")
        create_plots(rewards, episode_times, (reward_path, time_path), callback.moving_avg_window)
        csv_path = os.path.join(directories["plots"], f"episode_data_{model.num_timesteps}steps.csv")
        with open(csv_path, "w") as f:
            f.write("episode,reward,time_seconds,steps\n")
            for i, (r, t, s) in enumerate(zip(rewards, episode_times, episode_lengths)):
                f.write(f"{i+1},{r:.4f},{t:.4f},{s}\n")

    # Final summary
    summary = callback.get_summary()
    print("\nTRAINING SUMMARY")
    print(f"Time: {training_time/60:.2f}m | Steps: {summary['total_steps']}/{config.timesteps}")
    print(f"Episodes: {summary['total_episodes']} | Avg Reward: {summary['avg_reward']:.2f}")
    if len(rewards) >= callback.moving_avg_window:
        print(f"Last {callback.moving_avg_window} Avg: {np.mean(rewards[-callback.moving_avg_window:]):.2f}")
    print(f"Best: {summary['best_reward']:.2f} | Avg Steps: {summary['avg_steps_per_episode']:.1f}")

    env.close()
    return model_path, stats_path, rewards

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"run_{run_time}"
    run_dir = os.path.join(base_dir, run_dir_name)

    # Create directory structure
    os.makedirs(os.path.join(run_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'models', 'zip'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'models', 'pth'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'plots'), exist_ok=True)

    directories = {
        "logs": os.path.join(run_dir, 'logs'),
        "zip_models": os.path.join(run_dir, 'models', 'zip'),
        "pth_models": os.path.join(run_dir, 'models', 'pth'),
        "plots": os.path.join(run_dir, 'plots')
    }

    train_lunar_lander(TrainingConfig(), directories)  

if __name__ == "__main__":
    main()