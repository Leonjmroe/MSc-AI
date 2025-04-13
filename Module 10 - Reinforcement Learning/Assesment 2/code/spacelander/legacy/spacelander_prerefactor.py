import gymnasium as gym
import torch
import os
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from typing import Dict, List, Tuple, Optional, Callable
import warnings
import traceback

# Filter warnings individually instead of using a tuple
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class TrainingConfig:
    """Training configuration parameters."""
    def __init__(self):
        self.algorithm = "PPO"
        self.timesteps = 1000000
        self.n_envs = 8
        self.learning_rate = 3e-4
        self.n_steps = 2048 // self.n_envs
        self.batch_size = 64 * self.n_envs
        self.n_epochs = 10
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.policy_kwargs = {"net_arch": [dict(pi=[128, 128], vf=[128, 128])], "activation_fn": torch.nn.ReLU}
        self.env_id = "LunarLander-v3" 
        self.vec_normalize_filename = "vecnormalize.pkl"
        self.use_lr_schedule = True

def setup_directories(base_dir: str) -> Dict[str, str]:
    """Create directories for models, plots, and logs."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    directories = {
        "run_dir": run_dir,
        "models": os.path.join(run_dir, "models"),
        "zip_models": os.path.join(run_dir, "models", "zip"),
        "pth_models": os.path.join(run_dir, "models", "pth"),
        "plots": os.path.join(run_dir, "plots"),
        "logs": os.path.join(run_dir, "logs"),
    }
    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)
    print(f"Created directories: {run_dir}")
    return directories

class TrainingProgressCallback(BaseCallback):
    """Callback to monitor and log training progress."""
    def __init__(self, total_timesteps: int, n_envs: int, moving_avg_window: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.moving_avg_window = moving_avg_window
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.current_rewards = np.zeros(n_envs)
        self.current_lengths = np.zeros(n_envs, dtype=int)
        self.current_episode_start_times = np.full(n_envs, time.time())
        self.training_start_time = time.time()
        self.last_log_time = self.training_start_time

    def _on_step(self) -> bool:
        self.current_rewards += self.locals["rewards"]
        self.current_lengths += 1

        for i, done in enumerate(self.locals["dones"]):
            if done:
                ep_info = self.locals["infos"][i].get("episode", {})
                reward = ep_info.get("r", self.current_rewards[i])
                length = ep_info.get("l", self.current_lengths[i])
                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
                self.episode_times.append(time.time() - self.current_episode_start_times[i])
                self.current_rewards[i] = 0
                self.current_lengths[i] = 0
                self.current_episode_start_times[i] = time.time()

                if self.verbose and len(self.episode_rewards) % (self.n_envs * 2) == 0:
                    self._log_progress()

        return True

    def _log_progress(self):
        elapsed = time.time() - self.training_start_time
        steps_per_second = self.num_timesteps / (time.time() - self.last_log_time + 1e-8)
        self.last_log_time = time.time()
        completion = (self.num_timesteps / self.total_timesteps) * 100
        moving_avg = np.mean(self.episode_rewards[-self.moving_avg_window:]) if self.episode_rewards else -np.inf

        print(
            f"Steps: {self.num_timesteps}/{self.total_timesteps} ({completion:.1f}%) | "
            f"Episodes: {len(self.episode_rewards)} | "
            f"Mean Reward ({self.moving_avg_window}ep): {moving_avg:.1f} | "
            f"SPS: {int(steps_per_second)} | "
            f"Elapsed: {elapsed/60:.1f}m"
        )

    def get_summary(self) -> Dict[str, float]:
        total_time = time.time() - self.training_start_time
        return {
            "total_steps": self.num_timesteps,
            "total_episodes": len(self.episode_rewards),
            "total_training_time": total_time,
            "avg_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "avg_episode_time": np.mean(self.episode_times) if self.episode_times else 0,
            "avg_steps_per_episode": np.mean(self.episode_lengths) if self.episode_lengths else 0,
            "best_reward": max(self.episode_rewards) if self.episode_rewards else -np.inf,
            "completion_percentage": (self.num_timesteps / self.total_timesteps) * 100,
        }

def create_plots(rewards: List[float], episode_times: List[float], output_paths: Tuple[str, str], window_size: int = 100) -> Tuple[str, str]:
    """Generate and save reward and time plots."""
    if not rewards:
        print("No data to plot.")
        return "", ""

    reward_path, time_path = output_paths
    window = min(window_size, len(rewards))
    plt.style.use("seaborn-v0_8-whitegrid")

    # Reward plot
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label="Episode Reward", alpha=0.5, color="cornflowerblue")
    if window > 1:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(range(window - 1, len(rewards)), moving_avg, "r-", label=f"{window}-Episode Moving Avg", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(reward_path, dpi=300)
    plt.close()

    # Time plot
    if episode_times:
        plt.figure(figsize=(12, 6))
        plt.plot(episode_times, label="Episode Time (s)", alpha=0.6, color="mediumseagreen")
        if window > 1 and len(episode_times) >= window:
            time_avg = np.convolve(episode_times, np.ones(window) / window, mode="valid")
            plt.plot(range(window - 1, len(episode_times)), time_avg, "darkorange", label=f"{window}-Episode Moving Avg", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Time (seconds)")
        plt.title("Episode Duration")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(time_path, dpi=300)
        plt.close()
    else:
        time_path = ""

    return reward_path, time_path

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Return a linear learning rate schedule."""
    return lambda progress: initial_value * progress



def train_lunar_lander(config: TrainingConfig, directories: Dict[str, str]) -> Tuple[Optional[str], Optional[str], List[float]]:
    """Train PPO agent on LunarLander-v3, saving model and VecNormalize stats."""
    print(f"Training {config.algorithm} on {config.env_id} ({config.device.upper()})")

    # Setup environment
    vec_env_cls = SubprocVecEnv if config.n_envs > 1 else DummyVecEnv
    try:
        env = make_vec_env(config.env_id, n_envs=config.n_envs, vec_env_cls=vec_env_cls, seed=int(time.time() * 1000) % (2**32 - 1))
        env = VecNormalize(env, norm_obs=True, norm_reward=False, gamma=config.gamma)
        print(f"Initial VecNorm Stats: Obs Mean {env.obs_rms.mean[:4]}, Var {env.obs_rms.var[:4]}")
    except Exception as e:
        print(f"Environment setup failed: {e}")
        return None, None, []

    # Initialize model
    lr = linear_schedule(config.learning_rate) if config.use_lr_schedule else config.learning_rate
    try:
        logger = configure(directories["logs"], ["csv", "tensorboard"])
        model = PPO(
            "MlpPolicy", env, learning_rate=lr, n_steps=config.n_steps, batch_size=config.batch_size,
            n_epochs=config.n_epochs, gamma=config.gamma, gae_lambda=config.gae_lambda,
            clip_range=config.clip_range, ent_coef=config.ent_coef, vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm, device=config.device, policy_kwargs=config.policy_kwargs,
            seed=int(time.time() * 1000 + 1) % (2**32 - 1)
        )
        model.set_logger(logger)
    except Exception as e:
        print(f"Model setup failed: {e}")
        env.close()
        return None, None, []

    # Train
    callback = TrainingProgressCallback(config.timesteps, config.n_envs)
    start_time = time.time()
    try:
        model.learn(total_timesteps=config.timesteps, callback=callback)
    except Exception as e:
        print(f"Training failed: {e}")
        rewards = callback.episode_rewards
        env.close()
        return None, None, rewards

    training_time = time.time() - start_time
    rewards = callback.episode_rewards
    episode_times = callback.episode_times
    episode_lengths = callback.episode_lengths

    # Save model and stats
    model_path = os.path.join(directories["zip_models"], f"lunar_lander_ppo_{model.num_timesteps}steps.zip")
    stats_path = os.path.join(directories["zip_models"], config.vec_normalize_filename)
    torch_path = os.path.join(directories["pth_models"], f"lunar_lander_policy_{model.num_timesteps}steps.pth")

    try:
        model.save(model_path)
        vec_env = model.get_env()
        if isinstance(vec_env, VecNormalize):
            print(f"Saving VecNorm Stats: Obs Mean {vec_env.obs_rms.mean[:4]}, Var {vec_env.obs_rms.var[:4]}")
            vec_env.save(stats_path)
        else:
            print("Warning: No VecNormalize env found for saving stats.")
        torch.save(model.policy.state_dict(), torch_path)
        print(f"Saved: model={model_path}, stats={stats_path}, policy={torch_path}")
    except Exception as e:
        print(f"Save failed: {e}")

    # Summarize
    summary = callback.get_summary()
    print("\nTRAINING SUMMARY")
    print(f"Time: {training_time/60:.2f}m | Steps: {summary['total_steps']}/{config.timesteps}")
    print(f"Episodes: {summary['total_episodes']} | Avg Reward: {summary['avg_reward']:.2f}")
    if len(rewards) >= callback.moving_avg_window:
        print(f"Last {callback.moving_avg_window} Avg: {np.mean(rewards[-callback.moving_avg_window:]):.2f}")
    print(f"Best: {summary['best_reward']:.2f} | Avg Steps: {summary['avg_steps_per_episode']:.1f}")

    # Plot and save data
    if rewards:
        try:
            reward_path = os.path.join(directories["plots"], f"reward_curve_{model.num_timesteps}steps.png")
            time_path = os.path.join(directories["plots"], f"episode_times_{model.num_timesteps}steps.png")
            create_plots(rewards, episode_times, (reward_path, time_path), callback.moving_avg_window)
            print(f"Plots: {reward_path}")

            csv_path = os.path.join(directories["plots"], f"episode_data_{model.num_timesteps}steps.csv")
            with open(csv_path, "w") as f:
                f.write("episode,reward,time_seconds,steps\n")
                for i, (r, t, s) in enumerate(zip(rewards, episode_times, episode_lengths)):
                    f.write(f"{i+1},{r:.4f},{t:.4f},{s}\n")
            print(f"Data: {csv_path}")
        except Exception as e:
            print(f"Plotting failed: {e}")

    env.close()
    return model_path, stats_path, rewards


def evaluate_model(model_path: str, stats_path: str, num_episodes: int = 10, env_id: str = "LunarLander-v3", render: bool = True) -> Optional[List[float]]:
    """
    Evaluates the trained PPO agent, ensuring correct VecNormalize loading.

    Args:
        model_path: Path to the saved SB3 model (.zip).
        stats_path: Path to the saved VecNormalize statistics (.pkl).
        num_episodes: Number of episodes to run for evaluation.
        env_id: The environment ID to use for evaluation.
        render: Whether to render the environment during evaluation.

    Returns:
        A list of rewards obtained in each evaluation episode, or None if loading fails.
    """
    print(f"\n--- Evaluating Model ---")
    print(f"Model: {model_path}")
    print(f"Stats: {stats_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Render: {render}")
    print(f"------------------------")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    if not os.path.exists(stats_path):
        print(f"Error: VecNormalize statistics file not found at {stats_path}")
        print("Cannot evaluate accurately without normalization stats. Aborting.")
        return None

    eval_env = None # Initialize to None for cleanup
    try:
        # 1. Create the base environment (without normalization initially)
        # Use DummyVecEnv for evaluation as we only need one instance
        eval_env_raw = make_vec_env(env_id, n_envs=1, vec_env_cls=DummyVecEnv,
                                     env_kwargs={"render_mode": "human" if render else None})

        # 2. Load the VecNormalize statistics *into* the raw environment wrapper
        print(f"Loading VecNormalize stats from: {stats_path}")
        # This returns the VecNormalize object wrapping eval_env_raw
        eval_env = VecNormalize.load(stats_path, eval_env_raw)

        # 3. Crucially, set the loaded environment to evaluation mode
        eval_env.training = False
        # Deactivate reward normalization (we want to see the raw rewards)
        eval_env.norm_reward = False

        # Print some loaded stats for verification (optional debug)
        print(f"VecNormalize loaded. Obs Mean (sample): {eval_env.obs_rms.mean[:3]}")
        print(f"VecNormalize loaded. Obs Variance (sample): {eval_env.obs_rms.var[:3]}")
        print(f"Is eval_env in training mode? {eval_env.training}")

        # 4. Load the PPO model *separately* (let SB3 determine device or specify)
        print(f"Loading PPO model from: {model_path}")
        # Load the model directly with the environment to handle different number of environments
        model = PPO.load(model_path, env=eval_env)
        print("Model loaded with evaluation environment.")

        # Sanity check: Ensure the model is using the environment instance we prepared
        assert model.get_env() is eval_env, "Model's environment was not correctly set to the VecNormalize instance!"
        print("Model environment association verified.")


    except FileNotFoundError as e:
         print(f"Error: Required file not found: {e}")
         if eval_env: eval_env.close()
         return None
    except Exception as e:
        print(f"\nError during model or VecNormalize loading:")
        traceback.print_exc() # Print detailed traceback
        if eval_env: eval_env.close() # Ensure environment is closed on error
        return None

    # --- Evaluation Loop ---
    rewards: List[float] = []
    episode_lengths: List[int] = []
    successful_landings: int = 0

    try: # Add try...finally for robust environment closing
        for episode in range(num_episodes):
            obs = eval_env.reset()
            # Debug: Print initial observation (should be normalized)
            # print(f"Episode {episode + 1} Initial Obs (normalized): {obs[0][:4]}")

            episode_reward: float = 0.0
            steps: int = 0
            terminated: bool = False
            truncated: bool = False

            print(f"\nStarting Evaluation Episode {episode + 1}/{num_episodes}...")
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated_array, info_dict = eval_env.step(action)

                # Debug: Print observations during episode occasionally
                # if steps > 0 and steps % 100 == 0:
                #     print(f"  Step {steps}, Obs: {obs[0][:4]}, Reward: {reward[0]:.2f}")

                terminated = terminated_array[0]
                truncated = info_dict[0].get("TimeLimit.truncated", False) or info_dict[0].get("TimeLimitTruncated", False) # Handle slight variations

                episode_reward += reward[0]
                steps += 1

                # Rendering is handled by render_mode='human' if enabled
                # Optional small delay for smoother viewing if rendering
                # if render: time.sleep(0.01)

            # Check for success condition
            success = episode_reward > 200 # Standard LunarLander success threshold
            if success:
                print(f"Episode {episode + 1}: SUCCESS! Reward = {episode_reward:.1f}, Steps = {steps}")
                successful_landings += 1
            else:
                # Check for crash conditions (often large negative rewards)
                crash_threshold = -150 # Adjust as needed
                status = "CRASH?" if episode_reward < crash_threshold else "Finished"
                print(f"Episode {episode + 1}: {status} Reward = {episode_reward:.1f}, Steps = {steps}")


            rewards.append(episode_reward)
            episode_lengths.append(steps)

    except Exception as e:
        print("\nAn error occurred during the evaluation loop:")
        traceback.print_exc()
    finally:
        # IMPORTANT: Always close the environment
        if eval_env:
             print("\nClosing evaluation environment.")
             eval_env.close()

    # --- Evaluation Summary ---
    print("\n--- Evaluation Summary ---")
    if rewards:
        print(f"Average Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
        print(f"Min/Max Reward: {min(rewards):.2f} / {max(rewards):.2f}")
        print(f"Average Episode Length: {np.mean(episode_lengths):.1f} steps")
        print(f"Success Rate (>200 reward): {successful_landings / num_episodes * 100:.1f}% ({successful_landings}/{num_episodes})")
    else:
        print("No episodes were completed during evaluation (or an error occurred).")
    print("--------------------------")

    return rewards


def main():
    """Run training and optional evaluation."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = TrainingConfig()
    directories = setup_directories(base_dir)

    model_path, stats_path, _ = train_lunar_lander(config, directories)
    if not (model_path and stats_path):
        print("Training failed.")
        return

    if input("Evaluate model? (y/n): ").lower().strip() == "y":
        evaluate_model(model_path, stats_path, num_episodes=10, env_id=config.env_id)

if __name__ == "__main__":
    device = "MPS" if torch.backends.mps.is_available() else "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Using {device} device.")
    main()


