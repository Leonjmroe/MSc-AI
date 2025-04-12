import gymnasium as gym
import torch
import os
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv # Removed VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
# Import Atari wrapper for type hinting / explicit use if needed, though make_vec_env handles it
from stable_baselines3.common.atari_wrappers import AtariWrapper
from typing import Dict, List, Tuple, Optional, Callable
import warnings
import traceback

# Filter warnings individually instead of using a tuple
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class TrainingConfig:
    """Training configuration parameters for MsPacman."""
    def __init__(self):
        self.algorithm = "PPO"
        # Atari often requires more timesteps for good performance
        self.timesteps = 1_000_000 # Start with 1M, but 10M+ is common for Atari
        self.n_envs = 8 # Number of parallel environments
        self.learning_rate = 2.5e-4 # Common learning rate for Atari PPO
        self.n_steps = 128 # Steps per environment per update (SB3 default for PPO Atari)
        self.batch_size = self.n_steps * self.n_envs # PPO batch size
        self.n_epochs = 4 # Number of optimization epochs per batch (SB3 default for PPO Atari)
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.1 # Clip range for PPO (SB3 default for PPO Atari)
        self.ent_coef = 0.01 # Entropy coefficient (SB3 default for PPO Atari)
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        # Remove MLP policy_kwargs, SB3 CnnPolicy default is usually good
        self.policy_kwargs = None # Use default CNN architecture
        self.env_id = "ALE/MsPacman-v5" # Use ALE wrapper explicitely
        # No VecNormalize filename needed
        # self.vec_normalize_filename = "vecnormalize.pkl"
        self.use_lr_schedule = True # Keep linear schedule option

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
        # Use lists to store ongoing episode data for each env
        self.current_rewards = [0.0] * n_envs
        self.current_lengths = [0] * n_envs
        self.current_episode_start_times = [time.time()] * n_envs
        self.training_start_time = time.time()
        self.last_log_time = self.training_start_time
        self.total_episodes_started = 0 # Track total episodes started across all envs

    def _on_step(self) -> bool:
        # rewards and dones are numpy arrays of shape (n_envs,)
        for i in range(self.n_envs):
            self.current_rewards[i] += self.locals["rewards"][i]
            self.current_lengths[i] += 1

            # Check for episode termination (terminated or truncated)
            # The 'infos' dictionary contains termination signals per environment
            terminated = self.locals["dones"][i]
            truncated = self.locals["infos"][i].get("TimeLimit.truncated", False)

            if terminated or truncated:
                self.total_episodes_started += 1
                ep_info = self.locals["infos"][i].get("episode") # SB3 automatically logs this for VecEnvs
                if ep_info: # Use SB3's logged info if available
                    reward = ep_info["r"]
                    length = ep_info["l"]
                    ep_time = ep_info["t"]
                else: # Fallback if SB3 info isn't there (less likely but safe)
                    reward = self.current_rewards[i]
                    length = self.current_lengths[i]
                    ep_time = time.time() - self.current_episode_start_times[i]

                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
                self.episode_times.append(ep_time)

                # Reset trackers for this specific environment
                self.current_rewards[i] = 0.0
                self.current_lengths[i] = 0
                self.current_episode_start_times[i] = time.time()

                # Log progress periodically based on episodes finished
                if self.verbose > 0 and self.total_episodes_started % (self.n_envs * 2) == 0 : # Log roughly every 2*n_envs episodes
                     self._log_progress()

        return True

    def _log_progress(self):
        elapsed = time.time() - self.training_start_time
        # Ensure num_timesteps is correctly accessed (it's updated by the parent class)
        steps_per_second = (self.num_timesteps - getattr(self, '_last_log_step', 0)) / (time.time() - self.last_log_time + 1e-8)
        self._last_log_step = self.num_timesteps # Store current step count for next SPS calc

        self.last_log_time = time.time()
        completion = (self.num_timesteps / self.total_timesteps) * 100
        # Ensure window is not larger than the number of rewards collected
        current_window = min(self.moving_avg_window, len(self.episode_rewards))
        moving_avg = np.mean(self.episode_rewards[-current_window:]) if self.episode_rewards else 0.0

        print(
            f"Steps: {self.num_timesteps}/{self.total_timesteps} ({completion:.1f}%) | "
            f"Episodes: {self.total_episodes_started} | "
            f"Mean Reward ({current_window}ep): {moving_avg:.1f} | "
            f"SPS: {int(steps_per_second)} | "
            f"Elapsed: {elapsed/60:.1f}m"
        )

    def get_summary(self) -> Dict[str, float]:
        total_time = time.time() - self.training_start_time
        return {
            "total_steps": self.num_timesteps,
            "total_episodes": self.total_episodes_started, # Use total episodes started
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
    """Return a linear learning rate schedule function based on progress remaining."""
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end).
        :param progress_remaining: (float) 1.0 at the start of training, 0.0 at the end
        :return: (float) current learning rate
        """
        return initial_value * progress_remaining
    return func


# Rename function to be more general or specific to Atari/MsPacman
def train_mspacman(config: TrainingConfig, directories: Dict[str, str]) -> Tuple[Optional[str], List[float]]:
    """Train PPO agent on MsPacman-v5 using Atari wrappers."""
    print(f"Training {config.algorithm} on {config.env_id} ({config.device.upper()})")

    # Setup environment: make_vec_env handles Atari preprocessing with appropriate wrappers
        # --- DIAGNOSTIC BLOCK START ---
    print("\n--- Running Pre-check Diagnostics ---")
    import gymnasium as gym
    # Try importing ALE/Shimmy stuff directly to see if they exist
    try:
        import ale_py
        import shimmy
        print(f"Found ale_py version: {ale_py.__version__}")
        print(f"Found shimmy version: {shimmy.__version__}")
        print(f"Found gymnasium version: {gym.__version__}")
        print("Attempting to list all registered Gymnasium environments...")
        all_envs = gym.envs.registry.keys()
        print(f"Total environments registered: {len(all_envs)}")

        ale_envs = [env_id for env_id in all_envs if env_id.startswith("ALE/")]
        if ale_envs:
            print(f"Found {len(ale_envs)} environments starting with 'ALE/'. Sample: {ale_envs[:10]}") # Show more samples
            if "ALE/MsPacman-v5" in ale_envs:
                print("SUCCESS: 'ALE/MsPacman-v5' is present in the registry!")
            else:
                 print("ERROR: 'ALE/MsPacman-v5' NOT FOUND in registered 'ALE/' environments!")
                 print("This suggests shimmy might not be registering ALE envs correctly.")
        else:
            print("ERROR: No environments starting with 'ALE/' found in Gymnasium registry!")
            print("This strongly suggests the 'ale-py'/'shimmy' installation or entry points are broken.")

        # Check for non-ALE MsPacman as well
        non_ale_mspacman = [env_id for env_id in all_envs if "MsPacman" in env_id and not env_id.startswith("ALE/")]
        if non_ale_mspacman:
            print(f"Found non-ALE MsPacman versions: {non_ale_mspacman}")
        else:
            print("Also did not find any non-ALE MsPacman versions registered.")


        # Try making the env directly here *before* make_vec_env
        print("\nAttempting gym.make('ALE/MsPacman-v5') directly...")
        env_test = gym.make("ALE/MsPacman-v5")
        print("SUCCESS: Direct gym.make('ALE/MsPacman-v5') succeeded!")
        env_test.close()
        print("Direct gym.make seems okay, issue might be specific to make_vec_env interaction?")

    except ImportError as e:
        print(f"ERROR: Failed to import 'ale_py' or 'shimmy'. Check installation.")
        print(f"Import Error: {e}")
        traceback.print_exc()
    except gym.error.NamespaceNotFound as e:
        print(f"ERROR: gym.make failed directly with NamespaceNotFound: {e}")
        print("This confirms the core registration issue.")
        traceback.print_exc()
    except Exception as e:
        print(f"ERROR during diagnostic pre-check: {e}")
        traceback.print_exc()

    print("--- Finished Pre-check Diagnostics ---\n")
    # --- DIAGNOSTIC BLOCK END ---


    # Setup environment: make_vec_env handles Atari preprocessing...
    vec_env_cls = DummyVecEnv # Keep DummyVecEnv forced for debugging
    try:
        print(f"Attempting make_vec_env with vec_env_cls={vec_env_cls.__name__}...") # Add log
        # Original make_vec_env call follows
        env = make_vec_env(
            config.env_id,
            n_envs=config.n_envs,
            vec_env_cls=vec_env_cls,
            seed=int(time.time() * 1000) % (2**32 - 1)
        )
        # ... rest of the try block ...
        print(f"Created VecEnv with {config.n_envs} environments.")
        print(f"Observation Space: {env.observation_space}")
        print(f"Action Space: {env.action_space}")

    except Exception as e:
        print(f"Environment setup failed: {e}")
        traceback.print_exc()
        return None, []

    # Initialize model
    lr = linear_schedule(config.learning_rate) if config.use_lr_schedule else config.learning_rate
    try:
        logger = configure(directories["logs"], ["csv", "tensorboard"])
        model = PPO(
            "CnnPolicy",  # Use CNN policy for image inputs
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
            policy_kwargs=config.policy_kwargs, # Use default CNN architecture
            seed=int(time.time() * 1000 + 1) % (2**32 - 1),
            verbose=0 # Set verbose=0 to reduce SB3 console output, rely on callback
        )
        model.set_logger(logger)
        print("PPO Model with CnnPolicy initialized.")
    except Exception as e:
        print(f"Model setup failed: {e}")
        traceback.print_exc()
        env.close()
        return None, []

    # Train
    callback = TrainingProgressCallback(config.timesteps, config.n_envs, verbose=1)
    start_time = time.time()
    print("Starting training...")
    try:
        # Pass reset_num_timesteps=True if needed, False continues numbering if model was loaded
        model.learn(total_timesteps=config.timesteps, callback=callback, reset_num_timesteps=True)
    except Exception as e:
        print(f"Training failed: {e}")
        traceback.print_exc()
        rewards = callback.episode_rewards
        env.close()
        return None, rewards
    finally:
         # Ensure environment is closed even if training is interrupted
        try:
            env.close()
            print("Training environment closed.")
        except Exception as e:
            print(f"Error closing environment: {e}")


    training_time = time.time() - start_time
    rewards = callback.episode_rewards
    episode_times = callback.episode_times
    episode_lengths = callback.episode_lengths

    # Save model (no VecNormalize stats to save)
    model_zip_path = os.path.join(directories["zip_models"], f"mspacman_ppo_{model.num_timesteps}steps.zip")
    torch_path = os.path.join(directories["pth_models"], f"mspacman_policy_{model.num_timesteps}steps.pth")

    try:
        model.save(model_zip_path)
        # No VecNormalize stats to save for Atari usually
        # vec_env = model.get_env() # Would get the VecEnv wrapper
        torch.save(model.policy.state_dict(), torch_path)
        print(f"Saved: model={model_zip_path}, policy={torch_path}")
    except Exception as e:
        print(f"Save failed: {e}")

    # Summarize
    summary = callback.get_summary()
    print("\n--- TRAINING SUMMARY ---")
    print(f"Environment: {config.env_id}")
    print(f"Total Time: {training_time/60:.2f}m | Total Steps: {summary['total_steps']}/{config.timesteps}")
    print(f"Total Episodes Finished: {len(rewards)} (Started: {summary['total_episodes']})") # Distinguish finished vs started
    print(f"Average Reward (all finished ep): {summary['avg_reward']:.2f}")
    if len(rewards) >= callback.moving_avg_window:
         print(f"Avg Reward (last {min(callback.moving_avg_window, len(rewards))} ep): {np.mean(rewards[-callback.moving_avg_window:]):.2f}")
    print(f"Best Reward: {summary['best_reward']:.2f}")
    print(f"Average Steps per Episode: {summary['avg_steps_per_episode']:.1f}")
    print(f"Average Time per Episode: {summary['avg_episode_time']:.2f}s")
    print("------------------------")


    # Plot and save data
    if rewards:
        try:
            reward_plot_path = os.path.join(directories["plots"], f"reward_curve_{model.num_timesteps}steps.png")
            time_plot_path = os.path.join(directories["plots"], f"episode_times_{model.num_timesteps}steps.png")
            create_plots(rewards, episode_times, (reward_plot_path, time_plot_path), callback.moving_avg_window)
            print(f"Plots saved to: {directories['plots']}")

            csv_path = os.path.join(directories["plots"], f"episode_data_{model.num_timesteps}steps.csv")
            with open(csv_path, "w") as f:
                f.write("episode,reward,time_seconds,steps\n")
                for i, (r, t, s) in enumerate(zip(rewards, episode_times, episode_lengths)):
                    f.write(f"{i+1},{r:.4f},{t:.4f},{s}\n")
            print(f"Episode data saved to: {csv_path}")
        except Exception as e:
            print(f"Plotting/Saving data failed: {e}")

    # Return only model path and rewards, as there's no stats path
    return model_zip_path, rewards


def evaluate_model(model_path: str, num_episodes: int = 10, env_id: str = "ALE/MsPacman-v5", render: bool = True) -> Optional[List[float]]:
    """
    Evaluates the trained PPO agent on MsPacman.

    Args:
        model_path: Path to the saved SB3 model (.zip).
        num_episodes: Number of episodes to run for evaluation.
        env_id: The environment ID to use for evaluation (should match training).
        render: Whether to render the environment during evaluation.

    Returns:
        A list of rewards obtained in each evaluation episode, or None if loading fails.
    """
    print(f"\n--- Evaluating Model ---")
    print(f"Model: {model_path}")
    print(f"Environment: {env_id}")
    print(f"Episodes: {num_episodes}")
    print(f"Render: {render}")
    print(f"------------------------")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    eval_env = None # Initialize to None for cleanup
    try:
        # 1. Create the evaluation environment *with the same wrappers* as training.
        # Using make_vec_env with n_envs=1 is a reliable way to ensure this.
        # For rendering, make_vec_env might need help or use a single env directly.
        # Let's try creating a single env and wrapping it if rendering is needed.

        if render:
             # Create single env for rendering
             env_raw = gym.make(env_id, render_mode="human")
             # Apply necessary wrappers manually if make_vec_env's auto-wrapping isn't used
             # Note: This assumes AtariPreprocessing and FrameStack are the key wrappers SB3 uses.
             # You might need to verify the exact wrappers applied during training if issues arise.
             # env_wrapped = AtariWrapper(env_raw) # SB3's AtariWrapper includes preprocessing and frame stack
             # eval_env = DummyVecEnv([lambda: env_wrapped]) # Wrap in DummyVecEnv for SB3 model compatibility
             # Easier: Use make_vec_env and hope it handles render_mode correctly, or accept it might not render
             eval_env = make_vec_env(env_id, n_envs=1, vec_env_cls=DummyVecEnv,
                                      env_kwargs={"render_mode": "human" if render else None})

        else:
             # No rendering needed, make_vec_env is fine
             eval_env = make_vec_env(env_id, n_envs=1, vec_env_cls=DummyVecEnv)

        print(f"Evaluation environment created. Observation Space: {eval_env.observation_space}")

        # 2. Load the PPO model
        print(f"Loading PPO model from: {model_path} with the evaluation environment...")
        model = PPO.load(model_path, env=eval_env, device='auto')
        # No need for model.set_env(eval_env) anymore
        print("Model loaded with evaluation environment.")


    except FileNotFoundError as e:
         print(f"Error: Required file not found: {e}")
         if eval_env: eval_env.close()
         return None
    except Exception as e:
        print(f"\nError during model loading or environment setup for evaluation:")
        traceback.print_exc() # Print detailed traceback
        if eval_env: eval_env.close() # Ensure environment is closed on error
        return None

    # --- Evaluation Loop ---
    rewards: List[float] = []
    episode_lengths: List[int] = []

    try: # Add try...finally for robust environment closing
        for episode in range(num_episodes):
            # obs is a numpy array of shape (1, *obs_shape) for VecEnv
            obs = eval_env.reset()
            episode_reward: float = 0.0
            steps: int = 0
            terminated: bool = False
            truncated: bool = False # Atari envs might use truncation

            print(f"\nStarting Evaluation Episode {episode + 1}/{num_episodes}...")
            while not (terminated or truncated):
                # Use deterministic actions for evaluation
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated_array, info_dict = eval_env.step(action)

                terminated = terminated_array[0]
                # Check for truncation info (might be in 'TimeLimit.truncated' or just 'truncated')
                truncated = info_dict[0].get("TimeLimit.truncated", False) or info_dict[0].get("truncated", False)

                episode_reward += reward[0] # Reward is shape (1,)
                steps += 1

                # Rendering is handled by render_mode='human' if enabled during env creation
                # Optional small delay for smoother viewing if rendering
                if render: time.sleep(0.02) # Small delay for visibility

            print(f"Episode {episode + 1}: Finished. Reward = {episode_reward:.1f}, Steps = {steps}")
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
    else:
        print("No episodes were completed during evaluation (or an error occurred).")
    print("--------------------------")

    return rewards


def main():
    """Run training and optional evaluation for MsPacman."""
    # Ensure the script directory is used for relative paths
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    print(f"Using base directory: {base_dir}")

    config = TrainingConfig()
    directories = setup_directories(base_dir)

    # Train the model - returns model path and rewards list
    model_path, training_rewards = train_mspacman(config, directories)

    if not model_path:
        print("Training did not complete successfully or failed to save the model.")
        return

    print(f"\nTraining finished. Model saved at: {model_path}")

    # Ask user if they want to evaluate
    while True:
        eval_choice = input("Evaluate the trained model? (y/n): ").lower().strip()
        if eval_choice in ['y', 'n']:
            break
        print("Invalid input. Please enter 'y' or 'n'.")

    if eval_choice == "y":
        # Ask if rendering is desired
        while True:
            render_choice = input("Render the evaluation episodes? (y/n): ").lower().strip()
            if render_choice in ['y', 'n']:
                break
            print("Invalid input. Please enter 'y' or 'n'.")

        render_eval = (render_choice == 'y')
        num_eval_episodes = 5 # Evaluate fewer episodes if rendering, or make configurable
        print(f"Starting evaluation for {num_eval_episodes} episodes...")
        evaluate_model(model_path,
                       num_episodes=num_eval_episodes,
                       env_id=config.env_id,
                       render=render_eval)
    else:
        print("Skipping evaluation.")

if __name__ == "__main__":
    # Check for available hardware accelerator
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Using MPS (Apple Silicon GPU) device.")
        # Simple check to ensure MPS works, might fail on older macOS/PyTorch combinations
        try:
            _ = torch.tensor([1.0, 2.0], device="mps")
            print("MPS device check successful.")
        except Exception as e:
            print(f"MPS device check failed: {e}. Falling back to CPU.")
            # Force CPU if MPS check fails, preventing errors later
            TrainingConfig.device = "cpu"
    elif torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU device.")

    # Install necessary packages reminder
    print("\nEnsure you have installed the required packages:")
    print("pip install stable-baselines3[extra] torch gymnasium[atari] ale-py shimmy[atari] matplotlib")
    print("-" * 30)

    main()