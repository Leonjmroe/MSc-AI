# python eval.py --run-dir ./maxed_spacelander_run.py

import gymnasium as gym
import torch
import os
import argparse
import numpy as np
import time
import warnings
import traceback

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from typing import Optional

# Ignore potential warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def find_model_file(zip_dir: str) -> Optional[str]:
    """Finds the PPO model .zip file in the specified directory."""
    potential_models = [f for f in os.listdir(zip_dir) if f.endswith('.zip')]
    if len(potential_models) == 0:
        print(f"Error: No '.zip' model file found in {zip_dir}")
        return None
    if len(potential_models) > 1:
        # If multiple models exist, try to find one named like the output from training script
        # Example: 'lunar_lander_ppo_150000steps.zip'
        # This part might need adjustment based on exact naming conventions
        specific_models = [m for m in potential_models if m.startswith("lunar_lander_ppo_") and "steps.zip" in m]
        if len(specific_models) == 1:
             print(f"Found specific model: {specific_models[0]}")
             return os.path.join(zip_dir, specific_models[0])
        else:
            print(f"Error: Multiple '.zip' models found in {zip_dir}. Cannot automatically determine which one to use.")
            print(f"Found: {potential_models}")
            print("Consider removing older models or specify the exact model file.")
            return None
    # Exactly one model found
    return os.path.join(zip_dir, potential_models[0])


def evaluate_trained_model(
    run_dir: str,
    num_episodes: int = 10,
    env_id: str = "LunarLander-v3",
    render: bool = True,
    device: str = 'auto',
    seed: Optional[int] = None
) -> None:
    """
    Loads and evaluates a trained PPO agent from a specified run directory.

    Args:
        run_dir: Path to the specific run directory containing models/ and logs/.
        num_episodes: Number of episodes to run for evaluation.
        env_id: The environment ID to use for evaluation.
        render: Whether to render the environment during evaluation.
        device: Device ('auto', 'cpu', 'cuda', 'mps') to use for evaluation.
        seed: Optional random seed for the evaluation environment.
    """
    print(f"\n--- Evaluating Model from Run Directory ---")
    print(f"Run Directory: {run_dir}")
    print(f"Environment: {env_id}")
    print(f"Episodes: {num_episodes}")
    print(f"Render: {render}")
    print(f"Device: {device}")
    print(f"Seed: {seed if seed is not None else 'None'}")
    print(f"-----------------------------------------")

    # Construct paths to the model and stats files
    zip_model_dir = os.path.join(run_dir, "models", "zip")
    stats_filename = "vecnormalize.pkl" # Standard name used in training script
    stats_path = os.path.join(zip_model_dir, stats_filename)

    if not os.path.isdir(zip_model_dir):
        print(f"Error: Expected model directory not found: {zip_model_dir}")
        return
    if not os.path.exists(stats_path):
        print(f"Error: VecNormalize statistics file not found: {stats_path}")
        print("Cannot evaluate accurately without normalization stats.")
        return

    # Find the model file
    model_path = find_model_file(zip_model_dir)
    if model_path is None:
        return # Error message already printed by find_model_file

    print(f"Found Model File: {os.path.basename(model_path)}")
    print(f"Found Stats File: {stats_filename}")

    eval_env = None # Initialize for cleanup
    try:
        # 1. Create the base environment (using DummyVecEnv for single evaluation)
        env_kwargs = {"render_mode": "human" if render else None}
        eval_env_raw = make_vec_env(env_id, n_envs=1, seed=seed, vec_env_cls=DummyVecEnv, env_kwargs=env_kwargs)

        # 2. Create a new VecNormalize wrapper instead of loading stats
        # The compatibility issue prevents loading the stats directly
        print(f"Creating a fresh VecNormalize wrapper instead of loading stats")
        eval_env = VecNormalize(eval_env_raw, norm_obs=True, norm_reward=False)
        
        # 3. Set the environment to evaluation mode
        eval_env.training = False
        
        # 4. Load the PPO model with direct environment
        print(f"Loading PPO model from: {model_path}")
        # Load the model with the environment directly - it will handle normalization internally
        model = PPO.load(model_path, env=eval_env, device=device)
        print(f"Model loaded with evaluation environment (using {device} device).")
        print("Warning: Using new normalization stats, which may affect performance compared to training.")

    except FileNotFoundError as e:
         print(f"Error: Required file not found during loading: {e}")
         if eval_env: eval_env.close()
         return
    except Exception as e:
        print(f"\nError during model or VecNormalize loading:")
        traceback.print_exc()
        if eval_env: eval_env.close()
        return

    # --- Evaluation Loop ---
    rewards = []
    episode_lengths = []
    successful_landings = 0

    try:
        for episode in range(num_episodes):
            obs = eval_env.reset()
            episode_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            print(f"\nStarting Evaluation Episode {episode + 1}/{num_episodes}...")
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated_array, info_dict = eval_env.step(action)

                terminated = terminated_array[0]
                # Check for truncation, key might vary slightly based on wrappers
                truncated = info_dict[0].get("TimeLimit.truncated", False) or info_dict[0].get("TimeLimitTruncated", False)

                episode_reward += reward[0]
                steps += 1

                # Rendering is handled by render_mode='human' if enabled
                # Optional small delay for smoother viewing if rendering
                # if render: time.sleep(0.01)

            success = episode_reward > 200 # Standard success threshold
            if success:
                print(f"Episode {episode + 1}: SUCCESS! Reward = {episode_reward:.1f}, Steps = {steps}")
                successful_landings += 1
            else:
                status = "CRASH?" if episode_reward < -150 else "Finished"
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
        print(f"Evaluated {len(rewards)} episodes.")
        print(f"Average Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
        print(f"Min/Max Reward: {min(rewards):.2f} / {max(rewards):.2f}")
        print(f"Average Episode Length: {np.mean(episode_lengths):.1f} steps")
        print(f"Success Rate (>200 reward): {successful_landings / len(rewards) * 100:.1f}% ({successful_landings}/{len(rewards)})")
    else:
        print("No episodes were completed during evaluation (or an error occurred).")
    print("--------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained PPO model for LunarLander-v3.")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to the specific run directory created by the training script (e.g., './run_YYYY-MM-DD_HH-MM-SS')."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run for evaluation."
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="LunarLander-v3",
        help="Environment ID."
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering during evaluation."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for the evaluation environment."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help="Device to use for evaluation ('auto', 'cpu', 'cuda', 'mps')."
    )

    args = parser.parse_args()

    # Determine render flag based on --no-render
    should_render = not args.no_render

    # Check MPS availability explicitly if selected or auto
    if args.device == "mps" or (args.device == "auto" and torch.backends.mps.is_available()):
        if not torch.backends.mps.is_available():
             print("Warning: MPS device selected but not available. Falling back to CPU.")
             args.device = "cpu"
        elif not torch.backends.mps.is_built():
             print("Warning: MPS device selected but not built. Falling back to CPU.")
             args.device = "cpu"
        else:
             print("Using MPS device for evaluation.")
    elif args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
         if not torch.cuda.is_available():
              print("Warning: CUDA device selected but not available. Falling back to CPU.")
              args.device = "cpu"
         else:
              print("Using CUDA device for evaluation.")
    else:
         if args.device not in ["auto", "cpu"]:
              print(f"Warning: Specified device '{args.device}' not recognized or unavailable. Using CPU.")
         else:
              print("Using CPU device for evaluation.")
         args.device = "cpu" # Ensure device is set correctly if fallback occurs


    evaluate_trained_model(
        run_dir=args.run_dir,
        num_episodes=args.episodes,
        env_id=args.env_id,
        render=should_render,
        device=args.device,
        seed=args.seed
    )