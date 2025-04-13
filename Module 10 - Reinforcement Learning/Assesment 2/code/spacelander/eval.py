import gymnasium as gym
import torch
import os
import argparse
import numpy as np
import time
import importlib.util
import traceback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

def find_config_file(run_dir):
    """Try to find the configuration file used for training"""
    # Check if there's a config.py or similar in the run directory
    potential_config_files = [
        os.path.join(run_dir, "config.py"),
        os.path.join(run_dir, "training_config.py")
    ]
    
    # Check if the run directory name hints at the config
    run_name = os.path.basename(os.path.normpath(run_dir))
    if run_name.startswith("config_") and "_run" in run_name:
        config_name = run_name.split("_run")[0]
        # Look in the workspace configs directory
        base_dir = os.path.dirname(os.path.dirname(run_dir))  # Go up two levels from run_dir
        config_dir = os.path.join(base_dir, "configs")
        if os.path.exists(config_dir):
            config_path = os.path.join(config_dir, f"{config_name}.py")
            potential_config_files.append(config_path)
    
    for config_path in potential_config_files:
        if os.path.exists(config_path):
            print(f"Found config file: {config_path}")
            return config_path
    
    return None

def load_config(config_path):
    """Load a training configuration from a Python file"""
    try:
        module_name = os.path.basename(config_path).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load specification for {config_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'TrainingConfig'):
            config = getattr(module, 'TrainingConfig')()
            print(f"Successfully loaded configuration from {config_path}")
            return config
        else:
            print(f"No TrainingConfig found in {config_path}")
            return None
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return None

def evaluate_model_from_policy(
    run_dir: str,
    num_episodes: int = 10,
    env_id: str = "LunarLander-v3",
    render: bool = True,
    device: str = 'auto',
    seed: int = None,
    save_video: bool = False
):
    """
    Evaluates a model by rebuilding it and loading just the policy weights
    to avoid version compatibility issues.
    """
    print(f"\n--- Evaluating Model with Fresh Environment ---")
    print(f"Run Directory: {run_dir}")
    print(f"Environment: {env_id}")
    print(f"Episodes: {num_episodes}")
    print(f"Render: {render}")
    print(f"Device: {device}")
    print(f"Save Video: {save_video}")
    print(f"-----------------------------------------")

    # Try to find and load the original training config
    config = None
    config_path = find_config_file(run_dir)
    if config_path:
        config = load_config(config_path)
        if config:
            # Use the env_id from the config if not explicitly provided
            if hasattr(config, 'env_id') and env_id == "LunarLander-v3":
                env_id = config.env_id
                print(f"Using environment from config: {env_id}")

    # Locate the policy weight file (PyTorch state dict)
    pth_dir = os.path.join(run_dir, "models", "pth")
    if not os.path.exists(pth_dir):
        print(f"Error: PTH models directory not found: {pth_dir}")
        return
    
    # Find a .pth file
    pth_files = [f for f in os.listdir(pth_dir) if f.endswith('.pth')]
    if not pth_files:
        print(f"Error: No .pth policy files found in {pth_dir}")
        return
    
    policy_path = os.path.join(pth_dir, pth_files[0])
    print(f"Found policy weights: {os.path.basename(policy_path)}")
    
    # Create output directory for videos if needed
    video_dir = None
    if save_video:
        video_dir = os.path.join(run_dir, "eval_videos")
        os.makedirs(video_dir, exist_ok=True)
        print(f"Videos will be saved to: {video_dir}")
    
    try:
        # Create a fresh environment
        env_kwargs = {}
        if render:
            env_kwargs["render_mode"] = "human"
        elif save_video:
            # If not rendering but saving video, use rgb_array
            env_kwargs["render_mode"] = "rgb_array"
        
        eval_env_raw = make_vec_env(env_id, n_envs=1, seed=seed, vec_env_cls=DummyVecEnv, 
                          env_kwargs=env_kwargs)
        
        # Wrap with video recorder if needed
        if save_video and not render:
            from gymnasium.wrappers import RecordVideo
            # Need to unwrap to access the base env for recording
            base_env = eval_env_raw.envs[0].unwrapped
            eval_env_raw.close()
            base_env = RecordVideo(base_env, video_dir)
            eval_env_raw = DummyVecEnv([lambda: base_env])
            print("Configured video recording")
        
        # Wrap with VecNormalize
        norm_kwargs = {}
        if config and hasattr(config, 'norm_obs'):
            norm_kwargs["norm_obs"] = config.norm_obs
            print(f"Using norm_obs from config: {config.norm_obs}")
        if config and hasattr(config, 'norm_reward'):
            norm_kwargs["norm_reward"] = config.norm_reward
            print(f"Using norm_reward from config: {config.norm_reward}")
        
        eval_env = VecNormalize(eval_env_raw, **norm_kwargs)
        eval_env.training = False  # Set evaluation mode
        
        # Create a new model with original hyperparameters if available
        model_kwargs = {"policy": "MlpPolicy", "env": eval_env}
        
        # Handle the 'auto' device setting
        if device == 'auto':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        
        # Now the device is guaranteed to be a specific one, not 'auto'
        model_kwargs["device"] = device
        print(f"Using device: {device}")
        
        if config:
            print("Applying original training hyperparameters to model:")
            hp_mapping = {
                'learning_rate': 'learning_rate',
                'gamma': 'gamma',
                'gae_lambda': 'gae_lambda',
                'n_steps': 'n_steps',
                'ent_coef': 'ent_coef',
                'vf_coef': 'vf_coef',
                'max_grad_norm': 'max_grad_norm',
                'batch_size': 'batch_size',
                'n_epochs': 'n_epochs',
                'clip_range': 'clip_range',
                'policy_kwargs': 'policy_kwargs'
            }
            for config_attr, model_attr in hp_mapping.items():
                if hasattr(config, config_attr):
                    value = getattr(config, config_attr)
                    model_kwargs[model_attr] = value
                    print(f"  - {model_attr}: {value}")
        
        print("Creating a fresh model...")
        model = PPO(**model_kwargs)
        
        # Load just the policy weights
        print(f"Loading policy weights from: {policy_path}")
        state_dict = torch.load(policy_path, map_location=torch.device(device))
        model.policy.load_state_dict(state_dict)
        print("Policy weights loaded successfully!")
        
        # Evaluation loop
        rewards = []
        episode_lengths = []
        successful_landings = 0
        
        print("\nStarting evaluation...")
        for episode in range(num_episodes):
            # Fix the reset interface - handle both old and new Gymnasium API
            try:
                # New Gymnasium API (returns tuple of obs and info)
                reset_result = eval_env.reset()
                if isinstance(reset_result, tuple) and len(reset_result) == 2:
                    obs, _ = reset_result
                else:
                    # Old interface (returns just obs)
                    obs = reset_result
            except Exception as e:
                print(f"Error during reset: {e}")
                traceback.print_exc()
                break
                
            episode_reward = 0
            steps = 0
            done = False
            
            print(f"\nEvaluation Episode {episode + 1}/{num_episodes}...")
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                
                # Handle both VecEnv step API versions
                try:
                    step_result = eval_env.step(action)
                    
                    # Check the shape of step_result to determine API version
                    if len(step_result) == 4:  # Old VecEnv API: obs, reward, done, info
                        obs, reward, done_array, info = step_result
                        done = done_array[0]
                    elif len(step_result) == 5:  # New Gymnasium API: obs, reward, terminated, truncated, info
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    
                    episode_reward += reward[0]
                    steps += 1
                except Exception as e:
                    print(f"Error during step: {e}")
                    traceback.print_exc()
                    done = True
                
                # Small delay for smoother viewing if rendering
                if render:
                    time.sleep(0.01)
            
            success = episode_reward > 200  # Success threshold
            if success:
                print(f"Episode {episode + 1}: SUCCESS! Reward = {episode_reward:.1f}, Steps = {steps}")
                successful_landings += 1
            else:
                status = "CRASH?" if episode_reward < -150 else "Finished"
                print(f"Episode {episode + 1}: {status} Reward = {episode_reward:.1f}, Steps = {steps}")
            
            rewards.append(episode_reward)
            episode_lengths.append(steps)
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()
    finally:
        if 'eval_env' in locals():
            eval_env.close()
    
    # Always initialize rewards if somehow it wasn't defined above
    if 'rewards' not in locals():
        rewards = []
    
    # Evaluation summary
    if rewards:
        print("\n--- Evaluation Summary ---")
        print(f"Average Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
        print(f"Min/Max Reward: {min(rewards):.2f} / {max(rewards):.2f}")
        print(f"Average Episode Length: {np.mean(episode_lengths):.1f} steps")
        print(f"Success Rate: {successful_landings / len(rewards) * 100:.1f}% ({successful_landings}/{len(rewards)})")
        print("--------------------------")
    

if __name__ == "__main__":
    # Check if MPS (Apple Metal) is available and set it as default
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        default_device = 'mps'
        print("Using MPS device for evaluation.")
    elif torch.cuda.is_available():
        default_device = 'cuda'
        print("Using CUDA device for evaluation.")
    else:
        default_device = 'cpu'
        print("Using CPU for evaluation.")
    
    parser = argparse.ArgumentParser(description="Evaluate a trained model using only policy weights")
    parser.add_argument(
        "--run-dir", 
        type=str, 
        required=True,
        help="Path to the run directory containing the model"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=10,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--env-id", 
        type=str, 
        default="LunarLander-v3",
        help="Gymnasium environment ID"
    )
    parser.add_argument(
        "--no-render", 
        action="store_true",
        help="Disable rendering during evaluation"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=default_device,
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help="Device to use for evaluation"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save evaluation videos (only when --no-render is used)"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_model_from_policy(
        run_dir=args.run_dir,
        num_episodes=args.episodes,
        env_id=args.env_id,
        render=not args.no_render,
        device=args.device,
        seed=args.seed,
        save_video=args.save_video
    ) 