import gymnasium as gym
import argparse
import numpy as np
import torch
import os
import time
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ENV_NAME = "LunarLander-v3"
NUM_EPISODES = 5
RENDER = True

def evaluate_model(pth_path: str, pkl_path: str, num_episodes: int, render: bool):
    if not os.path.exists(pth_path) or not os.path.exists(pkl_path):
        print(f"Error: File not found - Model: {pth_path}, Stats: {pkl_path}")
        return

    eval_env_raw = make_vec_env(ENV_NAME, n_envs=1, vec_env_cls=DummyVecEnv, env_kwargs={"render_mode": "human" if render else None})
    
    with open(pkl_path, "rb") as f:
        vec_normalize = pickle.load(f)
        
    eval_env = VecNormalize(
        eval_env_raw,
        norm_obs=vec_normalize.norm_obs,
        norm_reward=False,
        clip_obs=vec_normalize.clip_obs,
        clip_reward=vec_normalize.clip_reward,
        gamma=vec_normalize.gamma,
        epsilon=vec_normalize.epsilon,
    )
    eval_env.obs_rms = vec_normalize.obs_rms
    eval_env.ret_rms = vec_normalize.ret_rms
    eval_env.training = False
    
    model = PPO("MlpPolicy", eval_env, device='auto')
    model.policy.load_state_dict(torch.load(pth_path))

    total_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            total_reward += reward[0]
            steps += 1
            if render and eval_env.envs[0].render_mode == "human":
                eval_env.render()
                time.sleep(0.01)

        total_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward:.2f}, Steps: {steps}")

    eval_env.close()

    if total_rewards:
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        mean_length = np.mean(episode_lengths)
        print(f"\nAverage Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Min/Max Reward: {np.min(total_rewards):.2f} / {np.max(total_rewards):.2f}")
        print(f"Average Episode Length: {mean_length:.1f} steps")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pth_path", type=str)
    parser.add_argument("pkl_path", type=str)
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    evaluate_model(args.pth_path, args.pkl_path, args.episodes, not args.no_render)