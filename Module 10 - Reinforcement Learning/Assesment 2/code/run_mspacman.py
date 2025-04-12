#!/usr/bin/env python3
"""
Run trained Ms. Pac-Man agent in visualization mode
"""
import os
import gymnasium as gym
import ale_py
import torch
import argparse
import logging
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mspacman_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mspacman_evaluation")

def parse_args():
    parser = argparse.ArgumentParser(description="Run a trained Ms. Pac-Man agent")
    parser.add_argument("--model_path", type=str, 
                        default="checkpoints/top_agent_final.zip",
                        help="Path to the trained model file")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return 1
    
    logger.info(f"Loading model from {args.model_path}")
    
    # Set up environment with rendering
    env = gym.make("ALE/MsPacman-v5", render_mode="human")
    env = AtariWrapper(env)
    
    # Load model
    try:
        model = DQN.load(args.model_path, env=env)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return 1
    
    # Run evaluation episodes
    total_rewards = []
    
    for episode in range(args.episodes):
        logger.info(f"Starting episode {episode+1}/{args.episodes}")
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0
        
        # Run episode
        while not (done or truncated):
            action, _ = model.predict(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated
            episode_reward += reward
            step_count += 1
            
            # Log progress periodically
            if step_count % 100 == 0:
                logger.info(f"Episode {episode+1} - Step {step_count}, current reward: {episode_reward:.2f}")
        
        total_rewards.append(episode_reward)
        logger.info(f"Episode {episode+1} finished with reward: {episode_reward:.2f} in {step_count} steps")
    
    # Display summary
    avg_reward = sum(total_rewards) / len(total_rewards)
    logger.info(f"Evaluation complete - Average reward over {args.episodes} episodes: {avg_reward:.2f}")
    
    env.close()
    return 0

if __name__ == "__main__":
    exit(main()) 