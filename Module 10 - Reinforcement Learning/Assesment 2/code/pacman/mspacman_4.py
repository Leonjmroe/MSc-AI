import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import os
import numpy as np
import datetime
import torch

EPISODES = 10000
MODEL_DIR = "models"

# Create models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Define a custom callback to stop training after 10 episodes and track rewards
class EpisodeCallback(BaseCallback):
    def __init__(self, max_episodes=EPISODES, verbose=0):
        super(EpisodeCallback, self).__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0
        self.episode_reward = 0
        self.rewards = []  # Store rewards for plotting
        self.moving_avg = 0  # For tracking moving average

    def _on_step(self):
        # Add the current step's reward to the episode total 
        self.episode_reward += self.locals['rewards'][0]
        # Check if the episode has ended
        if self.locals['dones'][0]:
            self.rewards.append(self.episode_reward)
            
            # Calculate moving average with window size of 50
            if len(self.rewards) >= 5:
                self.moving_avg = np.mean(self.rewards[-5:])
                print(f"Episode {self.episode_count + 1}: Reward = {self.episode_reward}, 5MAs = {self.moving_avg:.2f}")
            else:
                print(f"Episode {self.episode_count + 1}: Reward = {self.episode_reward}")
                
            self.episode_reward = 0  # Reset for the next episode
            self.episode_count += 1
            if self.episode_count >= self.max_episodes:
                return False  # Stop training after max_episodes
        return True

# Set the ROM path if using ROMs downloaded with AutoROM
os.environ['ATARI_ROM_PATH'] = os.path.expanduser('~/ALE-ROMs')

# Define environment name
env_name = 'ALE/MsPacman-v5'

def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

# Create a function that makes the environment
def make_env():
    return gym.make(env_name)

# Create vectorized environment using a function
train_env = DummyVecEnv([make_env])

# Initialize the PPO model with CnnPolicy for image inputs
model = PPO(
    "CnnPolicy",
    train_env,
    batch_size=256,
    n_steps=128,
    n_epochs=4,
    learning_rate=linear_schedule(2.5e-4),
    clip_range=linear_schedule(0.1),
    ent_coef=0.01,
    vf_coef=0.5,
    verbose=0,  # Changed to 0 to avoid printing the training grid
    device="mps"  # Use GPU for training
)

# Create the callback to limit training to 10 episodes
callback = EpisodeCallback(max_episodes=EPISODES)

# Train the model (set a large timestep limit; callback will stop it early)
model.learn(total_timesteps=1_000_000, callback=callback)

# Save the model with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(MODEL_DIR, f"mspacman_ppo_{timestamp}")
model.save(model_path)
print(f"Model saved to {model_path}")

# Save the model's state dictionary as a PyTorch .pth file
torch_model_path = os.path.join(MODEL_DIR, f"mspacman_ppo_{timestamp}.pth")
torch.save(model.policy.state_dict(), torch_model_path)
print(f"PyTorch model saved to {torch_model_path}")

# Close the training environment
train_env.close()

# Create evaluation environment with rendering
eval_env = gym.make(env_name, render_mode='human')

# Run an evaluation episode with rendering
obs, _ = eval_env.reset()
terminated = False
truncated = False
total_reward = 0
while not (terminated or truncated):
    action, _states = model.predict(obs, deterministic=True)  # Use deterministic actions
    obs, reward, terminated, truncated, _ = eval_env.step(action)
    total_reward += reward
print(f"Evaluation episode reward: {total_reward}")

# Close the evaluation environment
eval_env.close()

# Plot the reward curve
plt.figure(figsize=(10, 6))
plt.plot(callback.rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Reward Curve')
plt.grid(True)

# Add moving average to the plot
if len(callback.rewards) > 5:
    window_size = 5
    moving_avg = np.convolve(callback.rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(callback.rewards)), moving_avg, 'r-', label=f'{window_size}-episode Moving Average')
    plt.legend()

# Save the plot
plot_path = os.path.join(MODEL_DIR, f"reward_curve_{timestamp}.png")
plt.savefig(plot_path)
print(f"Reward curve saved to {plot_path}")

# Show the plot
plt.show()