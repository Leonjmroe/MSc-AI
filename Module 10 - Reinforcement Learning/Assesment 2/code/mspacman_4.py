import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import os

EPISODES = 100

# Define a custom callback to stop training after 10 episodes and track rewards
class EpisodeCallback(BaseCallback):
    def __init__(self, max_episodes=EPISODES, verbose=0):
        super(EpisodeCallback, self).__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0
        self.episode_reward = 0
        self.rewards = []  # Store rewards for plotting

    def _on_step(self):
        # Add the current step's reward to the episode total 
        self.episode_reward += self.locals['rewards'][0]
        # Check if the episode has ended
        if self.locals['dones'][0]:
            self.rewards.append(self.episode_reward)
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

# Create a function that makes the environment
def make_env():
    return gym.make(env_name)

# Create vectorized environment using a function
train_env = DummyVecEnv([make_env])

# Initialize the PPO model with CnnPolicy for image inputs
model = PPO("CnnPolicy", train_env, verbose=1, device='mps')

# Create the callback to limit training to 10 episodes
callback = EpisodeCallback(max_episodes=EPISODES)

# Train the model (set a large timestep limit; callback will stop it early)
model.learn(total_timesteps=1_000_000, callback=callback)

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
plt.plot(callback.rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Reward Curve')
plt.show()