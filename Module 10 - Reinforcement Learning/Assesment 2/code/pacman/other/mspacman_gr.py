import gymnasium as gym
import ale_py
import numpy as np
from collections import deque
import random
import time

# Register ALE environments
gym.register_envs(ale_py)

# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EXPLORATION_RATE = 1.0
MIN_EXPLORATION_RATE = 0.01
EXPLORATION_DECAY = 0.995
EPISODES = 100  # Reduced for quicker testing
MAX_STEPS = 10000

class QLearningAgent:
    def __init__(self, observation_space, action_space):
        self.action_space = action_space
        self.state_space = (20, 20)
        self.q_table = np.zeros(self.state_space + (action_space.n,))
        self.memory = deque(maxlen=10000)

    def preprocess_state(self, state):
        if len(state.shape) == 3:
            state = np.mean(state, axis=2).astype(np.uint8)
        state = state[::9, ::8][:20, :20]
        return state

    def get_state_key(self, state):
        processed = self.preprocess_state(state)
        discretized = (processed // 8).astype(np.int32)
        discretized = np.clip(discretized, 0, self.state_space[0] - 1)
        return tuple(np.unravel_index(discretized.flatten()[0], self.state_space))

    def choose_action(self, state, exploration_rate):
        state_key = self.get_state_key(state)
        if random.random() < exploration_rate:
            return self.action_space.sample()
        return np.argmax(self.q_table[state_key])

    def learn(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        current_q = self.q_table[state_key][action]
        next_max_q = np.max(self.q_table[next_state_key])
        new_q = current_q + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * next_max_q * (1 - done) - current_q
        )
        self.q_table[state_key][action] = new_q

def train(render=False):
    env = gym.make("ALE/MsPacman-v5", render_mode="human" if render else None)
    agent = QLearningAgent(env.observation_space, env.action_space)
    exploration_rate = EXPLORATION_RATE

    for episode in range(EPISODES):
        state, info = env.reset(seed=episode)
        total_reward = 0

        for step in range(MAX_STEPS):
            action = agent.choose_action(state, exploration_rate)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if render:
                env.render()
                time.sleep(0.02)  # Slow down rendering for visibility

            if done:
                break

        exploration_rate = max(MIN_EXPLORATION_RATE, exploration_rate * EXPLORATION_DECAY)

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Exploration Rate: {exploration_rate:.3f}")

    env.close()
    return agent

def play(agent):
    env = gym.make("ALE/MsPacman-v5", render_mode="human")
    try:
        state, info = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state, 0.05)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            env.render()
            time.sleep(0.02)  # Ensure smooth rendering
        print(f"Demonstration complete. Total Reward: {total_reward}")
    except Exception as e:
        print(f"Error during rendering: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    print("Starting training...")
    # Set render=True to debug training visualization, False to train faster
    trained_agent = train(render=True)
    print("Training complete. Starting demonstration...")
    play(trained_agent)