import gymnasium as gym
import ale_py
import numpy as np
import time
from collections import deque
import random
import os

# Register ALE environments with Gymnasium
gym.register_envs(ale_py)

class SimpleQLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = 0.01
        self.exploration_decay = exploration_decay
        self.q_table = {}
        
    def get_state_key(self, state):
        # Simplify the state representation for the Q-table
        # Downsampling the image to make it more manageable
        downsampled = state[::10, ::10, 0]  # Take only one color channel and sample every 10th pixel
        return str(downsampled.flatten())
    
    def choose_action(self, state):
        state_key = self.get_state_key(state)
        
        # Exploration: random action
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_size)
        
        # Exploitation: use Q-table or default to random if state not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[state_key])
    
    def learn(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Initialize Q values for states if they don't exist
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])
        
        # Q-value update formula
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q * (1 - done) - current_q)
        self.q_table[state_key][action] = new_q
        
        # Decay exploration rate
        if done:
            self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)


def train_agent(episodes=50, render=True, sound=False, render_every=1):
    """
    Train a Q-learning agent on Ms. Pacman
    
    Args:
        episodes: Number of training episodes
        render: Whether to render the game visually
        sound: Whether to play game sounds
        render_every: Only render every n episodes (to speed up training)
    """
    # Turn off sound if needed
    if not sound:
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
    
    # Set up environment with or without rendering
    render_mode = "human" if render else None
    env = gym.make("ALE/MsPacman-v5", render_mode=render_mode)
    
    # Get environment details
    action_size = env.action_space.n
    state_size = env.observation_space.shape
    
    # Initialize agent
    agent = SimpleQLearningAgent(state_size, action_size)
    
    # Keep track of scores
    scores = deque(maxlen=100)
    
    for episode in range(episodes):
        # Toggle rendering based on render_every setting
        if render and (episode % render_every != 0):
            env.close()
            env = gym.make("ALE/MsPacman-v5", render_mode=None)
        elif render and (episode % render_every == 0):
            env.close()
            env = gym.make("ALE/MsPacman-v5", render_mode="human")
            
        state, info = env.reset()
        score = 0
        
        # Game loop
        while True:
            # Agent selects action
            action = agent.choose_action(state)
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Agent learns from experience
            agent.learn(state, action, reward, next_state, done)
            
            # Update state and score
            state = next_state
            score += reward
            
            # End episode if game is over
            if done:
                break
                
        # Save score and print progress
        scores.append(score)
        print(f"Episode: {episode+1}/{episodes}, Score: {score}, Avg Score: {np.mean(scores):.2f}, Exploration Rate: {agent.exploration_rate:.2f}")
        
    env.close()
    return agent


def test_agent(agent, episodes=5, render=True, sound=False):
    """
    Test a trained agent
    
    Args:
        agent: The trained RL agent
        episodes: Number of test episodes
        render: Whether to render the game visually
        sound: Whether to play game sounds
    """
    # Turn off sound if needed
    if not sound:
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
    
    # Set up environment with or without rendering
    render_mode = "human" if render else None
    env = gym.make("ALE/MsPacman-v5", render_mode=render_mode)
    
    for episode in range(episodes):
        state, info = env.reset()
        score = 0
        done = False
        
        while not done:
            # Choose action with no exploration (pure exploitation)
            original_exploration_rate = agent.exploration_rate
            agent.exploration_rate = 0  # Turn off exploration for testing
            action = agent.choose_action(state)
            agent.exploration_rate = original_exploration_rate  # Restore exploration rate
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            score += reward
            
        print(f"Test Episode {episode+1}/{episodes}, Score: {score}")
    
    env.close()


if __name__ == "__main__":
    # Configuration
    EPISODES = 100
    RENDER = True  # Set to False for faster training with no visuals
    SOUND = False  # Set to False to mute sounds
    RENDER_EVERY = 5  # Only render every 5 episodes to speed up training
    
    # Train the agent
    print("Starting training...")
    agent = train_agent(episodes=EPISODES, render=RENDER, sound=SOUND, render_every=RENDER_EVERY)
    print("Training completed!")
    
    # Test the trained agent
    if RENDER:
        print("\nTesting the trained agent...")
        test_agent(agent, episodes=5, render=True, sound=SOUND)