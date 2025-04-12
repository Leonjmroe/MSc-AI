import gymnasium as gym
import os

# Set ROM path
os.environ['ATARI_ROM_PATH'] = os.path.expanduser('~/ALE-ROMs')

print("Attempting to create Ms. Pacman environment...")

try:
    # Try to create the environment
    env = gym.make('ALE/MsPacman-v5', render_mode='human')
    observation, info = env.reset()
    
    print("Success! Environment created.")
    print(f"Observation shape: {observation.shape}")
    print(f"Action space: {env.action_space}")
    
    print("Running 100 random steps...")
    
    # Run a few random steps
    for i in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if i % 10 == 0:
            print(f"Step {i}: reward = {reward}")
        
        if terminated or truncated:
            print("Episode ended, resetting...")
            observation, info = env.reset()
    
    env.close()
    print("Test completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nTroubleshooting tips:")
    print("1. Make sure you've installed all required packages:")
    print("   pip install 'gymnasium[atari]' 'autorom[accept-rom-license]'")
    print("2. Download ROMs using: AutoROM --install-dir ~/ALE-ROMs")
    print("3. Verify ROMs exist: ls ~/ALE-ROMs | grep ms_pacman") 