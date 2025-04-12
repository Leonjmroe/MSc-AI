# Ms. Pacman Installation Guide

This guide will help you set up the necessary packages to run Ms. Pacman in Gymnasium with Stable Baselines 3.

## Prerequisites

- Python 3.8+ installed
- pip package manager

## Installation Steps

1. **Install Gymnasium with Atari support**

```bash
pip install "gymnasium[atari]"
```

2. **Install AutoROM to manage Atari ROM files**

```bash
pip install "autorom[accept-rom-license]"
```

3. **Download and install Atari ROM files**

```bash
# Create a directory for ROMs
mkdir -p ~/ALE-ROMs

# Download and install ROMs
AutoROM --install-dir ~/ALE-ROMs
```

4. **Verify installation**

Create a simple test script to verify that Ms. Pacman can be loaded:

```python
# test_pacman.py
import gymnasium as gym
import os

# Set ROM path
os.environ['ATARI_ROM_PATH'] = os.path.expanduser('~/ALE-ROMs')

# Try to create the environment
env = gym.make('ALE/MsPacman-v5', render_mode='human')
observation, info = env.reset()

# Run a few random steps
for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

Run the test script:

```bash
python test_pacman.py
```

## Troubleshooting

If you encounter errors about namespace ALE not being found:

1. **Check if the ROM path is set correctly**:
   
   ```python
   os.environ['ATARI_ROM_PATH'] = os.path.expanduser('~/ALE-ROMs')
   ```

2. **Verify ROM installation**:
   
   ```bash
   ls ~/ALE-ROMs
   ```
   You should see `ms_pacman.bin` among other ROM files.

3. **Try reinstalling the packages**:
   
   ```bash
   pip uninstall gymnasium ale_py autorom
   pip install "gymnasium[atari]" "autorom[accept-rom-license]"
   ```

4. **Check for specific errors**:
   
   If you get `ImportError: cannot import name 'FrameStack'`, use `FrameStackObservation` instead:
   
   ```python
   from gymnasium.wrappers import FrameStackObservation
   env = FrameStackObservation(env, num_stack=4)
   ```

## Running the Ms. Pacman Script

Once the installation is complete, you can modify the `mspacman_4.py` script to use Ms. Pacman instead of CartPole:

1. Change the environment name:
   ```python
   env_name = 'ALE/MsPacman-v5'
   ```

2. Set the ROM path:
   ```python
   os.environ['ATARI_ROM_PATH'] = os.path.expanduser('~/ALE-ROMs')
   ```

3. Use CnnPolicy since Ms. Pacman uses image observations:
   ```python
   model = PPO("CnnPolicy", train_env, verbose=1, device='mps')
   ```

4. Adjust the episode steps to a higher value:
   ```python
   env = TimeLimit(env, max_episode_steps=10000)
   ```

## Additional Resources

- [Gymnasium Atari Documentation](https://gymnasium.farama.org/environments/atari/)
- [Stable Baselines 3 Documentation](https://stable-baselines3.readthedocs.io/en/master/)
- [AutoROM GitHub Repository](https://github.com/Farama-Foundation/AutoROM) 