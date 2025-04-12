import gymnasium as gym
import ale_py
import time 

# --- Parameters ---
ENV_NAME = "ALE/MsPacman-v5"
NUM_EPISODES = 50  # How many times to play the game
MAX_STEPS_PER_EPISODE = 500 # Safety break to prevent excessively long episodes
RENDER_MODE = "rgb_array" # rgb_array / human


# --- Environment Setup ---
try:
    # Create the environment
    # render_mode="human" opens a window to visualize the game.
    # Use render_mode="rgb_array" for training without visualization.
    env = gym.make(ENV_NAME, render_mode=RENDER_MODE)
    print(f"Environment '{ENV_NAME}' created successfully.")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")

except gym.error.Error as e:
    print(f"Error creating environment '{ENV_NAME}': {e}")
    print("Please ensure you have Atari ROMs installed.")
    print("You might need to install them using 'ale-import-roms <path_to_roms>'")
    exit() # Exit if environment creation fails

# --- Simple Random Agent Interaction Loop ---

print(f"\nRunning {NUM_EPISODES} episodes with a random agent...")

for episode in range(NUM_EPISODES):
    # Reset the environment at the start of each episode
    # Setting a seed ensures reproducibility for that specific episode start
    observation, info = env.reset(seed=42 + episode)
    
    terminated = False # Flag: Environment reached a terminal state (game over, win)
    truncated = False  # Flag: Episode ended due to external limit (e.g., time limit)
    total_reward = 0
    step_count = 0

    print(f"\n--- Starting Episode {episode + 1} ---")

    # Loop until the episode ends (terminated or truncated)
    while not terminated and not truncated:
        # 1. Choose an action (Policy)
        # For this "Hello World", we use a random policy
        action = env.action_space.sample()
        # print(f"Step: {step_count}, Action: {action}") # Uncomment for verbose action output

        # 2. Step the environment
        # Apply the action and get the outcome
        observation, reward, terminated, truncated, info = env.step(action)

        # 3. Accumulate reward
        total_reward += reward

        # 4. Optional: Add a small delay to make visualization easier to follow
        if RENDER_MODE == "human":
            time.sleep(0.01) # Adjust sleep time as needed

        # 5. Check for step limit (optional safety break)
        step_count += 1
        if step_count >= MAX_STEPS_PER_EPISODE:
             truncated = True # Manually truncate if episode is too long

        # The loop condition (while not terminated and not truncated) handles exit

    # Episode finished
    print(f"--- Episode {episode + 1} Finished ---")
    print(f"Reason: {'Terminated' if terminated else 'Truncated'}")
    print(f"Total Steps: {step_count}")
    print(f"Total Reward: {total_reward}")
    # The 'info' dictionary often contains useful episode-specific data like lives left
    if 'lives' in info:
        print(f"Lives remaining: {info['lives']}")


# --- Cleanup ---
print("\nClosing environment.")
env.close()

print("\nScript finished.")