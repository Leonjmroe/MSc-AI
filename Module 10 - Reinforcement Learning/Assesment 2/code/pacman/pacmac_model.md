# Reinforcement Learning Agent for Ms. Pac-Man using Enhanced DQN

## 1. Introduction

This document details the implementation of a Deep Q-Network (DQN) based reinforcement learning agent designed to play the Atari game Ms. Pac-Man. The agent leverages several advanced techniques built upon the standard DQN algorithm to improve learning efficiency, stability, and overall performance. The goal is to train an agent that can achieve high scores by navigating the maze, eating pellets, avoiding ghosts, and progressing through levels.

The key components and techniques employed include:

***Deep Q-Networks (DQN):** Using a Convolutional Neural Network (CNN) to approximate the action-value function (Q-function).

***Gymnasium Environment:** Utilizing the standard Atari environment interface.

***Preprocessing Wrappers:** Standard Atari preprocessing (downsampling, grayscale, frame stacking).

***Enhanced Network Architecture:** Dueling DQN with Batch Normalization and Dropout.

***Advanced Learning Techniques:** Prioritized Experience Replay (PER), Double DQN, Soft Target Updates, Action Repetition.

***Custom Reward Shaping:** Modifying the reward signal to guide learning more effectively.

***Configuration Management:** Easy setup for different modes (train, test, play) and hyperparameters.

## 2. Environment Setup

### 2.1. Base Environment

* The core environment is `ALE/MsPacman-v5` provided by the `ale-py` library and accessed via `gymnasium`.

*`full_action_space=False` is used to limit the action space to only meaningful actions for Ms. Pac-Man (typically 9 actions).

### 2.2. Gymnasium Wrappers

Several wrappers modify the raw environment output:

***`gym.wrappers.RecordEpisodeStatistics`**: Tracks episode returns and lengths automatically, simplifying logging.

***`gym.wrappers.AtariPreprocessing`**: Applies standard preprocessing steps crucial for Atari agents:

***Frame Skipping:** (Set to 1 here, meaning minimal skipping).

***Screen Resizing:** Downsamples the screen to 84x84 pixels.

***Grayscale Conversion:** Converts observations to grayscale.

***No-Op Max:** Executes random number of no-op actions at the start of episodes to introduce stochasticity.

***`RewardShapingWrapper` (Custom)**: Modifies the reward signal received from the environment to provide denser and potentially more informative feedback to the agent (detailed in Section 6).

***`gym.wrappers.FrameStackObservation`**: Stacks `NUM_FRAMES` (typically 4) consecutive frames together. This allows the agent to infer dynamic information like the direction and speed of ghosts and Ms. Pac-Man from static images.

## 3. Agent Architecture (Q-Network)

The agent uses an enhanced CNN based on common DQN architectures for Atari games.

### 3.1. Convolutional Layers

* Three convolutional layers extract spatial features from the stacked input frames (4x84x84).

*`Conv1`: 32 filters, 8x8 kernel, stride 4.

*`Conv2`: 64 filters, 4x4 kernel, stride 2.

*`Conv3`: 64 filters, 3x3 kernel, stride 1.

***ReLU Activation:** Used after each convolutional layer to introduce non-linearity.

***Batch Normalization (`BatchNorm2d`)**: Applied after each convolution. This helps stabilize training, allows for potentially higher learning rates, and can act as a regularizer.

### 3.2. Fully Connected Layers & Dueling DQN

* The output from the final convolutional layer is flattened.
* A fully connected layer (`fc1`) with 512 units processes the flattened features.

***Dropout**: Applied after `fc1` (with a rate of 0.2) to prevent overfitting by randomly setting a fraction of input units to 0 during training.

***Dueling DQN Architecture**: Instead of directly outputting Q-values, the network splits into two streams:

***Value Stream (`value`)**: Outputs a single scalar value representing the state value `V(s)`. This estimates how good it is to be in the current state `s`.

***Advantage Stream (`advantage`)**: Outputs a value for each action, representing the advantage `A(s, a)` of taking action `a` in state `s` compared to other actions.

***Combining Streams**: The Q-value is reconstructed using:

`Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))`

    Subtracting the mean advantage ensures identifiability and improves stability. Dueling DQN helps the agent learn the state value`V(s)` without having to learn the effect of *every* action when many actions might be irrelevant in certain states.

## 4. Learning Algorithm Enhancements

The agent incorporates several improvements over the basic DQN algorithm:

### 4.1. Experience Replay

***Concept:** Stores past transitions (state, action, reward, next\_state, done) in a replay buffer (`MEMORY_SIZE`). During training, batches of transitions are randomly sampled from this buffer.

***Benefits:**

* Breaks temporal correlations in sequential observations, making training samples more independent and identically distributed (i.i.d.).
* Increases data efficiency by reusing past experiences multiple times.

### 4.2. Prioritized Experience Replay (PER)

***Concept:** Instead of uniform random sampling, PER samples transitions based on their importance, typically measured by the magnitude of the Temporal Difference (TD) error. Transitions where the agent's prediction was highly inaccurate (large TD error) are sampled more frequently.

***Implementation:**

*`PrioritizedReplayBuffer` class stores transitions along with their priorities.

* New transitions are added with maximum priority.
* Sampling uses probabilities derived from priorities (`probs = priorities ** alpha`). `alpha` controls the degree of prioritization (0=uniform, 1=full).

***Importance Sampling (IS) Weights:** To correct the bias introduced by non-uniform sampling, updates are weighted by IS weights. `beta` controls the amount of correction, annealed from `beta_start` to 1.0 over training.

***Benefit:** Focuses training on "surprising" or difficult transitions, leading to faster and more efficient learning.

### 4.3. Target Network

***Concept:** Uses two networks: a `policy_net` (the main network being trained) and a `target_net`. The `target_net` is used to calculate the target Q-values in the TD error calculation: `Target = r + gamma * Q_target(s', argmax_a' Q_policy(s', a'))`.

***Benefit:** Provides a stable target during training, preventing the oscillations and divergence that can occur when the same rapidly changing network is used for both estimating current Q-values and target Q-values.

### 4.4. Soft Target Updates

***Concept:** Instead of periodically copying the `policy_net` weights directly to the `target_net` (hard update), a soft update is used. The `target_net` weights are updated slowly towards the `policy_net` weights at each update step: `θ_target = τ * θ_policy + (1 - τ) * θ_target`. `τ` (tau) is a small hyperparameter (e.g., 0.1).

***Benefit:** Further improves training stability compared to hard updates.

### 4.5. Double DQN

***Concept:** Standard DQN can suffer from overestimation of Q-values because the `max` operation uses the same network to both *select* the best next action and *evaluate* its Q-value. Double DQN decouples these steps:

1. Use the `policy_net` to select the best action for the next state: `a'_max = argmax_a' Q_policy(s', a')`.
2. Use the `target_net` to evaluate the Q-value of that selected action: `Q_target(s', a'_max)`.

***Benefit:** Reduces overestimation bias, leading to more accurate value estimates and often better performance.

### 4.6. Optimization

***Optimizer:**`Adam` optimizer is used, a common and effective adaptive learning rate method.

***Weight Decay:** A small L2 regularization term (`weight_decay=1e-5`) is added to the optimizer to discourage large weights and prevent overfitting.

***Loss Function:**`SmoothL1Loss` (Huber loss) is used. It behaves like Mean Squared Error (MSE) for small errors but like Mean Absolute Error (MAE) for large errors, making it less sensitive to outliers than pure MSE.

***Gradient Clipping:** Gradients are clipped (`clip_grad_norm_`) during the backward pass to prevent exploding gradients, which can destabilize training.

## 5. Exploration vs. Exploitation

***Epsilon-Greedy Strategy:** The agent balances exploring the environment to discover new strategies and exploiting its current knowledge to maximize rewards.

* With probability `epsilon`, a random action is chosen (exploration).
* With probability `1 - epsilon`, the action with the highest estimated Q-value from the `policy_net` is chosen (exploitation).

***Epsilon Decay:**`epsilon` starts high (`EPSILON_START = 1.0`) and linearly decays over `EPSILON_DECAY` steps towards a minimum value (`EPSILON_END = 0.01`). This encourages exploration early in training and more exploitation later as the agent becomes more confident.

## 6. Reward Shaping (`RewardShapingWrapper`)

Since the default rewards in Atari games can be sparse (only occurring when scoring points), custom reward shaping is used (`USE_REWARD_SHAPING = True`) to provide denser feedback:

***`SURVIVAL_REWARD` (Small Positive):** A tiny reward (+0.005) given at each step encourages the agent to stay alive longer.

***`NO_MOVEMENT_PENALTY` (Negative):** A penalty (-0.5) applied if the agent's estimated position (based on screen pixels) doesn't change significantly over several checks (~1 second). This discourages the agent from getting stuck.

***Life Lost Penalty (Implicit):** An extra penalty (-1.0) is added to the reward when a life is lost (`'ale.lives'` decreases).

***Scoring Bonuses:**

* Eating regular pellets/fruit results in a small positive reward (`score_diff * PELLET_REWARD_MULTIPLIER / 10.0`).
* Eating ghosts/power pellets (`score_diff > 50`) gives a larger bonus (`GHOST_EATEN_BONUS`).

***Level Completion Bonus (Implicit):** Detecting a large score jump (`score_diff > 500`) is assumed to be level completion and gives a large bonus (+10.0).

*Note: Score extraction relies on the `info` dictionary provided by the environment wrappers, with fallbacks.*

## 7. Action Repetition

***Concept:** During training, when the agent selects an action (either randomly or greedily), it might repeat that same action for a small, random number of subsequent steps (1-3 steps here, with a certain probability).

***Benefit:** This reflects the temporal coherence needed in many Atari games where holding down an action for a few frames is often necessary. It can lead to more stable and effective policies compared to selecting a new action every single frame.

## 8. Training Process

The main training loop (`if __name__ == "__main__":`) orchestrates the interaction between the agent and the environment:

1.**Initialization:** Sets up logging, device (CPU/GPU), random seeds, environment, agent, and loads any existing checkpoint if available (`check_existing_model`).

2.**Episode Loop:** Runs for `MAX_EPISODES` or until `TOTAL_TIMESTEPS` is reached.

3.**Step Loop (within each episode):**

***Select Action:** Agent chooses an action using the epsilon-greedy strategy (`agent.select_action`).

***Environment Step:** The chosen action is sent to the environment (`env.step`), receiving the next observation, reward, done flags, and info.

***Store Experience:** The transition `(state, action, reward, next_state, done)` is added to the prioritized replay buffer (`agent.memory.add`). *Note: The reward stored here is the shaped reward if `USE_REWARD_SHAPING` is True.*

***Train Agent:** If the buffer has enough samples, the agent performs a training step (`agent.train`): samples a batch using PER, calculates loss using Double DQN logic, computes gradients, updates `policy_net` weights, and updates priorities in the buffer.

***Update Target Network:** Periodically (every `TARGET_UPDATE_FREQ` steps), the `target_net` is updated using the soft update rule (`agent.update_target_network`).

***Logging/Reporting:** Periodically prints progress (steps, episodes, epsilon, loss, speed) and saves model checkpoints.

***End of Episode:** Logs detailed episode statistics (duration, steps, score, level, moving average reward), potentially plots progress, and resets the environment for the next episode.

4.**Post-Training:** Saves the final model, generates final plots, logs summary statistics.

5.**Auto-Play:** If enabled (`AUTO_PLAY_AFTER_TRAINING = True`), loads the final model and runs `play_trained_agent` for visualization.

## 9. Evaluation and Visualization

***`test_model` function (`MODE = "test"`):** Loads the latest model and runs it for `TEST_EPISODES` with minimal exploration (`epsilon = 0.01`) and no rendering. It logs performance statistics and saves plots summarizing test results (rewards, lengths, levels).

***`play_trained_agent` function (`MODE = "play"` or auto-play):** Loads the best available model (`find_best_model`), creates an environment with rendering (`render_mode="human"`), and runs the agent for `PLAY_EPISODES` with minimal exploration, allowing visual inspection of the agent's behavior.

## 10. Conclusion

This agent represents a sophisticated implementation of DQN for Atari Ms. Pac-Man. By combining a Dueling network architecture with Prioritized Experience Replay, Double DQN, reward shaping, action repetition, and careful hyperparameter tuning, it aims to learn complex strategies required to achieve high scores in the game. The modular design with configuration options allows for easy experimentation and adaptation.
