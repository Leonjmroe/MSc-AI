# Reinforcement Learning Agents for Lunar Lander

Implementations and scripts for training and evaluating RL agents (DQN, Double DQN, Dueling DQN, PPO) on the Gymnasium `LunarLander-v3` environment.

**Dependencies:** Requires Python 3.8+, `gymnasium[box2d]`, `torch`, `numpy`, `pandas`, `matplotlib`, `stable-baselines3`.

### DQN Variants (DQN, DDQN, Dueling DQN)

Run the agent script directly to train. Saves model (`.pt`), results (`.csv`), and plot (`.png`) locally.

- **DQN:** `python dqn.py`
- **Double DQN:** `python ddqn.py`
- **Dueling DQN:** `python duel_dqn.py`

*Note: Hyperparameters are constants within each script. Within the scripts, episodes are all hardcoded to 3000.*

### PPO

Train using configurations from `ppo_configs/`. Output saved to `training_runs/run_YYYYMMDD_HHMMSS/`.

- **Example command to train PPO:** `python ppo.py config_1.py --episodes 1000`

## Evaluation

Evaluate trained agents. Renders by default (`--no-render` to disable).

### Evaluating DQN Variants (`eval_dqn.py`)

Requires model path (`.pt`). Use `--arch duel` for Dueling DQN.

- **Example:** `python eval_dqn.py models/dqn.pt`
- **Example for duel dqn:** `python eval_dqn.py models/duel_dqn.pt --arch duel`
- **Optional:** `--episodes <N>` (default: 5)

### Evaluating PPO (`eval_ppo.py`)

Requires policy (`.pth`) and normalisation stats (`.pkl`) paths.

- **Example:** `python eval_ppo.py models/ppo_1.pth models/ppo_1.pkl`
- **Optional:** `--episodes <N>` (default: 5)
