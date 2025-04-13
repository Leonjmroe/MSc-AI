from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class TrainingConfig:
    env_id: str = 'LunarLander-v3'
    algorithm: str = 'PPO'
    device: str = 'cpu' 
    n_envs: int = 1
    timesteps: int = 200000
    learning_rate: float = 0.001
    use_lr_schedule: bool = False
    n_steps: int = 128
    batch_size: int = 32
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    policy_kwargs: Dict = None
    vec_normalize_filename: str = "vec_normalize_basic.pkl"
    target_episodes: Optional[int] = 3000
