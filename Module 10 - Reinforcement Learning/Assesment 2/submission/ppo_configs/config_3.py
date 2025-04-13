from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class TrainingConfig:
    env_id: str = 'LunarLander-v3'
    algorithm: str = 'PPO'
    device: str = 'cpu' 
    n_envs: int = 8
    timesteps: int = 3000000
    learning_rate: float = 0.003
    use_lr_schedule: bool = True
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy_kwargs: Dict = None
    vec_normalize_filename: str = "vec_normalize.pkl"
    target_episodes: Optional[int] = 3000