from dataclasses import dataclass
from typing import Optional

import tyro


# Hyperparameters
@dataclass
class Args:
    exp_name: str = "inverted_pendulum"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False  # True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: tyro.conf.FlagConversionOff[bool] = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm arguments
    env_id: str = "InvertedPendulum-v0"
    """the id of the environment"""
    total_timesteps: int = 10000  # 300000 #500000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 3000
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.995
    """the discount factor gamma"""
    gae_lambda: float = 0.98
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 10
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""
    bar_lr: float = 1e-3
    """learning rate for barrier compensator"""
    bar_train_steps: int = 100
    """training iterations for barrier compensator per policy update"""
    normalize: bool = False
    """use normalization (observation/reward)"""
    checkpoint_freq: int = 50
    """save checkpoint every N iterations"""
    timestep_batch_GP_max: int = 650
    """Update GP dynamics for certain number of timesteps"""

    # For the runtime
    batch_size: tyro.conf.Fixed[int] = 0
    """the batch size (computed in runtime)"""
    minibatch_size: tyro.conf.Fixed[int] = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: tyro.conf.Fixed[int] = 0
    """the number of iterations (computed in runtime)"""
