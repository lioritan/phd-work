import gym
import torch as th
from typing import Callable, Optional, List, Type, Dict, Any
from torch import nn

from stable_baselines3.common.torch_layers import create_mlp


class DynamicsNetwork(nn.Module):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable,

                 net_arch: Optional[List[int]] = None,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None, ):
        super(DynamicsNetwork, self).__init__()
        net = create_mlp(observation_space.shape[0] + action_space.shape[0], observation_space.shape[0], net_arch,
                         activation_fn)
        self.model = nn.Sequential(*net)
        self.n_points = 0

        self.optimizer_class = optimizer_class
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5
        self.optimizer_kwargs = optimizer_kwargs

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, action: th.Tensor):
        combined_in = th.cat((obs, action), dim=1)
        return self.model(combined_in)
