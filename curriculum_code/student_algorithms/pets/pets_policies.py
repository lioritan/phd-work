import gym
import torch as th
from stable_baselines3.common.policies import BasePolicy
from typing import Callable, Optional, List, Type, Dict, Any
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, create_mlp


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
        net = create_mlp(observation_space.shape[0] + action_space.shape[0], observation_space.shape[0], net_arch, activation_fn)
        self.model = nn.Sequential(*net)

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


class PETSPolicy(nn.Module):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Box,
                 lr_schedule: Callable,
                 ensemble_size: int,
                 net_arch: Optional[List[int]] = None,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 _init_setup_model=False):
        super(PETSPolicy, self).__init__()
        self.action_space = action_space

        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [64, 64]
            else:
                net_arch = []

        self.models = [
            DynamicsNetwork(observation_space, action_space, lr_schedule, net_arch, activation_fn, optimizer_class, optimizer_kwargs)
            for i in range(ensemble_size)]

        # TODO: extract

    def to(self, device=..., dtype=...,
           non_blocking: bool = ...):
        self.register_parameter("dummy", th.nn.Parameter(th.tensor([0], dtype=th.float32, device=device)))
        for net in self.models:
            net.to(device)
        return self

    def forward(self, obs: th.Tensor, action: th.Tensor):
        noisy_obs = obs + th.normal(0, 1, size=obs.size(), device=self.dummy.device)
        chosen_net_ind = th.randint(len(self.models), size=(1, )).item()
        return self.models[chosen_net_ind](noisy_obs, action)

    def get_next(self, observation, action):
        with th.no_grad():
            return self.forward(observation, action)

    def scale_action(self, action):
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action):
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))
