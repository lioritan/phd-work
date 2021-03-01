from typing import Optional, Callable, List, Type, Dict, Any, Union

import gym
import torch as th
from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from torch import nn
from stable_baselines3.dqn.policies import QNetwork


class EBQLEnsemblePolicy(BasePolicy):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable,
                 ensemble_size: int,
                 net_arch: Optional[List[int]] = None,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 ):
        super(EBQLEnsemblePolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [64, 64]
            else:
                net_arch = []

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.normalize_images = normalize_images

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        # TODO: shared feature extractor - requires re-writing some MLP code
        self.features_extractor = self.make_features_extractor()
        net_args = self._update_features_extractor(self.net_args, features_extractor=self.features_extractor)

        self.nets = [QNetwork(**net_args).to(self.device) for i in range(ensemble_size)]
        self.optimizers = [self.optimizer_class(self.nets[i].parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
                           for i in range(ensemble_size)]

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        q_values = self.forward(observation)
        # Greedy action
        action = q_values.argmax(dim=0).reshape(-1)
        return action

    def to(self, device=..., dtype=...,
           non_blocking: bool = ...):
        self.register_parameter("dummy", th.nn.Parameter(th.tensor([0], dtype=th.float32, device=device)))
        for net in self.nets:
            net.to(device)
        return self

    def forward(self, obs: th.Tensor) -> th.Tensor:
        clean_obs = self.extract_features(obs)
        q_tensor = th.vstack([net(clean_obs) for net in self.nets])
        return th.sum(q_tensor, dim=0)


# if discrete, action is x outs, if box action is box shape but not handling it right now
register_policy("MlpPolicy", EBQLEnsemblePolicy)
