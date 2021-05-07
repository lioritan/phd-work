import gym
import torch as th
from stable_baselines3.common.policies import BasePolicy
from typing import Callable, Optional, List, Type, Dict, Any

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from torch import nn

from student_algorithms.nn_model_ensemble.nnmm.nn_mixture import NNMixture
from student_algorithms.nn_model_ensemble.nnmm.nn_mixture_weighted import NNMixtureWeighted


class NNMMPolicy(nn.Module):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Box,
                 lr_schedule: Callable,
                 alpha: float,
                 merge_burnin: int,
                 merge_threshold: float,
                 is_mixed_model: bool,

                 net_arch: Optional[List[int]] = None,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 use_sde=None,
                 verbose=0,
                 _init_setup_model=False):
        super(NNMMPolicy, self).__init__()
        self.action_space = action_space

        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [64, 64]
            else:
                net_arch = []

        if is_mixed_model:
            model_class = NNMixtureWeighted
        else:
            model_class = NNMixture

        self.model = model_class(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            alpha=alpha,
            window_prob=1.0,
            self_prob=0.05,
            merge_burnin=merge_burnin,
            merge_threshold=merge_threshold,
            net_arch=net_arch,
            activation_fn=activation_fn,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def to(self, device=..., dtype=...,
           non_blocking: bool = ...):
        self.register_parameter("dummy", th.nn.Parameter(th.tensor([0], dtype=th.float32, device=device)))
        self.model.to(device)
        return self
