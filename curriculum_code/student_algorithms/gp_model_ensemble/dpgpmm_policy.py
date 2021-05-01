import gym
import torch as th
from stable_baselines3.common.policies import BasePolicy
from typing import Callable, Optional, List, Type, Dict, Any

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from torch import nn

from student_algorithms.gp_model_ensemble.dpgpmm.dpgpmm import DPGPMM


class DPGPMMPolicy(nn.Module):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Box,
                 lr_schedule: Callable,
                 alpha: float,
                 merge_burnin: int,
                 merge_threshold: float,
                 gp_iter: int,
                 max_inducing_point: int,
                 trigger_induce: int,
                 sample_number: int,

                 net_arch: Optional[List[int]] = None,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 use_sde=None,
                 _init_setup_model=False):
        super(DPGPMMPolicy, self).__init__()
        self.action_space = action_space

        self.gpmm_model = DPGPMM({
            "alpha": alpha,  # new model prior
            "ada_alpha": False,
            "merge": False,
            "merge_burnin": merge_burnin,  # num of points until new model trained
            "merge_threshold": merge_threshold,  # kl divergence distance to merge (if 2 models more similar, merge)
            "window_prob": 0.001,  # the initial transition bias to other cluster
            "self_prob": 1.0,  # the initial self-transition bias
            "lr": lr_schedule(1),
            "gp_iter": gp_iter,  # iteration loops of GP
            "model_type": "sample",
            "max_inducing_point": max_inducing_point,  # Used in [sparse/sample]. the data number after do a sparse operation
            "trigger_induce": trigger_induce,  # Used in [sparse/sample]. when n is larger than this value, do a sparse
            # operation
            "sample_number": sample_number,  # Used in [sample]. number of MC samples to find the highest lower bound
            "param": [  # GP initilize and constraint parameters
                0.1,  # noise_covar initilize
                0.001,  # noise_covar constraint
                0.0,  # constant initilize
                10.0,  # outputscale initilize
                1.0,  # lengthscale initilize
            ],

            "state_dim": observation_space.shape[0],
            "action_dim": action_space.shape[0],

        })


    def to(self, device=..., dtype=...,
           non_blocking: bool = ...):
        self.register_parameter("dummy", th.nn.Parameter(th.tensor([0], dtype=th.float32, device=device)))
        return self
