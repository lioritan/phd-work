from abc import ABC, abstractmethod
from typing import Dict, Any

from environment.environment_parameter import EnvironmentParameter


class EnvironmentWrapper(ABC):
    def __init__(self, env_name: str, parameter_space: Dict[str, EnvironmentParameter]):
        self.parameters = parameter_space
        self.name = env_name

    @abstractmethod
    def create_env(self, parameter_values: Dict[str, Any]):
        pass
