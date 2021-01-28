from abc import ABC, abstractmethod
from typing import Dict, Any


class EnvironmentWrapper(ABC):
    def __init__(self):
        self.parameters = {}
        self.name = ""

    @abstractmethod
    def create_env(self, parameter_values: Dict[str, Any]):
        pass
