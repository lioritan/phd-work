import random
from abc import ABC, abstractmethod


class EnvironmentParameter(ABC):
    @abstractmethod
    def sample(self, nearest_value=None):
        pass


class CategoricalParameter(EnvironmentParameter):
    def __init__(self, values):
        self.values = values

    def sample(self, nearest_value=None):
        if nearest_value is None:
            return random.choice(self.values)
        else:
            val_index = self.values.index(nearest_value)
            new_index = random.normalvariate(val_index, min(len(self.values)-val_index, val_index))
            if new_index < 0:
                return self.values[0]
            if new_index >= len(self.values) - 1:
                return self.values[-1]
            else:
                return self.values[round(new_index)]


class ContinuousParameter(EnvironmentParameter):
    def __init__(self, minimum, maximum):
        self.min_val = minimum
        self.max_val = maximum

    def sample(self, nearest_value=None):
        if nearest_value is None:
            return random.uniform(self.min_val, self.max_val)
        else:
            new_value = random.normalvariate(nearest_value, min(self.max_val - nearest_value, nearest_value - self.min_val))
            if new_value < self.min_val:
                return self.min_val
            if new_value > self.max_val:
                return self.max_val
            else:
                return new_value
