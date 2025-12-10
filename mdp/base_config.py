import numpy as np
import random

class BaseConfig:
    def __init__(self, seed=None):
        self.rng: random.Random = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def get_rng(self) -> random.Random:
        return self.rng

    def get_numpy_rng(self):
        return self.np_rng