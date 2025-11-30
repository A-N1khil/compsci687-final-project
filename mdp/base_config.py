import random

class BaseConfig:
    def __init__(self, seed=None):
        self.rng: random.Random = random.Random(seed)

    def get_rng(self) -> random.Random:
        return self.rng