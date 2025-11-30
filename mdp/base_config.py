import random

class BaseConfig:
    def __init__(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()

    def get_rng(self):
        return self.rng