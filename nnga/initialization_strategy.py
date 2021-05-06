# Copyright (c) [2021] Alessio Russo [alessior@kth.se]. All rights reserved.
# This file is part of nnGA.
# nnGA is free software: you can redistribute it and/or modify
# it under the terms of the MIT License. You should have received a copy of
# the MIT License along with nnGA.
# If not, see <https://opensource.org/licenses/MIT>.
#
# Code author: [Alessio Russo - alessior@kth.se]

import numpy as np

class InitializationStrategy(object):
    def __init__(self, network_structure: list):
        self.network_structure = network_structure

    def sample_network(self) -> list:
        pass


class GaussianInitializationStrategy(InitializationStrategy):
    def __init__(self, mean, std, *args):
        super().__init__(*args)
        self.mean = mean
        self.std = std

        assert len(self.mean) == len(self.std) \
            and len(self.mean) == len(self.network_structure)
        assert self.std > 0

    def sample_network(self) -> list:
        return [self.mean[idx] + self.std[x] * np.random.normal(size=x)
                for idx, x in enumerate(self.network_structure)]


class UniformInitializationStrategy(InitializationStrategy):
    def __init__(self, low, high, *args):
        super().__init__(*args)

        self.low = low
        self.high = high

        assert len(self.low) == len(self.high) \
            and len(self.low) == len(self.network_structure)
        assert np.all(self.high > self.low)

    def sample_network(self) -> list:
        return [np.random.uniform(low=self.low[x], high=self.high[idx], size=x)
                for idx, x in enumerate(self.network_structure)]

