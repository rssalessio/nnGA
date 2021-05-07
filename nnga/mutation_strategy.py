# Copyright (c) [2021] Alessio Russo [alessior@kth.se]. All rights reserved.
# This file is part of nnGA.
# nnGA is free software: you can redistribute it and/or modify
# it under the terms of the MIT License. You should have received a copy of
# the MIT License along with nnGA.
# If not, see <https://opensource.org/licenses/MIT>.
#
# Code author: [Alessio Russo - alessior@kth.se]

import numpy as np


class MutationStrategy(object):
    def __init__(self,
                 network_structure: list,
                 exploration_noise: list,
                 exploration_rate_scheduler: callable = None,
                 initial_epoch: int = 0):
        self.network_structure = network_structure
        self.exploration_noise = exploration_noise
        self.exploration_rate_scheduler = exploration_rate_scheduler
        self.initial_epoch = initial_epoch

        if isinstance(exploration_noise, float):
            self.exploration_noise = [exploration_noise
                                      ] * len(network_structure)

        if len(self.exploration_noise) != len(network_structure):
            raise ValueError(
                'You should define an exploration noise for each layer.')

        if initial_epoch > 0 and exploration_rate_scheduler:
            self.exploration_rate_scheduler(
                initial_epoch, self.network_structure, self.exploration_noise)

    def mutate(self, network: list) -> list:
        return network

    def update_exploration_rate(self, epoch):
        if self.exploration_rate_scheduler:
            x = self.initial_epoch + epoch
            self.exploration_noise = self.exploration_rate_scheduler(
                x, self.network_structure, self.exploration_noise)


class GaussianMutationStrategy(MutationStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mutate(self, network: list) -> list:
        return [
            network[i] + np.random.normal(size=x) * self.exploration_noise[i]
            for i, x in enumerate(self.network_structure)
        ]


class UniformMutationStrategy(MutationStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mutate(self, network: list) -> list:
        return [
            network[i] + np.random.uniform(low=-1, high=1, size=x) *
            self.exploration_noise[i]
            for i, x in enumerate(self.network_structure)
        ]
