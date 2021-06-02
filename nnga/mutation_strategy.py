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

    def update_parameters(self, epoch: int, elite_networks: list):
        pass

    def mutate(self, network: list) -> list:
        return network

    def update_exploration_rate(self, epoch):
        if self.exploration_rate_scheduler:
            x = self.initial_epoch + epoch
            self.exploration_noise = self.exploration_rate_scheduler(
                x, self.network_structure, self.exploration_noise)


class GaussianMutationStrategy(MutationStrategy):
    """
        This strategy implements Gaussian exploration,
        where we add to each layer noise from a Gaussian
        distribution
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mutate(self, network: list) -> list:
        return [
            network[i] + np.random.normal(size=x) * self.exploration_noise[i]
            for i, x in enumerate(self.network_structure)
        ]


class UniformMutationStrategy(MutationStrategy):
    """
        This strategy implements Uniform exploration,
        where we add to each layer noise from a Uniform
        distribution
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mutate(self, network: list) -> list:
        return [
            network[i] + np.random.uniform(low=-1, high=1, size=x) *
            self.exploration_noise[i]
            for i, x in enumerate(self.network_structure)
        ]

class SimpleCrossEntropyMutationStrategy(MutationStrategy):
    """
        This strategy implements a very simple version of the
        CrossEntropy method, where the standard deviation
        is computed elementwise.
        Check https://arxiv.org/pdf/1604.00772.pdf
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.means = []
        self.std = []

    def update_parameters(self, epoch: int, elite_networks: list):
        self.means = [
            np.mean([network[layer] for network in elite_networks], axis=0)
            for layer, _ in enumerate(self.network_structure)
        ]
        self.std = [
            np.std([network[layer] for network in elite_networks], axis=0) + self.exploration_noise[layer]
            for layer, _ in enumerate(self.network_structure)
        ]

    def mutate(self, network: list) -> list:
        return [
            self.means[i] + np.random.normal(size=x) * self.std[i]
            for i, x in enumerate(self.network_structure)
        ]


class CrossEntropyMutationStrategy(MutationStrategy):
    """
        This strategy implements a basic version of the
        CrossEntropy method.
        Check https://arxiv.org/pdf/1604.00772.pdf
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = 0
        self.cov = 0

    def _flatten_network(self, network: list) -> np.ndarray:
        return np.concatenate([layer.flatten() for layer in network])

    def _reshape_list_into_network(self, parameters: np.ndarray) -> list:
        # Reshape a list of parameters into a network
        network = []
        last = 0
        for S in self.network_structure:
            L = np.prod(S)
            network.append(np.reshape(parameters[last:last + L], S))
            last += L
        return network

    def update_parameters(self, epoch: int, elite_networks: list):
        networks = np.asarray([self._flatten_network(network) for network in elite_networks])

        # Compute mean
        mean = np.mean(networks, axis=0)

        # Compute covariance matrix
        cov = 0
        for idx, _ in enumerate(elite_networks):
            x = (networks[idx] - mean)[:, None]
            cov += x @ x.T

        cov /= max(1, len(elite_networks) - 1)

        # Add regularization
        cov += self.exploration_noise[0] * np.identity(len(mean))

        self.mean = mean
        self.cov = cov

    def mutate(self, network: list) -> list:
        new_sample = np.random.multivariate_normal(self.mean, self.cov)
        return self._reshape_list_into_network(new_sample)
