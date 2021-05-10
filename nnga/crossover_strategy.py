# Copyright (c) [2021] Alessio Russo [alessior@kth.se]. All rights reserved.
# This file is part of nnGA.
# nnGA is free software: you can redistribute it and/or modify
# it under the terms of the MIT License. You should have received a copy of
# the MIT License along with nnGA.
# If not, see <https://opensource.org/licenses/MIT>.
#
# Code author: [Alessio Russo - alessior@kth.se]

import numpy as np
from functools import reduce
from copy import deepcopy
from typing import List


def reshape(lst, base):
    last = 0
    res = []
    for ele in base:
        S = np.shape(ele)
        L = np.prod(S)
        res.append(np.reshape(lst[last:last + L], S))
        last += L

    return res


class CrossoverStrategy(object):
    def __init__(self, network_structure: list):
        self.network_structure = network_structure

    def crossover(self, x: list, y: list, z: list) -> list:
        pass


class MutateParentsCrossoverStrategy(CrossoverStrategy):
    def __init__(self, p, mutation_strategy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.mutation_strategy = mutation_strategy
        self.name = 'mutate-parents'

        assert mutation_strategy is not None
        assert p > 0. and p < 1.

    def crossover(self, x: list, y: list, z: list) -> list:
        if np.random.uniform() < 0.5:
            return self.mutation_strategy.mutate(x)
        else:
            return self.mutation_strategy.mutate(y)


class LayerBasedCrossoverStrategy(CrossoverStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'layer-based'

        if len(self.network_structure) == 1:
            raise ValueError(
                'Attention, you passed a structure with just 1 layer!'
                'I can\'t use layer-based crossover.')

    def crossover(self, x: list, y: list, z: list) -> list:
        # Choose a layer that acts as a crossover point
        crossover_point = np.random.randint(
            low=1, high=len(self.network_structure))
        return [
            x[i] if i < crossover_point else y[i]
            for i in range(len(self.network_structure))
        ]


class BasicCrossoverStrategy(CrossoverStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'basic'

    def crossover(self, x: list, y: list, z: list) -> list:
        # Flatten parents' parameters
        x_flattened = reduce(lambda a, b: a + b,
                             [k.flatten().tolist() for k in x])
        y_flattened = reduce(lambda a, b: a + b,
                             [k.flatten().tolist() for k in y])

        # Choose crossover point
        crossover_point = np.random.randint(low=1, high=len(x_flattened))

        # Generate offspring
        offspring = deepcopy(x_flattened)
        offspring[crossover_point:] = deepcopy(y_flattened[crossover_point:])
        return reshape(offspring, x)


class Best1BinCrossoverStrategy(CrossoverStrategy):
    def __init__(self, weights: List[float] = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'best1bin-df'

        self.weights = np.asarray(weights)
        if isinstance(weights, float):
            self.weights = np.asarray([weights] * len(self.network_structure))

        assert np.all(self.weights > 0.) and np.all(self.weights <= 2.)

    def crossover(self, x: list, y: list, z: list) -> list:
        return [
            z[i] + self.weights[i] * (x[i] - y[i])
            for i, _ in enumerate(self.network_structure)
        ]
