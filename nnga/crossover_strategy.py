# Copyright (c) [2021] Alessio Russo [alessior@kth.se]. All rights reserved.
# This file is part of nnGA.
# nnGA is free software: you can redistribute it and/or modify
# it under the terms of the MIT License. You should have received a copy of
# the MIT License along with nnGA.
# If not, see <https://opensource.org/licenses/MIT>.
#
# Code author: [Alessio Russo - alessior@kth.se]

import numpy as np

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

    def crossover(self, x: list, y: list) -> list:
        pass


class MutateParentsCrossoverStrategy(CrossoverStrategy):
    def __init__(self, p=0.5, mutation_strategy, *args):
        super().__init__(*args)
        self.p = p
        self.mutation_strategy = mutation_strategy
        self.name = 'mutate-parents'

        assert mutation_strategy is not None
        assert p > 0. and p < 1.

    def crossover(self, x: list, y: list) -> list:
        if np.random.uniform() < 0.5:
            return self.mutation_strategy.mutate(x)
        else:
            return self.mutation_strategy.mutate(y)


class LayerBasedCrossoverStrategy(CrossoverStrategy):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'layer-based'

    def crossover(self, x: list, y: list) -> list:
        # Choose a layer that acts as a crossover point
        crossover_point = np.random.randint(
            low=1, high=len(self.network_structure))
        return [
            x[i] if i < crossover_point else y[i]
            for i in range(len(self.network_structure))
        ]

class BasicCrossoverStrategy(CrossoverStrategy):
    def __init__(self, *args):
        super().__init__(*args)
        self.name = 'basic'

    def crossover(self, x: list, y: list) -> list:
       # Flatten parents' parameters
        x_flattened = reduce(lambda a, b: a + b,
                             [k.flatten().tolist() for k in x])
        y_flattened = reduce(lambda a, b: a + b,
                             [k.flatten().tolist() for k in y])

        # Choose crossover point
        crossover_point = np.random.randint(low=1, high=len(x_flattened))

        # Generate offspring
        offspring = deepcopy(x_flattened)
        offspring[crossover_point:] = deepcopy(
            y_flattened[crossover_point:])
        return reshape(offspring, x)

