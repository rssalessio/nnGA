# Copyright (c) [2021] Alessio Russo [alessior@kth.se]. All rights reserved.
# This file is part of nnGA.
# nnGA is free software: you can redistribute it and/or modify
# it under the terms of the MIT License. You should have received a copy of
# the MIT License along with nnGA.
# If not, see <https://opensource.org/licenses/MIT>.
#
# Code author: [Alessio Russo - alessior@kth.se]

import multiprocessing as mp
import numpy as np
from functools import reduce
from copy import deepcopy

def reshape(lst, base):
    last = 0
    res = []
    for ele in base:
        S = np.shape(ele)
        L = np.prod(S)
        res.append(np.reshape(lst[last : last + L], S))
        last += L
          
    return res


class nnGA(object):
    def __init__(
        self,
        network_structure: list,
        epochs: int,
        population_size: int,
        fitness_function: callable,
        fitness_function_args: tuple,
        exploration_noise: list,
        elite_fraction: float = 0.1,
        offsprings_from_elite_fraction: float = 0.2,
        crossover_fraction: float = 0.5,
        rnd_offsprings_fraction: float = 0.1,
        leader_offsprings_fraction: float = 0.1,
        crossover_type: str = 'layer-based',
        crossover_mutation_probability: float = 0.1,
        callbacks: dict = {},
        num_processors: int = 1
        ):

        self.__crossover_types = [
            'basic-crossover',
            'layer-based',
            'mutate-parents'
            'none'
        ]

        self.__callbacks = [
            'on_epoch_start',
            'on_epoch_end',
            'on_evaluation',
            'population_constraints'
        ]

        self.epochs = int(epochs)
        self.network_structure = deepcopy(network_structure)
        self.population_size = int(population_size)
        self.elite_fraction = float(elite_fraction)
        self.offsprings_from_elite_fraction = float(offsprings_from_elite_fraction)
        self.crossover_fraction = float(crossover_fraction)
        self.rnd_offsprings_fraction = float(rnd_offsprings_fraction)
        self.leader_offsprings_fraction = float(leader_offsprings_fraction)
        self.fitness_function = fitness_function
        self.fitness_function_args = fitness_function_args
        self.exploration_noise = exploration_noise
        self.crossover_type = crossover_type
        self.crossover_mutation_probability = crossover_mutation_probability
        self.callbacks = callbacks
        self.num_processors = num_processors

        if not np.isclose(elite_fraction + offsprings_from_elite_fraction + crossover_fraction \
                + rnd_offsprings_fraction + leader_offsprings_fraction, 1.):
            print('You have not provided a correct proportion for the elite/offsprings (it should sum to 1)')

        self._processes = mp.Pool(self.num_processors)

    def run(self):
        # Generate initial population
        population = [self._random_network() for _ in range(self.population_size)]
        elite_size = int(np.ceil(self.population_size *  self.elite_fraction))
        best_network, best_result = None, 0.
        results = []

        for epoch in range(self.epochs):
            if 'on_epoch_start' in self.callbacks:
                if self.callbacks['on_epoch_start'](epoch, population, best_result, best_network):
                    break

            # Evaluate population
            fitnesses = self._evaluate_population(population)
            results.append(fitnesses)

            # Select elite population
            elite = sorted(
                zip(fitnesses, population), key=lambda x: x[0], reverse=True)[:elite_size]
            elite_res, elite_pop = zip(*elite)
            best_network = deepcopy(elite_pop[0])
            best_result = elite_res[0]

            if 'on_evaluation' in self.callbacks:
                if self.callbacks['on_evaluation'](epoch, fitnesses, population, best_result, best_network):
                    break

            if epoch < self.epochs - 1:
                # Perform mutation/crossover
                population = self._evolve_population(elite_pop)
                if 'population_constraints' in self.callbacks:
                    population = self.callbacks['population_constraints'](population)

            if 'on_epoch_end' in self.callbacks:
                if self.callbacks['on_epoch_end'](epoch, fitnesses, population, best_result, best_network):
                    break

        self._processes.close()
        self._processes.join()
        return best_network,  best_result, results

    def _evolve_population(self, elite_population):
        offsprings_from_elite = int(np.ceil(self.population_size * self.offsprings_from_elite_fraction))
        crossover = int(np.ceil(self.population_size * self.crossover_fraction))
        rnd_offsprings = int(np.ceil(self.population_size * self.rnd_offsprings_fraction))
        leader_offsprings = int(np.ceil(self.population_size * self.leader_offsprings_fraction))
        offsprings = []

        # Add elite to population
        offsprings.extend(deepcopy(elite_population))

        # Mutate elite
        while len(offsprings) < offsprings_from_elite:
            for x in elite_population:
                offsprings.append(self._mutate_network(x))
                if len(offsprings) >= offsprings_from_elite:
                    break

        # Add additional offsprings for the leader
        # Idx 0 is the best network
        offsprings.extend([self._mutate_network(elite_population[0])
            for _ in range(leader_offsprings)])
        
        # Perform crossover
        offsprings.extend(self._crossover(crossover, elite_population))

        # Add random points
        while len(offsprings) < self.population_size:
            offsprings.append(self._random_network())

        offsprings = offsprings[:self.population_size]

        # Make sure we have a unique population, if not, add new elements.
        duplicates = len(offsprings) - len(set(map(lambda x: hash(str(x)), offsprings)))
        offsprings.extend([self._random_network() for _ in range(duplicates)])
        return offsprings

    def _random_network(self):
        return [
            np.random.normal(size=x) for x in self.network_structure
        ]

    def _mutate_network(self, network: list):
        return [
            network[i] + np.random.normal(size=x) * self.exploration_noise[i]
                for i, x in enumerate(self.network_structure, 0)
        ]

    def _evaluate_population(self, population: list):
        __args = [(x, *self.fitness_function_args, ) for x in population]
        fitnesses = list(self._processes.map(self.fitness_function, __args))
        return fitnesses

    def _crossover(self, n: int, elite_population: list):
        if not n > 0 or self.crossover_type == 'none':
            return []
        L = len(elite_population)

        # Pick couples
        pairs = np.array([
            (i,j) for i in range(L)
                for j in range(i+1, L)])
        L = len(pairs)
        l = n if L >= n else L
        idx = np.random.choice(len(pairs), size=l, replace=False)
        couples = pairs[idx].tolist()
        couples = [couples] if l == 1 else couples

        # Apply crossover to each couple
        return [
            self._crossover_couple(elite_population[x], elite_population[y])
                for x, y in couples]

    def _crossover_couple(self, x: list, y: list):
        if self.crossover_type == 'mutate-parents':
            # Randomly mutate the father or the mother
            return self._mutate_network(x) if np.random.uniform() < 0.5 else \
                        self._mutate_network(y)
        elif self.crossover_type == 'layer-based':
            # Choose a layer that acts as a crossover point
            l = np.random.randint(low=1, high=len(self.network_structure))
            offspring = [x[i] if i < l else y[i] for i in range(len(self.network_structure))]
        elif self.crossover_type == 'basic-crossover':
            # Flatten parents' parameters
            x_flattened = reduce(lambda a,b: a+b, [k.flatten().tolist() for k in x])
            y_flattened = reduce(lambda a,b: a+b, [k.flatten().tolist() for k in y])
            
            # Choose crossover point
            l = np.random.randint(low=1, high=len(x_flattened))

            # Generate offspring
            offspring = deepcopy(x_flattened)
            offspring[l:] = deepcopy(y_flattened[l:])
            offspring = reshape(offspring, x)
        else:
            raise ValueError('Crossover type {} not implemented'.format(self.crossover_type))


        if np.random.uniform() < self.crossover_mutation_probability:
            offspring = self._mutate_network(offspring)

        offspring_hash = hash(str(offspring))
        while hash(str(x)) == offspring_hash or hash(str(y)) == offspring_hash:
            offspring = self._mutate_network(offspring)
            offspring_hash = hash(str(offspring))
        return offspring



