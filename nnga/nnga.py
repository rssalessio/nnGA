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

from .crossover_strategy import CrossoverStrategy
from .initialization_strategy import InitializationStrategy
from .mutation_strategy import MutationStrategy
from .population_parameters import PopulationParameters


class nnGA(object):
    def __init__(self,
                 network_structure: list,
                 epochs: int,
                 fitness_function: callable,
                 fitness_function_args: tuple,
                 population_parameters: PopulationParameters,
                 crossover_strategy: CrossoverStrategy,
                 initialization_strategy: InitializationStrategy,
                 mutation_strategy: MutationStrategy,
                 initial_parameters: list = None,
                 callbacks: dict = {},
                 num_processors: int = 1):

        self.__callbacks = [
            'on_epoch_start', 'on_epoch_end', 'on_evaluation',
            'population_constraints'
        ]

        if np.any([x not in self.__callbacks for x in callbacks.keys()]):
            raise ValueError('One of the callbacks is not available')

        self.epochs = int(epochs)
        self.network_structure = deepcopy(network_structure)
        self.population = population_parameters


        self.fitness_function = fitness_function
        self.fitness_function_args = fitness_function_args
        self.callbacks = callbacks

        self.num_processors = num_processors
        self.initial_parameters = initial_parameters
        
        self.mutation_strategy = mutation_strategy
        self.initialization_strategy = initialization_strategy
        self.crossover_strategy = crossover_strategy


    def _generate_initial_population(self) -> list:
        # Generate initial population
        if not self.initial_parameters:
            print('[NNGA] Sampled random initial population.')
            population = [
                self.initialization_strategy.sample_network() for _ in range(self.population.size)
            ]
        else:
            print('[NNGA] Loaded intial popolation.')
            population = self._evolve_population([
                deepcopy(self.initial_parameters)
                for _ in range(self.population.elite_size)
            ])
        return population

    def _select_elite_population(self, population, fitnesses):
        # Select elite population
        elite = sorted(
            zip(fitnesses, population), key=lambda x: x[0],
            reverse=True)[:self.population.elite_size]
        elite_res, elite_pop = zip(*elite)
        best_network = deepcopy(elite_pop[0])
        best_result = elite_res[0]
        return best_result, best_network, elite_pop

    def run(self):
        best_network, best_result = None, 0.
        results = []

        population = self._generate_initial_population()

        for epoch in range(self.epochs):
            if 'on_epoch_start' in self.callbacks:
                if self.callbacks['on_epoch_start'](epoch, population,
                                                    best_result, best_network):
                    break

            # Evaluate population
            fitnesses = self._evaluate_population(population)
            results.append(fitnesses)

            # Select elite
            best_result, best_network, elite_pop = self._select_elite_population(
                population, fitnesses)

            if 'on_evaluation' in self.callbacks:
                if self.callbacks['on_evaluation'](epoch, fitnesses,
                                                   population, best_result,
                                                   best_network):
                    break

            if epoch < self.epochs - 1:
                # Perform mutation/crossover
                population = self._evolve_population(elite_pop)
                if 'population_constraints' in self.callbacks:
                    population = self.callbacks['population_constraints'](
                        population)

            if 'on_epoch_end' in self.callbacks:
                if self.callbacks['on_epoch_end'](epoch, fitnesses, population,
                                                  best_result, best_network):
                    break

        return best_network, best_result, results

    def _evolve_population(self, elite_population: list) -> list:
        # Add elite to population
        offsprings.extend(deepcopy(elite_population))

        # Mutate elite
        while len(offsprings) < self.population.offsprings_from_elite_group:
            for x in elite_population:
                offsprings.append(self.mutation_strategy.mutate(x))
                if len(offsprings) >= self.population.offsprings_from_elite_group:
                    break

        # Add additional offsprings for the leader
        # Idx 0 is the best network
        offsprings.extend([
            self.mutation_strategy.mutate(elite_population[0])
            for _ in range(self.population.offsprings_from_elite_leader)
        ])

        # Perform crossover
        offsprings.extend(self._crossover(crossover, elite_population))

        # Add random points
        while len(offsprings) < self.population.size:
            offsprings.append(self.initialization_strategy.sample_network())

        offsprings = offsprings[:self.population.size]

        # Make sure we have a unique population, if not, add new elements.
        duplicates = len(offsprings) - len(
            set(map(lambda x: hash(str(x)), offsprings)))
        offsprings.extend([self.initialization_strategy.sample_network() for _ in range(duplicates)])
        return offsprings

    def _evaluate_population(self, population: list) -> list:
        with mp.Pool(self.num_processors) as processes:
            __args = [(
                idx,
                x,
                *self.fitness_function_args,
            ) for idx, x in enumerate(population, 0)]
            fitnesses = list(processes.map(self.fitness_function, __args))
        return fitnesses

    def _crossover(self, n: int, elite_population: list) -> list:
        if not n > 0 or self.crossover_strategy is None:
            return []
        L = len(elite_population)

        # Pick couples
        pairs = np.array([(i, j) for i in range(L) for j in range(i + 1, L)])
        L = len(pairs)
        num_couples = n if L >= n else L
        idx = np.random.choice(len(pairs), size=num_couples, replace=False)
        couples = pairs[idx].tolist()
        couples = [couples] if num_couples == 1 else couples

        # Apply crossover to each couple
        return [
            self._crossover_couple(elite_population[x], elite_population[y])
            for x, y in couples
        ]

    def _crossover_couple(self, x: list, y: list) -> list:
        offspring = self.crossover_strategy.crossover(elite_population[x], elite_population[y])

        if np.random.uniform() < self.population.crossover_mutation_probability:
            offspring = self.mutation_strategy.mutate(offspring)

        offspring_hash = hash(str(offspring))
        while hash(str(x)) == offspring_hash or hash(str(y)) == offspring_hash:
            offspring = self.mutation_strategy.mutate(offspring)
            offspring_hash = hash(str(offspring))
        return offspring
