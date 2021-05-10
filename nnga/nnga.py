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
import logging
import time
from copy import deepcopy
from typing import Callable, Tuple, Mapping

from .crossover_strategy import CrossoverStrategy
from .initialization_strategy import InitializationStrategy
from .mutation_strategy import MutationStrategy
from .population_parameters import PopulationParameters

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[nnGA] [%(asctime)s] [%(levelname)-5.5s] %(message)s",
    handlers=[logging.FileHandler("nnGA.log"),
              logging.StreamHandler()])
logger = logging.getLogger('nnGA')


class nnGA(object):
    ''' Genetic Algorithm (GA) for Neural Networks
        GA is made out of 4 elements:
            1. A population of candidates
            2. A fitness function for the candidates
            3. A mutation strategy for the candidates
            4. A reproduction/crossover strategy for the candidates

        GA runs for N epochs. In each epoch every candidate
        is evaluated according to the fitness function.
        Unfit candidates are then removed from the population.
        New elements are then added to the population according to
        the mutation strategy, and the crossover strategy.

        Parameters
        -----------
        epochs: int
            Number of epochs
        fitness_function: Callable[[int, list], float]
            Fitness function. It should accept an integer, and a list
            of parameters. Returns a real value
        population_parameters: PopulationParameters
            Object that defines the properties of the population
        crossover_strategy: CrossoverStrategy
            Strategy that defines how to perform crossover
            on 2/3 candidates
        initialization_strategy: InitializationStrategy
            Strategy that defines how new candidates are randomly
            sampled from the space of candidates.
        mutation_strategy: MutationStrategy
            Strategy that defines how to perform mutation
            on a single candidate.
        initial_population: list, optional
            List of candidates that can be loaded at the beginning
        callbacks: Mapping[str, Callable[[int, list, list, float, list], float]], optional
            A dictionary of callbacks. The keys can be
            ['on_epoch_start', 'on_epoch_end', 'on_evaluation', 'population_constraints']
            Each callback should accept
            (epoch, fitnesses, population, best_result, best_network)
        num_processors: int, optional
            Number of cores to use
    '''
    def __init__(
            self,
            epochs: int,
            fitness_function: Callable[[int, list], float],
            population_parameters: PopulationParameters,
            crossover_strategy: CrossoverStrategy,
            initialization_strategy: InitializationStrategy,
            mutation_strategy: MutationStrategy,
            initial_population: list = None,
            callbacks: Mapping[
                str, Callable[[int, list, list, float, list], float]] = {},
            num_processors: int = 1):

        self.__callbacks = [
            'on_epoch_start', 'on_epoch_end', 'on_evaluation',
            'population_constraints'
        ]

        if np.any([x not in self.__callbacks for x in callbacks.keys()]):
            raise ValueError('One of the callbacks is not available')

        self.epochs = int(epochs)
        self.population = population_parameters

        self.fitness_function = fitness_function
        self.callbacks = callbacks

        self.num_processors = num_processors
        self.initial_population = initial_population

        self.mutation_strategy = mutation_strategy
        self.initialization_strategy = initialization_strategy
        self.crossover_strategy = crossover_strategy

    def _generate_initial_population(self) -> list:
        ''' Sample an initial population '''
        if not self.initial_population:
            # Sample a random population
            population = [
                self.initialization_strategy.sample_network()
                for _ in range(self.population.size)
            ]
            logger.info('Sampled random initial population.')
        else:
            # Load a population
            population = [deepcopy(x) for x in self.initial_population]
            while len(population) < self.population.size:
                for x in self.initial_population:
                    population.append(
                        self.mutation_strategy.mutate(deepcopy(x)))
                    if len(population) >= self.population.size:
                        break

            population = population[:self.population.size]
            logger.info('Loaded intial population.')
        return population

    def _select_elite_population(self, population, fitnesses):
        ''' Select elite candidates '''
        elite = sorted(
            zip(fitnesses, population), key=lambda x: x[0],
            reverse=True)[:self.population.elite_size]
        elite_res, elite_pop = zip(*elite)
        best_network = deepcopy(elite_pop[0])
        best_result = elite_res[0]
        return best_result, best_network, elite_pop

    def run(self):
        ''' Runs the GA algorithm
        Returns
        --------
        best_network: list
            Best candidate
        best_result: float
            Fitness value for the best candidate
        results: list[float]
            Fitness results for each epoch
        '''

        logger.info('Starting nnGA')
        best_network, best_result = None, 0.
        results = []

        population = self._generate_initial_population()

        for epoch in range(self.epochs):
            logger.info('Beginning of epoch {}'.format(epoch))
            if 'on_epoch_start' in self.callbacks:
                if self.callbacks['on_epoch_start'](epoch, population,
                                                    best_result, best_network):
                    break

            # Evaluate population
            fitnesses = self._evaluate_population(population)
            logger.info(
                'Best/mean/min/std values: {:.3f}/{:.3f}/{:.3f}/{:.3f}'.format(
                    np.max(fitnesses), np.mean(fitnesses), np.min(fitnesses),
                    np.std(fitnesses)))
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
            self.mutation_strategy.update_exploration_rate(epoch)

        logger.info('Search completed.')
        return best_network, best_result, results

    def _evolve_population(self, elite_population: list) -> list:
        # Add elite to population
        offsprings = []

        # Mutate elite
        while len(offsprings) < self.population.offsprings_from_elite_group:
            offsprings.append(
                self.mutation_strategy.mutate(
                    elite_population[np.random.choice(len(elite_population))]))

        offsprings.extend(deepcopy(elite_population))
        # Add additional offsprings for the leader
        # Idx 0 is the best network
        offsprings.extend([
            self.mutation_strategy.mutate(elite_population[0])
            for _ in range(self.population.offsprings_from_elite_leader)
        ])

        # Perform crossover
        offsprings.extend(
            self._crossover(self.population.crossover_size, elite_population))

        # Add random points
        while len(offsprings) < self.population.size:
            offsprings.append(self.initialization_strategy.sample_network())

        offsprings = offsprings[:self.population.size]

        # Make sure we have a unique population, if not, add new elements.
        duplicates = len(offsprings) - len(
            set(map(lambda x: hash(str(x)), offsprings)))
        offsprings.extend([
            self.initialization_strategy.sample_network()
            for _ in range(duplicates)
        ])
        return offsprings

    def _evaluate_population(self, population: list) -> list:
        time_start = time.time()
        with mp.Pool(self.num_processors) as processes:
            __args = [(
                idx,
                x,
            ) for idx, x in enumerate(population, 0)]
            fitnesses = list(processes.map(self.fitness_function, __args))
        logger.info('Completed evaluation in {:.3f} [s].'.format(time.time() -
                                                                 time_start))
        return fitnesses

    def _crossover(self, n: int, elite_population: list) -> list:
        if not n > 0 or self.crossover_strategy is None:
            return []
        L = len(elite_population)

        # Pick couples + a third one that is different from the other two
        # pairs = np.array([(i, j, 0 if i != 0 else min(j + 1, L - j + 1))
        #                   for i in range(L) for j in range(i + 1, L)])
        pairs = np.array([(i, j, 0 if i != 0 else j + 1 if j <= L//2 else j-1)
                          for i in range(L) for j in range(i + 1, L)])
        L = len(pairs)
        num_couples = n if L >= n else L
        idx = np.random.choice(len(pairs), size=num_couples, replace=False)
        couples = pairs[idx].tolist()
        couples = [couples] if num_couples == 1 else couples

        # Apply crossover to each couple
        return [
            self._crossover_couple(elite_population[x], elite_population[y],
                                   elite_population[z]) for x, y, z in couples
        ]

    def _crossover_couple(self, x: list, y: list, z: list) -> list:
        offspring = self.crossover_strategy.crossover(x, y, z)

        if np.random.uniform(
        ) < self.population.crossover_mutation_probability:
            offspring = self.mutation_strategy.mutate(offspring)

        offspring_hash = hash(str(offspring))
        while hash(str(x)) == offspring_hash or hash(str(y)) == offspring_hash:
            offspring = self.mutation_strategy.mutate(offspring)
            offspring_hash = hash(str(offspring))
        return offspring
