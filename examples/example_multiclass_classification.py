# Copyright (c) [2021] Alessio Russo [alessior@kth.se]. All rights reserved.
# This file is part of nnGA.
# nnGA is free software: you can redistribute it and/or modify
# it under the terms of the MIT License. You should have received a copy of
# the MIT License along with nnGA.
# If not, see <https://opensource.org/licenses/MIT>.
#
# Code author: [Alessio Russo - alessior@kth.se]
#

import sys
import numpy as np
import torch
sys.path.append("..")


from nnga import nnGA, GaussianInitializationStrategy, \
    GaussianMutationStrategy, Best1BinCrossoverStrategy, \
    PopulationParameters

# Example Multiclass classification
# ------------
# In this example we see how to use Genetic Algorithms
# to solve a multiclass classification problem
#
# Required dependencies:
# - Numpy
# - Pytorch
#


def make_network(parameters=None):
    neural_network = torch.nn.Sequential(
        torch.nn.Linear(2, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3))

    if parameters:
        state_dict = neural_network.state_dict()
        for x, k in enumerate(state_dict.keys(), 0):
            state_dict[k] = torch.tensor(parameters[x])
        neural_network.load_state_dict(state_dict)
    return neural_network


def fitness(idx, parameters, data):
    trn_data, labels = data[0], data[1].flatten()

    network = make_network(parameters)
    with torch.no_grad():
        y = network(trn_data)
        loss = -torch.nn.CrossEntropyLoss()(y, labels.long()).item()
    return loss


def on_evaluation(epoch, fitnesses, population, best_result, best_network,
                  data):
    val_data, labels = data[0], data[1].flatten()
    network = make_network(best_network)
    with torch.no_grad():
        y = network(val_data)
        loss = torch.nn.CrossEntropyLoss()(y, labels.long()).item()
        print('Evaluation loss: {} [Nats]'.format(loss))
    return False


def make_dataset():
    # Generate dataset
    N_training = 100
    N_validation = 50
    N = N_training + N_validation

    X0 = np.array([-1, 0]).T + 0.5 * np.random.normal(size=(N, 2))
    X1 = np.array([1, 0]).T + 0.5 * np.random.normal(size=(N, 2))
    X2 = np.array([0, 1]).T + 0.3 * np.random.normal(size=(N, 2))
    X = np.concatenate([X0, X1, X2])

    labels = np.zeros((3 * N, 1))
    labels[N:2 * N] = 1
    labels[2 * N:] = 2

    # Training dataset
    indices = np.random.permutation(3 * N)
    trn_indices, val_indices = indices[:N_training], indices[N_training:]
    Tdataset = ([
        torch.tensor(X[trn_indices], dtype=torch.float32),
        torch.tensor(labels[trn_indices], dtype=torch.float32)
    ])
    Vdataset = ([
        torch.tensor(X[val_indices], dtype=torch.float32),
        torch.tensor(labels[val_indices], dtype=torch.float32)
    ])
    return Tdataset, Vdataset


def _fitness(args):
    return fitness(*args, data=trn_data)


def _evaluate(*args):
    return on_evaluation(*args, data=val_data)


if __name__ == '__main__':
    nn = make_network().state_dict()
    network_structure = [list(v.shape) for _, v in nn.items()]
    population = PopulationParameters(population_size=22)
    mutation = GaussianMutationStrategy(network_structure, 1e-1)
    crossover = Best1BinCrossoverStrategy(1., network_structure)
    init = GaussianInitializationStrategy(
        mean=0., std=1., network_structure=network_structure)

    trn_data, val_data = make_dataset()

    ga = nnGA(
        epochs=100,
        fitness_function=_fitness,
        population_parameters=population,
        mutation_strategy=mutation,
        initialization_strategy=init,
        crossover_strategy=crossover,
        callbacks={'on_evaluation': _evaluate},
        num_processors=1)
    ga.run()
