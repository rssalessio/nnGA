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
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.append("..")
from nnga import nnGA, GaussianInitializationStrategy, \
    GaussianMutationStrategy, LayerBasedCrossoverStrategy, \
    PopulationParameters

# Example Reinforcement Learning - Cartpole
# ------------
# In this example we see how to use Genetic Algorithms
# to solve the cartpole environment

import gym


def make_network(parameters=None):
    neural_network = torch.nn.Sequential(
        torch.nn.Linear(4, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))

    if parameters:
        state_dict = neural_network.state_dict()
        for x, k in enumerate(state_dict.keys(), 0):
            state_dict[k] = torch.tensor(parameters[x])
        neural_network.load_state_dict(state_dict)
    return neural_network


def fitness(data: list):
    idx, parameters, episodes = data
    env = gym.make('CartPole-v1')
    episode_reward_list = []
    network = make_network(parameters)

    rewards = []
    for n in range(episodes):
        done = False
        state = env.reset()
        total_episode_reward = 0.
        while not done:
            with torch.no_grad():
                action = network(torch.tensor([state], dtype=torch.float32))
                action = torch.argmax(action).item()
            state, reward, done, _ = env.step(action)
            total_episode_reward += reward

        env.close()
        rewards.append(total_episode_reward)

    return np.mean(rewards) - (1.96 * np.std(rewards) / np.sqrt(episodes))


def on_evaluation(epoch, fitnesses, population, best_result, best_network):
    if best_result > 495:
        return True
    return False


if __name__ == '__main__':
    nn = make_network().state_dict()
    network_structure = [list(v.shape) for _, v in nn.items()]
    population = PopulationParameters(population_size=100)
    mutation = GaussianMutationStrategy(network_structure, 1e-1)
    crossover = LayerBasedCrossoverStrategy(network_structure)
    init = GaussianInitializationStrategy(
        mean=0., std=1., network_structure=network_structure)

    ga = nnGA(
        epochs=100,
        fitness_function=fitness,
        fitness_function_args=(100, ),
        population_parameters=population,
        mutation_strategy=mutation,
        initialization_strategy=init,
        crossover_strategy=crossover,
        callbacks={'on_evaluation': on_evaluation},
        num_processors=2)
    ga.run()