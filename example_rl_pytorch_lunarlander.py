# Copyright (c) [2021] Alessio Russo [alessior@kth.se]. All rights reserved.
# This file is part of nnGA.
# nnGA is free software: you can redistribute it and/or modify
# it under the terms of the MIT License. You should have received a copy of
# the MIT License along with nnGA.
# If not, see <https://opensource.org/licenses/MIT>.
#
# Code author: [Alessio Russo - alessior@kth.se]
#

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from nnga import nnGA

# Example Reinforcement Learning - LunarLander
# ------------
# In this example we see how to use Genetic Algorithms
# to solve the discrete LunarLander environment

import gym


def make_network(parameters=None):
    neural_network = torch.nn.Sequential(
        torch.nn.Linear(8, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4))

    if parameters:
        state_dict = neural_network.state_dict()
        for x, k in enumerate(state_dict.keys(), 0):
            state_dict[k] = torch.tensor(parameters[x])
        neural_network.load_state_dict(state_dict)
    return neural_network


def fitness(data: list):
    idx, parameters, episodes = data
    env = gym.make('LunarLander-v2')
    episode_reward_list = []
    network = make_network(parameters)

    avg_total_reward = 0.
    for n in range(episodes):
        done = False
        state = env.reset()
        total_episode_reward = 0.
        while not done:
            action = network(torch.tensor([state], dtype=torch.float32))
            action = torch.argmax(action).item()
            state, reward, done, _ = env.step(action)
            total_episode_reward += reward

        avg_total_reward = (n * avg_total_reward + total_episode_reward) / (
            n + 1)

        env.close()

    return avg_total_reward


def on_evaluation(epoch, fitnesses, population, best_result, best_network):
    print('[Epoch: {}] Best result: {:.3f}'.format(epoch, best_result))
    if best_result > 195:
        return True
    return False


if __name__ == '__main__':
    nn = make_network().state_dict()
    network_structure = [list(v.shape) for _, v in nn.items()]
    ga = nnGA(
        network_structure=network_structure,
        epochs=50,
        population_size=200,
        fitness_function=fitness,
        fitness_function_args=(10, ),
        exploration_noise=[1e-1] * len(network_structure),
        crossover_type='basic-crossover',
        callbacks={'on_evaluation': on_evaluation},
        num_processors=16)
    ga.run()