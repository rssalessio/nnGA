# nnGA Library - Neural Network Genetic Algorithm Library (v0.3)

Off the shelf Genetic Algorithm library for deep learning problems

_Author_: Alessio Russo (PhD Student at KTH - alessior@kth.se)

## License
Our code is released under the MIT license (refer to the [LICENSE](https://github.com/rssalessio/PoisoningDataDrivenControl/blob/master/LICENSE) file for details).

## Requirements
To use the library you need atleast Python 3.5. Examples may require additional libraries.

Other required dependencies:
- NumPy

## Usage/Examples
You can import the library by typing ```python from nnga import nnGA```.

To learn how to use nnGA, check the following examples:

1. Reinforcement learning examples
    * [Cartpole example](https://github.com/rssalessio/nnGA/blob/master/examples/example_rl_pytorch_cartpole.py)
    * [Lunarlander example](https://github.com/rssalessio/nnGA/blob/master/examples/example_rl_pytorch_lunarlander.py)

2. Supervised learning examples
    * [Multiclass classification](https://github.com/rssalessio/nnGA/blob/master/examples/example_rl_pytorch_lunarlander.py)

In general the code has the following structure
```python
from nnga import nnGA, GaussianInitializationStrategy, \
    GaussianMutationStrategy, BasicCrossoverStrategy, \
    PopulationParameters

def make_network(parameters=None):
    ''' Function that creates a network given a set of parameters '''
    neural_network = ...
    return neural_network


def fitness(idx, parameters):
    ''' Fitness function to evaluate a set of parameters '''
    # Evaluate parameters
    network = make_network(parameters)
    return evaluate_network(network)


if __name__ == '__main__':
    # Initialize GA parameters
    network = make_initial_network()
    network_structure = [list(layer.shape) for layer in network]  # List of tuples, containing the shape of each layer
    
    # Population parameters
    population = PopulationParameters(population_size=200)
    
    # Mutation strategy
    mutation = GaussianMutationStrategy(network_structure, 1e-1)
    
    # Crossover strategy
    crossover = BasicCrossoverStrategy(network_structure)
    
    # Initialization strategy
    init = GaussianInitializationStrategy(
        mean=0., std=1., network_structure=network_structure)

    ga = nnGA(
        epochs=50,  # Number of epochs
        fitness_function=fitness,
        population_parameters=population,
        mutation_strategy=mutation,
        initialization_strategy=init,
        crossover_strategy=crossover,
        num_processors=8)  # Number of cores

    # Run GA
    network_parameters, best_result, results = ga.run()

```

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
