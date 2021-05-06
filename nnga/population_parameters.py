# Copyright (c) [2021] Alessio Russo [alessior@kth.se]. All rights reserved.
# This file is part of nnGA.
# nnGA is free software: you can redistribute it and/or modify
# it under the terms of the MIT License. You should have received a copy of
# the MIT License along with nnGA.
# If not, see <https://opensource.org/licenses/MIT>.
#
# Code author: [Alessio Russo - alessior@kth.se]


class PopulationParameters(self):
    def __init__(self,
                 population_size: int,
                 elite_fraction: float = 0.1,
                 offsprings_from_elite_leader_fraction: float = 0.1,
                 offsprings_from_elite_group_fraction: float = 0.2,
                 crossover_fraction: float = 0.5,
                 rnd_offsprings_fraction: float = 0.1,
                 crossover_mutation_probability: float = 0.1,
                ):
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.offsprings_from_elite_leader_fraction = offsprings_from_elite_leader_fraction
        self.offsprings_from_elite_group_fraction = offsprings_from_elite_group_fraction
        self.crossover_fraction = crossover_fraction
        self.rnd_offsprings_fraction = rnd_offsprings_fraction
        self.crossover_mutation_probability = crossover_mutation_probability
        self.elite_size = int(
            np.ceil(self.population_size * self.elite_fraction))

        if not np.isclose(elite_fraction + offsprings_from_elite_leader + crossover_fraction \
                + rnd_offsprings_fraction + offsprings_from_elite_group, 1.):
            raise ValueError(
                'You have not provided a correct proportion for the elite/offsprings (it should sum to 1)'
            )

        @property
        def size(self):
            return self.population_size

        @property
        def elite_size(self):
            return self.elite_size
        
        def __len__(self):
            return self.population_size

        @property
        def offsprings_from_elite_leader(self):
            return int(self.size * self.offsprings_from_elite_leader_fraction)

        @property
        def offsprings_from_elite_group(self):
            return int(self.size * self.offsprings_from_elite_group_fraction)

        @property
        def random_offsprings(self):
            return int(self.size * self.rnd_offsprings_fraction)
