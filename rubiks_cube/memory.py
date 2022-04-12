import numpy as np


class Memory:
    def __init__(self, state_size: tuple(int), memory_size: int):
        self.state_size = state_size
        self.memory_size = memory_size
        self.memory = np.zeros((memory_size,) + state_size)

        self.used_mem = 0
        self.current_mem = 0

    def __getitem__(self, val):
        return self.memory[val]

    def __len__(self):
        return self.used_mem

    def extend(self, state):
        self.memory[self.current_mem] = state
        self.current_mem += 1

