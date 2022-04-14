import numpy as np


class Memory:
    def __init__(self, state_size: tuple[int], memory_size: int, dtype=float):
        self.state_size = state_size
        self.memory_size = memory_size
        self.memory = np.zeros((memory_size,) + state_size, dtype=dtype)

        self.current_mem = 0
        self.total_writes = 0

    @property
    def used_mem(self):
        return min(self.total_writes, self.memory_size)

    def __getitem__(self, val):
        return self.memory[val]

    def __len__(self):
        return self.used_mem

    def extend(self, state):
        self.memory[self.current_mem] = state
        self.current_mem += 1
        self.total_writes += 1
        if self.current_mem == self.memory_size:
            self.current_mem -= self.memory_size

