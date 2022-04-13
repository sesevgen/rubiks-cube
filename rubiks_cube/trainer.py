import torch.nn as nn
import torch.optim as optim

from rubiks_cube.agent import Agent
from rubiks_cube.cube import Cube
from rubiks_cube.memory import Memory


class Trainer:
    def __init__(self,
                 agent: Agent,
                 env: Cube,
                 memsize: int = 10000,
                 optim_config: dict = {"Adam": {"lr": 0.01}},
                 loss_config: dict = {"MSELoss": {}},
                 ):

        for k, v in optim_config.items():
            self.optimizer = getattr(optim, k)(self.model.parameters(), **v)

        for k, v in loss_config.items():
            self.loss = getattr(nn, k)(**v)

        self.agent = agent
        self.env = env

        self.state_memory = Memory(self.env.sides.shape, memsize)
        self.action_memory = Memory(self.env.action_space)

    def train(self, steps):


