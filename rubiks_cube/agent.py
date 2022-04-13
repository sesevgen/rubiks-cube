from typing import Union
import torch
import torch.nn as nn
import operator
import numpy as np

from rubiks_cube.lineardqn import LinearDQN

operator_dict = {
    "-": operator.sub,
    "*": operator.mul,
}


class Agent:
    def __init__(
            self,
            input_dim: int | tuple(int),
            output_dim: int | tuple(int),
            model: dict | nn.Module,
            epsilon=1,
            eps_decay_fcn='-',
            eps_decay_amt=1e-5,
    ):

        self.input_dim = input_dim
        self.output_dim = output_dim

        if isinstance(model, dict):
            model["input_dim"] = input_dim
            model["output_dim"] = output_dim
            self.model = LinearDQN(**model)
        else:
            assert model.input_dim == input_dim
            assert model.output_dim == output_dim
            self.model = model

        self.epsilon = epsilon
        self.eps_decay_fcn = operator_dict[eps_decay_fcn]
        self.eps_decay_amt = eps_decay_amt

    def choose_action(self, obs):
        if np.random.random() > self.epsilon:
            est_rewards = self.model(obs)
            optimal_action = torch.argmax(est_rewards)
            return optimal_action

        else:
            if isinstance(self.output_dim, int):
                return np.random.randint(0, self.output_dim)
            else:
                return tuple([np.random.randint(0, a) for a in self.output_dim])






