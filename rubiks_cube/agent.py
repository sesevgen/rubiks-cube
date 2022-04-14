from typing import Union
import torch
import torch.nn as nn
import operator
import numpy as np

from rubiks_cube.lineardqn import LinearDQN


class Agent:
    def __init__(
            self,
            model: LinearDQN,
    ):

        self.model = model

    def choose_action(self, obs, eps):
        if np.random.random() > eps:
            obs_oh = self.model.cube_state_to_onehot(obs)
            est_rewards = self.model(obs_oh.to(self.model.device))
            action = torch.argmax(est_rewards)
        else:
            action = np.random.randint(0, self.model.output_dim)

        return self.model.int_to_action(action)

    def attempt_solution(self, cube, max_steps=100, eps=0.0):
        self.model.eval()
        with torch.no_grad():
            steps = 0
            while (steps < max_steps) and (not cube.solved):
                action = self.choose_action(cube.sides, eps)
                cube.move(*action)
                steps += 1
            if cube.solved:
                print(f"Cube solved in {steps} steps!")
            else:
                print(f"Failed to solve cube in {steps} steps.")
