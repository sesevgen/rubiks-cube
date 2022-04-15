from typing import Union
import torch
import torch.nn as nn
import operator
import numpy as np

from rubiks_cube.fcdqn import FcDQN
from rubiks_cube.cube import Cube, permute_state_colors


class Agent:
    def __init__(
            self,
            model: FcDQN,
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

    def attempt_solution(self, cube, max_steps=100, eps=0.0, verbose=False):
        self.model.eval()
        with torch.no_grad():
            steps = 0
            while (steps < max_steps) and (not cube.solved):
                action = self.choose_action(cube.sides, eps)
                cube.move(*action)
                steps += 1
            if cube.solved:
                if verbose:
                    print(f"Cube solved in {steps} steps!")
            else:
                if verbose:
                    print(f"Failed to solve cube in {steps} steps.")

        return cube.solved, steps, cube

    def benchmark(self,
                  attempts_per_shuffle=256,
                  max_shuffles=8,
                  max_steps=128,
                  eps=0.0):

        n = self.model.n

        runs_dict = {}
        benchmark_dict = {}
        for s in range(1, max_shuffles+1):
            runs_dict[s] = {"solved": [],
                            "steps": [],
                            }
            benchmark_dict[s] = {
                "solve_fraction": None,
            }
            for a in range(attempts_per_shuffle):
                cube = Cube(n)
                cube.make_random_moves(s)
                # cube.sides = permute_state_colors(cube.sides, np.random.randint(0, 720))

                solved, steps, _ = self.attempt_solution(cube, max_steps=max_steps, eps=eps)
                runs_dict[s]["solved"].append(solved)
                runs_dict[s]["steps"].append(steps)

            benchmark_dict[s]["solve_fraction"] = sum(runs_dict[s]["solved"]) / len(runs_dict[s]["solved"])

        return benchmark_dict, runs_dict
