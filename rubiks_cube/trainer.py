import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from rubiks_cube.agent import Agent
from rubiks_cube.cube import Cube
from rubiks_cube.memory import Memory
from rubiks_cube.lineardqn import LinearDQN

import operator

operator_dict = {
    "-": operator.sub,
    "*": operator.mul,
}


class Trainer:
    def __init__(self,
                 agent: Agent,
                 env: Cube,
                 gamma=0.9,
                 eps_start: float = 1.0,
                 eps_min: float = 0.01,
                 eps_decay_fcn: str = '-',
                 eps_decay_amt: float = 0.01,
                 memsize: int = 10000,
                 batch_size: int = 64,
                 optim_config: dict = {"Adam": {"lr": 0.01}},
                 loss_config: dict = {"HuberLoss": {}},
                 ):

        self.agent = agent
        self.model = agent.model  # Alias for ease
        self.env = env

        self.gamma = gamma

        self.epsilon = eps_start
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_decay_fcn = operator_dict[eps_decay_fcn]
        self.eps_decay_amt = eps_decay_amt

        self.batch_size = batch_size

        # Stored in cube format, not onehot
        self.state_memory = Memory(self.env.sides.shape, memsize, dtype=np.ushort)
        self.f_state_memory = Memory(self.env.sides.shape, memsize, dtype=np.ushort)

        # Stored in action tuple, not int (but maybe should change?)
        self.action_memory = Memory((3,), memsize, dtype=np.ushort)

        self.reward_memory = Memory((1,), memsize)
        self.terminal_memory = Memory((1,), memsize, dtype=bool)

        for k, v in optim_config.items():
            self.optimizer = getattr(optim, k)(self.model.parameters(), **v)

        for k, v in loss_config.items():
            self.loss_function = getattr(nn, k)(**v)

        self.total_steps = 0

        self.model.train()

    def reset_environment(self, shuffles):
        self.env.reset()
        self.env.make_random_moves(shuffles)

    def make_moves(self, steps, shuffle, reset_period=0, reset_env=False, reset_eps=False):
        if reset_eps:
            self.epsilon = self.eps_start

        if reset_env:
            self.reset_environment(shuffle)

        reset_counter = 0
        for s in range(steps + 1):
            self.state_memory.extend(self.env.sides)
            action = self.agent.choose_action(self.env.sides, self.epsilon)
            self.action_memory.extend(action)
            self.env.move(*action)
            if self.env.solved:
                print(f"Solved on step {s}! Resetting environment to {shuffle} shuffles.")
                self.reward_memory.extend(1.0)
                self.terminal_memory.extend(1)
                self.reset_environment(shuffle)
                reset_counter = 0
            else:
                self.reward_memory.extend(0.0)
            self.f_state_memory.extend(self.env.sides)
            self.epsilon = self.eps_decay_fcn(self.epsilon, self.eps_decay_amt)

            reset_counter += 1
            if reset_counter == reset_period:
                self.reset_environment(shuffle)
                print("Resetting environment.")

        print(f"Took {s} steps.")

    def train(self):
        self.model.train()

        replay_size = self.state_memory.used_mem
        if replay_size < self.batch_size:
            print(f"Not enough recorded memory! {self.batch_size}, {replay_size}.")
            return

        self.optimizer.zero_grad()

        random_sample = np.random.choice(replay_size, self.batch_size, replace=False)

        # TODO: Going with inefficient batch creation for now
        # Need to add tests and verify onehot conversion functions with a batch dim.
        states = torch.zeros(self.batch_size, self.model.input_dim)
        next_states = torch.zeros(self.batch_size, self.model.input_dim)
        for i, s in enumerate(random_sample):
            states[i, :] = self.model.cube_state_to_onehot(self.state_memory[s])
            next_states[i, :] = self.model.cube_state_to_onehot(self.f_state_memory[s])

        rewards = torch.from_numpy(self.reward_memory[random_sample][:, 0])
        terminals = self.terminal_memory[random_sample][:, 0]
        actions = self.action_memory[random_sample]
        action_idx = np.array([self.model.action_to_int(a) for a in actions])

        q_current = self.model.forward(torch.Tensor(states).to(self.model.device))
        q_current = q_current[range(self.batch_size), action_idx]

        q_next = self.model.forward(torch.Tensor(next_states).to(self.model.device))
        q_next[terminals, :] = 0.0
        q_next = torch.max(q_next, dim=1)[0]

        q_target = rewards + self.gamma * q_next
        loss = self.loss_function(q_target.float(), q_current.float()).to(self.model.device)
        loss.backward()
        self.optimizer.step()

        return loss.item()


if __name__ == "__main__":
    cube = Cube(2)
    model = LinearDQN(2, [128, 128])
    agent = Agent(model)

    for i in range(4):
        for j in range(10):
            cube = Cube(2)
            while cube.solved:
                cube.make_random_moves(i+1)
            agent.attempt_solution(cube)

    trainer = Trainer(agent, cube)
    losses = []
    cube.make_random_moves(1)
    for i in range(1000):
        trainer.make_moves(100, int(i/1000)+1, reset_period=100)
        loss = trainer.train()
        losses.append(loss)

    plt.plot(losses)
    plt.show()

    for i in range(4):
        for j in range(10):
            cube = Cube(2)
            while cube.solved:
                cube.make_random_moves(i+1)
            agent.attempt_solution(cube)
