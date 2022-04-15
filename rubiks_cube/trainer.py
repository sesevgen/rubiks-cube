import copy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from rubiks_cube.agent import Agent
from rubiks_cube.cube import Cube, permute_state_colors
from rubiks_cube.memory import Memory
from rubiks_cube.fcdqn import FcDQN

import operator

operator_dict = {
    "-": operator.sub,
    "*": operator.mul,
}


class Trainer:
    def __init__(self,
                 agent: Agent,
                 env: Cube,
                 memsize: int = 10000,
                 batch_size: int = 64,
                 optim_config: dict = {"Adam": {"lr": 0.01}},
                 loss_config: dict = {"HuberLoss": {}},
                 ):

        self.agent = agent
        self.model = agent.model  # Alias for ease
        self.target_model = copy.deepcopy(self.model)
        self.env = env

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

    def make_moves(self, steps, shuffle, reset_period, eps, eps_min, eps_decay_fcn, eps_decay_amt, reset_env=False,
                   verbose=False):
        if reset_env:
            self.reset_environment(shuffle)

        reset_counter = 0
        for s in range(steps + 1):
            self.state_memory.extend(self.env.sides)
            action = self.agent.choose_action(self.env.sides, eps)
            self.action_memory.extend(action)
            self.env.move(*action)
            if self.env.solved:
                if verbose:
                    print(f"Solved on step {s}! Resetting environment to {shuffle} shuffles.")
                self.reward_memory.extend(10.0)
                self.terminal_memory.extend(1)
                self.reset_environment(shuffle)
                reset_counter = 0
            else:
                self.reward_memory.extend(0.0)
            self.f_state_memory.extend(self.env.sides)
            eps = max(eps_decay_fcn(eps, eps_decay_amt), eps_min)

            reset_counter += 1
            if reset_counter == reset_period:
                self.reset_environment(shuffle)
                if verbose:
                    print("Resetting environment.")

        if verbose:
            print(f"Took {s} steps.")

        return eps

    def model_step(self, gamma, verbose=False):
        self.model.train()

        replay_size = self.state_memory.used_mem
        if replay_size < self.batch_size:
            if verbose:
                print(f"Not enough recorded memory! {self.batch_size}, {replay_size}.")
            return

        random_sample = np.random.choice(replay_size, self.batch_size, replace=False)

        # TODO: Going with inefficient batch creation for now
        # Need to add tests and verify onehot conversion functions with a batch dim.
        states = torch.zeros(self.batch_size, self.model.input_dim)
        next_states = torch.zeros(self.batch_size, self.model.input_dim)
        for i, s in enumerate(random_sample):
            # Data augment with color swaps
            # TODO: Probably doesnt need to be the same for state and f_state actually?
            rand_permute_idx = np.random.randint(0, 720)

            # states[i, :] = self.model.cube_state_to_onehot(permute_state_colors(self.state_memory[s], rand_permute_idx))
            # next_states[i, :] = self.model.cube_state_to_onehot(
            #     permute_state_colors(self.f_state_memory[s], rand_permute_idx))

            states[i, :] = self.model.cube_state_to_onehot(self.state_memory[s])
            next_states[i, :] = self.model.cube_state_to_onehot(self.f_state_memory[s])

        rewards = torch.from_numpy(self.reward_memory[random_sample][:, 0])
        terminals = self.terminal_memory[random_sample][:, 0]
        actions = self.action_memory[random_sample]
        action_idx = np.array([self.model.action_to_int(a) for a in actions])

        q_current = self.model.forward(torch.Tensor(states).to(self.model.device))
        q_current = q_current[range(self.batch_size), action_idx]

        q_next = self.target_model.forward(torch.Tensor(next_states).to(self.model.device))
        # q_next[terminals, :] = 0.0
        q_next = torch.max(q_next, dim=1)[0]

        q_target = rewards + gamma * q_next
        loss = self.loss_function(q_target.float(), q_current.float()).to(self.model.device)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self,
              max_epochs=10000,
              success_ratio_cutoff: float = 0.9,
              moves_per_step: int = 1,
              gamma: float = 0.9,
              steps_per_bm: int = 256,
              eps: float = 1.0,
              eps_min: float = 0.05,
              eps_decay_fcn=operator.sub,
              eps_decay_amt: float = 0.0002,
              print_interval=1000,
              target_network_update_freq=10,
              ):
        """
        1) Start with shuffle = 1
        2) Take steps and record actions, resetting cube at every solve state
        3) After N steps, benchmark.
        4) If solved fraction > cutoff, increase shuffle by 1 and reset eps.
        5) Repeat 2-4 until shuffle is large.

        Notes:
            Trying dynamic epsilon

        """
        losses = []
        shuffle = 1

        self.model.train()
        for e in range(max_epochs):
            if e % target_network_update_freq:
                self.target_model.load_state_dict(self.model.state_dict())

            # Trying dynamic eps
            # try:
            #     eps = success_rate
            # except:
            #     pass

            eps = self.make_moves(moves_per_step, shuffle, 10 * shuffle, eps, eps_min, eps_decay_fcn, eps_decay_amt,
                                  reset_env=True)
            loss = self.model_step(gamma)
            if e % steps_per_bm == 0:
                bm_dict, _ = self.agent.benchmark(max_shuffles=shuffle)
                success_rate = bm_dict[shuffle]["solve_fraction"]
                if success_rate > success_ratio_cutoff:
                    print(f"Increasing shuffle at epoch {e} after {success_rate} success rate.")
                    shuffle += 1
                    eps = 1.0

            losses.append(loss)

            if ((e + 1) % print_interval == 0) or (e == 0):
                if loss is None:
                    loss = 0.0
                print(f"Epoch {e + 1} - Loss: {loss:.3f} | Latest success rate: {success_rate:.3f} | shuffle: {shuffle} | "
                      f"eps: {eps:.3f}")

        return losses


if __name__ == "__main__":
    cube = Cube(2)
    model = FcDQN(2, [256, 64])
    agent = Agent(model)

    bm_d, r_d = agent.benchmark()
    plt.plot(bm_d.keys(), [v["solve_fraction"] for v in bm_d.values()], label="Untrained")
    bm_d, r_d = agent.benchmark(eps=1.0)
    plt.plot(bm_d.keys(), [v["solve_fraction"] for v in bm_d.values()], label="Random")

    trainer = Trainer(agent, cube)
    trainer.train(max_epochs=500000, print_interval=200)
    bm_d, r_d = agent.benchmark()
    plt.plot(bm_d.keys(), [v["solve_fraction"] for v in bm_d.values()], label="Trained")
    plt.legend()
    plt.xlabel("Max steps away from solution")
    plt.ylabel("Fraction solved")
    plt.savefig("perf.pdf")

    # for i in range(4):
    #     for j in range(10):
    #         cube = Cube(2)
    #         while cube.solved:
    #             cube.make_random_moves(i+1)
    #         agent.attempt_solution(cube)
    #
    # trainer = Trainer(agent, cube)
    # losses = []
    # cube.make_random_moves(1)
    # for i in range(1000):
    #     trainer.make_moves(100, int(i/1000)+1, reset_period=100)
    #     loss = trainer.train()
    #     losses.append(loss)
    #
    # plt.plot(losses)
    # plt.show()
    #
    # for i in range(4):
    #     for j in range(10):
    #         cube = Cube(2)
    #         while cube.solved:
    #             cube.make_random_moves(i+1)
    #         agent.attempt_solution(cube)
