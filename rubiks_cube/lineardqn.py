import torch
import torch.nn as nn
import numpy as np


class LinearDQN(nn.Module):
    def __init__(
            self,
            n: int,
            hidden_layer_sizes: list[int],
            activations: str = "ReLU",
            device: str = None,
    ):

        super(LinearDQN, self).__init__()

        self.n = n
        self.input_dim = 6*n*n*6
        self.output_dim = n*3*2

        if activations is None:
            activations = "Identity"

        layers = [nn.Flatten]
        layer_sizes = [self.input_dim] + hidden_layer_sizes + [self.output_dim]
        for i, o in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(i, o))
            layers.append(getattr(nn, activations))
        # Drop the last activation
        layers = layers[:-1]
        self.layers = nn.Sequential(*layers)

        if device is None:
            self.device = torch.device('cuda:0' if T.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.to(self.device)
        print(self)

    def forward(self, x):
        x = self.layers(x)
        return x.view(-1, self.n, 3, 2)

    @staticmethod
    def cube_state_to_onehot(state):
        """
        Returns (6, n, n, 6) tensor from (6, n, n)
        """
        return onehot_initialization(state)

    @staticmethod
    def onehot_to_cube_state(onehot_state):
        """
        Returns (6, n, n) tensor from (6, n, n, 6)
        """
        return torch.argmax(onehot_state, -1)

    @staticmethod
    def action_to_onehot(action, action_space):
        """
        Returns (n*3*2) from (3)
        """
        oh = torch.ones(np.prod(action_space), dtype=torch.uint8)
        idx = (action[0] * np.prod(action_space[1:]) +
               action[1] * action_space[-1] +
               action[2])
        oh[idx] = 1

        return oh

    @staticmethod
    def onehot_to_action(onehot_action):
        """
        Returns (3) from (n*3*2)
        """
        idx = torch.argmax(onehot_action)
        axis = int(idx / (3*2))
        idx = idx % (3*2)
        depth = int(idx / 2)
        direction = int(idx % 2)

        return axis, depth. direction






# https://stackoverflow.com/a/36960495 @Divakar
def onehot_initialization(a):
    ncols = a.max()+1
    out = torch.zeros(a.shape + (ncols,), dtype=torch.uint8)
    out[all_idx(a, axis=3)] = 1
    return out

# https://stackoverflow.com/a/46103129/ @Divakar
def all_idx(idx, axis):
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)