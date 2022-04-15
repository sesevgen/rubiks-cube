import torch
import torch.nn as nn
import numpy as np


class FcDQN(nn.Module):
    def __init__(
            self,
            n: int,
            hidden_layer_sizes: list[int],
            activations: str = "ReLU",
            device: str = None,
    ):

        super(FcDQN, self).__init__()

        self._n = n
        self.input_dim = 6 * n * n * 6
        self.output_dim = n * 3 * 2

        if activations is None:
            activations = "Identity"

        layers = []
        layer_sizes = [self.input_dim] + hidden_layer_sizes + [self.output_dim]
        for i, o in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(i, o))
            layers.append(getattr(nn, activations)())
        # Drop the last activation
        layers = layers[:-1]
        self.layers = nn.Sequential(*layers)

        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.to(self.device)
        print(self)

    @property
    def n(self):
        return self._n

    def forward(self, x):
        return self.layers(x.float())

    def cube_state_to_onehot(self, state):
        """
        Returns (6, n, n, 6) tensor from (6, n, n)
        """
        return onehot_initialization(state).view(-1)

    def onehot_to_cube_state(self, onehot_state):
        """
        Returns (6, n, n) tensor from (6, n, n, 6)
        """
        return torch.argmax(onehot_state.view(6, self.n, self.n, 6), -1)

    def action_to_int(self, action):
        """
        Returns (1) from (3)
        """
        return (action[0] * 3 * 2 +
                action[1] * 2 +
                action[2])

    def int_to_action(self, int_action):
        """
        Returns (3) from (1)
        """
        depth = int(int_action / (3 * 2))
        int_action = int_action % (3 * 2)
        axis = int(int_action / 2)
        direction = int(int_action % 2)

        return depth, axis, direction


# https://stackoverflow.com/a/36960495 @Divakar
def onehot_initialization(a):
    ncols = a.max() + 1
    out = torch.zeros(a.shape + (ncols,), dtype=torch.uint8)
    out[all_idx(a.astype(float), axis=3)] = 1
    return out


# https://stackoverflow.com/a/46103129/ @Divakar
def all_idx(idx, axis):
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)
