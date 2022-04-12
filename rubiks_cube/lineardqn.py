import torch
import torch.nn as nn
import torch.optim as optim


class LinearDQN(nn.Module):
    def __init__(
            self,
            input_dim: int | tuple(int),
            hidden_layer_sizes: list[int],
            output_dim: int | tuple(int),
            activations: str = "ReLU",
            device: str = None,
    ):

        super(LinearDQN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if activations is None:
            activations = "Identity"

        layers = []
        if isinstance(input_dim, tuple):
            input_dim = 1
            for i in input_dim:
                input_dim *= i
            layers.append(nn.Flatten)

        if isinstance(output_dim, tuple):
            output_dim = 1
            for i in output_dim:
                output_dim *= i

        layer_sizes = [input_dim] + hidden_layer_sizes + [output_dim]

        for i, o in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(i, o))
            layers.append(getattr(nn, activations))
        layers = layers[:-1]
        if isinstance(output_dim, tuple):
            layers.append(nn.Unflatten(1, self.output_dim))
        self.layers = nn.Sequential(*layers)

        if device is None:
            self.device = torch.device('cuda:0' if T.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.to(self.device)
        print(self)

    def forward(self, x):
        return self.layers(x)
