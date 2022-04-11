import torch
import torch.nn as nn
import torch.optim as optim


class LinearDQN(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_layer_sizes: list[int],
            output_dim: int,
            activations: str = "ReLU",
            device: str = None,
    ):

        super(LinearDQN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if activations is None:
            activations = "Identity"

        layer_sizes = [input_dim] + hidden_layer_sizes + [output_dim]
        layers = []
        for i, o in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(i, o))
            layers.append(getattr(nn, activations))
        self.layers = nn.Sequential(*layers[:-1])

        if device is None:
            self.device = torch.device('cuda:0' if T.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.to(self.device)
        print(self)

    def forward(self, x):
        return self.layers(x)
