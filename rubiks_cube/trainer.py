import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optim_config: dict = {"Adam": {"lr": 0.01}},
                 loss_config: dict = {"MSELoss": {}},
                 ):

        self.model = model

        for k, v in optim_config.items():
            self.optimizer = getattr(optim, k)(self.model.parameters(), **v)

        for k, v in loss_config.items():
            self.loss = getattr(nn, k)(**v)
