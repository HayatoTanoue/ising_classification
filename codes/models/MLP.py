import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_classes, input_size):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        self.classifier = nn.Sequential(
            nn.Linear(self.input_size * self.input_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 30),
            nn.ReLU(inplace=True),
            nn.Linear(30, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
