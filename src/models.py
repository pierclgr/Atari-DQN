import torch
from torch import nn
from abc import ABC, abstractmethod


# TODO DOCUMENTATION
class ReinforcementLearningNN(ABC, nn.Module):
    @abstractmethod
    def __init__(self, action_space_size: int, observation_space_shape: tuple) -> None:
        super(ReinforcementLearningNN, self).__init__()
        self.observation_space_shape = observation_space_shape
        self.action_space_size = action_space_size

    @abstractmethod
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        pass


class ToyNN(ReinforcementLearningNN):
    def __init__(self, output_channels: int, input_shape: tuple) -> None:
        super(ToyNN, self).__init__(output_channels, input_shape)

        tensor_shape = (input_shape[2], input_shape[0], input_shape[1])

        self.conv2d1 = nn.Conv2d(in_channels=tensor_shape[0], out_channels=256, kernel_size=(3, 3))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.drop1 = nn.Dropout(0.2)

        tensor_shape = (256, (tensor_shape[1] - 2) / 2, (tensor_shape[2] - 2) / 2)

        self.conv2d2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.drop2 = nn.Dropout(0.2)

        tensor_shape = (256, round((tensor_shape[1] - 2) / 2), round((tensor_shape[2] - 2) / 2))
        tensor_shape = tensor_shape[0] * tensor_shape[1] * tensor_shape[2]

        self.linear = nn.Sequential(
            nn.Flatten(start_dim=0),
            nn.Linear(out_features=64, in_features=tensor_shape),
            nn.Linear(out_features=output_channels, in_features=64)
        )

    def forward(self, x: torch.Tensor) -> None:
        x = self.conv2d1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.drop1(x)
        x = self.conv2d2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.drop2(x)
        x = self.linear(x)
        return x
