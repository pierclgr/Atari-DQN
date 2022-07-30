import torch
from torch import nn
from functools import reduce
import operator


# TODO DOCUMENTATION
class RLNetwork(nn.Module):
    def __init__(self, observation_space_shape: torch.Size, action_space_size: int) -> None:
        super(RLNetwork, self).__init__()
        self.observation_space_shape = observation_space_shape
        self.action_space_size = action_space_size

        self.network = nn.Sequential()

    def _compute_in_features_linear(self, input_shape: tuple) -> int:
        dummy_tensor = torch.empty(size=input_shape)
        dummy_tensor = self.backbone(dummy_tensor)

        flattened_tensor_dimension = reduce(operator.mul, dummy_tensor.size())

        return flattened_tensor_dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        return x


class ToyNetwork(RLNetwork):
    def __init__(self, input_shape: torch.Size, output_channels: int) -> None:
        super(ToyNetwork, self).__init__(input_shape, output_channels)

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=256, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.2)
        )

        self.dense = nn.Sequential(
            nn.Flatten(start_dim=0),
            nn.Linear(out_features=64, in_features=super(ToyNetwork, self)._compute_in_features_linear(input_shape)),
            nn.Linear(out_features=output_channels, in_features=64)
        )

        self.network = nn.Sequential(
            self.backbone,
            self.dense
        )


model = ToyNetwork(torch.Size((3, 210, 160)), 18)
dummy_tensor = torch.empty(size=(3, 210, 160))
print(dummy_tensor.size())
dummy_tensor = model.forward(dummy_tensor)
print(dummy_tensor.size())
