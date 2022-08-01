import torch
from torch import nn
from functools import reduce
import operator


# TODO DOCUMENTATION
class RLNetwork(nn.Module):
    def __init__(self, observation_space_shape: tuple, action_space_size: int) -> None:
        super(RLNetwork, self).__init__()
        # convert observation space size to a shape suitable for PyTorch (i.e. (channels, height, width)
        self.observation_space_shape = observation_space_shape
        self.action_space_size = action_space_size
        self.network = nn.Sequential()

    def _compute_in_features_linear(self) -> int:
        dummy_tensor = torch.empty(size=self.observation_space_shape)
        dummy_tensor = self.backbone(dummy_tensor)
        dummy_tensor = torch.flatten(dummy_tensor, start_dim=1)

        flattened_tensor_dimension = dummy_tensor.size()[1]

        return flattened_tensor_dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        return x


class DQNNetwork(RLNetwork):
    def __init__(self, input_shape: tuple, output_channels: int) -> None:
        super(DQNNetwork, self).__init__(input_shape, output_channels)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=self.observation_space_shape[1], out_channels=32, kernel_size=(8, 8), stride=(2, 2)),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU()
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_features=512, in_features=super(DQNNetwork, self)._compute_in_features_linear()),
            nn.Linear(out_features=self.action_space_size, in_features=512)
        )

        self.network = nn.Sequential(
            self.backbone,
            self.dense
        )


if __name__ == "__main__":
    x = torch.empty(size=(2, 4, 64, 64))

    model = DQNNetwork(x.shape, 4)
    x = model(x)
