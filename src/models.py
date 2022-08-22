import torch
from torch import nn
from functools import reduce
import operator


class RLNetwork(nn.Module):
    """
    Class that describes a neural network used by a Reinforcement learning agent, subclass of torch.nn.Module
    """

    def __init__(self, input_shape: tuple, output_channels: int) -> None:
        """
        Constructor method of the network

        :param input_shape: the shape of the input that the network gets feed (tuple)
        :param output_channels: the number of output channels, or output dense units (int)

        :return: None
        """

        super(RLNetwork, self).__init__()

        self.input_shape = input_shape
        self.output_channels = output_channels
        self.backbone = nn.Sequential()
        self.dense = nn.Sequential()
        self.network = nn.Sequential()

    def _compute_in_features_linear(self) -> int:
        """
        Method to compute the number of input features that the fully-connected layer after the backbone (the
        convolutional layers) receive

        :return: returns the number of input features of the fully-connected layer (int)
        """

        # create a dummy tensor with the shape of the input data that the network receives
        dummy_tensor = torch.empty(size=self.input_shape)

        # feed the dummy tensor to the backbone of the network (the convolutional layers)
        dummy_tensor = self.backbone(dummy_tensor)

        # compute the flattened shape of the output of the backbone and return it
        flattened_tensor_dimension = reduce(operator.mul, dummy_tensor.shape)
        return flattened_tensor_dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method to forward an input tensor through the network

        :param x: the tensor to feed to the network (torch.Tensor)

        :return: the tensor outputted by the network (torch.Tensor)
        """
        x = self.network(x)
        return x


class DQNNetwork(RLNetwork):
    """
    Class that describes the Neural Network described in the 2015 Nature DQN paper
    """

    def __init__(self, input_shape: tuple, output_channels: int) -> None:
        """
        Constructor method of the network

        :param input_shape: the shape of the input that the network gets feed (tuple)
        :param output_channels: the number of output channels, or output dense units (int)

        :return: None
        """
        super(DQNNetwork, self).__init__(input_shape, output_channels)

        # define the backbone of the network
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=self.input_shape[0], out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU()
        )

        # define the dense layers of the network
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_features=512, in_features=super(DQNNetwork, self)._compute_in_features_linear()),
            nn.ReLU(),
            nn.Linear(out_features=self.output_channels, in_features=512)
        )

        # define the network as a sequence of backbone and dense layers
        self.network = nn.Sequential(
            self.backbone,
            self.dense
        )
