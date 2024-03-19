"""Define our models, and training and eval functions for CIFAR10 classification."""

import torch
import torch.nn.functional as F
from torch import nn

from project.types.common import IsolatedRNG, NetGen
from project.utils.utils import lazy_config_wrapper

class Net(nn.Module):
    """Convolutional Neural Network architecture.

    As described in McMahan 2017 paper :

    [Communication-Efficient Learning of Deep Networks from
    Decentralized Data] (https://arxiv.org/pdf/1602.05629.pdf)
    """

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize the network.

        Parameters
        ----------
        num_classes : int
            Number of classes in the dataset.

        Returns
        -------
        None
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.pool = nn.MaxPool2d(
            kernel_size=(2, 2),
            padding=1,
        )
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the CNN.

        Parameters
        ----------
        x : torch.Tensor
            Input Tensor that will pass through the network

        Returns
        -------
        torch.Tensor
            The resulting Tensor after it has passed through the network
        """
        output_tensor = F.relu(self.conv1(input_tensor))
        output_tensor = self.pool(output_tensor)
        output_tensor = F.relu(self.conv2(output_tensor))
        output_tensor = self.pool(output_tensor)
        output_tensor = torch.flatten(output_tensor, 1)
        output_tensor = F.relu(self.fc1(output_tensor))
        output_tensor = self.fc2(output_tensor)
        return output_tensor


# Simple wrapper to match the NetGenerator Interface
get_net: NetGen = lazy_config_wrapper(Net)


def get_net(_config: dict, rng_tuple: IsolatedRNG) -> nn.Module:
    """Return a model instance.

    Args:
    config: A dictionary with the model configuration.
    rng_tuple: The random number generator state for the training.
        Use if you need seeded random behavior

    Returns
    -------
    nn.Module
        A PyTorch model.
    """
    return Net()

# TODO: add Vision Transformer model as well
