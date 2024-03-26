"""CIFAR10 training and testing functions, local and federated."""

from collections.abc import Sized
from pathlib import Path
from typing import cast

import numpy as np

import json
import torch
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader

from project.task.default.train_test import get_fed_eval_fn as get_default_fed_eval_fn
from project.task.default.train_test import (
    get_on_evaluate_config_fn as get_default_on_evaluate_config_fn,
)
from project.task.default.train_test import (
    get_on_fit_config_fn as get_default_on_fit_config_fn,
)
from project.types.common import IsolatedRNG


class TrainConfig(BaseModel):
    """Training configuration, allows '.' member access and static checking.

    Guarantees that all necessary components are present, fails early if config is
    mismatched to client.
    """

    device: torch.device
    epochs: int
    learning_rate: float

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


def train(  # pylint: disable=too-many-arguments
    net: nn.Module,
    trainloader: DataLoader,
    _config: dict,
    _working_dir: Path,
    _rng_tuple: IsolatedRNG,
) -> tuple[int, dict]:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    _config : Dict
        The configuration for the training.
        Contains the device, number of epochs and learning rate.
        Static type checking is done by the TrainConfig class.
    _working_dir : Path
        The working directory for the training.
        Unused.
    _rng_tuple : IsolatedRNGTuple
        The random number generator state for the training.
        Use if you need seeded random behavior

    Returns
    -------
    Tuple[int, Dict]
        The number of samples used for training,
        the loss, and the accuracy of the input model on the given data.
    """
    if len(cast(Sized, trainloader.dataset)) == 0:
        raise ValueError(
            "Trainloader can't be 0, exiting...",
        )

    config: TrainConfig = TrainConfig(**_config)
    del _config

    net.to(config.device)
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=0.001,
    )

    final_epoch_per_sample_loss = 0.0
    num_correct = 0
    for _ in range(config.epochs):
        final_epoch_per_sample_loss = 0.0
        num_correct = 0
        for data, target in trainloader:
            data, target = (
                data.to(
                    config.device,
                ),
                target.to(config.device),
            )
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            final_epoch_per_sample_loss += loss.item()
            num_correct += (output.max(1)[1] == target).clone().detach().sum().item()
            loss.backward()
            optimizer.step()

    return len(cast(Sized, trainloader.dataset)), {
        "train_loss": final_epoch_per_sample_loss
        / len(cast(Sized, trainloader.dataset)),
        "train_accuracy": float(num_correct) / len(cast(Sized, trainloader.dataset)),
    }


class TestConfig(BaseModel):
    """Testing configuration, allows '.' member access and static checking.

    Guarantees that all necessary components are present, fails early if config is
    mismatched to client.
    """

    device: torch.device

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


def test(
    net: nn.Module,
    testloader: DataLoader,
    _config: dict,
    _working_dir: Path,
    _rng_tuple: IsolatedRNG,
) -> tuple[float, int, dict]:
    """Evaluate the network on the test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    _config : Dict
        The configuration for the testing.
        Contains the device.
        Static type checking is done by the TestConfig class.
    _working_dir : Path
        The working directory for the training.
        Unused.
    _rng_tuple : IsolatedRNGTuple
        The random number generator state for the training.
        Use if you need seeded random behavior

    Returns
    -------
    Tuple[float, int, float]
        The loss, number of test samples,
        and the accuracy of the input model on the given data.
    """
    if len(cast(Sized, testloader.dataset)) == 0:
        raise ValueError(
            "Testloader can't be 0, exiting...",
        )

    config: TestConfig = TestConfig(**_config)
    del _config

    net.to(config.device)
    net.eval()

    criterion = nn.CrossEntropyLoss()
    correct, per_sample_loss = 0, 0.0

    num_classes = 10

    confusion_matrix = np.zeros((num_classes, num_classes))

    batch_size = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = (
                images.to(
                    config.device,
                ),
                labels.to(config.device),
            )
            outputs = net(images)
            per_sample_loss += criterion(
                outputs,
                labels,
            ).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            for idx in range(len(labels)):
                confusion_matrix[predicted[idx], labels[idx]] += 1

            batch_size = len(images)

    # check whether client validation or centralised testing...
    # the centralised test set has 10000 observations, whereas
    # clients should have 50 observations (for 100 clients)
    #
    # if centralised testing, save the current model activations
    if 10000 // batch_size >= len(testloader):
        current_round = 0
        fname = "/nfs-share/lp647/L361/l361-project/project/current_round.json"
        with open(fname, encoding="utf-8") as f:
            current_round = json.load(f)["current_round"]

        torch.manual_seed(1337)
        global_activations_dir = "/nfs-share/lp647/L361/l361-project/outputs_custom\
            /cifar10_cnn_alpha0p1/results/state/global_activations"

        # get the activations after completing local training for the round
        rand_img = torch.rand((1, 3, 32, 32)).to(config.device)
        activations = net.get_activations(rand_img)
        activations_path = Path(global_activations_dir)
        activations_path.mkdir(parents=True, exist_ok=True)
        activations_file = activations_path / f"round_{current_round}.json"

        activations_dict = {
            "activations": activations,
        }

        with open(activations_file, "w", encoding="utf-8") as f:
            json.dump(activations_dict, f)

    # normalise confusion matrix
    confusion_matrix /= np.sum(confusion_matrix, axis=None)
    # make the confusion matrix json serializable
    confusion_matrix = confusion_matrix.tolist()

    return (
        per_sample_loss / len(cast(Sized, testloader.dataset)),
        len(cast(Sized, testloader.dataset)),
        {
            "test_accuracy": float(correct) / len(cast(Sized, testloader.dataset)),
            "test_confusion_matrix": confusion_matrix,
        },
    )


# Use defaults as they are completely determined
# by the other functions defined in cifar10_classification
get_fed_eval_fn = get_default_fed_eval_fn
get_on_fit_config_fn = get_default_on_fit_config_fn
get_on_evaluate_config_fn = get_default_on_evaluate_config_fn
