"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your code is to first
download the dataset and partition it and then run the experiments, please uncomment the
lines below and tell us in the README.md (see the "Running the Experiment" block) that
this file should be executed first.
"""

import numpy as np
import shutil
import logging
from pathlib import Path
from typing import Any
from collections.abc import Callable
from PIL import Image

import hydra
from omegaconf import DictConfig, OmegaConf
from flwr.common.logger import log


import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, VisionDataset

from project.task.cifar10_classification.dataset_utils import create_lda_partitions

ZERO = 0.0


class TorchVisionFL(VisionDataset):
    """A trimmed down version of torchvision.datasets.MNIST.

    Use this class by either passing a path to a torch file (.pt)
    containing (data, targets) or pass the data, targets directly
    instead.
    """

    def __init__(
        self,
        path_to_data: Path,
        data: Any = None,
        targets: Any = None,
        transform: Callable | None = None,
    ) -> None:
        path = path_to_data.parent if path_to_data else None
        super().__init__(path, transform=transform)
        self.transform = transform

        if path_to_data:
            # load data and targets (path_to_data points to an specific .pt file)
            self.data, self.targets = torch.load(path_to_data)
        else:
            self.data = data
            self.targets = targets

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Index a sample and label from the dataset."""
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if not isinstance(img, Image.Image):  # if not PIL image
            if not isinstance(img, np.ndarray):  # if torch tensor
                img = img.numpy()

            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.data)


def get_dataloader(
    path_to_data: str | Path,
    cid: str | int | Path,
    is_train: bool,
    batch_size: int,
    generator: Any | None,
) -> DataLoader:
    """Generate dataloader for a given client."""
    dataset = _get_dataset(Path(path_to_data), str(cid), is_train)

    # we use as number of workers all the cpu cores assigned to this actor
    kwargs = {"pin_memory": True, "drop_last": False}  # "num_workers": workers,
    return DataLoader(dataset, batch_size=batch_size, generator=generator, **kwargs)


def _get_dataset(path_to_data: Path, cid: str, is_train: bool) -> TorchVisionFL:
    """Generate Pytorch CIFAR10 object for a specific client's trainset/valset."""
    partition = "train" if is_train else "val"

    # path to client's data (depends on cid)
    path_to_data = path_to_data / cid / (partition + ".pt")

    return TorchVisionFL(path_to_data, transform=_get_cifar10_transform())


def _get_cifar10_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def _get_random_id_splits(
    total: int, val_ratio: float, shuffle: bool = True
) -> tuple[list, list]:
    """Split a list into train and validation sets.

    Split a list of length `total` into two following a \
    (1-val_ratio):val_ratio partitioning.

    By default the indices are shuffled before creating the split and
    returning.

    Parameters
    ----------
    total : int
        The length of the full (unsplit) list.
    val_ratio : float
        The portion of elements assigned to the validation split.
    shuffle : bool (default=True)
        Whether or not to shuffle the elements in the list prior to splitting.

    Returns
    -------
    train_idxs, val_idxs : tuple[list, list]
        The training and validation indices to be used for indexing the target dataset.
    """
    indices = list(range(total)) if isinstance(total, int) else total

    split = int(np.floor(val_ratio * len(indices)))

    if shuffle:
        np.random.shuffle(indices)
    return indices[split:], indices[:split]


def _download_cifar10_data(path_to_data: str | Path) -> tuple[str | Path, CIFAR10]:
    """Download CIFAR10 dataset.

    Generates a unified training set to be partitioned later using LDA.
    """
    # download dataset and load train set
    train_set = CIFAR10(root=path_to_data, train=True, download=True)

    # fuse all data splits into a single "training.pt"
    data_loc = Path(path_to_data) / "cifar-10-batches-py"
    training_data = data_loc / "training.pt"
    # print("Generating unified CIFAR dataset")
    torch.save([train_set.data, np.array(train_set.targets)], training_data)

    test_set = CIFAR10(
        root=path_to_data, train=False, transform=_get_cifar10_transform()
    )

    # return path of unified training set and the test set object
    return training_data, test_set


def _partition_data_lda(
    dataset_dir: str | Path,
    partition_dir: str | Path,
    num_partitions: int,
    alpha: float = 1000,
    num_classes: int = 10,
    val_ratio: float = 0.0,
    seed: int = 42,
) -> str | Path:  # TODO: define the return type
    """Partition torchvision datasets (e.g. CIFAR10) using LDA.

    Parameters
    ----------
    dataset_dir : Union[str, Path]
        The path to the centralised dataset.
    partition_dir : Union[str, Path]
        The target path for the generated client dataset partitions.
    num_clients : int
        The number of clients each holding a part of the data.
    seed : int
        Used to set a fix seed to replicate experiments, by default 42.
    iid : bool
        Whether the data should be independent and identically distributed between
        the clients (true) or if the data should be partitioned using LDA (false).
    alpha : float
        The degree of non-iidness for the LDA partitioning.

    """
    # ensure dataset_dir is of type Path, not str
    if isinstance(dataset_dir, str):
        dataset_dir = Path(dataset_dir)

    images, labels = torch.load(dataset_dir / "cifar-10-batches-py/training.pt")

    idx = np.array(range(len(images)))
    dataset = [idx, labels]

    # get the indices of the items for each partition using LDA
    partitions, _ = create_lda_partitions(
        dataset,
        num_partitions=num_partitions,
        concentration=alpha,
        accept_imbalanced=True,
    )

    # Show label distribution for first partition (purely informative)
    # partition_zero = partitions[0][1]
    # hist, _ = np.histogram(partition_zero, bins=list(range(num_classes + 1)))
    # print(
    #       f"Class histogram for 0-th partition (alpha={alpha},
    #       {num_classes} classes): {hist}"
    # )

    # now save partitioned dataset to disk
    # first delete dir containing splits (if exists), then create it
    splits_dir = Path(partition_dir)
    if splits_dir.exists():
        # print("Deleting existing partition directory")
        shutil.rmtree(splits_dir)

    # print(f"Creating new partition directory at {splits_dir}")
    Path.mkdir(splits_dir, parents=True)

    for p in range(num_partitions):

        labels = partitions[p][1]
        image_idx = partitions[p][0]
        imgs = images[image_idx]

        # create dir
        Path.mkdir(splits_dir / str(p))

        if val_ratio > ZERO:
            # split data according to val_ratio
            train_idx, val_idx = _get_random_id_splits(len(labels), val_ratio)
            val_imgs = imgs[val_idx]
            val_labels = labels[val_idx]

            with open(splits_dir / str(p) / "val.pt", "wb") as f:
                torch.save([val_imgs, val_labels], f)

            # remaining images for training
            imgs = imgs[train_idx]
            labels = labels[train_idx]

        with open(splits_dir / str(p) / "train.pt", "wb") as f:
            torch.save([imgs, labels], f)

    return partition_dir


@hydra.main(config_path="../../conf", config_name="cifar10", version_base=None)
def download_and_preprocess(cfg: DictConfig) -> None:
    """Do everything needed to get the dataset.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. print parsed config
    log(logging.INFO, OmegaConf.to_yaml(cfg))

    # Please include here all the logic
    # Please use the Hydra config style as much as possible especially
    # for parts that can be customised (e.g. how data is partitioned)

    # 2. Download the cifar-10 dataset
    # print("Downloading cifar10 dataset...")
    _, test_set = _download_cifar10_data(Path(cfg.dataset.dataset_dir))
    # print("Done!")

    # 3. Partition the dataset
    # _partition_data_lda(*) creates and populates the partition directory
    # print("\nPartitioning...")
    _partition_data_lda(
        dataset_dir=cfg.dataset.dataset_dir,
        partition_dir=cfg.dataset.partition_dir,
        num_partitions=cfg.dataset.num_clients,
        alpha=cfg.dataset.alpha,
        num_classes=cfg.dataset.num_classes,
        val_ratio=cfg.dataset.val_ratio,
        seed=cfg.dataset.seed,
    )

    # save the centralised test set in partition directory
    partition_dir = Path(cfg.dataset.partition_dir)
    torch.save(test_set, partition_dir / "test.pt")

    # print("Preprocessing finished.")


if __name__ == "__main__":

    download_and_preprocess()
