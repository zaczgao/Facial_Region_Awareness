'''
https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py
'''

from collections import Counter
import logging
from operator import itemgetter
import random

import os
import numpy as np
import math
from typing import TypeVar, Optional, Iterator, List, Union
from typing import Callable
import pandas as pd


import torch
from torch.utils.data import DistributedSampler
from torch.utils.data.distributed import Sampler, Dataset
import torch.distributed as dist
import torchvision

T_co = TypeVar('T_co', covariant=True)

LOGGER = logging.getLogger(__name__)


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.

    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.

        Args:
            index: index of the element in the dataset

        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class BalanceClassSampler(Sampler):
    """Allows you to create stratified sample on unbalanced classes.

    Args:
        labels: list of class label for each elem in the dataset
        mode: Strategy to balance classes.
            Must be one of [downsampling, upsampling]

    Python API examples:

    .. code-block:: python

        import os
        from torch import nn, optim
        from torch.utils.data import DataLoader
        from catalyst import dl
        from catalyst.data import ToTensor, BalanceClassSampler
        from catalyst.contrib.datasets import MNIST

        train_data = MNIST(os.getcwd(), train=True, download=True, transform=ToTensor())
        train_labels = train_data.targets.cpu().numpy().tolist()
        train_sampler = BalanceClassSampler(train_labels, mode=5000)
        valid_data = MNIST(os.getcwd(), train=False)

        loaders = {
            "train": DataLoader(train_data, sampler=train_sampler, batch_size=32),
            "valid": DataLoader(valid_data, batch_size=32),
        }

        model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.02)

        runner = dl.SupervisedRunner()
        # model training
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            num_epochs=1,
            logdir="./logs",
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
        )
    """

    def __init__(self, labels: List[int], mode: Union[str, int] = "downsampling"):
        """Sampler initialisation."""
        super().__init__(labels)

        labels = np.array(labels)
        samples_per_class = {label: (labels == label).sum() for label in set(labels)}

        self.lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)
        }

        if isinstance(mode, str):
            assert mode in ["downsampling", "upsampling"]

        if isinstance(mode, int) or mode == "upsampling":
            samples_per_class = (
                mode if isinstance(mode, int) else max(samples_per_class.values())
            )
        else:
            samples_per_class = min(samples_per_class.values())

        self.labels = labels
        self.samples_per_class = samples_per_class
        self.length = self.samples_per_class * len(set(labels))

    def __iter__(self) -> Iterator[int]:
        """
        Returns:
            iterator of indices of stratified sample
        """
        indices = []
        for key in sorted(self.lbl2idx):
            replace_flag = self.samples_per_class > len(self.lbl2idx[key])
            indices += np.random.choice(
                self.lbl2idx[key], self.samples_per_class, replace=replace_flag
            ).tolist()
        assert len(indices) == self.length
        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """
        Returns:
             length of result sample
        """
        return self.length


class BatchBalanceClassSampler(Sampler):
    """
    This kind of sampler can be used for both metric learning and classification task.

    BatchSampler with the given strategy for the C unique classes dataset:
    - Selection `num_classes` of C classes for each batch
    - Selection `num_samples` instances for each class in the batch
    The epoch ends after `num_batches`.
    So, the batch sise is `num_classes` * `num_samples`.

    One of the purposes of this sampler is to be used for
    forming triplets and pos/neg pairs inside the batch.
    To guarante existance of these pairs in the batch,
    `num_classes` and `num_samples` should be > 1. (1)

    This type of sampling can be found in the classical paper of Person Re-Id,
    where P (`num_classes`) equals 32 and K (`num_samples`) equals 4:
    `In Defense of the Triplet Loss for Person Re-Identification`_.

    Args:
        labels: list of classes labeles for each elem in the dataset
        num_classes: number of classes in a batch, should be > 1
        num_samples: number of instances of each class in a batch, should be > 1
        num_batches: number of batches in epoch
            (default = len(labels) // (num_classes * num_samples))

    .. _In Defense of the Triplet Loss for Person Re-Identification:
        https://arxiv.org/abs/1703.07737

    Python API examples:

    .. code-block:: python

        import os
        from torch import nn, optim
        from torch.utils.data import DataLoader
        from catalyst import dl
        from catalyst.data import ToTensor, BatchBalanceClassSampler
        from catalyst.contrib.datasets import MNIST

        train_data = MNIST(os.getcwd(), train=True, download=True)
        train_labels = train_data.targets.cpu().numpy().tolist()
        train_sampler = BatchBalanceClassSampler(
            train_labels, num_classes=10, num_samples=4)
        valid_data = MNIST(os.getcwd(), train=False)

        loaders = {
            "train": DataLoader(train_data, batch_sampler=train_sampler),
            "valid": DataLoader(valid_data, batch_size=32),
        }

        model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.02)

        runner = dl.SupervisedRunner()
        # model training
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            num_epochs=1,
            logdir="./logs",
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
        )
    """

    def __init__(
        self,
        labels: Union[List[int], np.ndarray],
        num_classes: int,
        num_samples: int,
        num_batches: int = None,
    ):
        """Sampler initialisation."""
        super().__init__(labels)
        classes = set(labels)

        assert isinstance(num_classes, int) and isinstance(num_samples, int)
        assert (1 < num_classes <= len(classes)) and (1 < num_samples)
        assert all(
            n > 1 for n in Counter(labels).values()
        ), "Each class shoud contain at least 2 instances to fit (1)"

        labels = np.array(labels)
        self._labels = list(set(labels.tolist()))
        self._num_classes = num_classes
        self._num_samples = num_samples
        self._batch_size = self._num_classes * self._num_samples
        self._num_batches = num_batches or len(labels) // self._batch_size
        self.lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)
        }

    @property
    def batch_size(self) -> int:
        """
        Returns:
            this value should be used in DataLoader as batch size
        """
        return self._batch_size

    @property
    def batches_in_epoch(self) -> int:
        """
        Returns:
            number of batches in an epoch
        """
        return self._num_batches

    def __len__(self) -> int:
        """
        Returns:
            number of samples in an epoch
        """
        return self._num_batches  # * self._batch_size

    def __iter__(self) -> Iterator[int]:
        """
        Returns:
            indeces for sampling dataset elems during an epoch
        """
        indices = []
        for _ in range(self._num_batches):
            batch_indices = []
            classes_for_batch = random.sample(self._labels, self._num_classes)
            while self._num_classes != len(set(classes_for_batch)):
                classes_for_batch = random.sample(self._labels, self._num_classes)
            for cls_id in classes_for_batch:
                replace_flag = self._num_samples > len(self.lbl2idx[cls_id])
                batch_indices += np.random.choice(
                    self.lbl2idx[cls_id], self._num_samples, replace=replace_flag
                ).tolist()
            indices.append(batch_indices)
        return iter(indices)


class DynamicBalanceClassSampler(Sampler):
    """
    This kind of sampler can be used for classification tasks with significant
    class imbalance.

    The idea of this sampler that we start with the original class distribution
    and gradually move to uniform class distribution like with downsampling.

    Let's define :math: D_i = #C_i/ #C_min where :math: #C_i is a size of class
    i and :math: #C_min is a size of the rarest class, so :math: D_i define
    class distribution. Also define :math: g(n_epoch) is a exponential
    scheduler. On each epoch current :math: D_i  calculated as
    :math: current D_i  = D_i ^ g(n_epoch),
    after this data samples according this distribution.

    Notes:
         In the end of the training, epochs will contain only
         min_size_class * n_classes examples. So, possible it will not
         necessary to do validation on each epoch. For this reason use
         ControlFlowCallback.

    Examples:

        >>> import torch
        >>> import numpy as np

        >>> from catalyst.data import DynamicBalanceClassSampler
        >>> from torch.utils import data

        >>> features = torch.Tensor(np.random.random((200, 100)))
        >>> labels = np.random.randint(0, 4, size=(200,))
        >>> sampler = DynamicBalanceClassSampler(labels)
        >>> labels = torch.LongTensor(labels)
        >>> dataset = data.TensorDataset(features, labels)
        >>> loader = data.dataloader.DataLoader(dataset, batch_size=8)

        >>> for batch in loader:
        >>>     b_features, b_labels = batch

    Sampler was inspired by https://arxiv.org/abs/1901.06783
    """

    def __init__(
        self,
        labels: List[Union[int, str]],
        exp_lambda: float = 0.9,
        start_epoch: int = 0,
        max_d: Optional[int] = None,
        mode: Union[str, int] = "downsampling",
        ignore_warning: bool = False,
    ):
        """
        Args:
            labels: list of labels for each elem in the dataset
            exp_lambda: exponent figure for schedule
            start_epoch: start epoch number, can be useful for multi-stage
            experiments
            max_d: if not None, limit on the difference between the most
            frequent and the rarest classes, heuristic
            mode: number of samples per class in the end of training. Must be
            "downsampling" or number. Before change it, make sure that you
            understand how does it work
            ignore_warning: ignore warning about min class size
        """
        assert isinstance(start_epoch, int)
        assert 0 < exp_lambda < 1, "exp_lambda must be in (0, 1)"
        super().__init__(labels)
        self.exp_lambda = exp_lambda
        if max_d is None:
            max_d = np.inf
        self.max_d = max_d
        self.epoch = start_epoch
        labels = np.array(labels)
        samples_per_class = Counter(labels)
        self.min_class_size = min(samples_per_class.values())

        if self.min_class_size < 100 and not ignore_warning:
            LOGGER.warning(
                f"the smallest class contains only"
                f" {self.min_class_size} examples. At the end of"
                f" training, epochs will contain only"
                f" {self.min_class_size * len(samples_per_class)}"
                f" examples"
            )

        self.original_d = {
            key: value / self.min_class_size for key, value in samples_per_class.items()
        }
        self.label2idxes = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in set(labels)
        }

        if isinstance(mode, int):
            self.min_class_size = mode
        else:
            assert mode == "downsampling"

        self.labels = labels
        self._update()

    def _update(self) -> None:
        """Update d coefficients."""
        current_d = {
            key: min(value ** self._exp_scheduler(), self.max_d)
            for key, value in self.original_d.items()
        }
        samples_per_classes = {
            key: int(value * self.min_class_size) for key, value in current_d.items()
        }
        self.samples_per_classes = samples_per_classes
        self.length = np.sum(list(samples_per_classes.values()))
        self.epoch += 1

    def _exp_scheduler(self) -> float:
        return self.exp_lambda ** self.epoch

    def __iter__(self) -> Iterator[int]:
        """
        Returns:
            iterator of indices of stratified sample
        """
        indices = []
        for key in sorted(self.label2idxes):
            samples_per_class = self.samples_per_classes[key]
            replace_flag = samples_per_class > len(self.label2idxes[key])
            indices += np.random.choice(
                self.label2idxes[key], samples_per_class, replace=replace_flag
            ).tolist()
        assert len(indices) == self.length
        np.random.shuffle(indices)
        self._update()
        return iter(indices)

    def __len__(self) -> int:
        """
        Returns:
             length of result sample
        """
        return self.length


class MiniEpochSampler(Sampler):
    """
    Sampler iterates mini epochs from the dataset used by ``mini_epoch_len``.

    Args:
        data_len: Size of the dataset
        mini_epoch_len: Num samples from the dataset used in one
          mini epoch.
        drop_last: If ``True``, sampler will drop the last batches
          if its size would be less than ``batches_per_epoch``
        shuffle: one of  ``"always"``, ``"real_epoch"``, or `None``.
          The sampler will shuffle indices
          > "per_mini_epoch" - every mini epoch (every ``__iter__`` call)
          > "per_epoch" -- every real epoch
          > None -- don't shuffle

    Example:
        >>> MiniEpochSampler(len(dataset), mini_epoch_len=100)
        >>> MiniEpochSampler(len(dataset), mini_epoch_len=100, drop_last=True)
        >>> MiniEpochSampler(len(dataset), mini_epoch_len=100,
        >>>     shuffle="per_epoch")
    """

    def __init__(
        self,
        data_len: int,
        mini_epoch_len: int,
        drop_last: bool = False,
        shuffle: str = None,
    ):
        """Sampler initialisation."""
        super().__init__(None)

        self.data_len = int(data_len)
        self.mini_epoch_len = int(mini_epoch_len)

        self.steps = int(data_len / self.mini_epoch_len)
        self.state_i = 0

        has_reminder = data_len - self.steps * mini_epoch_len > 0
        if self.steps == 0:
            self.divider = 1
        elif has_reminder and not drop_last:
            self.divider = self.steps + 1
        else:
            self.divider = self.steps

        self._indices = np.arange(self.data_len)
        self.indices = self._indices
        self.end_pointer = max(self.data_len, self.mini_epoch_len)

        if not (shuffle is None or shuffle in ["per_mini_epoch", "per_epoch"]):
            raise ValueError(
                "Shuffle must be one of ['per_mini_epoch', 'per_epoch']. "
                + f"Got {shuffle}"
            )
        self.shuffle_type = shuffle

    def shuffle(self) -> None:
        """Shuffle sampler indices."""
        if self.shuffle_type == "per_mini_epoch" or (
            self.shuffle_type == "per_epoch" and self.state_i == 0
        ):
            if self.data_len >= self.mini_epoch_len:
                self.indices = self._indices
                np.random.shuffle(self.indices)
            else:
                self.indices = np.random.choice(
                    self._indices, self.mini_epoch_len, replace=True
                )

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.

        Returns:
            python iterator
        """
        self.state_i = self.state_i % self.divider
        self.shuffle()

        start = self.state_i * self.mini_epoch_len
        stop = (
            self.end_pointer
            if (self.state_i == self.steps)
            else (self.state_i + 1) * self.mini_epoch_len
        )
        indices = self.indices[start:stop].tolist()

        self.state_i += 1
        return iter(indices)

    def __len__(self) -> int:
        """
        Returns:
            int: length of the mini-epoch
        """
        return self.mini_epoch_len


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.

    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """

        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.

        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))



class DistributedImbalancedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False,
                 labels: list = None, indices: list = None, callback_get_label: Callable = None) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

        self._calculate_weights(dataset, indices, labels, callback_get_label)

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torch.utils.data.TensorDataset):
            return dataset.tensors[1]
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def _calculate_weights(self, dataset, indices, labels, callback_get_label):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # # if num_samples is not provided, draw `len(indices)` samples in each iteration
        # self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()
        weights = 1.0 / label_to_count[df["label"]]
        self.weights = torch.DoubleTensor(weights.to_list())

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()
            indices = torch.multinomial(self.weights, len(self.dataset), replacement=True, generator=g).tolist()
        else:
            # indices = list(range(len(self.dataset)))
            indices = torch.multinomial(self.weights, len(self.dataset), replacement=True).tolist()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        dataset,
        labels: list = None,
        indices: list = None,
        num_samples: int = None,
        callback_get_label: Callable = None,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torch.utils.data.TensorDataset):
            return dataset.tensors[1]
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            if isinstance(dataset.dataset, torchvision.datasets.ImageFolder):
                return [dataset.dataset.imgs[dataset.indices[idx]][1] for idx in range(len(dataset))]
            else:
                return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


__all__ = [
    "BalanceClassSampler",
    "BatchBalanceClassSampler",
    "DistributedSamplerWrapper",
    "DynamicBalanceClassSampler",
    "MiniEpochSampler",
]
