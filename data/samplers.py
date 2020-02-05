from collections import Counter

import torch
import torch.utils.data
from torch.utils.data import SequentialSampler, RandomSampler

from data.DatabaseDataset import DatabaseDataset
from data.TabularDataset import TabularDataset

SequentialSampler
RandomSampler


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """
    Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, data_source, num_samples=None):
        assert isinstance(data_source, (DatabaseDataset, TabularDataset))

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(data_source) if num_samples is None else num_samples

        # weight for each sample
        print('Getting sampling weights for dataset')
        if isinstance(data_source, DatabaseDataset):
            labels = [dp[4] for dp_id, dp in data_source]
        elif isinstance(data_source, TabularDataset):
            labels = data_source.targets.tolist()
        label_to_count = Counter(labels)
        weights = [1.0 / label_to_count[l] for l in labels]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples
