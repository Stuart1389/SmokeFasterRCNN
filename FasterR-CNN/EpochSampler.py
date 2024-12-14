import torch
from torch.utils.data import Sampler

class EpochSampler(Sampler):
    def __init__(self, dataset, epochs):
        self.dataset = dataset
        self.epochs = epochs
        self.total_samples = len(dataset)
        self.current_epoch = 0

    def __iter__(self):
        # Return indices for the current epoch
        start_idx = self.current_epoch * self.total_samples
        end_idx = (self.current_epoch + 1) * self.total_samples
        indices = list(range(start_idx, end_idx))

        # After iteration, move to the next epoch
        self.current_epoch += 1
        if self.current_epoch >= self.epochs:
            self.current_epoch = 0  # Reset epoch after completing all epochs

        return iter(indices)

    def __len__(self):
        return self.total_samples