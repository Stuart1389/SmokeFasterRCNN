from torch.utils.data import Sampler
from GetValues import setTrainValues
"""
the purpose of this sampler is so that we can track what epoch we're on
this is necessary when using hdf5 with transforms since we need to read
from the current epoch
otherwise transforms will always be the same
"""
class EpochSampler(Sampler):
    def __init__(self, dataset, epochs):
        self.dataset = dataset
        self.epochs = epochs
        self.total_samples = len(dataset)
        self.current_epoch = 0
        self.force_first_epoch = setTrainValues("force_first_epoch")

    def __iter__(self):
        # Return indices for the current epoch
        print(self.current_epoch)
        if self.force_first_epoch:
            start_idx = 0
            end_idx = self.total_samples
        else:
            start_idx = self.current_epoch * self.total_samples
            end_idx = (self.current_epoch + 1) * self.total_samples

        indices = list(range(start_idx, end_idx))

        # After iteration, move to the next epoch
        if not self.force_first_epoch:
            self.current_epoch += 1
            if self.current_epoch >= self.epochs:
                self.current_epoch = 0  # Reset epoch after completing all epochs

        self.dataset.current_epoch = self.current_epoch
        return iter(indices)

    def __len__(self):
        return self.total_samples