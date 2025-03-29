from GetValues import checkColab, setTrainValues
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import h5py
import numpy as np
import os
from EpochSampler import EpochSampler
import matplotlib.pyplot as plt

# HDF5 DATASET EXTRACTS DATA FROM HDF5 FILE FOR TRAINING
class SmokeDatasetHdf5(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.epochs = setTrainValues("EPOCHS")
        self.debug_dataloader = None
        self.current_epoch = 0

    # used to track current epoch and how many datapoints in dataset
    def __len__(self):
        with h5py.File(self.file_path, 'r') as h5df_file:
            epoch_group = f"epoch_{self.current_epoch + 1}"
            self.total_samples = len(h5df_file[epoch_group]['images'])
        # nice to check things are correct at start
        print(f"Epoch {self.current_epoch + 1}: Total samples = {self.total_samples}")
        return self.total_samples


    # Extract data from hdf5
    def __getitem__(self, idx):
        epoch_idx = self.current_epoch + 1
        # find index of datapoint within epoch group
        actual_idx = idx % self.total_samples

        # Open the HDF5 file to retrieve data
        # Each dataloader worker opens the file independently
        with h5py.File(self.file_path, 'r') as h5df_file:
            epoch_group = f"epoch_{epoch_idx}"
            # checks that the epoch group exists
            if epoch_group in h5df_file:
                epoch_data = h5df_file[epoch_group]
                # get and reconstruct image
                flat_image = np.array(epoch_data['images'][actual_idx])
                height, width = epoch_data['image_dims'][actual_idx]
                image = torch.tensor(flat_image.reshape((height, width, -1)), dtype=torch.float32)
                # get and reconstruct bbox
                num_bbox = epoch_data['num_bbox'][actual_idx]
                boxes = torch.tensor(epoch_data['boxes'][actual_idx].reshape(num_bbox, 4), dtype=torch.float32)
                # get other targets
                labels = torch.tensor(epoch_data['labels'][actual_idx])
                image_id = epoch_data['image_ids'][actual_idx]
                area = torch.tensor(epoch_data['areas'][actual_idx])
                iscrowd = torch.tensor(epoch_data['iscrowds'][actual_idx])

                target = {
                    "boxes": boxes,
                    "labels": labels,
                    "image_id": image_id,
                    "area": area,
                    "iscrowd": iscrowd
                }
            return image, target

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    return tuple(zip(*batch))


# Run this script to check hdf5 file is working
# make sure to have a valid hdf5 file set in GetValues.py h5py_dir_load_name
if __name__ == '__main__':

    base_dir = checkColab()
    # select hdf5 file to load from GetValues.py
    write_main_path = Path(f"{base_dir}/DatasetH5py/" + setTrainValues("h5py_dir_load_name"))
    write_train_path = Path(f"{write_main_path}/Train.hdf5")

    num_workers = os.cpu_count() # cores available
    batch_size = setTrainValues("BATCH_SIZE")
    epochs = 2
    visualise = True

    # Load dataset and create dataloader
    debug_dir = SmokeDatasetHdf5(str(write_train_path))
    debug_sampler = EpochSampler(debug_dir, epochs=epochs)
    debug_dataloader = DataLoader(
        dataset=debug_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        shuffle=False,
        sampler=debug_sampler
    )

    # prints and optionally displays image/targets from hdf5 file
    for epoch in range(epochs):
        for batch, (images, targets) in enumerate(debug_dataloader):
            print(f"Processing batch {batch} out of {len(debug_dataloader)}")
            print(images, targets)
            if(visualise):
                for image in images:
                    plt.imshow(image.permute(1, 2, 0).numpy())
                    plt.show()

