from GetValues import setTrainValues, checkColab
from SmokeModel import SmokeModel
import h5py
from pathlib import Path
import numpy as np
import time

# class used to create hdf5 files from dataset
class WriteHdf5:
    def __init__(self):
        self.batch_size = setTrainValues("BATCH_SIZE")
        self.epochs = setTrainValues("EPOCHS")
        self.smoke_model = SmokeModel()
        self.train_dataloader, self.validate_dataloader, _ = self.smoke_model.get_dataloader()

        # Directory vars
        self.dir_name = setTrainValues("h5py_dir_save_name")
        self.base_dir = checkColab()
        self.write_main_path = Path(f"{self.base_dir}/DatasetH5py/" + setTrainValues("h5py_dir_save_name"))
        self.write_train_path = Path(f"{self.write_main_path}/Train.hdf5")
        self.write_validate_path = Path(f"{self.write_main_path}/Validate.hdf5")

        #write to file
        # create train hdf5 file
        self.write_to_file(self.train_dataloader, self.write_main_path, self.write_train_path)
        # create validate hdf5 file
        self.write_to_file(self.validate_dataloader, self.write_main_path, self.write_validate_path)

        #check file
        self.read_from_file(self.write_train_path)
        self.read_from_file(self.write_validate_path)

    # method reads from hdf5 file, used to examine a hdf5 file created using this script
    def read_from_file(self, file_path):
        with h5py.File(file_path, 'r') as file:
            # Function to print all groups and datasets recursively
            def print_items(group, prefix=''):
                for name, item in group.items():
                    if isinstance(item, h5py.Group):
                        print(f'Group: {prefix}{name}')
                        # Recursively print contents of the group
                        print_items(item, prefix + name + '/')
                    elif isinstance(item, h5py.Dataset):
                        print(f'Dataset: {prefix}{name}, Shape: {item.shape}, Dtype: {item.dtype}')
                        # Optionally print the actual data
                        #print(item[:])  # Uncomment to print data

            # Start from the root group and print all items
            print_items(file)


    # method to write to hdf5 file from dataloader, see README for more info
    def write_to_file(self, dataloader, dir_write_path, file_write_path):
        dir_write_path.mkdir(parents=True, exist_ok=True) # path to write to
        chunk_size = setTrainValues("hdf5_chunk_size")
        # get dataset length
        total_samples = len(dataloader.dataset)

        # open hdf5 and write
        with h5py.File(file_write_path, "w") as h5df_file:
            # Writing a single epoch, e.g. if 5 epochs in GetValues.py (see README) this will repeat 5 times
            for epoch in range(self.epochs):
                print(f"Processing epoch {epoch + 1} out of {self.epochs}")
                epoch_group = h5df_file.create_group(f"epoch_{epoch + 1}")  # Create a group for each epoch

                # creating a dataset for image and each target type
                image_storage = epoch_group.create_dataset("images", shape=(total_samples,), chunks=chunk_size,
                                                           dtype=h5py.special_dtype(vlen=np.dtype('float32')))
                print("image chunks = ", image_storage.chunks)
                # store image width and height so it can be red-assembled
                image_dims_storage = epoch_group.create_dataset("image_dims", shape=(total_samples, 2), chunks=(chunk_size, 2), dtype='int32')
                box_storage = epoch_group.create_dataset("boxes", shape=(total_samples,), chunks=chunk_size,
                                                         dtype=h5py.special_dtype(vlen='float32'))
                print(box_storage.chunks)
                #num off bboxes to reconstruct with shape
                num_bbox_storage = epoch_group.create_dataset("num_bbox", shape=(total_samples, ), chunks=chunk_size, dtype='int32')
                label_storage = epoch_group.create_dataset("labels", shape=(total_samples,), chunks=chunk_size, dtype=h5py.special_dtype(vlen='int64'))
                image_id_storage = epoch_group.create_dataset("image_ids", shape=(total_samples,), chunks=chunk_size, dtype='int64')
                area_storage = epoch_group.create_dataset("areas", shape=(total_samples,),
                                                          chunks=chunk_size, dtype=h5py.special_dtype(vlen='float32'))
                iscrowd_storage = epoch_group.create_dataset("iscrowds", shape=(total_samples,), chunks=chunk_size, dtype=h5py.special_dtype(vlen='int64'))

                # Create global index, this is so that we write to the correct spot when using dataloader
                global_index = 0

                # Loop through each batch from dataloader for current epoch
                for batch, (images, targets) in enumerate(dataloader):
                    # getting data from dataloader
                    print(f"Processing batch {batch + 1} out of {len(dataloader)}")
                    image_tensors = list(tensor.to("cpu", non_blocking=False) for tensor in images)
                    targets = list(target for target in targets)
                    batch_size = len(image_tensors)

                    # write to file
                    for i in range(batch_size):
                        # image
                        image = images[i].numpy()
                        image_shape = image.shape
                        image_storage[global_index + i] = image.flatten()
                        image_dims_storage[global_index + i] = image_shape[:2]

                        #targets
                        target = targets[i]
                        box_storage[global_index + i] = target["boxes"].flatten()
                        num_bbox_storage[global_index + i] = target["boxes"].shape[0]
                        label_storage[global_index + i] = target["labels"]
                        image_id_storage[global_index + i] = target["image_id"]
                        area_storage[global_index + i] = target["area"]
                        iscrowd_storage[global_index + i] = target["iscrowd"]

                    # Update global index for the next batch
                    global_index += batch_size

                print(f"Data successfully written to {file_write_path}")


if __name__ == '__main__':
    start_time = time.time()
    write_h5py = WriteHdf5()
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")




