import os
import shutil
import random
from GetValues import checkColab

base_dir = checkColab()

# Main dir, keep dataset intact
main_dir = os.path.join(base_dir, "Dataset", "Large data", "Main")
main_annot = os.path.join(main_dir, "annotations", "xmls")
main_image = os.path.join(main_dir, "images")

# Train dir
train_dir = os.path.join(base_dir, "Dataset", "Large data", "Train")
train_annot = os.path.join(train_dir, "annotations", "xmls")
train_image = os.path.join(train_dir, "images")

# Validate dir
val_dir = os.path.join(base_dir, "Dataset", "Large data", "Validate")
val_annot = os.path.join(val_dir, "annotations", "xmls")
val_image = os.path.join(val_dir, "images")

# Test dir
test_dir = os.path.join(base_dir, "Dataset", "Large data", "Test")
test_annot = os.path.join(test_dir, "annotations", "xmls")
test_image = os.path.join(test_dir, "images")

# split ratios (70/15/15)
train_split = 0.7
val_split = 0.15
test_split = 0.15

# function deletes files from dir
def clear_directory(directory):
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            os.remove(file_path)

# function copies files to dir
def copy_files(file_pairs, dest_image_dir, dest_annot_dir):
    for image_file, annot_file in file_pairs:
        shutil.copy(image_file, dest_image_dir)
        shutil.copy(annot_file, dest_annot_dir)

# creating train/validate/test splits
def train_test_split():
    # Clear dirs
    clear_directory(train_image)
    clear_directory(train_annot)
    clear_directory(val_image)
    clear_directory(val_annot)
    clear_directory(test_image)
    clear_directory(test_annot)

    # get list of images and annotations
    images = sorted([os.path.join(main_image, f) for f in os.listdir(main_image) if f.endswith(('.jpg', '.jpeg', '.png'))])
    annotations = sorted([os.path.join(main_annot, f.replace('.jpg', '.xml').replace('.jpeg', '.xml').replace('.png', '.xml')) for f in os.listdir(main_image) if f.endswith(('.jpg', '.jpeg', '.png'))])

    #pair images and annotations
    dataset = list(zip(images, annotations))

    #Shuffle dataset
    random.shuffle(dataset)

    #get split index
    #split_idx = int(len(dataset) * split_ratio)
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)

    #split into train and test
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]

    # make dirs if they dont exist
    os.makedirs(train_image, exist_ok=True)
    os.makedirs(train_annot, exist_ok=True)
    os.makedirs(val_image, exist_ok=True)
    os.makedirs(val_annot, exist_ok=True)
    os.makedirs(test_image, exist_ok=True)
    os.makedirs(test_annot, exist_ok=True)

    #copy files to dirs
    copy_files(train_dataset, train_image, train_annot)
    copy_files(val_dataset, val_image, val_annot)
    copy_files(test_dataset, test_image, test_annot)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

train_test_split()