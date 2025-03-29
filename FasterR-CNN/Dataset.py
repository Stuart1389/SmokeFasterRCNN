import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from SmokeUtils import extract_boxes
from GetValues import checkColab, setTrainValues, setGlobalValues
import time
import sys

### !!IMAGE TRANSFORMATIONS!!
# Albumentations library, can do transforms for image and bbox as one
# pipeline assumes pascal_voc is format (xmin, ymin, xmax, ymax) for bbox coords
transform_train = A.Compose([
    # EXAMPLE TRANSFORMS, SEE https://albumentations.ai/
    #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
    #A.PadIfNeeded(min_height=640, min_width=480),
    #A.RandomBrightnessContrast(p=0.4),
    #A.RandomContrast(p=0.5),
    #A.PadIfNeeded(min_height=480, min_width=640),
    #A.Resize(height=480, width=640),
    #A.ToGray(p=1.0),
    #A.SafeRotate(limit=10, p=1, border_mode=cv2.BORDER_CONSTANT),
    #A.GaussNoise(var_limit=(0.01, 0.005), p=1),
    #A.HorizontalFlip(p=1),
    #A.BBoxSafeRandomCrop(erosion_rate=0, p=1),
    #A.RandomScale(scale_limit=0.7, p=1),
    #A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
    #A.Resize(height=224, width=224),

    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels', 'class_id']))

transform_validate = A.Compose([
    #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
    #A.Resize(height=224, width=224),
    #A.ToGray(p=1.0),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels', 'class_id']))

transform_test = A.Compose([
    #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
    #A.Resize(height=224, width=224),
    #A.ToGray(p=1.0),
    ToTensorV2()
])

# this is used by utility files
# e.g. SmokeUpscale.py
# its only here to keep transforms for utility seperate
# from split transforms
transform_utility = A.Compose([
    ToTensorV2()
])


# !!CREATING DATASET!!
# setting basedir for dataset
base_dir = checkColab()
dataset_dir = Path(f"{base_dir}/Dataset/" + setTrainValues("dataset"))

# DEFAULT DATASET FOR PIPELINE
class smokeDataset(torch.utils.data.Dataset):
  # Constructor, setting instanced variables
  def __init__(self, main_dir: str, transform=None, testing = False, using_upscale_util = False):
    self.main_dir = main_dir
    # Sort image and annotations lists so they're retrieved in the same order
    self.images = sorted(list(Path(str(main_dir) + "/images/").glob("*.jpeg")))
    self.images += sorted(list(Path(str(main_dir) + "/images/").glob("*.jpg")))
    self.annotations = sorted(list(Path(str(main_dir) + "/annotations/xmls").glob("*.xml")))
    self.transform = transform # which transform to use
    self.testing = testing # Whether we are testing or not
    self.return_filenames = using_upscale_util # Whether the SmokeUpscale.py script is being run
    self.empty_image = None


  def __len__(self):
    # Return length of dataset e.g. number of images
    return len(self.images)

  def __getitem__(self, idx):
    # Get items from dataset
    img_path = self.images[idx] # return image at index
    filename = img_path.stem # get image filename
    # checking that annotation exists at index
    if(idx < len(self.annotations)):
        annotation_path = self.annotations[idx] # return annotation at index
        annotation_name = annotation_path.stem
        # checking that image and annotation names are the same
        if(filename != annotation_name):
            print(f"Filename {filename} and annotation name {annotation_name} don't match. Incorrect annotation will result in a poor model.")
            print("See Dataset.py to disable")
            sys.exit(1)

        #Parse/rerieve data from xml file
        if (setTrainValues("upscale_image") or setTrainValues("upscale_bbox")):
            # used to adjust bboxes when upscaling to match targets
            upscale_value = setTrainValues("upscale_value")
        else:
            upscale_value = 1
        # Global parse function from SmokeUtils see README for more info
        boxes, _, labels_int, labels = extract_boxes(annotation_path, upscale_value=upscale_value)
    else:
        # If there are no annotations then we know image is empty
        self.empty_image = True
        boxes = []
        labels = []
        labels_int = []

    #albumentations wants numpy
    image = np.array(Image.open(img_path))
    # normalize the image to [0, 1] so i can be seen visually
    if image.dtype == np.uint8:
      image = image / 255.0
    image = image.astype(np.float32)

    # this is transformations for training and validation
    if (self.transform and not self.testing and not self.return_filenames):
        """
        Transforming images and targets using albumentations
        this is necessary because transformations can alter ground truth position
        relative to objects, this ensures ground truth is correctly aligned with objects
        """
        transformed = self.transform(image=image, bboxes=boxes, class_labels=labels, class_id=labels_int)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        #transformed_class_labels = transformed['class_labels']
        transformed_class_id = transformed['class_id']
        transformed_bboxes = torch.tensor(transformed_bboxes, dtype=torch.float32)
        transformed_class_id = torch.tensor(transformed_class_id, dtype=torch.int64)
        """
        Faster RCNN expects image, target to be returned
        Target should be a dictionary containing bounding box and label
        boxes, labels, image_id, area, iscrowd
        """
        image = transformed_image
        image_id = idx
        target = {}
        if(self.empty_image):
            # empty targets if we dont have annotations, implemented for training on clouds
            target["boxes"] = torch.as_tensor(np.array(np.zeros((0, 4)), dtype=float))
            target["labels"] = torch.as_tensor(np.array([], dtype=int), dtype=torch.int64)
            target["area"] = torch.as_tensor(np.array([], dtype=float))
        else:
            target["boxes"] = transformed_bboxes
            target["labels"] = transformed_class_id
            area = (transformed_bboxes[:, 3] - transformed_bboxes[:, 1]) * (
                        transformed_bboxes[:, 2] - transformed_bboxes[:, 0])
            target["area"] = area
        target["image_id"] = image_id
        target["iscrowd"] = torch.zeros((transformed_bboxes.shape[0],), dtype=torch.int64)
        return image, target

    # Filenames are needed for the SmokeUpscaling to save the image to disk as the original filename
    if(self.return_filenames):
        self.transform = transform_utility
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        return image_tensor, filename

    # this is used when testing
    if (self.transform and self.testing):
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        return image_tensor, filename



# !!VISUALIZATION!!
# To see if bounding boxes are being displayed properly after transformations, etc.
# Function to display bbox over image
def visualise_bbox(img, bbox, class_name):
    color = (0, 0, 255) # bbox colour
    text_color = (255, 255, 255) # gt text colour
    index_to_class = {v: k for k, v in setGlobalValues("CLASS_INDEX_DICTIONARY").items()} # reverse dictionary
    class_name = index_to_class.get(class_name.item(), "N/a") # map label index to class name
    bbox_text = class_name + " Ground truth" # set bbox text
    # getting bbox cords
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

    # creating ground truth box
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=2)

    # creating text box
    ((text_width, text_height), _) = cv2.getTextSize(bbox_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=bbox_text,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=text_color,
        lineType=cv2.LINE_AA,
    )
    return img

# Creates plot with image
def plot_image(image, bboxes, labels, image_id):
    img = image.permute(1, 2, 0).numpy() # convert tensor to numpy array
    # Check if bboxes exists (e.g. if zoomed in), prevents index out of bounds error
    # Prevents index out of bounds error from below if no bbox
    if bboxes.nelement() > 0:
      # convert bboxs to a list
      if isinstance(bboxes[0], (int, float)):  # Check if it's a single bbox
          bboxes = [bboxes]
          labels = [labels]
          print (bboxes, labels)

    # overlay each bbox over image
    for bbox, label in zip(bboxes, labels):
        img = visualise_bbox(img, bbox, label)

    # create figure
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)

    # save image to disk
    save_path = './visualise_image'
    save_file = os.path.join(save_path, f"visualized_image_{image_id}.png")
    plt.savefig(save_file, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
    print(f"Image saved to {save_file}")

# only execute when running this script specifically
if __name__ == '__main__':
    # define directory to save images to
    directory_path = './visualise_image'
    # create the directory if it doesn't exist
    os.makedirs(directory_path, exist_ok=True)

    # visualise image from train split
    train_test = smokeDataset(str(dataset_dir) + "/Train", transform_train) # create instance of dataset
    # measure how long it takes to process the first image
    start_time = time.time()
    # get image and targets
    image, target = train_test.__getitem__(1)
    end_time = time.time()
    elapsed_time = (end_time - start_time)
    print(f"It took {elapsed_time:.2f} to process 1 image")
    bbox = target["boxes"]
    label = target["labels"]
    plot_image(image, bbox, label, 0)



    # visualise image from test split
    test_test = smokeDataset(str(dataset_dir) + "/Test", transform_train) # create instance of dataset
    image, target = test_test.__getitem__(1) # get image and annotation at index 1
    bbox = target["boxes"]
    label = target["labels"]
    plot_image(image, bbox, label, 1)