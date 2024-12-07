import os
import random
import xml.etree.ElementTree as ET
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from Get_Values import checkColab, setTrainValues
from torch.utils.data import Dataset
from multiprocessing import Pool
import time


### !!IMAGE TRANSFORMATIONS!!
# Albumentations library, can do transforms for image and bbox as one
# pascal_voc is format (xmin, ymin, xmax, ymax) we're using for bounding box coords
transform_train = A.Compose([
    #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
    #A.PadIfNeeded(min_height=320, min_width=240, border_mode=cv2.BORDER_CONSTANT), # prevents shape mismatch from image being cut off
    #A.PadIfNeeded(min_height=640, min_width=480), # doesnt work currently, need to fix
    #A.RandomCrop(width= round(320), height= round(240)), # needs padding or will throw error
    #A.BBoxSafeRandomCrop(erosion_rate=0.2, p=0.5),
    A.GaussNoise(var_limit=(0.05, 0.01), p=0.3),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.SafeRotate(limit=5, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels', 'class_id']))

transform_test_validate = A.Compose([
    #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
    ToTensorV2()
])


# !!CREATING DATASET!!
#Setting basedir for dataset
base_dir = checkColab()
dataset_dir = Path(f"{base_dir}/Dataset/" + setTrainValues("dataset"))

class smokeDataset(torch.utils.data.Dataset):
  # Constructor, setting instanced variables
  def __init__(self, main_dir: str, transform=None, testing = False):

    self.main_dir = main_dir
    self.images = list(Path(str(main_dir) + "/images/").glob("*.jpeg")) # set to list of all images
    self.annotations = list(Path(str(main_dir) + "/annotations/xmls").glob("*.xml")) # set to list of all xml files
    #self.loaded_images = [np.array(Image.open(img_path)) for img_path in self.images]
    self.transform = transform
    self.testing = testing



    # Constructor END

  def parse_xml(self, annotation_path, testing = False):
          # Parsing xml files for each image to find bbox
          tree = ET.parse(annotation_path)
          root = tree.getroot()

          # getting bounding boxes from xml file
          boxes = []
          areas = []
          for obj in root.findall("object"):
              xml_box = obj.find("bndbox")
              xmin = float(xml_box.find("xmin").text)
              ymin = float(xml_box.find("ymin").text)
              xmax = float(xml_box.find("xmax").text)
              ymax = float(xml_box.find("ymax").text)
              boxes.append([xmin, ymin, xmax, ymax])

          #Dataset has only background and smoke
          labels = [] # label name aka "smoke"
          labels_int = [] #label id aka 1
          class_to_idx = {"smoke": 1} # dictionary if "smoke" return 1
          for obj in root.findall("object"):
              label = obj.find("name").text # find name in xml
              labels.append(label)
              labels_int.append(class_to_idx[label])
              #boxes.append(class_to_idx[label])
              #boxes.append(label)

          #boxes.append([xmin, ymin, xmax, ymax, label])

          #print(f"Boxes: {boxes}")
          #print(f"labels: {label}")
          #print(f"Labels: {labels}, Labels_int {labels_int}")
          return boxes, labels, labels_int


  def __len__(self):
    # Return length of dataset/number of images
    return len(self.images)

  def __getitem__(self, idx):
    # Return items from dataset
    img_path = self.images[idx] # return image at index
    filename = img_path.stem # get filename
    annotation_path = self.annotations[idx] # return annotation at index
    #image = Image.open(img_path)
    #albumentations wants numpy
    image = np.array(Image.open(img_path))
    #image = self.loaded_images[idx]
    #Convert to float32 for model
    # normalize the image to [0, 1] if it is not already in float format, prevents image from being completely white
    if image.dtype == np.uint8:
      image = image / 255.0

    image = image.astype(np.float32)

    #print(f"Image dtype: {image.dtype}")
    #Parse data and return to instance
    boxes, labels, labels_int = self.parse_xml(annotation_path)
    #boxes, labels = self.parse_xml(annotation_path)
    #boxes = self.parse_xml(annotation_path)

    # this is transformations for training and validation
    if (self.transform and not self.testing):
        """
        # Created seperate labels list cause kept getting empty when i printed label and bbox
        because bbox was out of bounds because crop made image smaller,
        might change back to single list with label at end later lol
        """
        # Going through albumentations transform functions
        transformed = self.transform(image=image, bboxes=boxes, class_labels=labels, class_id=labels_int)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_class_labels = transformed['class_labels']
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
        #print(transformed_bboxes)
        # if there are no bboxes and we calculate area then it will throw error
        # check if bboxes exist, if they do then calculate area otherwise just set it to nothing
        area = None
        if(transformed_bboxes.numel() != 0):
            area = (transformed_bboxes[:, 3] - transformed_bboxes[:, 1]) * (transformed_bboxes[:, 2] - transformed_bboxes[:, 0])

        target = {}
        target["boxes"] = transformed_bboxes
        target["labels"] = transformed_class_id
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = torch.zeros((transformed_bboxes.shape[0],), dtype=torch.int64)
        #print(image, target)
        return image, target

    if (self.transform and self.testing):
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        #print(image.dtype)
        #print(image.shape)
        return image_tensor, filename



# !!VISUALIZATION!!
# To see if bounding boxes were being displayed properly during transformations, etc.
# Leaving incase i need it later for whatever reason

# Only want to execute these if im running this .py script specifically, prevents it from running when using other scripts
if __name__ == '__main__':
    train_test = smokeDataset(str(dataset_dir) + "/Train", transform_train) # create instance of dataset
    start_time = time.time()
    image, target = train_test.__getitem__(1)
    end_time = time.time()
    elapsed_time = (end_time - start_time)
    print(f"Elapsed time: {elapsed_time:.2f}")
    bbox = target["boxes"]
    label = target["labels"]

    # Made this because i didnt add collate fn so kept getting error because not list of dictionaries, was tryna debug
    """
    def check_for_strings_in_target(target):
        for key, value in target.items():
            if isinstance(value, list):
                # If the value is a list, check if any of the elements are strings
                if any(isinstance(v, str) for v in value):
                    print(f"String found in {key}: {value}")
            elif isinstance(value, str):
                print(f"String found in {key}: {value}")
            else:
                print(f"No strings found in {key}: {value}")
    """

    #image, target = smokeDataset(str(dataset_dir) + "/Train", transform_train)[0]
    #check_for_strings_in_target(target)
    ### End check for strings, reduntant

    ### Looking at image using dataset
    BOX_COLOR = (0, 0, 255) # red
    TEXT_COLOR = (255, 255, 255) # white

    # Function to display bbox over image
    def visualize_bbox_pascal_voc(img, bbox, class_name, color=BOX_COLOR, thickness=2):
        bbox_text = "Ground truth"
       # drawing bbox over image
        x_min, y_min, x_max, y_max = bbox
        x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

        ((text_width, text_height), _) = cv2.getTextSize(bbox_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
        cv2.putText(
            img,
            text=bbox_text, # can change this to class_name if you have multiple classes
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=TEXT_COLOR,
            lineType=cv2.LINE_AA,
        )
        return img


    # Creates plot with image
    def visualize(image, bboxes, labels):
        img = image.permute(1, 2, 0).numpy() # convert tensor to numpy array
        #img = image.copy() # throws error

        # Check if bboxes exists (e.g. if zoomed in), prevents index out of bounds error
        # Prevents index out of bounds error from below if no bbox
        #if bboxes:
        #Convert to if bboxes to work with tensors

        if bboxes.nelement() > 0:
          # If there's only one bounding box, convert it to a list
          if isinstance(bboxes[0], (int, float)):  # Check if it's a single bbox
              bboxes = [bboxes]
              labels = [labels]
              print (bboxes, labels)

        for bbox, label in zip(bboxes, labels):
            img = visualize_bbox_pascal_voc(img, bbox, label)

        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(img)
        plt.show()

    #image, bbox, label = train_test.__getitem__(1) # get image and annotation at index 1
    image, target = train_test.__getitem__(1) # get image and annotation at index 1
    bbox = target["boxes"]
    #label = target["labels"] cba
    label = "smoke"
    visualize(image, bbox, label)


    # Test the test dataset too
    test_test = smokeDataset(str(dataset_dir) + "/Test", transform_train) # create instance of dataset
    #image, bbox, label = test_test.__getitem__(1) # get image and annotation at index 1
    image, target = test_test.__getitem__(1) # get image and annotation at index 1
    bbox = target["boxes"]
    #label = target["labels"]
    label = "smoke"
    visualize(image, bbox, label)