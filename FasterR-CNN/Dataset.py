import torch
from PIL import Image, ImageDraw
from pathlib import Path
import os


import random
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from colabAdj import checkColab


### Transforms
from torchvision.transforms import transforms

""" Old, use v2 instead
train_image_transform = transforms.Compose([
    transforms.ToTensor()
])
"""
"""
train_image_transform = v2.Compose([
    v2.ToImage()
])
"""
"""
transform_b = v2.Compose([
    v2.ToTensor
])
"""

# Albumentations library, can do transforms for image and bbox as one
#https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

# pascal_voc is format (xmin, ymin, xmax, ymax) we're using for bounding box coords, alternatives inc yolo, coco, etc
transform_t = A.Compose([
    #A.PadIfNeeded(min_height=320, min_width=240, border_mode=cv2.BORDER_CONSTANT), # prevents shape mismatch from image being cut off
    #A.PadIfNeeded(min_height=320, min_width=240), # doesnt work currently, need to fix
    #A.RandomCrop(width= round(320), height= round(240)), # needs padding or wil lthrow error
    A.HorizontalFlip(p=0.5),
    #A.RandomBrightnessContrast(p=0.2),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels', 'class_id']))





#from albumentations.pytorch import ToTensorV2


### Dataset
from math import e
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from pathlib import Path
import xml.etree.ElementTree as ET
import os
import numpy as np
import albumentations as A
import cv2

#base dir
base_dir = checkColab()
dataset_dir = Path(f"{base_dir}/Dataset/Large data")


BATCH_SIZE = 2 # using 6GB/8GB of vram, > = mega slow
NUM_WORKERS = os.cpu_count()
device = "cuda" if torch.cuda.is_available() else "cpu"

class smokeDataset(torch.utils.data.Dataset):
  # Constructor, setting instanced variables
  def __init__(self,
               main_dir: str,
               transform=None,
               tensor_transform=None):

    self.main_dir = main_dir
    self.images = list(Path(str(main_dir) + "/images/").glob("*.jpeg")) # set to list of all images
    self.annotations = list(Path(str(main_dir) + "/annotations/xmls").glob("*.xml")) # set to list of all xml files
    self.transform = transform
    self.tensor_transform = tensor_transform # !!This doesnt exist atm

    # Constructor END

  def parse_xml(self, annotation_path):
          # Parsing xml files for each image to find bbox
          tree = ET.parse(annotation_path)
          root = tree.getroot()

          # getting bounding boxes from xml file
          boxes = []
          for obj in root.findall("object"):
              xml_box = obj.find("bndbox")
              xmin = float(xml_box.find("xmin").text)
              ymin = float(xml_box.find("ymin").text)
              xmax = float(xml_box.find("xmax").text)
              ymax = float(xml_box.find("ymax").text)
              boxes.append([xmin, ymin, xmax, ymax])

          #!!! Most of this gettin scrapped, using Albumentations lib

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
          return boxes, labels, labels_int#, torch.tensor(labels) # Return so we can use with __getitem__
          # Parsing END


  def __len__(self):
    # Return length of dataset/number of images
    return len(self.images)

  def __getitem__(self, idx):
    # Return items from dataset
    img_path = self.images[idx] # return image at index
    annotation_path = self.annotations[idx] # return annotation at index
    #image = Image.open(img_path)
    #albumentations wants numpy
    image = np.array(Image.open(img_path))
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

    # Turning bounding boxes into tensors
    # Cut this cause using albumentations
    """
    bounding_tensor = tv_tensors.BoundingBoxes(
        torch.tensor(boxes), # convert to tensors
        format="XYXY", # Format xmin, ymin, xman, ymax
        canvas_size=image.size # Image size need to be same as image actual size
    )"""


    # Leaving this here for augmentations later
    if self.transform:
            #image, bounding_tensor = self.transform(image, bounding_tensor)
            """
            image = self.transform(image)
            image = self.bounding_transform(image)
            bounding_tensor = self.bounding_transform(bounding_tensor)
            """
            #print(boxes, labels)
            """
            # Created seperate labels list cause kept getting empty when i printed label and bbox
            because bbox was out of bounds because crop made image smaller,
            might change back to single list with label at end later lol

            """
            # Going through albumentations transform functions
            transformed = self.transform(image=image, bboxes=boxes, class_labels=labels, class_id=labels_int)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            #print(f"transformed_bboxes: {transformed_bboxes}")
            transformed_class_labels = transformed['class_labels']
            transformed_class_id = transformed['class_id']

            transformed_bboxes = torch.tensor(transformed_bboxes, dtype=torch.float32)
            #transformed_class_labels = torch.tensor(transformed_class_labels, dtype=torch.int64)
            transformed_class_id = torch.tensor(transformed_class_id, dtype=torch.int64)

            """
            print(f"type transformed_image: {type(transformed_image)}")
            print(f"type transformed_bboxes: {type(transformed_bboxes)}")
            print(f"type transformed_class_labels: {type(transformed_class_labels)}")
            print(f"type transformed_class_id: {type(transformed_class_id)}")
            """

            #print(f"Transformed image: {transformed_image}")
            #print(f"Transformed bboxes: {transformed_bboxes}")

            # Faster RCNN expects image, target to be returned
            """
            Target should be a dictionary containing bounding box and label
            boxes, labels, image_id, area, iscrowd
            """

            image = transformed_image
            #image = image.to(torch.float32)
            image_id = idx
            area = (transformed_bboxes[:, 3] - transformed_bboxes[:, 1]) * (transformed_bboxes[:, 2] - transformed_bboxes[:, 0])
            target = {}
            target["boxes"] = transformed_bboxes
            target["labels"] = transformed_class_id
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = torch.zeros((transformed_bboxes.shape[0],), dtype=torch.int64)



            """
            #print(f"Target labels_int {labels_int}")
            print(f"type target: {type(target)}")
            print(f"type boxes: {type(target['boxes'])}")
            print(f"type labels: {type(target['labels'])})")
            print(target['labels'])
            print(f"BOXES: {target['boxes']}, {target['boxes'].shape}")

            print(f"Target boxes {target['boxes']}")
            print(f"Target labels {target['labels']}")
            print(f"Target boxes shape {target['boxes'].shape}")
            print(f"Target labels shape {target['labels'].shape}")
            """

    """
    else:
        transformed_image = image
        transformed_bboxes = boxes
        transformed_class_labels = labels
    """


    #return an image and its bounding box as a tensor
    #return transformed_image, transformed_bboxes, transformed_class_labels#, bounding_tensor #, labels
    #print(f"BOXES: {target['boxes']}")
    #print(f"\nShapes: boxes {target['labels'].shape}")
    return image, target
# Only want to execute these if im running this .py script specifically, prevents it from running when importing module
if __name__ == '__main__':
    #print(f"Dataset dir: {dataset_dir}")
    train_test = smokeDataset(str(dataset_dir) + "/Train", transform_t) # create instance of dataset
    #image, bounding, labels = test.__getitem__(1)
    #image, bbox, label = train_test.__getitem__(1)
    image, target = train_test.__getitem__(1)
    bbox = target["boxes"]
    label = target["labels"]

    #print(f"{image},\n Image shape:{image.shape} - Colour channels, height, width")
    #print(bbox)
    #print(label)
    #print(bounding.data, bounding.shape, bounding.squeeze())
    #print(f"\nLabels:{labels}")
    #test.__getitem__(0)
    #testB, testC = test.__getitem__(0)
    #print(testB)
    #print(testC)

    # Made this because i didnt add collate fn so kept getting error because not list of dictionaries
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

    image, target = smokeDataset(str(dataset_dir) + "/Train", transform_t)[0]
    check_for_strings_in_target(target)
    ### End check for strings, reduntant

    ### Looking at image using dataset

    import random

    import cv2
    from matplotlib import pyplot as plt

    import albumentations as A
    # If i write this like a proper british lad im gunna get errors o7
    BOX_COLOR = (255, 0, 0) # red
    TEXT_COLOR = (255, 255, 255) # white


    def visualize_bbox_pascal_voc(img, bbox, class_name, color=BOX_COLOR, thickness=2):
       # drawing bbox over image
        x_min, y_min, x_max, y_max = bbox
        x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
        cv2.putText(
            img,
            text=class_name,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=TEXT_COLOR,
            lineType=cv2.LINE_AA,
        )
        return img


    def visualize(image, bboxes, labels):
        img = image.permute(1, 2, 0).numpy() # convert tensor to numpy array
        #img = image.copy() # throws error

        # Check if bboxes exists (e.g. if zoomed in), prevents index out of bounds error
        # Prevents index out of bounds error from below if no bbox
        #if bboxes:
        #Convert to if bboxes to work witn tensors

        if bboxes.nelement() > 0:
          # If there's only one bounding box, convert it to a list
          # Will always happen in this dataset but keeping incase i want to use dif one with multiple
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


    # Test test dataset too
    test_test = smokeDataset(str(dataset_dir) + "/Test", transform_t) # create instance of dataset
    #image, bbox, label = test_test.__getitem__(1) # get image and annotation at index 1
    image, target = test_test.__getitem__(1) # get image and annotation at index 1
    bbox = target["boxes"]
    #label = target["labels"]
    label = "smoke"
    visualize(image, bbox, label)