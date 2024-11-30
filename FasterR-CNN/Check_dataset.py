import os
import random
import torch
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from Get_Values import checkColab

# !!Checking bounding box and dataset!!
# This was the first thing i made just to check out dataset, don't really need anymore

### DATASET: https://github.com/aiformankind/wildfire-smoke-dataset/blob/master/README.md
base_dir = checkColab()
print(torch.__version__)

# Setting directories for images and accompanying annotations/xml
image_path = Path(f"{base_dir}/Dataset/Large data/Train/images")
annotation_path = Path(f"{base_dir}/Dataset/Large data/Train/annotations/xmls")

def walk_through_dir(dir_path):
  """walk through dir path and return its contents """
  for dirpath, dirnames, filenames in os.walk(dir_path): # Go through target directory and go through each sub directory and print info
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

print(walk_through_dir(image_path))

# Set image path
image_path_list = list(image_path.glob("*.jpeg"))  # glob together/stick together all the files that follow a pattern. anything/anything/anything.jpeg. get all images

get_randy_image = random.choice(image_path_list)
file_name = get_randy_image.stem
print(file_name)
get_randy_annotation = str(annotation_path) + "/" + str(file_name) + ".xml"
print(get_randy_annotation)
print(get_randy_image)

img = Image.open(get_randy_image)

# Print metadata
print(f"Width: {img.width}")
print(f"Height: {img.height}")
print(f"Image format: {img.format}")

# Parse the annotation XML
tree = ET.parse(get_randy_annotation)
root = tree.getroot()

# Draw the bounding boxes
draw = ImageDraw.Draw(img)
for obj in root.findall('object'):
    bndbox = obj.find('bndbox')
    xmin = int(float(bndbox.find('xmin').text))
    ymin = int(float(bndbox.find('ymin').text))
    xmax = int(float(bndbox.find('xmax').text))
    ymax = int(float(bndbox.find('ymax').text))

    # Draw the bounding box on the image
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)

# Display the image with matplotlib
plt.imshow(img)
plt.axis('off')
plt.show()