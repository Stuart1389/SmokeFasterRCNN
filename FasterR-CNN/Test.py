### Testing model predictions
import Model
import torch
import torch
from PIL import Image, ImageDraw
from pathlib import Path
import os

# transform to convert image to tensor before going through model
def get_transform():
    import torch
    from torchvision.transforms import v2 as T
    transforms = []
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

# device agnostic
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model.getModel(True)

test_image_dir = r"N:\University subjects\Thesis\Python projects\SmokeFasterRCNN\Dataset\Small data\Test\images"

def test_dir(dir_path):
  """Walks through dir_path, for each file call predictBbox and pass file path """
  for dirpath, dirnames, filenames in os.walk(dir_path): # Go through target directory and go through each sub-directory and print info
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
    for filename in filenames:
        image_path = os.path.join(dir_path, filename)
        print(image_path)
        predictBbox(image_path)

def predictBbox(image_path):
    image = read_image(image_path)
    eval_transform = get_transform()
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        print(predictions)
        pred = predictions[0]

        print(pred)

    # Define the confidence threshold, only bbox with score above val will be displayed
    confidence_threshold = 0.5

    # Normalize the image
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]

    # Filter predictions based on confidence score
    filtered_labels = []
    filtered_boxes = []
    for label, score, box in zip(pred["labels"], pred["scores"], pred["boxes"]):
        if score >= confidence_threshold:
            filtered_labels.append(f"Smoke: {score:.3f}")
            filtered_boxes.append(box.long())

    # Convert filtered boxes to a tensor
    filtered_boxes = torch.stack(filtered_boxes) if filtered_boxes else torch.empty((0, 4), dtype=torch.long)

    # Draw bounding boxes on the image
    output_image = draw_bounding_boxes(image, filtered_boxes, filtered_labels, colors="red")

    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

test_dir(test_image_dir)