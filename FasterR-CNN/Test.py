### Testing model predictions
import Model
import torch
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from pathlib import Path
import os
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

# transform to convert image to tensor before going through model
def get_transform():
    import torch
    from torchvision.transforms import v2 as T
    transforms = []
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

# device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model.getModel(True) # get model, true used to tell function we want to test

# Test directories
test_image_dir = r"N:\University subjects\Thesis\Python projects\SmokeFasterRCNN\Dataset\Small data\Test\images"
test_annot_dir = r"N:\University subjects\Thesis\Python projects\SmokeFasterRCNN\Dataset\Small data\Test\annotations\xmls"

# Initialize mAP metric, intersection over union bbox
map_metric = MeanAveragePrecision(iou_type='bbox')

# Need ground truths to calculate mAP
# Function parse xml for ground truths (copied from Dataset class)
def parse_xml(annotation_path):
    # Parsing XML file for each image to find bounding boxes
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # Extract bounding boxes
    boxes = []
    for obj in root.findall("object"):
        xml_box = obj.find("bndbox")
        xmin = float(xml_box.find("xmin").text)
        ymin = float(xml_box.find("ymin").text)
        xmax = float(xml_box.find("xmax").text)
        ymax = float(xml_box.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])

    # Extract labels
    labels = []
    class_to_idx = {"smoke": 1}  # dictionary, if "smoke" return 1
    for obj in root.findall("object"):
        label = obj.find("name").text # find name in xml
        labels.append(class_to_idx.get(label, 0))  # 0 if the class isn't found in dictionary

    # Convert boxes and labels to tensors for torchmetrics
    ground_truth = {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }

    return ground_truth

def test_dir(dir_path):
  """Walks through dir_path, for each file call predictBbox and pass file path """
  for dirpath, dirnames, filenames in os.walk(dir_path): # Go through target directory and go through each sub-directory and print info
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
    for filename in filenames:
        image_path = os.path.join(dir_path, filename)
        print(image_path)
        predictBbox(image_path)

# Function to predict smoke and calulcate mAP
def predictBbox(image_path):
    image = read_image(image_path) # get image
    eval_transform = get_transform()
    model.to(device) # put model on cpu or gpu
    # set model to evaluation mode
    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        # Create predictions
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
    filtered_scores = []
    # Get prediction score, class and bbox if above threshold
    for label, score, box in zip(pred["labels"], pred["scores"], pred["boxes"]):
        if score >= confidence_threshold:
            filtered_labels.append(f"Smoke: {score:.3f}")
            #filtered_labels.append(label)
            filtered_boxes.append(box.long())
            filtered_scores.append(score)

    # annotation path for ground truth using test_annot_dir
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    annotation_path = os.path.join(test_annot_dir, f"{image_id}.xml")
    ground_truth = parse_xml(annotation_path)

    # format predictions for mAP calculation
    predicted = {
        "boxes": torch.stack(filtered_boxes) if filtered_boxes else torch.empty((0, 4), dtype=torch.float),
        "scores": torch.tensor(filtered_scores) if filtered_scores else torch.empty((0,)),
        "labels": torch.tensor([1] * len(filtered_boxes)) if filtered_boxes else torch.empty((0,), dtype=torch.long),
    }

    map_metric.update([predicted], [ground_truth])


    # Convert filtered boxes to a tensor
    filtered_boxes = torch.stack(filtered_boxes) if filtered_boxes else torch.empty((0, 4), dtype=torch.long)
    print("output labels", filtered_labels)

    # Draw bounding boxes over the image
    output_image = draw_bounding_boxes(image, filtered_boxes, filtered_labels, colors="red")

    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

test_dir(test_image_dir)
final_map = map_metric.compute()
print(f"Mean Average Precision (mAP): {final_map['map']}")