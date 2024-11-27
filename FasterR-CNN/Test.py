### Testing model predictions
import Model
import torch
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from pathlib import Path
from Get_Values import checkColab
import os
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torch.utils.benchmark as benchmark
import time
print(checkColab())

"""
 TO DO!
 Individual metrics, show how much each image contirbuted to mAP
 more global test metrics, print total false positive, true positive, etc.
 Display only matplotlib of false positive or false negative, etc
 add checkpointing
"""
base_dir = checkColab()
# Define the confidence threshold, only bbox with score above val will be used
confidence_threshold = 0.5
# Draw only predicted bbox with highest scor
draw_highest_only = False
# plot images
plot_image = False
# Whether to use torch.utils.benchmark
BENCHMARK = False
# use small dataset
small_data = False

# device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model.getModel(True) # get model, true used to tell function we want to test

if (small_data):
    # Test directories
    test_image_dir = f"{base_dir}/Dataset/Small data/Test/images"
    test_annot_dir = f"{base_dir}/Dataset/Small data/Test/annotations/xmls"
else:
    # Test directories
    test_image_dir = f"{base_dir}/Dataset/Large data/Test/images"
    test_annot_dir = f"{base_dir}/Dataset/Large data/Test/annotations/xmls"

# Initialize mAP metric, intersection over union bbox
map_metricA = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.5])
map_metricB = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.3])
# create global array for benchmark times
benchmark_times = []

# transform to convert image to tensor before going through model
def get_transform():
    import torch
    from torchvision.transforms import v2 as T
    transforms = []
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

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
    # !!! GETTING PREDICTIONS !!!
    image = read_image(image_path) # get image
    eval_transform = get_transform()
    model.to(device) # put model on cpu or gpu
    # set model to evaluation mode
    model.eval()

    x = eval_transform(image)
    #print(x)
    # convert RGBA -> RGB and move to device
    x = x[:3, ...].to(device)

    if BENCHMARK == True:
        # start torch.utils.benchmark
        # will run the below code a second time to measure performance
        timer = benchmark.Timer(
            stmt="model([x, ])",  #specify code to be benchmarked
            globals={"x": x, "model": model}  #pass x and model to be used by benchmark
        )

        # record time taken
        time_taken = timer.timeit(5)  # run code n times, gives average = time taken / n
        print(f"Prediction time taken: {time_taken.mean:.4f} seconds")
        benchmark_times.append(time_taken.mean)

    with torch.no_grad():
        # Create predictions
        predictions, _ = model([x, ]) # val loss
        #predictions = model([x, ])
        #print(type(predictions))
        #print(predictions)
        #print(predictions)
        pred = predictions[0]
        #print(pred)


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
    map_metricA.update([predicted], [ground_truth])
    map_metricB.update([predicted], [ground_truth])
    # call function to display image with overlayed bboxes
    display_prediction(filtered_labels, filtered_boxes, filtered_scores, image, ground_truth)

def display_prediction(filtered_labels, filtered_boxes, filtered_scores, image, ground_truth):
    if (plot_image == False):
        matplotlib.use('Agg')
    #!!! DISPLAYING PREDICTIONS THROUGH MATPLOT PLIB !!!
    filtered_boxes_tensor = torch.stack(filtered_boxes) if filtered_boxes else torch.empty((0, 4), dtype=torch.long)
    # Get prediction with highest score and draw only that if draw_only_highest is true
    if filtered_scores and draw_highest_only:
        max_score_idx = filtered_scores.index(max(filtered_scores)) # get index pos of highest score
        highest_score_box = filtered_boxes_tensor[max_score_idx].unsqueeze(0)  # Add batch dimension
        highest_score_label = [filtered_labels[max_score_idx]]  # get highest value using index pos

        # Draw highest scoring bbox in red
        output_image = draw_bounding_boxes(image, highest_score_box, highest_score_label, colors="red")
    else:
        # Draw all predicted bbox in red
        output_image = draw_bounding_boxes(image, filtered_boxes_tensor, filtered_labels, colors="red")

    # Convert filtered boxes to a tensor
    filtered_boxes = torch.stack(filtered_boxes) if filtered_boxes else torch.empty((0, 4), dtype=torch.long)
    print("output labels", filtered_labels)

    # convert ground truth boxes to tensor
    ground_truth_boxes = [torch.tensor(bbox) for bbox in ground_truth['boxes']]

    # draw ground truth bbox over predicted bbox over image
    output_image = draw_bounding_boxes(output_image, torch.stack(ground_truth_boxes),
                                       ["Ground Truth"] * len(ground_truth_boxes), colors="blue")
    if (plot_image):
        time.sleep(2)
    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.axis('off')
    plt.show()


# function to calculate the average time to make a prediction
def getAvgTime(benchmark_times):
    total_time = sum(benchmark_times)  # get sum
    avg_time = total_time / len(benchmark_times)  # get average
    return avg_time

# Start python timer
start_time = time.time()

# Begin testing
test_dir(test_image_dir)

# End timer
end_time = time.time()
elapsed_time = end_time - start_time

# Get final mAP
final_mapA = map_metricA.compute()
final_mapB = map_metricB.compute()

# Only benchmark if true, benchmarking does extra runs through model
if BENCHMARK == True:
    print(f"Average benchmark time (per image): {getAvgTime(benchmark_times):.4f} seconds")
print(f"Mean Average Precision @ 0.5 (mAP@0.5): {final_mapA['map']:.4f}")
print(f"Mean Average Precision @ 0.3 (mAP@0.3): {final_mapB['map']:.4f}")
print(f"Elapsed time: {elapsed_time:.4f} seconds")