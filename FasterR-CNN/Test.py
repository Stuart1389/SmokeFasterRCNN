### Testing model predictions
import torch
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from pathlib import Path
from Get_Values import checkColab, setTestValues
import os
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torch.utils.benchmark as benchmark
from torchvision.ops import box_iou
import time
from SmokeModel import SmokeModel
print(checkColab())

base_dir = checkColab()
# Define the confidence threshold, only bbox with score above val will be used
confidence_threshold = 0.5
# Draw only predicted bbox with highest score
draw_highest_only = False
# plot images
plot_image = False
# Whether to use torch.utils.benchmark
BENCHMARK = False
# use small dataset
small_data = False
ap_value = 0.5 # percentage of overlap necessary to count a bbox prediction as true positive
# Show images which have 0 true positive predictions
draw_no_true_positive_only = False

# device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

smoke_model = SmokeModel() # create instance of smoke model class

model = smoke_model.get_model(True) # get model, true used to tell function we are testing

# Test directories
test_image_dir = Path(f"{base_dir}/Dataset/") / setTestValues("dataset") / "Test/images"
test_annot_dir = Path(f"{base_dir}/Dataset/") / setTestValues("dataset") / "Test/annotations/xmls"


# Initialize mAP metric, intersection over union bbox
map_metricA = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.5])
map_metricB = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.3])

map_metricSmallA = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.5])
map_metricSmallB = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.3])

map_metricMediumA = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.5])
map_metricMediumB = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.3])

map_metricLargeA = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.5])
map_metricLargeB = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.3])

# create global array for benchmark times
benchmark_times = []
# Creating counters for precision/recall
total_tp = 0
total_fp = 0
total_fn = 0

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
def parse_xml(annotation_path, get_area=False):
    # Parsing XML file for each image to find bounding boxes
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # Extract bounding boxes
    boxes = []
    areas = []
    for obj in root.findall("object"):
        xml_box = obj.find("bndbox")
        xmin = float(xml_box.find("xmin").text)
        ymin = float(xml_box.find("ymin").text)
        xmax = float(xml_box.find("xmax").text)
        ymax = float(xml_box.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
        area = (xmax - xmin) * (ymax - ymin)
        if(get_area):
            areas.append(area)

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

    if get_area:
        if len(areas) == 1:
            return areas
        elif len(areas) > 1 or len(areas) == 0:
            return 0
    else:
        return ground_truth

def test_dir(dir_path):
  """Walks through dir_path, for each file call predictBbox and pass file path """
  for dirpath, dirnames, filenames in os.walk(dir_path): # Go through target directory and go through each sub-directory and print info
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
    for filename in filenames:
        image_path = os.path.join(dir_path, filename)
        print(image_path)
        predictBbox(image_path, filename)

# Function to predict smoke and calulcate mAP
def predictBbox(image_path, filename):
    global total_tp, total_fp, total_fn # this makes me feel sick
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

    # get image size
    image_size = predictions[0].get("image_size", [image.shape[1], image.shape[2]])

    # format predictions for mAP calculation
    predicted = {
        "boxes": torch.stack(filtered_boxes) if filtered_boxes else torch.empty((0, 4), dtype=torch.float),
        "scores": torch.tensor(filtered_scores) if filtered_scores else torch.empty((0,)),
        "labels": torch.tensor([1] * len(filtered_boxes)) if filtered_boxes else torch.empty((0,), dtype=torch.long),
    }

    print(filename)
    gt_size = image_gt_size.get(filename)
    print(gt_size)

    # getting tp, tn, fp, fn
    metrics = getImageVal(
        image_path=image_path,
        pred_score=predicted["scores"].to("cpu"), # cpu or cry
        pred_box=predicted["boxes"].to("cpu"),
        ground_truth=ground_truth,
        confidence_threshold=confidence_threshold,
        image_size = image_size,
        ap_value = ap_value,
        gt_size = gt_size
    )
    # Accumulate TP, FP, FN counts for the total
    total_tp += metrics["TP"]
    total_fp += metrics["FP"]
    total_fn += metrics["FN"]
    print(f"{metrics}")

    if gt_size == "small":
        map_metricSmallA.update([predicted], [ground_truth])
        map_metricSmallB.update([predicted], [ground_truth])
    elif gt_size == "medium":
        map_metricMediumA.update([predicted], [ground_truth])
        map_metricMediumB.update([predicted], [ground_truth])
    elif gt_size == "large":
        map_metricLargeA.update([predicted], [ground_truth])
        map_metricLargeB.update([predicted], [ground_truth])


    # mAP update
    map_metricA.update([predicted], [ground_truth])
    map_metricB.update([predicted], [ground_truth])




    if(draw_no_true_positive_only):
        if(metrics["TP"] == 0):
            # call function to display image with overlayed bboxes
            display_prediction(filtered_labels, filtered_boxes, filtered_scores, image, ground_truth)
    else:
        # call function to display image with overlayed bboxes
        display_prediction(filtered_labels, filtered_boxes, filtered_scores, image, ground_truth)


def getImageVal(image_path, pred_score, pred_box, ground_truth, confidence_threshold, image_size, ap_value, gt_size):
    # Filter predictions by confidence score
    filtered_boxes = []
    filtered_scores = []
    for score, box in zip(pred_score, pred_box):
        if score >= confidence_threshold:
            filtered_boxes.append(box)
            filtered_scores.append(score)

    # Initialize counters
    tp_count, fp_count, fn_count, tn_count = 0, 0, 0, 0

    # Extract ground truth boxes
    ground_truth_boxes = ground_truth['boxes']
    predicted_boxes = torch.stack(filtered_boxes) if filtered_boxes else torch.empty((0, 4), dtype=torch.float)

    # Calculate TP, FP, FN
    if len(predicted_boxes) > 0 and len(ground_truth_boxes) > 0:
        iou_matrix = box_iou(predicted_boxes, ground_truth_boxes)
        matched_gts = set()

        for i, pred_box in enumerate(predicted_boxes):
            max_iou, max_idx = iou_matrix[i].max(0)
            if max_iou >= ap_value:
                tp_count += 1
                matched_gts.add(max_idx.item())
            else:
                fp_count += 1

        # Ground truth boxes not matched are FN
        fn_count = len(ground_truth_boxes) - len(matched_gts)
    else:
        fp_count = len(predicted_boxes)
        fn_count = len(ground_truth_boxes)

    # Return a dictionary with results
    return {
        "image": image_path,
        "TP": tp_count,
        "FP": fp_count,
        "FN": fn_count
    }




def display_prediction(filtered_labels, filtered_boxes, filtered_scores, image, ground_truth):
    if (plot_image == False):
        matplotlib.use('Agg')
    #!!! DISPLAYING PREDICTIONS THROUGH MATPLOT LIB !!!
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
    ground_truth_boxes = [bbox.clone().detach() for bbox in ground_truth['boxes']]

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

# function to get precission and recall
def calculate_total_precision_recall():
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    return total_precision, total_recall


def get_ground_truth_size(test_image_dir, test_annot_dir):
    all_areas = []

    # Step 1: Iterate through the files in the test image directory
    for root, dirs, files in os.walk(test_image_dir):
        for file_name in files:
            if file_name.endswith(".jpeg"):  # Adjust for image types
                # Get the corresponding XML annotation file
                get_annotation = os.path.join(test_annot_dir, file_name.replace(".jpeg", ".xml"))

                # Parse XML to get areas (ground truth)
                ground_truth_area = parse_xml(get_annotation, True)

                # Append the area to the list for later processing
                if ground_truth_area:
                    all_areas.append(ground_truth_area[0])  # Assuming there's only one area per image

    # Step 2: Sort the areas in ascending order
    sorted_areas = sorted(all_areas)

    # Step 3: Compute boundaries for small, medium, and large sizes
    total_images = len(sorted_areas)
    small_limit = total_images // 3
    medium_limit = 2 * total_images // 3

    # Step 4: Create a dictionary for categorizing images by size
    categorized_images_dict = {}

    for root, dirs, files in os.walk(test_image_dir):
        for file_name in files:
            if file_name.endswith(".jpeg"):  # Adjust for image types
                # Get the corresponding XML annotation file
                get_annotation = os.path.join(test_annot_dir, file_name.replace(".jpeg", ".xml"))

                # Parse XML to get areas (ground truth)
                ground_truth_area = parse_xml(get_annotation, True)

                if ground_truth_area:
                    area = ground_truth_area[0]
                    if area == 0:
                        size_category = "none"
                    elif area <= sorted_areas[small_limit - 1]:
                        size_category = "small"
                    elif area <= sorted_areas[medium_limit - 1]:
                        size_category = "medium"
                    else:
                        size_category = "large"
                else:
                    size_category = "none"

                # Store the size category in the dictionary
                categorized_images_dict[file_name] = size_category

    # Step 5: Count the occurrences of each category
    size_counts = {"small": 0, "medium": 0, "large": 0, "none": 0}
    for size in categorized_images_dict.values():
        if size in size_counts:
            size_counts[size] += 1

    # Print the counts
    print(f"Small: {size_counts['small']}")
    print(f"Medium: {size_counts['medium']}")
    print(f"Large: {size_counts['large']}")
    print(f"None: {size_counts['none']}")

    return categorized_images_dict



image_gt_size = get_ground_truth_size(test_image_dir, test_annot_dir)
print(image_gt_size)

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

# get precission and recall
total_precision, total_recall = calculate_total_precision_recall()
print(f"Global Precision: {total_precision:.4f}")
print(f"Global Recall: {total_recall:.4f}")

# Only benchmark if true, benchmarking does extra run through model for each pred
if BENCHMARK == True:
    print(f"Global Average benchmark Time (per image): {getAvgTime(benchmark_times):.4f} seconds")

# Printing mAP values
print(f"Global Mean Average Precision @ 0.5 (mAP@0.5): {final_mapA['map']:.4f}")
print(f"Global Mean Average Precision @ 0.3 (mAP@0.3): {final_mapB['map']:.4f}")

print(f"Small Size mAP @ 0.5 (mAP@0.5): {map_metricSmallA.compute()['map']:.4f}")
print(f"Small Size mAP @ 0.3 (mAP@0.3): {map_metricSmallB.compute()['map']:.4f}")

print(f"Medium Size mAP @ 0.5 (mAP@0.5): {map_metricMediumA.compute()['map']:.4f}")
print(f"Medium Size mAP @ 0.3 (mAP@0.3): {map_metricMediumB.compute()['map']:.4f}")

print(f"Large Size mAP @ 0.5 (mAP@0.5): {map_metricLargeA.compute()['map']:.4f}")
print(f"Large Size mAP @ 0.3 (mAP@0.3): {map_metricLargeB.compute()['map']:.4f}")

print(f"Global time: {elapsed_time:.4f} seconds")