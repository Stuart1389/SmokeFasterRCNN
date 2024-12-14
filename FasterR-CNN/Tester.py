### Testing model predictions
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.utils.benchmark as benchmark
import concurrent.futures
from GetValues import checkColab, setTestValues, setGlobalValues
from SmokeModel import SmokeModel
from tabulate import tabulate
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes


class Tester:
    #Constructor
    def __init__(self):

        # Initialising instanced variables
        self.base_dir = checkColab()
        # scores over this will be counted towards mAP/precission/recall and will be displayed if plot
        self.confidence_threshold = 0.5
        self.draw_highest_only = False # only draw bbox with highest score on plot
        self.plot_image = False # plot images
        self.benchmark = True # measure how long it takes to make average prediction
        self.ap_value = 0.5 # ap value for precision/recall e.g. if 0.5 then iou > 50% overlap = true positive
        self.draw_no_true_positive_only = False # only plot images with no true positives

        # device agnostic code
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # device agnostic
        self.batch_size = setTestValues("BATCH_SIZE")

        # initialise model
        smoke_model = SmokeModel()
        self.model = smoke_model.get_model(True)

        # get test dataloader
        _, _, self.test_dataloader = smoke_model.get_dataloader()

        # Paths
        self.test_image_dir = Path(f"{self.base_dir}/Dataset/") / setTestValues("dataset") / "Test/images"
        self.test_annot_dir = Path(f"{self.base_dir}/Dataset/") / setTestValues("dataset") / "Test/annotations/xmls"

        # Metrics
        self.initialise_metrics()

        # Benchmark times
        self.benchmark_times = []

        # Total time
        self.start_time = None

        # Precision/recall counters
        self.total_tp = {"small": 0, "medium": 0, "large": 0, "global": 0}
        self.total_fp = {"small": 0, "medium": 0, "large": 0, "global": 0}
        self.total_fn = {"small": 0, "medium": 0, "large": 0, "global": 0}

        # dictionary containing ground truth sizes
        self.image_gt_size = self.get_ground_truth_size(self.test_image_dir, self.test_annot_dir)
        #print(self.image_gt_size)

        self.cur_batch = None
        self.model_name = setTestValues("model_name")


    # !!START TESTING CHAIN!!
    # function starts testing images
    def test_dir(self):
        test_dataloader = self.test_dataloader # change to dataloader
        # Start timer
        self.start_time = time.time()
        # Walks through dir_path, for each file call predictBbox and pass file path
        for batch, (image_tensor, filename) in enumerate(test_dataloader):
            print(f"Processing batch {batch} out of {len(test_dataloader)}")
            self.get_predictions(image_tensor, filename)
        self.get_results()

    # Transform images for model
    def get_transform(self):
        transforms = []
        transforms.append(T.ToDtype(torch.float, scale=True))
        transforms.append(T.ToPureTensor())
        return T.Compose(transforms)

    # function to calculate the average time to make a prediction
    def get_avg_time(self, benchmark_times):
        total_time = sum(benchmark_times)  # get sum
        avg_time = total_time / len(benchmark_times)  # get average
        return avg_time

    # !!GETTING PREDICTION!!
    @torch.inference_mode()
    def get_predictions(self, image_tensor, filename):
        # getting predictions
        self.model.to(self.device, non_blocking=False)  # put model on cpu or gpu
        # set model to evaluation mode
        self.model.eval()
        # print("image:", image, "image_tensor", image_tensor, "filename", filename)
        filenames = list(files for files in filename)
        image_tensors = list(tensor.to(self.device, non_blocking=False) for tensor in image_tensor)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.no_grad():
            # benchmarking
            if self.benchmark == True:
                # start torch.utils.benchmark
                # will run the below code a second time to measure performance
                timer = benchmark.Timer(
                    stmt="model(image_tensors)",  # specify code to be benchmarked
                    globals={"image_tensors": image_tensors, "model": self.model}  # pass x and model to be used by benchmark
                )

                # record time taken
                time_taken = timer.timeit(3) # run code n times, gives average = time taken / n
                time_taken_ind = time_taken.mean / self.batch_size
                print(f"Prediction time taken: {time_taken_ind:.4f} seconds")
                self.benchmark_times.append(time_taken_ind)

            #outputs = self.model(image_tensors)
            outputs, _ = self.model(image_tensors)
            temp_index = 1
            # parallel processing after getting model predictions, might aswell
            futures = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for image_tensor, prediction, filename in zip(image_tensor, outputs, filenames):
                    #print(f"{temp_index}! - pred:", prediction, filename)
                    #temp_index += 1
                    futures.append(executor.submit(self.process_predictions, image_tensor, prediction, filename))
                    #self.process_predictions(image_tensor, prediction, filename)
            #print("outputs", outputs)
            concurrent.futures.wait(futures)

    def process_predictions(self, image_tensor, predictions, filename):
        global total_tp, total_fp, total_fn
        # Normalize the image
        #x = image_tensor
        image = (255.0 * (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())).to(torch.uint8)
        # Filter predictions based on confidence score
        filtered_labels = []
        filtered_boxes = []
        filtered_scores = []
        # Get prediction score, class and bbox if above threshold
        for label, score, box in zip(predictions["labels"], predictions["scores"], predictions["boxes"]):
            if score >= self.confidence_threshold:
                filtered_labels.append(f"Smoke: {score:.3f}")
                filtered_boxes.append(box.long())
                filtered_scores.append(score)

        # annotation path for ground truth using test_annot_dir
        # CHANGE image ID to filename
        image_id = filename # temp
        #print(image_id)
        annotation_path = os.path.join(self.test_annot_dir, f"{image_id}.xml")
        ground_truth = self.parse_xml(annotation_path)

        # format predictions for mAP calculation
        predicted = {
            "boxes": torch.stack(filtered_boxes) if filtered_boxes else torch.empty((0, 4), dtype=torch.float),
            "scores": torch.tensor(filtered_scores) if filtered_scores else torch.empty((0,)),
            "labels": torch.tensor([1] * len(filtered_boxes)) if filtered_boxes else torch.empty((0,),
                                                                                                 dtype=torch.long),
        }

        # getting ground truth size (small, medium, large)
        # filename stored with format in dict, might change dict later
        gt_size = self.image_gt_size.get(filename + ".jpeg")
        #print(gt_size)

        # getting tp, tn, fp, fn for each image
        metrics = self.get_image_val(
            pred_score=predicted["scores"].to("cpu"),  # cpu or cry
            pred_box=predicted["boxes"].to("cpu"),
            ground_truth=ground_truth,
            confidence_threshold=self.confidence_threshold,
            ap_value=self.ap_value,
            gt_size=gt_size
        )

        # Accumulate TP, FP, FN counts for the total
        print(f"{metrics}")
        for size in ["small", "medium", "large", "global"]:
            self.total_tp[size] += metrics["TP"][size]
            self.total_fp[size] += metrics["FP"][size]
            self.total_fn[size] += metrics["FN"][size]

        if gt_size == "small":
            self.map_metricSmallA.update([predicted], [ground_truth])
            self.map_metricSmallB.update([predicted], [ground_truth])
        elif gt_size == "medium":
            self.map_metricMediumA.update([predicted], [ground_truth])
            self.map_metricMediumB.update([predicted], [ground_truth])
        elif gt_size == "large":
            self.map_metricLargeA.update([predicted], [ground_truth])
            self.map_metricLargeB.update([predicted], [ground_truth])

        # mAP update
        self.map_metricGlobalA.update([predicted], [ground_truth])
        self.map_metricGlobalB.update([predicted], [ground_truth])

        if (self.draw_no_true_positive_only):
            if (metrics["TP"] == 0):
                # call function to display image with overlayed bboxes
                self.display_prediction(filtered_labels, filtered_boxes, filtered_scores, image, ground_truth, metrics)
        else:
            # call function to display image with overlayed bboxes
            self.display_prediction(filtered_labels, filtered_boxes, filtered_scores, image, ground_truth, metrics)

    # function for each image gets the number of true positive, false positive, etc
    def get_image_val(self, pred_score, pred_box, ground_truth, confidence_threshold, ap_value,
                      gt_size):
        # Filter predictions by confidence score
        filtered_boxes = []
        filtered_scores = []
        for score, box in zip(pred_score, pred_box):
            if score >= confidence_threshold:
                filtered_boxes.append(box)
                filtered_scores.append(score)

        # Initialize counters
        tp_count = {'small': 0, 'medium': 0, 'large': 0, 'global': 0}
        fp_count = {'small': 0, 'medium': 0, 'large': 0, 'global': 0}
        fn_count = {'small': 0, 'medium': 0, 'large': 0, 'global': 0}

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
                    tp_count['global'] += 1
                    if gt_size != 'none':  # Update size counter if not none
                        tp_count[gt_size] += 1
                    matched_gts.add(max_idx.item())
                else:
                    fp_count['global'] += 1
                    if gt_size != 'none':  # Update size counter if not none
                        fp_count[gt_size] += 1

            # Ground truth boxes not matched are FN
            fn_count['global'] = len(ground_truth_boxes) - len(matched_gts)
            if gt_size != 'none':  # Update size counter if not none
                fn_count[gt_size] = len(ground_truth_boxes) - len(matched_gts)
        else:
            # No predictions
            fp_count['global'] = len(predicted_boxes)
            if gt_size != 'none':  # Update size counter if not none
                fp_count[gt_size] = len(predicted_boxes)
            fn_count['global'] = len(ground_truth_boxes)
            if gt_size != 'none':  # Update size counter if not none
                fn_count[gt_size] = len(ground_truth_boxes)

        # Return dictionary with results
        return {
            "TP": tp_count,
            "FP": fp_count,
            "FN": fn_count
        }



    # !!GETTING GROUND TRUTH AND MAP VALUES AND PRECISION/RECALL
    # function initialising mAP metrics
    def initialise_metrics(self):
        # kind of ugly but changing this would just make things way more complicated for no gain
        self.map_metricGlobalA = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.5])
        self.map_metricGlobalB = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.3])
        self.map_metricSmallA = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.5])
        self.map_metricSmallB = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.3])
        self.map_metricMediumA = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.5])
        self.map_metricMediumB = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.3])
        self.map_metricLargeA = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.5])
        self.map_metricLargeB = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.3])
        self.map_metricLocal = MeanAveragePrecision(iou_type='bbox', iou_thresholds=[0.3])

    # Need ground truths to calculate mAP
    # Function parse xml for ground truths (copied from Dataset class)
    # and get area of ground truth to assign each a size (small, medium, large)
    def parse_xml(self, annotation_path, get_area=False):
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
            area = (xmax - xmin) * (ymax - ymin) # get area of ground truth
            if (get_area):
                areas.append(area)

        # Extract labels
        labels = []
        class_to_idx = {"smoke": 1}  # dictionary, if "smoke" return 1
        for obj in root.findall("object"):
            label = obj.find("name").text  # find name in xml
            labels.append(class_to_idx.get(label, 0))  # 0 if the class isn't found in dictionary

        # Convert boxes and labels to tensors for torchmetrics
        ground_truth = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        # Some images have more than 1 ground truth, in that case only use that images AP for global
        if get_area:
            if len(areas) == 1:
                return areas
            elif len(areas) > 1 or len(areas) == 0:
                return 0
        else:
            return ground_truth

    # function to assign each ground truth a size (small, medium or large)
    # Used to calculate size mAP's
    def get_ground_truth_size(self, test_image_dir, test_annot_dir):
        all_areas = []

        # Iterate through files in test image dir
        for root, dirs, files in os.walk(test_image_dir):
            for file_name in files:
                if file_name.endswith(".jpeg"):  # dont really need but whatever
                    # Get the corresponding XML annotation
                    get_annotation = os.path.join(test_annot_dir, file_name.replace(".jpeg", ".xml"))

                    # Parse XML to get areas (ground truth)
                    ground_truth_area = self.parse_xml(get_annotation, True)

                    # append the area to list for later
                    if ground_truth_area:
                        all_areas.append(ground_truth_area[0])  # append ground truth

        # sort areas in asc order
        sorted_areas = sorted(all_areas)

        # get boundaries for small, medium, and large sizes
        # 1/3 - small, 1/3 - medium, 1/3 - large | 33/33/33 split
        total_images = len(sorted_areas)
        small_limit = total_images // 3
        medium_limit = 2 * total_images // 3

        # Create dictionary for categorizing images by size
        categorized_images_dict = {}

        for root, dirs, files in os.walk(test_image_dir):
            for filename in files:
                if filename.endswith(".jpeg"): # dont really need but whatever
                    # Get corresponding XML annotation file
                    get_annotation = os.path.join(test_annot_dir, filename.replace(".jpeg", ".xml")) # image and annot have same names

                    # Parse XML to get areas (ground truth)
                    ground_truth_area = self.parse_xml(get_annotation, True)

                    # assigning each ground truth a size
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

                    # Store the size category in dictionary
                    categorized_images_dict[filename] = size_category

        # Count occurrences of each size
        size_counts = {"small": 0, "medium": 0, "large": 0, "none": 0}
        for size in categorized_images_dict.values():
            if size in size_counts:
                size_counts[size] += 1

        # Print count for each size
        print(f"Small: {size_counts['small']}")
        print(f"Medium: {size_counts['medium']}")
        print(f"Large: {size_counts['large']}")
        print(f"None: {size_counts['none']}")

        return categorized_images_dict

    # function to get precission and recall
    def calculate_total_precision_recall(self):
        # Precision and Recall for each size category
        precision_recall = {
            "small": {},
            "medium": {},
            "large": {},
            "global": {}
        }

        for size in ["small", "medium", "large", "global"]:
            tp = self.total_tp.get(size, 0)
            fp = self.total_fp.get(size, 0)
            fn = self.total_fn.get(size, 0)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            precision_recall[size]["precision"] = precision
            precision_recall[size]["recall"] = recall

        return precision_recall


    # !!VISUALISATIONS!!
    def display_prediction(self, filtered_labels, filtered_boxes, filtered_scores, image, ground_truth, metrics):
        if (self.plot_image):
                #matplotlib.use('Agg')
            # !!! DISPLAYING PREDICTIONS THROUGH MATPLOT LIB !!!
            filtered_boxes_tensor = torch.stack(filtered_boxes) if filtered_boxes else torch.empty((0, 4), dtype=torch.long)
            # Get prediction with highest score and draw only that if draw_only_highest is true
            if filtered_scores and self.draw_highest_only:
                max_score_idx = filtered_scores.index(max(filtered_scores))  # get index pos of highest score
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

            # registry increased ide.rest.api.request.per.minute to 100
            if (self.plot_image):
                time.sleep(2)
            plt.figure(figsize=(12, 12))
            plt.imshow(output_image.permute(1, 2, 0))
            plt.axis('off')
            plt.show()
            matplotlib.pyplot.close()

    # starting chain to display results
    def get_results(self):
        if self.benchmark:
            avg_benchmark = str(round(self.get_avg_time(self.benchmark_times), 2)) + " seconds"
        else:
            avg_benchmark = "N/A"

        # End timer
        end_time = time.time()
        elapsed_time = round(end_time - self.start_time, 2)

        # Get final mAP, sync and compute
        self.map_metricSmallA.sync()
        self.final_mapSmallA = self.map_metricSmallA.compute()
        self.map_metricSmallB.sync()
        self.final_mapSmallB = self.map_metricSmallB.compute()
        self.map_metricMediumA.sync()
        self.final_mapMediumA = self.map_metricMediumA.compute()
        self.map_metricMediumB.sync()
        self.final_mapMediumB = self.map_metricMediumB.compute()
        self.map_metricLargeA.sync()
        self.final_mapLargeA = self.map_metricLargeA.compute()
        self.map_metricLargeB.sync()
        self.final_mapLargeB = self.map_metricLargeB.compute()
        self.map_metricGlobalA.sync()
        self.final_mapA = self.map_metricGlobalA.compute()
        self.map_metricGlobalB.sync()
        self.final_mapB = self.map_metricGlobalB.compute()
        # get precission and recall
        precision_recall = self.calculate_total_precision_recall()
        # get max vram
        max_vram = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
        torch.cuda.reset_peak_memory_stats()  # reset

        # start displaying
        self.nice_layout(precision_recall, elapsed_time, avg_benchmark, max_vram)
        self.ugly_layout(precision_recall, elapsed_time, avg_benchmark, max_vram)

    # !!DISPLAYING RESULTS!!
    # function for nice results
    def nice_layout(self, precision_recall, elapsed_time, avg_benchmark, max_vram):
        # colour codes
        COLOR_SMALL = "\033[94m"  # Blue
        COLOR_MEDIUM = "\033[92m"  # Green
        COLOR_LARGE = "\033[93m"  # Yellow
        COLOR_GLOBAL = "\033[91m"  # Red
        RESET = "\033[0m"  # Reset to default
        BOLD = "\033[1m"  # Bold text

        # total time and vram
        elapsed_time_data = [[f"{BOLD}Global Time{RESET}", f"{elapsed_time:} seconds"]]

        elapsed_time_data = [
            [f"{BOLD}Global Time{RESET}", f"{elapsed_time:} seconds"],
            [f"{BOLD}Max VRAM{RESET}", f"{max_vram:.2f} MB"],
            [f"{BOLD}Benchmark Time (per image){RESET}", f"{avg_benchmark}"]
        ]

        print(f"\n{BOLD}Time and VRAM:{RESET}")
        print(tabulate(elapsed_time_data, headers=[f"{BOLD}Metric{RESET}", f"{BOLD}Value{RESET}"],
                       tablefmt="fancy_grid"))

        # nice table
        combined_data = [
            [f"{BOLD}Precision{RESET}",
             f"{precision_recall['global']['precision'] * 100:.2f}%",
             f"{precision_recall['small']['precision'] * 100:.2f}%",
             f"{precision_recall['medium']['precision'] * 100:.2f}%",
             f"{precision_recall['large']['precision'] * 100:.2f}%"
             ],

            [f"{BOLD}Recall{RESET}",
             f"{precision_recall['global']['recall'] * 100:.2f}%",
             f"{precision_recall['small']['recall'] * 100:.2f}%",
             f"{precision_recall['medium']['recall'] * 100:.2f}%",
             f"{precision_recall['large']['recall'] * 100:.2f}%"
             ],

            [f"{BOLD}mAP @ 0.5{RESET}",
             f"{self.final_mapA['map'] * 100:.2f}%",
             f"{self.final_mapSmallA['map'] * 100:.2f}%",
             f"{self.final_mapMediumA['map'] * 100:.2f}%",
             f"{self.final_mapLargeA['map'] * 100:.2f}%"
             ],

            [f"{BOLD}mAP @ 0.3{RESET}",
             f"{self.final_mapB['map'] * 100:.2f}%",
             f"{self.final_mapSmallB['map'] * 100:.2f}%",
             f"{self.final_mapMediumB['map'] * 100:.2f}%",
             f"{self.final_mapLargeB['map'] * 100:.2f}%"
             ]
        ]

        # print table
        print(f"{BOLD}Precision, Recall, and mAP Values:{RESET}")
        print(tabulate(combined_data,
                       headers=[f"{BOLD}Metric{RESET}", f"{COLOR_GLOBAL}Global{RESET}",
                                f"{COLOR_SMALL}Small{RESET}", f"{COLOR_MEDIUM}Medium{RESET}",
                                f"{COLOR_LARGE}Large{RESET}"], tablefmt="fancy_grid"))

    def ugly_layout(self, precision_recall, elapsed_time, avg_benchmark, max_vram):
        # Ugly output for copy and paste
        print("\nUgly output for excel sheet copy/paste")
        output_data = [
            [   f"\"{self.model_name}\"",
                f"\"Precision: {precision_recall['global']['precision'] * 100:.2f}%\nRecall: {precision_recall['global']['recall'] * 100:.2f}%\nmAP @0.5: {self.final_mapA['map'] * 100:.2f}%\nmAP @0.3: {self.final_mapB['map'] * 100:.2f}%\"",
                f"\"Precision: {precision_recall['small']['precision'] * 100:.2f}%\nRecall: {precision_recall['small']['recall'] * 100:.2f}%\nmAP @0.5: {self.final_mapSmallA['map'] * 100:.2f}%\nmAP @0.3: {self.final_mapSmallB['map'] * 100:.2f}%\"",
                f"\"Precision: {precision_recall['medium']['precision'] * 100:.2f}%\nRecall: {precision_recall['medium']['recall'] * 100:.2f}%\nmAP @0.5: {self.final_mapMediumA['map'] * 100:.2f}%\nmAP @0.3: {self.final_mapMediumB['map'] * 100:.2f}%\"",
                f"\"Precision: {precision_recall['large']['precision'] * 100:.2f}%\nRecall: {precision_recall['large']['recall'] * 100:.2f}%\nmAP @0.5: {self.final_mapLargeA['map'] * 100:.2f}%\nmAP @0.3: {self.final_mapLargeB['map'] * 100:.2f}%\"",
                f"\"{max_vram}\"",
                f"\"{elapsed_time}\"",
                f"\"{avg_benchmark}\"",
                f"\"{self.batch_size}\""
            ]
        ]

        # headers
        headers = ["Model_name", "Global", "Small", "Medium", "Large", "Max size in vram (MB)", "Total time (seconds)",
                   "Benchmark (seconds), Batch Size"]

        # all this was tryna get excel to paste properly
        output = "\t".join(headers) + "\n"
        for row in output_data:
            output += "\t".join(row) + "\n"
        print(output.strip())


def main():
    # create instance of class
    tester = Tester()
    # start testing chain
    tester.test_dir()



if __name__ == '__main__':
    main()








