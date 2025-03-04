from Tester import Tester
from pathlib import Path
from GetValues import checkColab, setTrainValues
import os
import xml.etree.ElementTree as ET
import torch
import numpy as np
import shutil
import matplotlib.pyplot as plt
from SmokeUtils import extract_boxes

# class used to split images into small, medium and large based on ground truth area, see README for more info
class SplitImage():
    def __init__(self):
        # Setting dir of dataset to split
        self.base_dir = checkColab()
        #dataset_name = "Example dataset structure"  # set this to dataset name, see README
        dataset_name = "Large Data"

        self.split_images = False # whether to actually split the images in drive
        self.plot_no_large = True # dont plot large ground truths in scatter plot
        self.outlier_remove = True # whether to remove outliers

        # Creating lists to hold filenames for each split/size
        self.train_areas = {
            "small_vals": [],
            "medium_vals": [],
            "large_vals": [],
        }
        self.val_areas = {
            "small_vals": [],
            "medium_vals": [],
            "large_vals": [],
        }
        self.test_areas = {
            "small_vals": [],
            "medium_vals": [],
            "large_vals": [],
        }

        self.main_dir = Path(self.base_dir) / "Dataset" / dataset_name
        self.train_path = self.main_dir / "Train"
        self.val_path = self.main_dir / "Validate"
        self.test_path = self.main_dir / "Test"

        # methods split images
        self.subclass_images(self.train_path, self.train_areas)
        self.subclass_images(self.val_path, self.val_areas)
        self.subclass_images(self.test_path, self.test_areas)

        print(self.train_areas)
        print(self.val_areas)
        print(self.test_areas)

        # method to plot scatter of size distribution
        self.plot_scatter(self.train_areas, self.val_areas, self.test_areas)




    # method splits images into size and moves them to destination
    def subclass_images(self, dir, dict):
        image_dir = Path(str(dir) + "/images/")
        anot_dir = Path(str(dir) + "/annotations/xmls")
        # creating folder in directory for each size
        if(self.split_images):
            self.create_size_folders(image_dir, anot_dir)

        # This method is used to determine ground_truth area sizes
        self.image_gt_size = self.get_ground_truth_size(image_dir, anot_dir, dict)

        # move each image to new directory
        for image_path in image_dir.glob("*.jpeg"):
            filename = image_path.name
            anot_name = image_path.stem + ".xml"
            anot_path = anot_dir / anot_name

            size = self.image_gt_size.get(filename, "none")
            print(filename, size)

            size_dir_image = image_dir / size
            size_dir_anot = anot_dir / size
            if(self.split_images):
                shutil.move(str(image_path), size_dir_image / filename)
                shutil.move(str(anot_path), size_dir_anot / anot_name)

    # method plot scatter plot of size distributions
    def plot_scatter(self, train_area_dict, val_area_dict, test_area_dict):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Create a 1x3 grid of subplots

        colors = {
            "small_vals": 'blue',
            "medium_vals": 'orange',
            "large_vals": 'red'
        }

        for idx, (area_dict, title) in enumerate(zip([train_area_dict, val_area_dict, test_area_dict],
                                                     ['Train Area Scatter', 'Validation Area Scatter',
                                                      'Test Area Scatter'])):

            for area, values in area_dict.items():
                if(self.plot_no_large):
                    if area == "large_vals":
                        continue
                    filtered_values = values

                axs[idx].scatter(range(len(filtered_values)), filtered_values, color=colors[area], label=area)
            axs[idx].set_title(title)
            axs[idx].set_xlabel('Index')
            axs[idx].set_ylabel('Area')
            axs[idx].legend(loc="upper right")

        plt.tight_layout()
        plt.show()

    #method creates size folders
    def create_size_folders(self, image_dir, annotation_dir):
        sizes = ['small', 'medium', 'large', 'none']

        for size in sizes:
            (image_dir / size).mkdir(parents=True, exist_ok=True)
            (annotation_dir / size).mkdir(parents=True, exist_ok=True)

    # method to find area of ground truths, copied from Tester.py for simplisity
    def get_ground_truth_size(self, test_image_dir, test_annot_dir, area_dict):
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
                if filename.endswith(".jpeg"):  # dont really need but whatever
                    # Get corresponding XML annotation file
                    get_annotation = os.path.join(test_annot_dir, filename.replace(".jpeg",
                                                                                   ".xml"))  # image and annot have same names

                    # Parse XML to get areas (ground truth)
                    ground_truth_area = self.parse_xml(get_annotation, True)

                    # assigning each ground truth a size
                    if ground_truth_area:
                        area = ground_truth_area[0]
                        if area == 0:
                            size_category = "none"
                        elif area <= sorted_areas[small_limit - 1]:
                            size_category = "small"
                            area_dict["small_vals"].append(area)
                        elif area <= sorted_areas[medium_limit - 1]:
                            size_category = "medium"
                            area_dict["medium_vals"].append(area)
                        else:
                            size_category = "large"
                            area_dict["large_vals"].append(area)
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

        # Print area metrics
        # remove outliers
        q1 = np.percentile(all_areas, 25)
        q3 = np.percentile(all_areas, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_areas = [area for area in all_areas if lower_bound <= area <= upper_bound]

        # get min max and avg
        max_area = max(filtered_areas)
        min_area = min(filtered_areas)
        avg_area = sum(filtered_areas) / len(filtered_areas)

        print(f"Max Area (25th to 75th percentile): {max_area}")
        print(f"Min Area (25th to 75th percentile): {min_area}")
        print(f"Avg Area (25th to 75th percentile): {avg_area:.2f}")

        return categorized_images_dict

    # method to get area from ground truth bboxes
    def parse_xml(self, annotation_path, get_area=False):
        boxes, areas, labels_int, _ = (
            extract_boxes(annotation_path, get_area))
        # Convert boxes and labels to tensors for torchmetrics
        ground_truth = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels_int, dtype=torch.int64),
        }

        # Some images have more than 1 ground truth, in that case only use that images AP for global
        if get_area:
            if len(areas) == 1:
                return areas
            elif len(areas) > 1 or len(areas) == 0:
                return 0
        else:
            return ground_truth



if __name__ == '__main__':
    split_image = SplitImage()