"""
Settings -> Project structure -> set as source file and you dont need this anymore

"""

def getFunc():
    import os
    import urllib.request

    urls = [
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py",
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py",
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py",
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py",
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py"
    ]

    for url in urls:
        file_name = url.split("/")[-1]
        urllib.request.urlretrieve(url, file_name)
        print(f"{file_name} has been downloaded.")

    ### Set system path to include Libr
    import sys
    from pathlib import Path

    # Specify the path to the folder where your .py files are stored
    module_path = Path(r"N:\University subjects\Thesis\Python projects\SmokeFasterRCNN\Libr")

    # Add the folder to sys.path
    sys.path.append(str(module_path))

    # Specify the path to the folder where your .py files are stored
    module_path = Path(r"N:\University subjects\Thesis\Python projects\SmokeFasterRCNN\FasterR-CNN")

    # Add the folder to sys.path
    sys.path.append(str(module_path))


### Old test
image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
image = image[:3, ...]
pred_labels = [f"Smoke: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
pred_boxes = pred["boxes"].long()
output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0))
plt.show()



### Test transform
def get_transform(train):
    import torch
    from torchvision.transforms import v2 as T
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

###test
def predictBbox():
    image = read_image(
        r"N:\University subjects\Thesis\Python projects\SmokeFasterRCNN\Dataset\Small data\Test\images\ckagz7s5solbc0841r1aklq1g.jpeg")
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

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

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
                filtered_values = self.remove_outliers(values)
                axs[idx].scatter(range(len(filtered_values)), filtered_values, color=colors[area], label=area)

            axs[idx].set_title(title)
            axs[idx].set_xlabel('Index')
            axs[idx].set_ylabel('Area')
            axs[idx].legend()

            # Create inset
            axins = inset_axes(axs[idx], width="30%", height="30%", loc="upper left")  # Adjust size and position
            for area in ["small_vals", "medium_vals"]:  # Only plot small and medium values
                if area in area_dict:
                    filtered_values = self.remove_outliers(area_dict[area])
                    axins.scatter(range(len(filtered_values)), filtered_values, color=colors[area], label=area)

            axins.set_xlim(0, max(len(area_dict["small_vals"]), len(area_dict["medium_vals"])))  # Adjust x-axis
            axins.set_ylim(min(min(area_dict["small_vals"], default=0), min(area_dict["medium_vals"], default=0)),
                           max(max(area_dict["small_vals"], default=1),
                               max(area_dict["medium_vals"], default=1)))  # Adjust y-axis

            axins.set_xticks([])
            axins.set_yticks([])
            axins.set_title("Zoomed Inset", fontsize=10)

            mark_inset(axs[idx], axins, loc1=2, loc2=4, fc="none", ec="black")  # Draw connecting lines

        plt.tight_layout()
        plt.show()
