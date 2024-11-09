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