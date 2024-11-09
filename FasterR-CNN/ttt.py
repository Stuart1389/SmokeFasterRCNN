import torchvision
import os
import torch
from pathlib import Path

# Loading model

"""
Faster R-CNN/Regional convolutional neural network
Two stage detector = slower but more accurate compared to one stage (e.g. yolo)
Uses anchors, generates anchors on the image in a grid, compares to bounding box using IoU
IoU measures overlap between 2 bounding boxes

"""

num_classes = 2  # Dataset has only bounding boxes with label smoke + background = 2
# Transfer learning
model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
# Set in features to whatever region of interest(ROI) expects
in_features = model.roi_heads.box_predictor.cls_score.in_features
# Required to work with 2 classes
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

for param in model.rpn.parameters():
    param.requires_grad = True  # Unfreeze RPN layers

for param in model.roi_heads.parameters():
    param.requires_grad = True  # Unfreeze ROI heads

for param in model.backbone.parameters():
    param.requires_grad = True  # Unfreeze backbone layers (optional)