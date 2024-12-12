import os
import sys
current_dir = os.getcwd()
# add libr as source
relative_path = os.path.join(current_dir, '..', 'Libr')
sys.path.append(relative_path)

import math
import time
import matplotlib.pyplot as plt
import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

def train_step(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    iteration_loss_list = [] # loss graph iteration instead of epoch
    # Init loss values
    total_loss = 0
    total_loss_classifier = 0
    total_loss_box_reg = 0
    total_loss_objectness = 0
    total_loss_rpn_box_reg = 0
    num_batches = len(data_loader)


    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device, non_blocking=False) for image in images)
        targets = [{k: v.to(device, non_blocking=False) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item() # this batch
        # Getting loss values for iterations
        iteration_loss_list.append({
            'total_loss': loss_value,
            'loss_classifier': loss_dict_reduced['loss_classifier'].item(),
            'loss_box_reg': loss_dict_reduced['loss_box_reg'].item(),
            'loss_objectness': loss_dict_reduced['loss_objectness'].item(),
            'loss_rpn_box_reg': loss_dict_reduced['loss_rpn_box_reg'].item()
        })

        # gettting individual loss from dict for average epoch graphs
        total_loss += losses_reduced.item() # epoch
        total_loss_classifier += loss_dict_reduced['loss_classifier'].item()
        total_loss_box_reg += loss_dict_reduced['loss_box_reg'].item()
        total_loss_objectness += loss_dict_reduced['loss_objectness'].item()
        total_loss_rpn_box_reg += loss_dict_reduced['loss_rpn_box_reg'].item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            #print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        #train_loss.append(loss_value)

    # Getting average loss values and storing in dict
    avg_loss_dict = {
        "avg_total_loss": total_loss / num_batches,
        "avg_loss_classifier": total_loss_classifier / num_batches,
        "avg_loss_box_reg": total_loss_box_reg / num_batches,
        "avg_loss_objectness": total_loss_objectness / num_batches,
        "avg_loss_rpn_box_reg": total_loss_rpn_box_reg / num_batches
    }

    return metric_logger, iteration_loss_list, avg_loss_dict


def get_iou_type(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def validate_step(model, data_loader, device, scaler=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Validation:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = get_iou_type(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    # Init loss values
    iteration_loss_list = []  # loss graph iteration instead of epoch
    total_loss = 0
    total_loss_classifier = 0
    total_loss_box_reg = 0
    total_loss_objectness = 0
    total_loss_rpn_box_reg = 0
    num_batches = len(data_loader)



    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = list(img.to(device, non_blocking=False) for img in images)

        targets = [ # targets to get validation loss
            {k: (v.to(device, non_blocking=False) if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
            for t in targets
        ]

        if torch.cuda.is_available():
            # creates overhead by waiting for gpu operations to finish, use during validation for more reliable validating
            torch.cuda.synchronize()
        model_time = time.time()
        # validate
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            outputs, loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # Getting all loss values below
        # getting combined loss
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        # Getting loss values for iterations
        iteration_loss_list.append({
            'total_loss': losses_reduced.item(),
            'loss_classifier': loss_dict_reduced['loss_classifier'].item(),
            'loss_box_reg': loss_dict_reduced['loss_box_reg'].item(),
            'loss_objectness': loss_dict_reduced['loss_objectness'].item(),
            'loss_rpn_box_reg': loss_dict_reduced['loss_rpn_box_reg'].item()
        })
        #print("it loss:", iteration_loss_list)

        # gettting individual loss from dict for average epoch graphs
        total_loss += losses_reduced.item()
        total_loss_classifier += loss_dict_reduced['loss_classifier'].item()
        total_loss_box_reg += loss_dict_reduced['loss_box_reg'].item()
        total_loss_objectness += loss_dict_reduced['loss_objectness'].item()
        total_loss_rpn_box_reg += loss_dict_reduced['loss_rpn_box_reg'].item()

        #print(loss_dict_reduced)
        #print(loss_value)

        #outputs = model(images)
        #print(outputs)

        outputs = [{k: v.to(device, non_blocking=False) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # Getting average loss values and storing in dict
    avg_loss_dict = {
        "avg_total_loss": total_loss / num_batches,
        "avg_loss_classifier": total_loss_classifier / num_batches,
        "avg_loss_box_reg": total_loss_box_reg / num_batches,
        "avg_loss_objectness": total_loss_objectness / num_batches,
        "avg_loss_rpn_box_reg": total_loss_rpn_box_reg / num_batches
    }

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return coco_evaluator, iteration_loss_list, avg_loss_dict
