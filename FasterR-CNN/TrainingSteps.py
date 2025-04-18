import os
import sys
current_dir = os.getcwd()
# add libr as source
relative_path = os.path.join(current_dir, '..', 'Libr')
sys.path.append(relative_path)

import math
import time
import torch
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
import wandb
from GetValues import setTrainValues
from super_image import PanModel

# function used to dynamically upscale images during training
@torch.inference_mode()
def upscale_images(device, image_tensors):
    combined_tensor = torch.stack(image_tensors, dim=0).to(device)
    # initialise model to upscale
    upscale_model = PanModel.from_pretrained('eugenesiow/pan-bam', scale=setTrainValues("upscale_value"))
    upscale_model.to(device)
    upscale_model.eval()
    # pass images through upscale model and return upscaled images
    upscale_outputs = upscale_model(combined_tensor)
    # undo stack for faster rcnn input
    formatted_tensors = list(torch.unbind(upscale_outputs, dim=0))
    return formatted_tensors

# Training step function
def train_step(model, optimizer, data_loader, device, epoch, iteration, print_freq,
               scaler=None):
    iteration_loss_list = [] # loss graph iteration instead of epoch
    # Init loss values
    total_loss = 0
    total_loss_classifier = 0
    total_loss_box_reg = 0
    total_loss_objectness = 0
    total_loss_rpn_box_reg = 0
    num_batches = len(data_loader)
    non_blocking = setTrainValues("non_blocking")

    # metric logger used to display results to console for simplicity
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # model warmup to help mitigate instability
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    # training loop
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        if(setTrainValues("upscale_image")):
            images = upscale_images(device, images)
        images = list(image.to(device, non_blocking=non_blocking) for image in images)

        if(setTrainValues("half_precission")):
            images = [tensor.cuda().half() for tensor in images]

        targets = [{k: v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with torch.amp.autocast('cuda', enabled=scaler is not None): # Automatic mixed precision
            # input images and targets into faster-rcnn and return loss dictionary
            loss_dict, _, _, _, _ = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            iteration += 1

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item() # this batch
        # Getting loss values for iterations
        log_it_loss(iteration_loss_list, loss_value, loss_dict_reduced, epoch, iteration)

        # gettting individual loss from dict for average epoch graphs
        total_loss += losses_reduced.item()
        total_loss_classifier += loss_dict_reduced['loss_classifier'].item()
        total_loss_box_reg += loss_dict_reduced['loss_box_reg'].item()
        total_loss_objectness += loss_dict_reduced['loss_objectness'].item()
        total_loss_rpn_box_reg += loss_dict_reduced['loss_rpn_box_reg'].item()

        # indicates instability
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            # AMP
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

    # Logging average loss
    avg_loss_dict = log_avg_loss(total_loss, total_loss_classifier, total_loss_box_reg,
                                 total_loss_objectness, total_loss_rpn_box_reg, num_batches, epoch, iteration)

    return metric_logger, iteration_loss_list, avg_loss_dict, iteration



def log_avg_loss(total_loss, total_loss_classifier, total_loss_box_reg, total_loss_objectness, total_loss_rpn_box_reg, num_batches, epoch, iteration):
    avg_loss_dict = {
        "avg_total_loss": total_loss / num_batches,
        "avg_loss_classifier": total_loss_classifier / num_batches,
        "avg_loss_box_reg": total_loss_box_reg / num_batches,
        "avg_loss_objectness": total_loss_objectness / num_batches,
        "avg_loss_rpn_box_reg": total_loss_rpn_box_reg / num_batches,
    }
    wandb.log({
        "train_avg_total_loss_epoch": avg_loss_dict["avg_total_loss"],
        "train_avg_loss_classifier_epoch": avg_loss_dict["avg_loss_classifier"],
        "train_avg_loss_box_reg_epoch": avg_loss_dict["avg_loss_box_reg"],
        "train_avg_loss_objectness_epoch": avg_loss_dict["avg_loss_objectness"],
        "train_avg_loss_rpn_box_reg_epoch": avg_loss_dict["avg_loss_rpn_box_reg"],
        'current_epoch': epoch + 1,
        'current_iteration': iteration
    })
    return avg_loss_dict


def log_it_loss(iteration_loss_list, loss_value, loss_dict_reduced, epoch, iteration):
    iteration_loss_list.append({
        'total_loss': loss_value,
        'loss_classifier': loss_dict_reduced['loss_classifier'].item(),
        'loss_box_reg': loss_dict_reduced['loss_box_reg'].item(),
        'loss_objectness': loss_dict_reduced['loss_objectness'].item(),
        'loss_rpn_box_reg': loss_dict_reduced['loss_rpn_box_reg'].item(),
    })
    wandb.log({
        'train_total_loss_it': loss_value,
        'train_loss_classifier_it': loss_dict_reduced['loss_classifier'].item(),
        'train_loss_box_reg_it': loss_dict_reduced['loss_box_reg'].item(),
        'train_loss_objectness_it': loss_dict_reduced['loss_objectness'].item(),
        'train_loss_rpn_box_reg_it': loss_dict_reduced['loss_rpn_box_reg'].item(),
        'current_epoch': epoch + 1,
        'current_iteration': iteration
    })


# Validation step function - uses coco evaluatior for simplisity
# more complex evaluation implemented in Tester.py
@torch.inference_mode()
def validate_step(model, data_loader, device, epoch, iteration, scaler=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Validation:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    # method of calculating iou determined by target type, in our case bounding boxes
    coco_evaluator = CocoEvaluator(coco, ['bbox'])

    # Init loss values
    iteration_loss_list = []  # loss graph iteration instead of epoch
    total_loss = 0
    total_loss_classifier = 0
    total_loss_box_reg = 0
    total_loss_objectness = 0
    total_loss_rpn_box_reg = 0
    num_batches = len(data_loader)
    non_blocking = setTrainValues("non_blocking")

    # validation loop
    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = list(img.to(device, non_blocking=non_blocking) for img in images)
        if(setTrainValues("half_precission")):
            images = [tensor.cuda().half() for tensor in images]

        targets = [ # targets are necessary for validation loss
            {k: (v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
            for t in targets
        ]

        if torch.cuda.is_available():
            # creates overhead by waiting for gpu operations to finish, use during validation for more reliable validating
            torch.cuda.synchronize()
        model_time = time.time()

        with torch.amp.autocast('cuda', enabled=scaler is not None): # Automatic mixed precision
            iteration += 1
            # model input here, get predictions and validation loss
            outputs, loss_dict, _, _, _, _ = model(images, targets)

        # Getting all loss values below
        # getting combined loss
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        # Getting loss values for iteration
        iteration_loss_list.append({
            'total_loss': losses_reduced.item(),
            'loss_classifier': loss_dict_reduced['loss_classifier'].item(),
            'loss_box_reg': loss_dict_reduced['loss_box_reg'].item(),
            'loss_objectness': loss_dict_reduced['loss_objectness'].item(),
            'loss_rpn_box_reg': loss_dict_reduced['loss_rpn_box_reg'].item()
        })

        wandb.log({
            'val_total_loss_it': losses_reduced.item(),
            'val_loss_classifier_it': loss_dict_reduced['loss_classifier'].item(),
            'val_loss_box_reg_it': loss_dict_reduced['loss_box_reg'].item(),
            'val_loss_objectness_it': loss_dict_reduced['loss_objectness'].item(),
            'val_loss_rpn_box_reg_it': loss_dict_reduced['loss_rpn_box_reg'].item(),
            'current_epoch': epoch + 1,
            'current_iteration': iteration
        })
        

        # gettting individual loss from dict for average epoch graphs
        total_loss += losses_reduced.item()
        total_loss_classifier += loss_dict_reduced['loss_classifier'].item()
        total_loss_box_reg += loss_dict_reduced['loss_box_reg'].item()
        total_loss_objectness += loss_dict_reduced['loss_objectness'].item()
        total_loss_rpn_box_reg += loss_dict_reduced['loss_rpn_box_reg'].item()

        outputs = [{k: v.to(device, non_blocking=non_blocking) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        # process predictions
        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    avg_loss_dict = {
        "avg_total_loss": total_loss / num_batches,
        "avg_loss_classifier": total_loss_classifier / num_batches,
        "avg_loss_box_reg": total_loss_box_reg / num_batches,
        "avg_loss_objectness": total_loss_objectness / num_batches,
        "avg_loss_rpn_box_reg": total_loss_rpn_box_reg / num_batches
    }

    wandb.log({
        "val_avg_total_loss_epoch": total_loss / num_batches,
        "val_avg_loss_classifier_epoch": total_loss_classifier / num_batches,
        "val_avg_loss_box_reg_epoch": total_loss_box_reg / num_batches,
        "val_avg_loss_objectness_epoch": total_loss_objectness / num_batches,
        "val_avg_loss_rpn_box_reg_epoch": total_loss_rpn_box_reg / num_batches,
        'current_epoch': epoch + 1,
        'current_iteration': iteration
    })

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return coco_evaluator, iteration_loss_list, avg_loss_dict, iteration