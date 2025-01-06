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
import wandb
import torch.nn.functional as F
from torchsummary import summary
import torch.nn as nn

def train_step_know_distil(student, optimizer, data_loader, device, epoch, iteration, print_freq,
               scaler=None, profiler=None, teacher=None):
    iteration_loss_list = [] # loss graph iteration instead of epoch
    # Init loss values
    total_loss = 0
    total_loss_classifier = 0
    total_loss_box_reg = 0
    total_loss_objectness = 0
    total_loss_rpn_box_reg = 0
    num_batches = len(data_loader)
    temperature = 2
    alpha = 0.25
    lambda_bbox = 1.0

    running_loss = 0

    # set models to respective modes
    teacher.to(device)
    teacher.eval()
    student.train()



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
            # get logits from teacher
            with torch.no_grad():
                #student.rpn.pre_nms_top_n = teacher.rpn.pre_nms_top_n
                #student.rpn.post_nms_top_n = teacher.rpn.post_nms_top_n
                #print("pre student: ", student.rpn.pre_nms_top_n)
                #print("post student: ", student.rpn.post_nms_top_n)
                _, teacher_loss_dict, teacher_logits = teacher(images, targets)
                #print("teacher_logits:", teacher_logits.keys())
                #print(teacher.backbone)

            student_loss_dict, student_logits = student(images, targets)
            #print("student_logits: ", student_logits.keys())

            student_losses = sum(loss for loss in student_loss_dict.values())

            get_distil_loss(student_logits, teacher_logits)

            iteration += 1

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(student_loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item() # this batch
        # Getting loss values for iterations
        log_it_loss(iteration_loss_list, loss_value, loss_dict_reduced, epoch, iteration)

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
            #scaler.scale(total_distil_loss).backward()
            scaler.scale(student_losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            #total_distil_loss.backward()
            student_losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        #train_loss.append(loss_value)

    # Logging average loss
    avg_loss_dict = log_avg_loss(total_loss, total_loss_classifier, total_loss_box_reg,
                                 total_loss_objectness, total_loss_rpn_box_reg, num_batches, epoch, iteration)

    return metric_logger, iteration_loss_list, avg_loss_dict, iteration

def get_distil_loss(student_logits, teacher_logits, T = 10):
    #print(student_logits.keys())
    # Extract logits
    student_logit_objectness = student_logits["logit_objectness"]
    student_logit_rpn_box_reg = student_logits["logit_rpn_box_reg"]
    student_logit_classification = student_logits["logit_classification"]
    student_logit_bbox_delta = student_logits["logit_bbox_delta"]

    teacher_logit_objectness = teacher_logits["logit_objectness"]
    teacher_logit_rpn_box_reg = teacher_logits["logit_rpn_box_reg"]
    teacher_logit_classification = teacher_logits["logit_classification"]
    teacher_logit_bbox_delta = teacher_logits["logit_bbox_delta"]

    # Apply temperature scaling
    student_logit_objectness *= T
    teacher_logit_objectness *= T

    student_logit_classification *= T
    teacher_logit_classification *= T

    # Calculate the distillation losses
    objectness_loss = nn.KLDivLoss()(student_logit_objectness.log(), teacher_logit_objectness)  # Apply log on student logits for KLDivLoss
    print("objectness_loss: ", objectness_loss)

    rpn_box_loss = nn.MSELoss()(student_logit_rpn_box_reg, teacher_logit_rpn_box_reg)
    print("rpn_box_loss: ", rpn_box_loss)

    classification_loss = nn.KLDivLoss()(student_logit_classification.log(), teacher_logit_classification)  # Apply log on student logits for KLDivLoss
    print("classification_loss: ", classification_loss)

    bbox_delta_loss = nn.MSELoss()(student_logit_bbox_delta, teacher_logit_bbox_delta)
    print("bbox_delta_loss: ", bbox_delta_loss)

    # Return total distillation loss
    total_loss = objectness_loss + rpn_box_loss + classification_loss + bbox_delta_loss
    return total_loss


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