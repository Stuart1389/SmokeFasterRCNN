import os
import sys
from pathlib import Path
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import Dataset
import DatasetHd5f
import time
from EpochSampler import EpochSampler
from GetValues import checkColab, setTrainValues, setTestValues
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models import resnet50
import torch.nn as nn
import torch.nn.init as init
from torchsummary import summary
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np



"""
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    return inputs, list(targets)
"""

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    return tuple(zip(*batch))


class SmokeModel:
    # constructor
    def __init__(self, force_default=None):
        self.base_dir = checkColab() # get colab or local dirs
        self.num_classes = 2  # Smoke + background = 2
        # initializing variables
        self.model = None
        self.model_backbone = setTrainValues("alt_model_backbone")
        self.fpnv2 = setTrainValues("fpnv2")
        self.train_dataloader = None
        self.validate_dataloader = None
        self.test_dataloader = None
        self.debug_dataloader = None
        self.in_features = None

        self.num_train_epochs = setTrainValues("EPOCHS")

        self.load_path_teacher = Path("savedModels/" + setTrainValues("teacher_model_name"))
        self.load_path_train_checkpoint = Path("savedModels/" + setTrainValues("model_load_name"))
        self.load_path_test = Path("savedModels/" + setTestValues("model_name"))
        self.model_arch_path = model_arch_path = Path("savedModelsArch/")

        self.device = None
        self.force_default = force_default

        self.generate_weights = False
        self.load_qat_model = setTestValues("load_QAT_model")
        self.start_from_checkpoint = setTrainValues("start_from_checkpoint")

    def get_model(self, testing=None, know_distil=None, get_teacher=None):
        state_dict = None
        saved_dir = None
        if(setTrainValues("alt_model") or know_distil and not self.force_default and not self.generate_weights):
            self.model_builder()
        elif(not self.generate_weights):
            self.model_default()
        else:
            self.generate_fpnv2_weights()

        # loading model weights if necessary
        if get_teacher:
            saved_dir = Path(self.load_path_teacher) / f"{setTrainValues('teacher_model_name')}.pth"
            print("Load teacher")
        elif testing:
            saved_dir = Path(self.load_path_test) / f"{setTestValues('model_name')}.pth"
            print("Load model for testing")
        elif self.start_from_checkpoint:
            print("Load from checkpoint")
            saved_dir = Path(self.load_path_train_checkpoint) / f"{setTrainValues('model_load_name')}.pth"

        if testing and not self.load_qat_model or get_teacher:
            state_dict = torch.load(saved_dir, weights_only=True)
            self.model.load_state_dict(state_dict)
            return self.model
        elif self.load_qat_model:
            try:
                state_dict = torch.load(saved_dir, weights_only=True)
            except AttributeError:
                print("Ensure load QAT model is disabled (GetValues.py) if not TESTING and a quant aware trained model\n")
                sys.exit(1)
            return self.model, state_dict
        elif self.start_from_checkpoint:
            state_dict = torch.load(saved_dir, weights_only=True)
            self.model.load_state_dict(state_dict)
            return self.model, self.in_features, self.model.roi_heads.box_predictor
        else:
            return self.model, self.in_features, self.model.roi_heads.box_predictor

    def model_builder(self):
        print(f"Model builder, Backbone: {self.model_backbone}, FpnV2: {self.fpnv2}")
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_gen = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

        if (self.fpnv2):
            # print(combined_state_dict)
            fpn = torch.load(self.model_arch_path / "fpn.pth", weights_only=True)
            roi_heads = torch.load(self.model_arch_path / "roi.pth", weights_only=True)
            rpn = torch.load(self.model_arch_path / "rpn.pth", weights_only=True)
            self.model = torchvision.models.detection.stus_resnet_fpnv2_builder(weights_backbone="DEFAULT",#"DEFAULT"
                                                                                trainable_backbone_layers=3,
                                                                                roi_head_weights=roi_heads,
                                                                                rpn_weights=rpn, fpn_weights=fpn,
                                                                                model_backbone=self.model_backbone)
            self.model.rpn.anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        else:
            backbone = resnet_fpn_backbone(backbone_name=self.model_backbone, weights=None, trainable_layers=5)
            self.model = FasterRCNN(backbone=backbone, num_classes=self.num_classes,
                                    rpn_anchor_generator=anchor_gen)

        # Set in features to whatever region of interest(ROI) expects
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # set roi to 2 classes, mainly needed for fpnv2
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(self.in_features,
                                                                                                             self.num_classes)
        print(f"Number of parameters: {self.count_parameters(self.model)}")


        #self.model.rpn_pre_nms_top_n_test = 2000
        #self.model.rpn_post_nms_top_n_test = 2000

        #print(self.model.backbone)
        #print(self.model)


    def model_default(self):
        print("Default model Resnet50 FpnV2")
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT",
                                                                             trainable_backbone_layers=3)
        # load faster-rcnn


        #self.init_model()
        #state_dict = self.model.state_dict()
        #filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("backbone.body")}
        #torch.save(filtered_state_dict, self.model_arch_path / "fpnv2.pth")


        # Modify anchor box, defaults in detection/faster_rcnn
        # ((32,), (64,), (128,), (256,), (512,)), ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        # fpn takes 1 tuple per feature map, and has 5 feature maps
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        # anchor_sizes = ((8,), (32,), (128,), (256,), (512,))
        # anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
        # anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        self.model.rpn.anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

        # Set in features to whatever region of interest(ROI) expects
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # Required to work with 2 classes
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(self.in_features,
                                                                                                        self.num_classes)
        print(f"Number of parameters: {self.count_parameters(self.model)}")



    # generate weights for adapted fpnv2
    def generate_fpnv2_weights(self):
        # num classes is overwritten by roi using self.num_classes
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT",
                                                                             trainable_backbone_layers=3)
        state_dict = self.model.state_dict()
        rpn_weights = {}
        roi_weights = {}
        fpn_weights = {}

        for key, tensor in state_dict.items():
            print(key)
            if key.startswith("rpn.head"):
                # get rpn pre-trained weights
                new_key = key.replace("rpn.", "")
                rpn_weights[new_key] = tensor
            elif key.startswith("roi_heads"):
                # get roi pre-trained weights
                new_key = key.replace("roi_heads.", "")
                roi_weights[new_key] = tensor
            elif key.startswith("backbone.fpn"):
                # get fpn pre-trained weights
                new_key = key.replace("backbone.fpn.", "")
                fpn_weights[new_key] = tensor


        # Save RPN weights
        rpn_path = self.model_arch_path / "rpn.pth"
        torch.save(rpn_weights, rpn_path)
        print(f"Saved RPN weights to {rpn_path}")

        # Save ROI weights
        roi_path = self.model_arch_path / "roi.pth"
        torch.save(roi_weights, roi_path)
        print(f"Saved ROI weights to {roi_path}")

        # Save ROI weights
        fpn_path = self.model_arch_path / "fpn.pth"
        torch.save(fpn_weights, fpn_path)
        print(f"Saved ROI weights to {fpn_path}")

    # functions to get train, validate and test dataloaders
    def get_dataloader(self, testing = False):
        num_workers = os.cpu_count() # threads available
        batch_size = setTrainValues("BATCH_SIZE")
        test_batch_size = setTestValues("BATCH_SIZE")
        dataset_dir = Path(f"{self.base_dir}/Dataset/" + setTrainValues("dataset"))
        dataset_hd5f_dir = Path(f"{self.base_dir}/DatasetH5py/" + setTrainValues("h5py_dir_load_name"))
        test_dataset_dir = Path(f"{self.base_dir}/Dataset/" + setTestValues("dataset"))

        pinned_train = setTrainValues("pinned_memory")
        pinned_test = setTestValues("pinned_memory")

        # Load datasets
        if(setTrainValues("load_hd5f") == True and not testing):
            train_dir = DatasetHd5f.SmokeDatasetHd5f(str(dataset_hd5f_dir) + "/Train.hdf5")
            num_train_epochs = self.num_train_epochs
            train_sampler = EpochSampler(train_dir, epochs=num_train_epochs)
            val_dir = DatasetHd5f.SmokeDatasetHd5f(str(dataset_hd5f_dir) + "/Validate.hdf5")
            val_sampler = EpochSampler(val_dir, epochs=num_train_epochs)
        else:
            train_dir = Dataset.smokeDataset(str(dataset_dir) + "/Train", Dataset.transform_train)
            val_dir = Dataset.smokeDataset(str(dataset_dir) + "/Validate", Dataset.transform_validate)
            train_sampler = None
            val_sampler = None
        test_dir = Dataset.smokeDataset(str(test_dataset_dir) + "/Test", Dataset.transform_test, True)

        # Create dataloaders to iterate over dataset (batching)
        self.train_dataloader = DataLoader(
            dataset=train_dir, # dataset to use
            batch_size=batch_size,
            num_workers=num_workers, # set to all available threads
            collate_fn=collate_fn,
            pin_memory=pinned_train, # speeds up transfer of data between cpu and gpu/puts data in page locked memory
            shuffle=False, # only shuffle while training
            sampler=train_sampler
        )

        self.validate_dataloader = DataLoader(
            dataset=val_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pinned_train,
            shuffle=False,
            sampler=val_sampler
        )

        self.test_dataloader = DataLoader(
            dataset=test_dir,
            batch_size=test_batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pinned_test,
            shuffle=False
        )

        return self.train_dataloader, self.validate_dataloader, self.test_dataloader

    def get_validate_test_dataloader(self):
        test_dataset_dir = Path(f"{self.base_dir}/Dataset/" + setTestValues("dataset"))
        num_workers = os.cpu_count() # threads available
        test_batch_size = setTestValues("BATCH_SIZE")
        pinned_test = setTestValues("pinned_memory")

        # Load datasets
        val_dir = Dataset.smokeDataset(str(test_dataset_dir) + "/Validate", Dataset.transform_validate)
        self.test_validate_dataloader = DataLoader(
            dataset=val_dir,
            batch_size=test_batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pinned_test,
            shuffle=False
        )

        return self.test_validate_dataloader

    def get_debug_dataloader(self):
        num_workers = os.cpu_count() # threads available
        batch_size = setTrainValues("BATCH_SIZE")
        test_batch_size = setTestValues("BATCH_SIZE")
        dataset_dir = Path(f"{self.base_dir}/Dataset/" + setTrainValues("dataset"))
        test_dataset_dir = Path(f"{self.base_dir}/Dataset/" + setTestValues("dataset"))

        # Load datasets
        train_dir = Dataset.smokeDataset(str(dataset_dir) + "/Train", Dataset.transform_train)
        self.debug_dataloader = DataLoader(
            dataset=train_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            shuffle=True
        )

        return self.debug_dataloader



    #!!FINISHED!!

    # check dataloader when running this script
    def check_dataloader(self, test_runs = 1):
        for test_runs in range(test_runs):
            # Iterate through entire dataloader
            for batch_idx, (images, targets) in enumerate(self.train_dataloader):
                print(f"Batch {batch_idx}:")
            for batch_idx, (images, targets) in enumerate(self.validate_dataloader):
                print(f"Batch {batch_idx}:")
            for batch_idx, (images, targets) in enumerate(self.test_dataloader):
                print(f"Batch {batch_idx}:")


    # checking model and dataloaders
    def main(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # device agnostic
        #self.get_dataloader()
        #self.get_model()
        #self.check_dataloader()

    def checkModel(self):
        self.model.to(self.device)

        # Create dummy tensor
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

        # pass through model and print summary
        summary(self.model.backbone.body, input_size=(3, 224, 224), device=str(self.device))

        #print(self.model.backbone)
        #print(self.model.roi_heads)
        #print(self.model.rpn)


    def prnt_model_c(self):
        # Print general model settings
        print("General model settings:")
        print(f"Number of classes: {self.model.roi_heads.box_predictor.cls_score.out_features}")
        print(f"Backbone: {self.model.backbone}")
        print(f"RPN Anchor Generator: {self.model.rpn.anchor_generator}")

        # Print settings for the Region Proposal Network (RPN)
        print("\nRPN settings:")
        print(f"Anchor sizes: {self.model.rpn.anchor_generator.sizes}")
        print(f"Aspect ratios: {self.model.rpn.anchor_generator.aspect_ratios}")


        # Print settings for the RoI Heads (Region of Interest)
        print("\nRoI Heads settings:")
        print(f"Box predictor: {self.model.roi_heads.box_predictor}")
        print(f"Keypoint predictor: {self.model.roi_heads.keypoint_predictor}")

        # Print settings for the FPN (Feature Pyramid Network)
        print("\nFPN settings:")
        print(f"FPN in the backbone: {self.model.backbone.fpn}")
        print(f"FPN output channels: {self.model.backbone.out_channels}")


        # Print settings for the model's general hyperparameters and architecture
        print("\nModel hyperparameters and architecture:")
        print(f"Image Mean: {self.model.transform.image_mean}")
        print(f"Image Std: {self.model.transform.image_std}")


    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)



# only run if this script is being run (not being called from other)
if __name__ == '__main__':
    model = SmokeModel(force_default=True)
    model.main()
    model.get_model(know_distil=False)
    #model.checkModel()

    print("\n-----------------------------------------------------\n")

    modelS = SmokeModel()
    modelS.main()
    modelS.get_model(know_distil=True)
    #modelS.checkModel()
