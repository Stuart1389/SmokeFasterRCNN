import os
import sys
from pathlib import Path
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.models.vgg import vgg16, vgg16_bn, vgg19_bn
import Dataset
import DatasetHdf5
import time
from EpochSampler import EpochSampler
from GetValues import checkColab, setTrainValues, setTestValues, setGlobalValues
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, BackboneWithFPN, mobilenet_backbone
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
from collections import OrderedDict

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    return tuple(zip(*batch))

# class is used to create models and dataloaders for the pipeline
class SmokeModel:
    def __init__(self, force_default=None, using_upscale_util=False):
        """
        Example anchor sizes, leave as none for defaults
        # fpn takes 1 tuple per feature map
        example ((128,), (256,), (512,)) or for multi anchor size per feature map:
        ((32, 64, 128, 256, 512), (32, 64, 128, 256, 512), (32, 64, 128, 256, 512))
        number of feature maps depends on backbone e.g.
        default resnet50 fpnv2 is 5, default mobilenet is 3
        # example aspect ratios
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        """
        print("Beginning model setup")
        self.base_dir = checkColab() # get colab or local dirs
        self.num_classes = setGlobalValues("NUM_CLASSES")
        # initialising variables
        self.model = None
        # which resnet backbone to use if using a custom resnet model
        self.resnet_model_backbone = setGlobalValues("resnet_backbone")
        self.fpnv2 = setGlobalValues("fpnv2")
        self.train_dataloader = None
        self.validate_dataloader = None
        self.test_dataloader = None
        self.debug_dataloader = None
        self.in_features = None

        self.num_train_epochs = setTrainValues("EPOCHS")

        self.load_path_train_checkpoint = Path("savedModels/" + setTrainValues("model_load_name"))
        self.load_path_test = Path("savedModels/" + setTestValues("model_name"))
        self.model_arch_path = model_arch_path = Path("savedModelsArch/")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # device agnostic
        self.force_default = force_default

        self.generate_weights = False
        self.load_qat_model = setTestValues("load_QAT_model")
        self.start_from_checkpoint = setTrainValues("start_from_checkpoint")
        self.using_upscale_util = using_upscale_util

    # Method is responsible for build a model and loading any weights
    def get_model(self, testing=None):
        state_dict = None
        saved_dir = None
        self.model_builder()

        # loading model weights if necessary
        if testing:
            saved_dir = Path(self.load_path_test) / f"{setTestValues('model_name')}.pth"
            print("Load model for testing")
        elif self.start_from_checkpoint:
            print("Load from checkpoint")
            saved_dir = Path(self.load_path_train_checkpoint) / f"{setTrainValues('model_load_name')}.pth"

        if testing and not self.load_qat_model:
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



    # Method selects model building method based on GetValues.py settings
    def model_builder(self):
        if(setGlobalValues("backbone_to_use") == "default"):
            self.build_default()
        elif(setGlobalValues("backbone_to_use") == "mobilenet"):
            self.build_mobilenet()
        elif(setGlobalValues("backbone_to_use") == "resnet_builder"):
            self.resnet_builder()

    # Used to build a resnet model
    def resnet_builder(self):
        print(f"Model builder, Backbone: {self.resnet_model_backbone}, FpnV2: {self.fpnv2}")
        # default anchor sizes/ratios for IMAGENET1K
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        # generate anchors
        anchor_gen = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

        # Builds a model with more complex FPNv2 using resnet builder
        if (self.fpnv2):
            # Create faster-rcnn model with resnet backbone and fpnv2
            #This was for experimentation where COCO weights were extracted from default model
            #and used with the resnet builder. weights are provided in github for further experimentation
            if(setGlobalValues("load_coco_weights")):
                fpn = torch.load(self.model_arch_path / "fpn.pth", weights_only=True)
                roi_heads = torch.load(self.model_arch_path / "roi.pth", weights_only=True)
                rpn = torch.load(self.model_arch_path / "rpn.pth", weights_only=True)
                body = torch.load(self.model_arch_path / "body.pth", weights_only=True)
            else:
                fpn, roi_heads, rpn, body = None, None, None, None
            self.model = torchvision.models.detection.stus_resnet_fpnv2_builder(weights_backbone="DEFAULT", # IMAGENET1K
                                                                                trainable_backbone_layers=3,
                                                                                roi_head_weights=roi_heads,
                                                                                rpn_weights=rpn, fpn_weights=fpn, body_weights=body,
                                                                                model_backbone=self.resnet_model_backbone)
        else:
            # default resnet fpnv1 with IMAGENET1K weights
            backbone = resnet_fpn_backbone(backbone_name=self.resnet_model_backbone, weights="DEFAULT", trainable_layers=3)
            self.model = FasterRCNN(backbone=backbone, num_classes=2,
                                    rpn_anchor_generator=anchor_gen)
            body_weights = torch.load(self.model_arch_path / "body.pth", weights_only=True)
            self.model.backbone.body.load_state_dict(body_weights, strict=False)

        self.setup_model()

    # method builds default model
    # resnet-50 fpnv2 with coco weights
    def build_default(self):
        anchor_sizes = None
        aspect_ratios = None
        print("Default model Resnet50 FpnV2")
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT",
                                                                             trainable_backbone_layers=3,
                                                                             anchor_values=anchor_sizes,
                                                                             anchor_ratios=aspect_ratios)
        self.setup_model()

    # method builds a mobilenet model
    def build_mobilenet(self):
        anchor_sizes = None
        aspect_ratios = None
        print("Default model MobilenetV3 FPN")
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT",
                                                                             trainable_backbone_layers=3,
                                                                             anchor_values=anchor_sizes,
                                                                             anchor_ratios=aspect_ratios)
        self.setup_model()

    # extra steps needed for model setup
    def setup_model(self):
        print(f"Anchor sizes: {self.model.rpn.anchor_generator.sizes}")
        print(f"Anchor aspect ratios: {self.model.rpn.anchor_generator.aspect_ratios}")
        # Set in features to whatever region of interest(ROI) expects
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # Required to work with 2 classes
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(self.in_features,                                                                                          self.num_classes)
        print(f"Number of parameters (at model setup): {self.count_parameters(self.model)}")


    # This method was used to extract coco weights for experimentation
    def generate_fpnv2_weights(self):
        # num classes is overwritten by roi using self.num_classes
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT",
                                                                             trainable_backbone_layers=3)
        state_dict = self.model.state_dict()
        rpn_weights = {}
        roi_weights = {}
        fpn_weights = {}
        body_weights = {}

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
            elif key.startswith("backbone.body"):
                # get backbone.body pre-trained weights
                new_key = key.replace("backbone.body.", "")
                body_weights[new_key] = tensor


        # Save RPN weights
        rpn_path = self.model_arch_path / "rpn.pth"
        torch.save(rpn_weights, rpn_path)
        print(f"Saved RPN weights to {rpn_path}")

        # Save ROI weights
        roi_path = self.model_arch_path / "roi.pth"
        torch.save(roi_weights, roi_path)
        print(f"Saved ROI weights to {roi_path}")

        # Save FPN weights
        fpn_path = self.model_arch_path / "fpn.pth"
        torch.save(fpn_weights, fpn_path)
        print(f"Saved FPN weights to {fpn_path}")

        # Save body weights
        body_path = self.model_arch_path / "body.pth"
        torch.save(body_weights, body_path)
        print(f"Saved Body weights to {body_path}")

    # method to get train, validate and test dataloaders for their respective splits
    def get_dataloader(self, testing = False):
        num_workers = os.cpu_count() # threads available
        batch_size = setTrainValues("BATCH_SIZE")
        test_batch_size = setTestValues("BATCH_SIZE")
        # set dataset dirs based on values set in GetValues.py
        dataset_dir = Path(f"{self.base_dir}/Dataset/" + setTrainValues("dataset"))
        dataset_hdf5_dir = Path(f"{self.base_dir}/DatasetH5py/" + setTrainValues("h5py_dir_load_name"))
        test_dataset_dir = Path(f"{self.base_dir}/Dataset/" + setTestValues("dataset"))
        pinned_train = setTrainValues("pinned_memory")
        pinned_test = setTestValues("pinned_memory")
        # Load datasets
        if(setTrainValues("load_hdf5") == True and not testing):
            if(self.using_upscale_util):
                raise ValueError(f"Current util pre-upscale image file currently only works with default "
                                 f"dataset because hdf5 dataset doesnt return filename."
                                 f"This should be an easy to implement later")
            train_dir = DatasetHdf5.SmokeDatasetHdf5(str(dataset_hdf5_dir) + "/Train.hdf5")
            num_train_epochs = self.num_train_epochs
            train_sampler = EpochSampler(train_dir, epochs=num_train_epochs)
            val_dir = DatasetHdf5.SmokeDatasetHdf5(str(dataset_hdf5_dir) + "/Validate.hdf5")
            val_sampler = EpochSampler(val_dir, epochs=num_train_epochs)
        else:
            train_dir = Dataset.smokeDataset(str(dataset_dir) + "/Train", Dataset.transform_train, using_upscale_util = self.using_upscale_util)
            val_dir = Dataset.smokeDataset(str(dataset_dir) + "/Validate", Dataset.transform_validate, using_upscale_util = self.using_upscale_util)
            train_sampler = None
            val_sampler = None
        if(setTestValues("test_on_val")):
            test_dir = Dataset.smokeDataset(str(test_dataset_dir) + "/Validate", Dataset.transform_test, True)
        else:
            test_dir = Dataset.smokeDataset(str(test_dataset_dir) + "/Test", Dataset.transform_test, True, using_upscale_util = self.using_upscale_util)

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


    # this is used to evaluate models on the validate dataloader, even though its not good practice
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

    # checking model
    def checkModel(self):
        print(self.device)
        self.model.to(self.device)
        # Create dummy tensor
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        # pass through model and print summary
        summary(self.model.backbone.body, input_size=(3, 224, 224), device=str(self.device))
        self.prnt_model_c()


    def prnt_model_c(self):
        #Print general model settings
        print("General model settings:")
        print(f"Number of classes: {self.model.roi_heads.box_predictor.cls_score.out_features}")
        print(f"Backbone: {self.model.backbone}")
        print(f"RPN Anchor Generator: {self.model.rpn.anchor_generator}")

        # Print settings for the Region Proposal Network (RPN)
        print("\nRPN settings:")
        print(f"Anchor sizes: {self.model.rpn.anchor_generator.sizes}")
        print(f"Aspect ratios: {self.model.rpn.anchor_generator.aspect_ratios}")
        print(f"RPN: {self.model.rpn}")


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


    # method used to find number of model parameters
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)



# run this script to test model currently selected in GetValues.py
if __name__ == '__main__':
    """
    #This just prints information about the default resnet-50 model to compare 
    #with whatever other model is currently selected in cofig
    model = SmokeModel(force_default=True)
    model.main()
    model.get_model()
    model.checkModel()
    """
    print("\n-----------------------------------------------------\n")

    modelS = SmokeModel()
    modelS.get_model()
    modelS.checkModel()

