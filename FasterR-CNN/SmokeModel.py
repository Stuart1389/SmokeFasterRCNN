import os
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
from torchsummary import summary


# altering data to format that model expects at input or cry
# i hate this function so much the amount of hours ive spent on this little rascal
# you put this in class and it pretends it's cool but it secretly breaks everything
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
    def __init__(self):
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

        self.load_path_train = Path("savedModels/" + setTrainValues("teacher_model_name"))
        self.load_path_test = Path("savedModels/" + setTestValues("model_name"))
        self.model_arch_path = model_arch_path = Path("savedModelsArch/")

        self.device = None

    def get_model(self, testing=None, know_distil=None, get_teacher=None):
        if(setTrainValues("alt_model") or know_distil):
            self.model_builder()
        else:
            self.model_default()

        # loading model weights if necessary
        if get_teacher:
            saved_dir = Path(self.load_path_train) / f"{setTrainValues('teacher_model_name')}.pth"
        elif testing:
            saved_dir = Path(self.load_path_test) / f"{setTestValues('model_name')}.pth"

        if testing or get_teacher:
            state_dict = torch.load(saved_dir, weights_only=True)
            self.model.load_state_dict(state_dict)
            return self.model

        return self.model, self.in_features, self.model.roi_heads.box_predictor

    def model_builder(self):
        print(f"model builder, backbone: {self.model_backbone}")
        backbone = resnet_fpn_backbone(backbone_name=self.model_backbone, pretrained=True, trainable_layers=3)
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_gen = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        # self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        """
        box_head = FastRCNNConvFCHead(
            (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
        ) box_head=box_head
        """

        # print("student ROI Head:")
        # print(self.model.roi_heads)
        self.model = FasterRCNN(backbone=backbone, num_classes=self.num_classes,
                                rpn_anchor_generator=anchor_gen)

        # Set in features to whatever region of interest(ROI) expects
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # Required to work with 2 classes
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(self.in_features,
                                                                                                        self.num_classes)
        print(f"Number of parameters: {self.count_parameters(self.model)}")
        if(self.fpnv2):
            self.model.rpn = torch.load(self.model_arch_path / "rpn.pth")
            self.model.roi = torch.load(self.model_arch_path / "roi.pth")
            print(f"Number of parameters: {self.count_parameters(self.model)}")

        #self.model.rpn_pre_nms_top_n_test = 2000
        #self.model.rpn_post_nms_top_n_test = 2000

        #print(self.model.backbone)
        #print(self.model)

    def model_default(self):
        # load faster-rcnn
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")

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
        #print(self.model.backbone)
        # print("Teacher ROI Head:")
        # print(self.model.roi_heads)
        # print(self.model)
        # print(self.model.roi_heads)
        """
        print(self.model.roi_heads)
        print(self.model.rpn)
        print("saving default model stuff")
        rpn = self.model.rpn
        roi = self.model.roi_heads
        model_arch_path = Path("savedModelsArch/")
        torch.save(rpn, model_arch_path / "rpn.pth")
        torch.save(roi, model_arch_path / "roi.pth")
        """



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
        self.get_dataloader()
        self.get_model()
        #self.check_dataloader()

    def checkModel(self):
        self.model.to(self.device)

        # Create dummy tensor
        #dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

        # pass through model and print summary
        #summary(self.model.backbone.body, input_size=(3, 224, 224), device=str(self.device))

        #print(self.model.backbone)
        #print(self.model.roi_heads)
        #print(self.model.rpn)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)



# only run if this script is being run (not being called from other)
if __name__ == '__main__':
    model = SmokeModel()
    model.main()
    model.get_model(know_distil=False)
    model.checkModel()

    modelS = SmokeModel()
    modelS.main()
    modelS.get_model(know_distil=True)
    modelS.checkModel()
