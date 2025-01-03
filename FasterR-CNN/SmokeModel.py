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
        self.train_dataloader = None
        self.validate_dataloader = None
        self.test_dataloader = None
        self.debug_dataloader = None

        self.num_train_epochs = setTrainValues("EPOCHS")

        self.load_path = Path("savedModels/" + setTestValues("model_name"))

    def get_model(self, testing=None, resnet101=None):
        if(resnet101):
            backbone = resnet_fpn_backbone(backbone_name="resnet18", pretrained=True, trainable_layers=3)
            self.model = FasterRCNN(backbone=backbone, num_classes=self.num_classes)
            #self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
            print(os.path.expanduser('~/.cache/torch/hub/checkpoints'))
            # Set in features to whatever region of interest(ROI) expects
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            # Required to work with 2 classes
            self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, self.num_classes)
            print(f"Number of parameters in ResNet101 backbone: {self.count_parameters(self.model)}")
            #print(self.model.backbone)
        else:
            # load faster-rcnn
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT",
                                                                                 trainable_backbone_layers=3)
            #self.model = torchvision.models.detection.fasterrcnn_resnet101_fpn_v2(weights="DEFAULT")
            print(os.path.expanduser('~/.cache/torch/hub/checkpoints'))
            # Set in features to whatever region of interest(ROI) expects
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            # Required to work with 2 classes
            self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, self.num_classes)
            print(f"Number of parameters in ResNet50 backbone: {self.count_parameters(self.model)}")

        # if testing then load a model and use its parameters
        if testing:
            #saved_dir = Path(f"{self.base_dir}/FasterR-CNN/savedModels/" + setTestValues("model_name") + ".pth")
            saved_dir = Path(self.load_path) / f"{setTestValues('model_name')}.pth"
            state_dict = torch.load(saved_dir, weights_only=True)
            self.model.load_state_dict(state_dict)
            return self.model

        return self.model, in_features, self.model.roi_heads.box_predictor

    # functions to get train, validate and test dataloaders
    def get_dataloader(self, testing = False):
        num_workers = os.cpu_count() # threads available
        batch_size = setTrainValues("BATCH_SIZE")
        test_batch_size = setTestValues("BATCH_SIZE")
        dataset_dir = Path(f"{self.base_dir}/Dataset/" + setTrainValues("dataset"))
        dataset_hd5f_dir = Path(f"{self.base_dir}/DatasetH5py/" + setTrainValues("h5py_dir_load_name"))
        test_dataset_dir = Path(f"{self.base_dir}/Dataset/" + setTestValues("dataset"))

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
            pin_memory=False, # speeds up transfer of data between cpu and gpu/puts data in page locked memory
            shuffle=False, # only shuffle while training
            sampler=train_sampler
        )

        self.validate_dataloader = DataLoader(
            dataset=val_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            shuffle=False,
            sampler=val_sampler
        )

        self.test_dataloader = DataLoader(
            dataset=test_dir,
            batch_size=test_batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
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
        device = "cuda" if torch.cuda.is_available() else "cpu"  # device agnostic
        self.get_dataloader()
        self.get_model()
        #self.check_dataloader()

    def checkModel(self):
        print(self.model)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)



# only run if this script is being run (not being called from other)
if __name__ == '__main__':
    model = SmokeModel()
    model.main()
    model.get_model(resnet101=True)
    model.checkModel()
