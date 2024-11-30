import os
from pathlib import Path
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import Dataset
from Get_Values import checkColab, setTrainValues, setTestValues

# altering data to format that model expects at input or cry
# i hate this function so much the amount of hours ive spent on this little rascal
# you put this in class and it pretends it's cool but it secretly breaks everything
def collate_fn(batch):
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

    def get_model(self, testing=None):
        # load faster-rcnn
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        # Set in features to whatever region of interest(ROI) expects
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # Required to work with 2 classes
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, self.num_classes)

        # if testing then load a model and use its parameters
        if testing:
            saved_dir = Path(f"{self.base_dir}/FasterR-CNN/savedModels/" + setTestValues("model_name") + ".pth")
            state_dict = torch.load(saved_dir, weights_only=True)
            self.model.load_state_dict(state_dict)
            return self.model

        return self.model, in_features, self.model.roi_heads.box_predictor

    # function to get train, validate and test dataloaders
    def get_dataloader(self):
        num_workers = os.cpu_count() # threads available
        batch_size = setTrainValues("BATCH_SIZE")
        dataset_dir = Path(f"{self.base_dir}/Dataset/" + setTrainValues("dataset"))

        # Load datasets
        train_dir = Dataset.smokeDataset(str(dataset_dir) + "/Train", Dataset.transform_t)
        val_dir = Dataset.smokeDataset(str(dataset_dir) + "/Validate", Dataset.transform_t)
        test_dir = Dataset.smokeDataset(str(dataset_dir) + "/Test", Dataset.transform_t)

        # Create dataloaders to iterate over dataset (batching)
        self.train_dataloader = DataLoader(
            dataset=train_dir, # dataset to use
            batch_size=batch_size,
            num_workers=num_workers, # set to all available threads
            collate_fn=collate_fn,
            shuffle=True # only shuffle while training
        )
        self.validate_dataloader = DataLoader(
            dataset=val_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            shuffle=False
        )
        self.test_dataloader = DataLoader(
            dataset=test_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            shuffle=False
        )

        return self.train_dataloader, self.validate_dataloader, self.test_dataloader

    #!!FINISHED!!

    # check dataloader when running this script
    def check_dataloader(self):
        images, targets = next(iter(self.train_dataloader))
        for target in targets:
            print(target)
            print(type(target))

    # checking model and dataloaders
    def main(self):
        self.get_dataloader()
        self.get_model()
        self.check_dataloader()


# only run if this script is being run (not being called from other)
if __name__ == '__main__':
    model = SmokeModel()
    model.main()