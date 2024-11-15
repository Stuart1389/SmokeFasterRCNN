def getModel(test=None, fine_tune=False):
    ### Defining model
    import torchvision
    import os
    import torch
    from pathlib import Path
    from colabAdj import checkColab
    # Loading model

    """
    Faster R-CNN/Regional convolutional neural network
    Two stage detector = slower but more accurate compared to one stage (e.g. yolo)
    Uses anchors, generates anchors on the image in a grid, compares to bounding box using IoU
    IoU measures overlap between 2 bounding boxes
     
    """
    base_dir = checkColab()
    num_classes = 2  # Dataset has only bounding boxes with label smoke + background = 2
    # Transfer learning
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    # Set in features to whatever region of interest(ROI) expects
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Required to work with 2 classes
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    #print(model.roi_heads.box_predictor)
    # det_model
    # det_model.state_dict()

    # Unfreezing layers so model can update then during training
    # Allows tensor to be updated during back propogation
    # Aka fine tune
    if fine_tune is not False:
        for param in model.rpn.parameters():
            param.requires_grad = True
        for param in model.roi_heads.parameters():
            param.requires_grad = True
        for param in model.backbone.parameters():
            param.requires_grad = True
        print("Finetune")

    for param in model.rpn.parameters():
        print(f"RPN unfrozen/requires_grad: {param.requires_grad}")
    for param in model.roi_heads.parameters():
        print(f"ROI Heads unfrozen/requires_grad: {param.requires_grad}")
    for param in model.backbone.parameters():
        print(f"Backbone unfrozen/requires_grad: {param.requires_grad}")

    if test is None:
        # If no params then get model which hasnt been trained on smokeDataset yet
        return model, in_features, model.roi_heads.box_predictor
    else:
        # If params are passed then we want to load state_dict, aka weights i've trained
        saved_dir = Path(rf"{base_dir}\savedModels\smokeDetBaseLarge.pth")
        state_dict = torch.load(saved_dir)
        model.load_state_dict(state_dict)
        return model

def getDataloader():
    ### Create dataloaders
    # Creating iterable to go through dataset in batches
    from torch.utils.data import DataLoader
    import torchvision.utils as utils
    import utils
    import Dataset
    import os
    from pathlib import Path
    from colabAdj import checkColab
    # from Dataset import smokeDataset, transform_t

    NUM_WORKERS = os.cpu_count()  # set to num cores available
    BATCH_SIZE = 2

    base_dir = checkColab()
    dataset_dir = Path(rf"{base_dir}\Dataset\Large data")
    train_test = Dataset.smokeDataset(str(dataset_dir) + "/Train", Dataset.transform_t)
    test_test = Dataset.smokeDataset(str(dataset_dir) + "/Test", Dataset.transform_t)

    train_dataloader = DataLoader(
        dataset=train_test,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,  # Number of cpu cores
        collate_fn=utils.collate_fn, # Creates list of dictionaries, without this we error
        shuffle=False # Temporary while testing manual seeds
    )

    test_dataloader = DataLoader(
        dataset=test_test,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=utils.collate_fn,
        shuffle=False
    )

    print(f"Dataloader objects: {train_dataloader}, {test_dataloader}")
    # displays batch size, !!! is (1, 1) because using DATASET SMALLER THAN BATCHSIZE ATM!!
    print(f"{len(train_dataloader)}, {len(test_dataloader)}")
    print(len(train_test)), print(len(test_test))
    return train_dataloader, test_dataloader

# This was to check if target was xstructured correctly, mainly cause no collate function
# Will leave in
def main():
    train_dataloader, test_dataloader = getDataloader()
    model, in_features, model.roi_heads.box_predictor = getModel(fine_tune=True)
    images, targets = next(iter(train_dataloader))
    print(targets)

    for t in targets:
        print(t)
        print(type(t))

    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.transforms import functional as F

    # Get a batch of data
    #images, targets = next(iter(train_dataloader))
    #print(images, targets)

if __name__ == '__main__':
    main()





