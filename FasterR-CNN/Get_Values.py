import os
import torch
# using this file to quickly change values

def setTrainValues(val_to_get):
    # Creating dictionary with values
    train_values = {
        "BATCH_SIZE": 8,
        "EPOCHS": 25,
        "PATIENCE": 4,
        "dataset": "Large data", # "Small data" OR "Large data"
        "model_name": "augmentationL8", # name of saved model
        "plotJSON_fname": "augmentationL8" # json filename
    }
    # return value corresponding with val_to_get
    return train_values.get(val_to_get, None)

def setTestValues(val_to_get):
    # Create dictionary with  values
    test_values = {
        "BATCH_SIZE": 4,
        "dataset": "Large data", # "Small data" OR "Large data"
        "model_name": "BaselineLocalRTX2080" # name of model to test
    }

    # return value corresponding with val_to_get
    return test_values.get(val_to_get, None)

def setGlobalValues(val_to_get):
    use_gpu = True # whether to use GPU if available
    if(use_gpu):
        device = "cuda" if torch.cuda.is_available() else "cpu" # device agnostic code
    else:
        device = "cpu"
    print(device)

    global_values = {
        "device": device
    }

def checkColab():
    if "COLAB_GPU" in os.environ: # think thesis is american for bsc but idc
        base_dir = "/content/drive/My Drive/Colab Notebooks/Thesis" # this is to run the program in colab
    else: # change this to (Project folder\\SmokeFasterRCNN) to run locally
        base_dir = "N:\\University subjects\\Thesis\\Python projects\\SmokeFasterRCNN"
    return base_dir
