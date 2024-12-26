import os
import torch
# using this file to quickly change values

def setTrainValues(val_to_get):
    # Creating dictionary with values
    train_values = {
        "BATCH_SIZE": 2,
        "EPOCHS": 25,
        "PATIENCE": 4,
        "dataset": "Large data", # "Small data" , "Medium data" OR "Large data"
        "model_name": "torch_update_L4", # name of saved model
        "plotJSON_fname": "torch_update_L4", # json filename

        "load_hd5f" : False, # whether to load from hd5f MAKE SURE THIS IS OFF WHEN CREATING HD5F
        "h5py_dir_save_name": "test_file", # file name for h5py file
        "h5py_dir_load_name": "test_file", #basic_transform_large

        # PROFILER
        "start_profiler" : False,
        "record_trace" : False,

        # PARAMETERS
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "step_size": 3,
        "gamma": 0.1
    }
    # return value corresponding with val_to_get
    return train_values.get(val_to_get, None)

def setTestValues(val_to_get):
    # Create dictionary with  values
    test_values = {
        "BATCH_SIZE": 4,
        "dataset": "Large data", # "Small data" OR "Large data"
        "model_name": "reduce_batch_L4", # name of model to test

        # PROFILER
        "start_profiler": False,
        "record_trace": False,
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
    if "COLAB_GPU" in os.environ:
        base_dir = "/content/drive/My Drive/Colab Notebooks/Thesis" # this is to run the program in colab
    else: # change this to (Project folder\\SmokeFasterRCNN) to run locally
        base_dir = "N:\\University subjects\\Thesis\\Python projects\\SmokeFasterRCNN"
    return base_dir
