import os
import torch
# using this file to quickly change values

def setTrainValues(val_to_get):
    # Creating dictionary with values
    train_values = {
        "BATCH_SIZE": 8,
        "EPOCHS": 4,
        "PATIENCE": 4,
        "dataset": "Small data", # "Small data" , "Medium data" OR "Large data", "Small data cloud"
        "model_name": "qat_csj_a100_test", # name of saved model
        "plotJSON_fname": "qat_csj_a100_test", # json filename
        "model_id": "43",
        "save_at_end" : False,
        # temp resnet101_2080_101,

        # use alternate model
        "alt_model": False,
        "alt_model_backbone": "resnet101", #resnet18, resnet50, resnet101, etc
        "fpnv2": True, # Sets alternate model to use ehnhanced feature pyramid

        # Knowlege distillation, uses alt model
        "know_distil": False,
        "teacher_model_name": "transform_csj_a100",

        "load_hd5f" : False, # whether to load from hd5f MAKE SURE THIS IS OFF WHEN CREATING HD5F
        "h5py_dir_save_name": "transform_csj_3_def", # file name for h5py file
        "h5py_dir_load_name": "test_file", #large_15_no_transform, transform_csj, large_1_transform, test_file
        # transform_csj_3_def, transform_csj_3_100, transform_csj_3_1

        # PROFILER
        "start_profiler" : False,
        "record_trace" : False,
        "logWB" : False,

        # PARAMETERS
        "learning_rate": 0.003,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "step_size": 3,
        "gamma": 0.1,

        # DATALOADER/TODEVICE/MIXED-PRECISSION
        "non_blocking": True,
        "pinned_memory": True,
        "amp_mixed_precission": False,

        # QUANT
        "quant_aware_training": True
    }
    # return value corresponding with val_to_get
    return train_values.get(val_to_get, None)

def setTestValues(val_to_get):
    # Create dictionary with  values
    test_values = {
        "BATCH_SIZE": 4,
        "dataset": "Small data", # "Small data" OR "Large data"
        "model_name": "qat_csj_a100_test", # name of model to test

        # PROFILER
        "start_profiler": False,
        "record_trace": False,

        # DATALOADER/TODEVICE
        "non_blocking": True,
        "pinned_memory": True,

        # Quants, only enable if testing
        "static_quant": True,
        "CPU_inference": True,
        "calibrate_full_set": False,
        "load_QAT_model": False,

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
