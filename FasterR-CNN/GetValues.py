import os
import torch
# using this file to quickly change values

def setTrainValues(val_to_get):
    # Creating dictionary with values
    train_values = {
        "BATCH_SIZE": 4,
        "EPOCHS": 2,
        "PATIENCE": 3,
        "dataset": "Large data", # Name of dataset to use
        "model_name": "test_file", # Names model, including directory, weights, etc
        "plotJSON_fname": "", # json and manual plot filename, leave black to make same as model
        "model_id": "39B",
        "save_at_end" : False,
        # temp resnet101_2080_101,

        # Default model settings
        "mobilenet": False,

        # Model builder settings
        "alt_model": False,
        "alt_model_backbone": "resnet101", #resnet18, resnet34, resnet50, resnet101, etc
        "fpnv2": False, # Sets alternate model to use fpnv2

        "load_hd5f" : True, # whether to load from hd5f MAKE SURE THIS IS OFF WHEN CREATING HD5F
        "force_first_epoch" : False, #
        "h5py_dir_save_name": "test_file", # file name for h5py file
        "h5py_dir_load_name": "test_file", #large_15_no_transform, transform_csj, large_1_transform, test_file
        # transform_csj_3_def, transform_csj_3_100, transform_csj_3_1, transform_csj_c_20

        # PROFILER
        "start_profiler" : False,
        "record_trace" : False,
        "logWB" : False,

        # PARAMETERS
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "step_size": 3,
        "gamma": 0.1,

        # DATALOADER/TODEVICE/MIXED-PRECISSION
        "non_blocking": True,
        "pinned_memory": True,
        "amp_mixed_precission": True,
        "half_precission": False,

        # QUANT
        "quant_aware_training": False,
        "start_from_checkpoint": False,
        "model_load_name": "transform_csj_a100",

        # UPSCALE
        "upscale_image": False,
        # only needed when using previously upscaled images with original bbox
        "upscale_bbox": False,
        "upscale_value": 2,
        # SmokeUpscale can be used to upscale images before training
        "upscale_dataset_save_name": "Large data upscale",

        # Misc
        #e.g. when using dataloader to upscale and filenames are needed
        "return_filenames": False,

        "plot_feature_maps": False,
        # start testing straight after training
        "test_after_train": False
    }
    # return value corresponding with val_to_get
    # filenames will be returned INSTEAD of TARGETS
    return train_values.get(val_to_get, None)

def setTestValues(val_to_get):
    # Create dictionary with  values
    test_values = {
        "BATCH_SIZE": 8,
        "dataset": "Large data", # "Small data" OR "Large data"
        "model_name": "transform_csj_a100", # name of model to test
        "test_on_val": False, # test on validation instead of test set

        # PROFILER
        "start_profiler": False,
        "record_trace": False,

        # DATALOADER/TODEVICE
        "non_blocking": True,
        "pinned_memory": True,

        # Quants, only enable if testing
        "static_quant": False,
        "CPU_inference": False, # force cpu inference even if cuda is available
        "calibrate_full_set": False,
        "load_QAT_model": False,

        "half_precission": True, # float 16

        #Pruning
        "prune_model": False,
        "unstructured": False, # or unstructured, uses l1/l2 to minimize effects
        "prune_amount": 0.3,
        "tensor_type": "csr", # coo or csr

        # Upscale
        "upscale_image": False,
        "upscale_value": 2, # image * upscale_value
        "split_images": False


    }

    # return value corresponding with val_to_get
    return test_values.get(val_to_get, None)

def checkColab():
    if "COLAB_GPU" in os.environ:
        base_dir = "/content/drive/My Drive/Colab Notebooks/Thesis" # this is to run the program in colab
    else: # change this to (Project folder\\SmokeFasterRCNN) to run locally
        base_dir = "N:\\University subjects\\Thesis\\Python projects\\SmokeFasterRCNN"
    return base_dir

