import os
import torch
# CONFIGURATION FILE, used to set values throughout the pipeline
def setGlobalValues(val_to_get):
    global_values = {
        "NUM_CLASSES": 2, # number of targets, E.g. background + smoke = 2
        "CLASS_INDEX_DICTIONARY": {"smoke": 1}, # each target must have an id, e.g. smoke = 1, fire = 2, etc

        # Backbone model settings
        # available backbones to use:
        # weights can be manually changed at Lib/site-packages/torchvision/models/detection/faster_rcnn.py
        # default - coco weight resnet 50 fpnv2
        # mobilenet  - IMAGENET1K weights for fpnv1, it's recommeneded to resize to 244*244 when using this
        # resnet_builder - IMAGENET1K weights for fpnv1
        "backbone_to_use": "default",

        # Settings for resnet builder, only applicable when using "resnet_builder" above
        "resnet_backbone": "resnet50",  # resnet18, resnet34, resnet50, resnet101, etc
        "fpnv2": False,  # Sets alternate model to use fpnv2
        "load_coco_weights": False,  # used to experiment with using coco weights on resnet101 with fpnv2
    }

    return global_values.get(val_to_get, None)


def setTrainValues(val_to_get):
    # Creating dictionary with values
    train_values = {
        # TRAIN SETTINGS
        # Basic train settings
        "BATCH_SIZE": 2,
        "EPOCHS": 2,
        "PATIENCE": 4,
        "dataset": "Small data", # Name of dataset to use e.g. Example dataset structure in README (doesn't apply when using hdf5 file)
        "model_name": "presentation_2080", # Names model, including directory, weights, etc
        "plotJSON_fname": "", # json and manual plot filename, leave empty to make same as model
        "model_id": "999", # model is is used for W&B runs only: id_modelname
        "save_at_end" : False, # forces the model to save at end of training, ignoring validation

        # HDF5 settings. TO create HDF5 file use WriteHdf5.py
        # NOTE WriteHdf5.py uses dataloader to write, so if this is enabled when writing
        # hdf5 will be used to write instead of the default dataset
        "h5py_dir_save_name": "", # file name for hdf5 file when written
        # chunk size for hdf5, needs to be less than number of samples in epoch
        "hdf5_chunk_size": 8,  # this only applies when creating the hdf5 file, not when loading
        # Loading hdf5
        "load_hdf5": True, # whether to load hdf5 file instead of default dataset
        "force_first_epoch": True,  # forces hdf5 to repeat the first epoch for all training epochs
        "h5py_dir_load_name": "test_file", # file name of hdf5 file to load

        # PROFILER
        "start_profiler" : True,
        "record_trace" : True,

        # Enables or disables Weights&Biases logging
        "logWB" : False, # conflicts with profiler

        # TRAINING HYPERPARAMETERS
        "learning_rate": 0.004,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "step_size": 4,
        "gamma": 0.1,

        # DATALOADER/TODEVICE/MIXED-PRECISSION
        "non_blocking": True,
        "pinned_memory": True,
        "amp_mixed_precission": False,
        "half_precission": False,

        # QUANT
        "quant_aware_training": False,

        # Load previously trained model to continue training
        "start_from_checkpoint": False,
        "model_load_name": "transform_csj_a100",

        # UPSCALE
        "upscale_image": False,
        # only needed when using previously upscaled images with original bbox
        "upscale_bbox": False,
        "upscale_value": 4,
        # SmokeUpscale.py can be used to upscale images before training
        "upscale_dataset_save_name": "Example upscale",

        "plot_feature_maps": False,
        # start testing straight after training
        "test_after_train": False
    }
    # return value corresponding with val_to_get
    return train_values.get(val_to_get, None)

def setTestValues(val_to_get):
    # TEST SETTINGS
    test_values = {
        "BATCH_SIZE": 4,
        "dataset": "Small data", # dataset to load from
        "model_name": "transform_csj_a100", # name of model to load
        "test_on_val": False, # test on validation instead of test set

        "CPU_inference": False,  # force cpu inference even if cuda is available

        # PROFILER
        "start_profiler": False,
        "record_trace": False,

        # DATALOADER/TODEVICE
        "non_blocking": True,
        "pinned_memory": True,

        # Quants, only enable if testing
        "static_quant": False,
        "calibrate_full_set": False, # calibrate on full dataset rather than 1 image
        "load_QAT_model": False, # this needs to be enabled if loading quant aware trained model

        "half_precission": True, # convert tensor to float 16

        #Pruning
        "prune_model": False,
        "unstructured": False, # or unstructured, uses l1/l2 to minimize effects
        "prune_amount": 0.3,
        "tensor_type": "csr", # coo or csr

        # Upscale
        "upscale_image": False,
        "upscale_value": 2, # image resolution * upscale value
        "split_images": False
    }

    # return value corresponding with val_to_get
    return test_values.get(val_to_get, None)

# sets working directory, used to ensure notebook runs correctly in colab
def checkColab():
    if "COLAB_GPU" in os.environ:
        base_dir = "/content/drive/My Drive/Colab Notebooks/Honours" # this is to run the program in colab
    else: # change this to (Project folder\\SmokeFasterRCNN) to run locally
        base_dir = "N:\\University subjects\\Thesis\\Python projects\\SmokeFasterRCNN"
    return base_dir

