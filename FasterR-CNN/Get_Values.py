import os


def setTrainValues(val_to_get):
    # Creating dictionary with values
    train_values = {
        "BATCH_SIZE": 4,
        "EPOCHS": 3,
        "dataset": "Large data", # "Small data" OR "Large data"
        "model_name": "setTrainTest" # name of saved model
    }
    # return value corresponding with val_to_get
    return train_values.get(val_to_get, None)

def setTestValues(val_to_get):
    # Create dictionary with  values
    test_values = {
        "dataset": "Small Data", # "Small Data" OR "Large Data"
        "model_name": "valTestB" # name of model to test
    }

    # return value corresponding with val_to_get
    return test_values.get(val_to_get, None)



def checkColab():
    if "COLAB_GPU" in os.environ:
        base_dir = "/content/drive/My Drive/Colab Notebooks/Thesis"
    else:
        base_dir = "N:\\University subjects\\Thesis\\Python projects\\SmokeFasterRCNN"
    return base_dir
