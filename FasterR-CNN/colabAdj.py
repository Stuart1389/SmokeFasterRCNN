import os

def checkColab():
    if "COLAB_GPU" in os.environ:
        base_dir = "/content/drive/My Drive/Colab Notebooks/Thesis"
    else:
        base_dir = "N:\\University subjects\\Thesis\\Python projects\\SmokeFasterRCNN"
    return base_dir
