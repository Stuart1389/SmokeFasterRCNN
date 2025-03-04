import torchvision.utils
from super_image import PanModel, ImageLoader
import os
import numpy as np
from PIL import Image
from pathlib import Path
from GetValues import setTrainValues
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch
from GetValues import checkColab, setTrainValues
from SmokeModel import SmokeModel

# This script takes a dataset and upscales it
# See README for more details
class SmokeUpscale:
    def __init__(self):
        self.batch_size = setTrainValues("BATCH_SIZE")
        self.epochs = setTrainValues("EPOCHS")
        self.smoke_model = SmokeModel(using_upscale_util = True)
        self.train_dataloader, self.validate_dataloader, _ = self.smoke_model.get_dataloader()
        self.upscale_model = PanModel.from_pretrained('eugenesiow/pan-bam', scale=setTrainValues("upscale_value"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.base_dir = checkColab()
        self.dataset_dir = Path(f"{self.base_dir}/Dataset/" + setTrainValues("upscale_dataset_save_name"))
        self.train_dir = Path(str(self.dataset_dir) + "/Train/images")
        self.val_dir = Path(str(self.dataset_dir) + "/Validate/images")


    # Method takes images and upscales them using super-resolution for pytorch model
    def upscale_images(self, image_tensors):
        combined_tensor = torch.stack(image_tensors, dim=0).to(self.device)
        upscale_model = PanModel.from_pretrained('eugenesiow/pan-bam', scale=setTrainValues("upscale_value"))
        upscale_model.to(self.device)
        upscale_outputs = upscale_model(combined_tensor)
        formatted_tensors = list(torch.unbind(upscale_outputs, dim=0))
        return formatted_tensors

    # Method saves upscaled images to disk
    def save_upscaled_images(self, image_tensor, filename, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir / f"{filename}.jpeg"
        torchvision.utils.save_image(image_tensor, save_path, "jpeg")



    # Loop upscales images using train and validate dataloaders
    def upscale_loop(self):
        for batch, (image_tensors, filenames) in enumerate(self.train_dataloader):
            print(f"Processing batch: {batch} of {len(self.train_dataloader)}")
            upscaled_images = self.upscale_images(image_tensors)
            for tensor, filename in zip(upscaled_images, filenames):
                self.save_upscaled_images(tensor, filename, self.train_dir)


        for batch, (image_tensors, filenames) in enumerate(self.validate_dataloader):
            print(f"Processing batch: {batch} of {len(self.validate_dataloader)}")
            upscaled_images = self.upscale_images(image_tensors)
            for tensor, filename in zip(upscaled_images, filenames):
                self.save_upscaled_images(tensor, filename, self.val_dir)


if __name__ == '__main__':
    smoke_upcale = SmokeUpscale()
    smoke_upcale.upscale_loop()