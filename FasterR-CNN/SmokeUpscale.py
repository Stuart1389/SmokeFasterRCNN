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
class SmokeUpscale:
    def __init__(self):
        self.batch_size = setTrainValues("BATCH_SIZE")
        self.epochs = setTrainValues("EPOCHS")
        self.smoke_model = SmokeModel()
        self.train_dataloader, self.validate_dataloader, _ = self.smoke_model.get_dataloader()
        self.upscale_model = PanModel.from_pretrained('eugenesiow/pan-bam', scale=setTrainValues("upscale_value"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.base_dir = checkColab()
        self.dataset_dir = Path(f"{self.base_dir}/Dataset/" + setTrainValues("upscale_dataset_save_name"))
        self.train_dir = str(self.dataset_dir) + "/Train/images"
        self.val_dir = str(self.dataset_dir) + "/Validate/images"

    def upscale_images(self, image_tensors):
        combined_tensor = torch.stack(image_tensors, dim=0).to(self.device)
        upscale_model = PanModel.from_pretrained('eugenesiow/pan-bam', scale=setTrainValues("upscale_value"))
        upscale_model.to(self.device)
        upscale_outputs = upscale_model(combined_tensor)
        # undo stack for faster rcnn input
        formatted_tensors = list(torch.unbind(upscale_outputs, dim=0))
        return formatted_tensors

    def save_upscaled_images(self, image_tensor, filename, save_dir):
        print("image_tensor: ", image_tensor, "filename: ", filename, "save dir:" , save_dir)
        os.makedirs(save_dir, exist_ok=True)
        #move to cpu or cry
        image_tensor = image_tensor.cpu().detach()
        image_tensor = image_tensor.permute(1, 2, 0)
        image_array = (image_tensor.numpy() * 255).astype(np.uint8)
        image = Image.fromarray(image_array)
        save_path = os.path.join(save_dir, filename)
        image.save(save_path)


    def upscale_loop(self):
        for batch, (image_tensors, filenames) in enumerate(self.train_dataloader):
            upscaled_images = self.upscale_images(image_tensors)
            for tensor, filename in zip(upscaled_images, filenames):
                self.save_upscaled_images(tensor, filename, self.train_dir)

        for batch, (image_tensors, _) in enumerate(self.validate_dataloader):
            upscaled_images = self.upscale_images(image_tensors)
            for tensor, filename in zip(upscaled_images, filenames):
                self.save_upscaled_images(tensor, filename, self.val_dir)

if __name__ == '__main__':
    smoke_upcale = SmokeUpscale()
    smoke_upcale.upscale_loop()
"""
image_path = Path("N:\\University subjects\\Thesis\\Python projects\\SmokeFasterRCNN\\Dataset\\Small data\\Train\\images\\ckagzh7gxonuc0841rdaell0k.jpeg")
image = Image.open(image_path)

model = PanModel.from_pretrained('eugenesiow/pan-bam', scale=2)
model.to("cuda")
inputs = ImageLoader.load_image(image)
print(inputs.shape)
preds = model(inputs.to("cuda")).to("cuda")
print(preds)


ImageLoader.save_image(preds, './scaled_2x.png')
ImageLoader.save_compare(inputs, preds, './scaled_2x_compare.png')

"""