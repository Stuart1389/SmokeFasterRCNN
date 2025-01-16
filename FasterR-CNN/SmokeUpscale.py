from super_image import PanModel, ImageLoader
from PIL import Image
from pathlib import Path

image_path = Path("N:\\University subjects\\Thesis\\Python projects\\SmokeFasterRCNN\\Dataset\\Small data\\Train\\images\\ckagzh7gxonuc0841rdaell0k.jpeg")
image = Image.open(image_path)

model = PanModel.from_pretrained('eugenesiow/pan-bam', scale=2)
model.to("cuda")
inputs = ImageLoader.load_image(image)
preds = model(inputs.to("cuda"))

ImageLoader.save_image(preds, './scaled_2x.png')
ImageLoader.save_compare(inputs, preds, './scaled_2x_compare.png')