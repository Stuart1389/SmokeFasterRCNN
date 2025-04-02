<h1> Resources </h1>
<p>
In the root directory of the repository:
requirements.txt can be used to easily download/install required packages <br>
Pipeline.xlsx contains information about model training and evaluation. Along with some testing for the pipeline. <br>
SmokePipeline.ipynb is a jupyter notebook designed for use with google colab. (See bottom of README for more info).
</p>

<h1> Using Weights & Biases</h1>
<p>
To use weights and biased login to a W&B account. <a href = "https://docs.wandb.ai/quickstart/"> W&B quickstart </a> <br>
Create a W&B project, go to Trainer.py and edit: wandb.init(project="Example_project_name" <br>
</p>

<h2> Getting the pipeline ready </h2>
<p>
*NOTE: This ReadMe is a prototype and should be refined further to better help environmental researchers. <br>
Please change base_dir in checkColab function within GetValues.py to set base directory. <br>
This pipeline is designed for image and xml annotation pairs. <br>
Image and xml annotations should have the exact same name to be associated with each other. <br>
The pipeline assumes that datasets are structured in the following format: <br>
</p>

```bash
├───Example dataset structure    
│   ├───Test                     
│   │   ├───annotations
│   │   │   └───xmls
│   │   │       └─── A9X3Z1.xml
│   │   └───images
│   │           └─── A9X3Z1.jpg
│   ├───Train
│   │   ├───annotations
│   │   │   └───xmls
│   │   │       └─── M7B4Q8.xml
│   │   └───images
│   │           └─── M7B4Q8.jpg
│   └───Validate
│       ├───annotations
│       │   └───xmls
│       │       └─── X2Y9W5.xml
│       └───images
│           └─── X2Y9W5.jpg
```
<p>
A utility file named ShuffleData.py can be used to split a dataset into the above format. <br>
ShuffleData assumes the below file structure, again image and xml names should be identical.
</p>

```bash
├───Example dataset structure
│   ├───Main
│   │   ├───annotations
│   │   │   └───xmls
│   │   │       └─── P5L8D2.xml
│   │   └───images
│   │       └─── P5L8D2.jpg
```
Output of ShuffleData.py, keeps Main dataset intact:
```bash
├───Example dataset structure
│   ├───Main
│   │   ├───annotations
│   │   │   └───xmls
│   │   └───images
│   ├───Test
│   │   ├───annotations
│   │   │   └───xmls
│   │   └───images
│   ├───Train
│   │   ├───annotations
│   │   │   └───xmls
│   │   └───images
│   └───Validate
│       ├───annotations
│       │   └───xmls
│       └───images

```


<p>
Another consideration is the XML annotation structure. <br>
This pipeline assumes the following xml format with bounding box coordinates in PASCAL VOC format. <br>
To modify the pipeline for another xml structure or BBOX format, open SmokeUtils.py and edit extract_boxes. <br>
extract_boxes is used for all xml extraction EXCEPT in CheckDataset.py. <br>
CheckDataset is intended to be a simple way to check a bbox on a dataset image, this was done to keep the file simple.
</p>

```xml
<annotation>
    <folder></folder>
    <filename></filename>
    <path></path>
    <source>
        <database></database>
    </source>
    <size>
        <width></width>
        <height></height>
        <depth></depth>
    </size>
    <segmented></segmented>
    <object>
        <name></name>
        <pose></pose>
        <truncated></truncated>
        <difficult></difficult>
        <bndbox>
            <xmin></xmin>
            <ymin></ymin>
            <xmax></xmax>
            <ymax></ymax>
        </bndbox>
    </object>
</annotation>
```

<h2> Pipeline directory </h2>

```bash
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----        03/03/2025     15:21                Dataset - Create and store any dataset folders here, as seen above
d-----        04/02/2025     17:48                DatasetH5py - Create and store any hdf5 files here 
d-----        03/03/2025     15:16                FasterR-CNN - Contains pipeline scripts, models will also be saved here automatically
d-----        19/09/2024     18:18                Lib - Contains modified site packages, including Faster-RCNN, backbones, etc
d-----        12/01/2025     15:00                Libr - Includes third-party scripts, including coco evaluator
-a----        03/03/2025     15:45            575 .gitignore
-a----        03/03/2025     15:42           2485 README.md
```
<h2> Main pipeline files </h2>

```bash
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----        02/04/2025     15:35                savedModels - Directory contains files created during training/testing, e.g. model weights, logs, etc
-a----        29/03/2025     21:13           2425 CheckDataset.py - Simple file to test image and annotations, plots image with ground truth overlayed
-a----        31/03/2025     14:57          10849 Dataset.py - PyTorch dataset, used to get data ready for model input, including transforms/converting to tensor
-a----        29/03/2025     21:30           4207 DatasetHdf5.py - Dataset to handle created HDF5 files, used instead of Dataset.py
-a----        29/03/2025     21:30           1410 EpochSampler.py - Custom epoch sampler to keep track of position in HDF5 file, necessary for HDF5 dataset only
-a----        02/04/2025     15:36           5783 GetValues.py - Contains pipeline settings, used as a convenient hub for model management
-a----        02/04/2025     13:42            891 Logger.py - Used to log script output to console and txt files simultaneously
-a----        29/03/2025     21:30           5833 PlotGraphs.py - Plots loss graphs for training and validation loss during training
-a----        04/03/2025     02:22           4307 ShuffleData.py - Creates random train, validation and test splits, gets data ready for training
-a----        31/03/2025     15:15          18804 SmokeModel.py - Initialises models and dataloaders
-a----        29/03/2025     21:32           2841 SmokeUpscale.py - Can be used to AI upscale a dataset for repeated use, dynamic upscale is also available in Trainer/Tester
-a----        30/03/2025     00:37           8523 SmokeUtils.py - Contains misc files that are likely to require modification during pipeline usage 
-a----        29/03/2025     21:34           9604 SplitImageIntoSize.py - Splits images based on ground truth size, into small, medium and large dirs
-a----        02/04/2025     15:36          50012 Tester.py - Can be used to test any models trained using this pipeline
-a----        02/04/2025     12:55          17946 Trainer.py - Script to train a model on a target dataset/problem
-a----        31/03/2025     15:48          11944 TrainingSteps.py - Contains training and validation steps used during model training
-a----        29/03/2025     23:08           6777 WriteHdf5.py - Creates HDF5 files using Dataset.py, HDF5 file can then be loaded during training
-a----        09/01/2025     14:22              0 __init__.py

```
<h2> Modified site packages </h2>

```bash
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----        26/02/2025     01:19          54274 mean_ap.py - Fixed inconsistent exception
-a----        02/04/2025     12:40          11219 feature_pyramid_network.py - Plot feature maps
-a----        03/03/2025     13:40          40686 resnet.py - Modified to be compatible with quantization
-a----        10/02/2025     20:27          11308 backbone_utils.py - Modified to be compatible with quantization
-a----        12/01/2025     17:06          36024 roi_heads.py - Modified return values
-a----        12/01/2025     15:42          16621 rpn.py - Modified return values
-a----        02/02/2025     18:13           6033 generalized_rcnn.py - Modified return values
-a----        04/03/2025     02:46          42491 faster_rcnn.py - Make FPNv2 available for other resnet backbones





```

<h2> Training a model: </h2>
<p>
Once data preperation is complete, and dataset/directory is correctly setup <br>
Open GetValues, this is the configuration file for the pipeline. <br>
Hyperparameters, model backbone and etc can be changed here. <br>
Once configured open Trainer.py and run the script, the model will automatically begin training. <br>
Model logs, weights, etc will automatically be saved inside - <br>
FasterR-CNN > SavedModels > model_name set in GetValues <br>
</p>

<h2> Evaluating a model: </h2>
<p>
Make sure within GetValues the model_name value in test settings is the same as the name of the trained model <br>
Open Tester.py and run the file. The model will be automatically evaluated.
There are many options at the top of the Tester.py file relating to visualisation, metrics, etc
</p>


<h2> Using strategies </h2>
<p> 
Several strategies were outlined in the paper which accompanies this file <br>
Below is a guide on how to use each strategy.


</p>

<h3> Quantization </h3>
<p>
*Note quantization is implemented for CPU only, device will change automatically when enabled <br>
Setup (ignore if using default or resnet_builder as backbone): <br>
Open SmokeUtils.py, edit the get_layers_to_fuse function <br>
Ensure arithmetic doesn't occur between fused layers. <br>
Edit the backbone model, ensuring that any arithmetic is replaced with floatfunctional equivalents <br>  
https://pytorch.org/docs/stable/generated/torch.ao.nn.quantized.FloatFunctional.html <br>
Editing backbone_utils.py and faster_rcnn.py (from torchvision/detection) may be necessary depending on the model used. <br>
Post-static quants: <br>
Enable static_quant in test setting of GetValues.py and test as normal <br>
Quant-Aware-Training: <br>
Enable quant_aware_training in training settings of GetValues.py and train <br>
When evaluating, enable load_QAT_model in test setting of GetValues.py and evaluate as normal

</p>
<h3> Reduced precision </h3>
<p> 
Enable half_precision in GetValues.py for training or testing
</p>
<h3> Automatic mixed precision training </h3>
<p> 
Enable amp_mixed_precission within GetValues.py within training settings 
</p>
<h3> Pruning </h3>
<p> 
Enable pruning within testing setting of GetValues.py <br>
If not using default or resnet_builder backbones (see GetValues.py) <br>
In SmokeUtils.py alter get_layers_to_prune, adding layers to prune based on the backbone used <br>
Pruning strategy can be altered further, see prune_model method in Tester.py
</p>
<h3> AI upscaling </h3>
<p> 
Pre-upscaling images before training: <br>
A utility file is provided to upscale images for repeated use, SmokeUpscale.py <br>
In GetValues.py set upscale value to the desired multiplier, e.g. 2 upscales images to 2x their resolution <br>
batch size in training settings are also used during pre-upscaling. <br>
The upscaler creates a new dataset with upscaled images but doesn't copy xml. <br>
These images can then be copied into another dataset or xml can be added to the created dataset. <br>
SmokeUpscale output:
</p>

``` bash
├───SmokeUpscale output
│   ├───Train
│   │   └───images
│   └───Validate
│       └───images
```

<p>
Training: <br>
Using pre-upscaled images: <br>
Within GetValues.py enable upscale_bbox and ensure upscale_value is the value used to upscale images during pre-upscaling <br>
Then train as normal. <br>

Dynamic upscaling: <br>
In GetValues enable upscale_image, and set upscale_value in either train or test settings then train or evaluate as normal <br>

Future improvements:
Looking back it would have been much easier to just take the xml annotations and update the coordinates using the upscale value <br>
before placing them in the created dataset with the upscaled images. This should be changed for future use. <br>
However the current implementation will remain for this prototype.

Split or merge bboxes during evaluation:
In GetValues enable split_images in test settings
For mergin within Tester.py enable combine_bboxes at the top of the file with other settings <br>
Below combine_bboxes, merge_tolerance can be changed. This is the max distance between bboxes that can be merged
</p>



<h3> Image transformations </h3>
<p> 
Open Dataset.py and add transformations at the top of the page using albumentations. <br>
Transform types are labelled e.g. train_transform, validate_transform, etc <br>
Some transforms are left commented out for convenience, see albumentations for full list <br>
https://albumentations.ai/docs/api_reference/augmentations/transforms/
</p>
<h3> Finetune from previously trained model </h3>
<p> 
Enable start_from_checkpoint in GetValues.py within the training settings <br>
Make sure to enter the name of the model to be loaded in model_load_name just below it <br>
This assumes model was trained with this pipeline and is in the same format/dir
</p>
<h3> Create/use hdf5 files </h3>
<p> 
Create hdf5 file:
In GetValues.py set the name to save the hdf5 file as. <br> 
Make sure load hdf5 is disabled if you want to create a hdf5 file from default dataset/dataloaders <br>
Training epoch is used to set how many epochs to write to hdf5. Batch size is used for dataloaders as normal.
A hdf5 file will be automatically created using whatever values, transforms, etc settings are set for dataloaders in SmokeModel.py and Dataset.py <br>
Load hdf5 file:
Once the file has been created, enable load_hdf5 in GetValues.py, make sure the h5py_dir_load_name is correct!
</p>
<h3> Pinned/non-blocking </h3>
<p> 
Enable pinned/non-blocking in GetValues.py for training or testing <br>
Its recommended to use both simultaneously instead of individually
</p>

<h3> Splitting images based on ground truth area </h3>
<p>
SplitImageIntoSize can be used to split images based on gt area. <br>
Takes the following dataset structure: 
</p>

```bash
├───Example dataset structure
│   ├───Main
│   │   ├───annotations
│   │   │   └───xmls
│   │   └───images
│   ├───Test
│   │   ├───annotations
│   │   │   └───xmls
│   │   └───images
│   ├───Train
│   │   ├───annotations
│   │   │   └───xmls
│   │   └───images
│   └───Validate
│       ├───annotations
│       │   └───xmls
│       └───images

```

<p>
It then categorises images by size in each split. This could probably be improved in the future <br>
</p>

```bash
├───Example dataset structure
│   ├───Test
│   │   ├───annotations
│   │   │   └───xmls
│   │   │       ├───large
│   │   │       ├───medium
│   │   │       ├───none
│   │   │       └───small
│   │   └───images
│   │       ├───large
│   │       ├───medium
│   │       ├───none
│   │       └───small
│   ├───Train
│   │   ├───annotations
│   │   │   └───xmls
│   │   │       ├───large
│   │   │       ├───medium
│   │   │       ├───none
│   │   │       └───small
│   │   └───images
│   │       ├───large
│   │       ├───medium
│   │       ├───none
│   │       └───small
│   └───Validate
│       ├───annotations
│       │   └───xmls
│       │       ├───large
│       │       ├───medium
│       │       ├───none
│       │       └───small
│       └───images
│           ├───large
│           ├───medium
│           ├───none
│           └───small

```

<h1> Cloud computing</h1>
<p>
A jupyter notebooks is provided in the repo for use with google colab <br>
Setup: <br>
In google drive create a base directory for dataset files and to store any saved models. <br>
Add a folder named "Dataset" and store datasets within, as you would when running locally. <br>
Add a folder named "DatasetH5py" and store hdf5 files within, as you would when running locally. <br>
Model weights, json, plots, etc will automatically be stored in a folder called "FasterR-CNN within the base directory" <br>
Within the jupyter notebook, at the top alter "base_dir" and point it to the created base directory within google drive <br>
The following directory structure should be used in google drive. <br>
By default, the notebook points to this repository, to use for other tasks clone this repository and use it to create a new one. <br>
Change "!git clone https://github.com/Stuart1389/SmokeFasterRCNN.git" within SmokePipeline.ipynb to the new repository.
</p>

``` bash
├───BaseDir
│   ├───Dataset
│   │   └───Example dataset structure (see above)
│   └───DatasetH5py
│       └───Example hdf5 dataset
│           ├───Train.hdf5
│           └───Validate.hdf5
```