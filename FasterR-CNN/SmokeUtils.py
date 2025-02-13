from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union
from torch import nn, Tensor
import torch
import torch.ao.quantization as quantization
import matplotlib.pyplot as plt
import cv2
import numpy as np
from GetValues import setTrainValues

# modified layer getter class to qork with quants
class SmokeIntermediateLayerGetter(nn.ModuleDict):
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        out = OrderedDict()
        extracted_features = []
        extracted_names = []

        for name, module in self.items():
            # if we try to quant when already quanted
            # will throw error, so if module has a quant stub then dequant before
            # cant really use module since other layer types contain quant
            # resnet names their quant layers "quant" so better than just dequanting at specific layers
            # since it works with all resnet
            #print(f"Name: {name}, module: {str(module)}")
            if ("quant" in name):
                x = self.dequant(x)
            x = module(x)
            if name in self.return_layers:
                #print(self.return_layers)
                out_name = self.return_layers[name]
                out[out_name] = self.dequant(x)
                #print(out[out_name].shape)
                extracted_features.append(out[out_name])
                extracted_names.append(name)
        return out

def get_layers_to_fuse(module_names):
    """
    resnet built with multiple bottleneck blocks which can be seen below. ideally we look for this
    set of layers, c1,b1,r,c2,b2,r,c3,b3 and if we find them we add them to a list
    this list will then be used to fuse bottleneck blocks for qat
    just doing conv and bn because relus have same name (.relu) meaning that its
    a massive headache to extract them, could also list them manually if wanted to
    fusing layers together, notice that these are the layers
    defined in resnet Bottleneck which makes up the bulk
    of the resnet backbones

    """

    # looking for conv and bn, put occurence in layers to fuse list
    # which we will use during quantisation
    layers_to_fuse = []  # get a list of layers in backbone to fuse
    layers_to_fuse.append(['conv1', 'bn1', 'relu'])
    # example of adding manually
    """
    layers_to_fuse.append(['conv1', 'bn1', 'relu'])
    layers_to_fuse.append(['layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu1'])
    layers_to_fuse.append(['layer1.0.conv2', 'layer1.0.bn2', 'layer1.0.relu2'])
    layers_to_fuse.append(['layer1.0.conv3', 'layer1.0.bn3', 'layer1.0.relu'])
    layers_to_fuse.append(['layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.relu1'])
    layers_to_fuse.append(['layer1.1.conv2', 'layer1.1.bn2', 'layer1.1.relu2'])
    """
    for i in range(len(module_names)):
        name, module = module_names[i]
        #print(name)
        if(i > 2):
            # skip bn3 and relu fuse because down sampling happens in between
            if 'conv' in name and 'bn' in module_names[i + 1][0] and 'relu' in module_names[i + 2][0] and 'bn3' not in module_names[i + 1][0]:
                layers_to_fuse.append([module_names[i][0], module_names[i + 1][0], module_names[i + 2][0]])
            elif 'conv' in name and 'bn' in module_names[i + 1][0]:
                layers_to_fuse.append([module_names[i][0], module_names[i + 1][0]])

    #print(layers_to_fuse)
    return layers_to_fuse
