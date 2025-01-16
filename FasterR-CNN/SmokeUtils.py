from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union
from torch import nn, Tensor
import torch
import torch.ao.quantization as quantization

# for static quants
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
        for name, module in self.items():
            # so pretty much if we try to quant when already quanted
            # will throw error, so if module has a quant stub then dequant before
            # cant really use module since other layer types contain quant
            # resnet names their quant layers "quant" so better than just dequanting at specific layers
            # sicne it works with all resnet
            # come back to this
            #print(f"Name: {name}, module: {str(module)}")
            if ("quant" in name):
                x = self.dequant(x)
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = self.dequant(x)

        return out

def get_layers_to_fuse(module_names):
    """
    resnet built with multiple bottleneck blocks which can be seen below. ideally we look for this
    set of layers, c1,b1,r,c2,b2,r,c3,b3 and if we find them we add them to a list
    this list will then be used to fuse bottleneck blocks for qat
    just doing conv and bn because relus have same name (.relu) meaning that its
    a massive headache to extract them, could also list them manually if wanted to
     # fusing layers together, notice that these are the layers
     # defined in resnet Bottleneck which makes up the bulk
     # of the resnet backbones

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)
    """

    # looking for conv and bn, put occurence in layers to fuse list
    # which we will use during quantisation
    layers_to_fuse = []  # get a list of layers in backbone to fuse
    # example of adding manually
    layers_to_fuse.append(['conv1', 'bn1', 'relu'])
    layers_to_fuse = []  # get a list of layers in backbone to fuse
    # example of adding manually
    layers_to_fuse.append(['conv1', 'bn1', 'relu'])

    for i in range(len(module_names)):
        name, module = module_names[i]
        print(name)
        if(i > 2):
            if 'conv' in name and 'bn' in module_names[i + 1][0]:
                    layers_to_fuse.append([module_names[i][0], module_names[i + 1][0]])
    return layers_to_fuse
    #print(layers_to_fuse)
