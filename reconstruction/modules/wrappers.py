from torch import Tensor
from torch.nn import Module, ModuleList
from torchvision.models._utils import IntermediateLayerGetter
from nntransfer.models.wrappers.intermediate_layer_getter import (
    IntermediateLayerGetter as ILG
)


class IntermediateLayerModelVGG():
    def __init__(self, model, blocks, return_layer):
        submodel = model
        for block in blocks:
            submodel = getattr(submodel, block)
        self.model = IntermediateLayerGetter(submodel, return_layer)
        self.return_key = list(return_layer.values())[0]

    def __call__(self, x, *args, **kwargs):
        output = self.model(x, *args, **kwargs)[self.return_key]
        return output

class IntermediateLayerResNet50():
    def __init__(self, model, return_layers):
        self.model = ILG(model, return_layers, keep_output=True)
        self.return_key = list(return_layers.values())[0]

    def __call__(self, x, *args, **kwargs):
        activations, _ = self.model(x, *args, **kwargs)
        return activations[self.return_key]