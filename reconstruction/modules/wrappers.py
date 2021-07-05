from torch import Tensor
from torch.nn import Module, ModuleList
from torchvision.models._utils import IntermediateLayerGetter


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
