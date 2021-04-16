from torch import Tensor
from torch.nn import Module, ModuleList
from torchvision.models._utils import IntermediateLayerGetter


class IntermediateLayerModelVGG():
    def __init__(self, model, return_layer):
        self.model = IntermediateLayerGetter(model.features, return_layer)
        self.return_key = list(return_layer.values())[0]

    def __call__(self, x, *args, **kwargs):
        output = self.model(x, *args, **kwargs)[self.return_key]
        return output