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


class IntermediateLayerModelVGGLoop():
    def __init__(self, model, layer):
        self.features = model.features
        self.layer = layer

    def __call__(self, x, *args, **kwargs):
        for name, child in self.features.named_children():
            if int(name) <= self.layer:
                x = self.features[int(name)](x)
        return x