from torch import Tensor
from torch.nn import Module, ModuleList
from torchvision.models._utils import IntermediateLayerGetter


class IntermediateLayerOutput(Module):
    """A model that has its output constrained.

    Attributes:
        model: A PyTorch module.
        return_layer: example:  {'0': 'out_layer0',}
        target_fn: Callable, that gets as an input the constrained output of the model.
    """

    def __init__(self, model, return_layer, target_fn=None, ):
        """Initializes ConstrainedOutputModel."""
        super().__init__()
        if target_fn is None:
            target_fn = lambda x: x
        if len(list(return_layer.keys())) > 1:
            raise ValueError("Output can only be constraint with a single layer")
        self.model = IntermediateLayerGetter(model.features, return_layer)
        self.return_key = list(return_layer.values())[0]
        self.target_fn = target_fn

    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Computes the constrained output of the model.

        Args:
            x: A tensor representing the input to the model.
            *args: Additional arguments will be passed to the model.
            **kwargs: Additional keyword arguments will be passed to the model.

        Returns:
            A tensor representing the constrained output of the model.
        """
        output = self.model(x, *args, **kwargs)[self.return_key]
        return self.target_fn(output)

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self.model}, )"
