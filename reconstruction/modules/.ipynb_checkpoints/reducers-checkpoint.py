from torch import Tensor
from torch.nn import Module, ModuleList


class ConstrainedOutputModel(Module):
    """A model that has its output constrained.

    Attributes:
        model: A PyTorch module.
        constraint: An integer representing the index of a neuron in the model's output. Only the value corresponding
            to that index will be returned.
        target_fn: Callable, that gets as an input the constrained output of the model.
        forward_kwargs: A dictionary containing keyword arguments that will be passed to the model every time it is
            called. Optional.
    """

    def __init__(self, model: Module, target_fn=None, forward_kwargs=None):
        """Initializes ConstrainedOutputModel."""
        super().__init__()
        if target_fn is None:
            target_fn = lambda x: x
        self.model = model
        self.target_fn = target_fn
        self.forward_kwargs = forward_kwargs if forward_kwargs is not None else dict()

    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Computes the constrained output of the model.

        Args:
            x: A tensor representing the input to the model.
            *args: Additional arguments will be passed to the model.
            **kwargs: Additional keyword arguments will be passed to the model.

        Returns:
            A tensor representing the constrained output of the model.
        """
        output = self.model(x, *args, **self.forward_kwargs, **kwargs)
        return self.target_fn(output)

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self.model})"
