from mei.initial import InitialGuessCreator
from torch import Tensor, randn


class RandomNormalRGB(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        initial = self._create_random_tensor(*shape)
        return initial.repeat(1, 3, 1, 1)

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"