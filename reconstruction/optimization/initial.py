from mei.initial import InitialGuessCreator
from torch import Tensor, randn
import torch
from ..schema.main import ReconstructionImages


class RandomNormalRGB(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        initial = self._create_random_tensor(*shape)
        return initial.repeat(1, 3, 1, 1)

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


class OriginalImage(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    def __init__(self, image_class, image_id):

        image = torch.from_numpy((ReconstructionImages & dict(image_class=image_class, image_id=image_id)).fetch1("image"))
        self.natural_img = image.repeat(1, 3, 1, 1)

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        return self.natural_img

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"