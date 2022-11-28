import torch
from mei.initial import InitialGuessCreator
from torch import Tensor, randn

from ..schema.main import ReconstructionImages


class RandomNormalRGB(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    _create_random_tensor = randn

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        initial = self._create_random_tensor(*shape)
        if initial.shape[1] == 1:
            initial = initial.repeat(1, 3, 1, 1)
        return initial

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


class OriginalImage(InitialGuessCreator):
    """Used to create an initial guess tensor filled with values distributed according to a normal distribution."""

    def __init__(self, image_class, image_id):

        image = torch.from_numpy(
            (
                ReconstructionImages & dict(image_class=image_class, image_id=image_id)
            ).fetch1("image")
        )
        self.natural_img = image.repeat(1, 3, 1, 1)

    def __call__(self, *shape):
        """Creates a random initial guess from which to start the MEI optimization process given a shape."""
        return self.natural_img

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


class OriginalImageScaled(InitialGuessCreator):
    """
    Fetches the original image - i.e. the image on which the reconstruction is
    based - scales it down according to the norm constraint, and returns
    this scaled image
    """

    def __init__(self, key, norm_fraction):

        img = (ReconstructionImages & key).fetch1("image")  # returns numpy array
        img_tensor = torch.Tensor(img.copy())
        self.scaled_img = img_tensor * norm_fraction

    def __call__(self, *shape):
        return self.scaled_img

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"
