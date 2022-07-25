import numpy as np

class ImageTransforms(object):
    """
    Class to handle the transforms.
    """
    def __init__(self) -> None:
        """
        Initialize the transforms.
        Parameters:
            None
        Returns:
            None
        """
        pass

    def resize_image(self, image: np.ndarray, dims: tuple) -> np.ndarray:
        """
        Resize an image to the given dimensions.
        Parameters:
            dims: (type tuple) dimensions to resize the image to.
            image: (type numpy.ndarray) image to resize.
        Returns:
            image: (type numpy.ndarray) resized image.
        """
        return image.resize(dims)

    def normalize_tensor(self, tensor: np.ndarray ) -> np.ndarray:
        """
        Normalize a tensor.
        Parameters:
            tensor: (type numpy.ndarray) tensor to normalize.
        Returns:
            tensor: (type numpy.ndarray) normalized tensor.
        """
        return tensor / 255.0
