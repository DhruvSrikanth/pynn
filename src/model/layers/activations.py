import numpy as np

class Sigmoid(object):
    def __init__(self, name: str = 'Sigmoid') -> None:
        '''
        Initialize the Sigmoid layer.
        '''
        self.fx = None
        self.name = name

    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Forward pass of the Sigmoid layer.
        Params:
            x - input (n_features, 1)
        Returns:
            fx - output of sigmoid function (n_features, 1)
        '''
        self.fx = 1 / (1 + np.exp(-x))
        return self.fx

    def backward(self, dy):
        '''
        Backward pass of the Sigmoid layer.
        Params:
            dy - gradient of the loss with respect to the output of the Sigmoid layer (n_features, 1)
        Returns:
            dl - gradient of the loss with respect to the input of the Sigmoid layer (n_features, 1)
        '''
        dl = dy * (1 - self.fx) * self.fx
        return dl

class Relu(object):
    def __init__(self, name: str = 'Relu') -> None:
        '''
        Initializes a new Relu layer.
        '''
        self.fx = None
        self.name = name

    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Forward pass of the Relu layer.
        Params:
            x - input (n_features, 1)
        Returns:
            fx - output of Relu layer (n_features, 1)
        '''
        self.fx = np.maximum(0, x)
        return self.fx

    def backward(self):
        '''
        Backward pass of the Relu layer.
        Params:
            dl - gradient of the loss with respect to the output of the Relu layer (n_features, 1)
        '''
        dl = np.where(self.fx > 0, 1, 0)
        return dl
