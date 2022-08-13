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

    def backward(self, dy) -> np.ndarray:
        '''
        Backward pass of the Sigmoid layer.
        Params:
            dy - gradient of the loss with respect to the output of the Sigmoid layer (n_features, 1)
        Returns:
            dl - gradient of the loss with respect to the input of the Sigmoid layer (n_features, 1)
        '''
        dl = dy * (1 - self.fx) * self.fx
        return dl
    
    def __repr__(self) -> str:
        return f'{self.name}'
    
    def __str__(self) -> str:
        return f'{self.name}'

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

    def backward(self) -> np.ndarray:
        '''
        Backward pass of the Relu layer.
        Params:
            dl - gradient of the loss with respect to the output of the Relu layer (n_features, 1)
        '''
        dl = np.where(self.fx > 0, 1, 0)
        return dl
    
    def __repr__(self) -> str:
        return f'{self.name}'
    
    def __str__(self) -> str:
        return f'{self.name}'

class Softmax(object):
    def __init__(self, name: str = 'Softmax') -> None:
        '''
        Initializes a new Softmax layer.
        '''
        self.fx = None
        self.name = name

    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Forward pass of the Softmax layer.
        Params:
            x - input (n_features, 1)
        Returns:
            fx - output of Softmax layer (n_features, 1)
        '''
        ex = np.exp(x - np.max(x))
        self.fx =  ex / ex.sum()
        return self.fx

    def backward(self, dl: np.ndarray) -> np.ndarray:
        '''
        Backward pass of the Softmax layer.
        Params:
            dl - gradient of the loss with respect to the output of the Softmax layer (n_features, 1)
        '''
        sm = self.fx.reshape((-1,1))
        return np.diagflat(self.fx) - np.dot(sm, sm.T)
    
    def __repr__(self) -> str:
        return f'{self.name}'
    
    def __str__(self) -> str:
        return f'{self.name}'