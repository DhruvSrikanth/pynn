import numpy as np

class Sigmoid(object):
    '''Sigmoid Layer.'''
    def __init__(self, name:np.str="Sigmoid"):
        '''
        Initialize the Sigmoid Layer.
        **Parameters:**
            `name`: name of the layer.
        '''
        self.name = name
        self.x = None
        self.fx = None
        self.dfx = None
    
    def __repr__(self):
        return f"Sigmoid Layer: {self.name}"
    
    def __str__(self):
        return f"Sigmoid Layer: {self.name}"
    
    def _sigmoid(self) -> np.ndarray:
        '''
        Sigmoid activation function - **y = 1 / (1 + e^-x)**.
        **Returns:**
            `y`: output data.
        '''
        return np.exp(self.x) / (1 + np.exp(self.x))

    def _sigmoid_grad(self) -> np.ndarray:
        '''
        Sigmoid gradient - **y = fx * (1 - fx)**, where **fx** is the output of the sigmoid function.
        **Returns:**
            `y`: output data.
        '''
        return self.fx * (1. - self.fx)

    def __call__(self, x:np.ndarray) -> np.ndarray:
        '''
        Compute forward transformation - **y = 1 / (1 + e^-x)**.
        **Parameters:**
            `x`: input data.
        **Returns:**
            `y`: output data.
        '''
        self.x = np.copy(x)
        self.fx = self._sigmoid()
        return self.fx

    def backward(self, upstream_grad:np.ndarray) -> np.ndarray:
        '''
        Compute downstream gradient.
        **Parameters:**
            `upstream_grad`: upstream gradient.
        **Returns:**
            `downstream_grad`: downstream gradient.
        '''
        self.dfx = upstream_grad * self._sigmoid_grad()
        return self.dfx
