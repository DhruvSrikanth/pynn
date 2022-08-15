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

        # Forward prop
        self.fx = None

        # Backward prop
        self.dfx = None

    
    def __repr__(self):
        return f"Sigmoid Layer: {self.name}"
    
    def __str__(self):
        return f"Sigmoid Layer: {self.name}"
    
    def _sigmoid(self, x:np.ndarray) -> np.ndarray:
        '''
        Sigmoid activation function - **y = 1 / (1 + e^-x)**.
        **Parameters:**
            `x`: input data.
        **Returns:**
            `y`: output data.
        '''
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_grad(self) -> np.ndarray:
        '''
        Sigmoid gradient - **y = fx * (1 - fx)**, where **fx** is the output of the sigmoid function.
        **Returns:**
            `y`: output data.
        '''
        return self.fx * (1 - self.fx)

    def forward(self, x:np.ndarray) -> np.ndarray:
        '''
        Compute forward transformation - **y = 1 / (1 + e^-x)**.
        **Parameters:**
            `x`: input data.
        **Returns:**
            `y`: output data.
        '''
        self.fx = self._sigmoid(x)
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