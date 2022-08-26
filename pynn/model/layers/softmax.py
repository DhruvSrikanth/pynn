import numpy as np

class Softmax(object):
    '''Softmax Layer.'''
    def __init__(self, name:np.str="Softmax"):
        '''
        Initialize the Softmax Layer.
        **Parameters:**
            `name`: name of the layer.
        '''
        self.name = name
        self.x = None
        self.fx = None
        self.dfx = None
    
    def __repr__(self):
        return f"Softmax Layer: {self.name}"
    
    def __str__(self):
        return f"Softmax Layer: {self.name}"
    
    def _softmax(self) -> np.ndarray:
        '''
        Softmax activation function - **y = e^x / sum(e^x)**.
        **Returns:**
            `y`: output data.
        '''
        return np.exp(self.x) / np.exp(self.x).sum(axis=1)[:, None]
    
    def _softmax_grad(self, upstream_grad) -> np.ndarray:
        '''
        Softmax gradient - **y = fx * (1 - fx)**, where **fx** is the output of the softmax function.
        **Parameters:**
            `upstream_grad`: upstream gradient.
        **Returns:**
            `y`: output data.
        '''
        return self.fx * (upstream_grad - (upstream_grad * self.fx).sum(axis=1)[:, None])

    def __call__(self, x:np.ndarray) -> np.ndarray:
        '''
        Compute forward transformation - **y = e^x / sum(e^x)**.
        **Parameters:**
            `x`: input data.
        **Returns:**
            `y`: output data.
        '''
        self.x = np.copy(x)
        self.fx = self._softmax()
        return self.fx

    def backward(self, upstream_grad:np.ndarray) -> np.ndarray:
        '''
        Compute downstream gradient.
        **Parameters:**
            `upstream_grad`: upstream gradient.
        **Returns:**
            `downstream_grad`: downstream gradient.
        '''
        self.dfx = self._softmax_grad(upstream_grad)
        return self.dfx
