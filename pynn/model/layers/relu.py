import numpy as np

class ReLU(object):
    '''Rectified Linear Unit'''
    def __init__(self, name:np.str="ReLU"):
        '''
        Initialize the ReLU Layer.
        **Parameters:**
            `name`: name of the layer.
        '''
        self.name = name
        self.x = None
        self.fx = None
        self.dfx = None
    
    def __repr__(self):
        return f"ReLU Layer: {self.name}"
    
    def __str__(self):
        return f"ReLU Layer: {self.name}"
    
    def _relu(self) -> np.ndarray:
        '''
        ReLU activation function - **y = max(0, x)**.
        **Returns:**
            `y`: output data.
        '''
        return np.clip(self.x, 0, None)
    
    def _relu_grad(self) -> np.ndarray:
        '''
        ReLU gradient - **y = fx > 0**, where **fx** is the output of the ReLU function.
        **Returns:**
            `y`: output data.
        '''
        return np.where(self.x > 0, 1, 0)
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        '''
        Compute forward transformation - **y = max(0, x)**.
        **Parameters:**
            `x`: input data.
        **Returns:**
            `y`: output data.
        '''
        self.x = np.copy(x)
        self.fx = self._relu()
        return self.fx
    
    def backward(self, upstream_grad:np.ndarray) -> np.ndarray:
        '''
        Compute downstream gradient.
        **Parameters:**
            `upstream_grad`: upstream gradient.
        **Returns:**
            `dfx`: downstream gradient.
        '''
        self.dfx = upstream_grad * self._relu_grad()
        return self.dfx
