import numpy as np


class LeakyReLU(object):
    '''Leaky Rectified Linear Unit'''
    def __init__(self, negative_slope:np.float64=0.2, name:np.str="ReLU"):
        '''
        Initialize the ReLU Layer.
        **Parameters:**
            `name`: name of the layer.
        '''
        self.negative_slope = negative_slope
        self.name = name
        self.x = None
        self.fx = None
        self.dfx = None
    
    def __repr__(self):
        return f"Leaky ReLU Layer: {self.name}"
    
    def __str__(self):
        return f"Leaky ReLU Layer: {self.name}"
    
    def _leaky_relu(self):
        '''
        Leaky ReLU activation function - **y = max(negative_slope*x, x)**.
        **Returns:**
            `y`: output data.
        '''
        fx = []
        for x in self.x:
            if x >= 0:
                fx.append(x)
            else:
                fx.append(self.negative_slope * x)
        return fx
    
    def _leaky_relu_grad(self) -> np.ndarray:
        '''
        ReLU gradient - **y = 1 if x > 0 else negative slope**.
        **Returns:**
            `y`: output data.
        '''
        return np.where(self.x > 0, 1, self.negative_slope)
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        '''
        Compute forward transformation - **y = max(0, x)**.
        **Parameters:**
            `x`: input data.
        **Returns:**
            `y`: output data.
        '''
        self.x = np.copy(x)
        self.fx = self._leaky_relu()
        return self.fx
    
    def backward(self, upstream_grad:np.ndarray) -> np.ndarray:
        '''
        Compute downstream gradient.
        **Parameters:**
            `upstream_grad`: upstream gradient.
        **Returns:**
            `dfx`: downstream gradient.
        '''
        self.dfx = upstream_grad * self._leaky_relu_grad()
        return self.dfx

