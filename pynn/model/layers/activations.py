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
    
    def _positive_sigmoid(self, x):
        '''Positive Sigmoid Function.'''
        return 1 / (1 + np.exp(-x))
    
    def _negative_sigmoid(self, x):
        '''Negative Sigmoid Function.'''
        exp = np.exp(x)
        return exp / (exp + 1)
    
    def _sigmoid(self, x:np.ndarray) -> np.ndarray:
        '''
        Sigmoid activation function - **y = 1 / (1 + e^-x)**.
        **Parameters:**
            `x`: input data.
        **Returns:**
            `y`: output data.
        '''
        positive = x >= 0
        negative = ~positive
        y = np.empty_like(x)
        y[positive] = self._positive_sigmoid(x[positive])
        y[negative] = self._negative_sigmoid(x[negative])
        return y

    def _sigmoid_grad(self) -> np.ndarray:
        '''
        Sigmoid gradient - **y = fx * (1 - fx)**, where **fx** is the output of the sigmoid function.
        **Returns:**
            `y`: output data.
        '''
        return self.fx * (1 - self.fx)

    def __call__(self, x:np.ndarray) -> np.ndarray:
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

class Softmax(object):
    '''Softmax Layer.'''
    def __init__(self, name:np.str="Softmax"):
        '''
        Initialize the Softmax Layer.
        **Parameters:**
            `name`: name of the layer.
        '''
        self.name = name
        # Forward prop
        self.fx = None
        # Backward prop
        self.dfx = None
    
    def __repr__(self):
        return f"Softmax Layer: {self.name}"
    
    def __str__(self):
        return f"Softmax Layer: {self.name}"
    
    def _softmax(self, x:np.ndarray) -> np.ndarray:
        '''
        Softmax activation function - **y = e^x / sum(e^x)**.
        **Parameters:**
            `x`: input data.
        **Returns:**
            `y`: output data.
        '''
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)
    
    def _softmax_grad(self, upstream_grad) -> np.ndarray:
        '''
        Softmax gradient - **y = fx * (1 - fx)**, where **fx** is the output of the softmax function.
        **Parameters:**
            `upstream_grad`: upstream gradient.
        **Returns:**
            `y`: output data.
        '''
        return self.fx * (upstream_grad  - (upstream_grad - self.fx).sum(axis=1)[:, None])

    def __call__(self, x:np.ndarray) -> np.ndarray:
        '''
        Compute forward transformation - **y = e^x / sum(e^x)**.
        **Parameters:**
            `x`: input data.
        **Returns:**
            `y`: output data.
        '''
        self.fx = self._softmax(x)
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