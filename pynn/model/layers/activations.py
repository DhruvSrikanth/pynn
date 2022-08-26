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

