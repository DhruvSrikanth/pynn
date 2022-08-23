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
        max_x = np.zeros((x.shape[0], 1) ,dtype=x.dtype)
        for i in range(x.shape[0]):
            max_x[i,0] = np.max(x[i,:])
        e_x = np.exp(x - max_x)
        return e_x / e_x.sum(axis=1).reshape((-1, 1))
    
    def _softmax_grad(self) -> np.ndarray:
        '''
        Softmax gradient - **y = fx * (1 - fx)**, where **fx** is the output of the softmax function.
        **Returns:**
            `y`: output data.
        '''
        identity = np.eye(self.fx.shape[-1])
        return np.einsum('ij,jk->ijk', self.fx, identity) - np.einsum('ij,ik->ijk', self.fx, self.fx)

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
        self.dfx = upstream_grad * self._softmax_grad()
        return self.dfx

class ReLU(object):
    '''ReLU Layer.'''
    def __init__(self, name:np.str="ReLU"):
        '''
        Initialize the ReLU Layer.
        **Parameters:**
            `name`: name of the layer.
        '''
        self.name = name

        # Forward prop
        self.fx = None

        # Backward prop
        self.dfx = None

    
    def __repr__(self):
        return f"ReLU Layer: {self.name}"
    
    def __str__(self):
        return f"ReLU Layer: {self.name}"
    
    def _relu(self, x:np.ndarray) -> np.ndarray:
        '''
        ReLU activation function - **y = max(0, x)**.
        **Parameters:**
            `x`: input data.
        **Returns:**
            `y`: output data.
        '''
        return np.maximum(0, x)
    
    def _relu_grad(self) -> np.ndarray:
        '''
        ReLU gradient - **y = 1 if x > 0 else 0**.
        **Returns:**
            `y`: output data.
        '''
        return (self.fx > 0).astype(self.fx.dtype)

    def __call__(self, x:np.ndarray) -> np.ndarray:
        '''
        Compute forward transformation - **y = max(0, x)**.
        **Parameters:**
            `x`: input data.
        **Returns:**
            `y`: output data.
        '''
        self.fx = self._relu(x)
        return self.fx

    def backward(self, upstream_grad:np.ndarray) -> np.ndarray:
        '''
        Compute downstream gradient.
        **Parameters:**
            `upstream_grad`: upstream gradient.
        **Returns:**
            `downstream_grad`: downstream gradient.
        '''
        self.dfx = upstream_grad * self._relu_grad()
        return self.dfx

class LeakyReLU(object):
    '''Leaky ReLU Layer.'''
    def __init__(self, alpha:float=0.01, name:np.str="LeakyReLU"):
        '''
        Initialize the Leaky ReLU Layer.
        **Parameters:**
            `alpha`: slope of the negative part.
            `name`: name of the layer.
        '''
        self.name = name
        self.alpha = alpha

        # Forward prop
        self.fx = None

        # Backward prop
        self.dfx = None

    
    def __repr__(self):
        return f"LeakyReLU Layer: {self.name}"
    
    def __str__(self):
        return f"LeakyReLU Layer: {self.name}"
    
    def _leaky_relu(self, x:np.ndarray) -> np.ndarray:
        '''
        Leaky ReLU activation function - **y = max(alpha * x, x)**.
        **Parameters:**
            `x`: input data.
        **Returns:**
            `y`: output data.
        '''
        return np.maximum(self.alpha * x, x)
    
    def _leaky_relu_grad(self) -> np.ndarray:
        '''
        Leaky ReLU gradient - **y = alpha if x < 0 else 1**.
        **Returns:**
            `y`: output data.
        '''
        return (self.fx < 0).astype(self.fx.dtype) * self.alpha + (self.fx >= 0).astype(self.fx.dtype)

    def __call__(self, x:np.ndarray) -> np.ndarray:
        '''
        Compute forward transformation - **y = max(alpha * x, x)**.
        **Parameters:**
            `x`: input data.
        **Returns:**
            `y`: output data.
        '''
        self.fx = self._leaky_relu(x)
        return self.fx

    def backward(self, upstream_grad:np.ndarray) -> np.ndarray:
        '''
        Compute downstream gradient.
        **Parameters:**
            `upstream_grad`: upstream gradient.
        **Returns:**
            `downstream_grad`: downstream gradient.
        '''
        self.dfx = upstream_grad * self._leaky_relu_grad()
        return self.dfx

class Tanh(object):
    '''Tanh Layer.'''
    def __init__(self, name:np.str="Tanh"):
        '''
        Initialize the Tanh Layer.
        **Parameters:**
            `name`: name of the layer.
        '''
        self.name = name

        # Forward prop
        self.fx = None

        # Backward prop
        self.dfx = None

    
    def __repr__(self):
        return f"Tanh Layer: {self.name}"
    
    def __str__(self):
        return f"Tanh Layer: {self.name}"
    
    def _tanh(self, x:np.ndarray) -> np.ndarray:
        '''
        Tanh activation function - **y = (e^x - e^-x) / (e^x + e^-x)**.
        **Parameters:**
            `x`: input data.
        **Returns:**
            `y`: output data.
        '''
        return np.tanh(x)
    
    def _tanh_grad(self) -> np.ndarray:
        '''
        Tanh gradient - **y = 1 - fx^2**.
        **Returns:**
            `y`: output data.
        '''
        return 1 - self.fx**2

    def __call__(self, x:np.ndarray) -> np.ndarray:
        '''
        Compute forward transformation - **y = (e^x - e^-x) / (e^x + e^-x)**.
        **Parameters:**
            `x`: input data.
        **Returns:**
            `y`: output data.
        '''
        self.fx = self._tanh(x)
        return self.fx

    def backward(self, upstream_grad:np.ndarray) -> np.ndarray:
        '''
        Compute downstream gradient.
        **Parameters:**
            `upstream_grad`: upstream gradient.
        **Returns:**
            `downstream_grad`: downstream gradient.
        '''
        self.dfx = upstream_grad * self._tanh_grad()
        return self.dfx

