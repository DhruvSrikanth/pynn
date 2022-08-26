import numpy as np

class Pool1D(object):
    '''Pooling layer.'''
    def __init__(self, kernel_size:np.int64, stride:np.int64, type:np.str='max', name:np.str='Pooling 1D'):
        '''
        Initialize the ReLU Layer.
        **Parameters:**
            `kernel_size`: k sized kernel to be moved over input.
            `stride`: step size of kernel movement over input.
            `type`: type of pooling layer. Valid types include - `max`, `min`, `average`.
            `name`: name of the layer.
        '''
        valid_types = {
            'max' : np.max, 
            'min' : np.min, 
            'average' : np.mean
        }
        if type not in valid_types:
            raise ValueError(f"Invalid pooliing type - {type}. Valid types include {', '.join(valid_types.keys())}")
        self.type = type.lower()
        self.name = f'{self.type.capitalize()} {name}'
        self.pooling_op = valid_types[self.type]

        self.kernel_size = kernel_size
        self.stride = stride

        self.x = None
        self.fx = None
        self.dfx = None
    
    def __repr__(self):
        return f"Pooling 1D Layer: {self.name}"
    
    def __str__(self):
        return f"Pooling 1D Layer: {self.name}"
    
    def _pool1D(self):
        '''
        1D Pooling function - **y = Pool(x)**.
        **Returns:**
            `y`: output data.
        '''
        y = []
        for start in range(self.kernel_size, len(self.x), self.stride + 1):
            end = start + self.kernel_size
            y.append(self.pooling_op(self.x[start:end]))
        return y
    
    def _pool1D_grad(self) -> np.ndarray:
        '''
        1D Pooling gradient - **y = [1...] where size = size(Pool(x))**.
        **Returns:**
            `y`: output data.
        '''
        return np.ones(self.fx.shape)
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        '''
        Compute forward transformation - **y = Pool(x)**.
        **Parameters:**
            `x`: input data.
        **Returns:**
            `y`: output data.
        '''
        self.x = np.copy(x)
        self.fx = self._pool1D()
        return self.fx
    
    def backward(self, upstream_grad:np.ndarray) -> np.ndarray:
        '''
        Compute downstream gradient.
        **Parameters:**
            `upstream_grad`: upstream gradient.
        **Returns:**
            `dfx`: downstream gradient.
        '''
        self.dfx = upstream_grad * self._pool1D_grad()
        return self.dfx