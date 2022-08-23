import numpy as np  # import numpy library

class Linear(object):
    '''Perceptron Layer.'''
    def __init__(self, in_features:np.ndarray, out_features:np.ndarray, bias:np.bool=True, initialization:np.str="random", name:np.str="Linear"):
        '''
        Intialize the Perceptron Layer.
        **Parameters:**
            `in_features`: number of input features.
            `out_features`: number of output features.
            `bias`: whether to use bias or not.
            `initialization`: method of initialization. 
                **Default:** `"random"`. 
                **Valid values:** `"random"`, `"uniform"`, `"zeros"`, `"ones"`.
            `name`: name of the layer.
        '''
        self.in_features = in_features
        self.out_features = out_features
        self.initialization = initialization
        self.bias = bias
        self.name = name

        # initialize weights and bias
        initializer = {
            'random' : np.random.randn, 
            'uniform' : np.random.uniform, 
            'zeros' : np.zeros,
            'ones' : np.ones
        }

        self.params = {
            'W': initializer[self.initialization](out_features, in_features), 
            'b': np.zeros((out_features, 1))
        }

        # Forward prop
        self.downstream_activation = None
        self.fx = None

        # Backward prop
        self.dW = None
        self.db = None
        self.downstream_grad = None
    
    def __repr__(self):
        return f"Linear Layer: {self.in_features} -> {self.out_features}"
    
    def __str__(self):
        return f"Linear Layer: {self.in_features} -> {self.out_features}"
    
    def _linear_transformation(self) -> np.ndarray:
        '''
        Linear transformation - **Z = Wx + b**.
        **Returns:**
            `Z`: transformed data.
        '''
        return (np.dot(self.params['W'], self.downstream_activation.T) + self.params['b']).T

    def __call__(self, x:np.ndarray):
        '''
        Compute forward transformation - **y = Wx + b**.
        **Parameters:**
            `x`: input data.
        **Returns:**
            `fx`: transformed data.
        '''
        self.downstream_activation = x
        self.fx = self._linear_transformation()
        return self.fx

    def backward(self, upstream_grad:np.ndarray) -> np.ndarray:
        '''
        Compute backward transformation - **dZ = dWx + db**.
        **Parameters:**
            `upstream_grad`: upstream gradient.
        **Returns:**
            `downstream_grad`: downstream gradient.
        '''
        self.dW = np.dot(upstream_grad.T, self.downstream_activation)
        if self.bias:
            self.db = np.sum(upstream_grad.T, axis=1, keepdims=True)
        self.downstream_grad = np.array([np.sum(upstream_grad_sample * self.params['W'].T) for upstream_grad_sample in upstream_grad]).reshape(-1, 1)
        return self.downstream_grad

    def update_params(self, lr:np.float64):
        '''
        Update parameters based on gradient descent.
        **Parameters:**
            `lr`: learning rate.
        '''
        self.params['W'] = self.params['W'] - lr * self.dW
        if self.bias:
            self.params['b'] = self.params['b'] - lr * self.db
    