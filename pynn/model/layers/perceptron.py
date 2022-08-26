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
                **Valid values:** `"random"`, `"uniform"`.
            `name`: name of the layer.
        '''
        self.in_features = in_features
        self.out_features = out_features
        
        valid_initialization = ['random', 'uniform']
        if initialization not in valid_initialization:
            raise ValueError(f"Invalid weight initialization - {initialization}. Valid types include {', '.join(valid_initialization)}.")

        self.initialization = initialization
        self.bias_flag = bias
        self.name = name

        # initialize weights and bias
        initializer = {
            'random' : np.random.randn, 
            'uniform' : np.random.uniform
        }

        self.weight = initializer[self.initialization](self.in_features, self.out_features) * np.sqrt(2 / self.in_features)
        self.bias = np.zeros(self.out_features)

        self.x = None
        self.fx = None
        self.dfx = None
        self.dW = None
        self.db = None
    
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
        return np.dot(self.x, self.weight) + self.bias
    
    def _linear_transformation_grad(self, upstream_grad):
        '''
        Linear transformation gradient.
        **Returns:**
            `dfx`: gradient of linear transformation. 
        '''
        self.dW = (np.matmul(self.x[:, :, None], upstream_grad[:, None, :])).mean(axis=0)
        if self.bias_flag:
            self.db = upstream_grad.mean(axis=0)
        return np.dot(upstream_grad, self.weight.transpose())

    def __call__(self, x:np.ndarray):
        '''
        Compute forward transformation - **y = Wx + b**.
        **Parameters:**
            `x`: input data.
        **Returns:**
            `fx`: transformed data.
        '''
        self.x = np.copy(x)
        self.fx =  self._linear_transformation()
        return self.fx

    def backward(self, upstream_grad:np.ndarray) -> np.ndarray:
        '''
        Compute backward transformation - **dZ = dWx + db**.
        **Parameters:**
            `upstream_grad`: upstream gradient.
        **Returns:**
            `dfx`: downstream gradient.
        '''
        self.dfx = self._linear_transformation_grad(upstream_grad)
        return self.dfx

    def update_params(self, lr:np.float64):
        '''
        Update parameters based on gradient descent.
        **Parameters:**
            `lr`: learning rate.
        '''
        self.weight -= lr * self.dW
        if self.bias_flag:
            self.bias -= lr * self.db