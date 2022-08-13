import numpy as np

class Linear(object):
    def __init__(self, in_features: np.int64, out_features: np.int64, bias: np.bool=True, name: str = 'Linear') -> None:
        '''
        Initialize linear layer.
        Params:
            in_features: number of input features
            out_features: number of output features
            bias: boolean to determine whether to include bias term
        '''
        self.name = name
        self.in_features = in_features
        self.out_features = out_features

        # Weight matrix (in_features, out_features)
        self.weight = np.random.randn(in_features, out_features)

        # Bias vector (out_features)
        if bias:
            self.bias = np.zeros(out_features)
        else:
            self.bias = None
        
        # Gradient of loss w.r.t. weight matrix
        self.dweight = None

        # Gradient of loss w.r.t. bias vector
        self.dbias = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Forward pass of linear layer.
        Params:
            x: input data of shape (in_features, 1)
        Returns:
            out: output data of shape (out_features, 1)
        '''
        return np.dot(x, self.weight) + self.bias

    def backward(self, dout):
        '''
        Backward pass of linear layer.
        Params:
            dout: gradient of loss w.r.t. output data of shape (out_features, 1)
        Returns:
            dx: gradient of loss w.r.t. input data of shape (in_features, 1)
        '''
        self.dweight = np.dot(dout.T, self.x)
        self.dbias = np.sum(dout, axis=0)
        return np.dot(dout, self.weight.T)
    
    def update(self, lr: np.float64) -> None:
        '''
        Update weights and biases.
        Params:
            lr: learning rate
        '''
        self.weight -= lr * self.dweight
        self.bias -= lr * self.dbias
    
    def __repr__(self) -> str:
        return f'{self.name}({self.in_features}, {self.out_features})'
    def __str__(self) -> str:
        return f'{self.name}({self.in_features}, {self.out_features})'
