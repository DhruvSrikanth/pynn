import numpy as np

class MSE(object):
    '''Objective function for - **Mean Squared Error.**'''
    def __init__(self, name:np.str = 'Mean Squared Error'):
        '''
        Initialize a Mean Squared Error objective function.
        **Parameters:**
            `name`: name of the objective function.
        '''
        self.name = name
        self.fx = None
        self.local_grad = None
    
    def _mse(self, y, y_hat):
        '''
        Compute the Mean Squared Error.
        **Parameters:**
            `y`: true labels.
            `y_hat`: predicted labels.
        **Returns:**
            `mse`: Mean Squared Error.
        '''
        return 0.5*np.sum((y - y_hat)**2)/y_hat.shape[0]
    
    def _mse_grad(self, y, y_hat):
        '''
        Compute the gradient of the Mean Squared Error.
        **Parameters:**
            `y`: true labels.
            `y_hat`: predicted labels.
        **Returns:**
            `mse_grad`: gradient of the Mean Squared Error.
        '''
        return (y_hat - y) / y_hat.shape[0]
    
    def __call__(self, y, y_hat):
        '''
        Forward pass of the Mean Squared Error.
        **Parameters:**
            `y`: true labels.
            `y_hat`: predicted labels.
        **Returns:**
            `mse`: Mean Squared Error.
        '''
        self.fx = self._mse(y, y_hat)
        self.local_grad = self._mse_grad(y, y_hat)
        return self.fx
    
    def backward(self):
        '''
        Backward pass of the Mean Squared Error.
        **Returns:**
            `downstream_derivative`: gradient of the Mean Squared Error.
        '''
        upstream_derivative = 1
        downstream_derivative = upstream_derivative * self.local_grad
        return downstream_derivative
    
    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.name