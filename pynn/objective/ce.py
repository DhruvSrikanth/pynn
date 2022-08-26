import numpy as np

class CrossEntropy(object):
    def __init__(self, name:np.str="Cross Entropy"):
        '''
        Initialize the Cross Entropy objective function.
        '''
        self.name = name
        self.y = None
        self.y_hat = None
        self.loss = None
        self.dloss = None
    
    def _cross_entropy(self):
        '''
        Compute the Cross Entropy.
        **Returns:**
            `cross_entropy`: Cross Entropy.
        '''
        return (np.where(self.y == 1, -np.log(self.y_hat), 0)).sum(axis=1) 
    
    def _cross_entropy_grad(self):
        '''
        Compute the gradient of the Cross Entropy.
        **Returns:**
            `cross_entropy_grad`: gradient of the Cross Entropy.
        '''
        return np.where(self.y == 1, -1 / self.y_hat, 0)
    
    def __call__(self, y_hat, y):
        '''
        Forward pass of the Cross Entropy objective function.
        **Parameters:**
            `y`: true labels.
            `y_hat`: predicted labels.
        **Returns:**
            `cross_entropy`: Cross Entropy.
        '''
        self.y = np.copy(y)
        self.y_hat = y_hat.clip(min=1e-8, max=None)
        self.loss = self._cross_entropy()
        return self.loss
    
    def backward(self):
        '''
        Backward pass of the Cross Entropy objective function.
        **Returns:**
            `dloss`: gradient of the Cross Entropy.
        '''
        upstream_grad = 1
        self.dloss = upstream_grad * self._cross_entropy_grad()
        return self.dloss
    
    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.name