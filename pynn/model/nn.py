import numpy as np

class NeuralNetwork(object):
    '''Neural Network.'''
    def __init__(self, name:np.str="Neural Network"):
        '''
        Initialize a neural network.
        '''
        self.blocks = {
            'input': np.array([]),
            'hidden': np.array([]), 
            'output': np.array([]),
        }

        self.name = name
 
    def add(self, block_name:np.str, layer):
        '''
        Add a layer to the model.
        **Parameters:**
            `block_name`: name of the block to add the layer to.
            `layer`: layer to add to the model.
        '''
        if block_name not in self.blocks:
            raise ValueError(f"Block name {block_name} not found. Valid block names are {', '.join(self.blocks.keys())}.")
        self.blocks[block_name].append(layer)
    
    def __call__(self, x:np.ndarray) -> np.ndarray:
        '''
        Forward pass through the model.
        **Parameters:**
            `x`: input data.
        **Returns:**
            `y_hat`: output of the model.
        '''
        for block in ['input', 'hidden', 'output']:
            for layer in self.blocks[block]:
                x = layer(x)
        y_hat = x
        return x
    
    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        '''
        Backward pass through the model.
        **Parameters:**
            `upstream_grad`: upstream gradient.
        **Returns:**
            `downstream_grad`: downstream gradient.
        '''
        for block in ['output', 'hidden', 'input']:
            for layer in self.blocks[block]:
                upstream_grad = layer.backward(upstream_grad)
        downstream_grad = upstream_grad
        return downstream_grad
        
    def update_weights(self, learning_rate: np.float64):
        '''
        Update weights of the model.
        **Parameters:**
            `learning_rate`: learning rate.
        '''
        for block in ['input', 'hidden', 'output']:
            for layer in self.blocks[block]:
                if hasattr(layer, 'update_weights'):
                    layer.update_weights(learning_rate)