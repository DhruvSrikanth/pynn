import numpy as np
from collections import OrderedDict

class NeuralNetwork(object):
    def __init__(self) -> None:
        '''
        Initialize neural network.
        '''
        self.blocks = OrderedDict(
            {
                'input': np.array([]),
                'hidden': np.array([]), 
                'output': np.array([]),
            }
        )

        print(f"Blocks to be initialized:{','.join(self.blocks.keys())}")
 
    def add(self, block_name: np.str, layer) -> None:
        '''
        Add a block to the neural network.
        Params:
            block_name: name of the block to add
        '''
        if block_name not in self.blocks:
            raise ValueError(f"Block name {block_name} not found. Valid block names are {', '.join(self.blocks.keys())}.")
        self.blocks[block_name].append(layer)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        '''
        Forward pass through the model.
        Params:
            x: input data of shape (n_features, 1)
        Returns:
            out: output data of shape (n_features, 1)
        '''
        # Input layer pass
        for layer in self.blocks['input']:
            x = layer.forward(x)
        
        # Hidden layer pass
        for layer in self.blocks['hidden']:
            x = layer.forward(x)
        
        # Output layer pass
        for layer in self.blocks['output']:
            x = layer.forward(x)
        
        return x
    
    def backward(self, dl: np.float64) -> None:
        '''
        Backward pass through the model.
        Params:
            dl - gradient of the loss with respect to the output of the model
        '''
        # Output layer backward pass
        for layer in reversed(self.blocks['output']):
            dl = layer.backward(dl)
        
        # Hidden layer backward pass
        for layer in reversed(self.blocks['hidden']):
            dl = layer.backward(dl)
        
        # Input layer backward pass
        for layer in reversed(self.blocks['input']):
            dl = layer.backward(dl)
        
    def update_weights(self, learning_rate: np.float64) -> None:
        '''
        Update weights of the model.
        Params:
            learning_rate: learning rate of the model
        '''
        for layer in self.blocks['input']:
            layer.update_weights(learning_rate)
        
        for layer in self.blocks['hidden']:
            layer.update_weights(learning_rate)
        
        for layer in self.blocks['output']:
            layer.update_weights(learning_rate)