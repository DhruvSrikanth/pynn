import numpy as np
from collections import OrderedDict
from .layers import Linear, Sigmoid, Relu
import typing

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

        print(f"Blocks to be initialized:{'\n'.join(self.blocks.keys())}")

    
    def add_layer(self, block_name: np.str, layer: typing.Union([Linear, Sigmoid, Relu])) -> None:
        '''
        Add a block to the neural network.
        Params:
            block_name: name of the block to add
        '''
        if block_name not in self.blocks:
            raise ValueError(f'Block name {block_name} not found')
        self.blocks[f'{block_name}'].append(layer)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
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
    
    def backward(self, dy: np.ndarray) -> None:
        '''
        Backward pass through the model.
        Params:
            dy - gradient of the loss with respect to the output of the model
        '''
        # Output layer backward pass
        for layer in reversed(self.blocks['output']):
            dy = layer.backward(dy)
        
        # Hidden layer backward pass
        for layer in reversed(self.blocks['hidden']):
            dy = layer.backward(dy)
        
        # Input layer backward pass
        for layer in reversed(self.blocks['input']):
            dy = layer.backward(dy)
        
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
        
    def train(self, x: np.ndarray, y: np.ndarray, learning_rate: np.float64, loss: Loss) -> None:
        '''
        Train the model.
        Params:
            x: input data of shape (n_features, 1)
            y: target data of shape (n_features, 1)
            learning_rate: learning rate of the model
            loss: loss function to use
        '''
        # Forward pass
        out = self.forward(x)
        # Compute loss
        loss_value = loss.forward(out, y)

        # Backward pass
        dy = loss.backward(out, y)

        # Update weights
        self.backward(dy)
        
        # Update weights
        self.update_weights(learning_rate)
        
        return loss_value

    