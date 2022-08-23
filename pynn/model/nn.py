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
        self.blocks[block_name] = np.append(self.blocks[block_name], layer)
    
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
        return y_hat
        
    def no_grad(self):
        '''
        Disable gradient calculation.
        '''
        for block in ['input', 'hidden', 'output']:
            for layer in self.blocks[block]:
                if hasattr(layer, 'no_grad'):
                    layer.no_grad()
    
    def grad(self):
        '''
        Allow computation of the gradients of the model.
        '''
        for block in ['input', 'hidden', 'output']:
            for layer in self.blocks[block]:
                if hasattr(layer, 'grad'):
                    layer.grad()

    def zero_grad(self):
        '''
        Zero out the gradients of the model.
        '''
        for block in ['input', 'hidden', 'output']:
            for layer in self.blocks[block]:
                if hasattr(layer, 'zero_grad'):
                    layer.zero_grad()
    
    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        '''
        Backward pass through the model.
        **Parameters:**
            `upstream_grad`: upstream gradient.
        **Returns:**
            `downstream_grad`: downstream gradient.
        '''
        for block in ['output', 'hidden', 'input']:
            for layer in self.blocks[block][::-1]:
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
                if hasattr(layer, 'update_params'):
                    layer.update_params(learning_rate)
    
    def __repr__(self):
        '''
        Print the model summary.
        '''
        name = f"Model Summary - {self.name}:\n"
        summary = '-' * len(name) + '\n'
        for block in ['input', 'hidden', 'output']:
            summary += f"{block.capitalize()} - \n"
            for layer in self.blocks[block]:
                summary += f"{layer.__repr__()}\n"
        summary += '-' * len(name) + '\n'
        summary = name + summary
        return summary
    
    def __str__(self):
        '''
        Print the model summary.
        '''
        return self.__repr__()
    
    def summary(self):
        '''
        Print the model summary.
        '''
        return self.__repr__()