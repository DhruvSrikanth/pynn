import numpy as np
from tqdm import tqdm
from time import time
from .nn import NeuralNetwork
class Train(object):
    def __init__(self):
        '''Initialize a trainer.'''
    
    def __call__(self, model:NeuralNetwork, train_set, val_set, L, lr, epochs:np.int64):
        '''
        Train a model.
        **Parameters:**
            `model`: model to train.
            `train_set`: training set.
            `val_set`: validation set.
            `L`: objective function.
            `lr`: learning rate.
            `epochs`: number of epochs to train for.
        '''
        for epoch in range(epochs):
            print('-' * 50)
            print(f'Starting Epoch {epoch + 1}/{epochs}:')
            start_time = time()
            for stage in ['train', 'val']:
                dataset = train_set if stage == 'train' else val_set
                with tqdm(dataset, desc=f"{'Training' if stage == 'train' else 'Validation'} : {model.name}") as pbar:
                    for minibatch_x, minibatch_y in pbar:
                        # Forward pass
                        minibatch_inference = model(minibatch_x)
                        # Compute loss
                        minibatch_loss = L(minibatch_inference, minibatch_y)

                        if stage == 'train':
                            # Backward pass
                            model.backward(dl=minibatch_loss)
                            # Update weights
                            model.update(learning_rate=lr)
                        
                        # Compute accuracy
                        predictions = np.argmax(minibatch_inference, axis=0)
                        minibatch_accuracy = np.mean(predictions == minibatch_y)
                        # Update the progress bar
                        pbar.set_postfix(Metrics=f"Loss: {minibatch_loss:.4f} - Accuracy: {minibatch_accuracy * 100:.4f}%")
                        pbar.update()
            print(f"Epoch: {epoch + 1} - Loss: {minibatch_loss:.6f} - Accuracy: {minibatch_accuracy:.6f} - Time Taken: {time() - start_time:.2f}s.")
            print('-' * 50 + '\n')
            
            
                
                    

                        
        
