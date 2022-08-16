import numpy as np
from tqdm import tqdm
from time import time
from .nn import NeuralNetwork

class Learner(object):
    def __init__(self, name:np.str = 'Learner'):
        '''
        Initialize a trainer.
        **Parameters:**
            `name`: name of the trainer.
        '''
        self.name = name
    
    def train(self, model:NeuralNetwork, train_set, val_set, L, lr:np.float64, epochs:np.int64):
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
                        minibatch_loss = 0
                        minibatch_y = np.asarray(minibatch_y)
                        minibatch_inference = np.asarray([])
                        for x in minibatch_x:
                            x = np.asarray(x)
                            
                            # Forward pass
                            y_hat = model(x)
                            minibatch_inference = np.append(minibatch_inference, y_hat)
                            
                        # Compute loss
                        minibatch_loss = L(minibatch_inference, minibatch_y)

                        if stage == 'train':
                            # Backward pass
                            model.backward(L.backward())
                            # Update weights
                            model.update_weights(lr)
                        
                        # Compute accuracy
                        predictions = np.argmax(minibatch_inference, axis=0)
                        minibatch_accuracy = np.mean(predictions == minibatch_y)
                        # Update the progress bar
                        pbar.set_postfix(Metrics=f"Loss: {minibatch_loss:.4f} - Accuracy: {minibatch_accuracy * 100:.4f}%")
                        pbar.update()
            print(f"Epoch: {epoch + 1} - Loss: {minibatch_loss:.6f} - Accuracy: {minibatch_accuracy:.6f} - Time Taken: {time() - start_time:.2f}s.")
            print('-' * 50 + '\n')
    
    def test(self, model: NeuralNetwork, test_set):
        '''
        Test the model.
        **Parameters:**
            `model`: model to test.
            `test_set`: test set.
        '''
        print('-' * 50)
        print(f'Testing {model.name}:')
        start_time = time()
        with tqdm(test_set, desc=f"Testing : {model.name}") as pbar:
            for minibatch_x, minibatch_y in pbar:
                minibatch_inference = np.asarray([])
                for x in minibatch_x:
                    x = np.asarray(x)
                    # Forward pass
                    y_hat = model(x)
                    minibatch_inference = np.append(minibatch_inference, y_hat)
                    
                # Compute accuracy
                predictions = np.argmax(minibatch_inference, axis=0)
                minibatch_accuracy = np.mean(predictions == minibatch_y)
                # Update the progress bar
                pbar.set_postfix(Metrics=f"Accuracy: {minibatch_accuracy * 100:.4f}%")
                pbar.update()
        print(f"Test Accuracy: {minibatch_accuracy:.6f} - Time Taken: {time() - start_time:.2f}s.")
        print('-' * 50 + '\n')