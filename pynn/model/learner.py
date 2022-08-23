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
                        # Flatten the minibatch
                        minibatch_x = list(map(lambda x: np.asarray(x).flatten(), minibatch_x))
                        minibatch_y = list(map(lambda y: np.expand_dims(np.asarray(y), axis=0), minibatch_y))
                        minibatch_x = np.asarray(minibatch_x)
                        minibatch_y = np.asarray(minibatch_y)

                        # Forward pass
                        minibatch_yhat = model(minibatch_x)
                        minibatch_inference = np.expand_dims(np.argmax(minibatch_yhat, axis=1), axis=1)

                        # Compute loss
                        minibatch_loss = L(minibatch_inference, minibatch_y)
                        if stage == 'train':
                            # Backward pass
                            model.backward(L.backward())
                            # Update 
                            model.update_weights(lr)
                        
                        minibatch_accuracy = np.mean(minibatch_inference == minibatch_y)
                        
                        # Update the progress bar
                        pbar.set_postfix(Metrics=f"Loss: {minibatch_loss:.4f} - Accuracy: {minibatch_accuracy * 100:.4f}%")
                        pbar.update()
            print(f"Epoch: {epoch + 1} - Loss: {minibatch_loss:.6f} - Accuracy: {minibatch_accuracy * 100:.4f} - Time Taken: {time() - start_time:.2f}s.")
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
                # Flatten the minibatch
                minibatch_x = list(map(lambda x: np.asarray(x).flatten(), minibatch_x))
                minibatch_y = list(map(lambda y: np.expand_dims(np.asarray(y), axis=0), minibatch_y))
                minibatch_x = np.asarray(minibatch_x)
                minibatch_y = np.asarray(minibatch_y)
                minibatch_inference = model(minibatch_x) 
                    
                # Compute accuracy
                predictions = np.expand_dims(np.argmax(minibatch_inference, axis=1), axis=1)
                minibatch_accuracy = np.mean(predictions == minibatch_y)

                # Update the progress bar
                pbar.set_postfix(Metrics=f"Accuracy: {minibatch_accuracy * 100:.4f}%")
                pbar.update()
        print(f"Test Accuracy: {minibatch_accuracy * 100:.4f} - Time Taken: {time() - start_time:.2f}s.")
        print('-' * 50 + '\n')