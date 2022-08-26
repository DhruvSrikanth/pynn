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
    
    def train(self, model:NeuralNetwork, train_set, val_set, L, lr:np.float64, n_classes:np.int64, epochs:np.int64):
        '''
        Train a model.
        **Parameters:**
            `model`: model to train.
            `train_set`: training set.
            `val_set`: validation set.
            `L`: objective function.
            `lr`: learning rate.
            `n_classes`: number of classes in classification problem.
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
                        minibatch_x = np.asarray(list(map(lambda x: np.asarray(x).flatten(), minibatch_x)))
                        minibatch_y = np.asarray(list(map(lambda y: np.array([1 if i == y else 0 for i in range(n_classes)]), minibatch_y)))

                        # Forward pass
                        minibatch_yhat = model(minibatch_x)
                        minibatch_y_hat = np.expand_dims(np.argmax(minibatch_yhat, axis=1), axis=1)

                        # Compute loss
                        minibatch_loss = np.mean(L(minibatch_y_hat, minibatch_y))
                        if stage == 'train':
                            model.backward(L.backward())
                            model.update_weights(lr)
                        
                        minibatch_accuracy = np.mean(minibatch_y_hat == minibatch_y)
                        
                        # Update the progress bar
                        pbar.set_postfix(Metrics=f"Loss: {minibatch_loss:.4f} - Accuracy: {minibatch_accuracy * 100:.4f}%")
                        pbar.update()
            print(f"Epoch: {epoch + 1} - Loss: {minibatch_loss:.6f} - Accuracy: {minibatch_accuracy * 100:.4f} - Time Taken: {time() - start_time:.2f}s.")
            print('-' * 50 + '\n')
    
    def test(self, model: NeuralNetwork, test_set, n_classes:np.int64):
        '''
        Test the model.
        **Parameters:**
            `model`: model to test.
            `test_set`: test set.
            `n_classes`: number of classes for classification problem.
        '''
        print('-' * 50)
        print(f'Testing {model.name}:')
        start_time = time()
        # No need to compute gradients
        with tqdm(test_set, desc=f"Testing : {model.name}") as pbar:
            for minibatch_x, minibatch_y in pbar:
                minibatch_x = np.asarray(list(map(lambda x: np.asarray(x).flatten(), minibatch_x)))
                minibatch_y = np.asarray(list(map(lambda y: np.array([1 if i == y else 0 for i in range(n_classes)]), minibatch_y)))
                
                # Forward pass
                minibatch_y_hat = model(minibatch_x) 
                    
                # Compute accuracy
                minibatch_accuracy = np.mean(minibatch_y_hat == minibatch_y)

                # Update the progress bar
                pbar.set_postfix(Metrics=f"Accuracy: {minibatch_accuracy * 100:.4f}%")
                pbar.update()
        print(f"Test Accuracy: {minibatch_accuracy * 100:.4f} - Time Taken: {time() - start_time:.2f}s.")
        print('-' * 50 + '\n')