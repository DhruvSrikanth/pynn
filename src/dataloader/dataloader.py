from torch.utils.data import DataLoader
from torchvision import datasets

class MNISTDataLoader:
    def __init__(self, type: str, batch_size: int, num_workers: int=1, transform: object=None):
        """
        Initialize MNIST data loader.
        Params:
            batch_size : (type int) batch size of data loader.
            num_workers : (type int) number of workers to use for data loader.
            transform : (type object) transform to apply to the dataset.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        # Type check the type
        self.type = type.lower()

        # Check if the type is valid
        if self.type != 'train' and self.type != 'test':
            raise ValueError(f"Invalid type: {self.type}. Expected 'train' or 'test'.")
        
        # Get the dataset
        self.dataset = datasets.MNIST(f'./data/{self.type}', train = self.type == 'train', download=True, transform=self.transform)
        
        # Create the data loader
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def __len__(self):
        """
        Get the length of the data loader.
        Returns:
            len: (type int) length of the data loader.
        """
        return len(self.dataloader)
        
    def get_dataloader(self):
        """
        Get the data loader.
        Returns:
            dataloader: (type torch.utils.data.DataLoader) data loader.
        """
        return self.dataloader