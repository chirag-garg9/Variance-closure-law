import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from datasets_our.base_dataset import BaseTaskDataset

class MNISTSplit(BaseTaskDataset):
    def __init__(self):
        # CRITICAL FIX 1: Normalize inputs to prevent artificially stretched 
        # Hessian eigenvalues (pathological geometry).
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )

    def get_task_loaders(self, batch_size):
        # CRITICAL FIX 2: Vectorized indexing.
        # Do NOT iterate over the dataset tuples. Query the raw targets tensor.
        # This reduces split time from minutes to 1 millisecond.
        targets = self.dataset.targets
        
        idx_task1 = torch.where(targets < 5)[0].tolist()
        idx_task2 = torch.where(targets >= 5)[0].tolist()

        task1 = Subset(self.dataset, idx_task1)
        task2 = Subset(self.dataset, idx_task2)

        # Using drop_last=True ensures consistent batch sizes for your gradient buffer
        loader1 = DataLoader(task1, batch_size=batch_size, shuffle=True, drop_last=True)
        loader2 = DataLoader(task2, batch_size=batch_size, shuffle=True, drop_last=True)

        return loader1, loader2