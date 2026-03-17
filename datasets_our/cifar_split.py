# Auto-generated file
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

class CIFAR10Split:
    def __init__(self):
        # Strict CIFAR-10 Normalization to preserve isotropic geometry
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        self.num_classes = 10

        self.dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )

    def get_task_loaders(self, batch_size):
        # CIFAR targets are lists, cast to tensor for lightning-fast splitting
        targets = torch.tensor(self.dataset.targets)
        
        # Task 1: Classes 0-4. Task 2: Classes 5-9.
        idx_task1 = torch.where(targets < 5)[0].tolist()
        idx_task2 = torch.where(targets >= 5)[0].tolist()

        task1 = Subset(self.dataset, idx_task1)
        task2 = Subset(self.dataset, idx_task2)

        # drop_last=True ensures HVP matrices don't jitter at epoch boundaries
        loader1 = DataLoader(task1, batch_size=batch_size, shuffle=True, drop_last=True)
        loader2 = DataLoader(task2, batch_size=batch_size, shuffle=True, drop_last=True)

        return loader1, loader2


class CIFAR100Split:
    def __init__(self):
        # Strict CIFAR-100 Normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])
        self.num_classes = 100
        self.dataset = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )

    def get_task_loaders(self, batch_size):
        targets = torch.tensor(self.dataset.targets)
        
        # Task 1: First 50 classes. Task 2: Last 50 classes.
        idx_task1 = torch.where(targets < 50)[0].tolist()
        idx_task2 = torch.where(targets >= 50)[0].tolist()

        task1 = Subset(self.dataset, idx_task1)
        task2 = Subset(self.dataset, idx_task2)

        loader1 = DataLoader(task1, batch_size=batch_size, shuffle=True, drop_last=True)
        loader2 = DataLoader(task2, batch_size=batch_size, shuffle=True, drop_last=True)

        return loader1, loader2