import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Subset, Dataset

def load_data(batch_size, train_size=10000):

    # Load MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Print the size of the training and test datasets
    print(f"Training set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")

    train_set = select_random(train_set, subset_size=train_size)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader
    
# Select a random subset from the training set
def select_random(dataset, subset_size=10000):
    indices = np.random.permutation(len(dataset))
    subset_indices = indices[:subset_size]
    subset = Subset(dataset, subset_indices)
    # Print the size of the random subset
    print(f"Random subset size: {len(subset)}")
    return subset

# select digit
class MNISTSubset(Dataset):
    def __init__(self, mnist_dataset, min_label=0, max_label=4):
        self.mnist_dataset = mnist_dataset
        self.indices = [i for i, (_, label) in enumerate(mnist_dataset) if label <= max_label and label >= min_label]

    def __getitem__(self, index):
        return self.mnist_dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)