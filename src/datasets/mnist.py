import torch
from torchvision import datasets, transforms


class MNISTDataset:
    input_dim = 784
    output_dim = 10
    def __init__(self, n_train=60000, n_test=10000, **kwargs):
        self.n_train = n_train
        self.n_test = n_test

    def create(self):
        train_ds = datasets.MNIST(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )
        test_ds = datasets.MNIST(
            root="./data", train=False, download=True, transform=transforms.ToTensor()
        )

        return {
            "train_input": train_ds.data[: self.n_train].float().view(-1, 784) / 255.0,
            "train_label": train_ds.targets[: self.n_train],
            "test_input": test_ds.data[: self.n_test].float().view(-1, 784) / 255.0,
            "test_label": test_ds.targets[: self.n_test],
        }
