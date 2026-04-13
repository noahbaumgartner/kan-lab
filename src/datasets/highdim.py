import torch

from kan.utils import create_dataset


class HighDimDataset:
    input_dim = 100
    output_dim = 1
    ranges = [-1, 1]

    def __init__(self, n_train=1000, n_test=100, **kwargs):
        self.n_train = n_train
        self.n_test = n_test

    def create(self, device="cpu"):
        f = lambda x: torch.exp(
            torch.sin(torch.pi * x / 2).pow(2).mean(dim=1, keepdim=True)
        )
        return create_dataset(
            f,
            n_var=self.input_dim,
            ranges=[-1, 1],
            train_num=self.n_train,
            test_num=self.n_test,
            device=device,
        )
