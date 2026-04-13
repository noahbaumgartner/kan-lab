import torch

from kan.utils import create_dataset


class ExpSinDataset:
    input_dim = 2
    output_dim = 1

    def __init__(self, n_train=1000, n_test=100, **kwargs):
        self.n_train = n_train
        self.n_test = n_test

    def create(self, device="cpu"):
        f = lambda x: torch.exp(
            torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2
        )
        return create_dataset(
            f,
            n_var=self.input_dim,
            ranges=[-1, 1],
            train_num=self.n_train,
            test_num=self.n_test,
            normalize_input=True,
            normalize_label=True,
            device=device,
        )
