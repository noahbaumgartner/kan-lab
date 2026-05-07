import torch

from kan.utils import create_dataset


class DeepFormulaDataset:
    input_dim = 4
    output_dim = 1

    def __init__(self, n_train=1000, n_test=100, **kwargs):
        self.n_train = n_train
        self.n_test = n_test

    def create(self):
        f = lambda x: torch.exp(
            0.5
            * (
                torch.sin(torch.pi * (x[:, [0]] ** 2 + x[:, [1]] ** 2))
                + torch.sin(torch.pi * (x[:, [2]] ** 2 + x[:, [3]] ** 2))
            )
        )
        return create_dataset(
            f,
            n_var=self.input_dim,
            ranges=[-1, 1],
            train_num=self.n_train,
            test_num=self.n_test,
            normalize_input=True,
            normalize_label=True,
            device="cpu",
        )
