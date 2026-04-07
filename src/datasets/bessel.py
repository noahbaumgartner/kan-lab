import torch
from scipy.special import j0

from src.datasets.base import create_dataset


class BesselDataset:
    input_dim = 1
    output_dim = 1

    f = staticmethod(lambda x: j0(20 * x[:, 0]))

    def __init__(self, n_train=1000, n_test=100, **kwargs):
        self.n_train = n_train
        self.n_test = n_test

    def create(self, device="cpu"):
        dataset = create_dataset(
            self.f,
            n_var=self.input_dim,
            train_num=self.n_train,
            test_num=self.n_test,
            ranges=[-1, 1],
            device=device,
        )
        return dataset
