import torch


class ExpSinDataset:
    kan_width = [2, 1, 1]

    def __init__(self, n_train=1000, n_test=100):
        self.n_train = n_train
        self.n_test = n_test

    def create(self, device="cpu"):
        train_input = torch.rand(self.n_train, 2, device=device) * 2 - 1
        test_input = torch.rand(self.n_test, 2, device=device) * 2 - 1
        train_label = torch.exp(
            torch.sin(torch.pi * train_input[:, [0]]) + train_input[:, [1]] ** 2
        )
        test_label = torch.exp(
            torch.sin(torch.pi * test_input[:, [0]]) + test_input[:, [1]] ** 2
        )
        return {
            "train_input": train_input,
            "train_label": train_label,
            "test_input": test_input,
            "test_label": test_label,
        }
