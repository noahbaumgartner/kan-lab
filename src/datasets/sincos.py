import torch


class SinCosDataset:
    """Dataset for f(x1, x2) = sin(x1) + cos(x2)."""

    def __init__(self, n_train=1000, n_test=100):
        self.n_train = n_train
        self.n_test = n_test

    def create(self, device="cpu"):
        train_input = torch.rand(self.n_train, 2, device=device) * 2 * torch.pi
        test_input = torch.rand(self.n_test, 2, device=device) * 2 * torch.pi
        train_label = (
            torch.sin(train_input[:, [0]]) + torch.cos(train_input[:, [1]])
        )
        test_label = (
            torch.sin(test_input[:, [0]]) + torch.cos(test_input[:, [1]])
        )
        return {
            "train_input": train_input,
            "train_label": train_label,
            "test_input": test_input,
            "test_label": test_label,
        }
