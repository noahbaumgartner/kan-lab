import torch


class HighDimDataset:
    input_dim = 100
    output_dim = 1

    def __init__(self, n_train=1000, n_test=100, **kwargs):
        self.n_train = n_train
        self.n_test = n_test

    def create(self, device="cpu"):
        train_input = torch.rand(self.n_train, 100, device=device) * 2 - 1
        test_input = torch.rand(self.n_test, 100, device=device) * 2 - 1
        train_label = torch.exp(
            torch.sin(torch.pi * train_input / 2).pow(2).mean(dim=1, keepdim=True)
        )
        test_label = torch.exp(
            torch.sin(torch.pi * test_input / 2).pow(2).mean(dim=1, keepdim=True)
        )
        return {
            "train_input": train_input,
            "train_label": train_label,
            "test_input": test_input,
            "test_label": test_label,
        }
