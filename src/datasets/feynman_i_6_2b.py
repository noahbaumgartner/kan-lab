import torch


class FeynmanI62bDataset:
    input_dim = 3
    output_dim = 1

    def __init__(self, n_train=1000, n_test=100, **kwargs):
        self.n_train = n_train
        self.n_test = n_test

    def create(self, device="cpu"):
        train_input = torch.rand(self.n_train, 3, device=device) * 2 - 1
        test_input = torch.rand(self.n_test, 3, device=device) * 2 - 1
        train_label = self._compute(train_input)
        test_label = self._compute(test_input)
        return {
            "train_input": train_input,
            "train_label": train_label,
            "test_input": test_input,
            "test_label": test_label,
        }

    def _compute(self, x):
        theta = x[:, [0]]
        theta1 = x[:, [1]]
        sigma = x[:, [2]]
        sigma_sq = sigma ** 2
        return torch.exp(-(theta - theta1) ** 2 / (2 * sigma_sq)) / torch.sqrt(2 * torch.pi * sigma_sq)
