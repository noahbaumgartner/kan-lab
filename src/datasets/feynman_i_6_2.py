import torch


class FeynmanI62Dataset:
    input_dim = 2
    output_dim = 1

    def __init__(self, n_train=1000, n_test=100, **kwargs):
        self.n_train = n_train
        self.n_test = n_test

    def create(self, device="cpu"):
        train_input = torch.rand(self.n_train, 2, device=device) * 2 - 1
        test_input = torch.rand(self.n_test, 2, device=device) * 2 - 1
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
        sigma = x[:, [1]]
        sigma_sq = sigma ** 2
        return torch.exp(-theta ** 2 / (2 * sigma_sq)) / torch.sqrt(2 * torch.pi * sigma_sq)
