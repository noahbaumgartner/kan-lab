import torch


class DeepFormulaDataset:
    input_dim = 4
    output_dim = 1

    def __init__(self, n_train=1000, n_test=100, **kwargs):
        self.n_train = n_train
        self.n_test = n_test

    def create(self, device="cpu"):
        train_input = torch.rand(self.n_train, 4, device=device) * 2 - 1
        test_input = torch.rand(self.n_test, 4, device=device) * 2 - 1
        train_label = self._formula(train_input)
        test_label = self._formula(test_input)
        return {
            "train_input": train_input,
            "train_label": train_label,
            "test_input": test_input,
            "test_label": test_label,
        }

    def _formula(self, x):
        a = torch.sin(torch.pi * (x[:, 0] ** 2 + x[:, 1] ** 2))
        b = torch.sin(torch.pi * (x[:, 2] ** 2 + x[:, 3] ** 2))
        return torch.exp(0.5 * (a + b)).unsqueeze(1)
