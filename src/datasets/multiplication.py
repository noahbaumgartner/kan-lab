import torch


class MultiplicationDataset:
    input_dim = 2
    output_dim = 1

    def __init__(self, n_train=1000, n_test=100, **kwargs):
        self.n_train = n_train
        self.n_test = n_test

    def create(self, device="cpu"):
        train_input = torch.rand(self.n_train, 2, device=device) * 2 - 1
        test_input = torch.rand(self.n_test, 2, device=device) * 2 - 1
        train_label = (train_input[:, [0]] * train_input[:, [1]])
        test_label = (test_input[:, [0]] * test_input[:, [1]])
        return {
            "train_input": train_input,
            "train_label": train_label,
            "test_input": test_input,
            "test_label": test_label,
        }
