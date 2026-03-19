import torch
from scipy.special import j0


class BesselDataset:
    kan_width = [1, 1]

    def __init__(self, n_train=1000, n_test=100):
        self.n_train = n_train
        self.n_test = n_test

    def create(self, device="cpu"):
        train_input = torch.rand(self.n_train, 1, device=device) * 2 - 1
        test_input = torch.rand(self.n_test, 1, device=device) * 2 - 1
        train_label = torch.tensor(j0(20 * train_input.cpu().numpy()), dtype=torch.float32, device=device)
        test_label = torch.tensor(j0(20 * test_input.cpu().numpy()), dtype=torch.float32, device=device)
        return {
            "train_input": train_input,
            "train_label": train_label,
            "test_input": test_input,
            "test_label": test_label,
        }
