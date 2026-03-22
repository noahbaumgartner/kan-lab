import torch


class FeynmanI1211Dataset:
    input_dim = 5
    output_dim = 1

    def __init__(self, n_train=1000, n_test=100, **kwargs):
        self.n_train = n_train
        self.n_test = n_test

    def create(self, device="cpu"):
        train_input = torch.rand(self.n_train, 5, device=device) * 2 - 1
        test_input = torch.rand(self.n_test, 5, device=device) * 2 - 1
        train_label = self._compute(train_input)
        test_label = self._compute(test_input)
        return {
            "train_input": train_input,
            "train_label": train_label,
            "test_input": test_input,
            "test_label": test_label,
        }

    def _compute(self, x):
        # f = q(Ef + Bv·sinθ)
        q = x[:, [0]]
        Ef = x[:, [1]]
        B = x[:, [2]]
        v = x[:, [3]]
        theta = x[:, [4]]
        return q * (Ef + B * v * torch.sin(theta))
