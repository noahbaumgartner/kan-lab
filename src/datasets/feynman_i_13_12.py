import torch


class FeynmanI1312Dataset:
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
        # f = Gm₁m₂(1/r₂ - 1/r₁)
        G = x[:, [0]]
        m1 = x[:, [1]]
        m2 = x[:, [2]]
        r1 = x[:, [3]]
        r2 = x[:, [4]]
        return G * m1 * m2 * (1 / r2 - 1 / r1)
