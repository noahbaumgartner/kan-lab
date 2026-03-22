import torch


class FeynmanI918Dataset:
    input_dim = 9
    output_dim = 1

    def __init__(self, n_train=1000, n_test=100, **kwargs):
        self.n_train = n_train
        self.n_test = n_test

    def create(self, device="cpu"):
        train_input = torch.rand(self.n_train, 9, device=device) * 2 - 1
        test_input = torch.rand(self.n_test, 9, device=device) * 2 - 1
        train_label = self._compute(train_input)
        test_label = self._compute(test_input)
        return {
            "train_input": train_input,
            "train_label": train_label,
            "test_input": test_input,
            "test_label": test_label,
        }

    def _compute(self, x):
        # Variables: G, m1, m2, x1, x2, y1, y2, z1, z2
        G = x[:, [0]]
        m1 = x[:, [1]]
        m2 = x[:, [2]]
        x1 = x[:, [3]]
        x2 = x[:, [4]]
        y1 = x[:, [5]]
        y2 = x[:, [6]]
        z1 = x[:, [7]]
        z2 = x[:, [8]]
        r_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
        return G * m1 * m2 / r_sq
