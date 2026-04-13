from kan.feynman import get_feynman_dataset
from kan.utils import create_dataset


class FeynmanDataset:
    output_dim = 1

    def __init__(self, name, n_train=1000, n_test=100, **kwargs):
        self.name = name
        self.n_train = n_train
        self.n_test = n_test

        symbol, expr, self.f, self.ranges = get_feynman_dataset(name)
        if isinstance(symbol, (list, tuple)):
            self.input_dim = len(symbol)
        else:
            self.input_dim = 1

    def create(self, device="cpu"):
        return create_dataset(
            self.f,
            n_var=self.input_dim,
            ranges=self.ranges,
            train_num=self.n_train,
            test_num=self.n_test,
            normalize_input=True,
            normalize_label=True,
            device=device,
        )
