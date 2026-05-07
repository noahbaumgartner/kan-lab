from scipy.special import j0

from kan.utils import create_dataset


class BesselDataset:
    input_dim = 1
    output_dim = 1

    f = staticmethod(lambda x: j0(20 * x[:, 0]))

    def __init__(self, n_train=1000, n_test=100, **kwargs):
        self.n_train = n_train
        self.n_test = n_test

    def create(self):
        return create_dataset(
            self.f,
            n_var=self.input_dim,
            ranges=[-1, 1],
            train_num=self.n_train,
            test_num=self.n_test,
            normalize_input=True,
            normalize_label=True,
            device="cpu",
        )
