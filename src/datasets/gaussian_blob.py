import torch


class GaussianBlobDataset:
    input_dim = 100  # 10x10 flattened
    output_dim = 4   # x, y, width, amplitude

    def __init__(self, n_train=1000, n_test=100, image_size=10, **kwargs):
        self.n_train = n_train
        self.n_test = n_test
        self.image_size = image_size

    def _generate(self, n, device="cpu"):
        size = self.image_size

        # random blob parameters
        x_center = torch.rand(n, device=device) * (size - 1)
        y_center = torch.rand(n, device=device) * (size - 1)
        width = torch.rand(n, device=device) * 2.0 + 0.5      # [0.5, 2.5]
        amplitude = torch.rand(n, device=device) * 0.8 + 0.2   # [0.2, 1.0]

        # pixel grid
        yy, xx = torch.meshgrid(
            torch.arange(size, dtype=torch.float32, device=device),
            torch.arange(size, dtype=torch.float32, device=device),
            indexing="ij",
        )
        xx = xx.flatten()  # (100,)
        yy = yy.flatten()  # (100,)

        # gaussian blob images: (n, 100)
        dx = xx[None, :] - x_center[:, None]
        dy = yy[None, :] - y_center[:, None]
        images = amplitude[:, None] * torch.exp(
            -(dx ** 2 + dy ** 2) / (2 * width[:, None] ** 2)
        )

        # normalize parameters to [0, 1] for regression targets
        labels = torch.stack([
            x_center / (size - 1),
            y_center / (size - 1),
            (width - 0.5) / 2.0,
            (amplitude - 0.2) / 0.8,
        ], dim=1)

        return images, labels

    def create(self, device="cpu"):
        train_input, train_label = self._generate(self.n_train, device)
        test_input, test_label = self._generate(self.n_test, device)
        return {
            "train_input": train_input,
            "train_label": train_label,
            "test_input": test_input,
            "test_label": test_label,
        }
