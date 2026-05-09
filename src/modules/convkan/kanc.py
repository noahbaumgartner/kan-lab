from torch import nn

from .kanconv import KAN_Convolutional_Layer


class KANC_MLP(nn.Module):
    def __init__(
        self,
        grid_size: int = 5,
        img_size: int = 28,
        in_chans: int = 1,
        num_classes: int = 10,
    ):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(
            in_channels=in_chans, out_channels=5, kernel_size=(3, 3), grid_size=grid_size
        )
        self.conv2 = KAN_Convolutional_Layer(
            in_channels=5, out_channels=5, kernel_size=(3, 3), grid_size=grid_size
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flat = nn.Flatten()

        # Two (3x3 conv → 2x2 pool) stages applied to img_size.
        s = (img_size - 2) // 2
        s = (s - 2) // 2
        flat_dim = 5 * s * s

        self.linear1 = nn.Linear(flat_dim, num_classes)
        self.name = f"KANC MLP (Small) (gs = {grid_size})"

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.flat(x)
        return self.linear1(x)
