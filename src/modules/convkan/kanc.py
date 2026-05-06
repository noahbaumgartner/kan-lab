from torch import nn
import torch.nn.functional as F

from .kanconv import KAN_Convolutional_Layer


# KANC_MLP from the paper https://arxiv.org/pdf/2406.13155
class KANC_MLP(nn.Module):
    def __init__(
        self,
        in_chans: int = 1,
        img_size: int = 28,
        num_classes: int = 10,
        conv1_out: int = 5,
        conv2_out: int = 10,
        kernel_size: int = 3,
        hidden_dim: int = 100,
        grid_size: int = 5,
    ):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(
            in_channels=in_chans,
            out_channels=conv1_out,
            kernel_size=(kernel_size, kernel_size),
            grid_size=grid_size,
        )
        self.conv2 = KAN_Convolutional_Layer(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=(kernel_size, kernel_size),
            grid_size=grid_size,
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flat = nn.Flatten()

        s = (img_size - (kernel_size - 1)) // 2
        s = (s - (kernel_size - 1)) // 2
        flat_dim = conv2_out * s * s

        self.linear1 = nn.Linear(flat_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.flat(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)
