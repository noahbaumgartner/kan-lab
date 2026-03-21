import torch
import torch.nn as nn

from .base import BaseKANModel
from modules.chebykan.ChebyKANLayer import ChebyKANLayer


class ChebyKANNetwork(nn.Module):
    def __init__(self, layers_hidden, degree=4):
        super().__init__()
        layers = []
        for i in range(len(layers_hidden) - 1):
            layers.append(ChebyKANLayer(layers_hidden[i], layers_hidden[i + 1], degree))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChebyKANModel(BaseKANModel):
    def __init__(self, layers_hidden, degree=4, **kwargs):
        self.layers_hidden = layers_hidden
        self.degree = degree
        self.model = None
        self.device = "cpu"

    def build(self, device="cpu"):
        self.model = ChebyKANNetwork(
            layers_hidden=list(self.layers_hidden),
            degree=self.degree,
        ).to(device)
        self.device = device

    def fit(self, dataset, steps, lr, optimizer, loss_fn, batch_size, lamb, **kwargs):
        if loss_fn is None:
            loss_fn = lambda pred, target: torch.mean((pred - target) ** 2)

        if optimizer == "Adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == "LBFGS":
            opt = torch.optim.LBFGS(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        results = {"train_loss": [], "test_loss": [], "reg": []}

        for _ in range(steps):
            n_train = dataset["train_input"].shape[0]
            if batch_size == -1 or batch_size >= n_train:
                x = dataset["train_input"]
                y = dataset["train_label"]
            else:
                idx = torch.randperm(n_train, device=self.device)[:batch_size]
                x = dataset["train_input"][idx]
                y = dataset["train_label"][idx]

            if optimizer == "LBFGS":
                def closure():
                    opt.zero_grad()
                    pred = self.model(x)
                    loss = loss_fn(pred, y)
                    loss.backward()
                    return loss

                opt.step(closure)
            else:
                opt.zero_grad()
                pred = self.model(x)
                train_loss = loss_fn(pred, y)
                train_loss.backward()
                opt.step()

            with torch.no_grad():
                train_pred = self.model(dataset["train_input"])
                train_loss_val = loss_fn(train_pred, dataset["train_label"])
                test_pred = self.model(dataset["test_input"])
                test_loss_val = loss_fn(test_pred, dataset["test_label"])

            results["train_loss"].append(train_loss_val.sqrt().item())
            results["test_loss"].append(test_loss_val.sqrt().item())
            results["reg"].append(0.0)

        return results

    def predict(self, x):
        return self.model(x)

    def parameter_count(self):
        return sum(p.numel() for p in self.model.parameters())

    def get_model(self):
        return self.model
