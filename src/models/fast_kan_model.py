import torch
import torch.nn as nn

from .base import BaseKANModel
from modules.fastkan.fastkan import FastKAN, FastKANLayer


class FastKANModel(BaseKANModel):
    def __init__(self, layers_hidden, grid_min=-2.0, grid_max=2.0, num_grids=8):
        self.layers_hidden = layers_hidden
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.model = None
        self.device = "cpu"

    def set_width(self, width):
        self.layers_hidden = width

    def build(self, device="cpu"):
        layers_hidden = list(self.layers_hidden)
        needs_custom = any(d == 1 for d in layers_hidden[:-1])
        if needs_custom:
            model = FastKAN(layers_hidden=[2, 1], num_grids=self.num_grids)
            model.layers = nn.ModuleList([
                FastKANLayer(
                    in_dim, out_dim,
                    grid_min=self.grid_min,
                    grid_max=self.grid_max,
                    num_grids=self.num_grids,
                    use_layernorm=in_dim > 1,
                )
                for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
            ])
            self.model = model.to(device)
        else:
            self.model = FastKAN(
                layers_hidden=layers_hidden,
                grid_min=self.grid_min,
                grid_max=self.grid_max,
                num_grids=self.num_grids,
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
