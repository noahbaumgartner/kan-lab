import torch

from .base import BaseKANModel
from modules.efficientkan.src.efficient_kan import KAN


class EfficientKANModel(BaseKANModel):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3):
        self.layers_hidden = layers_hidden
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.model = None
        self.device = "cpu"

    def build(self, device="cpu"):
        self.model = KAN(
            layers_hidden=list(self.layers_hidden),
            grid_size=self.grid_size,
            spline_order=self.spline_order,
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
                    if lamb > 0:
                        loss = loss + lamb * self.model.regularization_loss()
                    loss.backward()
                    return loss

                opt.step(closure)
            else:
                opt.zero_grad()
                pred = self.model(x)
                train_loss = loss_fn(pred, y)
                total = train_loss
                if lamb > 0:
                    total = total + lamb * self.model.regularization_loss()
                total.backward()
                opt.step()

            with torch.no_grad():
                train_pred = self.model(dataset["train_input"])
                train_loss_val = loss_fn(train_pred, dataset["train_label"])
                test_pred = self.model(dataset["test_input"])
                test_loss_val = loss_fn(test_pred, dataset["test_label"])

            # Store RMSE (sqrt of MSE) to match pykan's convention
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
