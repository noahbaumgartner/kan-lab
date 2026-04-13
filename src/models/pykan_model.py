import torch
from kan import KAN
from .base import BaseKANModel


class PyKANModel(BaseKANModel):
    reports_rmse = True

    def __init__(
        self,
        width,
        k,
        grid=20,
        base_fun="silu",
        **kwargs,
    ):
        self.width = width
        self.k = k
        self.grid = grid
        self.base_fun = base_fun
        self.model = None

    def build(self, device="cpu"):
        self.device = device
        self.model = KAN(
            width=list(self.width),
            grid=self.grid,
            k=self.k,
            base_fun=self.base_fun,
            seed=0,
            device=device,
            grid_range=[0, 20],
        )

    def _make_metrics(self, dataset, task_type):
        if task_type != "classification":
            return None
        model_ref = self.model

        def train_acc():
            return torch.mean(
                (
                    torch.argmax(model_ref(dataset["train_input"]), dim=1)
                    == dataset["train_label"]
                ).float()
            )

        def test_acc():
            return torch.mean(
                (
                    torch.argmax(model_ref(dataset["test_input"]), dim=1)
                    == dataset["test_label"]
                ).float()
            )

        return (train_acc, test_acc)

    def fit(
        self,
        dataset,
        epochs,
        lr,
        optimizer,
        loss_fn,
        batch_size,
        lamb,
        task_type="regression",
        **kwargs,
    ):
        self.model.update_grid_from_samples(dataset["train_input"])

        metrics = self._make_metrics(dataset, task_type)

        fit_kwargs = dict(
            opt=optimizer,
            steps=epochs,
            lr=lr,
            lamb=lamb,
            loss_fn=loss_fn,
            batch=batch_size,
        )
        if metrics is not None:
            fit_kwargs["metrics"] = metrics

        results = self.model.fit(dataset, **fit_kwargs)
        return results

    def predict(self, x):
        return self.model.forward(x)

    def parameter_count(self):
        return sum(p.numel() for p in self.model.parameters())

    def get_model(self):
        return self.model
