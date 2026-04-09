import torch
from kan import KAN
from .base import BaseKANModel

# Fixed grid refinement schedule from the KAN paper (Section 2.5.1)
GRID_SCHEDULE = [20]
STEPS_PER_GRID = 1000


class PyKANModel(BaseKANModel):
    reports_rmse = True

    def __init__(
        self,
        width,
        k,
        base_fun="silu",
        **kwargs,
    ):
        self.width = width
        self.k = k
        self.base_fun = base_fun
        self.model = None

    def build(self, device="cpu"):
        self.device = device
        self.model = KAN(
            width=list(self.width),
            grid=GRID_SCHEDULE[0],
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

    @staticmethod
    def _collect_results(all_results, results, task_type):
        all_results["train_loss"].extend(results["train_loss"])
        all_results["test_loss"].extend(results["test_loss"])
        all_results["reg"].extend(results["reg"])
        if task_type == "classification":
            all_results["train_acc"].extend(results.get("train_acc", []))
            all_results["test_acc"].extend(results.get("test_acc", []))

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
        all_results = {"train_loss": [], "test_loss": [], "reg": []}
        if task_type == "classification":
            all_results["train_acc"] = []
            all_results["test_acc"] = []

        metrics = self._make_metrics(dataset, task_type)

        for i, grid_size in enumerate(GRID_SCHEDULE):
            if i > 0:
                self.model = self.model.refine(grid_size)
                metrics = self._make_metrics(dataset, task_type)

            fit_kwargs = dict(
                opt=optimizer,
                steps=STEPS_PER_GRID,
                lr=lr,
                lamb=lamb,
                loss_fn=loss_fn,
                batch=batch_size,
            )
            if metrics is not None:
                fit_kwargs["metrics"] = metrics

            results = self.model.fit(dataset, **fit_kwargs)
            self._collect_results(all_results, results, task_type)

        return all_results

    def predict(self, x):
        return self.model.forward(x)

    def parameter_count(self):
        return sum(p.numel() for p in self.model.parameters())

    def get_model(self):
        return self.model
