import torch
from kan import KAN
from .base import BaseKANModel


class PyKANModel(BaseKANModel):
    def __init__(
        self,
        width,
        grid,
        k,
        base_fun="silu",
        seed=1,
        grid_refinement_schedule=None,
        grid_refinement_steps=20,
        **kwargs,
    ):
        self.width = width
        self.grid = grid
        self.k = k
        self.base_fun = base_fun
        self.seed = seed
        self.grid_refinement_schedule = (
            list(grid_refinement_schedule) if grid_refinement_schedule else None
        )
        self.grid_refinement_steps = grid_refinement_steps
        self.model = None

    def build(self, device="cpu"):
        self.device = device
        grid = (
            self.grid_refinement_schedule[0]
            if self.grid_refinement_schedule
            else self.grid
        )
        self.model = KAN(
            width=list(self.width),
            grid=grid,
            k=self.k,
            base_fun=self.base_fun,
            seed=self.seed,
            device=device,
        )

    def fit(self, dataset, steps, lr, optimizer, loss_fn, batch_size, lamb, task_type="regression", **kwargs):

        all_results = {"train_loss": [], "test_loss": [], "reg": []}

        # Build accuracy metric callbacks for classification
        metrics = None
        if task_type == "classification":
            all_results["train_acc"] = []
            all_results["test_acc"] = []
            model_ref = self.model

            def train_acc():
                return torch.mean(
                    (torch.argmax(model_ref(dataset["train_input"]), dim=1) == dataset["train_label"]).float()
                )

            def test_acc():
                return torch.mean(
                    (torch.argmax(model_ref(dataset["test_input"]), dim=1) == dataset["test_label"]).float()
                )

            metrics = (train_acc, test_acc)

        remaining_steps = steps

        schedule = self.grid_refinement_schedule or [self.grid]

        for i, grid_size in enumerate(schedule):
            if remaining_steps <= 0:
                break

            # Refine grid for stages after the first (first grid is set in build)
            if i > 0:
                self.model = self.model.refine(grid_size)
                model_ref = self.model

            stage_steps = min(self.grid_refinement_steps, remaining_steps)

            fit_kwargs = dict(
                opt=optimizer,
                steps=stage_steps,
                lr=lr,
                lamb=lamb,
                loss_fn=loss_fn,
                batch=batch_size,
            )
            if metrics is not None:
                fit_kwargs["metrics"] = metrics

            results = self.model.fit(dataset, **fit_kwargs)

            all_results["train_loss"].extend(results["train_loss"])
            all_results["test_loss"].extend(results["test_loss"])
            all_results["reg"].extend(results["reg"])
            if task_type == "classification":
                all_results["train_acc"].extend(results.get("train_acc", []))
                all_results["test_acc"].extend(results.get("test_acc", []))

            remaining_steps -= stage_steps

        return all_results

    def predict(self, x):
        return self.model.forward(x)

    def parameter_count(self):
        return sum(p.numel() for p in self.model.parameters())

    def get_model(self):
        return self.model
