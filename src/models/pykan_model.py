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
        optimizer_factory,
        loss_fn,
        batch_size,
        lamb,
        task_type="regression",
        **kwargs,
    ):
        self.model.update_grid_from_samples(dataset["train_input"])

        metrics = self._make_metrics(dataset, task_type)

        # pykan's internal fit takes opt as a string ("Adam"/"LBFGS") and builds
        # the optimizer itself via torch.optim.Adam(params, lr=lr). To route our
        # wrapper-based factory through it, we monkey-patch torch.optim.Adam (in
        # kan.MultKAN's namespace) to ignore pykan's lr and call our factory,
        # which already carries all its hyperparams.
        import sys
        kan_mod = sys.modules.get("kan.MultKAN")
        original_adam = kan_mod.torch.optim.Adam if kan_mod is not None else None
        if kan_mod is not None:
            kan_mod.torch.optim.Adam = lambda params, lr=None: optimizer_factory(params)

        fit_kwargs = dict(
            opt="Adam",
            steps=epochs,
            lamb=lamb,
            loss_fn=loss_fn,
            batch=batch_size,
        )
        if metrics is not None:
            fit_kwargs["metrics"] = metrics

        try:
            results = self.model.fit(dataset, **fit_kwargs)
        finally:
            if original_adam is not None:
                kan_mod.torch.optim.Adam = original_adam
        return results

    def predict(self, x):
        return self.model.forward(x)

    def parameter_count(self):
        return sum(p.numel() for p in self.model.parameters())

    def get_model(self):
        return self.model
