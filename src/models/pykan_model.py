import torch
from kan import KAN
from .base import BaseKANModel


class PyKANModel(BaseKANModel):
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

    def _make_metrics(self, dataset, task_type, eval_batch_size=1024):
        if task_type != "classification":
            return None
        model_ref = self.model

        def _batched_acc(inputs, labels):
            correct = 0
            total = inputs.shape[0]
            with torch.no_grad():
                for start in range(0, total, eval_batch_size):
                    end = start + eval_batch_size
                    preds = torch.argmax(model_ref(inputs[start:end]), dim=1)
                    correct += (preds == labels[start:end]).sum().item()
            return torch.tensor(correct / max(total, 1))

        def train_acc():
            return _batched_acc(dataset["train_input"], dataset["train_label"])

        def test_acc():
            return _batched_acc(dataset["test_input"], dataset["test_label"])

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
        dataset = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in dataset.items()
        }

        metrics = self._make_metrics(dataset, task_type)
        fit_kwargs = dict(
            opt=getattr(optimizer_factory, "PYKAN_OPT", "Adam"),
            lr=optimizer_factory.lr,
            steps=epochs,
            lamb=lamb,
            loss_fn=loss_fn,
            batch=batch_size,
        )
        if metrics is not None:
            fit_kwargs["metrics"] = metrics

        return self.model.fit(dataset, **fit_kwargs)

    def predict(self, x):
        return self.model.forward(x)

    def parameter_count(self):
        return sum(p.numel() for p in self.model.parameters())

    def get_model(self):
        return self.model
