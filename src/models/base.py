from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from tqdm import tqdm


class BaseKANModel(ABC):
    model: torch.nn.Module | None = None
    device: str = "cpu"
    reports_rmse: bool = False

    @abstractmethod
    def build(self, device: str = "cpu") -> None:
        """Construct the underlying model from config."""

    def regularization_loss(self) -> float:
        return 0.0

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def get_model(self) -> torch.nn.Module:
        return self.model

    def fit(self, dataset, steps, lr, optimizer, loss_fn, batch_size, lamb, task_type="regression", **kwargs):
        if optimizer == "Adam":
            opt = torch.optim.AdamW(self.get_model().parameters(), lr=lr)
        elif optimizer == "LBFGS":
            opt = torch.optim.LBFGS(self.get_model().parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        results = {"train_loss": [], "test_loss": [], "reg": []}
        if task_type == "classification":
            results["train_acc"] = []
            results["test_acc"] = []

        pbar = tqdm(range(steps), desc="Training")
        for _ in pbar:
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
                    pred = self.predict(x)
                    loss = loss_fn(pred, y)
                    reg = self.regularization_loss()
                    if reg > 0:
                        loss = loss + lamb * reg
                    loss.backward()
                    return loss

                opt.step(closure)
            else:
                opt.zero_grad()
                pred = self.predict(x)
                loss = loss_fn(pred, y)
                reg = self.regularization_loss()
                if reg > 0:
                    loss = loss + lamb * reg
                loss.backward()
                opt.step()

            with torch.no_grad():
                train_pred = self.predict(dataset["train_input"])
                train_loss_val = loss_fn(train_pred, dataset["train_label"]).item()
                test_pred = self.predict(dataset["test_input"])
                test_loss_val = loss_fn(test_pred, dataset["test_label"]).item()

            results["train_loss"].append(train_loss_val)
            results["test_loss"].append(test_loss_val)
            results["reg"].append(0.0)

            if task_type == "classification":
                with torch.no_grad():
                    train_acc = (
                        (train_pred.argmax(dim=1) == dataset["train_label"])
                        .float()
                        .mean()
                        .item()
                    )
                    test_acc = (
                        (test_pred.argmax(dim=1) == dataset["test_label"])
                        .float()
                        .mean()
                        .item()
                    )
                results["train_acc"].append(train_acc)
                results["test_acc"].append(test_acc)
                pbar.set_postfix(loss=f"{test_loss_val:.4f}", acc=f"{test_acc:.4f}")
            else:
                pbar.set_postfix(rmse=f"{test_loss_val ** 0.5:.4f}")

        return results
