from __future__ import annotations
from abc import ABC, abstractmethod
import torch


class BaseKANModel(ABC):
    model: torch.nn.Module | None = None
    device: str = "cpu"

    @abstractmethod
    def build(self, device: str = "cpu") -> None:
        """Construct the underlying model from config."""

    def fit(
        self,
        dataset: dict,
        steps: int,
        lr: float,
        optimizer: str,
        loss_fn: callable | None,
        batch_size: int,
        lamb: float,
        **kwargs,
    ) -> dict:
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
                    reg = self.regularization_loss()
                    if reg > 0:
                        loss = loss + lamb * reg
                    loss.backward()
                    return loss

                opt.step(closure)
            else:
                opt.zero_grad()
                pred = self.model(x)
                loss = loss_fn(pred, y)
                reg = self.regularization_loss()
                if reg > 0:
                    loss = loss + lamb * reg
                loss.backward()
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

    def regularization_loss(self) -> float:
        return 0.0

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def get_model(self) -> torch.nn.Module:
        return self.model
