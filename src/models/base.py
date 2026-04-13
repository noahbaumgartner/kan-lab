from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader, TensorDataset
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

    def predict(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        return self.model(x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def get_model(self) -> torch.nn.Module:
        return self.model

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
        # Build DataLoaders
        train_ds = TensorDataset(dataset["train_input"], dataset["train_label"])
        val_ds = TensorDataset(dataset["test_input"], dataset["test_label"])

        n_train = len(train_ds)
        bs = n_train if (batch_size == -1 or batch_size >= n_train) else batch_size

        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

        # Optimizer
        if optimizer == "Adam":
            opt = torch.optim.Adam(self.get_model().parameters(), lr=lr)
        elif optimizer == "LBFGS":
            opt = torch.optim.LBFGS(self.get_model().parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        results = {"train_loss": [], "test_loss": [], "reg": []}
        if task_type == "classification":
            results["train_acc"] = []
            results["test_acc"] = []

        for epoch in tqdm(range(epochs), desc="Training"):
            # --- Train ---
            self.get_model().train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

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

            # --- Validate ---
            self.get_model().eval()
            train_mse = 0.0
            train_correct = 0
            train_total = 0
            with torch.no_grad():
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    pred = self.predict(x)
                    train_mse += loss_fn(pred, y).item() * x.shape[0]
                    if task_type == "classification":
                        train_correct += (pred.argmax(dim=1) == y).sum().item()
                    train_total += x.shape[0]
            train_mse /= train_total

            val_mse = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    pred = self.predict(x)
                    val_mse += loss_fn(pred, y).item() * x.shape[0]
                    if task_type == "classification":
                        val_correct += (pred.argmax(dim=1) == y).sum().item()
                    val_total += x.shape[0]
            val_mse /= val_total

            results["train_loss"].append(train_mse)
            results["test_loss"].append(val_mse)
            results["reg"].append(0.0)

            if task_type == "classification":
                train_acc = train_correct / train_total
                val_acc = val_correct / val_total
                results["train_acc"].append(train_acc)
                results["test_acc"].append(val_acc)

        return results
